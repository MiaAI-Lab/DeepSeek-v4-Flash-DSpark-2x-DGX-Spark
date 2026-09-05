# SPDX-License-Identifier: Apache-2.0
"""
Disk backed KV cache tier for DeepSeek-V4-Flash-0731 (NVFP4 MLA) on dual GB10.

Ships two pieces, both loaded through documented vLLM extension points -- no
vLLM source is patched:

1. MultiGroupTieringOffloadingSpec
   vLLM's stock TieringOffloadingSpec is the only spec that supports secondary
   (disk) tiers, but it hard-asserts a SINGLE kv cache group:

       assert len(self.gpu_block_size) == 1     # tiering/spec.py

   This model has two groups -- the NVFP4 MLA attention layers and the FP8
   Lightning-Indexer cache -- so that assert trips and the engine dies before
   the disk tier is ever built. The assert is stricter than the surrounding
   code needs: `gpu_block_size` is one entry PER GROUP, and base.py's own
   check on the same field is `len(set(gpu_block_size)) == 1` (all groups must
   share a block size, not "there must be one group"). Here both groups are
   block_size=256, and the plain CPUOffloadingSpec -- which does the identical
   page-size math and has no such assert -- was verified booting on this exact
   model. So we re-implement get_manager() with the set() form.

2. CappedFileSystemTierManager
   vLLM's stock `fs_python` tier (FileSystemTierManager) has NO capacity bound
   and NO eviction: lookup() is a bare os.path.exists() and nothing is ever
   deleted, so it writes block files until the filesystem is full. This adds a
   byte cap with LRU eviction of the least-recently-touched block, which is
   the intended policy: fill the disk budget, then evict oldest-last-touched.
   Nothing expires on a timer.

Wire up via --kv-transfer-config:

  {"kv_connector":"OffloadingConnector","kv_role":"kv_both",
   "kv_connector_extra_config":{
     "spec_name":"MultiGroupTieringOffloadingSpec",
     "spec_module_path":"dsv4_kv_disk_tier",
     "cpu_bytes_to_use": 2147483648,
     "eviction_policy":"lru",
     "secondary_tiers":[{"type":"fs_capped","root_dir":"/kvdisk",
                         "max_bytes": 300000000000}]}}
"""

from __future__ import annotations

import functools
import os
import threading
from collections import OrderedDict
from collections.abc import Collection, Iterable

from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    OffloadingManager,
    ReqContext,
)
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
try:  # GLM53-TIER-IMPORT-FIX: JobMetadata moved base -> manager
    from vllm.v1.kv_offload.tiering.base import JobMetadata
except ImportError:  # newer vLLM (GLM image)
    from vllm.v1.kv_offload.tiering.manager import JobMetadata
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

# vLLM's dictConfig (vllm/logger.py:201) configures ONLY the "vllm" logger --
# DEFAULT_LOGGING_CONFIG has no "root" key, so the root logger keeps Python's
# defaults (level=WARNING, handlers=[]). A top-level logger named
# "dsv4_kv_disk_tier" therefore drops every .info() at the isEnabledFor()
# check before a record is even built. Living under "vllm." inherits
# VLLM_LOGGING_LEVEL, the vLLM formatter, and the (EngineCore pid=..) /
# (Worker_TP*) stream prefix.
logger = init_logger(f"vllm.{__name__}")

# Runtime fixes for upstream vLLM bugs found while building this tier:
#   P0 multi-node guard (PROVEN corruption), O_DIRECT alignment + non-destructive
#   load, descriptor-array lifetime on cuMemcpyBatchAsync, SWA over-promotion,
#   handler tensor aliasing, num_blocks>=1. Each is env-gated; see the module.
# Imported here because this module is loaded via spec_module_path BEFORE the
# connector is constructed, in BOTH the EngineCore and Worker processes.
import dsv4_vllm_patches  # noqa: E402
from vllm.utils.math_utils import round_up  # noqa: E402

dsv4_vllm_patches.apply_all()

# Registers the "shard_dist" tier: a per-node disk shard with an all-nodes
# completion barrier. This is the ONLY tier type that is correct under
# multi-node TP -- see that module's header for why a head-only tier persists
# zeros for every non-head rank.
import dsv4_shard_tier  # noqa: E402


def _load_block_buffered(
    source_path: str, view: memoryview, offset: int, block_size: int
) -> None:
    """Buffered replacement for vllm/v1/kv_offload/tiering/fs/io.py:load_block.

    Two deliberate differences from the stock implementation:
      1. No ``O_DIRECT`` -- see CappedFileSystemTierManager.submit_load for the
         alignment arithmetic that makes the stock read fail with EINVAL.
      2. No ``os.remove(source_path)`` on failure. Stock io.py deletes the block
         file when a read raises, which turns a transient I/O error into
         permanent cache loss.
    Loops on short reads rather than assuming one readv() drains the file.
    """
    try:
        if offset < 0 or offset + block_size > view.nbytes:
            raise ValueError(
                f"load offset/size out of range: offset={offset} "
                f"block_size={block_size} region={view.nbytes}"
            )
        view_slice = view.cast("B")[offset : offset + block_size]
        fd = os.open(source_path, os.O_RDONLY)
        try:
            n = 0
            while n < block_size:
                r = os.readv(fd, [view_slice[n:]])
                if r == 0:
                    raise OSError(
                        f"short read of {source_path}: {n}/{block_size} bytes"
                    )
                n += r
        finally:
            os.close(fd)
    except BaseException as e:
        # Count and SURFACE failures. Stock io.py swallowed the cause into a
        # generic thread-pool log line; without this the promotion path is
        # unobservable (which is exactly how the O_DIRECT EINVAL hid for so long).
        _LOAD_STATS.fail(e)
        raise
    else:
        _LOAD_STATS.ok(block_size)


class _LoadStats:
    """Module-level counters for disk->CPU promotion reads.

    The tier's own counters live on the scheduler-side manager and are only
    sampled from lookup(), which made them stale snapshots. These are updated
    on the I/O threads where the work actually happens.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.n_ok = 0
        self.n_fail = 0
        self.bytes_ok = 0
        self.first_error = None

    def ok(self, nbytes: int) -> None:
        with self._lock:
            self.n_ok += 1
            self.bytes_ok += nbytes
            if self.n_ok in (1, 10, 100, 500, 1000) or self.n_ok % 2000 == 0:
                logger.info(
                    "[dsv4-disk] load OK #%d (%.1f MiB total), failures=%d",
                    self.n_ok, self.bytes_ok / (1 << 20), self.n_fail,
                )

    def fail(self, exc: BaseException) -> None:
        with self._lock:
            self.n_fail += 1
            if self.first_error is None:
                self.first_error = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "[dsv4-disk] FIRST load failure (%s) -- all later failures "
                    "suppressed; ok=%d", self.first_error, self.n_ok,
                )
            elif self.n_fail in (10, 100) or self.n_fail % 1000 == 0:
                logger.error(
                    "[dsv4-disk] load failures=%d ok=%d first=%s",
                    self.n_fail, self.n_ok, self.first_error,
                )


_LOAD_STATS = _LoadStats()


class CappedFileSystemTierManager(FileSystemTierManager):
    """FileSystemTierManager + a byte budget + LRU eviction.

    The parent writes one file per offloaded block and never removes it. We
    keep an LRU ordering of on-disk blocks and delete the coldest ones when a
    store would exceed ``max_bytes``.

    Accounting is deliberately optimistic: the parent's thread pool writes
    asynchronously, so we reserve a block's bytes at submit_store() time
    rather than when the file lands. Evicting a block whose write is still in
    flight is harmless -- the file simply gets recreated on the next store, and
    a missing file on unlink is ignored.
    """

    def __init__(
        self,
        offloading_spec,
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        max_bytes: int | float = 0,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
        scan_existing: bool = True,
    ):
        super().__init__(
            offloading_spec=offloading_spec,
            primary_kv_view=primary_kv_view,
            tier_type=tier_type,
            root_dir=root_dir,
            n_read_threads=n_read_threads,
            n_write_threads=n_write_threads,
        )
        self._root_dir = root_dir
        self._max_bytes = int(max_bytes)
        self._lock = threading.RLock()
        # path -> size in bytes, ordered least-recently-used FIRST
        self._lru: OrderedDict[str, int] = OrderedDict()
        self._total_bytes = 0
        self._evicted_blocks = 0
        self._evicted_bytes = 0
        # Paths with a load in flight (pin count) -- never evict these out from
        # under a reader, and the job_id -> paths map so completion can unpin.
        self._pinned: dict[str, int] = {}
        self._load_job_keys: dict[int, list[str]] = {}
        self._store_job_keys: dict[int, list[str]] = {}

        if self._max_bytes <= 0:
            logger.warning(
                "CappedFileSystemTierManager: max_bytes<=0, running UNBOUNDED "
                "(identical to the stock fs_python tier). The disk will fill."
            )
        else:
            logger.info(
                "CappedFileSystemTierManager: root_dir=%s cap=%.1f GiB "
                "block=%d B (~%d blocks)",
                root_dir,
                self._max_bytes / (1 << 30),
                self._block_size,
                self._max_bytes // max(self._block_size, 1),
            )

        if scan_existing:
            # Adopt blocks left by a previous run so the cap survives restarts.
            # Ordered by mtime so the oldest are evicted first.
            self._scan_existing()

    # ---------------------------------------------------------------- helpers

    def _scan_existing(self) -> None:
        found: list[tuple[float, str, int]] = []
        # Walk ONLY this run's FileMapper subtree, never the whole root_dir, and
        # accept only files of exactly one block. _evict_for() unlinks whatever
        # this adopts, and ".bin" is also the HuggingFace weight-shard extension
        # -- the big NVMe holding a 300 GB KV budget is very often the same disk
        # holding the model. Scanning root_dir would make the cache capable of
        # silently deleting weights.
        scan_root = f"{self.file_mapper.base_path}_r{self.file_mapper.rank}"
        try:
            for dirpath, _dirnames, filenames in os.walk(scan_root):
                for fn in filenames:
                    if not fn.endswith(".bin"):
                        continue
                    p = os.path.join(dirpath, fn)
                    try:
                        st = os.stat(p)
                    except OSError:
                        continue
                    if st.st_size != self._block_size:
                        continue
                    found.append((st.st_mtime, p, st.st_size))
        except OSError as e:
            logger.warning("CappedFileSystemTier: scan of %s failed: %s",
                           scan_root, e)
            return

        found.sort(key=lambda t: t[0])  # oldest first == LRU front
        with self._lock:
            for _mtime, p, size in found:
                if p not in self._lru:
                    self._lru[p] = size
                    self._total_bytes += size
        if found:
            logger.info(
                "CappedFileSystemTier: adopted %d existing blocks (%.1f GiB)",
                len(found),
                self._total_bytes / (1 << 30),
            )

    def _mark_recent(self, paths: Iterable[str]) -> None:
        with self._lock:
            for p in paths:
                if p in self._lru:
                    self._lru.move_to_end(p)

    def _evict_for(self, incoming_bytes: int) -> None:
        """Evict least-recently-used blocks until incoming_bytes fits."""
        if self._max_bytes <= 0:
            return
        with self._lock:
            while self._lru and self._total_bytes + incoming_bytes > self._max_bytes:
                # Skip blocks with a load in flight: unlinking between the
                # lookup() hit and the read thread's os.open turns a hit into a
                # load failure (the same race DistributedShardTier guards with
                # its _pinned set).
                victim = None
                for path in self._lru:
                    if path not in self._pinned:
                        victim = path
                        break
                if victim is None:
                    # Everything resident is pinned; stop rather than evict a
                    # block a reader is about to open.
                    break
                size = self._lru.pop(victim)
                self._total_bytes -= size
                self._evicted_blocks += 1
                self._evicted_bytes += size
                try:
                    os.unlink(victim)
                except FileNotFoundError:
                    pass
                except OSError as e:
                    logger.warning("CappedFileSystemTier: unlink %s: %s", victim, e)

    def _account(self, paths: Iterable[str]) -> None:
        with self._lock:
            for p in paths:
                if p in self._lru:
                    self._lru.move_to_end(p)
                else:
                    self._lru[p] = self._block_size
                    self._total_bytes += self._block_size

    # ------------------------------------------------------- tier interface

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext | None = None):
        hit = super().lookup(key, req_context)
        # GLM53-LOOKUPRESULT: the parent returns a LookupResult enum, and every
        # member (MISS/HIT_PENDING/RETRY included) is truthy. Only a real HIT
        # should refresh LRU position.
        if hit is LookupResult.HIT:
            self._mark_recent([self.file_mapper.get_file_name(key)])
        return hit

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        # Base SecondaryTierManager.touch() is a no-op; this is the documented
        # hook for exactly this ("Mark blocks as recently used for eviction").
        self._mark_recent(self.file_mapper.get_file_name(k) for k in keys)

    @override
    def submit_store(self, job_metadata: JobMetadata) -> None:
        paths = [self.file_mapper.get_file_name(k) for k in job_metadata.keys]
        with self._lock:
            new_paths = [p for p in paths if p not in self._lru]
            self._evict_for(len(new_paths) * self._block_size)
            self._store_job_keys[job_metadata.job_id] = new_paths
        super().submit_store(job_metadata)
        self._account(paths)

    @override
    def submit_load(self, job_metadata: JobMetadata) -> None:
        """Load without O_DIRECT. THIS is the fix that makes promotion work.

        Stock ``tiering/fs/io.py:load_block`` opens the block file with
        ``O_DIRECT`` and issues one ``readv()`` of ``self._block_size`` bytes.
        O_DIRECT requires the transfer LENGTH to be a multiple of the
        filesystem's logical block size (512 on the ext4 behind /kvdisk). Here
        ``self._block_size`` is ``primary_kv_view.strides[0]`` == 2,103,552 and

            2_103_552 % 512 == 256

        so every promotion read returns EINVAL -- 100% of the time, at any
        buffer alignment. Verified on the real block files: length 2103552 ->
        EINVAL, length 2103296 -> OK.

        Stores are unaffected because ext4's O_DIRECT *write* path tolerates a
        trailing sub-block on an extending write. That asymmetry is exactly why
        writes always worked and promotion never could.

        Worse, stock io.py then ``os.remove()``s the source file on failure and
        the manager calls ``complete_write(success=False)``, which frees the CPU
        entry too -- so each attempt DESTROYED the cached block in both tiers.
        This override drops O_DIRECT and never deletes on failure.
        """
        # A load is a use: keep these blocks hot.
        paths = [self.file_mapper.get_file_name(k) for k in job_metadata.keys]
        with self._lock:
            for p in paths:
                self._pinned[p] = self._pinned.get(p, 0) + 1
            self._load_job_keys[job_metadata.job_id] = paths
        self._mark_recent(paths)
        tasks = (
            functools.partial(
                _load_block_buffered,
                self.file_mapper.get_file_name(key),
                self._primary_kv_view,
                int(bid) * self._block_size,
                self._block_size,
            )
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
        )
        self._pool.enqueue_load(job_metadata.job_id, len(job_metadata.keys), tasks)

    def get_finished_jobs(self):
        """Poll completions and release load pins before they can block eviction.

        A block stays pinned from submit_load() until its job reports finished,
        so _evict_for() can never unlink a file a read thread is about to open.
        """
        for result in super().get_finished_jobs():
            paths = self._load_job_keys.pop(result.job_id, None)
            if paths:
                with self._lock:
                    for p in paths:
                        n = self._pinned.get(p, 0) - 1
                        if n <= 0:
                            self._pinned.pop(p, None)
                        else:
                            self._pinned[p] = n
            # A failed store never landed its file, but submit_store already
            # accounted its bytes optimistically; drop the phantom entry so
            # _total_bytes and the LRU do not drift after a write error.
            store_paths = self._store_job_keys.pop(result.job_id, None)
            if store_paths is not None and not result.success:
                with self._lock:
                    for p in store_paths:
                        if p in self._lru:
                            self._total_bytes -= self._lru.pop(p)
            yield result

    def stats(self) -> dict:
        with self._lock:
            return {
                "blocks": len(self._lru),
                "bytes": self._total_bytes,
                "cap_bytes": self._max_bytes,
                "evicted_blocks": self._evicted_blocks,
                "evicted_bytes": self._evicted_bytes,
            }


class InstrumentedTieringManager(TieringOffloadingManager):
    """TieringOffloadingManager with counters on the promotion path.

    Writes to disk demonstrably work, but nothing is ever promoted back
    (metrics show GPU_to_CPU only, external hit rate 0.0%). This narrows where
    the chain breaks: are secondary lookups hitting at all, is
    _initiate_promotion() refusing because the primary tier is full, and is
    take_events() -- the deferred flush point -- actually being drained?
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._n_lookup = 0
        self._n_primary_hit = 0
        self._n_secondary_hit = 0
        self._n_promo_ok = 0
        self._n_promo_full = 0
        self._n_take_events = 0
        self._n_flushed = 0
        self._last_report = 0

    def lookup(self, key, req_context):
        self._n_lookup += 1
        r = super().lookup(key, req_context)
        if r is LookupResult.HIT:
            self._n_primary_hit += 1
        self._maybe_report()
        return r

    def _initiate_promotion(self, tier, key, req_context):
        self._n_secondary_hit += 1
        ok = super()._initiate_promotion(tier, key, req_context)
        if ok:
            self._n_promo_ok += 1
        else:
            self._n_promo_full += 1
        return ok

    def _flush_pending_promotions(self):
        pending = sum(len(v) for v in self._pending_load_submissions.values())
        super()._flush_pending_promotions()
        if pending:
            self._n_flushed += pending
        return None

    def take_events(self):
        self._n_take_events += 1
        return super().take_events()

    def _maybe_report(self):
        # Periodic snapshot, not per-lookup: a fully-hitting request yields one
        # lookup per offload block, so logging every lookup floods the log in
        # steady state while adding no signal (all counters are cumulative).
        # Every 1000 lookups keeps the counters visible without the spam.
        if self._n_lookup - self._last_report < 1000:
            return
        self._last_report = self._n_lookup
        logger.info(
            "[dsv4-disk] lookups=%d primary_hits=%d secondary_hits=%d "
            "promo_ok=%d promo_primary_full=%d take_events=%d flushed=%d",
            self._n_lookup,
            self._n_primary_hit,
            self._n_secondary_hit,
            self._n_promo_ok,
            self._n_promo_full,
            self._n_take_events,
            self._n_flushed,
        )


class _HandlerProbeMixin:
    """Dump the canonical-tensor -> group mapping.

    This is the measurement that decides whether per-group offload regions are
    worth building. Every CPU cell is currently sized for ALL canonical tensors
    (`cpu_page_size`), but a key belongs to exactly ONE group
    (`get_offload_group_idx` reads the key's last 4 bytes). The achievable
    saving is therefore

        cpu_page_size / (bytes belonging to that key's group)

    which requires knowing each group's SUMMED tensor bytes -- a group spans
    many layers, so it is NOT one page_size_bytes.
    """

    def create_handlers(self, kv_caches):  # type: ignore[override]
        try:
            tensors = kv_caches.tensors
            refs = kv_caches.group_data_refs
            per_tensor = [t.page_size_bytes for t in tensors]
            total = sum(per_tensor)
            logger.info(
                "[dsv4-disk] canonical tensors=%d total_page_bytes=%d "
                "(cpu cell/worker=%d)",
                len(tensors), total, getattr(self, "cpu_page_size_per_worker", -1),
            )
            seen: dict[int, list[int]] = {}
            for gi, group in enumerate(refs):
                idxs, gbytes = [], 0
                for r in group:
                    ti = getattr(r, "tensor_idx", getattr(r, "index", None))
                    pb = getattr(r, "page_size_bytes", None)
                    idxs.append(ti)
                    if pb is not None:
                        gbytes += pb
                    seen.setdefault(ti, []).append(gi)
                logger.info(
                    "[dsv4-disk]   group %d: %d refs, summed page bytes=%d "
                    "(%.1f%% of cell) -> per-group cell would save %.1fx",
                    gi, len(group), gbytes,
                    100.0 * gbytes / total if total else 0.0,
                    (total / gbytes) if gbytes else float("nan"),
                )
            shared = {t: g for t, g in seen.items() if len(set(g)) > 1}
            logger.info(
                "[dsv4-disk] tensors shared across groups: %d %s",
                len(shared),
                ("(SHARING PRESENT - per-group regions need care)" if shared
                 else "(none - clean per-group partition)"),
            )
        except Exception as e:  # never break serving for a diagnostic
            logger.warning("[dsv4-disk] handler probe failed: %r", e)
        return super().create_handlers(kv_caches)


class _BlockSizeFactorMixin:
    """Enable ``block_size_factor`` > 1 on a hybrid (multi-block-size) model.

    WHY THIS IS THE BIGGEST LEVER. Staging cost is dominated by sliding-window
    groups, and vLLM already keeps only the trailing ``tail`` blocks of each
    full-attention *alignment segment* (offloading/scheduler.py store path).
    The kept fraction per SWA group is

        tail / per_segment = cdiv(window, bs*F) / (align_tokens / (bs*F))

    ``per_segment`` is independent of F, while ``tail`` shrinks toward 1 as F
    grows -- so raising F collapses SWA stores. Measured projection for this
    model (window sizes 128/128/8/128 tokens, block sizes 256/64/64/4/8):

        F=1  -> 198.2 GB per 1M tokens   (28.0x the GPU KV density)
        F=16 ->  43.1 GB per 1M tokens   ( 6.1x)
        79K prompt: 14.9 GB -> 3.2 GB    (4.6x less staging)

    WHY IT IS NOT JUST A CONFIG KNOB. Setting ``block_size`` in
    kv_connector_extra_config trips this assert in
    ``vllm/v1/kv_offload/base.py``::

        gpu_block_sizes = set(self.gpu_block_size)
        assert len(gpu_block_sizes) == 1, "... all groups must have the same
                                           block size"

    That guard is over-strict. ``block_size_factor`` is only ever used as a
    per-group multiplier (``offloaded_block_size = gpu_block_size * factor``),
    which is perfectly well defined when groups differ -- the assert exists
    solely because the stock code derives F from a single target token count.
    We therefore set the factor directly and redo the arithmetic that
    ``CPUOffloadingSpec.__init__`` performed with F=1.
    """

    def __init__(self, *args, **kwargs):
        # GLM53-SPEC-PORT: newer vLLM calls OffloadingSpec.__init__(config)
        # instead of (vllm_config, kv_cache_config). Forward whatever we got.
        super().__init__(*args, **kwargs)

        # Both reasons this mixin exists are obsolete on the new API:
        #   - multi-group tolerance: OffloadingConfig.groups is native and the
        #     single-group assert is gone from base.py
        #   - block_size_factor: removed from vllm/v1/kv_offload entirely
        # Detect the new spec (it carries .config) and no-op.
        if getattr(self, "config", None) is not None and not hasattr(
            self, "vllm_config"
        ):
            if int(self.extra_config.get("dsv4_block_size_factor", 1)) > 1:
                logger.warning(
                    "[dsv4-disk] dsv4_block_size_factor is ignored on this vLLM: "
                    "block_size_factor was removed upstream and multi-group is "
                    "native. Drop it from kv_connector_extra_config."
                )
            return

        factor = int(self.extra_config.get("dsv4_block_size_factor", 1))
        if factor <= 1:
            return
        if self.block_size_factor != 1:
            logger.warning(
                "[dsv4-disk] block_size_factor already %d; not overriding",
                self.block_size_factor,
            )
            return

        # Every group's offloaded_block_size must stay a whole number of
        # hash blocks, or Request.block_hashes slicing desynchronises.
        for gb in self.gpu_block_size:
            if (gb * factor) % self.hash_block_size != 0:
                raise ValueError(
                    f"dsv4_block_size_factor={factor}: offloaded block "
                    f"{gb * factor} not divisible by hash_block_size "
                    f"{self.hash_block_size}"
                )

        self.block_size_factor = factor

        # Redo CPUOffloadingSpec.__init__'s sizing with the new factor.
        world_size = self.vllm_config.parallel_config.world_size
        cpu_bytes = int(self.extra_config["cpu_bytes_to_use"])
        if self.kv_cache_config.num_blocks > 0:
            total_gpu_kv_bytes = sum(t.size for t in self.kv_cache_config.kv_cache_tensors)
            kv_bytes_per_block = (
                total_gpu_kv_bytes // self.kv_cache_config.num_blocks
            ) * world_size
        else:
            kv_bytes_per_block = 0
        per_offloaded_block = kv_bytes_per_block * factor
        aligned_kv_bytes_per_offloaded_block = round_up(
            per_offloaded_block, self.BLOCK_SIZE_ALIGNMENT
        )
        self.kv_bytes_per_offloaded_block = aligned_kv_bytes_per_offloaded_block
        self.num_blocks = (
            cpu_bytes // aligned_kv_bytes_per_offloaded_block
            if aligned_kv_bytes_per_offloaded_block > 0 else 0
        )
        self.cpu_page_size_per_worker = (
            per_offloaded_block // world_size if world_size > 0 else 0
        )
        logger.info(
            "[dsv4-disk] block_size_factor=%d -> offloaded blocks %s tokens, "
            "cell=%d B/worker, num_blocks=%d (%.1f GiB tier)",
            factor,
            [gb * factor for gb in self.gpu_block_size],
            self.cpu_page_size_per_worker,
            self.num_blocks,
            cpu_bytes / (1 << 30),
        )


class _ShardAgentMixin:
    """Start this worker's ShardAgent once its offload region exists.

    ``create_handlers`` is the one hook that runs in EVERY worker process (on
    both nodes) and has the node-local ``SharedOffloadRegion`` in hand, which
    is exactly what the agent needs to address its own KV slot.
    """

    def create_handlers(self, kv_caches):  # type: ignore[override]
        # Older vLLM (DeepSeek image). Unchanged.
        handlers = super().create_handlers(kv_caches)
        dsv4_shard_tier.maybe_start_shard_agent(self, handlers)
        return handlers

    def get_worker(self, kv_caches):  # type: ignore[override]
        # GLM53-WORKER-HOOK: newer vLLM renamed create_handlers -> get_worker.
        # Without this override the mixin is dead code and the agent never
        # starts: the tier attaches and logs healthy while _ready stays False,
        # every lookup returns MISS and /kvdisk stays empty. The returned
        # worker carries _mmap_region (cpu/gpu_worker.py), which is what
        # maybe_start_shard_agent needs to find this node's KV slot.
        worker = super().get_worker(kv_caches)
        # This vLLM's CPUOffloadingWorker stores the node-local region on its
        # store handler, not on the worker itself; expose it for the shard
        # agent (which reads getattr(handlers, "_mmap_region")).
        if getattr(worker, "_mmap_region", None) is None:
            _h = getattr(worker, "_store_handler", None)
            worker._mmap_region = getattr(_h, "_mmap_region", None)
        dsv4_shard_tier.maybe_start_shard_agent(self, worker)
        return worker


class MultiGroupTieringOffloadingSpec(
    _BlockSizeFactorMixin, _ShardAgentMixin, _HandlerProbeMixin,
    TieringOffloadingSpec,
):
    """TieringOffloadingSpec that tolerates >1 KV cache group.

    Identical to the parent's get_manager() except the single-group assert is
    replaced by the weaker (and actually required) invariant that all groups
    share one GPU block size -- the same check base.py applies to this field.
    """

    @override
    def get_manager(self) -> OffloadingManager:
        if self._manager:
            return self._manager

        # P0. This subclass OVERRIDES get_manager, so patching the base class
        # does not protect us -- the override shadows it. Call the check
        # explicitly. It raises on multi-node TP, where a HEAD-ONLY disk tier
        # silently serves ZEROS for every non-head rank (proven by needle test).
        #
        # The "shard_dist" tier is the fix for exactly that failure, so it is
        # the one configuration allowed to proceed multi-node: every node
        # persists and restores its own slot, and a block is only servable once
        # all nodes have acked it.
        # GLM53-GETMANAGER-PORT: newer vLLM removed the single-group assert this
        # whole override was written to bypass, and its stock get_manager() now
        # builds the region + primary + secondary tiers itself. Delegate.
        # The P0 corruption guard is preserved verbatim below.
        _new_api = getattr(self, "config", None) is not None and not hasattr(
            self, "vllm_config"
        )
        if _new_api:
            if dsv4_shard_tier.shard_tier_config(self) is None:
                import types as _types

                # check_multinode_safe only reads parallel_config.world_size.
                # Newer vLLM carries this on self.config (parallel.world_size),
                # not self.vllm_config.parallel_config -- see shard_tier_config
                # and DistributedShardTier.__init__ for the same split.
                _shim = _types.SimpleNamespace(
                    parallel_config=_types.SimpleNamespace(
                        world_size=int(self.config.parallel.world_size)
                    )
                )
                dsv4_vllm_patches.check_multinode_safe(_shim)
            else:
                logger.info(
                    "[dsv4-disk] shard_dist tier configured -- per-node shards "
                    "make multi-node TP safe; skipping the head-only-tier guard."
                )
            logger.info(
                "[dsv4-disk] newer vLLM detected: delegating to stock "
                "TieringOffloadingSpec.get_manager() (multi-group is native; "
                "the group-count workaround this override provided is obsolete)."
            )
            return super().get_manager()

        if dsv4_shard_tier.shard_tier_config(self) is None:
            dsv4_vllm_patches.check_multinode_safe(self.vllm_config)
        else:
            logger.info(
                "[dsv4-disk] shard_dist tier configured -- per-node shards make "
                "multi-node TP safe; skipping the head-only-tier guard."
            )

        kv_events_config = self.vllm_config.kv_events_config
        enable_events = (
            kv_events_config is not None and kv_events_config.enable_kv_cache_events
        )

        world_size = self.vllm_config.parallel_config.world_size
        scheduler_mmap = SharedOffloadRegion(
            instance_id=self.vllm_config.instance_id,
            num_blocks=self.num_blocks,
            rank=None,
            kv_bytes_per_block=self.kv_bytes_per_offloaded_block,
            cpu_page_size=self.cpu_page_size_per_worker,
        )
        self._scheduler_mmap = scheduler_mmap

        # vLLM stock asserts len(self.gpu_block_size) == 1 (one GROUP), and an
        # earlier version of this file asserted a single block size in TOKENS.
        # Both are the wrong quantity. The flat CPU staging region indexes
        # block i at i * cpu_page_size, so what must be uniform is the page
        # size in BYTES -- not tokens per block.
        #
        # vLLM guarantees exactly that: unify_kv_cache_spec_page_size() in
        # v1/core/kv_cache_utils.py raises the block_size of any layer with a
        # smaller page (ratio = max_page_size // layer_page_size) until
        # page_size_bytes matches the max, asserting equality. Differing token
        # block sizes such as (256, 64, 64, 4, 8) are the *signature* of that
        # unification, not evidence of heterogeneous pages.
        groups = self.kv_cache_config.kv_cache_groups
        page_bytes = [g.kv_cache_spec.page_size_bytes for g in groups]
        for i, (g, blk, pb) in enumerate(zip(groups, self.gpu_block_size, page_bytes)):
            sw = getattr(g.kv_cache_spec, "sliding_window", None)
            logger.info(
                "  KV group %d: %s block_size=%d tokens, page=%d bytes, "
                "sliding_window=%s (blocks=%s)",
                i,
                type(g.kv_cache_spec).__name__,
                blk,
                pb,
                sw,
                (sw + blk - 1) // blk if sw else None,
            )
        # NO uniformity assert. Groups here legitimately differ in page size
        # (measured: 8640 and 37440 bytes, ratio 4.33 so vLLM's
        # unify_kv_cache_spec_page_size() cannot equalise them), and that is
        # fine: SharedOffloadRegion does NOT lay out one group per row. Per
        # its own docstring each block cell is cpu_page_size bytes holding
        # ALL canonical tensors concatenated --
        #     [ tensor0_data | tensor1_data | ... ]
        # -- and create_next_view(tensor_page_size) carves each tensor's
        # sub-view at its own offset inside that cell. Secondary tiers address
        # block b as view[b], i.e. a whole row, so rows are uniform by
        # construction whatever the per-group sizes are.
        #
        # cpu_page_size_per_worker is total_gpu_kv_bytes // num_blocks, which
        # is exactly that concatenated per-block size -- not a lossy average.
        # GPU<->CPU copies are already per-group: gpu_worker.py sets
        #     all_sizes[...] = data_ref.page_size_bytes
        # per data ref, and its comments explicitly cover HMA hybrid models.
        logger.info(
            "MultiGroupTieringOffloadingSpec: %d KV cache groups, page sizes "
            "%s bytes, concatenated cell=%d bytes/block -- proceeding "
            "(stock vLLM aborts on group count alone).",
            len(groups),
            sorted(set(page_bytes)),
            self.cpu_page_size_per_worker,
        )

        primary_tier = CPUPrimaryTierOffloadingManager(
            num_blocks=self.num_blocks,
            cache_policy=self.eviction_policy,  # type: ignore[arg-type]
            enable_events=enable_events,
            mmap_region=scheduler_mmap,
        )

        primary_kv_view = primary_tier.get_kv_memoryview()
        secondary_tiers = []
        for i, tier_config in enumerate(self.secondary_tier_configs):
            tier = SecondaryTierFactory.create_secondary_tier(
                tier_config, primary_kv_view, self
            )
            secondary_tiers.append(tier)
            logger.info("Created secondary tier #%d (%s)", i, tier.tier_type)

        if int(self.extra_config.get("store_threshold", 0)) >= 2:
            raise ValueError(
                "store_threshold is not supported for TieringOffloadingSpec"
            )

        self._manager = InstrumentedTieringManager(
            primary_tier=primary_tier,
            secondary_tiers=secondary_tiers,
            enable_events=enable_events,
        )
        logger.info(
            "MultiGroupTieringOffloadingSpec ready: primary=%s (%d blocks), "
            "%d secondary tier(s)",
            self.eviction_policy,
            self.num_blocks,
            len(secondary_tiers),
        )
        return self._manager


# Register the capped tier under its own type so the stock `fs_python`
# behaviour stays available for comparison.
try:
    SecondaryTierFactory.register_tier(
        "fs_capped", "dsv4_kv_disk_tier", "CappedFileSystemTierManager"
    )
except ValueError:
    # Already registered (module imported twice: scheduler + worker process).
    pass
