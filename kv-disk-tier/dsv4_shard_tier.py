# SPDX-License-Identifier: Apache-2.0
"""Per-node sharded disk KV tier for multi-node TP (spark1 + spark2).

WHY THIS EXISTS
---------------
The stock ``fs_python`` secondary tier is built ONLY in the scheduler process
(``offloading_connector.py`` gates secondary tiers to the SCHEDULER role), and
it copies whole rows of the primary staging region:

    self._block_size = primary_kv_view.strides[0]      # fs/manager.py
                     = cpu_page_size * world_size

On a single node that row genuinely holds every worker's shard, because all
TP ranks mmap the SAME ``/dev/shm/vllm_offload_<id>.mmap``. Across two nodes
they do not: each node has its own /dev/shm, and each worker picks its slot
with

    rank = torch.accelerator.current_device_index()    # tiering/spec.py:170

which is the LOCAL device index -- 0 on both spark1 and spark2, since each has
one GB10. So the layout is:

    spark1 /dev/shm:  [ spark1 shard | ZEROS ]
    spark2 /dev/shm:  [ spark2 shard | ZEROS ]

The head-only tier therefore persisted ``[spark1 shard | zeros]`` and restored
it to spark1 alone. spark2's half of every block came back as zeros. That is
not a slow cache -- it is silent corruption, and it is exactly what the needle
probe caught: cold run FOUND the secret, restored run reported MISSING while
every head-side metric (99.3% hit rate, zero I/O failures) looked perfect.

THE FIX
-------
Give every node its own tier over its own disk, and persist only that node's
own slot:

    offset(block b) = b * row_stride + rank_local * cpu_page_size
    length          = cpu_page_size          (NOT the full row)

Each node writes ``/kvdisk`` on its own NVMe with its own byte cap. On restore
each node reads its own slice back into its own /dev/shm slot, and the union
reconstitutes a correct distributed block. The shards are disjoint by
construction, so no cross-node byte-layout negotiation is needed at all.

Consequence: per-node I/O halves (1 slice, not a 2-slice row), and each node reads
its shard from its OWN NVMe instead of pulling bytes across the fabric.

NOT a consequence -- capacity. An earlier version of this docstring claimed
``max_bytes`` being per node makes 300 GB x 2 = 600 GB of distinct cache. That is
WRONG. The two nodes' slot bytes are byte-identical (md5-verified across all 1020
rows): this is MLA, and the latent KV (one kv_lora + rope vector per token, shared
by every head) is REPLICATED across TP ranks rather than sharded. So both nodes
store the same content and distinct capacity is 300 GB, not 600 GB.

That does not weaken the correctness argument -- the head-only tier was broken
because spark2's region was never POPULATED, which is a different failure from the
shards differing -- but it does mean this design buys locality and correctness,
not capacity.

HOW THE BARRIER WORKS
---------------------
``SecondaryTierManager`` is already an async submit/poll interface, and
``TieringOffloadingManager`` holds the primary-tier ref_cnt (``prepare_read``
for stores, ``prepare_write`` for promotions) until ``get_finished()`` reports
the job. So correctness across nodes needs exactly one rule:

    report a job finished ONLY when every node has acked it.

Until then the framework keeps the CPU slots pinned, ``lookup()`` keeps
returning None ("promotion in flight, retry later"), and the GPU load cannot
start. No change to the scheduler/worker transfer protocol is required -- the
existing tier contract already provides the ordering guarantee.

TOPOLOGY
--------
  * ``DistributedShardTier`` runs in the EngineCore (scheduler) process on the
    head. It moves NO bytes. It owns the authoritative key index, the LRU, and
    the completion barrier, and talks to agents over a ZMQ ROUTER.
  * ``ShardAgent`` runs inside every worker process (head included), started
    from ``create_handlers``. It owns its node's disk and its node's /dev/shm
    slot, and does all the actual I/O.

On startup each agent reports its on-disk inventory; the head keeps only the
INTERSECTION and tells the agents to drop the rest. A block is cacheable only
if every node still has its shard, which makes a half-populated cache
impossible to observe rather than merely unlikely.
"""

from __future__ import annotations

import errno
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Collection, Iterable

import msgspec.msgpack
import zmq
from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.base import (
    JobResult,
    SecondaryTierManager,
)

try:  # GLM53-TIER-IMPORT-FIX: JobMetadata moved base -> manager
    from vllm.v1.kv_offload.tiering.base import JobMetadata
except ImportError:  # newer vLLM (GLM image)
    from vllm.v1.kv_offload.tiering.manager import JobMetadata

# GLM53-ABC-PORT: newer vLLM replaced the bool lookup contract with an enum
# and added RequestOffloadingContext. Both absent on the older DeepSeek image,
# where lookup() must keep returning a plain bool.
try:
    from vllm.v1.kv_offload.base import LookupResult as _LookupResult
except ImportError:
    _LookupResult = None
try:
    from vllm.v1.kv_offload.base import (
        RequestOffloadingContext as _RequestOffloadingContext,
    )
except ImportError:
    _RequestOffloadingContext = None
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

# See dsv4_kv_disk_tier for why the logger must live under "vllm.".
logger = init_logger(f"vllm.{__name__}")

_ENC = msgspec.msgpack.Encoder()
_DEC = msgspec.msgpack.Decoder()

TIER_TYPE = "shard_dist"
DEFAULT_PORT = 25055


def _head_addr_from_env() -> str:
    """Resolve the head endpoint agents dial.

    ``DSV4_SHARD_HEAD`` wins; otherwise fall back to the same master address
    vLLM's own multinode bootstrap uses, so a stock launch needs no new env.
    """
    explicit = os.environ.get("DSV4_SHARD_HEAD")
    if explicit:
        return explicit
    host = os.environ.get("MASTER_ADDR") or os.environ.get("VLLM_HOST_IP") or "127.0.0.1"
    port = os.environ.get("DSV4_SHARD_PORT", str(DEFAULT_PORT))
    return f"tcp://{host}:{port}"


# ---------------------------------------------------------------- agent side


# O_DIRECT keeps the page cache out of the transfer entirely. That is not an
# optimisation here, it is a survival requirement: these are 121 GiB boxes running
# at ~116 GiB, and a buffered restore of a 1M-token context would pull 8.4 GB
# through the page cache. Measured consequence of the buffered version --
# "NVRM: Check failed: Out of memory [NV_ERR_NO_MEMORY]" in dmesg on BOTH nodes
# and a dead worker, on a restore of only 262 cells (2.2 GB).
#
# Alignment works out here where it did NOT for stock vLLM. O_DIRECT needs the
# length, the file offset and the buffer address all aligned to the device
# logical block size (512 on this NVMe):
#
#     per-node slice   4,207,104 % 512 == 0   <- ours, fine
#     row_stride       8,414,208 % 512 == 0   <- so every block offset is aligned
#     upstream's unit  2,103,552 % 512 == 256 <- EINVAL, every single read
#
# Persisting only this node's slice is what made the length divisible; the stock
# full-row I/O unit could never be aligned. Falls back to buffered on EINVAL so a
# different filesystem or block size degrades instead of failing.
_USE_ODIRECT = os.environ.get("DSV4_SHARD_ODIRECT", "1") != "0"
_ODIRECT_OK = _USE_ODIRECT


def _aligned(offset: int, nbytes: int) -> bool:
    return _ODIRECT_OK and (offset % 512 == 0) and (nbytes % 512 == 0)


def _store_one(path: str, mv: memoryview, offset: int, nbytes: int) -> None:
    """Write this node's slice of one block, atomically."""
    global _ODIRECT_OK
    # Unique per writer: a fixed "<path>.tmp" lets two threads from the 16-thread
    # pool interleave writes into one temp file and then both rename it, which
    # publishes a block built from two different sources under a valid key.
    tmp = f"{path}.{os.getpid()}.{threading.get_ident():x}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    direct = _aligned(offset, nbytes)
    if direct:
        flags |= os.O_DIRECT
    try:
        fd = os.open(tmp, flags, 0o600)
    except OSError:
        direct = False
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        src = mv[offset : offset + nbytes]
        written = 0
        while written < nbytes:
            try:
                written += os.write(fd, src[written:])
            except OSError as e:
                if direct and e.errno == errno.EINVAL and written == 0:
                    # Retry this block buffered, and stop trying O_DIRECT.
                    _ODIRECT_OK = False
                    logger.warning(
                        "[dsv4-shard] O_DIRECT write rejected (EINVAL); "
                        "falling back to buffered I/O for the rest of this run"
                    )
                    os.close(fd)
                    fd = -1        # so `finally` cannot double-close on reopen failure
                    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                    direct = False
                    continue
                raise
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    finally:
        if fd >= 0:
            os.close(fd)
    # Rename last so a reader never observes a partial block.
    os.replace(tmp, path)


def _load_one(path: str, mv: memoryview, offset: int, nbytes: int) -> None:
    """Read this node's slice of one block back into its /dev/shm slot."""
    global _ODIRECT_OK
    dst = mv[offset : offset + nbytes]
    direct = _aligned(offset, nbytes)
    try:
        fd = os.open(path, os.O_RDONLY | (os.O_DIRECT if direct else 0))
    except OSError:
        direct = False
        fd = os.open(path, os.O_RDONLY)
    try:
        got = 0
        while got < nbytes:
            try:
                n = os.readv(fd, [dst[got:]])
            except OSError as e:
                if direct and e.errno == errno.EINVAL and got == 0:
                    _ODIRECT_OK = False
                    logger.warning(
                        "[dsv4-shard] O_DIRECT read rejected (EINVAL); "
                        "falling back to buffered I/O for the rest of this run"
                    )
                    os.close(fd)
                    fd = -1        # so `finally` cannot double-close on reopen failure
                    fd = os.open(path, os.O_RDONLY)
                    direct = False
                    continue
                raise
            if n == 0:
                raise OSError(f"short read of {path}: {got}/{nbytes} bytes")
            got += n
    finally:
        if fd >= 0:
            os.close(fd)
    # Never unlink on failure -- a transient read error must not become
    # permanent cache loss (stock tiering/fs/io.py does exactly that).


class ShardAgent:
    """Per-node disk worker. One instance per vLLM worker process.

    Holds the node's own ``SharedOffloadRegion`` and its own ``/kvdisk``, and
    services store/load/evict commands from the head. All I/O is this node's
    slot only.
    """

    def __init__(
        self,
        region,
        slice_bytes: int,
        world_size: int,
        global_rank: int,
        root_dir: str,
        offloading_spec,
        head_addr: str,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
    ) -> None:
        self._slice_bytes = int(slice_bytes)
        # GLM53-ROW-STRIDE: newer vLLM builds SharedOffloadRegion with
        # kv_bytes_per_block (row stride) and cpu_page_size (per-worker slice)
        # as INDEPENDENT parameters, so the row may carry padding beyond
        # slice_bytes * world_size (measured: +1024 B on GLM-5.3). Take the row
        # stride from the region rather than recomputing it; guessing it wrong
        # offsets every read/write inside each slot -- silent cross-slot
        # corruption, not a crash.
        _computed_row = self._slice_bytes * int(world_size)
        self._row_stride = int(getattr(region, "_row_stride", _computed_row))
        # Must match SharedOffloadRegion's own arithmetic. region.rank is the
        # LOCAL device index; _worker_offset drifts as create_next_view() is
        # called, so recompute rather than reading it back.
        local_rank = 0 if region.rank is None else int(region.rank)
        self._slot_off = local_rank * self._slice_bytes
        self._rank = int(global_rank)
        self._root_dir = root_dir
        self._head_addr = head_addr

        # The old equality check (row == slice*world) no longer holds; what must
        # still be true is that every worker's slice FITS inside the row and
        # this node's slot does not overrun it. Same guarantee, fail-closed.
        if _computed_row > self._row_stride or (
            self._slot_off + self._slice_bytes > self._row_stride
        ):
            raise RuntimeError(
                f"shard slot does not fit the region row: row={self._row_stride} "
                f"slice={self._slice_bytes} world={int(world_size)} "
                f"slot_off={self._slot_off} (need slot_off+slice <= row)"
            )

        self._mv = memoryview(region.mmap_obj)
        # GLM53-BSF: newer vLLM removed block_size_factor; the successor is
        # blocks_per_chunk (GPU blocks per offloaded chunk).
        _blocks_per_file = int(
            getattr(
                offloading_spec,
                "block_size_factor",
                getattr(offloading_spec, "blocks_per_chunk", 1),
            )
        )
        # GLM53-FILEMAPPER-KW: the kwarg was renamed gpu_blocks_per_file ->
        # blocks_per_file. Introspect rather than guess, so this works on both
        # vLLM generations.
        import inspect as _inspect

        _fm_params = _inspect.signature(FileMapper.from_offloading_spec).parameters
        _bpf_kw = (
            "blocks_per_file" if "blocks_per_file" in _fm_params else "gpu_blocks_per_file"
        )
        self._mapper = FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=offloading_spec,
            **{_bpf_kw: _blocks_per_file},
        )
        # Keep each rank's blocks in its own subtree. The disks are already
        # separate, but this makes a mistakenly-shared root self-diagnosing
        # instead of silently mixing two ranks' shards.
        self._mapper.rank = self._rank

        self._known_dirs: set[str] = set()
        self._pool = DualQueueThreadPool(
            n_read_threads, n_write_threads, thread_name_prefix="dsv4_shard"
        )
        self._stop = False
        self._n_store = 0
        self._n_load = 0
        self._n_evict = 0
        self._n_fail = 0
        self._thread = threading.Thread(
            target=self._run, name="dsv4-shard-agent", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def shutdown(self) -> None:
        """Stop I/O, then drop the region view.

        Order matters and mirrors CPUPrimaryTierOffloadingManager.shutdown():
        the mmap cannot be closed while a memoryview over it is still
        exported, so the view has to go before the region is cleaned up.
        """
        self._stop = True
        self._pool.shutdown(wait=True)
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        try:
            self._mv.release()
        except BufferError:
            logger.warning("[dsv4-shard] region view still exported at shutdown")

    # ------------------------------------------------------------- internals

    def _path(self, key: bytes) -> str:
        return self._mapper.get_file_name(OffloadKey(key))

    def _offset(self, block_id: int) -> int:
        return block_id * self._row_stride + self._slot_off

    def _ensure_dir(self, path: str) -> None:
        d = os.path.dirname(path)
        if d not in self._known_dirs:
            os.makedirs(d, exist_ok=True)
            self._known_dirs.add(d)

    def _inventory(self) -> list[bytes]:
        """Rebuild the key set already on this node's disk, oldest first.

        Filenames encode everything needed to reconstruct the OffloadKey:
        ``<hhh>/<hh>_g<group>/<hash_hex>.bin`` -- see FileMapper.get_file_name.
        """
        found: list[tuple[float, bytes]] = []
        root = f"{self._mapper.base_path}_r{self._rank}"
        try:
            for dirpath, _dirs, files in os.walk(root):
                group_part = os.path.basename(dirpath)
                if "_g" not in group_part:
                    continue
                try:
                    group_idx = int(group_part.rsplit("_g", 1)[1])
                except ValueError:
                    continue
                for fn in files:
                    if fn.endswith(".tmp"):
                        # Leftover from a writer that died mid-store. Never a
                        # valid block; remove it rather than leave it to age.
                        try:
                            os.unlink(os.path.join(dirpath, fn))
                        except OSError:
                            pass
                        continue
                    if not fn.endswith(".bin"):
                        continue
                    try:
                        st = os.stat(os.path.join(dirpath, fn))
                    except OSError:
                        continue
                    if st.st_size != self._slice_bytes:
                        # Truncated or written by a different geometry.
                        continue
                    try:
                        h = bytes.fromhex(fn[: -len(".bin")])
                    except ValueError:
                        continue
                    found.append(
                        (st.st_mtime, h + group_idx.to_bytes(4, "big", signed=False))
                    )
        except OSError as e:
            logger.warning("[dsv4-shard] inventory scan of %s failed: %s", root, e)
            return []
        found.sort(key=lambda t: t[0])
        return [k for _m, k in found]

    def _run(self) -> None:
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.IDENTITY, f"dsv4-rank{self._rank}".encode())
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self._head_addr)

        inv = self._inventory()
        logger.info(
            "[dsv4-shard] agent rank=%d slot_off=%d slice=%d B root=%s -> head %s "
            "(%d blocks already on disk, %.1f GiB)",
            self._rank, self._slot_off, self._slice_bytes, self._root_dir,
            self._head_addr, len(inv), len(inv) * self._slice_bytes / (1 << 30),
        )
        # DEALER queues until the head binds, so ordering with get_manager()
        # in the other process does not matter.
        sock.send(_ENC.encode({"t": "hello", "rank": self._rank, "keys": inv}))

        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        while not self._stop:
            try:
                if poller.poll(20):
                    while True:
                        try:
                            raw = sock.recv(zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        try:
                            self._handle(raw, sock)
                        except Exception as e:
                            logger.error(
                                "[dsv4-shard] agent failed to handle a frame: %s",
                                e,
                            )
                for job_id, ok, *_ in self._pool.get_finished():  # GLM53-POOL-UNPACK: now (job_id, ok, transfer_time)
                    if not ok:
                        self._n_fail += 1
                    sock.send(
                        _ENC.encode({"t": "ack", "job": int(job_id), "ok": bool(ok)})
                    )
            except zmq.ZMQError as e:
                if self._stop:
                    break
                logger.error("[dsv4-shard] agent socket error: %s", e)
                time.sleep(0.1)
        sock.close(0)

    def _handle(self, raw: bytes, sock) -> None:
        msg = _DEC.decode(raw)
        kind = msg.get("t")
        if kind in ("store", "load"):
            job_id = int(msg["job"])
            keys = msg["keys"]
            bids = msg["bids"]
            if not keys:
                # DualQueueThreadPool would never complete a 0-task JobState.
                sock.send(_ENC.encode({"t": "ack", "job": job_id, "ok": True}))
                return
            paths = [self._path(k) for k in keys]
            if kind == "store":
                for p in paths:
                    self._ensure_dir(p)
                self._n_store += len(keys)
                self._pool.enqueue_store(
                    job_id,
                    len(keys),
                    [
                        _bind(_store_one, p, self._mv, self._offset(int(b)),
                              self._slice_bytes)
                        for p, b in zip(paths, bids)
                    ],
                )
            else:
                self._n_load += len(keys)
                self._pool.enqueue_load(
                    job_id,
                    len(keys),
                    [
                        _bind(_load_one, p, self._mv, self._offset(int(b)),
                              self._slice_bytes)
                        for p, b in zip(paths, bids)
                    ],
                )
        elif kind == "evict":
            for k in msg["keys"]:
                try:
                    os.unlink(self._path(k))
                    self._n_evict += 1
                except FileNotFoundError:
                    pass
                except OSError as e:
                    logger.warning("[dsv4-shard] unlink failed: %s", e)
        elif kind == "stats":
            sock.send(
                _ENC.encode(
                    {
                        "t": "stats",
                        "rank": self._rank,
                        "stores": self._n_store,
                        "loads": self._n_load,
                        "evicts": self._n_evict,
                        "failures": self._n_fail,
                    }
                )
            )
        elif kind == "rehello":
            # The head asks us to re-report our on-disk inventory (e.g. after a
            # peer reconnected with a changed inventory). Rescan and re-send the
            # same hello frame we send at startup.
            sock.send(
                _ENC.encode(
                    {"t": "hello", "rank": self._rank, "keys": self._inventory()}
                )
            )
        elif kind == "bye":
            self._stop = True


def _bind(fn, *args):
    return lambda: fn(*args)


# ----------------------------------------------------------------- head side


class _JobAcc:
    """Completion accumulator for one job, across all nodes."""

    __slots__ = ("need", "acks", "ok", "deadline", "keys", "is_promotion")

    def __init__(self, need: int, keys: list[bytes], is_promotion: bool,
                 deadline: float) -> None:
        self.need = need
        self.acks = 0
        self.ok = True
        self.keys = keys
        self.is_promotion = is_promotion
        self.deadline = deadline


class DistributedShardTier(SecondaryTierManager):
    """Scheduler-side coordinator for per-node disk shards.

    Moves no KV bytes. Owns the authoritative index of blocks that every node
    holds, the LRU/capacity policy, and the all-nodes completion barrier.
    """

    def __init__(
        self,
        offloading_spec,
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str = "/kvdisk",
        max_bytes: int | float = 0,
        bind_host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        job_timeout_s: float = 900.0,
    ) -> None:
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        # GLM53-ABC-PORT: newer vLLM moved these onto a precomputed
        # OffloadingConfig; the spec no longer carries vllm_config at all.
        _cfg = getattr(offloading_spec, "config", None)
        if _cfg is not None and hasattr(_cfg, "parallel"):
            self._num_agents = int(_cfg.parallel.world_size)
            # cpu_page_size_per_worker, NOT worker_kv_bytes_per_block: the
            # latter is the region's ROW stride and omits per-worker padding.
            self._slice_bytes = int(
                getattr(
                    offloading_spec,
                    "cpu_page_size_per_worker",
                    _cfg.worker_kv_bytes_per_block,
                )
            )
        else:  # older vLLM (DeepSeek image)
            pc = offloading_spec.vllm_config.parallel_config
            self._num_agents = int(pc.world_size)
            self._slice_bytes = int(offloading_spec.cpu_page_size_per_worker)
        self._max_bytes = int(max_bytes)
        self._root_dir = root_dir
        self._job_timeout_s = float(job_timeout_s)

        # Present on EVERY node. LRU order: least-recently-used first.
        self._present: OrderedDict[bytes, None] = OrderedDict()
        self._bytes = 0
        # Keys with I/O in flight; never evict these out from under a reader.
        self._pinned: dict[bytes, int] = {}
        self._jobs: dict[int, _JobAcc] = {}
        self._done: list[JobResult] = []

        self._identities: dict[bytes, int] = {}
        self._hellos: dict[int, list[bytes]] = {}
        self._ready = False

        self._evicted_blocks = 0
        self._evicted_bytes = 0
        self._n_store_jobs = 0
        self._n_load_jobs = 0
        self._n_timeouts = 0

        ctx = zmq.Context.instance()
        self._sock = ctx.socket(zmq.ROUTER)
        # Surface sends to a peer that has not registered instead of dropping
        # them silently -- a dropped job would pin CPU blocks forever.
        self._sock.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self._sock.setsockopt(zmq.LINGER, 0)
        bind = f"tcp://{bind_host}:{int(port)}"
        self._sock.bind(bind)

        logger.info(
            "[dsv4-shard] head bound %s, expecting %d agents; per-node cap "
            "%.1f GiB, slice=%d B (~%d blocks/node, %.1f GiB of full blocks "
            "across the cluster)",
            bind, self._num_agents, self._max_bytes / (1 << 30),
            self._slice_bytes,
            self._max_bytes // max(self._slice_bytes, 1),
            self._num_agents * self._max_bytes / (1 << 30),
        )
        if self._max_bytes <= 0:
            logger.warning(
                "[dsv4-shard] max_bytes<=0: running UNBOUNDED, disks will fill."
            )

    # ------------------------------------------------------------- transport

    def _broadcast(self, payload: dict) -> bool:
        raw = _ENC.encode(payload)
        for ident in self._identities:
            try:
                self._sock.send_multipart([ident, raw])
            except zmq.ZMQError as e:
                logger.error("[dsv4-shard] send to %s failed: %s", ident, e)
                return False
        return True

    def _drain(self) -> None:
        while True:
            try:
                frames = self._sock.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                return
            except zmq.ZMQError as e:
                logger.error("[dsv4-shard] recv failed: %s", e)
                return
            if len(frames) != 2:
                continue
            ident, raw = frames
            try:
                msg = _DEC.decode(raw)
            except Exception as e:
                logger.error("[dsv4-shard] undecodable frame from %s: %s", ident, e)
                continue
            kind = msg.get("t")
            if kind == "hello":
                self._on_hello(ident, msg)
            elif kind == "ack":
                self._on_ack(int(msg["job"]), bool(msg["ok"]))

    def _on_hello(self, ident: bytes, msg: dict) -> None:
        rank = int(msg["rank"])
        self._identities[ident] = rank

        # A re-registration after READY means an agent reconnected with a
        # possibly-changed inventory (e.g. its process restarted). The
        # reconciled index is now stale, so discard it and ask EVERY agent to
        # re-report before serving again -- otherwise the reconnect leaves
        # _hellos at 1/N forever and the head keeps serving a stale index.
        if self._ready:
            logger.warning(
                "[dsv4-shard] agent rank=%d reconnected after READY; discarding "
                "the stale reconciled index and re-requesting inventory from "
                "all %d agents",
                rank, self._num_agents,
            )
            self._ready = False
            self._present.clear()
            self._bytes = 0
            self._hellos.clear()
            self._broadcast({"t": "rehello"})

        self._hellos[rank] = [bytes(k) for k in msg.get("keys", ())]
        logger.info(
            "[dsv4-shard] agent rank=%d registered (%d blocks on its disk); "
            "%d/%d agents up",
            rank, len(self._hellos[rank]), len(self._hellos), self._num_agents,
        )
        if len(self._hellos) < self._num_agents:
            return

        # Reconcile: a block is usable only if EVERY node still holds its
        # shard. Anything else is a half-block, which is precisely the state
        # that produced fluent-but-wrong output before.
        ranks = sorted(self._hellos)
        common: set[bytes] | None = None
        for r in ranks:
            s = set(self._hellos[r])
            common = s if common is None else (common & s)
        common = common or set()

        # Order by the head's own inventory (mtime-sorted by the agent) so the
        # restored LRU keeps its coldest-first shape.
        ordered = [k for k in self._hellos[ranks[0]] if k in common]
        for k in ordered:
            self._present[k] = None
        self._bytes = len(ordered) * self._slice_bytes

        for r in ranks:
            orphans = [k for k in self._hellos[r] if k not in common]
            if orphans:
                logger.warning(
                    "[dsv4-shard] rank %d holds %d blocks no peer has; dropping "
                    "them (a partial block cannot be served correctly)",
                    r, len(orphans),
                )
        stale = set()
        for r in ranks:
            stale |= {k for k in self._hellos[r] if k not in common}
        if stale:
            self._broadcast({"t": "evict", "keys": list(stale)})

        self._hellos.clear()
        self._ready = True
        logger.info(
            "[dsv4-shard] READY: %d blocks present on all %d nodes (%.1f GiB "
            "per node)",
            len(self._present), self._num_agents, self._bytes / (1 << 30),
        )

    def _on_ack(self, job_id: int, ok: bool) -> None:
        acc = self._jobs.get(job_id)
        if acc is None:
            return
        acc.acks += 1
        acc.ok = acc.ok and ok
        if acc.acks < acc.need:
            return
        self._finish(job_id, acc, acc.ok)

    def _finish(self, job_id: int, acc: _JobAcc, ok: bool) -> None:
        self._jobs.pop(job_id, None)
        for k in acc.keys:
            n = self._pinned.get(k, 0) - 1
            if n <= 0:
                self._pinned.pop(k, None)
            else:
                self._pinned[k] = n
        if ok:
            if not acc.is_promotion:
                # Only now is the block genuinely on every node's disk.
                self._admit(acc.keys)
        else:
            # Do not keep serving a block we could not fully read or write.
            # A failed job must not yank a block out from under a concurrent
            # in-flight job (e.g. two requests restoring the same prefix). The
            # pin count for this job's keys was already decremented above, so a
            # key still in _pinned belongs to another live job: leave it alone.
            doomed = [k for k in acc.keys if k not in self._pinned]
            self._forget(doomed)
            # And tell the agents to delete it. Dropping it from the index alone
            # is not enough: a timed-out or failed store can leave a
            # correctly-named, correctly-sized but TORN file on disk, which the
            # next boot's inventory scan (which can only check the size) would
            # adopt and serve. Forgetting locally while leaving the bytes behind
            # is precisely how a half-written block becomes a silent bad hit.
            if doomed:
                self._broadcast({"t": "evict", "keys": doomed})
        self._done.append(JobResult(job_id=job_id, success=ok))

    # ------------------------------------------------------------ index/LRU

    def _admit(self, keys: Iterable[bytes]) -> None:
        for k in keys:
            if k in self._present:
                self._present.move_to_end(k)
            else:
                self._present[k] = None
                self._bytes += self._slice_bytes

    def _forget(self, keys: Iterable[bytes]) -> None:
        # _present maps key -> None (membership only), so pop(k, None) cannot
        # tell present from absent by its return value -- the byte counter
        # would never be decremented and the capacity accounting would drift.
        for k in keys:
            if k in self._present:
                del self._present[k]
                self._bytes -= self._slice_bytes

    def _touch(self, keys: Iterable[bytes]) -> None:
        for k in keys:
            if k in self._present:
                self._present.move_to_end(k)

    def _evict_for(self, incoming_bytes: int) -> None:
        if self._max_bytes <= 0 or not incoming_bytes:
            return
        victims: list[bytes] = []
        skipped = 0
        while self._present and self._bytes + incoming_bytes > self._max_bytes:
            for k in self._present:
                if k not in self._pinned:
                    break
            else:
                # Everything left is in flight; stop rather than spin.
                skipped += 1
                break
            del self._present[k]
            self._bytes -= self._slice_bytes
            victims.append(k)
        if victims:
            self._evicted_blocks += len(victims)
            self._evicted_bytes += len(victims) * self._slice_bytes
            self._broadcast({"t": "evict", "keys": victims})
        if skipped:
            logger.warning(
                "[dsv4-shard] cap exceeded but all resident blocks are pinned; "
                "%d bytes over budget this step", incoming_bytes
            )

    # ------------------------------------------------------- tier interface

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext | None = None):
        # GLM53-ABC-PORT: the ABC now returns a LookupResult enum. Returning a
        # bool here is silently wrong -- LookupResult uses auto(), so True is
        # never == LookupResult.HIT and every block reads as a miss.
        if not self._ready:
            return _LookupResult.MISS if _LookupResult is not None else False
        k = bytes(key)
        if k in self._present:
            self._present.move_to_end(k)
            return _LookupResult.HIT if _LookupResult is not None else True
        return _LookupResult.MISS if _LookupResult is not None else False

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        self._touch(bytes(k) for k in keys)

    @override
    def submit_store(self, job_metadata: JobMetadata) -> None:
        job_id = int(job_metadata.job_id)
        keys = [bytes(k) for k in job_metadata.keys]
        bids = [int(b) for b in job_metadata.block_ids]

        if not self._ready:
            # Agents are not all up yet. Complete immediately without caching
            # so the framework releases its ref_cnt on the primary blocks.
            self._done.append(JobResult(job_id=job_id, success=True))
            return

        # A store is a use: keep these blocks hot before deciding what to
        # evict, so an incoming batch never evicts its own neighbours.
        self._touch(keys)

        # Skip blocks every node already has -- rewriting them is pure I/O.
        # The framework still releases ref_cnt for the FULL key set, because
        # complete_read() uses job_metadata.keys, not what we chose to send.
        fresh = [(k, b) for k, b in zip(keys, bids) if k not in self._present]
        if not fresh:
            self._done.append(JobResult(job_id=job_id, success=True))
            return

        self._evict_for(len(fresh) * self._slice_bytes)
        fkeys = [k for k, _b in fresh]
        fbids = [b for _k, b in fresh]
        for k in fkeys:
            self._pinned[k] = self._pinned.get(k, 0) + 1
        self._jobs[job_id] = _JobAcc(
            self._num_agents, fkeys, False, time.monotonic() + self._job_timeout_s
        )
        self._n_store_jobs += 1
        if not self._broadcast(
            {"t": "store", "job": job_id, "keys": fkeys, "bids": fbids}
        ):
            self._finish(job_id, self._jobs[job_id], False)

    @override
    def submit_load(self, job_metadata: JobMetadata) -> None:
        job_id = int(job_metadata.job_id)
        keys = [bytes(k) for k in job_metadata.keys]
        bids = [int(b) for b in job_metadata.block_ids]

        if not self._ready:
            # Agents are not all up (or just reconnected after READY): we
            # cannot restore reliably. Fail immediately so the primary tier
            # recomputes instead of letting the job sit until its timeout
            # against agents that will never ack it.
            self._done.append(JobResult(job_id=job_id, success=False))
            return

        self._touch(keys)
        for k in keys:
            self._pinned[k] = self._pinned.get(k, 0) + 1
        self._jobs[job_id] = _JobAcc(
            self._num_agents, keys, True, time.monotonic() + self._job_timeout_s
        )
        self._n_load_jobs += 1
        if not self._broadcast(
            {"t": "load", "job": job_id, "keys": keys, "bids": bids}
        ):
            self._finish(job_id, self._jobs[job_id], False)

    @override
    def get_finished(self) -> Iterable[JobResult]:
        self._drain()
        if self._jobs:
            now = time.monotonic()
            for job_id, acc in list(self._jobs.items()):
                if now > acc.deadline:
                    self._n_timeouts += 1
                    logger.error(
                        "[dsv4-shard] job %d timed out after %.0fs with %d/%d "
                        "acks -- failing it so the primary tier can release "
                        "its blocks", job_id, self._job_timeout_s, acc.acks,
                        acc.need,
                    )
                    self._finish(job_id, acc, False)
        out = self._done
        self._done = []
        return out

    # ---------------- GLM53-ABC-PORT: newer-vLLM SecondaryTierManager surface

    def get_finished_jobs(self):
        """Renamed from get_finished() in newer vLLM. Same semantics."""
        return self.get_finished()

    def has_pending_work(self) -> bool:
        """GLM53-ABC-PORT: this tier completes jobs on remote-agent ACKs, which
        arrive asynchronously. The ABC default is False ("does not need the
        engine to keep stepping"), which would leave outstanding jobs uncollected
        whenever no request is scheduled -- stores would hang instead of
        finishing. While True, on_schedule_end() and get_finished_jobs() keep
        being called.

        _done must keep the engine stepping too: submit_store/submit_load
        push early results there without ever creating a _jobs entry (e.g.
        when not ready), so "no jobs in flight" must not mean "nothing to
        hand back".
        """
        return bool(self._jobs) or bool(self._done)

    def on_new_request(self, req_context: ReqContext):
        """New abstract method. fs/manager.py returns the default context
        (OffloadPolicy.BLOCK_LEVEL); this tier has no different preference."""
        if _RequestOffloadingContext is None:
            return None
        return _RequestOffloadingContext()

    def drain_jobs(self) -> None:
        """Block until every submitted job has completed or failed.

        fs/manager.py delegates to its threadpool's wait_idle(), but this tier
        is ack-driven: a job completes when every remote ShardAgent acks it, so
        we pump _drain() until _jobs empties.

        Per the ABC contract we must NOT abort a mid-flight transfer -- a
        partial copy would corrupt the primary memoryview or the backing store
        -- so we wait out the job timeout rather than cancelling. Anything
        still outstanding at the deadline is failed explicitly, which is what
        the contract requires: its result must still surface from
        get_finished_jobs() so the primary tier can release its blocks.
        """
        deadline = time.monotonic() + self._job_timeout_s
        while self._jobs and time.monotonic() < deadline:
            self._drain()
            if not self._jobs:
                break
            time.sleep(0.005)
        if self._jobs:
            logger.error(
                "[dsv4-shard] drain_jobs: %d job(s) still unacked after %.0fs;"
                " failing them so the primary tier can release its blocks",
                len(self._jobs),
                self._job_timeout_s,
            )
            for job_id, acc in list(self._jobs.items()):
                self._finish(job_id, acc, False)

    def stats(self) -> dict:
        return {
            "ready": self._ready,
            "agents": len(self._identities),
            "blocks": len(self._present),
            "bytes_per_node": self._bytes,
            "cap_bytes_per_node": self._max_bytes,
            "evicted_blocks": self._evicted_blocks,
            "evicted_bytes": self._evicted_bytes,
            "store_jobs": self._n_store_jobs,
            "load_jobs": self._n_load_jobs,
            "timeouts": self._n_timeouts,
            "inflight": len(self._jobs),
        }

    @override
    def shutdown(self) -> None:
        try:
            self._broadcast({"t": "bye"})
        except Exception:
            pass
        self._sock.close(0)


# ------------------------------------------------------------- worker wiring

_AGENT: ShardAgent | None = None


def shard_tier_config(offloading_spec) -> dict | None:
    """Return the shard_dist tier config from the spec, if one is configured."""
    for cfg in getattr(offloading_spec, "secondary_tier_configs", ()) or ():
        if cfg.get("type") == TIER_TYPE:
            return cfg
    return None


def maybe_start_shard_agent(offloading_spec, handlers) -> None:
    """Start this worker's ShardAgent. Called from create_handlers().

    Raises rather than warning on failure: a node without an agent contributes
    no shard, and the resulting cache reads back as zeros for that node's half
    of every block -- the exact silent-corruption mode this tier exists to
    eliminate. Refusing to serve is strictly better.
    """
    global _AGENT
    cfg = shard_tier_config(offloading_spec)
    if cfg is None or _AGENT is not None:
        return

    region = getattr(handlers, "_mmap_region", None)
    if region is None:
        raise RuntimeError(
            "shard_dist tier configured but the worker handlers exposed no "
            "SharedOffloadRegion; cannot locate this node's KV slot."
        )

    # GLM53-AGENT-CONFIG: newer vLLM moved these onto OffloadingConfig; the
    # spec no longer carries vllm_config.
    _cfg = getattr(offloading_spec, "config", None)
    if _cfg is not None and hasattr(_cfg, "parallel"):
        _world_size = int(_cfg.parallel.world_size)
        _global_rank = int(_cfg.parallel.rank)
        # See DistributedShardTier.__init__: the per-worker slice is
        # cpu_page_size_per_worker; worker_kv_bytes_per_block is the row stride.
        _slice_bytes = int(
            getattr(
                offloading_spec,
                "cpu_page_size_per_worker",
                _cfg.worker_kv_bytes_per_block,
            )
        )
    else:  # older vLLM (DeepSeek image)
        pc = offloading_spec.vllm_config.parallel_config
        _world_size = int(pc.world_size)
        _global_rank = int(pc.rank)
        _slice_bytes = int(offloading_spec.cpu_page_size_per_worker)

    _AGENT = ShardAgent(
        region=region,
        slice_bytes=_slice_bytes,
        world_size=_world_size,
        global_rank=_global_rank,
        root_dir=cfg.get("root_dir", "/kvdisk"),
        offloading_spec=offloading_spec,
        head_addr=_head_addr_from_env(),
        n_read_threads=int(cfg.get("n_read_threads", 16)),
        n_write_threads=int(cfg.get("n_write_threads", 16)),
    )
    _AGENT.start()


try:
    SecondaryTierFactory.register_tier(
        TIER_TYPE, "dsv4_shard_tier", "DistributedShardTier"
    )
except ValueError:
    # Already registered (module imported in both scheduler and worker).
    pass
