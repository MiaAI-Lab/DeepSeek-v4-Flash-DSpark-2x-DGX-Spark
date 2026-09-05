# SPDX-License-Identifier: Apache-2.0
"""
Runtime monkeypatches for the vLLM KV-offload disk tier on DeepSeek-V4-Flash-0731
(2x DGX Spark, TP=2 across nodes, vLLM 0.25.2.dev0+g752a3a504.d20260714).

Import this from dsv4_kv_disk_tier.py BEFORE the connector is constructed:

    import dsv4_vllm_patches
    dsv4_vllm_patches.apply_all()

Every patch is individually gated by an environment variable so the whole set
can be bisected without editing code. Defaults are the recommended
first-restart configuration.

    DSV4_PATCH_BUG2_KEEPALIVE=1   retain cuMemcpyBatchAsync descriptor arrays
                                  until the transfer's end_event completes
    DSV4_PATCH_BUG2_WAITSTREAM=0  make CPU->GPU loads wait on the compute
                                  stream (hunk 2; keep OFF on the first run so
                                  the two mechanisms stay bisectable)
    DSV4_PIN_DESCRIPTORS=1        page-lock the descriptor buffers
    DSV4_PATCH_TENSOR_ALIAS=1     stop shutdown() of one handler emptying the
                                  other handler's tensor lists
    DSV4_PATCH_BUG1_IO=1          O_DIRECT alignment + non-destructive load
    DSV4_PATCH_SWLOOKUP=1         stop the 7.4x over-promotion on SWA groups
    DSV4_PATCH_NUMBLOCKS_GUARD=1  actionable error instead of "cannot mmap an
                                  empty file" when the CPU tier rounds to 0
    DSV4_BATCH_LOG=1              log num_copy_ops / refs-per-group per transfer
    DSV4_MAX_COPIES_PER_BATCH=8192  bound copies per driver/SG launch (default
                                  8192). Slices the batch with a per-slice
                                  stream sync so a huge restore can't submit one
                                  unbounded batch to the driver.
    DSV4_SG_THRESHOLD=20000       batches with >= this many descriptors go
                                  through the scatter-gather kernel; below it
                                  the driver's cuMemcpyBatchAsync path is faster
                                  and is used directly (the driver segfaults
                                  above ~23k descriptors, so 20000 stays clear).
    DSV4_MAX_OFFLOAD_BLOCKS_PER_REQUEST=0  cap on the number of offload keys a
                                  single request may store (0 = unlimited).
                                  Stops a prompt that fits in GPU KV from
                                  spilling its whole prefix to disk.

Everything here is pure Python. No vLLM file needs to be bind-mounted and no
container rebuild is required. The file must be importable in BOTH the
EngineCore process on spark1 and the Worker process on spark2.
"""

from __future__ import annotations

import ctypes
import errno
import os
import threading

from vllm.logger import init_logger

logger = init_logger(f"vllm.{__name__}")


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _intenv(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


_APPLIED: set[str] = set()


# ===========================================================================
# BUG 2 - descriptor-array lifetime on cuMemcpyBatchAsync
# ===========================================================================
#
# CONFIDENCE: MEDIUM (mechanism PLAUSIBLE, not CONFIRMED).
#
# vllm/v1/kv_offload/cpu/gpu_worker.py:230-232 allocates all_src/all_dst/
# all_sizes as per-call np.empty(num_copy_ops, int64). Line 295-297 wraps them
# with torch.from_numpy, line 329-334 hands the raw host addresses to
# cuMemcpyBatchAsync (csrc/libtorch_stable/cache_kernels.cu takes
# mutable_data_ptr<int64_t>() and passes it straight to the driver), and line
# 349 returns. The Transfer dataclass (gpu_worker.py:30-36) has only
# job_id/stream/start_event/end_event/num_bytes, so the LAST reference to
# those buffers dies at enqueue time, not at completion time.
#
# At this workload's batch size (~40.6K copy ops => ~325 KB per array) glibc
# serves np.empty from mmap and free() issues munmap, so the arrays are
# unmapped from the worker's address space microseconds after the copy is
# enqueued. CUDA_LAUNCH_BLOCKING=1 masks it because the copy then finishes
# inside swap_blocks_batch() while the arrays are still mapped.
#
# NOT PROVEN: that the driver actually reads the descriptor arrays after
# cuMemcpyBatchAsync returns. The CUDA 13.0 headers do not document the
# lifetime of the srcs/dsts/sizes arrays either way, and the same call passes
# &attr/&attrs_idx/&fail_idx as C++ stack locals with fail_idx read
# immediately after return, which implies at least some synchronous
# validation. What IS confirmed: upstream vLLM main has independently made
# exactly this change (Transfer gained batch_src/batch_dst/batch_sizes, the
# buffers come from a pinned _new_descriptor_buffers() helper, and they are
# returned to a _buffer_pool only inside get_finished() after
# end_event.query()).
#
# IMPLEMENTATION NOTE. Rather than reimplementing the 120-line transfer_async
# (and risking a transcription error), this intercepts
# vllm._custom_ops.swap_blocks_batch - which has exactly ONE caller,
# gpu_worker.py:329 - copies the three descriptor tensors into pooled,
# page-locked buffers, submits the copy from those, and holds them on the
# handler until get_finished() observes the job complete. The extra host->host
# copy is ~3 x 325 KB (tens of microseconds) and buys full control over the
# buffers' lifetime and residency.

_MAX_POOLED_BUFFER_SETS = 4

# Set by the transfer_async wrapper for the duration of one call. The offload
# worker is single-threaded (vllm/v1/kv_offload/worker/worker.py has no Thread
# and no Lock), so a plain module global is correct and is more robust than a
# thread-local if that ever changes: worst case the buffers are returned to
# another handler's pool, which is still safe.
_CTX: dict | None = None

_ORIG_SWAP_BLOCKS_BATCH = None
_ORIG_TRANSFER_ASYNC = None
_ORIG_GET_FINISHED = None
_ORIG_HANDLER_SHUTDOWN = None
_ORIG_HANDLER_INIT = None

_PIN_DESCRIPTORS = True
_WAITSTREAM_ON_LOAD = False
_BATCH_LOG = True
_MAX_COPIES_PER_BATCH = 8192

_log_counter = [0]


def _new_descriptor_buffers(num_copy_ops: int):
    """Allocate one (src, dst, sizes) descriptor-buffer set.

    Page-locked when available: pinned pages are guaranteed resident for the
    DMA path and torch's CachingHostAllocator never returns them to the OS,
    so they can never be munmap()ed out from under an in-flight copy.
    """
    import torch

    pin = _PIN_DESCRIPTORS
    if pin:
        try:
            from vllm.utils.platform_utils import is_pin_memory_available

            pin = bool(is_pin_memory_available())
        except Exception:  # pragma: no cover - defensive
            pin = False
    try:
        return (
            torch.empty(num_copy_ops, dtype=torch.int64, pin_memory=pin),
            torch.empty(num_copy_ops, dtype=torch.int64, pin_memory=pin),
            torch.empty(num_copy_ops, dtype=torch.int64, pin_memory=pin),
        )
    except RuntimeError:
        # cudaHostAlloc can fail under memory pressure. Pageable buffers still
        # fix the lifetime bug, which is the part that matters.
        logger.warning(
            "[dsv4-patch] pinned descriptor allocation failed for n=%d; "
            "falling back to pageable",
            num_copy_ops,
        )
        return (
            torch.empty(num_copy_ops, dtype=torch.int64),
            torch.empty(num_copy_ops, dtype=torch.int64),
            torch.empty(num_copy_ops, dtype=torch.int64),
        )


def _swap_blocks_batch_patched(
    src_ptrs, dst_ptrs, sizes, is_src_access_order_any: bool = False
):
    import torch

    ctx = _CTX
    if ctx is None:
        # Not inside a patched transfer_async. Pass through unchanged.
        return _ORIG_SWAP_BLOCKS_BATCH(
            src_ptrs, dst_ptrs, sizes, is_src_access_order_any=is_src_access_order_any
        )

    handler = ctx["handler"]
    captured = ctx["captured"]
    n = int(src_ptrs.numel())

    # --- BUG 2, hunk 2: barrier against the compute stream on the LOAD path.
    # Upstream gpu_worker.py:311-313 guards stream.wait_stream(
    # torch.cuda.current_stream()) with `if self.gpu_to_cpu:`, so a CPU->GPU
    # load has no barrier against work already queued on the compute stream
    # (including destination-block zeroing), which can then land on top of the
    # restored blocks. Upstream main has removed the guard. We are inside
    # `with torch.cuda.stream(stream)`, so current_stream() here IS the
    # transfer stream; ctx["compute_stream"] was captured before the context
    # manager was entered. The barrier is enqueued before the copy, which is
    # what matters; it lands after start_event.record(), so it is included in
    # the reported transfer_time.
    compute_stream = ctx.get("compute_stream")
    if compute_stream is not None:
        torch.cuda.current_stream().wait_stream(compute_stream)

    if _BATCH_LOG:
        _log_counter[0] += 1
        c = _log_counter[0]
        if c <= 50 or c % 200 == 0 or n > 60000:
            # transfer_type is only set by the tensor-alias patch's
            # _handler_init_patched; don't hard-depend on it (the patches are
            # independently gated).
            tt = getattr(handler, "transfer_type", ("?", "?"))
            refs = getattr(handler, "kv_cache_groups_data_refs", None)
            refs_per_group = [len(r) for r in refs] if refs is not None else None
            logger.info(
                "[dsv4-patch] %s->%s job=%d copy_ops=%d refs_per_group=%s "
                "pinned=%d chunk=%d",
                tt[0],
                tt[1],
                ctx["job_id"],
                n,
                refs_per_group,
                int(_PIN_DESCRIPTORS),
                _MAX_COPIES_PER_BATCH,
            )

    if n == 0:
        return None

    pool = handler._dsv4_pool
    buf = pool.pop() if pool else None
    if buf is None or buf[0].numel() < n:
        buf = _new_descriptor_buffers(n)
    bsrc, bdst, bsz = buf

    # narrow(0, 0, n) is a zero-offset contiguous view, so data_ptr() is the
    # buffer base and size(0) is n - exactly what cache_kernels.cu reads via
    # mutable_data_ptr<int64_t>() and src_ptrs.size(0).
    vsrc = bsrc.narrow(0, 0, n)
    vdst = bdst.narrow(0, 0, n)
    vsz = bsz.narrow(0, 0, n)
    vsrc.copy_(src_ptrs)
    vdst.copy_(dst_ptrs)
    vsz.copy_(sizes)

    # Retain until get_finished() proves end_event completed.
    captured.append(buf)

    # Above DSV4_SG_THRESHOLD descriptors, bypass cuMemcpyBatchAsync entirely and
    # do the scatter-gather in our own kernel. The driver segfaults INSIDE
    # libcuda.so.1 above ~23k descriptors (core dump: PC in libcuda's range, x0=0,
    # i.e. an unchecked NULL from a failed internal allocation; dmesg shows NVRM
    # NV_ERR_NO_MEMORY at the same instant on both nodes). It is not host memory --
    # reproduced with 11-13 GB free -- and chunking with a per-chunk stream sync
    # bounds only the PENDING count, which bought ~25% and no more. A 1M-token
    # restore needs ~244k descriptors, so the driver path cannot reach it at all.
    sg = _sg_copy_threshold()
    if sg > 0 and n >= sg:
        chunk = _MAX_COPIES_PER_BATCH
        if chunk <= 0 or n <= chunk:
            rc = _sg_batch_copy(vsrc, vdst, vsz, n, handler, captured)
        else:
            # Bound the per-launch descriptor count (and therefore the UVA
            # traffic the driver must track at once). The kernel is
            # grid-strided, so the slices are semantically identical to one big
            # launch; synchronising between slices caps the driver's outstanding
            # memdesc/UVA state, which is what exhausts _memdescAllocInternal on
            # very large offloads (NVRM NV_ERR_NO_MEMORY, seen on a 750K-token
            # offload). The descriptor device buffer is reused across slices,
            # and the sync guarantees a slice's upload+launch has drained before
            # the next slice's upload overwrites it.
            rc = 0
            for lo in range(0, n, chunk):
                hi = min(lo + chunk, n)
                rc = _sg_batch_copy(
                    vsrc.narrow(0, lo, hi - lo),
                    vdst.narrow(0, lo, hi - lo),
                    vsz.narrow(0, lo, hi - lo),
                    hi - lo,
                    handler,
                    captured,
                )
                if rc != 0:
                    break
                torch.cuda.current_stream().synchronize()
        if rc == 0:
            return None
        # Do NOT fall through to the driver above the wall: that path is known to
        # segfault at this size, and an exception kills one request while a
        # segfault takes the whole worker down.
        raise RuntimeError(
            f"dsv4-patch: SG copy failed (rc={rc}) for n={n} descriptors, which "
            f"is above the ~23k cuMemcpyBatchAsync wall; refusing to fall back "
            f"to the driver path because it would segfault the worker."
        )

    chunk = _MAX_COPIES_PER_BATCH
    if chunk <= 0 or n <= chunk:
        return _ORIG_SWAP_BLOCKS_BATCH(
            vsrc, vdst, vsz, is_src_access_order_any=is_src_access_order_any
        )

    # EXPERIMENTAL chunking (BUG 3). Copies within one cuMemcpyBatchAsync have
    # no mutual ordering guarantee and every copy targets a distinct block, so
    # partitioning into ordered sub-batches on the same stream is a legal
    # refinement. Sub-batches after the first have a non-zero storage_offset;
    # TensorBase::data_ptr() accounts for storage_offset, so the C++ side sees
    # the sub-batch base.
    # DSV4_CHUNK_SYNC bounds the copies the driver has PENDING, not just the
    # number per call. Chunking alone was measured NOT to fix the large-restore
    # segfault (chunk=16384 verified applied in the log, same fault at the same
    # copy_ops), which is consistent with a limit on outstanding work per stream
    # rather than per cuMemcpyBatchAsync call. Synchronising between sub-batches
    # is the only way to tell those two apart, and the only remaining lever on
    # the ~20k-descriptor wall. Cheap: the copy is ~0.145 s for a 1M restore
    # against ~1.7 s of NVMe feeding it, so a dozen syncs cost nothing.
    sync_between = _flag("DSV4_CHUNK_SYNC", "0")
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        _ORIG_SWAP_BLOCKS_BATCH(
            vsrc.narrow(0, lo, hi - lo),
            vdst.narrow(0, lo, hi - lo),
            vsz.narrow(0, lo, hi - lo),
            is_src_access_order_any=is_src_access_order_any,
        )
        if sync_between and hi < n:
            torch.cuda.current_stream().synchronize()
        if _BATCH_LOG and n > 20000:
            # Locate the fault. If the last line printed is "enqueued lo..hi"
            # for the final sub-batch, the copy itself survived and the segfault
            # is DOWNSTREAM (attention over the restored blocks), not in
            # cuMemcpyBatchAsync. Every root-cause guess so far has been wrong;
            # this distinguishes the two halves of the search space.
            logger.info("[dsv4-patch]   enqueued %d..%d of %d", lo, hi, n)
    return None


def _transfer_async_patched(self, job_id: int, src_spec, dst_spec) -> bool:
    global _CTX
    import torch

    if not hasattr(self, "_dsv4_pool"):
        self._dsv4_pool = []
        self._dsv4_inflight = {}

    captured: list = []
    compute_stream = None
    if _WAITSTREAM_ON_LOAD and not self.gpu_to_cpu:
        compute_stream = torch.cuda.current_stream()

    prev = _CTX
    _CTX = {
        "handler": self,
        "captured": captured,
        "job_id": job_id,
        "compute_stream": compute_stream,
    }
    try:
        return _ORIG_TRANSFER_ASYNC(self, job_id, src_spec, dst_spec)
    finally:
        _CTX = prev
        if captured:
            # Attached even if transfer_async raised after the enqueue: the
            # buffers must outlive the copy regardless. If the job never
            # reports completion the set simply leaks, which is safe.
            self._dsv4_inflight[job_id] = captured


def _get_finished_patched(self):
    results = _ORIG_GET_FINISHED(self)
    inflight = getattr(self, "_dsv4_inflight", None)
    if inflight:
        pool = self._dsv4_pool
        for result in results:
            bufs = inflight.pop(result.job_id, None)
            if not bufs:
                continue
            # Safe: end_event.query() inside the original get_finished()
            # already proved the copy is complete, so the driver can no longer
            # touch these arrays.
            for buf in bufs:
                # Only HOST descriptor buffers may be pooled. The SG path also
                # appends its CUDA staging buffer to `captured` (to keep it
                # alive until the job completes); pooling a device buffer here
                # would poison the pool -- the next small transfer would
                # copy_() host pointers into it (silent cross-device) and pass
                # a device pointer to cuMemcpyBatchAsync, which expects host
                # addresses.
                if len(pool) < _MAX_POOLED_BUFFER_SETS and not buf[0].is_cuda:
                    pool.append(buf)
    return results


def _handler_shutdown_patched(self) -> None:
    try:
        _ORIG_HANDLER_SHUTDOWN(self)
    finally:
        if hasattr(self, "_dsv4_inflight"):
            self._dsv4_inflight.clear()
        if hasattr(self, "_dsv4_pool"):
            self._dsv4_pool.clear()


def _handler_init_patched(self, *args, **kwargs):
    _ORIG_HANDLER_INIT(self, *args, **kwargs)
    # gpu_worker.py:437-452 passes the SAME gpu_tensors/cpu_tensors list
    # objects to both handlers, and shutdown() at :386-387 does
    # src_tensors.clear() / dst_tensors.clear(), so shutting down one handler
    # empties the other's tensor lists. Decouple them.
    self.src_tensors = list(self.src_tensors)
    self.dst_tensors = list(self.dst_tensors)
    # This vLLM's SingleDirectionOffloadingHandler has no transfer_type attr;
    # set it for the patch's direction log (handler.transfer_type[0]->[1]).
    self.transfer_type = ('GPU', 'CPU') if self.gpu_to_cpu else ('CPU', 'GPU')


def apply_bug2_keepalive() -> None:
    """Retain (and page-lock) the cuMemcpyBatchAsync descriptor arrays."""
    global _ORIG_SWAP_BLOCKS_BATCH, _ORIG_TRANSFER_ASYNC, _ORIG_GET_FINISHED
    global _ORIG_HANDLER_SHUTDOWN

    if "bug2" in _APPLIED:
        return

    from vllm import _custom_ops as ops
    from vllm.v1.kv_offload.cpu.gpu_worker import SingleDirectionOffloadingHandler

    _ORIG_SWAP_BLOCKS_BATCH = ops.swap_blocks_batch
    _ORIG_TRANSFER_ASYNC = SingleDirectionOffloadingHandler.transfer_async
    _ORIG_GET_FINISHED = SingleDirectionOffloadingHandler.get_finished
    _ORIG_HANDLER_SHUTDOWN = SingleDirectionOffloadingHandler.shutdown

    ops.swap_blocks_batch = _swap_blocks_batch_patched
    SingleDirectionOffloadingHandler.transfer_async = _transfer_async_patched
    SingleDirectionOffloadingHandler.get_finished = _get_finished_patched
    SingleDirectionOffloadingHandler.shutdown = _handler_shutdown_patched

    _APPLIED.add("bug2")
    logger.info(
        "[dsv4-patch] BUG2 descriptor keepalive ACTIVE "
        "(pin=%s wait_stream_on_load=%s max_copies_per_batch=%d)",
        _PIN_DESCRIPTORS,
        _WAITSTREAM_ON_LOAD,
        _MAX_COPIES_PER_BATCH,
    )


def apply_tensor_alias_fix() -> None:
    """Stop one handler's shutdown() from emptying the other's tensor lists."""
    global _ORIG_HANDLER_INIT

    if "tensor_alias" in _APPLIED:
        return

    from vllm.v1.kv_offload.cpu.gpu_worker import SingleDirectionOffloadingHandler

    _ORIG_HANDLER_INIT = SingleDirectionOffloadingHandler.__init__
    SingleDirectionOffloadingHandler.__init__ = _handler_init_patched
    _APPLIED.add("tensor_alias")
    logger.info("[dsv4-patch] handler tensor-list aliasing fix ACTIVE")


# ===========================================================================
# BUG 1 - O_DIRECT alignment, and non-destructive load failures
# ===========================================================================
#
# CONFIDENCE: HIGH for the mechanism, verified against the real device.
#
# FileSystemTierManager transfers one primary-tier ROW per file:
# tiering/fs/manager.py:79 sets _block_size = primary_kv_view.strides[0]
# (= cpu_page_size_per_worker * num_workers = 2,103,552 B here) and :112-132
# pass int(bid) * _block_size as the offset. tiering/fs/io.py:87-88 opens the
# file with O_DIRECT and issues one os.readv() of _block_size bytes.
#
# The kernel checks TWO different limits (fs/iomap/direct-io.c):
#   (file_offset | length) & (bdev_logical_block_size(bdev) - 1)  -> EINVAL
#   buffer_address              & bdev_dma_alignment(bdev)        -> EINVAL
# They are NOT the same number. On /kvdisk (ext4 on nvme0n1p2):
#   logical_block_size = 512, dma_alignment = 3 (i.e. 4-byte addresses).
# statvfs f_bsize is 4096 and is the WRONG source - a 2,103,296-byte read is
# 4096-misaligned yet succeeds.
#
# 2,103,552 % 512 == 256, so at block_size_factor=1 every load returns EINVAL.
# At bsf=4 the block is 8,414,208 B, which IS 512-aligned, so stock O_DIRECT
# reads are legal at the currently deployed factor - re-test at bsf=1 to see
# the original EINVAL. bsf=1 is the DEFAULT (base.py:373), so this is the
# stock configuration, not an exotic one.
#
# A naive fix that also demands 512-byte ADDRESS alignment would force fully
# buffered I/O on every odd block index (offset 256 within the row), i.e.
# exactly the 50% it exists to rescue. Hence the two-value model below.
#
# Second defect, unconditional: io.py:91-98 os.remove()s the source file on ANY
# exception, and the failed job then drives tiering/manager.py:200-207 ->
# cpu/manager.py:210-215 to free the CPU entry, destroying the block in BOTH
# tiers. But simply deleting the unlink is WRONG: fs/manager.py:106 lookup()
# is a bare os.path.exists(), so a permanently unreadable file would make
# _initiate_promotion re-fire every step and defer the request forever. The
# unlink is a liveness mechanism. Correct behaviour: retry buffered in-process
# first (which is what an alignment EINVAL needs), and only retire the file if
# THAT also fails.

O_DIRECT = getattr(os, "O_DIRECT", 0)

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]

_DEFAULT_IO_ALIGNMENT = 4096
_MIN_IO_ALIGNMENT = 512
_MAX_IO_ALIGNMENT = 65536
# Granularity the BUFFER ADDRESS must satisfy. The kernel checks this against
# the queue's DMA alignment, which is far weaker than the logical block size
# (this NVMe reports a mask of 3, i.e. 4 bytes).
_MIN_MEM_ALIGNMENT = 4

# st_dev -> (mem_align, io_align). (0, 0) means "never use O_DIRECT here".
_dio_alignment_cache: dict[int, tuple[int, int]] = {}

_ORIG_STORE_BLOCK = None
_ORIG_LOAD_BLOCK = None

_tmp_suffix_local = threading.local()


def _get_tmp_suffix() -> str:
    """Thread-local unique suffix for temp files.

    Stock io.py:18-24 seeds from random.randint, which is fork-unsafe: vLLM
    forks workers, so a shared pre-fork RNG state can hand two processes the
    same suffix. Bind to pid + thread id + OS entropy instead.
    """
    try:
        return _tmp_suffix_local.tmp_suffix
    except AttributeError:
        _tmp_suffix_local.tmp_suffix = (
            f"_{os.getpid()}_{threading.get_ident()}_"
            f"{int.from_bytes(os.urandom(8), 'little')}.tmp"
        )
        return _tmp_suffix_local.tmp_suffix


def _ensure_dirs(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _read_int_file(path: str):
    try:
        with open(path, "rb") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _round_pow2(n: int) -> int:
    if n <= 0:
        return 0
    if n & (n - 1):
        return 1 << n.bit_length()
    return n


def _compute_dio_alignment(st_dev: int, dir_path: str) -> tuple[int, int]:
    """Return (mem_align, io_align) for O_DIRECT on *dir_path*.

    /sys/dev/block/<maj>:<min> is a symlink into /sys/devices/... A whole
    device carries the request queue itself; a partition inherits it from its
    parent, which ".." reaches because the kernel resolves the symlink before
    the "..". The path must therefore NOT be normalised.

    Never raises: any failure degrades to a conservative default.
    """
    dev = f"{os.major(st_dev)}:{os.minor(st_dev)}"
    io_align = 0
    mem_align = 0
    for queue in (
        f"/sys/dev/block/{dev}/queue",
        f"/sys/dev/block/{dev}/../queue",
    ):
        lbs = _read_int_file(f"{queue}/logical_block_size")
        if lbs and lbs > 0:
            io_align = lbs
            dma = _read_int_file(f"{queue}/dma_alignment")
            mem_align = _round_pow2(dma + 1) if dma is not None else lbs
            break
    if io_align <= 0:
        # No block device behind this path (network / FUSE / overlay), or
        # sysfs is not mounted. f_bsize is the FILESYSTEM block size, a
        # coarser and needlessly pessimistic bound, but it is safe.
        try:
            io_align = os.statvfs(dir_path).f_bsize
        except OSError:
            io_align = 0
        io_align = io_align or _DEFAULT_IO_ALIGNMENT
        mem_align = io_align
    io_align = _round_pow2(io_align)
    if io_align > _MAX_IO_ALIGNMENT:
        return (0, 0)
    return (
        max(_round_pow2(mem_align), _MIN_MEM_ALIGNMENT),
        max(io_align, _MIN_IO_ALIGNMENT),
    )


def _disable_direct_io(dir_path: str) -> None:
    try:
        _dio_alignment_cache[os.stat(dir_path).st_dev] = (0, 0)
    except OSError:
        pass


def _buffer_address(view: memoryview):
    try:
        # The temporary keeps a buffer export alive only for this expression.
        return ctypes.addressof(ctypes.c_char.from_buffer(view))
    except (TypeError, ValueError, BufferError):
        # Read-only or empty view: caller falls back to buffered I/O.
        return None


def _direct_io_alignment(dir_path: str, view: memoryview) -> int:
    """Alignment to open an O_DIRECT fd with, or 0 for buffered I/O."""
    if not O_DIRECT or fcntl is None:
        return 0
    try:
        st_dev = os.stat(dir_path).st_dev
    except OSError:
        return 0
    alignments = _dio_alignment_cache.get(st_dev)
    if alignments is None:
        alignments = _compute_dio_alignment(st_dev, dir_path)
        _dio_alignment_cache[st_dev] = alignments
        logger.info(
            "[dsv4-patch] O_DIRECT alignment for %s (dev %d:%d): "
            "mem=%d io=%d",
            dir_path,
            os.major(st_dev),
            os.minor(st_dev),
            alignments[0],
            alignments[1],
        )
    mem_align, io_align = alignments
    if io_align <= 0 or len(view) < io_align:
        return 0
    address = _buffer_address(view)
    if address is None or (address & (mem_align - 1)):
        return 0
    return io_align


def _clear_o_direct(fd: int) -> None:
    assert fcntl is not None
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~O_DIRECT)


def _open_maybe_direct(
    path: str,
    flags: int,
    mode: int = 0o644,
    *,
    direct: bool,
    dir_path: str | None = None,
) -> tuple[int, bool]:
    """Open *path*, requesting O_DIRECT when *direct*.

    Some filesystems (tmpfs, several FUSE/network mounts) reject O_DIRECT at
    open() time. Treat that as "no direct I/O on this device" rather than
    failing the transfer, and remember it so the wasted syscall is paid once.
    O_EXCL is preserved on the retry by unlinking any inode the rejected
    open may already have created.
    """
    if direct:
        try:
            return os.open(path, flags | O_DIRECT, mode), True
        except OSError as exc:
            if exc.errno != errno.EINVAL:
                raise
            logger.debug(
                "[dsv4-patch] O_DIRECT unsupported for %s; using buffered I/O",
                path,
            )
            if flags & os.O_CREAT:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            if dir_path is not None:
                _disable_direct_io(dir_path)
    return os.open(path, flags, mode), False


def _byte_slice(buffer: memoryview, offset: int, block_size: int) -> memoryview:
    """Flat byte view of [offset, offset+block_size).

    memoryview slicing silently CLAMPS, which would turn an out-of-range block
    id into a short, silently-wrong transfer. Stock load_block caught that via
    its `bytes_read < block_size` check; restore the invariant explicitly.
    """
    flat = buffer.cast("B")
    if offset < 0 or offset + block_size > len(flat):
        raise OSError(
            f"block slice [{offset}, {offset + block_size}) out of range "
            f"for a view of {len(flat)} bytes"
        )
    view_slice = flat[offset : offset + block_size]
    if len(view_slice) != block_size:
        raise OSError(
            f"short slice: got {len(view_slice)} bytes, expected {block_size}"
        )
    return view_slice


def _transfer_exact(fd: int, view: memoryview, align: int, *, write: bool) -> bool:
    """Transfer exactly len(view) bytes, looping until done.

    A single read()/write() is not guaranteed to move a multi-megabyte buffer.
    *align* is the alignment O_DIRECT is active with, or 0 for a buffered fd.
    Direct I/O can only move whole logical blocks, so as soon as the remainder
    is shorter than *align* (or a short transfer leaves the offset unaligned)
    O_DIRECT is cleared and the rest is finished buffered. The direct and
    buffered regions are disjoint and strictly sequential on a single fd.

    Returns True if any part of the transfer went through the page cache.
    """
    total = len(view)
    done = 0
    direct = align > 0
    buffered = not direct
    while done < total:
        if direct and ((total - done) < align or (done & (align - 1))):
            _clear_o_direct(fd)
            direct = False
            buffered = True
        if direct:
            chunk = view[done : done + ((total - done) & ~(align - 1))]
        else:
            chunk = view[done:]
        try:
            if write:
                transferred = os.write(fd, chunk)
            else:
                transferred = os.readv(fd, [chunk])
        except OSError as exc:
            if not direct or exc.errno != errno.EINVAL:
                raise
            # The filesystem wants a stricter alignment than the device
            # reports (btrfs uses its sector size, XFS its own geometry).
            # Nothing was transferred, so redo the remainder buffered.
            logger.debug(
                "[dsv4-patch] O_DIRECT rejected a %d byte transfer; "
                "using buffered I/O",
                len(chunk),
            )
            _clear_o_direct(fd)
            direct = False
            buffered = True
            continue
        if transferred <= 0:
            verb, past = ("write", "wrote") if write else ("read", "read")
            raise OSError(
                f"Short {verb}: expected {total} bytes, {past} {done}"
            )
        done += transferred
    return buffered


def _read_block(source_path: str, view_slice: memoryview, align: int) -> None:
    fd, direct = _open_maybe_direct(
        source_path,
        os.O_RDONLY,
        direct=align > 0,
        dir_path=os.path.dirname(source_path),
    )
    try:
        _transfer_exact(fd, view_slice, align if direct else 0, write=False)
    finally:
        os.close(fd)


def store_block(
    dest_path: str, buffer: memoryview, offset: int, block_size: int
) -> None:
    """Store callback: write to a temp file then atomically replace."""
    if os.path.exists(dest_path):
        return

    tmp_path = dest_path + _get_tmp_suffix()
    _ensure_dirs(dest_path)
    dir_path = os.path.dirname(dest_path)

    view_slice = _byte_slice(buffer, offset, block_size)
    align = _direct_io_alignment(dir_path, view_slice)
    try:
        fd, direct = _open_maybe_direct(
            tmp_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_TRUNC,
            0o644,
            direct=align > 0,
            dir_path=dir_path,
        )
        try:
            if _transfer_exact(fd, view_slice, align if direct else 0, write=True):
                # Part of the block went through the page cache and is not
                # necessarily on stable storage. Flush before the rename
                # publishes it, otherwise a crash can leave a visible but
                # incomplete block that would later be read back as silently
                # corrupt KV data. The pure-O_DIRECT path skips this.
                os.fdatasync(fd)
        finally:
            os.close(fd)
        os.replace(tmp_path, dest_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError as cleanup_exc:
            logger.warning(
                "[dsv4-patch] failed to remove temp file %s: %s",
                tmp_path,
                cleanup_exc,
            )
        raise


def load_block(
    source_path: str, view: memoryview, offset: int, block_size: int
) -> None:
    """Load callback: read one KV block from disk.

    Direct first; on any I/O error other than a missing file, retry the whole
    block buffered in-process. Only if THAT fails is the file retired, because
    fs/manager.py lookup() is a bare os.path.exists() and a permanently
    unreadable file that stays on disk defers every request touching it
    forever.
    """
    view_slice = _byte_slice(view, offset, block_size)
    dir_path = os.path.dirname(source_path)
    align = _direct_io_alignment(dir_path, view_slice)

    if align:
        try:
            _read_block(source_path, view_slice, align)
            return
        except FileNotFoundError:
            raise
        except OSError as exc:
            logger.warning(
                "[dsv4-patch] direct read of %s failed (%s); retrying buffered",
                source_path,
                exc,
            )

    try:
        _read_block(source_path, view_slice, 0)
    except Exception:
        try:
            os.remove(source_path)
        except OSError as cleanup_exc:
            logger.warning(
                "[dsv4-patch] failed to remove unreadable file %s: %s",
                source_path,
                cleanup_exc,
            )
        raise


def apply_bug1_io() -> None:
    """Install the corrected store_block / load_block.

    fs/manager.py:32 does `from ...fs.io import load_block, store_block`, so
    the names are bound into the MANAGER module's namespace at import time.
    Both modules are patched.
    """
    global _ORIG_STORE_BLOCK, _ORIG_LOAD_BLOCK

    if "bug1" in _APPLIED:
        return

    from vllm.v1.kv_offload.tiering.fs import io as fs_io
    from vllm.v1.kv_offload.tiering.fs import manager as fs_manager

    _ORIG_STORE_BLOCK = fs_io.store_block
    _ORIG_LOAD_BLOCK = fs_io.load_block

    fs_io.store_block = store_block
    fs_io.load_block = load_block
    fs_manager.store_block = store_block
    fs_manager.load_block = load_block

    _APPLIED.add("bug1")
    logger.info("[dsv4-patch] BUG1 O_DIRECT alignment + safe load ACTIVE")


# ===========================================================================
# Over-promotion: _sliding_window_lookup treats a deferred lookup as a MISS
# ===========================================================================
#
# CONFIDENCE: HIGH. Verified arithmetic reproduces the measured promo_ok=648.
#
# offloading/scheduler.py:305-309 (_maximal_prefix_lookup) sets result = True
# when the backend defers; :328-332 (_sliding_window_lookup) sets result =
# False for the identical condition. With result = False, consecutive_hits is
# reset on every iteration of a cold-tier scan, `consecutive_hits ==
# sliding_window_size` never fires, and the backward loop runs to idx 0 -
# kicking off a promotion for EVERY block in the range. But
# update_state_after_alloc only ever LOADS the trailing window: :586-591
# asserts num_pending_gpu_blocks <= sliding_window_size_in_blocks *
# block_size_factor and :595-599 slices offload_keys[start_block_idx:
# num_blocks].
#
# DeepSeek-V4-Flash, 82,944 tokens, bsf=4, with the alignment_block_count
# filter (scheduler.py:98-131) applied to the measured group layout
# (blocks 256/64/64/4/8 tok, windows -/128/128/8/128):
#     g0  81 promoted, 81 loaded
#     g1  81 promoted,  1 loaded
#     g2  81 promoted,  1 loaded
#     g3  81 promoted,  1 loaded
#     g4 324 promoted,  4 loaded
#   total 648 promoted, 88 loaded  -> 7.4x. 648 == the measured promo_ok.
# Total lookup() CALLS also drop from ~8,505 to ~92 per scan pass.
#
# CRITICAL DETAIL - do NOT "fix" the off-by-one by raising the threshold to
# sliding_window_size + 1. update_state_after_alloc's slice length is
# cdiv(G, f) - L // f, which equals W only when G % f == 0 and is W + 1
# otherwise; every key handed to prepare_load must already be resident or
# cpu/manager.py:121 `assert block is not None` kills the EngineCore. But for
# the W == 1 groups here the store-side alignment filter stores runs of
# exactly one block, so a run never reaches length 2 and a threshold of W + 1
# never fires - the scan degenerates back to a full scan. The correct shape is
# "terminate at W".
#
# (G % f == 0 holds for every group at bsf=4 and bsf=16 on this model, so the
# W vs W+1 boundary is moot here. The deferred path returns None -- mirroring
# stock vLLM 0.25.x -- so no load is issued on a deferred streak and the
# prepare_load assert above is never reached; no preemptive extra lookup is
# needed.)

_ORIG_SW_LOOKUP = None


def _sliding_window_lookup_patched(self, keys, sliding_window_size, req_context):
    """Return the end index (in `keys`) of the last run of
    `sliding_window_size` consecutive hits, scanning from the end.
    Returns 0 on miss, None if the backend deferred a lookup."""
    from vllm.v1.kv_offload.base import LookupResult

    defer_lookup = False
    consecutive_hits = 0
    for idx in range(len(keys) - 1, -1, -1):
        result = self.manager.lookup(keys[idx], req_context)
        if result is LookupResult.HIT:
            consecutive_hits += 1
        elif result is LookupResult.HIT_PENDING:
            # Block is in cache but not yet readable (a store is in-flight).
            # It counts as a hit for the consecutive streak, but the load must
            # be deferred: update_state_after_alloc would otherwise hand this
            # key to prepare_load(), whose stock `assert block.is_ready` kills
            # the EngineCore. This is the vLLM 0.25.x enum API; the old
            # bool/None contract no longer exists, so `if not result` would
            # misclassify every enum member (all truthy) as a hit.
            defer_lookup = True
            consecutive_hits += 1
        elif result is LookupResult.RETRY:
            # Block location uncertain -- does not count as a hit. Keep
            # scanning so the manager can kick off async lookups.
            defer_lookup = True
            consecutive_hits = 0
        else:  # LookupResult.MISS
            consecutive_hits = 0
        if consecutive_hits == sliding_window_size:
            # Identical to stock vLLM 0.25.x. A deferred streak returns None,
            # which makes _lookup() defer the whole request (no load is issued
            # this step, so update_state_after_alloc/prepare_load are never
            # reached). A non-deferred streak returns the end index.
            return idx + sliding_window_size if not defer_lookup else None
    return consecutive_hits if not defer_lookup else None


def apply_sliding_window_lookup() -> None:
    global _ORIG_SW_LOOKUP

    if "swlookup" in _APPLIED:
        return

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    _ORIG_SW_LOOKUP = OffloadingConnectorScheduler._sliding_window_lookup
    OffloadingConnectorScheduler._sliding_window_lookup = (
        _sliding_window_lookup_patched
    )
    _APPLIED.add("swlookup")
    logger.info("[dsv4-patch] sliding-window over-promotion fix ACTIVE")


# ===========================================================================
# num_blocks >= 1 guard
# ===========================================================================
#
# cpu/spec.py:42-46 yields num_blocks = 0 when cpu_bytes_to_use is smaller
# than one offloaded block; tiering/spec.py then builds SharedOffloadRegion(
# total_size_bytes=0) -> ftruncate(fd, 0) -> mmap.mmap(fd, 0), which raises
# "ValueError: cannot mmap an empty file". Reachable purely by raising
# block_size_factor (an 8 GiB tier at bsf=16 already gives only ~255 blocks).

_ORIG_CPU_SPEC_INIT = None


def _cpu_spec_init_patched(self, *args, **kwargs):
    _ORIG_CPU_SPEC_INIT(self, *args, **kwargs)
    num_blocks = getattr(self, "num_blocks", None)
    page = getattr(self, "cpu_page_size_per_worker", 0)
    if num_blocks is not None and page > 0 and num_blocks < 1:
        raise ValueError(
            "kv_connector_extra_config: cpu_bytes_to_use is too small to hold "
            f"a single offloaded block at block_size_factor="
            f"{getattr(self, 'block_size_factor', '?')} "
            f"(cpu_page_size_per_worker={page} B). Raise cpu_bytes_to_use or "
            "lower block_size_factor."
        )


def apply_num_blocks_guard() -> None:
    global _ORIG_CPU_SPEC_INIT

    if "numblocks" in _APPLIED:
        return

    from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

    _ORIG_CPU_SPEC_INIT = CPUOffloadingSpec.__init__
    CPUOffloadingSpec.__init__ = _cpu_spec_init_patched
    _APPLIED.add("numblocks")


# ===========================================================================
# P0: multi-node corruption guard
# ===========================================================================

_ORIG_TIERING_GET_MANAGER = None


def check_multinode_safe(vllm_config) -> None:
    """Raise if a secondary (disk) tier would silently corrupt on this topology.

    Call this from ANY spec that overrides get_manager() -- a subclass override
    shadows the patched parent method, so patching the base class alone is not
    sufficient. (Learned the hard way: the guard silently did nothing because
    MultiGroupTieringOffloadingSpec defines its own get_manager.)
    """
    try:
        import torch

        world = int(vllm_config.parallel_config.world_size)
        local = int(torch.cuda.device_count() or 1)
    except Exception:
        return

    if world <= local:
        return
    if not _flag("DSV4_ALLOW_MULTINODE_TIER", "0"):
        raise RuntimeError(
            "dsv4-patch: refusing to build a disk/secondary KV tier on a "
            f"MULTI-NODE deployment (world_size={world}, local GPUs={local}).\n"
            "The secondary tier is scheduler-side only and its I/O unit spans "
            "ALL workers, but SharedOffloadRegion is /dev/shm and therefore "
            "node-local. The peer node's KV shard is never persisted, and on "
            "restore it is served as ZEROS -- fluent but WRONG output while "
            "promo_ok, hit-rate and IO-failure metrics all look perfect. "
            "Verified here by needle test: secret recovered cold, LOST after a "
            "disk restore.\nUse single-node TP, or set "
            "DSV4_ALLOW_MULTINODE_TIER=1 if you have verified all workers "
            "share the scheduler's /dev/shm."
        )
    logger.warning(
        "[dsv4-patch] MULTI-NODE tier explicitly allowed (world=%d local=%d) "
        "-- restored KV for non-head ranks WILL be zeros. Output correctness "
        "is NOT guaranteed.",
        world, local,
    )


def _multinode_guard_patched(self, *args, **kwargs):
    """Refuse to build a secondary (disk) tier on a MULTI-NODE deployment.

    PROVEN CORRUPTION, not a theoretical risk. Needle-in-haystack test on this
    cluster (67,047-token context, secret buried at 25% depth):

        cold      -> NEEDLE FOUND    'ZQ7-MAGENTA-4419'   39.82 s
        restored  -> NEEDLE MISSING  answer=''             8.34 s
        External prefix cache hit rate: 99.3%
        promo_ok=520  promo_primary_full=0    <- every metric green

    Why. The secondary tier is SCHEDULER-side only, so it exists solely on the
    head node. Its I/O unit is ``primary_kv_view.strides[0]`` == the full row,
    ``cpu_page_size * world_size`` -- bytes belonging to EVERY worker. But
    ``SharedOffloadRegion`` lives at ``/dev/shm/vllm_offload_<id>.mmap``, which
    is NODE-LOCAL, and the worker offset comes from the local device index,
    which is 0 on every single-GPU node. So each node writes worker-area 0 of
    its own file; the head persists its own shard plus a megabyte of zeros
    where the peer's shard belongs; the peer's shard is never persisted at all.
    On restore, rank 1 gets zeros promoted into its GPU shard -> fluent,
    confident, WRONG output, while every head-side metric stays green.

    Detection: the region is node-local, so the deployment is only safe when
    all ``world_size`` workers are on this node. ``torch.cuda.device_count()``
    is the local world size.

    DSV4_ALLOW_MULTINODE_TIER=1 bypasses this ONLY if you have verified every
    worker shares the scheduler's /dev/shm (true single-node TP).
    """
    try:
        import torch

        world = int(self.vllm_config.parallel_config.world_size)
        local = int(torch.cuda.device_count() or 1)
    except Exception:  # never block startup on a detection failure
        return _ORIG_TIERING_GET_MANAGER(self, *args, **kwargs)

    if world > local:
        if not _flag("DSV4_ALLOW_MULTINODE_TIER", "0"):
            raise RuntimeError(
                "dsv4-patch: refusing to build a disk/secondary KV tier on a "
                f"MULTI-NODE deployment (world_size={world}, local GPUs="
                f"{local}).\nThe secondary tier is scheduler-side only and its "
                "I/O unit spans ALL workers, but SharedOffloadRegion is "
                "/dev/shm and therefore node-local. The peer node's KV shard "
                "is never persisted, and on restore it is served as ZEROS -- "
                "producing fluent but WRONG output while promo_ok, hit-rate "
                "and IO-failure metrics all look perfect. Verified here with a "
                "needle test: the secret was recovered cold and LOST after a "
                "disk restore.\nUse single-node TP, or set "
                "DSV4_ALLOW_MULTINODE_TIER=1 if you have verified that all "
                "workers share the scheduler's /dev/shm."
            )
        logger.warning(
            "[dsv4-patch] MULTI-NODE tier explicitly allowed (world=%d "
            "local=%d) -- restored KV for non-head ranks WILL be zeros. "
            "Output correctness is NOT guaranteed.",
            world, local,
        )
    return _ORIG_TIERING_GET_MANAGER(self, *args, **kwargs)


def apply_multinode_guard() -> None:
    """Fail fast instead of silently corrupting on multi-node TP."""
    global _ORIG_TIERING_GET_MANAGER

    if "multinode_guard" in _APPLIED:
        return

    from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

    _ORIG_TIERING_GET_MANAGER = TieringOffloadingSpec.get_manager
    TieringOffloadingSpec.get_manager = _multinode_guard_patched
    _APPLIED.add("multinode_guard")


# ===========================================================================
# Bound eager offload (DSV4_MAX_OFFLOAD_BLOCKS_PER_REQUEST)
# ===========================================================================
#
# vLLM's offloading scheduler (kv_role=kv_both) stores EVERY request's prompt
# blocks eagerly, even when the request fits entirely in GPU KV. On this
# deployment the GPU pool holds ~1.4M tokens, so a single huge prompt (e.g.
# 750K tokens) spills its whole prefix to the CPU staging tier and then to
# disk, driving a large GPU->CPU->disk copy that the NVRM driver's memdesc pool
# cannot always absorb -- and writing bytes to SSD that will never be a useful
# prefix-cache hit (no reuse at that size on a 6-seq deployment).
#
# This caps the number of offload keys a single request may store. Keys past
# the cap are never stored, so the oversized request serves from GPU memory
# alone. 0 = unlimited (stock behaviour). Enforced on the manager's
# prepare_store(), the single chokepoint every eager store goes through.

_MAX_OFFLOAD_BLOCKS = 0
_REQ_OFFLOADED: dict = {}
_ORIG_PREPARE_STORE = None


def _offload_budget_prune(manager, force: bool = False) -> None:
    """Drop counters for requests the manager no longer tracks."""
    if not force and len(_REQ_OFFLOADED) < 10000:
        return
    live = getattr(manager, "_req_state", None)
    for rid in list(_REQ_OFFLOADED):
        # If the manager exposes no _req_state (should not happen for the
        # tiering manager), it cannot tell us which requests are live; dropping
        # everything still bounds the dict instead of leaking unbounded.
        if live is not None and rid in live:
            continue
        del _REQ_OFFLOADED[rid]


def _prepare_store_patched(self, keys, req_context):
    """Clamp the eager store to a per-request offload-key budget.

    Passing an empty key list through the original manager yields a falsy
    ``keys_to_store`` (verified: CPUPrimaryTierOffloadingManager.prepare_store
    short-circuits on empty input), which _build_store_jobs handles by
    advancing ``next_stored_block_idx`` and skipping -- so the request stops
    being re-offered instead of retrying forever.
    """
    budget = _MAX_OFFLOAD_BLOCKS
    if budget <= 0:
        return _ORIG_PREPARE_STORE(self, keys, req_context)

    rid = req_context.req_id
    used = _REQ_OFFLOADED.get(rid, 0)
    if used >= budget:
        keys = []
    else:
        keys = list(keys)
        room = budget - used
        if len(keys) > room:
            logger.info(
                "[dsv4-patch] offload budget: req=%s capping store at %d "
                "blocks (budget %d, already %d, offered %d)",
                rid, room, budget, used, len(keys),
            )
            keys = keys[:room]
    # Count conservatively (may over-count keys the primary tier already holds,
    # which only makes the cap stricter, never looser).
    _REQ_OFFLOADED[rid] = used + len(keys)
    _offload_budget_prune(self)
    return _ORIG_PREPARE_STORE(self, keys, req_context)


def apply_offload_budget() -> None:
    """Install the per-request offload cap (no-op at budget 0)."""
    global _ORIG_PREPARE_STORE

    if "offload_budget" in _APPLIED:
        return

    from vllm.v1.kv_offload.tiering.manager import TieringOffloadingManager

    _ORIG_PREPARE_STORE = TieringOffloadingManager.prepare_store
    TieringOffloadingManager.prepare_store = _prepare_store_patched
    _APPLIED.add("offload_budget")
    logger.info(
        "[dsv4-patch] offload budget ACTIVE (max_blocks_per_request=%d)",
        _MAX_OFFLOAD_BLOCKS,
    )


# ===========================================================================
# entry point
# ===========================================================================


def apply_all() -> None:
    global _PIN_DESCRIPTORS, _WAITSTREAM_ON_LOAD, _BATCH_LOG
    global _MAX_COPIES_PER_BATCH, _MAX_OFFLOAD_BLOCKS

    _PIN_DESCRIPTORS = _flag("DSV4_PIN_DESCRIPTORS", "1")
    _WAITSTREAM_ON_LOAD = _flag("DSV4_PATCH_BUG2_WAITSTREAM", "0")
    _BATCH_LOG = _flag("DSV4_BATCH_LOG", "1")
    _MAX_COPIES_PER_BATCH = _intenv("DSV4_MAX_COPIES_PER_BATCH", 8192)
    _MAX_OFFLOAD_BLOCKS = _intenv("DSV4_MAX_OFFLOAD_BLOCKS_PER_REQUEST", 0)

    logger.info(
        "[dsv4-patch] applying in pid=%d (%s)",
        os.getpid(),
        threading.current_thread().name,
    )

    # P0 first: abort before anything else if this deployment would corrupt.
    if _flag("DSV4_PATCH_MULTINODE_GUARD", "1"):
        apply_multinode_guard()
    if _flag("DSV4_PATCH_BUG1_IO", "1"):
        apply_bug1_io()
    if _flag("DSV4_PATCH_SWLOOKUP", "1"):
        apply_sliding_window_lookup()
    if _flag("DSV4_PATCH_NUMBLOCKS_GUARD", "1"):
        apply_num_blocks_guard()
    if _flag("DSV4_PATCH_TENSOR_ALIAS", "1"):
        apply_tensor_alias_fix()
    # Install the eager-offload cap. The wrapper is a no-op passthrough at
    # budget 0, so installing it unconditionally lets the knob flip at runtime
    # without reimporting the module.
    if _flag("DSV4_PATCH_OFFLOAD_BUDGET", "1"):
        apply_offload_budget()
    # Applied last so the handler class is fully patched before any instance
    # exists.
    if _flag("DSV4_PATCH_BUG2_KEEPALIVE", "1"):
        apply_bug2_keepalive()
    # Opt-in: off by default until proven under the real MLA/DSpark kernels.
    if _flag("DSV4_HOST_KV", "0"):
        apply_host_kv_alloc()
        try:
            apply_host_kv_alloc_v2()
        except ImportError:
            # V1-only vLLM image: no vllm.v1.worker.gpu.model_runner module.
            pass

    logger.info("[dsv4-patch] active patches: %s", sorted(_APPLIED))


# ---------------------------------------------------------------------------
# Design B: allocate the KV cache from cudaHostAlloc memory.
#
# A cudaMalloc pointer lives in a PROT_NONE anonymous VA reservation (---p,
# Rss 0, VmFlags "mr mw me" -- *may*-read/write, no rd/wr), so it has no CPU
# page-table mapping and no struct page. get_user_pages() rejects it in
# check_vma_flags(), and every file read into it returns EFAULT before it ever
# reaches the block layer. Measured on both nodes, buffered and O_DIRECT alike.
# THAT is the real reason the KV offload path needs a CPU staging tier -- not
# bandwidth, and not anything about discrete VRAM (there is none here).
#
# cudaHostAlloc memory has neither problem. O_DIRECT lands in it at 7.25 GB/s,
# and the GPU reads it at DEVICE SPEED -- measured with a paged gather at the
# real nvfp4_ds_mla page sizes, 240 MiB working set, L2-bypassing loads:
#
#     page  8,640 B:  device  72.8  cudaHostAlloc  73.3 (100.8%)  ATS  70.5
#     page 37,440 B:  device 205.3  cudaHostAlloc 208.3 (101.4%)  ATS 160.0
#
# The well-known ~25-30% penalty belongs to the PAGEABLE/ATS path; cudaHostAlloc
# bypasses it by getting real GPU page-table mappings.
#
# With the KV cache allocated here, the staging buffer and the KV cache become
# the same allocation: the 8.58 GB/node region disappears, and so does the
# whole-prefix residency cliff -- there are no staging cells left to run out of.
#
# Hooked at initialize_kv_cache_tensors because it dominates BOTH allocation
# paths (allocate_uniform_kv_caches when use_uniform_kv_cache() is true, and
# _allocate_kv_cache_tensors otherwise). No conflict with vLLM's own pool:
# _maybe_get_memory_pool_context returns nullcontext() unless
# enable_cumem_allocator, which is off here (it breaks the DSpark draft config).
# ---------------------------------------------------------------------------

_HOST_KV_POOL = None  # module-level: the pool must outlive every tensor from it
_HOST_KV_ALLOC = None

_HOST_KV_SO = os.environ.get("DSV4_HOST_KV_SO", "/usr/local/lib/libdsv4_host_kv.so")


def _build_host_kv_pool():
    """Create (once) the cudaHostAlloc-backed MemPool."""
    global _HOST_KV_POOL, _HOST_KV_ALLOC
    if _HOST_KV_POOL is not None:
        return _HOST_KV_POOL
    from torch.cuda.memory import CUDAPluggableAllocator, MemPool

    if not os.path.exists(_HOST_KV_SO):
        raise RuntimeError(
            f"dsv4-patch: DSV4_HOST_KV=1 but {_HOST_KV_SO} is missing. Build it with:\n"
            "  nvcc -O3 -arch=sm_121a -shared -Xcompiler -fPIC "
            "-o /usr/local/lib/libdsv4_host_kv.so dsv4_host_kv_alloc.cu"
        )
    _HOST_KV_ALLOC = CUDAPluggableAllocator(
        _HOST_KV_SO, "dsv4_host_malloc", "dsv4_host_free"
    )
    _HOST_KV_POOL = MemPool(_HOST_KV_ALLOC.allocator())
    logger.info("[dsv4-patch] host-KV pool ready from %s", _HOST_KV_SO)
    return _HOST_KV_POOL


def _verify_host_kv_tensors(tensors: dict) -> None:
    """Prove the KV tensors are host-addressable and dump their layout.

    A silent fallback to cudaMalloc would look identical until the first disk
    read returned EFAULT in production, so this fails loud. mincore(2) returns
    -1/ENOMEM for an unmapped (PROT_NONE cudaMalloc) address instead of
    faulting; a raw memmove would SIGSEGV the process on that same address.
    """
    import ctypes as _ct

    sample = next(iter(tensors.values()))
    ptr = sample.data_ptr()
    libc = _ct.CDLL("libc.so.6", use_errno=True)
    mincore = libc.mincore
    mincore.argtypes = [_ct.c_void_p, _ct.c_size_t, _ct.POINTER(_ct.c_ubyte)]
    mincore.restype = _ct.c_int
    vec = (_ct.c_ubyte * 1)()
    if mincore(_ct.c_void_p(ptr), _ct.c_size_t(1), vec) != 0:
        err = _ct.get_errno()
        raise RuntimeError(
            f"mincore probe failed (errno={err}) -- KV at 0x{ptr:x} "
            "is not host-addressable"
        )
    logger.info("[dsv4-patch] host-KV verified: CPU can address KV at 0x%x", ptr)
    # Dump the distinct backing tensors (deduped by data_ptr) so the
    # direct-I/O path can map a disk cell -> per-group host pointers.
    _seen: set[int] = set()
    for _name, _t in tensors.items():
        _p = _t.data_ptr()
        if _p in _seen:
            continue
        _seen.add(_p)
        logger.info(
            "[dsv4-patch] host-KV tensor %s shape=%s stride=%s ptr=0x%x "
            "nbytes=%d",
            _name, tuple(_t.shape), tuple(_t.stride()), _p,
            _t.numel() * _t.element_size(),
        )


def apply_host_kv_alloc() -> None:
    """Route KV cache allocation through cudaHostAlloc on the V1 model runner.

    The V2 model runner (forced on for DSpark) allocates KV through
    ``vllm.v1.worker.gpu.model_runner.init_kv_cache`` instead of this method;
    see apply_host_kv_alloc_v2.
    """
    if "hostkv" in _APPLIED:
        return
    from torch.cuda.memory import use_mem_pool
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    orig = GPUModelRunner.initialize_kv_cache_tensors

    def patched(self, kv_cache_config, kernel_block_sizes):
        pool = _build_host_kv_pool()
        total = sum(t.size for t in kv_cache_config.kv_cache_tensors)
        logger.info(
            "[dsv4-patch] allocating %.2f GiB of KV cache from cudaHostAlloc "
            "(disk can DMA straight into it; cudaMalloc cannot -- EFAULT)",
            total / (1 << 30),
        )
        with use_mem_pool(pool):
            out = orig(self, kv_cache_config, kernel_block_sizes)
        try:
            _verify_host_kv_tensors(out)
        except Exception as e:
            logger.error(
                "[dsv4-patch] host-KV VERIFICATION FAILED (%r) -- KV is NOT "
                "host-addressable, so disk->KV DMA will EFAULT", e
            )
            raise
        return out

    GPUModelRunner.initialize_kv_cache_tensors = patched
    _APPLIED.add("hostkv")
    logger.info("[dsv4-patch] hostkv (V1): KV cache will be cudaHostAlloc-backed")


def apply_host_kv_alloc_v2() -> None:
    """Route KV cache allocation through cudaHostAlloc on the V2 model runner.

    V2 allocates KV via the module-level ``init_kv_cache`` imported into
    ``vllm.v1.worker.gpu.model_runner`` (``from ...attn_utils import
    init_kv_cache``), so that bound name -- not the one in attn_utils -- is
    what ``GPUModelRunner.initialize_kv_cache`` calls at runtime. Wrapping the
    V1 method is a silent no-op under V2: the sitecustomize hook still arms,
    the wrapped method is simply never called.
    """
    if "hostkv_v2" in _APPLIED:
        return
    from torch.cuda.memory import use_mem_pool
    import vllm.v1.worker.gpu.model_runner as _mr

    orig = _mr.init_kv_cache
    if getattr(orig, "_dsv4_host_kv", False):
        _APPLIED.add("hostkv_v2")
        return

    def patched(*args, **kwargs):
        pool = _build_host_kv_pool()
        logger.info(
            "[dsv4-patch] allocating KV cache from cudaHostAlloc "
            "(V2 init_kv_cache)"
        )
        with use_mem_pool(pool):
            out = orig(*args, **kwargs)
        try:
            _verify_host_kv_tensors(out)
        except Exception as e:
            logger.error(
                "[dsv4-patch] host-KV VERIFICATION FAILED (%r) -- KV is NOT "
                "host-addressable, so disk->KV DMA will EFAULT", e
            )
            raise
        return out

    patched._dsv4_host_kv = True
    _mr.init_kv_cache = patched
    _APPLIED.add("hostkv_v2")
    logger.info("[dsv4-patch] hostkv (V2): init_kv_cache will be cudaHostAlloc-backed")


# ---------------------------------------------------------------------------
# Scatter-gather batch copy (replaces cuMemcpyBatchAsync for large restores).
# See dsv4_batch_copy.cu for the full rationale and the measured driver wall.
# ---------------------------------------------------------------------------

_SG_LIB = None
_SG_FN = None


def _sg_copy_threshold() -> int:
    # Default matches the compose/docs default (20000): below the ~23k driver
    # wall, the driver's cuMemcpyBatchAsync is faster and is used directly. A
    # bare `vllm serve` (no compose env) still gets the SG fallback above the
    # wall rather than segfaulting the driver.
    return _intenv("DSV4_SG_THRESHOLD", 20000)


def _sg_load():
    """Load libdsv4_batch_copy.so once."""
    global _SG_LIB, _SG_FN
    if _SG_FN is not None:
        return _SG_FN
    path = os.environ.get("DSV4_SG_SO", "/usr/local/lib/libdsv4_batch_copy.so")
    if not os.path.exists(path):
        raise RuntimeError(f"DSV4_SG_THRESHOLD set but {path} is missing")
    _SG_LIB = ctypes.CDLL(path)
    fn = _SG_LIB.dsv4_batch_copy
    fn.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    _SG_FN = fn
    logger.info("[dsv4-patch] SG copy kernel loaded from %s", path)
    return _SG_FN


def _sg_batch_copy(vsrc, vdst, vsz, n: int, handler, captured: list) -> int:
    """Run the descriptor list through our own kernel on the current stream.

    The descriptor arrays live in host memory; the kernel needs them device-side,
    so they are staged through a device buffer (3 * n * 8 bytes -- ~5.9 MB even
    for a 1M-token restore).

    THE BUFFER IS PER HANDLER, NOT PER DEVICE. vLLM builds TWO independent
    SingleDirectionOffloadingHandlers (gpu_worker.py:437-452) -- one per
    direction -- each with its own _transfers deque and stream pool, and
    `stream.wait_event(last_transfer.end_event)` only serialises transfers
    *within* one handler. OffloadingConnectorWorker.start_kv_transfers submits
    store jobs and load jobs in the same call, so a load's descriptor upload can
    land while a store's kernel is still grid-striding over a shared buffer. The
    store would then execute the load's descriptors, its CPU staging slots would
    never be written, and stale bytes would be flushed to disk under a valid
    block key -- persistent, restart-surviving, and invisible to a single-load
    benchmark. That is the same silent-corruption class this tier exists to
    eliminate, so the buffer hangs off the handler exactly like _dsv4_pool does.

    It is also appended to `captured`, which is released only after the job's
    end_event completes: growing the buffer must not free one an in-flight
    kernel is still reading.
    """
    import torch

    fn = _sg_load()
    buf = getattr(handler, "_dsv4_sg_buf", None)
    if buf is None or buf[0].numel() < n:
        buf = (
            torch.empty(n, dtype=torch.int64, device="cuda"),
            torch.empty(n, dtype=torch.int64, device="cuda"),
            torch.empty(n, dtype=torch.int64, device="cuda"),
        )
        handler._dsv4_sg_buf = buf
    captured.append(buf)
    dsrc, ddst, dsz = (b.narrow(0, 0, n) for b in buf)
    # non_blocking is safe: these are ordered on the same stream as the kernel,
    # and the host-side sources are retained in `captured` for the job's lifetime.
    dsrc.copy_(vsrc, non_blocking=True)
    ddst.copy_(vdst, non_blocking=True)
    dsz.copy_(vsz, non_blocking=True)

    stream = torch.cuda.current_stream()
    rc = fn(
        ctypes.c_void_p(dsrc.data_ptr()),
        ctypes.c_void_p(ddst.data_ptr()),
        ctypes.c_void_p(dsz.data_ptr()),
        ctypes.c_int64(n),
        ctypes.c_void_p(stream.cuda_stream),
    )
    if rc == 0 and _BATCH_LOG:
        logger.info("[dsv4-patch] SG copy ok: %d descriptors (driver path bypassed)", n)
    return int(rc)
