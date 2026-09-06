# SPDX-License-Identifier: Apache-2.0
"""Apply early monkeypatches in EVERY python process (engine and workers).

Python imports `sitecustomize` automatically at interpreter startup, which is the
only seam that reliably reaches vLLM's spawned worker processes. (The disk tier gets
there via `spec_module_path`, but that only exists when a --kv-transfer-config is
configured.)

The patches cannot be applied at startup -- vLLM is not imported yet -- so this hooks
`SourceFileLoader.exec_module` and fires each patch the moment its target module
finishes loading, which is the earliest point the patch subject exists and still
well before any KV cache is allocated or any config object is constructed.

Two patches, both a no-op unless the master switch is on:
  * KV_DISK_CACHE_DIRECT_IO=1 -- route the KV cache through cudaHostAlloc (see apply_host_kv_alloc).
  * always -- exempt the OffloadingConnector from vLLM's expandable_segments
    rejection (the disk tier requires expandable segments; see below).
"""
import os
import sys

_HOOKS = []  # (target_module_name, callable)

# The disk tier is default-off. When the master switch is off this module must
# be a complete no-op: no import hooks, no monkeypatches, and no host-KV
# allocation changes leaked in by any other experimental knob. Everything below
# is gated on this one flag so an off launch leaves Python/config untouched and
# needs no staged KV_DISK_CACHE_SRC.
_ENABLED = os.environ.get("KV_DISK_CACHE_ENABLE", "0").strip() == "1"


def _apply_host_kv():
    import dsv4_vllm_patches

    dsv4_vllm_patches.apply_host_kv_alloc()


def _apply_host_kv_v2():
    import dsv4_vllm_patches

    dsv4_vllm_patches.apply_host_kv_alloc_v2()


def _apply_expandable_segments_exempt():
    """Let the disk tier run with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.

    vLLM's VllmConfig._verify_kv_transfer_compat rejects expandable_segments:True
    whenever ANY KV connector is configured, because RDMA-pinning connectors
    (Nixl via ibv_reg_mr, Mooncake) would hold stale physical-page registrations
    once CUDA VMM remaps the KV VA range. The OffloadingConnector used by this
    disk tier does NOT pin KV memory -- it copies GPU<->CPU staging with
    cuMemcpyBatchAsync and persists to NVMe -- so VMM remapping is harmless and
    the rejection is over-broad. Keeping expandable_segments:True collapses the
    per-allocation memdesc pressure during prefill that otherwise exhausts the
    NVIDIA driver's fixed-size memdesc pool (NVRM _memdescAllocInternal
    NV_ERR_NO_MEMORY).
    """
    from vllm.config.vllm import VllmConfig

    _orig = VllmConfig._verify_kv_transfer_compat
    if getattr(_orig, "_dsv4_exempt", False):
        return

    def _patched(self):
        # Run the stock check first so every unrelated check it performs (no
        # connector, expandable_segments not set, cumem allocator, and any
        # future additions) is preserved. We only swallow the ONE rejection we
        # know is over-broad for this tier: the expandable_segments guard that
        # protects RDMA-pinning connectors. The OffloadingConnector copies
        # GPU<->CPU staging with cuMemcpyBatchAsync and persists to NVMe, so it
        # never pins KV memory and CUDA VMM remapping is harmless.
        try:
            _orig(self)
        except ValueError as e:
            kt = self.kv_transfer_config
            if (
                kt is not None
                and kt.kv_connector == "OffloadingConnector"
                and "expandable_segments" in str(e)
            ):
                return
            raise

    _patched._dsv4_exempt = True
    VllmConfig._verify_kv_transfer_compat = _patched


if _ENABLED and os.environ.get("KV_DISK_CACHE_DIRECT_IO") == "1":
    _HOOKS.append(("vllm.v1.worker.gpu_model_runner", _apply_host_kv))
    _HOOKS.append(("vllm.v1.worker.gpu.model_runner", _apply_host_kv_v2))

if _ENABLED:
    _HOOKS.append(("vllm.config.vllm", _apply_expandable_segments_exempt))

if _HOOKS:
    import importlib.machinery

    _orig_exec_module = importlib.machinery.SourceFileLoader.exec_module

    def _exec_module(self, module):
        _orig_exec_module(self, module)
        name = getattr(module, "__name__", None)
        for target, cb in _HOOKS:
            if name == target:
                try:
                    cb()
                except Exception as e:  # fail loud: a silent miss is worse
                    print(
                        f"[dsv4-sitecustomize] hook for {target} FAILED in "
                        f"pid {os.getpid()}: {e!r}",
                        file=sys.stderr,
                        flush=True,
                    )
                    raise

    importlib.machinery.SourceFileLoader.exec_module = _exec_module
    print(
        f"[dsv4-sitecustomize] hooks armed in pid {os.getpid()}: "
        f"{[t for t, _ in _HOOKS]}",
        file=sys.stderr,
        flush=True,
    )
