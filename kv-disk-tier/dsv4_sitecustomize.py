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

Two patches, each env-gated and a no-op otherwise:
  * DSV4_HOST_KV=1 -- route the KV cache through cudaHostAlloc (see apply_host_kv_alloc).
  * DSV4_ALLOW_EXPANDABLE_SEGMENTS=1 -- exempt the OffloadingConnector from vLLM's
    expandable_segments rejection (see below).
"""
import os
import sys

_HOOKS = []  # (target_module_name, callable)


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
        kt = self.kv_transfer_config
        if kt is not None and kt.kv_connector == "OffloadingConnector":
            return
        return _orig(self)

    _patched._dsv4_exempt = True
    VllmConfig._verify_kv_transfer_compat = _patched


if os.environ.get("DSV4_HOST_KV") == "1":
    _HOOKS.append(("vllm.v1.worker.gpu_model_runner", _apply_host_kv))
    _HOOKS.append(("vllm.v1.worker.gpu.model_runner", _apply_host_kv_v2))

if os.environ.get("DSV4_ALLOW_EXPANDABLE_SEGMENTS") == "1":
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
