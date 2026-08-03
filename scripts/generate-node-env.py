#!/usr/bin/env python3
"""Generate separate, mode-0600 head and worker DSpark environments."""

from __future__ import annotations

import argparse
from pathlib import Path


COMMON_KEYS = (
    "MASTER_ADDR", "MASTER_PORT", "DSPARK_MODEL", "DSPARK_MODEL_REVISION",
    "DSPARK_ENCODING_FILE", "SERVED_MODEL_NAME", "DSPARK_VLLM_IMAGE",
    "MAX_MODEL_LEN", "MAX_NUM_SEQS", "MAX_NUM_BATCHED_TOKENS",
    "GPU_MEMORY_UTILIZATION", "MTP_NUM_TOKENS", "DEFAULT_THINKING",
    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_XET",
    "NCCL_CROSS_NIC", "NCCL_IB_GID_AUTO", "NCCL_IB_ADDR_FAMILY",
    "NCCL_IB_ROCE_VERSION_NUM", "NCCL_NET", "NCCL_IB_DISABLE",
    "NCCL_CUMEM_ENABLE", "NCCL_IGNORE_CPU_AFFINITY", "NCCL_DEBUG",
    "NCCL_NVLS_ENABLE", "PYTORCH_CUDA_ALLOC_CONF", "CUTE_DSL_ARCH",
    "TORCH_CUDA_ARCH_LIST", "FLASHINFER_CUDA_ARCH_LIST",
    "FLASHINFER_DISABLE_VERSION_CHECK", "TILELANG_CLEANUP_TEMP_FILES",
    "DG_JIT_USE_NVRTC", "DG_JIT_NVCC_COMPILER",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN", "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB",
    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "VLLM_USE_FLASHINFER_SAMPLER",
    "VLLM_USE_BREAKABLE_CUDAGRAPH", "VLLM_USE_B12X_MOE",
    "VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM", "VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M",
    "VLLM_B12X_W4A16_FORCE_TILE_CONFIG", "ENABLE_VLLM_GB10_PATCH",
    "GB10_HYBRID_NVFP4_M_THRESHOLD",
)
HEAD_ONLY_KEYS = (
    "WORKER_HOST", "WORKER_DIR", "WORKER_SCRIPT_DIR", "WORKER_HF_CACHE",
    "WORKER_NCCL_IB_HCA", "WORKER_NCCL_SOCKET_IFNAME",
    "WORKER_TP_SOCKET_IFNAME", "WORKER_GLOO_SOCKET_IFNAME",
    "WORKER_NCCL_IB_GID_INDEX", "WORKER_NCCL_IB_GID_MATCH_IP",
    "WORKER_VLLM_HOST_IP", "VLLM_ORIGIN_KEY_FILE", "VLLM_PROXY_HOST",
    "VLLM_PROXY_PORT",
)
LOCAL_KEYS = (
    "NODE_RANK", "HEADLESS", "HF_CACHE", "NCCL_IB_HCA",
    "NCCL_SOCKET_IFNAME", "TP_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME",
    "NCCL_IB_GID_INDEX", "NCCL_IB_GID_MATCH_IP", "VLLM_HOST",
    "VLLM_PORT", "VLLM_HOST_IP",
)


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.replace("_", "").isalnum():
            values[key] = value
    return values


def write_env(path: Path, values: dict[str, str], keys: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}={values[key]}" for key in keys if key in values]
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o600)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--head-output", type=Path, required=True)
    parser.add_argument("--worker-output", type=Path, required=True)
    args = parser.parse_args()
    values = parse_env(args.source)

    worker = {key: values[key] for key in COMMON_KEYS if key in values}
    worker.update({
        "NODE_RANK": "1",
        "HEADLESS": "1",
        "HF_CACHE": values.get("WORKER_HF_CACHE", values.get("HF_CACHE", "")),
        "NCCL_IB_HCA": values.get("WORKER_NCCL_IB_HCA", values.get("NCCL_IB_HCA", "")),
        "NCCL_SOCKET_IFNAME": values.get("WORKER_NCCL_SOCKET_IFNAME", values.get("NCCL_SOCKET_IFNAME", "")),
        "TP_SOCKET_IFNAME": values.get("WORKER_TP_SOCKET_IFNAME", values.get("TP_SOCKET_IFNAME", "")),
        "GLOO_SOCKET_IFNAME": values.get("WORKER_GLOO_SOCKET_IFNAME", values.get("GLOO_SOCKET_IFNAME", "")),
        "NCCL_IB_GID_INDEX": values.get("WORKER_NCCL_IB_GID_INDEX", ""),
        "NCCL_IB_GID_MATCH_IP": values.get("WORKER_NCCL_IB_GID_MATCH_IP", ""),
        "VLLM_HOST": "127.0.0.1",
        "VLLM_PORT": values.get("VLLM_PORT", "8889"),
        "VLLM_HOST_IP": values.get("WORKER_VLLM_HOST_IP", ""),
    })
    write_env(args.head_output, values, COMMON_KEYS + HEAD_ONLY_KEYS + LOCAL_KEYS)
    write_env(args.worker_output, worker, COMMON_KEYS + LOCAL_KEYS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
