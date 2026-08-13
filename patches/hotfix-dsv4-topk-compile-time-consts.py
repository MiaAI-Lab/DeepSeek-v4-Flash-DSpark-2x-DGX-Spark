#!/usr/bin/env python3
"""hotfix-dsv4-topk-compile-time-consts.py — Compile-time constants for the
global top-k index kernel (upstream vLLM PR #51967 backport).

Backport of upstream vLLM PR #51967 ("[Perf][DSV4] Optimize global top-k
index kernel with compile-time constants") applied to the Anemll
dspark-vllm-gx10 0.1.1 image (vLLM 0.25.2.dev0+g752a3a504.d20260714).

WHY: the `_compute_global_topk_indices_and_lens_kernel` Triton kernel took
`global_topk_indices_stride`, `topk_indices_stride`, `topk`,
`block_table_stride`, and `block_size` as *runtime* arguments. Triton JIT
compiles a generic loop with dynamic bounds checks for runtime values. With
those values promoted to `tl.constexpr`, the compiler knows them at compile
time (topk=128, block_size=256 on DSV4) and can fully unroll the loop and
drop the dynamic bounds checks, shrinking per-invocation instruction count.
This kernel runs on every prefill step's global top-k pass, so the saving
compounds on long prompts.

MEASURED (this repo's 2x DGX Spark, TP=2, DSV4-Flash-0731): decode C1 +2.8%
vs baseline; prefill within measurement noise. Kernel is numerically
identical (constexpr promotion only — no math change).

Usage (idempotent — re-running skips already-applied hunks):
  docker cp hotfix-dsv4-topk-compile-time-consts.py <container>:/tmp/ && \
  docker exec <container> python3 /tmp/hotfix-dsv4-topk-compile-time-consts.py
  # then restart the vLLM process inside the container.
"""

from pathlib import Path

VLLM_ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")
TARGET = VLLM_ROOT / "models/deepseek_v4/common/ops/cache_utils.py"


def main() -> None:
    src = TARGET.read_text()
    marker = "global_topk_indices_stride: tl.constexpr"
    if marker in src:
        print("[skip] hotfix already applied")
        return

    old = """def _compute_global_topk_indices_and_lens_kernel(
    global_topk_indices_ptr,
    global_topk_indices_stride,
    topk_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    topk,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    is_valid_token_ptr,"""
    new = """def _compute_global_topk_indices_and_lens_kernel(
    global_topk_indices_ptr,
    global_topk_indices_stride: tl.constexpr,
    topk_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride: tl.constexpr,
    topk: tl.constexpr,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride: tl.constexpr,
    block_size: tl.constexpr,
    is_valid_token_ptr,"""
    if old not in src:
        raise SystemExit(f"[ERR] anchor not found in {TARGET}")
    src = src.replace(old, new, 1)
    TARGET.write_text(src)
    print(f"[OK]   promoted 5 kernel args to tl.constexpr in {TARGET}")


if __name__ == "__main__":
    main()
