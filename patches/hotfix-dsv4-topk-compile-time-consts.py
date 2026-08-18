#!/usr/bin/env python3
"""hotfix-dsv4-topk-compile-time-consts.py — Compile-time constants for the
global top-k index kernel (upstream vLLM PR #51967 backport).

Backport of upstream vLLM PR #51967 ("[Perf][DSV4] Optimize global top-k
index kernel with compile-time constants"), merged upstream 2026-08-16,
applied to the Anemll dspark-vllm-gx10 0.1.1 image
(vLLM 0.25.2.dev0+g752a3a504.d20260714).

WHAT: promotes five parameters of the `_compute_global_topk_indices_and_lens_kernel`
Triton kernel — `global_topk_indices_stride`, `topk_indices_stride`, `topk`,
`block_table_stride`, `block_size` — from runtime arguments to `tl.constexpr`.
Byte-for-byte the same five promotions upstream merged. This fork's extra
`num_blocks` bound (stale block-table guard) sits after the anchor and stays a
runtime argument.

WHERE IT RUNS: `compute_global_topk_indices_and_lens` is called from
`models/deepseek_v4/nvidia/sm120.py` on
  - the C4A **decode** path (`compress_ratio == 4`, cudagraph-captured, hence
    the persistent padded `topk_indices` buffer and the `is_valid_token` mask
    the kernel exists to honour), and
  - the C4A/C128A **prefill** paths.
C128A decode does not call it (its global indices are pre-computed during
metadata build).

WHAT THE PROMOTION ACTUALLY BUYS: literal comparands for `offset < topk` and
`block_indices < block_table_stride`, and shift/mask instead of `//` and `%`
for `local_idx // block_size` / `local_idx % block_size`. It does **not**
unroll the loop: `for i in range(0, topk, TRITON_BLOCK_SIZE)` already receives
`TRITON_BLOCK_SIZE=1024` as a `tl.constexpr`, and the reachable widths are
`topk` = 512 on C4A (512 index_topk x 4 compress ratio) and the adaptive
C128A prefill width under the live #50004 hotfix, so the C4A loop is a single
trip. Real constants on this fork are `block_size` = 256 // 4 = **64** (C4A)
and 256 // 128 = **2** (C128A prefill) — not the 256 an earlier revision of
this docstring claimed.

COST: the promoted tuple becomes part of the Triton specialization key, so the
kernel is compiled once per distinct
`(global_topk_indices_stride, topk_indices_stride, topk, block_table_stride,
block_size)`. C4A contributes one tuple; C128A prefill contributes one per
adaptive width band. Compilations are lazy and first-touch, so a fresh
container start can pay them on the serving path.

NUMERICS: identical. `tl.constexpr` changes *when* a value is known, not the
value. No accumulation-order, dtype, masking, or allocation change; the kernel
is integer/index-only.

MEASURED: **not measured on this fork.** Upstream's author measured this exact
change with `vllm bench serve` (128 prompts, `--ignore-eos`, `--temperature 0`,
`--seed 701`, TP=8, 1024-in/64-out): +0.50% mean output throughput,
-0.98% mean TPOT. Earlier revisions of this file carried a "+3.3% decode C1"
figure that is withdrawn: it had no n or dispersion, its statistics belonged to
a three-patch stack, and it is not reproducible from anything committed here.
Treat the fork-local effect as unknown until a baseline -> candidate -> baseline
run under `scripts/bench-patches.sh` records it in `results/`.

Default OFF. The compose entrypoint applies this only when
`DSPARK_ENABLE_TOPK_CONSTEXPR_HOTFIX` is exactly `1`, and fails the container
boot (`|| exit 1`) if application fails — see `.env.dspark.example`.

Usage (idempotent — re-running skips an already-applied file):
  python3 hotfix-dsv4-topk-compile-time-consts.py            # apply
  python3 hotfix-dsv4-topk-compile-time-consts.py --status   # probe, no write
`VLLM_ROOT` overrides the vLLM install root (default
/usr/local/lib/python3.12/dist-packages/vllm).
"""

import os
import sys
from pathlib import Path

VLLM_ROOT = Path(os.environ.get("VLLM_ROOT", "/usr/local/lib/python3.12/dist-packages/vllm"))
TARGET = VLLM_ROOT / "models/deepseek_v4/common/ops/cache_utils.py"
MARKER = "global_topk_indices_stride: tl.constexpr"


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        status_src = TARGET.read_text() if TARGET.is_file() else ""
        print("dsv4 topk constexpr (#51967)       :",
              "APPLIED" if MARKER in status_src else "NOT APPLIED")
        raise SystemExit(0)

    src = TARGET.read_text()
    marker = MARKER
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
