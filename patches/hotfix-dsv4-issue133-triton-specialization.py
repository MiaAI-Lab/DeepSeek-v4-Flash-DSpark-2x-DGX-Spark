#!/usr/bin/env python3
"""Disable traffic-dependent Triton alignment specialization on global top-k.

Issue #133: mixed prefill slices token_to_req_indices and is_valid_token at the
same token offset, so Triton's 16-byte alignment specialization produced three
pointer-key states. Combined with effective block sizes 64 and 2 that became
six persistent-cache entries and mid-serve JIT. Both pointers are scalar loads,
so alignment specialization does not vectorize them.

This startup patch matches the overlay in recipe/overlay/.../cache_utils.py
against the Anemll 0.1.1 kernel signature (no num_blocks bound).
"""
from __future__ import annotations

import sys
from pathlib import Path

P = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/common/ops/cache_utils.py"
)
MARK = "# [issue133-hotfix] do_not_specialize_on_alignment for sliced metadata ptrs"

OLD = """@triton.jit
def _compute_global_topk_indices_and_lens_kernel(
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
    is_valid_token_ptr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
"""

NEW = """@triton.jit(
    do_not_specialize_on_alignment=[
        "token_to_req_indices_ptr",
        "is_valid_token_ptr",
    ]
)
def _compute_global_topk_indices_and_lens_kernel(
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
    is_valid_token_ptr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
""" + MARK + "\n"


def patch_text(source: str) -> tuple[str, str]:
    if MARK in source and "do_not_specialize_on_alignment" in source:
        return source, "skipped"
    if source.count(OLD) != 1:
        return source, f"drift:old={source.count(OLD)}"
    updated = source.replace(OLD, NEW, 1)
    compile(updated, "cache_utils.py", "exec")
    return updated, "applied"


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        status_src = P.read_text() if P.is_file() else ""
        print(
            "issue133 triton specialization     :",
            "APPLIED" if MARK in status_src else "NOT APPLIED",
        )
        return 0
    if not P.is_file():
        print(f"[issue133-hotfix] missing {P}", file=sys.stderr)
        return 1
    src = P.read_text()
    updated, status = patch_text(src)
    if status == "skipped":
        print(f"[issue133-hotfix] already applied to {P}")
        return 0
    if status != "applied":
        print(f"[issue133-hotfix] refusing to patch ({status})", file=sys.stderr)
        return 1
    P.write_text(updated)
    print(f"[issue133-hotfix] applied to {P}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
