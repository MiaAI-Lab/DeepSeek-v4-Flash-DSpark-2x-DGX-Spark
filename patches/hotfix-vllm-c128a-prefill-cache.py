#!/usr/bin/env python3
"""Reuse C128A prefill index conversion within one attention metadata lifetime.

The first SM120 C128A consumer uses the unchanged conversion and its original
inputs. Later layers sharing that metadata reuse the result. C4A, decode and
other attention implementations are unchanged. No persistent buffers are added.

Opt-in hotfix for the pinned Anemll 0.1.1 vLLM image. Check validates both
source regions before either file is written; repeated application is a no-op.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")
EXPECTED_VLLM_VERSION = "0.25.2.dev0+g752a3a504.d20260714"
MARK = "# [dspark-c128a-prefill-cache]"
SM = "models/deepseek_v4/sparse_mla.py"
FS = "models/deepseek_v4/nvidia/flashinfer_sparse.py"

FIELDS_OLD = """    c128a_prefill_topk_indices: torch.Tensor | None = None
"""
FIELDS_NEW = FIELDS_OLD + """    # [dspark-c128a-prefill-cache] Shared only within this forward's metadata.
    c128a_prefill_global_topk: tuple[torch.Tensor, torch.Tensor] | None = None
"""
CALL_OLD = """            extra_sparse_indices, extra_sparse_lengths = (
                compute_global_topk_indices_and_lens(
                    local_topk_indices,
                    swa_metadata.token_to_req_indices[prefill_token_slice],
                    attn_metadata.block_table,
                    block_size,
                    swa_metadata.is_valid_token[prefill_token_slice],
                )
            )"""
CALL_NEW = """            # [dspark-c128a-prefill-cache] C4A indices remain layer-dependent.
            prefill_topk = (
                attn_metadata.c128a_prefill_global_topk
                if self.compress_ratio == 128
                else None
            )
            if prefill_topk is None:
                prefill_topk = compute_global_topk_indices_and_lens(
                    local_topk_indices,
                    swa_metadata.token_to_req_indices[prefill_token_slice],
                    attn_metadata.block_table,
                    block_size,
                    swa_metadata.is_valid_token[prefill_token_slice],
                )
                if self.compress_ratio == 128:
                    attn_metadata.c128a_prefill_global_topk = prefill_topk
            extra_sparse_indices, extra_sparse_lengths = prefill_topk"""
REGIONS = {SM: (FIELDS_OLD, FIELDS_NEW), FS: (CALL_OLD, CALL_NEW)}


def transform(text: str, relative: str) -> str:
    old, new = REGIONS[relative]
    if "[dspark-c128a-hoist-v1]" in text:
        raise ValueError("recreate the container; builder experiment is not supported")
    if text.count(MARK) == 1 and text.count(new) == 1:
        # FIELDS_NEW contains FIELDS_OLD; only the complete new region is valid.
        restored = text.replace(new, old, 1)
        if restored.count(old) != 1:
            raise ValueError(f"duplicate target region: {relative}")
        compile(text, relative, "exec")
        return text
    if "c128a_prefill_global_topk" in text or MARK in text or text.count(old) != 1:
        raise ValueError(f"unsupported target region: {relative}")
    updated = text.replace(old, new, 1)
    compile(updated, relative, "exec")
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        version = importlib.metadata.version("vllm")
        if version != EXPECTED_VLLM_VERSION:
            raise ValueError(f"unsupported vllm version: {version}")
        staged = []
        for relative in REGIONS:
            path = args.root / relative
            original = path.read_text(encoding="utf-8")
            updated = transform(original, relative)
            staged.append((path, original, updated))
        for path, original, updated in staged:
            state = "patched" if original == updated else "patchable"
            if not (args.check or args.status) and original != updated:
                fd, name = tempfile.mkstemp(prefix=".c128a-prefill-", dir=path.parent)
                temporary = Path(name)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as handle:
                        handle.write(updated)
                    temporary.chmod(path.stat().st_mode & 0o777)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
                state = "applied"
            print(f"dspark-c128a-prefill-cache: {state} ({path})")
        return 0
    except (OSError, SyntaxError, ValueError, importlib.metadata.PackageNotFoundError) as error:
        print(f"dspark-c128a-prefill-cache: FAIL-CLOSED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
