#!/usr/bin/env python3
"""A/B the issue #133 Triton specialization family: baseline vs patched overlay.

Compares compile-key cardinality, SM121 compilation, and (when CUDA is present)
bit-exact kernel outputs against a Python reference and against each other.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, make_backend
from triton.compiler import compile as triton_compile
from triton.runtime.jit import MockTensor, create_function_from_signature

OFFSETS = (0, 4, 1, 16)
BLOCK_SIZES = (64, 2)
NUM_BLOCKS = 8
REQ_INDICES = (0, 1, 0)
VALID_TOKENS = (True, False, True)


def _stub_and_load(path: Path, module_name: str):
    triton_utils = types.ModuleType("vllm.triton_utils")
    triton_utils.triton = triton
    triton_utils.tl = tl
    import_utils = types.ModuleType("vllm.utils.import_utils")
    import_utils.has_cutedsl = lambda: False
    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    vllm_utils = types.ModuleType("vllm.utils")
    vllm_utils.__path__ = []
    vllm.triton_utils = triton_utils
    vllm.utils = vllm_utils
    vllm_utils.import_utils = import_utils
    stub_modules = {
        "vllm": vllm,
        "vllm.triton_utils": triton_utils,
        "vllm.utils": vllm_utils,
        "vllm.utils.import_utils": import_utils,
    }
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in stub_modules}
    try:
        sys.modules.update(stub_modules)
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, prev in previous.items():
            if prev is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
    return module


def _specialize(kernel, backend, binder, token_offset: int, block_size: int):
    class OffsetTensor(MockTensor):
        def __init__(self, dtype, address: int) -> None:
            super().__init__(dtype)
            self.address = address

        def data_ptr(self) -> int:
            return self.address

    aligned_i32 = OffsetTensor(torch.int32, 0)
    token_to_req_indices = OffsetTensor(torch.int32, 4 * token_offset)
    is_valid_token = OffsetTensor(torch.bool, token_offset)
    params, specialization, _ = binder(
        aligned_i32,
        1024,
        aligned_i32,
        aligned_i32,
        1024,
        1024,
        token_to_req_indices,
        aligned_i32,
        1024,
        block_size,
        is_valid_token,
        4096,
        1024,
    )
    named = {
        parameter.name: item
        for parameter, item in zip(kernel.params, specialization)
    }
    return params, specialization, named


def _key(named: dict) -> str:
    return json.dumps(named, default=str, sort_keys=True)


def _analyze(label: str, path: Path, compile_sm121: bool) -> dict:
    module = _stub_and_load(path, f"issue133_{label}")
    kernel = module._compute_global_topk_indices_and_lens_kernel
    target = GPUTarget("cuda", 121, 32)
    backend = make_backend(target)
    binder = create_function_from_signature(kernel.signature, kernel.params, backend)
    rows = []
    unique = {}
    for block_size in BLOCK_SIZES:
        for offset in OFFSETS:
            params, specialization, named = _specialize(
                kernel, backend, binder, offset, block_size
            )
            digest = hashlib.sha256(_key(named).encode()).hexdigest()[:12]
            unique.setdefault(digest, {"named": named, "cases": []})
            unique[digest]["cases"].append({"offset": offset, "block_size": block_size})
            rows.append(
                {
                    "offset": offset,
                    "block_size": block_size,
                    "i32_aligned_16": (4 * offset) % 16 == 0,
                    "bool_aligned_16": offset % 16 == 0,
                    "key": digest,
                    "named": {k: str(v) for k, v in named.items()},
                }
            )
            if compile_sm121 and offset == 0:
                options, signature, constexprs, attrs = kernel._pack_args(
                    backend, {}, params, specialization, {}
                )
                compiled = triton_compile(
                    ASTSource(kernel, signature, constexprs, attrs),
                    target=target,
                    options=options.__dict__,
                )
                ttir = compiled.asm["ttir"]
                unique[digest]["ttir_has_kernel"] = (
                    "_compute_global_topk_indices_and_lens_kernel" in ttir
                )
                unique[digest]["ttir_sha256"] = hashlib.sha256(ttir.encode()).hexdigest()[
                    :16
                ]
    return {
        "label": label,
        "path": str(path),
        "unique_keys": len(unique),
        "rows": rows,
        "families": [
            {
                "key": digest,
                "cases": payload["cases"],
                "ttir_has_kernel": payload.get("ttir_has_kernel"),
                "ttir_sha256": payload.get("ttir_sha256"),
            }
            for digest, payload in unique.items()
        ],
        "module": module,
    }


def _reference(topk_indices, block_table, block_size, req_indices, valid_tokens):
    expected_indices = torch.full_like(topk_indices, -1)
    expected_lens = torch.zeros(len(req_indices), dtype=torch.int32, device=topk_indices.device)
    for token_idx, (req_idx, is_valid) in enumerate(zip(req_indices, valid_tokens)):
        if not is_valid:
            continue
        valid_count = 0
        for topk_idx, local_idx in enumerate(topk_indices[token_idx].tolist()):
            if local_idx < 0:
                continue
            block_idx = local_idx // block_size
            if block_idx >= block_table.shape[1]:
                continue
            block_number = int(block_table[req_idx, block_idx].item())
            if block_number < 0 or block_number >= NUM_BLOCKS:
                continue
            expected_indices[token_idx, topk_idx] = (
                block_number * block_size + local_idx % block_size
            )
            valid_count += 1
        expected_lens[token_idx] = valid_count
    return expected_indices, expected_lens


def _numerical(baseline_mod, patched_mod) -> dict:
    device = torch.device("cuda")
    block_table = torch.tensor(
        [
            [1, 2, -1, 30, 4, 5, 6, 7],
            [3, 4, 5, 6, 7, -1, 30, 0],
        ],
        dtype=torch.int32,
        device=device,
    )
    mismatches = []
    cases = 0
    for block_size in BLOCK_SIZES:
        topk_indices = torch.tensor(
            [
                [0, 1, block_size, block_size + 1, -1, 2 * block_size + 1, 9999, 3],
                [0, 1, block_size, block_size + 1, -1, 2 * block_size + 1, 9999, 3],
                [1, block_size, block_size + 1, 3 * block_size, -1, 9999, 0, 2],
            ],
            dtype=torch.int32,
            device=device,
        )
        expected_indices, expected_lens = _reference(
            topk_indices, block_table, block_size, REQ_INDICES, VALID_TOKENS
        )
        for token_offset in OFFSETS:
            token_parent = torch.empty(
                token_offset + len(REQ_INDICES), dtype=torch.int32, device=device
            )
            valid_parent = torch.empty(
                token_offset + len(VALID_TOKENS), dtype=torch.bool, device=device
            )
            token_to_req_indices = token_parent[token_offset:]
            is_valid_token = valid_parent[token_offset:]
            token_to_req_indices.copy_(
                torch.tensor(REQ_INDICES, dtype=torch.int32, device=device)
            )
            is_valid_token.copy_(
                torch.tensor(VALID_TOKENS, dtype=torch.bool, device=device)
            )
            kwargs = dict(
                topk_indices=topk_indices,
                token_to_req_indices=token_to_req_indices,
                block_table=block_table,
                block_size=block_size,
                is_valid_token=is_valid_token,
                num_blocks=NUM_BLOCKS,
            )
            b_idx, b_lens = baseline_mod.compute_global_topk_indices_and_lens(**kwargs)
            p_idx, p_lens = patched_mod.compute_global_topk_indices_and_lens(**kwargs)
            cases += 1
            if not torch.equal(b_idx, expected_indices) or not torch.equal(
                b_lens, expected_lens
            ):
                mismatches.append(
                    {"side": "baseline", "offset": token_offset, "block_size": block_size}
                )
            if not torch.equal(p_idx, expected_indices) or not torch.equal(
                p_lens, expected_lens
            ):
                mismatches.append(
                    {"side": "patched", "offset": token_offset, "block_size": block_size}
                )
            if not torch.equal(b_idx, p_idx) or not torch.equal(b_lens, p_lens):
                mismatches.append(
                    {
                        "side": "baseline_vs_patched",
                        "offset": token_offset,
                        "block_size": block_size,
                    }
                )
    return {
        "cuda": True,
        "cases": cases,
        "mismatches": mismatches,
        "ok": not mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--patched", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    compile_sm121 = not args.no_compile
    baseline = _analyze("baseline", args.baseline, compile_sm121)
    patched = _analyze("patched", args.patched, compile_sm121)
    numerical = {"cuda": False, "skipped": True}
    if torch.cuda.is_available():
        numerical = _numerical(baseline["module"], patched["module"])

    baseline.pop("module")
    patched.pop("module")
    report = {
        "triton": getattr(triton, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "baseline_unique_keys": baseline["unique_keys"],
        "patched_unique_keys": patched["unique_keys"],
        "key_family_collapsed": baseline["unique_keys"] == 6
        and patched["unique_keys"] == 2,
        "baseline": baseline,
        "patched": patched,
        "numerical": numerical,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")

    ok = patched["unique_keys"] == 2 and baseline["unique_keys"] > patched["unique_keys"]
    if compile_sm121:
        ok = ok and all(fam.get("ttir_has_kernel") for fam in patched["families"])
    if numerical.get("cuda"):
        ok = ok and numerical["ok"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
