#!/usr/bin/env python3
"""Driverless regression test for issue #133's Triton specialization family."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

try:
    import torch
    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, make_backend
    from triton.compiler import compile as triton_compile
    from triton.runtime.jit import MockTensor, create_function_from_signature
except ModuleNotFoundError:
    torch = None
    triton = None


REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_UTILS = (
    REPO_ROOT / "recipe/overlay/vllm/models/deepseek_v4/common/ops/cache_utils.py"
)


@unittest.skipIf(triton is None, "requires the pinned vLLM/Triton runtime")
class Issue133SpecializationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
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
        previous_modules = {
            name: sys.modules.get(name, missing) for name in stub_modules
        }
        try:
            sys.modules.update(stub_modules)
            spec = importlib.util.spec_from_file_location(
                "issue133_cache_utils", CACHE_UTILS
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot load {CACHE_UTILS}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            for name, previous in previous_modules.items():
                if previous is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous
        cls.kernel = module._compute_global_topk_indices_and_lens_kernel
        cls.module = module
        cls.target = GPUTarget("cuda", 121, 32)
        cls.backend = make_backend(cls.target)
        cls.binder = create_function_from_signature(
            cls.kernel.signature, cls.kernel.params, cls.backend
        )

    def _specialization(self, token_offset: int, block_size: int):
        class OffsetTensor(MockTensor):
            def __init__(self, dtype, address: int) -> None:
                super().__init__(dtype)
                self.address = address

            def data_ptr(self) -> int:
                return self.address

        aligned_i32 = OffsetTensor(torch.int32, 0)
        token_to_req_indices = OffsetTensor(torch.int32, 4 * token_offset)
        is_valid_token = OffsetTensor(torch.bool, token_offset)
        params, specialization, _ = type(self).binder(
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
        return params, specialization

    def test_slice_alignment_does_not_change_specialization(self) -> None:
        # Same-slice offsets cover both aligned, int32-only aligned, and neither.
        offsets = (0, 4, 1, 16)
        for block_size in (64, 2):
            specializations = [
                self._specialization(offset, block_size)[1] for offset in offsets
            ]
            self.assertTrue(
                all(item == specializations[0] for item in specializations[1:])
            )

    def test_block_size_is_the_only_remaining_key_axis(self) -> None:
        specialization_64 = self._specialization(0, 64)[1]
        specialization_2 = self._specialization(0, 2)[1]
        differing_parameters = [
            parameter.name
            for parameter, left, right in zip(
                type(self).kernel.params, specialization_64, specialization_2
            )
            if left != right
        ]
        self.assertEqual(differing_parameters, ["block_size"])

    def test_both_block_sizes_compile_for_sm121(self) -> None:
        for block_size in (64, 2):
            params, specialization = self._specialization(0, block_size)
            options, signature, constexprs, attrs = type(self).kernel._pack_args(
                type(self).backend, {}, params, specialization, {}
            )
            source = ASTSource(type(self).kernel, signature, constexprs, attrs)
            compiled = triton_compile(
                source,
                target=type(self).target,
                options=options.__dict__,
            )
            ttir = compiled.asm["ttir"]
            self.assertIn("_compute_global_topk_indices_and_lens_kernel", ttir)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "requires a CUDA device for numerical validation",
    )
    def test_outputs_match_reference_for_all_alignment_classes(self) -> None:
        device = torch.device("cuda")
        num_blocks = 8
        block_table = torch.tensor(
            [
                [1, 2, -1, 30, 4, 5, 6, 7],
                [3, 4, 5, 6, 7, -1, 30, 0],
            ],
            dtype=torch.int32,
            device=device,
        )
        req_indices = (0, 1, 0)
        valid_tokens = (True, False, True)

        for block_size in (64, 2):
            topk_indices = torch.tensor(
                [
                    [0, 1, block_size, block_size + 1, -1, 2 * block_size + 1, 9999, 3],
                    [0, 1, block_size, block_size + 1, -1, 2 * block_size + 1, 9999, 3],
                    [1, block_size, block_size + 1, 3 * block_size, -1, 9999, 0, 2],
                ],
                dtype=torch.int32,
                device=device,
            )

            expected_indices = torch.full_like(topk_indices, -1)
            expected_lens = torch.zeros(
                len(req_indices), dtype=torch.int32, device=device
            )
            for token_idx, (req_idx, is_valid) in enumerate(
                zip(req_indices, valid_tokens)
            ):
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
                    if block_number < 0 or block_number >= num_blocks:
                        continue
                    expected_indices[token_idx, topk_idx] = (
                        block_number * block_size + local_idx % block_size
                    )
                    valid_count += 1
                expected_lens[token_idx] = valid_count

            for token_offset in (0, 4, 1, 16):
                token_parent = torch.empty(
                    token_offset + len(req_indices),
                    dtype=torch.int32,
                    device=device,
                )
                valid_parent = torch.empty(
                    token_offset + len(valid_tokens),
                    dtype=torch.bool,
                    device=device,
                )
                token_to_req_indices = token_parent[token_offset:]
                is_valid_token = valid_parent[token_offset:]
                token_to_req_indices.copy_(
                    torch.tensor(req_indices, dtype=torch.int32, device=device)
                )
                is_valid_token.copy_(
                    torch.tensor(valid_tokens, dtype=torch.bool, device=device)
                )

                actual_indices, actual_lens = type(
                    self
                ).module.compute_global_topk_indices_and_lens(
                    topk_indices,
                    token_to_req_indices,
                    block_table,
                    block_size,
                    is_valid_token,
                    num_blocks,
                )
                self.assertTrue(torch.equal(actual_indices, expected_indices))
                self.assertTrue(torch.equal(actual_lens, expected_lens))

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "requires a CUDA device for persistent-cache validation",
    )
    def test_fresh_triton_cache_has_two_stable_entries(self) -> None:
        cache_dir = tempfile.mkdtemp(prefix="issue133-triton-cache-")
        env = os.environ.copy()
        env["TRITON_CACHE_DIR"] = cache_dir
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--cache-gate"],
            cwd=str(REPO_ROOT),
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            self.fail(completed.stdout + completed.stderr)
        report = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(report["unique_kernel_dirs"], 2)
        self.assertTrue(report["mtimes_stable"])


def _kernel_cache_dirs(cache_root: Path) -> dict[Path, float]:
    found: dict[Path, float] = {}
    if not cache_root.exists():
        return found
    for child in cache_root.iterdir():
        if not child.is_dir():
            continue
        files = [path for path in child.rglob("*") if path.is_file()]
        if not any(
            "_compute_global_topk_indices_and_lens_kernel" in path.read_text(errors="ignore")
            for path in files
        ):
            continue
        found[child] = max(path.stat().st_mtime for path in files)
    return found


def _launch_alignment_family(module) -> None:
    device = torch.device("cuda")
    num_blocks = 8
    block_table = torch.tensor(
        [[1, 2, -1, 30, 4, 5, 6, 7], [3, 4, 5, 6, 7, -1, 30, 0]],
        dtype=torch.int32,
        device=device,
    )
    req_indices = (0, 1, 0)
    valid_tokens = (True, False, True)
    # Production DSV4-Flash index_topk is 512; keep a short row so the
    # constexpr top-k axis matches the live C4A buffer width.
    topk = 512
    for block_size in (64, 2):
        row = [0, 1, block_size, block_size + 1, -1, 2 * block_size + 1, 9999, 3]
        topk_indices = torch.full((3, topk), -1, dtype=torch.int32, device=device)
        topk_indices[:, : len(row)] = torch.tensor(row, dtype=torch.int32, device=device)
        for token_offset in (0, 4, 1, 16):
            token_parent = torch.empty(
                token_offset + len(req_indices), dtype=torch.int32, device=device
            )
            valid_parent = torch.empty(
                token_offset + len(valid_tokens), dtype=torch.bool, device=device
            )
            token_to_req_indices = token_parent[token_offset:]
            is_valid_token = valid_parent[token_offset:]
            token_to_req_indices.copy_(
                torch.tensor(req_indices, dtype=torch.int32, device=device)
            )
            is_valid_token.copy_(
                torch.tensor(valid_tokens, dtype=torch.bool, device=device)
            )
            module.compute_global_topk_indices_and_lens(
                topk_indices,
                token_to_req_indices,
                block_table,
                block_size,
                is_valid_token,
                num_blocks,
            )
    torch.cuda.synchronize()


def _cache_gate_main() -> int:
    cache_root = Path(os.environ["TRITON_CACHE_DIR"])
    cache_root.mkdir(parents=True, exist_ok=True)
    if any(cache_root.iterdir()):
        raise SystemExit(f"TRITON_CACHE_DIR is not empty: {cache_root}")
    Issue133SpecializationTest.setUpClass()
    _launch_alignment_family(Issue133SpecializationTest.module)
    first = _kernel_cache_dirs(cache_root)
    time.sleep(1.1)
    _launch_alignment_family(Issue133SpecializationTest.module)
    second = _kernel_cache_dirs(cache_root)
    report = {
        "unique_kernel_dirs": len(first),
        "dirs": [str(path) for path in sorted(first)],
        "mtimes_stable": first == second,
        "first_mtimes": {str(path): mtime for path, mtime in first.items()},
        "second_mtimes": {str(path): mtime for path, mtime in second.items()},
    }
    print(json.dumps(report))
    return 0 if report["unique_kernel_dirs"] == 2 and report["mtimes_stable"] else 1


if __name__ == "__main__":
    if "--cache-gate" in sys.argv:
        raise SystemExit(_cache_gate_main())
    unittest.main()
