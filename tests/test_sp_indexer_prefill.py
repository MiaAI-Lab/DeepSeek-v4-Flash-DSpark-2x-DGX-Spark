#!/usr/bin/env python3
"""CPU tests for the sequence-parallel indexer prefill hotfix.

1. The patcher applies to the real image ``sparse_attn_indexer.py`` (extracted
   with ``docker run``; skipped when docker/image are unavailable), compiles,
   and is idempotent.
2. The split / local-bounds math (pure torch, CPU) matches a brute-force
   reference for many request shapes and TP sizes: slices are page-aligned,
   disjoint, cover every request exactly, and every query's local [ks, ke)
   covers exactly the keys of its global range that fall in this rank's slice.
"""
from __future__ import annotations

import importlib.util
import os
import py_compile
import shutil
import platform
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOTFIX = ROOT / "patches" / "hotfix-dsv4-sp-indexer-prefill.py"
REAL = (
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "sparse_attn_indexer.py"
)
IMAGE = "ghcr.io/anemll/dspark-vllm-gx10:0.1.1"


def _load_patch_module():
    spec = importlib.util.spec_from_file_location("sp_patch", HOTFIX)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _image_runnable() -> bool:
    """True only when the pinned image is already present locally and its
    architecture matches this host (CI runners are amd64 and must neither pull
    the 19 GB arm64 image nor try to exec it)."""
    if shutil.which("docker") is None:
        return False
    try:
        r = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Architecture}}", IMAGE],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if r.returncode != 0:
        return False
    arch, host = r.stdout.strip(), platform.machine()
    return (arch, host) in {("arm64", "aarch64"), ("arm64", "arm64"), ("amd64", "x86_64")}


_docker_available = _image_runnable


class PatchApplyTest(unittest.TestCase):
    @unittest.skipUnless(_docker_available(), "docker image not available")
    def test_applies_compiles_idempotent(self):
        mod = _load_patch_module()
        with tempfile.TemporaryDirectory() as tmpd:
            target = Path(tmpd) / "sparse_attn_indexer.py"
            with target.open("w") as fh:
                subprocess.run(["docker", "run", "--rm", "--entrypoint", "cat", IMAGE, REAL],
                               check=True, stdout=fh, timeout=120)
            src = target.read_text()
            self.assertIn("for chunk in prefill_metadata.chunks:", src)
            self.assertTrue(mod.apply(target))
            patched = target.read_text()
            self.assertIn(mod.MARK, patched)
            self.assertIn("_sp_indexer_prefill_chunk(", patched)
            self.assertIn("get_tp_group", patched)
            py_compile.compile(str(target), doraise=True)
            self.assertFalse(mod.apply(target))
            self.assertEqual(target.read_text(), patched)


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


@unittest.skipUnless(_torch_available(), "torch not installed (run inside the image)")
class SplitMathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch
        mod = _load_patch_module()
        ns: dict = {"torch": torch, "os": os}
        # Only the pure-torch helpers are needed; stub the rest.
        for name in ("ops", "fp8_fp4_mqa_logits", "has_cutedsl", "current_platform",
                     "get_tensor_model_parallel_world_size",
                     "get_tensor_model_parallel_rank", "get_tp_group"):
            ns[name] = None
        exec(compile(mod.HELPER_SRC, "helper", "exec"), ns)
        cls.helpers = ns  # plain dict: avoids bound-method self injection

    def _check(self, lens, block_rows, tp):
        torch = self.torch
        cu = torch.zeros(len(lens) + 1, dtype=torch.int32)
        cu[1:] = torch.cumsum(torch.tensor(lens, dtype=torch.int32), 0)
        # queries: for each request a few (ks, ke) with ks == request start
        ks, ke, exp = [], [], []
        for i, L in enumerate(lens):
            for rel in sorted({L, max(L - 1, 0), L // 2, min(L, 1), 0, min(L, 513)}):
                ks.append(int(cu[i]))
                ke.append(int(cu[i]) + rel)
                exp.append((i, rel))
        ks_t = torch.tensor(ks, dtype=torch.int32)
        ke_t = torch.tensor(ke, dtype=torch.int32)
        covered = [torch.zeros(L, dtype=torch.int32) for L in lens]
        for r in range(tp):
            lo, local_len, local_cu = self.helpers["_sp_indexer_split"](cu, block_rows, tp, r)
            self.assertEqual(lo.dtype, torch.int32)
            self.assertEqual(local_cu.dtype, cu.dtype)
            for i, L in enumerate(lens):
                l, n = int(lo[i]), int(local_len[i])
                self.assertEqual(l % block_rows, 0, "lo must be page aligned")
                self.assertGreaterEqual(n, 0)
                self.assertLessEqual(l + n, L)
                covered[i][l : l + n] += 1
            self.assertEqual(int(local_cu[-1]), int(local_len.sum()))
            lo_q, ks_l, ke_l = self.helpers["_sp_indexer_local_bounds"](cu, ks_t, ke_t, lo, local_len, local_cu)
            for qi, (i, rel) in enumerate(exp):
                l, n = int(lo[i]), int(local_len[i])
                # keys of the query's global range [0, rel) that fall in [l, l+n)
                want = max(0, min(rel, l + n) - l)
                if lens[i] > 0:
                    # empty requests share a cu value with their successor; the
                    # helper deliberately resolves to the non-empty one
                    # (searchsorted right=True), which is harmless because an
                    # empty request has no keys (range stays empty below).
                    self.assertEqual(int(lo_q[qi]), l)
                    self.assertEqual(int(ks_l[qi]), int(local_cu[i]))
                self.assertEqual(int(ke_l[qi]) - int(ks_l[qi]), want,
                                 f"lens={lens} tp={tp} r={r} q={(i, rel)}")
        for i, L in enumerate(lens):
            self.assertTrue(bool((covered[i] == 1).all()),
                            f"rank slices must partition request {i}: {lens} tp={tp}")

    def test_partitions_and_bounds(self):
        for tp in (2, 3, 4):
            for lens in ([3000, 5000], [64], [63], [1], [0, 500], [65, 7000, 130],
                         [100000], [8192, 8192, 8192], [129, 128, 127]):
                self._check(lens, 64, tp)

    def test_zero_and_small_requests_do_not_break_search(self):
        torch = self.torch
        cu = torch.tensor([0, 0, 10, 10, 20], dtype=torch.int32)  # requests 0 and 2 are empty
        lo, local_len, local_cu = self.helpers["_sp_indexer_split"](cu, 4, 2, 1)
        ks = torch.tensor([0, 0, 10, 10], dtype=torch.int32)
        ke = torch.tensor([0, 10, 10, 20], dtype=torch.int32)
        lo_q, ks_l, ke_l = self.helpers["_sp_indexer_local_bounds"](cu, ks, ke, lo, local_len, local_cu)
        self.assertTrue(bool((ke_l >= ks_l).all()))
        self.assertTrue(bool((ke_l <= int(local_cu[-1])).all()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
