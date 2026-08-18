#!/usr/bin/env python3
"""CPU gates for the #51967 top-k constexpr hotfix (patches/hotfix-dsv4-topk-compile-time-consts.py).

Deterministic and offline: the patcher is run as a subprocess against a copy of
this repo's own vendored kernel source
(`recipe/overlay/vllm/models/deepseek_v4/common/ops/cache_utils.py`), pointed
there by the `VLLM_ROOT` override the patcher now honours. No GPU, no docker,
no network.

Asserted:
  * the 12-line signature anchor occurs exactly once in the vendored source
    (so `replace(..., 1)` cannot mis-target), and the idempotence marker is
    absent before patching;
  * applying produces exactly upstream #51967's transformation — 5 added and 5
    removed lines, and those ten lines are precisely the five parameter
    promotions upstream merged, nothing else;
  * a second apply prints `[skip]` and leaves the file byte-identical;
  * `--status` reports NOT APPLIED before and APPLIED after, exits 0, and never
    writes;
  * a corrupted anchor exits non-zero with `[ERR] anchor not found` and leaves
    the file byte-identical (fail-closed, no partial write).

The vendored overlay is the repo's own copy of the target file, not the served
image's; the served copy is checked at boot by the patcher's own anchor guard,
which fails the container closed under the `DSPARK_ENABLE_TOPK_CONSTEXPR_HOTFIX`
gate.
"""
from __future__ import annotations

import difflib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOTFIX = ROOT / "patches" / "hotfix-dsv4-topk-compile-time-consts.py"
OVERLAY = ROOT / "recipe/overlay/vllm/models/deepseek_v4/common/ops/cache_utils.py"
REL_TARGET = "models/deepseek_v4/common/ops/cache_utils.py"

MARKER = "global_topk_indices_stride: tl.constexpr"

# The 12-line signature prefix the patcher anchors on.
ANCHOR = """def _compute_global_topk_indices_and_lens_kernel(
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

# Upstream vllm-project/vllm#51967 promotes exactly these five parameters and
# nothing else (+5/-5 in one file). Encoded as data so the gate fails if the
# backport ever drifts from upstream's hunk.
UPSTREAM_PROMOTIONS = [
    ("    global_topk_indices_stride,", "    global_topk_indices_stride: tl.constexpr,"),
    ("    topk_indices_stride,", "    topk_indices_stride: tl.constexpr,"),
    ("    topk,", "    topk: tl.constexpr,"),
    ("    block_table_stride,", "    block_table_stride: tl.constexpr,"),
    ("    block_size,", "    block_size: tl.constexpr,"),
]


def run_hotfix(vllm_root: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, VLLM_ROOT=str(vllm_root))
    return subprocess.run(
        [sys.executable, str(HOTFIX), *args],
        env=env, capture_output=True, text=True,
    )


class TopkConstexprPatchApplyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="topk-constexpr-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.vllm_root = self.tmp / "vllm"
        self.target = self.vllm_root / REL_TARGET
        self.target.parent.mkdir(parents=True)
        shutil.copyfile(OVERLAY, self.target)
        self.original = self.target.read_text(encoding="utf-8")

    # --- anchor -----------------------------------------------------------
    def test_anchor_unique_and_marker_absent_before_patch(self):
        self.assertEqual(self.original.count(ANCHOR), 1)
        self.assertNotIn(MARKER, self.original)

    # --- exact upstream diff ---------------------------------------------
    def test_apply_produces_exactly_upstream_51967_diff(self):
        r = run_hotfix(self.vllm_root)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("[OK]", r.stdout)

        patched = self.target.read_text(encoding="utf-8")
        diff = list(difflib.unified_diff(
            self.original.splitlines(), patched.splitlines(), n=0, lineterm=""))
        removed = [l[1:] for l in diff if l.startswith("-") and not l.startswith("---")]
        added = [l[1:] for l in diff if l.startswith("+") and not l.startswith("+++")]

        self.assertEqual(len(removed), 5, diff)
        self.assertEqual(len(added), 5, diff)
        self.assertEqual(removed, [old for old, _ in UPSTREAM_PROMOTIONS])
        self.assertEqual(added, [new for _, new in UPSTREAM_PROMOTIONS])

        # The fork-local `num_blocks` bound stays a runtime argument, and the
        # already-constexpr TRITON_BLOCK_SIZE is untouched.
        self.assertIn("\n    num_blocks,\n", patched)
        self.assertEqual(patched.count("TRITON_BLOCK_SIZE: tl.constexpr"), 1)
        self.assertEqual(patched.count(MARKER), 1)

    # --- idempotence ------------------------------------------------------
    def test_second_apply_skips_and_is_byte_identical(self):
        self.assertEqual(run_hotfix(self.vllm_root).returncode, 0)
        once = self.target.read_bytes()

        r = run_hotfix(self.vllm_root)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("[skip] hotfix already applied", r.stdout)
        self.assertEqual(self.target.read_bytes(), once)

    # --- --status ---------------------------------------------------------
    def test_status_reports_state_without_writing(self):
        before = run_hotfix(self.vllm_root, "--status")
        self.assertEqual(before.returncode, 0, before.stderr)
        self.assertIn("NOT APPLIED", before.stdout)
        self.assertEqual(self.target.read_text(encoding="utf-8"), self.original)

        self.assertEqual(run_hotfix(self.vllm_root).returncode, 0)
        patched = self.target.read_bytes()

        after = run_hotfix(self.vllm_root, "--status")
        self.assertEqual(after.returncode, 0, after.stderr)
        self.assertIn("APPLIED", after.stdout)
        self.assertNotIn("NOT APPLIED", after.stdout)
        self.assertEqual(self.target.read_bytes(), patched)

    def test_status_on_missing_target_is_not_applied(self):
        r = run_hotfix(self.tmp / "absent", "--status")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("NOT APPLIED", r.stdout)

    # --- fail-closed ------------------------------------------------------
    def test_corrupted_anchor_fails_closed_without_writing(self):
        corrupted = self.original.replace(ANCHOR, ANCHOR.replace("\n    topk,\n", "\n    topk_n,\n"), 1)
        self.assertNotEqual(corrupted, self.original)
        self.assertNotIn(MARKER, corrupted)
        self.target.write_text(corrupted, encoding="utf-8")

        r = run_hotfix(self.vllm_root)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("[ERR] anchor not found", r.stderr)
        self.assertEqual(self.target.read_text(encoding="utf-8"), corrupted)


if __name__ == "__main__":
    unittest.main()
