#!/usr/bin/env python3
"""CPU tests for patches/hotfix-vllm-dspark-block-k.py (no vLLM needed)."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "scripts" / "fixtures" / "dspark-block-k" / "speculative-752a3a504-stock.py"
PATCHER = ROOT / "patches" / "hotfix-vllm-dspark-block-k.py"
COMPOSE = ROOT / "docker-compose.dspark.yml"
START = ROOT / "start-deepseek-v4-flash-dspark.sh"
ENV_EXAMPLE = ROOT / ".env.dspark.example"
CI = ROOT / "scripts" / "ci-validate.sh"


def _load():
    spec = importlib.util.spec_from_file_location("dspark_block_k", PATCHER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BK = _load()
GOOD = lambda name: BK.EXPECTED_VLLM_VERSION  # noqa: E731


class Transform(unittest.TestCase):
    def test_fixture_matches_stock_pin(self):
        data = FIXTURE.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), BK.STOCK_SHA256)
        self.assertEqual(len(data), BK.STOCK_SIZE)

    def test_transform_is_pinned_minimal_and_compiles(self):
        stock = FIXTURE.read_bytes()
        patched = BK.transform(stock)
        self.assertEqual(hashlib.sha256(patched).hexdigest(), BK.PATCHED_SHA256)
        self.assertEqual(len(patched), BK.PATCHED_SIZE)
        compile(patched, "speculative.py", "exec")
        self.assertEqual(patched.count(BK.MARK.encode()), 1)
        # exactly one added condition line; everything else byte-identical
        stock_lines = stock.splitlines()
        patched_lines = patched.splitlines()
        self.assertEqual(len(patched_lines), len(stock_lines) + 2)
        added = [line for line in patched_lines if line not in stock_lines]
        removed = [line for line in stock_lines if line not in patched_lines]
        self.assertEqual(
            [line.strip() for line in added],
            [
                b'self.method != "dspark"',
                BK.MARK.encode(),
                b"and self.num_speculative_tokens > n_predict",
            ],
        )
        self.assertEqual(
            [line.strip() for line in removed],
            [b"self.num_speculative_tokens > n_predict"],
        )
        self.assertIn(b"# Ensure divisibility for MTP module reuse.", patched)

    def test_transform_refuses_foreign_or_patched_bytes(self):
        with self.assertRaises(BK.HotfixError):
            BK.transform(b"def nothing():\n    pass\n")
        patched = BK.transform(FIXTURE.read_bytes())
        with self.assertRaises(BK.HotfixError):
            BK.transform(patched)

    def test_condition_semantics(self):
        """The patched predicate: dspark never raises; mtp keeps the rule."""
        patched = BK.transform(FIXTURE.read_bytes()).decode("utf-8")
        start = patched.index('self.method != "dspark"')
        block = patched[start - 40 : start + 400]
        self.assertIn("self.num_speculative_tokens > n_predict", block)
        self.assertIn("self.num_speculative_tokens % n_predict != 0", block)

        def raises(method, k, n_predict):
            return (
                method != "dspark" and k > n_predict and k % n_predict != 0
            )

        self.assertFalse(raises("dspark", 5, 3))
        self.assertFalse(raises("dspark", 7, 3))
        self.assertFalse(raises("dspark", 6, 3))
        self.assertTrue(raises("mtp", 5, 3))
        self.assertFalse(raises("mtp", 6, 3))
        self.assertFalse(raises("mtp", 2, 3))


class Patcher(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dspark-block-k-"))
        self.target = self.tmp / "speculative.py"
        shutil.copyfile(FIXTURE, self.target)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_apply_then_idempotent(self):
        self.assertEqual(BK.apply(self.target, provider=GOOD), "applied")
        data = self.target.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), BK.PATCHED_SHA256)
        self.assertEqual(BK.inspect(self.target, provider=GOOD)[0], "patched")
        self.assertEqual(BK.apply(self.target, provider=GOOD), "already-patched")
        self.assertEqual(self.target.read_bytes(), data)
        self.assertEqual([p.name for p in self.tmp.iterdir()], ["speculative.py"])

    def test_refuses_foreign_bytes(self):
        self.target.write_bytes(b"x = 1\n")
        with self.assertRaises(BK.HotfixError):
            BK.inspect(self.target, provider=GOOD)
        with self.assertRaises(BK.HotfixError):
            BK.apply(self.target, provider=GOOD)
        self.assertEqual(self.target.read_bytes(), b"x = 1\n")

    def test_refuses_wrong_vllm_version(self):
        with self.assertRaises(BK.HotfixError):
            BK.inspect(self.target, provider=lambda name: "0.26.0")
        self.assertEqual(hashlib.sha256(self.target.read_bytes()).hexdigest(), BK.STOCK_SHA256)

    def test_refuses_symlink(self):
        link = self.tmp / "link.py"
        os.symlink(self.target, link)
        with self.assertRaises(BK.HotfixError):
            BK.inspect(link, provider=GOOD)

    def test_cli_check_and_status_do_not_write(self):
        import subprocess

        env = dict(os.environ)
        before = self.target.read_bytes()
        for flag in ("--check", "--status"):
            proc = subprocess.run(
                [sys.executable, str(PATCHER), flag, "--target", str(self.target)],
                capture_output=True,
                text=True,
                env=env,
            )
            # the real vllm is not installed here -> fail closed, no write
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            self.assertIn("FAIL-CLOSED", proc.stderr)
            self.assertEqual(self.target.read_bytes(), before)


class Wiring(unittest.TestCase):
    def test_compose_gate_default_off_fail_closed(self):
        compose = COMPOSE.read_text()
        self.assertIn('DSPARK_ENABLE_DSPARK_BLOCK_K: "${DSPARK_ENABLE_DSPARK_BLOCK_K:-0}"', compose)
        self.assertIn(
            'if [ "$${DSPARK_ENABLE_DSPARK_BLOCK_K:-0}" = "1" ]; then '
            "python3 /opt/hotfix-vllm-dspark-block-k.py || exit 1; fi;",
            compose,
        )
        self.assertIn(
            "${DSPARK_DSPARK_BLOCK_K_HOTFIX:-./patches/hotfix-vllm-dspark-block-k.py}"
            ":/opt/hotfix-vllm-dspark-block-k.py:ro",
            compose,
        )

    def test_launcher_passthrough_sync_preflight_and_k_check(self):
        start = START.read_text()
        self.assertIn("DSPARK_DSPARK_BLOCK_K_HOTFIX='./patches/hotfix-vllm-dspark-block-k.py'", start)
        self.assertIn("DSPARK_ENABLE_DSPARK_BLOCK_K=$REMOTE_DSPARK_BLOCK_K", start)
        self.assertIn("/opt/hotfix-vllm-dspark-block-k.py --check", start)
        self.assertIn('patches/hotfix-vllm-dspark-block-k.py"', start)
        # the launcher's own k rule is relaxed only when the unlock is on
        self.assertIn('if [ "${DSPARK_ENABLE_DSPARK_BLOCK_K:-0}" = "1" ]; then', start)
        self.assertIn("MTP_NUM_TOKENS >= 5 and divisible by 3", start)

    def test_env_example_and_ci(self):
        env = ENV_EXAMPLE.read_text()
        self.assertIn("DSPARK_ENABLE_DSPARK_BLOCK_K=0", env)
        ci = CI.read_text()
        self.assertIn("scripts/test-dspark-block-k.py", ci)
        self.assertIn("hotfix-vllm-dspark-block-k.py", ci)


if __name__ == "__main__":
    unittest.main()
