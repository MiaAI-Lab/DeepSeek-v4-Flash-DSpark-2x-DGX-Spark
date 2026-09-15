#!/usr/bin/env python3
"""CPU tests for patches/hotfix-vllm-dspark-swa-prefix.py (no vLLM needed)."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "scripts" / "fixtures" / "dspark-swa-prefix"
KV_FIXTURE = FIXTURES / "kv_cache_manager-752a3a504-stock.py"
SCHED_FIXTURE = FIXTURES / "scheduler-752a3a504-stock.py"
PATCHER = ROOT / "patches" / "hotfix-vllm-dspark-swa-prefix.py"
EMPTY_ENCODER = ROOT / "patches" / "hotfix-vllm-empty-encoder-output.py"
COMPOSE = ROOT / "docker-compose.dspark.yml"
START = ROOT / "start-deepseek-v4-flash-dspark.sh"
ENV_EXAMPLE = ROOT / ".env.dspark.example"
CI = ROOT / "scripts" / "ci-validate.sh"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SWA = _load(PATCHER, "dspark_swa_prefix")
GOOD = lambda name: SWA.EXPECTED_VLLM_VERSION  # noqa: E731

KV_ADDED_LINES = [
    b"dspark_window_size: int | None = None,",
    b"self.dspark_window_size = dspark_window_size",
    b"# [dspark-swa-prefix]",
    b"# DSpark fix: the draft model attends over a sliding window of",
    b"# `dspark_window_size` tokens, populated from the target's hidden",
    b"# states via `precompute_and_store_context_kv`. On a prefix-cache hit",
    b"# the target skips computing the cached prefix, so the draft's SWA",
    b"# cache would be missing the window and the draft degenerates. Force",
    b"# the target to always recompute the last `dspark_window_size` tokens",
    b"# so the draft always has its full window populated. The cost is a",
    b"# small recompute (128 tokens) relative to the cached prefix.",
    b"if self.dspark_window_size is not None and self.dspark_window_size > 0:",
    b"max_cache_hit_length = max(",
    b"request.num_tokens - 1 - self.dspark_window_size, 0",
]
SCHED_ADDED_LINES = [
    b"# [dspark-swa-prefix]",
    b"# DSpark fix: the draft model attends over a sliding window of",
    b"# `dspark_window_size` tokens. On a prefix-cache hit the target skips",
    b"# computing the cached prefix, so the draft's SWA cache would be",
    b"# missing the window and the draft degenerates. We force the target to",
    b"# always recompute the last `dspark_window_size` tokens (see",
    b"# KVCacheManager.get_computed_blocks). Read the window size from the",
    b"# draft model's HF config.",
    b"self.dspark_window_size: int | None = None",
    b"if speculative_config is not None and speculative_config.use_dspark():",
    b'draft_cfg = getattr(speculative_config, "draft_model_config", None)',
    b'hf_cfg = getattr(draft_cfg, "hf_config", None)',
    b'window = getattr(hf_cfg, "sliding_window", None)',
    b"if window is not None and window > 0:",
    b"self.dspark_window_size = int(window)",
    b"dspark_window_size=self.dspark_window_size,",
]


class Transform(unittest.TestCase):
    def test_fixtures_match_stock_pins(self):
        for spec, fixture in ((SWA.KV, KV_FIXTURE), (SWA.SCHED, SCHED_FIXTURE)):
            data = fixture.read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), spec.stock_sha256)
            self.assertEqual(len(data), spec.stock_size)

    def test_transform_is_pinned_minimal_and_compiles(self):
        for spec, fixture, grown, added_lines in (
            (SWA.KV, KV_FIXTURE, 15, KV_ADDED_LINES),
            (SWA.SCHED, SCHED_FIXTURE, 17, SCHED_ADDED_LINES),
        ):
            stock = fixture.read_bytes()
            patched = SWA.transform(spec, stock)
            self.assertEqual(hashlib.sha256(patched).hexdigest(), spec.patched_sha256)
            self.assertEqual(len(patched), spec.patched_size)
            compile(patched, spec.label, "exec")
            self.assertEqual(patched.count(SWA.MARK.encode()), 1)
            # pure insertion: no stock line is removed or rewritten
            stock_lines = stock.splitlines()
            patched_lines = patched.splitlines()
            self.assertEqual(len(patched_lines), len(stock_lines) + grown)
            added = [line for line in patched_lines if line not in stock_lines]
            removed = [line for line in stock_lines if line not in patched_lines]
            self.assertEqual([line.strip() for line in added], added_lines)
            self.assertEqual(removed, [])

    def test_transform_refuses_foreign_or_patched_bytes(self):
        for spec, fixture in ((SWA.KV, KV_FIXTURE), (SWA.SCHED, SCHED_FIXTURE)):
            with self.assertRaises(SWA.HotfixError):
                SWA.transform(spec, b"def nothing():\n    pass\n")
            patched = SWA.transform(spec, fixture.read_bytes())
            with self.assertRaises(SWA.HotfixError):
                SWA.transform(spec, patched)

    def test_cache_hit_cap_semantics(self):
        """The patched cap: DSpark recomputes the last window; else stock."""
        patched = SWA.transform(SWA.KV, KV_FIXTURE.read_bytes()).decode("utf-8")
        self.assertIn(
            "max_cache_hit_length = max(\n"
            "                request.num_tokens - 1 - self.dspark_window_size, 0\n"
            "            )",
            patched,
        )

        def hit_cap(num_tokens, window):
            max_cache_hit_length = num_tokens - 1
            if window is not None and window > 0:
                max_cache_hit_length = max(num_tokens - 1 - window, 0)
            return max_cache_hit_length

        self.assertEqual(hit_cap(4096, None), 4095)  # no DSpark: stock arithmetic
        self.assertEqual(hit_cap(4096, 0), 4095)  # degenerate window: stock
        self.assertEqual(hit_cap(4096, 128), 3967)  # last window always recomputed
        self.assertEqual(hit_cap(130, 128), 1)
        self.assertEqual(hit_cap(129, 128), 0)
        self.assertEqual(hit_cap(64, 128), 0)  # short prompts never go negative

    def test_scheduler_transform_commutes_with_empty_encoder_co_owner(self):
        """scheduler.py is co-owned at boot; both orders yield identical bytes."""
        ee = _load(EMPTY_ENCODER, "empty_encoder_output")
        stock = SCHED_FIXTURE.read_text(encoding="utf-8")
        ee_first, status = ee.patch_scheduler_text(stock)
        self.assertEqual(status, "applied")
        swa_after_ee = SWA.transform(SWA.SCHED, ee_first.encode("utf-8"))
        swa_first = SWA.transform(SWA.SCHED, stock.encode("utf-8"))
        ee_after_swa, status = ee.patch_scheduler_text(swa_first.decode("utf-8"))
        self.assertEqual(status, "applied")
        self.assertEqual(swa_after_ee, ee_after_swa.encode("utf-8"))
        self.assertEqual(SWA.classify(SWA.SCHED, swa_after_ee), "patched")


class Patcher(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dspark-swa-prefix-"))
        self.kv = self.tmp / "kv_cache_manager.py"
        self.sched = self.tmp / "scheduler.py"
        shutil.copyfile(KV_FIXTURE, self.kv)
        shutil.copyfile(SCHED_FIXTURE, self.sched)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _apply(self):
        return SWA.apply(self.kv, self.sched, provider=GOOD)

    def test_apply_then_idempotent(self):
        self.assertEqual(
            self._apply(),
            {"kv_cache_manager.py": "applied", "scheduler.py": "applied"},
        )
        kv_data = self.kv.read_bytes()
        sched_data = self.sched.read_bytes()
        self.assertEqual(hashlib.sha256(kv_data).hexdigest(), SWA.KV.patched_sha256)
        self.assertEqual(hashlib.sha256(sched_data).hexdigest(), SWA.SCHED.patched_sha256)
        self.assertEqual(
            self._apply(),
            {"kv_cache_manager.py": "already-patched", "scheduler.py": "already-patched"},
        )
        self.assertEqual(self.kv.read_bytes(), kv_data)
        self.assertEqual(self.sched.read_bytes(), sched_data)
        self.assertEqual(
            sorted(p.name for p in self.tmp.iterdir()),
            ["kv_cache_manager.py", "scheduler.py"],
        )

    def test_apply_heals_a_torn_pair(self):
        self.kv.write_bytes(SWA.transform(SWA.KV, self.kv.read_bytes()))
        self.assertEqual(
            self._apply(),
            {"kv_cache_manager.py": "already-patched", "scheduler.py": "applied"},
        )

    def test_apply_accepts_co_owned_scheduler_preimage(self):
        """In the boot chain the always-on scheduler patchers run first."""
        ee = _load(EMPTY_ENCODER, "empty_encoder_output_pre")
        pre, status = ee.patch_scheduler_text(self.sched.read_text(encoding="utf-8"))
        self.assertEqual(status, "applied")
        self.sched.write_text(pre, encoding="utf-8")
        self.assertEqual(
            self._apply(),
            {"kv_cache_manager.py": "applied", "scheduler.py": "applied"},
        )
        after = self.sched.read_text(encoding="utf-8")
        self.assertEqual(ee.patch_scheduler_text(after)[1], "skipped")
        self.assertEqual(SWA.classify(SWA.SCHED, after.encode("utf-8")), "patched")

    def test_refuses_foreign_bytes_and_writes_nothing(self):
        self.kv.write_bytes(b"x = 1\n")
        sched_before = self.sched.read_bytes()
        with self.assertRaises(SWA.HotfixError):
            self._apply()
        self.assertEqual(self.kv.read_bytes(), b"x = 1\n")
        self.assertEqual(self.sched.read_bytes(), sched_before)

    def test_preflights_both_targets_before_writing_either(self):
        kv_before = self.kv.read_bytes()
        self.sched.write_bytes(b"x = 1\n")
        with self.assertRaises(SWA.HotfixError):
            self._apply()
        self.assertEqual(self.kv.read_bytes(), kv_before)
        self.assertEqual(self.sched.read_bytes(), b"x = 1\n")

    def test_refuses_wrong_vllm_version(self):
        with self.assertRaises(SWA.HotfixError):
            SWA.apply(self.kv, self.sched, provider=lambda name: "0.26.0")
        self.assertEqual(hashlib.sha256(self.kv.read_bytes()).hexdigest(), SWA.KV.stock_sha256)

    def test_refuses_symlink(self):
        link = self.tmp / "link.py"
        os.symlink(self.kv, link)
        with self.assertRaises(SWA.HotfixError):
            SWA.inspect(SWA.KV, link, provider=GOOD)

    def test_cli_check_and_status_do_not_write(self):
        env = dict(os.environ)
        before = (self.kv.read_bytes(), self.sched.read_bytes())
        for flag in ("--check", "--status"):
            proc = subprocess.run(
                [
                    sys.executable,
                    str(PATCHER),
                    flag,
                    "--kv-target",
                    str(self.kv),
                    "--sched-target",
                    str(self.sched),
                ],
                capture_output=True,
                text=True,
                env=env,
            )
            # the real vllm is not installed here -> fail closed, no write
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            self.assertIn("FAIL-CLOSED", proc.stderr)
            self.assertEqual((self.kv.read_bytes(), self.sched.read_bytes()), before)


class Wiring(unittest.TestCase):
    def test_compose_gate_default_off_fail_closed(self):
        compose = COMPOSE.read_text()
        self.assertIn(
            'DSPARK_ENABLE_DSPARK_SWA_PREFIX: "${DSPARK_ENABLE_DSPARK_SWA_PREFIX:-0}"',
            compose,
        )
        self.assertIn(
            'if [ "$${DSPARK_ENABLE_DSPARK_SWA_PREFIX:-0}" = "1" ]; then '
            "python3 /opt/hotfix-vllm-dspark-swa-prefix.py || exit 1; fi;",
            compose,
        )
        self.assertIn(
            "${DSPARK_DSPARK_SWA_PREFIX_HOTFIX:-./patches/hotfix-vllm-dspark-swa-prefix.py}"
            ":/opt/hotfix-vllm-dspark-swa-prefix.py:ro",
            compose,
        )

    def test_launcher_passthrough_sync_and_preflight(self):
        start = START.read_text()
        self.assertIn(
            "DSPARK_DSPARK_SWA_PREFIX_HOTFIX='./patches/hotfix-vllm-dspark-swa-prefix.py'",
            start,
        )
        self.assertIn("DSPARK_ENABLE_DSPARK_SWA_PREFIX=$REMOTE_DSPARK_SWA_PREFIX", start)
        self.assertIn("/opt/hotfix-vllm-dspark-swa-prefix.py --check", start)
        self.assertIn('patches/hotfix-vllm-dspark-swa-prefix.py"', start)

    def test_env_example_and_ci(self):
        env = ENV_EXAMPLE.read_text()
        self.assertIn("DSPARK_ENABLE_DSPARK_SWA_PREFIX=0", env)
        ci = CI.read_text()
        self.assertIn("scripts/test-dspark-swa-prefix.py", ci)
        self.assertIn("hotfix-vllm-dspark-swa-prefix.py", ci)


if __name__ == "__main__":
    unittest.main()
