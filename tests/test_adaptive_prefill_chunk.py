#!/usr/bin/env python3
"""Behavioral tests for the adaptive long-prefill chunk-cap hotfix."""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOTFIX = ROOT / "patches" / "hotfix-dsv4-adaptive-prefill-chunk.py"

FIXTURE = '''\
class Req:
    def __init__(self, is_prefill_chunk):
        self.is_prefill_chunk = is_prefill_chunk


class Scheduler:
    def __init__(self, threshold=1024, max_batched=8192):
        self.scheduler_config = type(
            "Cfg",
            (),
            {
                "long_prefill_token_threshold": threshold,
                "max_num_batched_tokens": max_batched,
            },
        )()
        self.running = []

    def cap_running(self, num_new_tokens):
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            return num_new_tokens

    def cap_waiting(self, num_new_tokens):
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold
                    return num_new_tokens
'''


def _apply_to(path: Path) -> None:
    txt = HOTFIX.read_text()
    marker = 'Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py")'
    txt = txt.replace(marker, f"Path({str(path)!r})")
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            exec(compile(txt, str(HOTFIX), "exec"), {})
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise


def _load(**kwargs):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "scheduler.py"
        path.write_text(FIXTURE)
        _apply_to(path)
        ns: dict = {}
        exec(compile(path.read_text(), str(path), "exec"), ns)
        sched = ns["Scheduler"](**kwargs)
        return sched, ns["Req"], path.read_text()


class AdaptivePrefillChunkTest(unittest.TestCase):
    def test_cold_prefill_uses_full_batch(self):
        sched, _, _ = _load()
        self.assertEqual(sched.cap_running(8192), 8192)
        self.assertEqual(sched.cap_waiting(8192), 8192)

    def test_running_prefills_only_still_uncapped(self):
        sched, Req, _ = _load()
        sched.running = [Req(True), Req(True)]
        self.assertEqual(sched.cap_running(4096), 4096)

    def test_decode_active_restores_1024(self):
        sched, Req, _ = _load()
        sched.running = [Req(True), Req(False)]
        self.assertEqual(sched.cap_running(8192), 1024)
        self.assertEqual(sched.cap_waiting(8192), 1024)

    def test_disabled_threshold_stays_disabled(self):
        sched, Req, _ = _load(threshold=0)
        sched.running = []
        self.assertEqual(sched.cap_running(8192), 8192)
        sched.running = [Req(False)]
        self.assertEqual(sched.cap_running(8192), 8192)

    def test_apply_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scheduler.py"
            path.write_text(FIXTURE)
            _apply_to(path)
            patched = path.read_text()
            _apply_to(path)
            self.assertEqual(path.read_text(), patched)
            self.assertIn("# [dspark-adaptive-chunk]", patched)


if __name__ == "__main__":
    sys.exit(0 if unittest.main(verbosity=2) else 1)
