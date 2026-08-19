#!/usr/bin/env python3
"""The shipped #27 hotfix must honor DSPARK_MAX_INFLIGHT_PREFILLS (clamped 1–3)
because this image rejects --max-num-partial-prefills. Applies the real hotfix
entry point to a fixture copy of the admission-loop anchor.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOTFIX = ROOT / "patches" / "hotfix-dsv4-issue27-partial-prefill-concurrency.py"

FIXTURE = (
    "                num_running = len(self.running) + self.num_waiting_for_streaming_input\n"
    "                if num_running >= self.max_num_running_reqs:\n"
    "                    break\n"
    "                leftover = True\n"
)


def _apply_to(path: Path) -> None:
    txt = HOTFIX.read_text()
    marker = 'Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py")'
    txt = txt.replace(marker, f"Path({str(path)!r})")
    ns: dict = {}
    exec(compile(txt, str(HOTFIX), "exec"), ns)


class Issue27InflightCapTest(unittest.TestCase):
    def test_hotfix_source_has_env_override(self):
        src = HOTFIX.read_text()
        self.assertIn("DSPARK_MAX_INFLIGHT_PREFILLS", src)
        self.assertIn("if _pp_cap > 3:", src)
        env_at = src.find("'DSPARK_MAX_INFLIGHT_PREFILLS'")
        # fallback to SchedulerConfig still present after the env read
        self.assertIn("self.scheduler_config.max_num_partial_prefills", src)
        self.assertGreater(env_at, 0)

    def test_apply_injects_env_cap_into_admission_loop(self):
        tmp = Path(tempfile.mkdtemp()) / "scheduler.py"
        tmp.write_text(FIXTURE)
        try:
            _apply_to(tmp)
        except SystemExit as e:
            self.assertEqual(e.code, 0)
        patched = tmp.read_text()
        self.assertIn("# [issue27-hotfix]", patched)
        self.assertIn("DSPARK_MAX_INFLIGHT_PREFILLS", patched)
        self.assertIn("_pp_cap", patched)
        self.assertIn("len(self._inflight_prefills)", patched)
        try:
            _apply_to(tmp)
        except SystemExit as e:
            self.assertEqual(e.code, 0)
        self.assertEqual(patched, tmp.read_text())


if __name__ == "__main__":
    sys.exit(0 if unittest.main(verbosity=2) else 1)
