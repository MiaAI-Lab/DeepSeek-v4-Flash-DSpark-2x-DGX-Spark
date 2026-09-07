#!/usr/bin/env python3
"""CPU regressions for the bench-baseline container lifecycle.

Both bench-baseline scripts used to (a) tear down only the HEAD project with a
bare `docker compose down`, leaving the worker rank serving so the following
start failed its worker precheck, and (b) launch the start script in the
background, never check its exit code, wait up to 10 minutes for an API that
could never come, and finally `kill $START_PID` — a PID that by then belonged
to a recycled, unrelated process.

The launcher blocks until the API is up (or exits non-zero), so the scripts
now stop through stop-deepseek-v4-flash-dspark.sh (both nodes) and run the
start script in the foreground. These tests pin that structure.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    ROOT / "scripts" / "bench-baseline-issue22-only.sh",
    ROOT / "scripts" / "bench-baseline-no-patches.sh",
]


def code_of(script: Path) -> str:
    # Comments keep the history of why the old pattern was wrong; they must not
    # trip the guards. Assertions below run on code only.
    return "\n".join(line for line in script.read_text().splitlines()
                     if not line.strip().startswith("#"))


class Teardown(unittest.TestCase):
    def test_no_bare_compose_down(self):
        for script in SCRIPTS:
            with self.subTest(script=script.name):
                self.assertIsNone(
                    re.search(r"docker compose[^\n]*\bdown\b", code_of(script)),
                    "head-only compose down returned",
                )

    def test_stop_script_used_for_teardown(self):
        for script in SCRIPTS:
            text = script.read_text()
            with self.subTest(script=script.name):
                # One stop before the baseline start, one before the patched
                # restart.
                self.assertGreaterEqual(
                    text.count('bash "$SCRIPT_DIR/stop-deepseek-v4-flash-dspark.sh"'), 2,
                )


class Startup(unittest.TestCase):
    def test_start_runs_in_foreground(self):
        for script in SCRIPTS:
            for lineno, line in enumerate(script.read_text().splitlines(), 1):
                if "start-deepseek-v4-flash-dspark.sh" in line and not line.strip().startswith("#"):
                    with self.subTest(script=script.name, line=lineno):
                        self.assertFalse(line.rstrip().endswith("&"),
                                         f"backgrounded launcher: {line!r}")

    def test_no_start_pid_bookkeeping(self):
        for script in SCRIPTS:
            with self.subTest(script=script.name):
                self.assertNotIn("START_PID", code_of(script))
                self.assertNotRegex(code_of(script), r"kill \$")

    def test_no_local_wait_loop(self):
        # The launcher waits for readiness itself; a local curl loop only hid
        # launcher failure behind 10 minutes of dots.
        for script in SCRIPTS:
            with self.subTest(script=script.name):
                self.assertNotIn("Waiting for API", script.read_text())

    def test_stop_precedes_every_start(self):
        for script in SCRIPTS:
            text = script.read_text()
            starts = [m.start() for m in re.finditer(
                r"bash \"\$SCRIPT_DIR/start-deepseek-v4-flash-dspark\.sh\"", text)]
            stops = [m.start() for m in re.finditer(
                r"bash \"\$SCRIPT_DIR/stop-deepseek-v4-flash-dspark\.sh\"", text)]
            with self.subTest(script=script.name):
                self.assertEqual(len(starts), len(stops))
                for s_start, s_stop in zip(starts, stops):
                    self.assertLess(s_stop, s_start,
                                    "a start is not preceded by a stop")


if __name__ == "__main__":
    unittest.main()
