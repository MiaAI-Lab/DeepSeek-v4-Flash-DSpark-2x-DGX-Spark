#!/usr/bin/env python3
"""Host-only lifecycle regressions for the two bench-baseline scripts.

Each shipped bench script runs in a temporary checkout whose stop script, start
launcher, docker and bench-ttft.py are recorders; no Docker, SSH, GPU or API is
touched, and the recorders log the order in which the bench reaches them. The
launcher recorder returns only after a delay, so a launcher that stops being
awaited would be observed measuring before it finished. Assertions are on that
recorded order and on the bench's exit status:

  * a failed stop or a failed launcher aborts the bench before it measures
  * a successful launcher completes before the bench measures
  * both teardowns (before the baseline start and before the patched restart)
    go through the coordinated stop script
"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash") or "/bin/bash"
BENCH = (
    ROOT / "scripts" / "bench-baseline-issue22-only.sh",
    ROOT / "scripts" / "bench-baseline-no-patches.sh",
)

# Stands in for the shipped two-node stop script: records the teardown, then
# reports the operator's knob for a stop that could not bring the ranks down.
STOP_RECORDER = """#!/bin/sh
printf '{"step": "stop"}\\n' >> "$LIFECYCLE_LOG"
exit "${FAIL_STOP:-0}"
"""

# Stands in for the shipped launcher and its block-until-ready contract: the
# record is written when the launcher returns, from a background process only
# after the delay, so an unawaited launcher loses the race to the measurement.
START_RECORDER = """#!/bin/sh
"$LIFECYCLE_PYTHON" -c 'import time; time.sleep(0.5)'
printf '{"step": "start"}\\n' >> "$LIFECYCLE_LOG"
exit "${FAIL_START:-0}"
"""

MEASURE_RECORDER = """#!/usr/bin/env python3
import json
import os

with open(os.environ["LIFECYCLE_LOG"], "a") as log:
    log.write(json.dumps({"step": "measure"}) + "\\n")
"""

# The bench scripts' own downtime pauses, container queries and any endpoint
# probe: instant and inert, so an unstubbed call cannot reach a live system.
COMMAND_STUBS = {
    "docker": "printf '0 0 0 0\\n'\n",
    "sleep": "exit 0\n",
    "curl": "exit 7\n",
}


def write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


class BenchLifecycle(unittest.TestCase):
    def setUp(self):
        self.workdir = Path(tempfile.mkdtemp(prefix="bench-baseline-lifecycle-"))
        self.addCleanup(shutil.rmtree, self.workdir)
        self.checkout = self.workdir / "checkout"
        scripts = self.checkout / "scripts"
        scripts.mkdir(parents=True)
        for script in BENCH:
            shutil.copyfile(script, scripts / script.name)
        write_executable(self.checkout / "stop-deepseek-v4-flash-dspark.sh", STOP_RECORDER)
        write_executable(self.checkout / "start-deepseek-v4-flash-dspark.sh", START_RECORDER)
        write_executable(scripts / "bench-ttft.py", MEASURE_RECORDER)
        bindir = self.workdir / "bin"
        bindir.mkdir()
        for command, body in COMMAND_STUBS.items():
            write_executable(bindir / command, "#!/bin/sh\n" + body)
        self.log = self.workdir / "lifecycle.jsonl"
        self.env = {
            "PATH": f"{bindir}:/usr/bin:/bin",
            "HOME": str(self.workdir),
            "LC_ALL": "C",
            "LIFECYCLE_LOG": str(self.log),
            "LIFECYCLE_PYTHON": sys.executable or "python3",
        }

    def run_bench(self, script: Path, fail_stop: bool = False, fail_start: bool = False):
        self.log.write_text("")
        env = dict(self.env, FAIL_STOP="1" if fail_stop else "0",
                   FAIL_START="1" if fail_start else "0")
        result = subprocess.run(
            [BASH, str(self.checkout / "scripts" / script.name)],
            cwd=self.checkout, env=env, input="y\n",
            capture_output=True, text=True, timeout=60,
        )
        events = [json.loads(line) for line in self.log.read_text().splitlines()]
        return result, [event["step"] for event in events]

    def test_failed_stop_aborts_before_measurement(self):
        for script in BENCH:
            with self.subTest(script=script.name):
                result, steps = self.run_bench(script, fail_stop=True)
                self.assertNotEqual(result.returncode, 0, result.stderr)
                self.assertEqual(steps, ["stop"])

    def test_failed_start_aborts_before_measurement(self):
        for script in BENCH:
            with self.subTest(script=script.name):
                result, steps = self.run_bench(script, fail_start=True)
                self.assertNotEqual(result.returncode, 0, result.stderr)
                self.assertEqual(steps, ["stop", "start"])

    def test_successful_launcher_completes_before_measurement(self):
        for script in BENCH:
            with self.subTest(script=script.name):
                result, steps = self.run_bench(script)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertGreater(steps.index("measure"), steps.index("start"),
                                   "measured before the launcher returned")

    def test_both_teardowns_use_the_coordinated_stop_script(self):
        for script in BENCH:
            with self.subTest(script=script.name):
                result, steps = self.run_bench(script)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(steps, ["stop", "start", "measure", "stop", "start"])


if __name__ == "__main__":
    unittest.main()
