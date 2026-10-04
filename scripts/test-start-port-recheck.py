#!/usr/bin/env python3
"""CPU regressions for the head-port re-check before `compose up -d`.

The launcher's head-port check ran minutes before the head bind (file syncs,
GID resolve, worker `up -d` intervene): a process taking the port in between
surfaced only as a failed bind at `up -d`, after worker ranks had already
started — a half-up cluster to clean up. The check is now a function called
early (fail fast) AND immediately before the head `up -d`, with guidance that
the worker ranks need ./stop-… when the late check trips.

These tests pin the call placement and exercise the shipped function against
a fake `ss`.
"""
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "start-deepseek-v4-flash-dspark.sh"
SOURCE = LAUNCHER.read_text()

_fn_start = SOURCE.index("assert_head_port_free() {")
FN = SOURCE[_fn_start:SOURCE.index("\n}", _fn_start) + 2]

FAKE_SS = """#!/usr/bin/env bash
echo "State Recv-Q Send-Q Local Address:Port Peer Address:Port"
if [ "${SS_BUSY:-0}" = "1" ]; then
  echo "LISTEN 0 4096 0.0.0.0:${VLLM_PORT:-8888} 0.0.0.0:*"
fi
exit 0
"""


def run_check(busy: bool, ss_present: bool = True) -> subprocess.CompletedProcess:
    workdir = Path(tempfile.mkdtemp())
    bindir = workdir / "bin"
    bindir.mkdir()
    if ss_present:
        f = bindir / "ss"
        f.write_text(FAKE_SS)
        f.chmod(0o755)
    script = f"""set -euo pipefail
PATH={shlex.quote(str(bindir))}:/usr/bin:/bin
export VLLM_PORT=8888
export SS_BUSY={"1" if busy else "0"}
{FN}
assert_head_port_free
"""
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    shutil.rmtree(workdir)
    return r


class CheckBehavior(unittest.TestCase):
    def test_port_free_passes(self):
        r = run_check(busy=False)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_port_busy_fails_with_message(self):
        r = run_check(busy=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("Port 8888 is already listening", r.stderr)

    def test_missing_ss_skips_check(self):
        r = run_check(busy=True, ss_present=False)
        self.assertEqual(r.returncode, 0, r.stderr)


class CallPlacement(unittest.TestCase):
    def test_function_defined_once_called_twice(self):
        self.assertEqual(SOURCE.count("assert_head_port_free() {"), 1)
        self.assertEqual(SOURCE.count("assert_head_port_free ||"), 2)

    def test_early_call_precedes_worker_sync(self):
        early = SOURCE.index("assert_head_port_free || exit 1")
        sync = SOURCE.index('"mkdir -p $REMOTE_WORKER_DIR"')
        self.assertLess(early, sync)

    def test_late_call_immediately_precedes_head_up(self):
        late = SOURCE.index('compose_base 0 "" up -d')
        prefix = SOURCE[:late]
        self.assertIn("assert_head_port_free || {", prefix)
        # Nothing but the failure block and the echo separate the re-check
        # from the bind.
        between = prefix[prefix.rindex("assert_head_port_free || {"):]
        self.assertIn("./stop-deepseek-v4-flash-dspark.sh", between)
        self.assertLess(between.count("\n"), 10)


if __name__ == "__main__":
    unittest.main()
