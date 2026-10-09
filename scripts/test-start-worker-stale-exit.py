#!/usr/bin/env python3
"""CPU regressions for the worker stale-rank exit-code contract in start.

README documents (and systemd consumes via SuccessExitStatus=3) that
./start-… exits **3** when the cluster is already running. The head
already-running path honoured that, but the worker/worker2 stale-rank
prechecks exited with the raw ssh status — 1 for "container found", 255 for
"unreachable" — so a supervised start treated a healthy, already-up pair as
a failed start, and conflated the two cases in one message.

The checks now answer with sentinel 42 from the remote probe: 42 (stale
rank) -> exit 3 with the already-running hint; any other failure (ssh
unreachable etc.) -> exit 1. These tests extract the shipped blocks and run
them with the real transport wrapper against fake ssh/docker binaries.
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


def extract(start_marker: str, end_marker: str) -> str:
    i = SOURCE.index(start_marker)
    return SOURCE[i:SOURCE.index(end_marker, i) + len(end_marker)]


HINT_FN = extract("already_running_hint() {", "\n}")
TRANSPORT_FN = extract("dssh() {", "\n")
WORKER_BLOCK = extract("worker_rc=0", "esac")
WORKER2_BLOCK = extract("worker2_rc=0", "esac")

FAKE_SSH = """#!/usr/bin/env bash
while [ "${1:-}" = "-o" ]; do shift 2; done
host="$1"; shift
if [ "$host" = "unreachable.example" ]; then exit 255; fi
exec bash -c "$*"
"""


def run_block(block: str, *, stale: bool,
              unreachable: bool = False) -> subprocess.CompletedProcess:
    workdir = Path(tempfile.mkdtemp())
    bindir = workdir / "bin"
    bindir.mkdir()
    (bindir / "ssh").write_text(FAKE_SSH)
    (bindir / "ssh").chmod(0o755)
    docker_stub = "#!/usr/bin/env bash\n"
    if stale:
        docker_stub += 'echo "deepseek-v4-flash-vllm-dspark-1"\n'
    (bindir / "docker").write_text(docker_stub)
    (bindir / "docker").chmod(0o755)
    host = "unreachable.example" if unreachable else "worker.example"
    script = f"""set -euo pipefail
PATH={shlex.quote(str(bindir))}:/usr/bin:/bin
PROJECT_NAME=deepseek-v4-flash
WORKER_HOST={host}
WORKER2_HOST=worker2.example
DSPARK_TP3=1
{HINT_FN}
{TRANSPORT_FN}
{block}
echo BLOCK_PASSED
"""
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    shutil.rmtree(workdir)
    return result


class WorkerCheck(unittest.TestCase):
    def test_clean_worker_passes(self):
        result = run_block(WORKER_BLOCK, stale=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("BLOCK_PASSED", result.stdout)

    def test_stale_worker_exits_3_with_hint(self):
        result = run_block(WORKER_BLOCK, stale=True)
        self.assertEqual(result.returncode, 3, result.stderr)

    def test_unreachable_worker_exits_1(self):
        result = run_block(WORKER_BLOCK, stale=False, unreachable=True)
        self.assertEqual(result.returncode, 1, result.stderr)

    def test_stale_worker2_exits_3_with_hint(self):
        result = run_block(WORKER2_BLOCK, stale=True)
        self.assertEqual(result.returncode, 3, result.stderr)


if __name__ == "__main__":
    unittest.main()
