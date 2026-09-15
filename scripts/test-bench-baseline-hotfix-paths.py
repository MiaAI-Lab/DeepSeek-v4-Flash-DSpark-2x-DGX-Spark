#!/usr/bin/env python3
"""Host-only regression: the issue22 baseline reaches its hotfix where compose mounts it.

bench-baseline-issue22-only.sh runs in a temporary checkout whose `docker` is a
fake transport: `exec … bash <path>` resolves <path> inside an isolated
representation of the compose mount (the checkout's `patches/` directory stands
for the container's /opt/dspark-patches) and fails when the mount cannot back it,
the way the container's bash would. The launcher and scripts/bench-ttft.py are
fixtures, so the recorded steps are the order in which the shipped script reaches
them. No Docker, SSH, GPU, model reload or benchmark endpoint is touched.

  * the mounted hotfix is invoked, and the measurement runs after it
  * a hotfix missing from the mount aborts the run before any measurement

The pre-fix Step 3 targeted /tmp/hotfix-nvfp4-ds-mla-issue22.sh, which no mount
backs, so that run aborted at Step 3 without reaching the measurement.
"""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash") or "/bin/bash"
BENCH = ROOT / "scripts" / "bench-baseline-issue22-only.sh"

HOTFIX = "hotfix-nvfp4-ds-mla-issue22.sh"

# Fake docker transport. Compose teardown is inert, the patch-state query returns
# zero counts, and a `bash <path>` exec succeeds only for a file the mount
# representation holds: anything else fails like the container's bash, which
# under the bench's `pipefail` aborts the run at Step 3.
DOCKER_TRANSPORT = """#!/bin/sh
case "$1" in
  compose)
    exit 0 ;;
  exec)
    shift 2
    [ "$1" = "bash" ] && shift
    if [ "$1" = "-c" ]; then printf '0 0 0 0\\n'; exit 0; fi
    case "$1" in
      /opt/dspark-patches/*) file="$FAKE_MOUNT_ROOT/${1#/opt/dspark-patches/}" ;;
      *) file="" ;;
    esac
    if [ -n "$file" ] && [ -f "$file" ]; then
      printf '{"step": "hotfix"}\\n' >> "$LIFECYCLE_LOG"
      exit 0
    fi
    printf 'bash: %s: No such file or directory\\n' "$1" >&2
    exit 127 ;;
esac
exit 0
"""

# Stands in for the shipped launcher, which cannot start a server here.
START_FIXTURE = """#!/bin/sh
exit 0
"""

MEASURE_FIXTURE = """#!/usr/bin/env python3
import json
import os

with open(os.environ["LIFECYCLE_LOG"], "a") as log:
    log.write(json.dumps({"step": "measure"}) + "\\n")
"""

# The bench's own downtime pauses and its endpoint probe: inert, so an unstubbed
# call cannot reach a live system.
COMMAND_STUBS = {
    "docker": DOCKER_TRANSPORT,
    "sleep": "#!/bin/sh\nexit 0\n",
    "curl": "#!/bin/sh\nexit 7\n",
}


def write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


class Issue22HotfixInvocation(unittest.TestCase):
    def setUp(self):
        self.workdir = Path(tempfile.mkdtemp(prefix="bench-hotfix-paths-"))
        self.addCleanup(shutil.rmtree, self.workdir)
        self.checkout = self.workdir / "checkout"
        scripts = self.checkout / "scripts"
        scripts.mkdir(parents=True)
        shutil.copyfile(BENCH, scripts / BENCH.name)
        write_executable(self.checkout / "start-deepseek-v4-flash-dspark.sh", START_FIXTURE)
        write_executable(self.checkout / "stop-deepseek-v4-flash-dspark.sh", "#!/bin/sh\nexit 0\n")
        write_executable(scripts / "bench-ttft.py", MEASURE_FIXTURE)
        self.mount = self.checkout / "patches"
        self.mount.mkdir()
        shutil.copyfile(ROOT / "patches" / HOTFIX, self.mount / HOTFIX)
        self.bindir = self.workdir / "bin"
        self.bindir.mkdir()
        for command, content in COMMAND_STUBS.items():
            write_executable(self.bindir / command, content)
        self.log = self.workdir / "lifecycle.jsonl"
        self.env = {
            "PATH": f"{self.bindir}:/usr/bin:/bin",
            "HOME": str(self.workdir),
            "LC_ALL": "C",
            "LIFECYCLE_LOG": str(self.log),
            "FAKE_MOUNT_ROOT": str(self.mount),
        }

    def steps(self):
        return [json.loads(line)["step"] for line in self.log.read_text().splitlines()]

    def run_bench(self):
        self.log.write_text("")
        return subprocess.run(
            [BASH, str(self.checkout / "scripts" / BENCH.name)],
            cwd=self.checkout, env=self.env, input="y\n",
            capture_output=True, text=True, timeout=60,
        )

    def test_mounted_hotfix_is_invoked_before_measurement(self):
        result = self.run_bench()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.steps(), ["hotfix", "measure"])

    def test_unmounted_hotfix_aborts_before_measurement(self):
        (self.mount / HOTFIX).unlink()
        result = self.run_bench()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.steps(), [])


if __name__ == "__main__":
    unittest.main()
