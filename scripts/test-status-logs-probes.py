#!/usr/bin/env python3
"""CPU regressions for status/logs probe exit codes and the status port fix.

Two defects lived in the ops scripts:

1. `status-deepseek-v4-flash-dspark.sh` resolved `PORT="${PORT:-8888}"` BEFORE
   sourcing .env.dspark and used `$PORT` for the `ss` listing, while the API
   probe used `${VLLM_PORT:-8888}` — with VLLM_PORT=9000 the port listing
   watched the wrong port.
2. Both scripts ended every probe in `|| true` and always exited 0, so a
   supervisor or runbook treating them as health probes got a green light on a
   dead cluster.

The scripts now resolve VLLM_PORT once (PORT kept as a legacy alias), count
probe failures via note_failure(), and exit 1 when any real probe fails
(container-grep/ss listings stay informational; compose-logs absence on a
stopped cluster is not an error). ssh probes carry BatchMode+ConnectTimeout
so a stalled node fails fast instead of hanging the probe.

These tests run the shipped scripts end-to-end against fake docker/ssh/curl/ss
binaries.
"""
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "status-deepseek-v4-flash-dspark.sh"
LOGS = ROOT / "logs-deepseek-v4-flash-dspark.sh"

FAKE_SSH = """#!/usr/bin/env bash
if [ "${SSH_BEHAVIOR:-ok}" = "fail" ]; then exit 255; fi
printf 'ssh %s\\n' "$*" >> "$PROBE_LOG"
exit 0
"""
FAKE_DOCKER = """#!/usr/bin/env bash
printf 'docker %s\\n' "$*" >> "$PROBE_LOG"
exit 0
"""
FAKE_CURL = """#!/usr/bin/env bash
if [ "${CURL_BEHAVIOR:-ok}" = "fail" ]; then exit 7; fi
printf 'curl %s\\n' "$*" >> "$PROBE_LOG"
echo '{"data":[{"id":"deepseek-v4-flash-vision-exp"}]}'
exit 0
"""
FAKE_SS = """#!/usr/bin/env bash
printf 'ss %s\\n' "$*" >> "$PROBE_LOG"
exit 0
"""


def run(script: Path, env_over: dict) -> subprocess.CompletedProcess:
    workdir = Path(tempfile.mkdtemp())
    bindir = workdir / "bin"
    bindir.mkdir()
    for name, body in (("ssh", FAKE_SSH), ("docker", FAKE_DOCKER),
                       ("curl", FAKE_CURL), ("ss", FAKE_SS)):
        f = bindir / name
        f.write_text(body)
        f.chmod(0o755)
    log = workdir / "probe.log"
    env = dict(os.environ)
    env.update({
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "PROBE_LOG": str(log),
        "ENV_FILE": "/dev/null",
        "WORKER_HOST": "worker.example",
        "VLLM_PORT": "9999",
    })
    env.pop("WORKER2_HOST", None)
    env.pop("API_URL", None)
    env.update(env_over)
    result = subprocess.run(["bash", str(script)], capture_output=True,
                            text=True, env=env, cwd=ROOT)
    result.probe_log = log.read_text() if log.exists() else ""  # type: ignore[attr-defined]
    shutil.rmtree(workdir)
    return result


class StatusProbes(unittest.TestCase):
    def test_healthy_cluster_exits_0(self):
        r = run(STATUS, {})
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_api_failure_exits_1(self):
        r = run(STATUS, {"CURL_BEHAVIOR": "fail"})
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("API probe failed", r.stderr)

    def test_unreachable_worker_exits_1(self):
        r = run(STATUS, {"SSH_BEHAVIOR": "fail"})
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("worker compose ps failed", r.stderr)

    def test_ss_watches_vllm_port_not_legacy_default(self):
        r = run(STATUS, {})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("sport = :9999", r.probe_log)
        self.assertNotIn("sport = :8888", r.probe_log)

    def test_ssh_probes_fail_fast(self):
        r = run(STATUS, {})
        self.assertEqual(r.returncode, 0, r.stderr)
        for line in r.probe_log.splitlines():
            if line.startswith("ssh "):
                self.assertIn("BatchMode=yes", line)
                self.assertIn("ConnectTimeout=10", line)


class LogsProbes(unittest.TestCase):
    def test_healthy_cluster_exits_0(self):
        r = run(LOGS, {})
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_unreachable_worker_exits_1(self):
        r = run(LOGS, {"SSH_BEHAVIOR": "fail"})
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("worker logs ssh failed", r.stderr)

    def test_compose_logs_failure_is_not_an_error(self):
        # A stopped cluster legitimately has no logs; only the ssh transport
        # is counted.
        r = run(LOGS, {})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("logs --tail=", r.probe_log)

    def test_ssh_probes_fail_fast(self):
        r = run(LOGS, {})
        for line in r.probe_log.splitlines():
            if line.startswith("ssh "):
                self.assertIn("ConnectTimeout=10", line)


class SourceShape(unittest.TestCase):
    def test_status_has_no_pre_env_port_constant(self):
        text = STATUS.read_text()
        env_source = text.index('source "$ENV_FILE"')
        self.assertNotIn('PORT="${PORT:-8888}"', text[:env_source])
        self.assertIn('ss -ltn "( sport = :$VLLM_PORT )"', text)

    def test_both_scripts_exit_1_on_failures(self):
        self.assertIn('if [ "$STATUS_FAILURES" -gt 0 ]', STATUS.read_text())
        self.assertIn('if [ "$LOGS_FAILURES" -gt 0 ]', LOGS.read_text())


if __name__ == "__main__":
    unittest.main()
