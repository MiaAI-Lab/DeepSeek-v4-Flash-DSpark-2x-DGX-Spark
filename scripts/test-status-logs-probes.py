#!/usr/bin/env python3
"""CPU regressions for status/logs probe exit codes and the status port fix.

Two defects lived in the ops scripts:

1. `status-deepseek-v4-flash-dspark.sh` resolved `PORT="${PORT:-8888}"` BEFORE
   sourcing .env.dspark and used `$PORT` for the `ss` listing, while the API
   probe used `${VLLM_PORT:-8888}` — with a non-default VLLM_PORT the port
   listing watched the wrong port.
2. Both scripts ended every probe in `|| true` and always exited 0, so a
   supervisor or runbook treating them as health probes got a green light on a
   dead cluster.

The scripts now resolve VLLM_PORT once, after the env file is sourced (PORT
kept as a legacy alias), count real probe failures via note_failure() — head
compose ps, head image inspect, worker ssh on status; worker ssh on logs — and
exit 1 when any of them fails. Informational listings, and compose logs that
cannot be read on a stopped cluster, stay non-errors. ssh probes carry
BatchMode=yes and ConnectTimeout=10, which bound the connection attempt only;
neither option bounds the remote command that runs after it.

These tests run the shipped scripts end-to-end against fake docker/ssh/curl/ss
binaries, with host and port supplied through a scratch ENV_FILE, so no case
can reach a real node, container or port.
"""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "status-deepseek-v4-flash-dspark.sh"
LOGS = ROOT / "logs-deepseek-v4-flash-dspark.sh"

# Host and port enter through the file the probes source, so a case fails if
# either value is resolved from a pre-source default instead.
ENV_FILE_BODY = "WORKER_HOST=worker.example\nVLLM_PORT=9999\n"

# Printed by the fake docker when a failure is injected; asserting on it proves
# the failure really happened rather than resting on an always-0 fake.
DOCKER_LOGS_ERROR = "docker compose: no containers to show"

# Ambient values that would redirect a probe or select another branch if they
# leaked in from the calling shell or CI.
SCRUBBED_VARS = (
    "API_URL", "COMPOSE_FILE", "DSPARK_API_KEYS", "DSPARK_VLLM_IMAGE",
    "ENV_FILE", "LEGACY_PROJECT_NAME", "PORT", "PROJECT_NAME", "TAIL",
    "VLLM_API_KEY", "VLLM_HOST", "VLLM_PORT", "WORKER2_DIR", "WORKER2_HOST",
    "WORKER2_SCRIPT_DIR", "WORKER_DIR", "WORKER_HOST", "WORKER_SCRIPT_DIR",
)

FAKE_SSH = """#!/usr/bin/env bash
if [ "${SSH_BEHAVIOR:-ok}" = "fail" ]; then exit 255; fi
printf 'ssh %s\\n' "$*" >> "$PROBE_LOG"
exit 0
"""
FAKE_DOCKER = f"""#!/usr/bin/env bash
printf 'docker %s\\n' "$*" >> "$PROBE_LOG"
if [ "${{DOCKER_BEHAVIOR:-ok}}" = "fail" ]; then
  echo "{DOCKER_LOGS_ERROR}" >&2
  exit 1
fi
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
    """Run `script` end-to-end under the fake binaries and a scratch env.

    A value in env_over sets a variable; None removes it, so a case can assert
    explicit absence even when the parent shell carries one. The timeout bounds
    a hung probe, and the scratch tree is removed on every path.
    """
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        bindir = workdir / "bin"
        bindir.mkdir()
        for name, body in (("ssh", FAKE_SSH), ("docker", FAKE_DOCKER),
                           ("curl", FAKE_CURL), ("ss", FAKE_SS)):
            f = bindir / name
            f.write_text(body)
            f.chmod(0o755)
        env_file = workdir / ".env.dspark"
        env_file.write_text(ENV_FILE_BODY)
        log = workdir / "probe.log"
        env = dict(os.environ)
        for name in SCRUBBED_VARS:
            env.pop(name, None)
        env.update({
            "PATH": f"{bindir}:{env.get('PATH', '/usr/bin:/bin')}",
            "HOME": str(workdir),
            "ENV_FILE": str(env_file),
            "PROBE_LOG": str(log),
        })
        for key, value in env_over.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        result = subprocess.run(["bash", str(script)], capture_output=True,
                                text=True, env=env, cwd=ROOT, timeout=60)
        result.probe_log = log.read_text() if log.exists() else ""  # type: ignore[attr-defined]
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


class LogsProbes(unittest.TestCase):
    def test_healthy_cluster_exits_0(self):
        r = run(LOGS, {})
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_unreachable_worker_exits_1(self):
        r = run(LOGS, {"SSH_BEHAVIOR": "fail"})
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("worker logs ssh failed", r.stderr)

    def test_compose_logs_failure_is_not_an_error(self):
        # A stopped cluster's `docker compose logs` fails for real here — the
        # injected message has to reach the caller — yet unreadable logs stay
        # acceptable. Only a node that could not be queried is an error
        # (test_unreachable_worker_exits_1).
        r = run(LOGS, {"DOCKER_BEHAVIOR": "fail"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(DOCKER_LOGS_ERROR, r.stderr)


if __name__ == "__main__":
    unittest.main()
