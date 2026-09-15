#!/usr/bin/env python3
"""CPU regressions for env normalization and private worker publication."""
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "start-deepseek-v4-flash-dspark.sh"
SOURCE = LAUNCHER.read_text()


def extract_before(start: str, end: str) -> str:
    i = SOURCE.index(start)
    return SOURCE[i:SOURCE.index(end, i)]


ENV_BLOCK = extract_before("_dspark_env_clean=", "# GPU util comes from GPU_MEMORY_UTILIZATION_TEXT")
PUBLISH_BLOCK = extract_before(
    "# Stream into a private sibling",
    'ssh "$WORKER_HOST" "mkdir -p $REMOTE_WORKER_DIR/recipe/vllm/v1/spec_decode"',
)


def run_env(content: bytes, extra: str = "") -> subprocess.CompletedProcess:
    workdir = Path(tempfile.mkdtemp())
    tmpdir = workdir / "tmp"
    tmpdir.mkdir()
    env_file = workdir / ".env.dspark"
    env_file.write_bytes(content)
    script = f"""set -euo pipefail
export TMPDIR={shlex.quote(str(tmpdir))}
ENV_FILE={shlex.quote(str(env_file))}
{ENV_BLOCK}
printf 'WORKER_HOST=%q\nVLLM_PORT=%q\n' "${{WORKER_HOST:-<unset>}}" "${{VLLM_PORT:-<unset>}}"
{extra}
"""
    env = dict(os.environ)
    env.pop("DSPARK_API_KEYS", None)
    env.pop("VLLM_API_KEY", None)
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env=env,
    )
    result.operator_bytes = env_file.read_bytes()  # type: ignore[attr-defined]
    result.leftovers = sorted(path.name for path in tmpdir.iterdir())  # type: ignore[attr-defined]
    shutil.rmtree(workdir)
    return result


def write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


class EnvNormalisationTest(unittest.TestCase):
    def test_plain_file(self):
        result = run_env(b"WORKER_HOST=10.0.0.2\nVLLM_PORT=8888\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("WORKER_HOST=10.0.0.2", result.stdout)
        self.assertIn("VLLM_PORT=8888", result.stdout)
        self.assertEqual(result.leftovers, [])

    def test_bom_crlf_and_private_compose_snapshot(self):
        result = run_env(
            b"\xef\xbb\xbfWORKER_HOST=10.0.0.2\r\n\r\nVLLM_PORT=8888\r\n",
            'printf "MODE=%s\\n" "$(stat -c %a "$COMPOSE_ENV_FILE")"; cat "$COMPOSE_ENV_FILE"',
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("WORKER_HOST=10.0.0.2", result.stdout)
        self.assertIn("MODE=600", result.stdout)
        self.assertNotIn("\r", result.stdout)
        self.assertNotIn("\ufeff", result.stdout)
        self.assertEqual(result.leftovers, [])

    def test_operator_file_is_unchanged(self):
        raw = b"\xef\xbb\xbfWORKER_HOST=10.0.0.2\r\nVLLM_PORT=8888\r\n"
        result = run_env(raw)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.operator_bytes, raw)

    def test_source_failure_preserves_status_and_cleans(self):
        result = run_env(b"WORKER_HOST=(unbalanced\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.leftovers, [])

    def test_explicit_exit_preserves_status_and_cleans(self):
        result = run_env(b"exit 7\n")
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.leftovers, [])

    def test_signals_terminate_and_clean(self):
        for signal, status in (("HUP", 129), ("INT", 130), ("TERM", 143)):
            with self.subTest(signal=signal):
                result = run_env(f"kill -{signal} $$\n".encode())
                self.assertEqual(result.returncode, status, result.stderr)
                self.assertEqual(result.leftovers, [])

    def test_cleanup_is_armed_before_mktemp(self):
        self.assertLess(ENV_BLOCK.index("trap _cleanup_dspark_env EXIT"),
                        ENV_BLOCK.index("mktemp"))

    def test_all_local_compose_calls_use_normalized_snapshot(self):
        self.assertNotIn('--env-file "$ENV_FILE"', SOURCE)
        self.assertIn('docker compose -p "$PROJECT_NAME" --env-file "$COMPOSE_ENV_FILE"', SOURCE)
        self.assertNotIn("SIDECAR_COMPOSE_FILE", SOURCE)
        self.assertNotIn("docker-compose.vl-sidecar.yml", SOURCE)


class WorkerPublishTest(unittest.TestCase):
    def run_publish(self, *, space=False, fail_cat=False, fail_mv=False):
        workdir = Path(tempfile.mkdtemp(prefix="worker publish " if space else "worker-publish-"))
        bindir = workdir / "bin"
        bindir.mkdir()
        worker_dir = workdir / "remote dir" if space else workdir / "remote"
        worker_dir.mkdir()
        source = workdir / "normalized.env"
        source.write_text("VLLM_API_KEY=secret\n")
        source.chmod(0o600)
        final = worker_dir / ".env.dspark"
        final.write_text("OLD=1\n")
        final.chmod(0o644)
        before_log = workdir / "before.log"

        write_executable(bindir / "ssh", """#!/usr/bin/env bash
set -euo pipefail
shift
exec bash -c "$*"
""")
        write_executable(bindir / "cat", """#!/usr/bin/env bash
set -euo pipefail
if [ "${FAIL_CAT:-0}" = 1 ]; then
  IFS= read -r -n 4 partial || true
  printf %s "$partial"
  exit 7
fi
exec /bin/cat "$@"
""")
        write_executable(bindir / "mv", """#!/usr/bin/env bash
set -euo pipefail
printf '%s:%s\n' "$(/bin/cat "$FINAL_PATH")" "$(stat -c %a "$FINAL_PATH")" > "$BEFORE_LOG"
[ "${FAIL_MV:-0}" != 1 ] || exit 9
exec /bin/mv "$@"
""")

        script = f"""set -euo pipefail
PATH={shlex.quote(str(bindir))}:/usr/bin:/bin
WORKER_HOST=worker
REMOTE_ENV_FILE="$(printf %q {shlex.quote(str(final))})"
COMPOSE_ENV_FILE={shlex.quote(str(source))}
{PUBLISH_BLOCK}
echo OK
"""
        env = dict(os.environ)
        env.update({
            "FAIL_CAT": "1" if fail_cat else "0",
            "FAIL_MV": "1" if fail_mv else "0",
            "FINAL_PATH": str(final),
            "BEFORE_LOG": str(before_log),
        })
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
        result.final_text = final.read_text()  # type: ignore[attr-defined]
        result.final_mode = final.stat().st_mode & 0o777  # type: ignore[attr-defined]
        result.staged = sorted(path.name for path in worker_dir.glob(".env.dspark.tmp.*"))  # type: ignore[attr-defined]
        result.before_text = before_log.read_text() if before_log.exists() else None  # type: ignore[attr-defined]
        shutil.rmtree(workdir)
        return result

    def test_private_atomic_replace_with_space_path(self):
        result = self.run_publish(space=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.final_text, "VLLM_API_KEY=secret\n")
        self.assertEqual(result.final_mode, 0o600)
        self.assertEqual(result.before_text, "OLD=1:644\n")
        self.assertEqual(result.staged, [])

    def test_partial_stage_failure_keeps_old_destination_and_status(self):
        result = self.run_publish(fail_cat=True)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.final_text, "OLD=1\n")
        self.assertEqual(result.final_mode, 0o644)
        self.assertEqual(result.staged, [])
        self.assertNotIn("OK", result.stdout)

    def test_publish_failure_keeps_old_destination_and_status(self):
        result = self.run_publish(fail_mv=True)
        self.assertEqual(result.returncode, 9)
        self.assertEqual(result.final_text, "OLD=1\n")
        self.assertEqual(result.final_mode, 0o644)
        self.assertEqual(result.staged, [])
        self.assertNotIn("OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
