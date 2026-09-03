#!/usr/bin/env python3
"""Behavioral tests for fail-closed Python hotfix startup wiring."""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.dspark.yml"
START = ROOT / "start-deepseek-v4-flash-dspark.sh"
ENCODING_TOKEN = "python3 /opt/hotfix-encoding-dsv4-issue21.py"
RUNTIME_TOKEN = "python3 /opt/hotfix-dsv4-issue27-partial-prefill-concurrency.py"
RESPONSES_STORE_TOKEN = "python3 /opt/hotfix-dsv4-responses-store.py"
RESPONSES_LAUNCHER_BEGIN = "# Bounded Responses API store pre-flight (begin)."
RESPONSES_LAUNCHER_END = "# Bounded Responses API store pre-flight (end)."
ISSUE138_TOKEN = "python3 /opt/hotfix-vllm-issue138-responses-history.py"
LAUNCHER_BEGIN = "# Issue #138 Responses history compatibility pre-flight (begin)."
LAUNCHER_END = "# Issue #138 Responses history compatibility pre-flight (end)."
CODEX_AGENT_TOKEN = "python3 /opt/hotfix-vllm-codex-agent-message.py"
CODEX_LAUNCHER_BEGIN = "# Codex agent_message compatibility pre-flight (begin)."
CODEX_LAUNCHER_END = "# Codex agent_message compatibility pre-flight (end)."
ISSUE136_TOKEN = "python3 /opt/hotfix-vllm-issue136-xgrammar-termination.py"

PYTHON_STUB = """#!/usr/bin/env bash
case "${1:-}" in
  -c) step=inline-reasoning-map ;;
  *) step="${1##*/}" ;;
esac
printf '%s\n' "$step" >> "$INVOCATIONS"
if [ "$step" = "${FAIL_STEP:-}" ]; then
  exit "${FAIL_CODE:-7}"
fi
"""

CP_STUB = """#!/usr/bin/env bash
step=encoding-copy
printf '%s\n' "$step" >> "$INVOCATIONS"
if [ "$step" = "${FAIL_STEP:-}" ]; then
  exit "${FAIL_CODE:-7}"
fi
"""


def _compose_line(token: str) -> str:
    matches = [
        line.strip()
        for line in COMPOSE.read_text().splitlines()
        if token in line
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly one Compose command line containing {token!r}, "
            f"got {len(matches)}"
        )
    return matches[0]


def _optional_compose_line(token: str) -> str:
    matches = [
        line.strip()
        for line in COMPOSE.read_text().splitlines()
        if token in line
    ]
    return matches[0] if len(matches) == 1 else ""


class PythonHotfixFailClosedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoding_line = _compose_line(ENCODING_TOKEN)
        cls.runtime_line = _compose_line(RUNTIME_TOKEN)
        cls.responses_store_line = _compose_line(RESPONSES_STORE_TOKEN)
        cls.issue138_line = _compose_line(ISSUE138_TOKEN)
        start_source = START.read_text()
        cls.launcher_preflight = start_source.split(LAUNCHER_BEGIN, 1)[1].split(
            LAUNCHER_END, 1
        )[0]
        cls.responses_launcher_preflight = start_source.split(
            RESPONSES_LAUNCHER_BEGIN, 1
        )[1].split(RESPONSES_LAUNCHER_END, 1)[0]
        cls.codex_agent_line = _optional_compose_line(CODEX_AGENT_TOKEN)
        if CODEX_LAUNCHER_BEGIN in start_source and CODEX_LAUNCHER_END in start_source:
            cls.codex_launcher_preflight = start_source.split(
                CODEX_LAUNCHER_BEGIN, 1
            )[1].split(CODEX_LAUNCHER_END, 1)[0]
        else:
            cls.codex_launcher_preflight = ""
        cls.issue136_line = _compose_line(ISSUE136_TOKEN)

    def _run_line(
        self,
        line: str,
        *,
        fail_step: str | None = None,
        env_extra: dict[str, str | None] | None = None,
        missing_encoding: bool = False,
        after_command: str | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], list[str], bool]:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            for name, body in (("python3", PYTHON_STUB), ("cp", CP_STUB)):
                path = bin_dir / name
                path.write_text(body)
                path.chmod(0o755)

            invocations = root / "invocations.txt"
            invocations.touch()
            reached = root / "service-exec-reached"
            encoding = root / "encoding_dsv4.py"
            if not missing_encoding:
                encoding.write_text("# fixture\n")

            command = line.replace("$$", "$")
            if after_command is not None:
                command += "\n" + after_command
            command += f'\nprintf x > "{reached}"\n'
            env = dict(os.environ)
            env.update(
                {
                    "PATH": f"{bin_dir}:/usr/bin:/bin",
                    "INVOCATIONS": str(invocations),
                    "ENCODING_SOURCE": str(encoding),
                    "DSPARK_ENABLE_ISSUE31_GPU_HOTFIX": "0",
                    "DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX": "0",
                    "DSPARK_SKIP_SUPPRESS_STOPS_HOTFIX": "0",
                    # Opt-in (compose default 0); the chain tests model them
                    # enabled so their fail-closed behaviour is covered.
                    "DSPARK_ENABLE_ADAPTIVE_CHUNK": "1",
                    "DSPARK_ENABLE_REPLICATE_MARKOV": "1",
                }
            )
            env.pop("DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT", None)
            env.pop("DSPARK_ISSUE138_HOTFIX", None)
            env.pop("DSPARK_ENABLE_CODEX_AGENT_MESSAGE_COMPAT", None)
            env.pop("DSPARK_CODEX_AGENT_MESSAGE_HOTFIX", None)
            env.pop("VLLM_ENABLE_RESPONSES_API_STORE", None)
            env.pop("DSPARK_RESPONSES_STORE_MAX_ENTRIES", None)
            env.pop("DSPARK_RESPONSES_STORE_HOTFIX", None)
            env.pop("DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK", None)
            if fail_step is not None:
                env["FAIL_STEP"] = fail_step
                env["FAIL_CODE"] = "7"
            else:
                env.pop("FAIL_STEP", None)
                env.pop("FAIL_CODE", None)
            if env_extra:
                for key, value in env_extra.items():
                    if value is None:
                        env.pop(key, None)
                    else:
                        env[key] = value

            proc = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
            return proc, invocations.read_text().splitlines(), reached.exists()
    def _run_responses_preflight(
        self,
        *,
        enabled: str | None,
        capacity: str | None,
        patch_exists: bool,
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            patch = root / "hotfix.py"
            if patch_exists:
                patch.touch()
            env = dict(os.environ)
            env["SCRIPT_DIR"] = str(root)
            env["DSPARK_RESPONSES_STORE_HOTFIX"] = str(patch)
            if enabled is None:
                env.pop("VLLM_ENABLE_RESPONSES_API_STORE", None)
            else:
                env["VLLM_ENABLE_RESPONSES_API_STORE"] = enabled
            if capacity is None:
                env.pop("DSPARK_RESPONSES_STORE_MAX_ENTRIES", None)
            else:
                env["DSPARK_RESPONSES_STORE_MAX_ENTRIES"] = capacity
            command = (
                self.responses_launcher_preflight
                + "\nprintf '%s\\n%s\\n' "
                + '"$VLLM_ENABLE_RESPONSES_API_STORE" '
                + '"$DSPARK_RESPONSES_STORE_MAX_ENTRIES"'
            )
            return subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )


    def test_encoding_line_runs_enabled_steps_in_order(self):
        proc, invocations, reached = self._run_line(self.encoding_line)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(
            invocations,
            [
                "encoding-copy",
                "inline-reasoning-map",
                "hotfix-encoding-dsv4-issue21.py",
                "hotfix-dsv4-issue55-tool-truncation.py",
            ],
        )
        self.assertTrue(reached)

    def test_each_encoding_step_failure_blocks_later_steps_and_service_exec(self):
        order = [
            "encoding-copy",
            "inline-reasoning-map",
            "hotfix-encoding-dsv4-issue21.py",
            "hotfix-dsv4-issue55-tool-truncation.py",
        ]
        for step in order:
            with self.subTest(step=step):
                proc, invocations, reached = self._run_line(
                    self.encoding_line,
                    fail_step=step,
                )
                self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
                self.assertEqual(invocations, order[: order.index(step) + 1])
                self.assertFalse(reached)

    def test_missing_encoding_remains_warning_only_but_issue55_still_runs(self):
        proc, invocations, reached = self._run_line(
            self.encoding_line,
            missing_encoding=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(invocations, ["hotfix-dsv4-issue55-tool-truncation.py"])
        self.assertIn("encoding_dsv4.py not found", proc.stderr)
        self.assertTrue(reached)

    def test_enabled_issue31_failure_blocks_issue55_and_service_exec(self):
        proc, invocations, reached = self._run_line(
            self.encoding_line,
            fail_step="hotfix-dsv4-issue31-v2-thinking-budget-gpu.py",
            env_extra={"DSPARK_ENABLE_ISSUE31_GPU_HOTFIX": "1"},
        )
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertEqual(
            invocations,
            [
                "encoding-copy",
                "inline-reasoning-map",
                "hotfix-encoding-dsv4-issue21.py",
                "hotfix-dsv4-issue31-v2-thinking-budget-gpu.py",
            ],
        )
        self.assertFalse(reached)

    def test_enabled_issue31_success_runs_before_issue55(self):
        proc, invocations, reached = self._run_line(
            self.encoding_line,
            env_extra={"DSPARK_ENABLE_ISSUE31_GPU_HOTFIX": "1"},
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(
            invocations,
            [
                "encoding-copy",
                "inline-reasoning-map",
                "hotfix-encoding-dsv4-issue21.py",
                "hotfix-dsv4-issue31-v2-thinking-budget-gpu.py",
                "hotfix-dsv4-issue55-tool-truncation.py",
            ],
        )
        self.assertTrue(reached)

    def test_responses_store_only_invokes_for_exact_one(self):
        for value in (None, "0", "true", "2"):
            with self.subTest(value=value):
                extra = (
                    {}
                    if value is None
                    else {"VLLM_ENABLE_RESPONSES_API_STORE": value}
                )
                proc, invocations, reached = self._run_line(
                    self.responses_store_line,
                    env_extra=extra,
                    after_command=(
                        'printf "normalized=%s\\n" '
                        '"$VLLM_ENABLE_RESPONSES_API_STORE"'
                    ),
                )
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                self.assertEqual(invocations, [])
                self.assertTrue(reached)
                self.assertEqual(proc.stdout, "normalized=0\n")

        proc, invocations, reached = self._run_line(
            self.responses_store_line,
            env_extra={"VLLM_ENABLE_RESPONSES_API_STORE": "1"},
            after_command=(
                'printf "normalized=%s\\n" '
                '"$VLLM_ENABLE_RESPONSES_API_STORE"'
            ),
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(invocations, ["hotfix-dsv4-responses-store.py"])
        self.assertTrue(reached)
        self.assertEqual(proc.stdout, "normalized=1\n")

        proc, invocations, reached = self._run_line(
            self.responses_store_line,
            fail_step="hotfix-dsv4-responses-store.py",
            env_extra={"VLLM_ENABLE_RESPONSES_API_STORE": "1"},
        )
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertEqual(invocations, ["hotfix-dsv4-responses-store.py"])
        self.assertFalse(reached)

    def test_responses_store_preflight_is_fail_closed(self):
        disabled = self._run_responses_preflight(
            enabled="true", capacity="not-an-int", patch_exists=False
        )
        self.assertEqual(disabled.returncode, 0, disabled.stdout + disabled.stderr)
        self.assertEqual(disabled.stdout.splitlines(), ["0", "256"])

        enabled = self._run_responses_preflight(
            enabled="1", capacity="7", patch_exists=True
        )
        self.assertEqual(enabled.returncode, 0, enabled.stdout + enabled.stderr)
        self.assertEqual(enabled.stdout.splitlines(), ["1", "7"])

        missing = self._run_responses_preflight(
            enabled="1", capacity="7", patch_exists=False
        )
        self.assertEqual(missing.returncode, 1)
        self.assertIn("patcher is missing", missing.stderr)

        invalid = self._run_responses_preflight(
            enabled="1", capacity="0", patch_exists=True
        )
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("must be a positive integer", invalid.stderr)

    def test_issue138_unset_zero_and_non_one_values_do_not_invoke_patcher(self):
        for value in (None, "0", "true", "2"):
            with self.subTest(value=value):
                extra = ({} if value is None else {
                    "DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT": value
                })
                proc, invocations, reached = self._run_line(
                    self.issue138_line, env_extra=extra
                )
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                self.assertEqual(invocations, [])
                self.assertTrue(reached)

    def test_issue138_exact_one_runs_before_permanent_chain(self):
        proc, invocations, reached = self._run_line(
            self.issue138_line + "\n" + self.runtime_line,
            env_extra={"DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT": "1"},
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(
            invocations,
            [
                "hotfix-vllm-issue138-responses-history.py",
                "hotfix-dsv4-vision-exp.py",
                "hotfix-vllm-empty-encoder-output.py",
                "hotfix-dsv4-issue27-partial-prefill-concurrency.py",
                "hotfix-dsv4-adaptive-prefill-chunk.py",
                "hotfix-dsv4-replicate-markov-head.py",
                "hotfix-dsv4-issue43-decode-fairness-and-diag.py",
                "hotfix-dsv4-issue26-hybrid-swa-min.py",
                "hotfix-dsv4-issue133-triton-specialization.py",
                "hotfix-dsv4-suppress-stops-in-reasoning.py",
            ],
        )
        self.assertTrue(reached)

    def test_issue138_enabled_failure_blocks_later_patchers_and_service_exec(self):
        proc, invocations, reached = self._run_line(
            self.issue138_line + "\n" + self.runtime_line,
            fail_step="hotfix-vllm-issue138-responses-history.py",
            env_extra={"DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT": "1"},
        )
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertEqual(invocations, ["hotfix-vllm-issue138-responses-history.py"])
        self.assertFalse(reached)

    def test_codex_agent_message_unset_zero_and_non_one_do_not_invoke_patcher(self):
        self.assertTrue(self.codex_agent_line, "Codex agent_message gate is missing")
        for value in (None, "0", "true", "2"):
            with self.subTest(value=value):
                extra = (
                    {}
                    if value is None
                    else {"DSPARK_ENABLE_CODEX_AGENT_MESSAGE_COMPAT": value}
                )
                proc, invocations, reached = self._run_line(
                    self.codex_agent_line, env_extra=extra
                )
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                self.assertEqual(invocations, [])
                self.assertTrue(reached)

    def test_issue138_and_codex_agent_message_run_in_compatible_order(self):
        self.assertTrue(self.codex_agent_line, "Codex agent_message gate is missing")
        proc, invocations, reached = self._run_line(
            self.issue138_line + "\n" + self.codex_agent_line,
            env_extra={
                "DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT": "1",
                "DSPARK_ENABLE_CODEX_AGENT_MESSAGE_COMPAT": "1",
            },
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(
            invocations,
            [
                "hotfix-vllm-issue138-responses-history.py",
                "hotfix-vllm-codex-agent-message.py",
            ],
        )
        self.assertTrue(reached)

    def test_codex_agent_message_failure_blocks_service_exec(self):
        self.assertTrue(self.codex_agent_line, "Codex agent_message gate is missing")
        proc, invocations, reached = self._run_line(
            self.codex_agent_line,
            fail_step="hotfix-vllm-codex-agent-message.py",
            env_extra={"DSPARK_ENABLE_CODEX_AGENT_MESSAGE_COMPAT": "1"},
        )
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertEqual(invocations, ["hotfix-vllm-codex-agent-message.py"])
        self.assertFalse(reached)

    def test_codex_launcher_enabled_missing_source_exits_before_side_effect(self):
        self.assertTrue(
            self.codex_launcher_preflight,
            "Codex agent_message launcher pre-flight is missing",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reached = root / "side-effect"
            command = self.codex_launcher_preflight + f"\nprintf reached > '{reached}'\n"
            proc = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "SCRIPT_DIR": str(root),
                    "DSPARK_ENABLE_CODEX_AGENT_MESSAGE_COMPAT": "1",
                    "DSPARK_CODEX_AGENT_MESSAGE_HOTFIX": "missing.py",
                },
                timeout=30,
            )
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            self.assertIn("patcher is missing", proc.stderr)
            self.assertFalse(reached.exists())

    def test_codex_launcher_normalizes_flag_and_resolves_relative_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            patcher = root / "selected.py"
            patcher.write_text("# fixture\n")
            result = root / "result"
            command = (
                self.codex_launcher_preflight
                + "\nprintf '%s\\n%s\\n' "
                + '"$DSPARK_ENABLE_CODEX_AGENT_MESSAGE_COMPAT" '
                + '"$DSPARK_CODEX_AGENT_MESSAGE_HOTFIX"'
                + f" > '{result}'\n"
            )
            proc = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "SCRIPT_DIR": str(root),
                    "DSPARK_ENABLE_CODEX_AGENT_MESSAGE_COMPAT": "true",
                    "DSPARK_CODEX_AGENT_MESSAGE_HOTFIX": "selected.py",
                },
                timeout=30,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertEqual(result.read_text().splitlines(), ["0", str(patcher)])

    def test_launcher_enabled_missing_source_exits_before_side_effect(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reached = root / "side-effect"
            command = (
                self.launcher_preflight
                + f"\nprintf reached > '{reached}'\n"
            )
            proc = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "SCRIPT_DIR": str(root),
                    "DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT": "1",
                    "DSPARK_ISSUE138_HOTFIX": "missing.py",
                },
                timeout=30,
            )
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            self.assertIn("patcher is missing", proc.stderr)
            self.assertFalse(reached.exists())

    def test_launcher_normalizes_flag_resolves_relative_path_and_reaches_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            patcher = root / "selected.py"
            patcher.write_text("# fixture\n")
            result = root / "result"
            command = (
                self.launcher_preflight
                + "\nprintf '%s\\n%s\\n' "
                + '"$DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT" '
                + '"$DSPARK_ISSUE138_HOTFIX"'
                + f" > '{result}'\n"
            )
            proc = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "SCRIPT_DIR": str(root),
                    "DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT": "true",
                    "DSPARK_ISSUE138_HOTFIX": "selected.py",
                },
                timeout=30,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertEqual(result.read_text().splitlines(), ["0", str(patcher)])

    def test_launcher_carries_normalized_flag_and_canonical_path_to_both_worker_commands(self):
        source = START.read_text()
        worker_env = (
            "DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT="
            "'$DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT' "
            "DSPARK_ISSUE138_HOTFIX='./patches/"
            "hotfix-vllm-issue138-responses-history.py'"
        )
        self.assertEqual(source.count(worker_env), 4)
        self.assertIn(
            'scp "$DSPARK_ISSUE138_HOTFIX" "${WORKER_HOST}:'
            '${REMOTE_WORKER_DIR}/patches/'
            'hotfix-vllm-issue138-responses-history.py"',
            source,
        )
        codex_worker_env = (
            "DSPARK_ENABLE_CODEX_AGENT_MESSAGE_COMPAT="
            "'$DSPARK_ENABLE_CODEX_AGENT_MESSAGE_COMPAT' "
            "DSPARK_CODEX_AGENT_MESSAGE_HOTFIX='./patches/"
            "hotfix-vllm-codex-agent-message.py'"
        )
        self.assertEqual(source.count(codex_worker_env), 4)
        self.assertIn(
            'scp "$DSPARK_CODEX_AGENT_MESSAGE_HOTFIX" "${WORKER_HOST}:'
            '${REMOTE_WORKER_DIR}/patches/hotfix-vllm-codex-agent-message.py"',
            source,
        )
        self.assertIn(
            "NODE_RANK=2 HEADLESS=1 $WORKER2_HF_COMPOSE_ENV",
            source,
        )

    def test_runtime_line_runs_enabled_steps_in_order(self):
        proc, invocations, reached = self._run_line(self.runtime_line)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(
            invocations,
            [
                "hotfix-dsv4-vision-exp.py",
                "hotfix-vllm-empty-encoder-output.py",
                "hotfix-dsv4-issue27-partial-prefill-concurrency.py",
                "hotfix-dsv4-adaptive-prefill-chunk.py",
                "hotfix-dsv4-replicate-markov-head.py",
                "hotfix-dsv4-issue43-decode-fairness-and-diag.py",
                "hotfix-dsv4-issue26-hybrid-swa-min.py",
                "hotfix-dsv4-issue133-triton-specialization.py",
                "hotfix-dsv4-suppress-stops-in-reasoning.py",
            ],
        )
        self.assertTrue(reached)

    def test_each_runtime_patcher_failure_blocks_later_steps_and_service_exec(self):
        order = [
            "hotfix-dsv4-vision-exp.py",
            "hotfix-vllm-empty-encoder-output.py",
            "hotfix-dsv4-issue27-partial-prefill-concurrency.py",
            "hotfix-dsv4-adaptive-prefill-chunk.py",
            "hotfix-dsv4-replicate-markov-head.py",
            "hotfix-dsv4-issue43-decode-fairness-and-diag.py",
            "hotfix-dsv4-issue26-hybrid-swa-min.py",
            "hotfix-dsv4-issue133-triton-specialization.py",
            "hotfix-dsv4-suppress-stops-in-reasoning.py",
        ]
        for step in order:
            with self.subTest(step=step):
                proc, invocations, reached = self._run_line(
                    self.runtime_line,
                    fail_step=step,
                )
                self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
                self.assertEqual(invocations, order[: order.index(step) + 1])
                self.assertFalse(reached)

    def test_issue141_unset_zero_and_nonone_are_byte_neutral(self):
        expected = [
            "hotfix-dsv4-vision-exp.py",
            "hotfix-vllm-empty-encoder-output.py",
            "hotfix-dsv4-issue27-partial-prefill-concurrency.py",
            "hotfix-dsv4-adaptive-prefill-chunk.py",
            "hotfix-dsv4-replicate-markov-head.py",
            "hotfix-dsv4-issue43-decode-fairness-and-diag.py",
            "hotfix-dsv4-issue26-hybrid-swa-min.py",
            "hotfix-dsv4-issue133-triton-specialization.py",
            "hotfix-dsv4-suppress-stops-in-reasoning.py",
        ]
        for value in (None, "0", "2", "true"):
            with self.subTest(value=value):
                env_extra = {}
                if value is not None:
                    env_extra["DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK"] = value
                proc, invocations, reached = self._run_line(
                    self.runtime_line,
                    env_extra=env_extra,
                )
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                self.assertEqual(invocations, expected)
                self.assertNotIn(
                    "hotfix-dsv4-issue141-sparse-mla-decode-chunk.py",
                    invocations,
                )
                self.assertTrue(reached)

    def test_issue141_exact_one_runs_first_then_service_chain(self):
        proc, invocations, reached = self._run_line(
            self.runtime_line,
            env_extra={"DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK": "1"},
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(
            invocations,
            [
                "hotfix-dsv4-vision-exp.py",
                "hotfix-dsv4-issue141-sparse-mla-decode-chunk.py",
                "hotfix-vllm-empty-encoder-output.py",
                "hotfix-dsv4-issue27-partial-prefill-concurrency.py",
                "hotfix-dsv4-adaptive-prefill-chunk.py",
                "hotfix-dsv4-replicate-markov-head.py",
                "hotfix-dsv4-issue43-decode-fairness-and-diag.py",
                "hotfix-dsv4-issue26-hybrid-swa-min.py",
                "hotfix-dsv4-issue133-triton-specialization.py",
                "hotfix-dsv4-suppress-stops-in-reasoning.py",
            ],
        )
        self.assertTrue(reached)

    def test_issue141_enabled_failure_blocks_every_later_step_and_service_exec(self):
        proc, invocations, reached = self._run_line(
            self.runtime_line,
            fail_step="hotfix-dsv4-issue141-sparse-mla-decode-chunk.py",
            env_extra={"DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK": "1"},
        )
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertEqual(
            invocations,
            [
                "hotfix-dsv4-vision-exp.py",
                "hotfix-dsv4-issue141-sparse-mla-decode-chunk.py",
            ],
        )
        self.assertFalse(reached)

    def test_easy_knobs_default_off_skip_adaptive_and_markov_patchers(self):
        for adaptive, markov in ((None, None), ("0", "0"), ("2", "true")):
            with self.subTest(adaptive=adaptive, markov=markov):
                proc, invocations, reached = self._run_line(
                    self.runtime_line,
                    env_extra={
                        "DSPARK_ENABLE_ADAPTIVE_CHUNK": adaptive,
                        "DSPARK_ENABLE_REPLICATE_MARKOV": markov,
                    },
                )
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                self.assertEqual(
                    invocations,
                    [
                        "hotfix-dsv4-vision-exp.py",
                        "hotfix-vllm-empty-encoder-output.py",
                        "hotfix-dsv4-issue27-partial-prefill-concurrency.py",
                        "hotfix-dsv4-issue43-decode-fairness-and-diag.py",
                        "hotfix-dsv4-issue26-hybrid-swa-min.py",
                        "hotfix-dsv4-issue133-triton-specialization.py",
                        "hotfix-dsv4-suppress-stops-in-reasoning.py",
                    ],
                )
                self.assertTrue(reached)

    def test_suppress_stops_skip_switch_keeps_other_patchers_fail_closed(self):
        proc, invocations, reached = self._run_line(
            self.runtime_line,
            env_extra={"DSPARK_SKIP_SUPPRESS_STOPS_HOTFIX": "1"},
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(
            invocations,
            [
                "hotfix-dsv4-vision-exp.py",
                "hotfix-vllm-empty-encoder-output.py",
                "hotfix-dsv4-issue27-partial-prefill-concurrency.py",
                "hotfix-dsv4-adaptive-prefill-chunk.py",
                "hotfix-dsv4-replicate-markov-head.py",
                "hotfix-dsv4-issue43-decode-fairness-and-diag.py",
                "hotfix-dsv4-issue26-hybrid-swa-min.py",
                "hotfix-dsv4-issue133-triton-specialization.py",
            ],
        )
        self.assertTrue(reached)

    def test_issue136_default_off_and_nonone_values_skip_patcher(self):
        for value in ("", "0", "true", "01"):
            with self.subTest(value=value):
                proc, invocations, reached = self._run_line(
                    self.issue136_line,
                    env_extra={"DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX": value},
                )
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                self.assertEqual(invocations, [])
                self.assertTrue(reached)

    def test_issue136_enabled_success_reaches_service_exec(self):
        proc, invocations, reached = self._run_line(
            self.issue136_line,
            env_extra={"DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX": "1"},
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(
            invocations, ["hotfix-vllm-issue136-xgrammar-termination.py"]
        )
        self.assertTrue(reached)

    def test_issue136_enabled_failure_blocks_service_exec(self):
        proc, invocations, reached = self._run_line(
            self.issue136_line,
            fail_step="hotfix-vllm-issue136-xgrammar-termination.py",
            env_extra={"DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX": "1"},
        )
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertEqual(
            invocations, ["hotfix-vllm-issue136-xgrammar-termination.py"]
        )
        self.assertFalse(reached)


if __name__ == "__main__":
    unittest.main()
