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
ENCODING_TOKEN = "python3 /opt/hotfix-encoding-dsv4-issue21.py"
RUNTIME_TOKEN = "python3 /opt/hotfix-dsv4-issue27-partial-prefill-concurrency.py"

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


class PythonHotfixFailClosedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoding_line = _compose_line(ENCODING_TOKEN)
        cls.runtime_line = _compose_line(RUNTIME_TOKEN)

    def _run_line(
        self,
        line: str,
        *,
        fail_step: str | None = None,
        env_extra: dict[str, str] | None = None,
        missing_encoding: bool = False,
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
            command += f'\nprintf x > "{reached}"\n'
            env = dict(os.environ)
            env.update(
                {
                    "PATH": f"{bin_dir}:/usr/bin:/bin",
                    "INVOCATIONS": str(invocations),
                    "ENCODING_SOURCE": str(encoding),
                    "DSPARK_ENABLE_ISSUE31_GPU_HOTFIX": "0",
                    "DSPARK_SKIP_SUPPRESS_STOPS_HOTFIX": "0",
                }
            )
            if fail_step is not None:
                env["FAIL_STEP"] = fail_step
                env["FAIL_CODE"] = "7"
            else:
                env.pop("FAIL_STEP", None)
                env.pop("FAIL_CODE", None)
            if env_extra:
                env.update(env_extra)

            proc = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
            return proc, invocations.read_text().splitlines(), reached.exists()

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

    def test_runtime_line_runs_enabled_steps_in_order(self):
        proc, invocations, reached = self._run_line(self.runtime_line)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(
            invocations,
            [
                "hotfix-vllm-empty-encoder-output.py",
                "hotfix-dsv4-issue27-partial-prefill-concurrency.py",
                "hotfix-dsv4-issue43-decode-fairness-and-diag.py",
                "hotfix-dsv4-issue26-hybrid-swa-min.py",
                "hotfix-dsv4-issue133-triton-specialization.py",
                "hotfix-dsv4-runtime-ablation.py",
                "hotfix-dsv4-suppress-stops-in-reasoning.py",
            ],
        )
        self.assertTrue(reached)

    def test_each_runtime_patcher_failure_blocks_later_steps_and_service_exec(self):
        order = [
            "hotfix-vllm-empty-encoder-output.py",
            "hotfix-dsv4-issue27-partial-prefill-concurrency.py",
            "hotfix-dsv4-issue43-decode-fairness-and-diag.py",
            "hotfix-dsv4-issue26-hybrid-swa-min.py",
            "hotfix-dsv4-issue133-triton-specialization.py",
            "hotfix-dsv4-runtime-ablation.py",
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

    def test_suppress_stops_skip_switch_keeps_other_patchers_fail_closed(self):
        proc, invocations, reached = self._run_line(
            self.runtime_line,
            env_extra={"DSPARK_SKIP_SUPPRESS_STOPS_HOTFIX": "1"},
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(
            invocations,
            [
                "hotfix-vllm-empty-encoder-output.py",
                "hotfix-dsv4-issue27-partial-prefill-concurrency.py",
                "hotfix-dsv4-issue43-decode-fairness-and-diag.py",
                "hotfix-dsv4-issue26-hybrid-swa-min.py",
                "hotfix-dsv4-issue133-triton-specialization.py",
                "hotfix-dsv4-runtime-ablation.py",
            ],
        )
        self.assertTrue(reached)


if __name__ == "__main__":
    unittest.main()
