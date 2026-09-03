#!/usr/bin/env python3
"""CPU regressions for model selection in startup probes and warmup."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "start-deepseek-v4-flash-dspark.sh"
SOURCE = LAUNCHER.read_text(encoding="utf-8")
SELECTION_BEGIN = "# Probe/warmup model selection (begin)."
SELECTION_END = "# Probe/warmup model selection (end)."
SMOKE = ROOT / "smoke-deepseek-v4-flash-dspark.sh"
SMOKE_SOURCE = SMOKE.read_text(encoding="utf-8")
SMOKE_SELECTION_BEGIN = "# Smoke model selection (begin)"
SMOKE_SELECTION_END = "# Smoke model selection (end)"


def extract_selection() -> str:
    start = SOURCE.index(SELECTION_BEGIN)
    end = SOURCE.index(SELECTION_END, start) + len(SELECTION_END)
    return SOURCE[start:end]


class SelectionBehaviorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.selection = extract_selection()

    def select(self, value: str | None) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env.pop("SERVED_MODEL_NAME", None)
        if value is not None:
            env["SERVED_MODEL_NAME"] = value
        return subprocess.run(
            [
                "bash",
                "-c",
                "set -euo pipefail\n"
                f"{self.selection}\n"
                "printf '%s\\n' \"$PROBE_MODEL\"\n",
            ],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

    def assert_selection(self, value: str | None, expected: str) -> None:
        result = self.select(value)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, expected + "\n")

    def test_unset_and_empty_use_default(self) -> None:
        for value in (None, "", "   "):
            with self.subTest(value=value):
                self.assert_selection(value, "deepseek-v4-flash-dspark")

    def test_single_alias_is_unchanged(self) -> None:
        self.assert_selection("deepseek-v4-flash-0731", "deepseek-v4-flash-0731")

    def test_multiple_aliases_select_first(self) -> None:
        self.assert_selection(
            "deepseek-v4-flash-0731 deepseek-v4-flash-dspark",
            "deepseek-v4-flash-0731",
        )

    def test_shell_whitespace_is_normalized(self) -> None:
        self.assert_selection("  alias-a\t alias-b  ", "alias-a")


class LauncherWiringTest(unittest.TestCase):
    def test_every_ready_probe_uses_selected_alias(self) -> None:
        start = SOURCE.index(SELECTION_BEGIN)
        ready_block = SOURCE[start : SOURCE.index("    exit 0", start)]
        after_selection = ready_block[ready_block.index(SELECTION_END) :]

        self.assertNotIn("${SERVED_MODEL_NAME", after_selection)
        self.assertEqual(
            ready_block.count("'\"${PROBE_MODEL}\"'"),
            2,
            "both startup chat payloads must use the selected alias",
        )
        self.assertIn(
            '"${CHAT_URL%/v1/chat/completions}" "$PROBE_MODEL"',
            ready_block,
        )

    def test_smoke_uses_selected_alias(self) -> None:
        start = SMOKE_SOURCE.index(SMOKE_SELECTION_BEGIN)
        after_selection = SMOKE_SOURCE[
            SMOKE_SOURCE.index(SMOKE_SELECTION_END, start)
        :]

        self.assertIn(
            'read -r MODEL _ <<< "${SERVED_MODEL_NAME:-deepseek-v4-flash-dspark}"',
            SMOKE_SOURCE[start : SMOKE_SOURCE.index(SMOKE_SELECTION_END, start)],
        )
        self.assertNotIn("${SERVED_MODEL_NAME", after_selection)
        self.assertIn(
            '''-d '{"model":"'"$MODEL"'","messages":''',
            after_selection,
        )


if __name__ == "__main__":
    unittest.main(
        verbosity=0 if "-q" in sys.argv else 2,
        argv=[argument for argument in sys.argv if argument != "-q"],
    )
