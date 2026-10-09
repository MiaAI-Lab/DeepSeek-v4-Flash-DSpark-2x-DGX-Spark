#!/usr/bin/env python3
"""CPU regressions for model selection in startup probes, warmup and smoke."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "start-deepseek-v4-flash-dspark.sh"
SOURCE = LAUNCHER.read_text(encoding="utf-8")
SELECTION_BEGIN = "# Probe/warmup model selection (begin)."
SELECTION_END = "# Probe/warmup model selection (end)."
SMOKE = ROOT / "smoke-deepseek-v4-flash-dspark.sh"

# Transport stand-in for the smoke CLI: records the request body it was handed
# and answers with the minimal response shape the CLI accepts. Nothing leaves
# the process — no endpoint, key or server is involved.
CURL_STUB = """#!/usr/bin/env bash
set -euo pipefail
body=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -d)
      body="$2"
      shift 2
      ;;
    --max-time|-H)
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
printf '%s' "$body" >"${SMOKE_CAPTURE_DIR:?}/request.$$.$RANDOM.json"
printf '{"choices":[{"message":{"role":"assistant","content":"OK"}}]}'
"""


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


class SmokeRequestModelTest(unittest.TestCase):
    """Run the shipped smoke CLI and read the model it puts in the request.

    The no-env fallback is the lane this repo serves: `.env.dspark.example`
    publishes SERVED_MODEL_NAME=deepseek-v4-flash-vision-exp, so the smoke CLI
    fallback is that contract rather than an arbitrary default.
    """

    def run_smoke(self, served_model: str | None) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            shim = work / "bin"
            shim.mkdir()
            curl = shim / "curl"
            curl.write_text(CURL_STUB, encoding="utf-8")
            curl.chmod(0o755)
            requests = work / "requests"
            requests.mkdir()
            env_file = work / "env.dspark"
            env_file.write_text("", encoding="utf-8")

            env = {
                "PATH": f"{shim}{os.pathsep}{os.environ.get('PATH', os.defpath)}",
                "HOME": os.environ.get("HOME", str(work)),
                "ENV_FILE": str(env_file),
                "CHAT_URL": "http://smoke.invalid/v1/chat/completions",
                "CONCURRENCY": "1",
                "SMOKE_CAPTURE_DIR": str(requests),
            }
            if served_model is not None:
                env["SERVED_MODEL_NAME"] = served_model

            result = subprocess.run(
                ["bash", str(SMOKE)],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            captured = sorted(requests.glob("*.json"))
            self.assertEqual(
                len(captured), 1, "the smoke CLI must send exactly one request"
            )
            return json.loads(captured[0].read_text(encoding="utf-8"))

    def test_unset_env_sends_the_served_vision_exp_alias(self) -> None:
        self.assertEqual(
            self.run_smoke(None)["model"],
            "deepseek-v4-flash-vision-exp",
        )

    def test_explicit_alias_reaches_the_request(self) -> None:
        self.assertEqual(self.run_smoke("alias-a")["model"], "alias-a")

    def test_first_of_several_aliases_reaches_the_request(self) -> None:
        self.assertEqual(self.run_smoke("alias-a alias-b")["model"], "alias-a")


if __name__ == "__main__":
    unittest.main(
        verbosity=0 if "-q" in sys.argv else 2,
        argv=[argument for argument in sys.argv if argument != "-q"],
    )
