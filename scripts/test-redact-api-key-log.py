#!/usr/bin/env python3
"""Behavioral CPU tests for the vLLM startup API-key log redaction patch.

The suite materializes the pinned api_utils.py anchor under a temporary
VLLM_ROOT and invokes the real shell patch. No GPU, container, or third-party
Python package is required.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches" / "hotfix-vllm-redact-api-key-log.sh"
RELATIVE_API_UTILS = Path("entrypoints/serve/utils/api_utils.py")
RAW_LOG_BODY = '''    non_default_args = get_non_default_args(args)
    logger.info("non-default args: %s", non_default_args)'''
FIXTURE = '''from argparse import Namespace
from typing import Any

EngineArgs = Namespace


class _Logger:
    def info(self, *args: Any) -> None:
        pass


logger = _Logger()


def get_non_default_args(args: Namespace | EngineArgs) -> dict[str, Any]:
    return {}


def log_non_default_args(args: Namespace | EngineArgs):
    non_default_args = get_non_default_args(args)
    logger.info("non-default args: %s", non_default_args)
'''


def materialize(root: Path, text: str = FIXTURE) -> Path:
    target = root / RELATIVE_API_UTILS
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def run_patch(root: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["VLLM_ROOT"] = str(root)
    return subprocess.run(
        ["bash", str(PATCH), *args],
        capture_output=True,
        text=True,
        env=env,
    )


class RedactApiKeyLogPatch(unittest.TestCase):
    def apply_fixture(self, root: Path) -> tuple[Path, subprocess.CompletedProcess]:
        target = materialize(root)
        result = run_patch(root)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        return target, result

    def test_pristine_apply_then_status_is_fully_applied(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target, applied = self.apply_fixture(root)
            text = target.read_text(encoding="utf-8")
            self.assertIn("applied=1", applied.stdout)
            self.assertNotIn(RAW_LOG_BODY, text)
            self.assertIn("_dspark_redact_non_default", text)

            status = run_patch(root, "--status")
            self.assertEqual(status.returncode, 0, status.stderr + status.stdout)
            self.assertNotIn("NOT APPLIED", status.stdout)

    def test_reapply_skips_and_is_byte_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target, _ = self.apply_fixture(root)
            before = target.read_bytes()
            reapplied = run_patch(root)
            self.assertEqual(reapplied.returncode, 0, reapplied.stderr + reapplied.stdout)
            self.assertIn("[skip]", reapplied.stdout)
            self.assertIn("skipped=1", reapplied.stdout)
            self.assertEqual(target.read_bytes(), before)

    def test_drifted_anchor_fails_without_modifying_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = materialize(
                root,
                FIXTURE.replace(
                    'logger.info("non-default args: %s", non_default_args)',
                    'logger.warning("non-default args: %s", non_default_args)',
                ),
            )
            before = target.read_bytes()
            applied = run_patch(root)
            self.assertEqual(applied.returncode, 1, applied.stderr + applied.stdout)
            self.assertIn("FAILED: applied=0 skipped=0 errors=1", applied.stdout)
            self.assertEqual(target.read_bytes(), before)
            self.assertEqual(run_patch(root, "--status").returncode, 1)

    def test_missing_api_utils_and_missing_root_fail(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td, "vllm")
            root.mkdir()
            missing_file = run_patch(root)
            self.assertEqual(missing_file.returncode, 1)
            self.assertIn("File not found", missing_file.stdout + missing_file.stderr)

            missing_root = run_patch(Path(td, "absent"))
            self.assertEqual(missing_root.returncode, 1)
            self.assertIn("vLLM not found", missing_root.stdout + missing_root.stderr)

    def test_partial_patch_is_rejected_and_status_reports_raw_logger(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target, _ = self.apply_fixture(root)
            partial = target.read_text(encoding="utf-8") + '''


def leaked_again(args):
    non_default_args = get_non_default_args(args)
    logger.info("non-default args: %s", non_default_args)
'''
            target.write_text(partial, encoding="utf-8")

            applied = run_patch(root)
            self.assertEqual(applied.returncode, 1, applied.stderr + applied.stdout)
            self.assertIn(
                "partial patch: replacement present but raw logger still defined",
                applied.stdout,
            )
            self.assertNotIn("[skip]", applied.stdout)
            self.assertIn("FAILED: applied=0 skipped=0 errors=1", applied.stdout)

            status = run_patch(root, "--status")
            self.assertEqual(status.returncode, 1)
            self.assertRegex(
                status.stdout,
                r"raw dict no longer logged\s+: NOT APPLIED",
            )

    def test_unpatched_status_is_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            materialize(root)
            status = run_patch(root, "--status")
            self.assertEqual(status.returncode, 1)
            self.assertIn("NOT APPLIED", status.stdout)

    def test_sabotaged_redactor_fails_behavioural_status(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target, _ = self.apply_fixture(root)
            text = target.read_text(encoding="utf-8")
            rebind = '            non_default_args[key] = [f"<redacted:{len(value)} value(s)>"]\n'
            self.assertIn(rebind, text)
            target.write_text(text.replace(rebind, "", 1), encoding="utf-8")

            status = run_patch(root, "--status")
            self.assertEqual(status.returncode, 1)
            self.assertRegex(
                status.stdout,
                r"redaction verified behaviourally\s+: NOT APPLIED",
            )


if __name__ == "__main__":
    unittest.main()
