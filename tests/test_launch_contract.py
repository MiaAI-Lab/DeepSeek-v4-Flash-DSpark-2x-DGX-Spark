#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import stat
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "scripts/entrypoint.sh"
LAUNCHER = ROOT / "run-dspark-dual-gb10.sh"


def staged_entrypoint(tmp: Path, audit_exit: int = 0) -> tuple[Path, Path]:
    marker = tmp / "audit-ran"
    audit = tmp / "audit_runtime.py"
    audit.write_text(f"#!/usr/bin/env bash\nprintf ran > {marker}\nexit {audit_exit}\n")
    audit.chmod(audit.stat().st_mode | stat.S_IEXEC)
    staged = tmp / "entrypoint.sh"
    staged.write_text(ENTRYPOINT.read_text().replace(
        "AUDIT_BIN=/usr/local/bin/audit_runtime.py", f"AUDIT_BIN={audit}"
    ))
    staged.chmod(staged.stat().st_mode | stat.S_IEXEC)
    return staged, marker


class LaunchContractTests(unittest.TestCase):
    def test_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(ENTRYPOINT)], check=True)
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)

    def test_legacy_stage_c_knobs_are_lane_scoped(self) -> None:
        text = LAUNCHER.read_text()
        lane_start = text.index('if [[ "${RUNTIME_LANE}" == "legacy-stage-c" ]]')
        lane_end = text.index("\nfi", lane_start)
        native_prefix = text[:lane_start]
        legacy_block = text[lane_start:lane_end]
        self.assertNotIn("VLLM_DSPARK_CONFIDENCE_SCHEDULER", native_prefix)
        self.assertIn("VLLM_DSPARK_CONFIDENCE_SCHEDULER", legacy_block)
        self.assertIn("VLLM_USE_B12X_WO_PROJECTION", legacy_block)
        self.assertIn("Unsupported DSPARK_RUNTIME_LANE", legacy_block)

    def test_audit_then_exact_argv(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            staged, marker = staged_entrypoint(Path(raw))
            expected = ["alpha beta", "gamma"]
            result = subprocess.run([
                staged, sys.executable, "-c",
                "import json,sys; print(json.dumps(sys.argv[1:]))", *expected,
            ], capture_output=True, text=True, check=True)
            self.assertIn('["alpha beta", "gamma"]', result.stdout)
            self.assertTrue(marker.exists())

    def test_audit_failure_blocks_child(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            staged, marker = staged_entrypoint(tmp, audit_exit=23)
            child = tmp / "child-ran"
            result = subprocess.run([staged, "bash", "-c", f"touch {child}"], check=False)
            self.assertEqual(result.returncode, 23)
            self.assertTrue(marker.exists())
            self.assertFalse(child.exists())

    def test_non_nvfp4_cache_is_rejected_before_audit(self) -> None:
        for argv, env_value in ((["true", "--kv-cache-dtype", "fp8"], None), (["true"], "fp8")):
            with self.subTest(argv=argv, env=env_value), tempfile.TemporaryDirectory() as raw:
                staged, marker = staged_entrypoint(Path(raw))
                env = os.environ.copy()
                if env_value is not None:
                    env["KV_CACHE_DTYPE"] = env_value
                result = subprocess.run([staged, *argv], env=env, capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 64)
                self.assertIn("nvfp4_ds_mla", result.stderr)
                self.assertFalse(marker.exists())

    def test_zero_args_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            staged, marker = staged_entrypoint(Path(raw))
            result = subprocess.run([staged], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 64)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
