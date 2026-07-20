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
        self.assertIn('if [[ "${RUNTIME_LANE}" == "legacy-stage-c" ]]', text)
        self.assertEqual(text.count("VLLM_DSPARK_CONFIDENCE_SCHEDULER"), 2)
        self.assertEqual(text.count("VLLM_USE_B12X_WO_PROJECTION"), 2)
        self.assertIn("Unsupported DSPARK_RUNTIME_LANE", text)
        self.assertIn("legacy-stage-c requires DSPARK_VLLM_IMAGE", text)
        self.assertIn("native-v025 cannot use the legacy Stage-C image", text)

    def test_native_lane_rejects_legacy_capacity_without_experiment_opt_in(self) -> None:
        env = {
            **os.environ,
            "DSPARK_RUNTIME_LANE": "native-v025",
            "MAX_MODEL_LEN": "1048576",
            "KV_CACHE_DTYPE": "nvfp4_ds_mla",
        }
        result = subprocess.run(
            [LAUNCHER], env=env, capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("qualified 200K/16/16K/0.84/K5", result.stderr)

    def test_legacy_lane_requires_stage_c_image(self) -> None:
        env = {
            **os.environ,
            "DSPARK_RUNTIME_LANE": "legacy-stage-c",
            "DSPARK_VLLM_IMAGE": "dspark-r0b0tlab:production-candidate",
            "MAX_MODEL_LEN": "384000",
            "MAX_NUM_SEQS": "4",
            "MAX_NUM_BATCHED_TOKENS": "8192",
            "GPU_MEMORY_UTILIZATION": "0.88",
            "MTP_NUM_TOKENS": "5",
            "KV_CACHE_DTYPE": "nvfp4_ds_mla",
        }
        result = subprocess.run(
            [LAUNCHER], env=env, capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("legacy-stage-c requires", result.stderr)

    def test_audit_then_exact_argv(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            staged, marker = staged_entrypoint(tmp)
            fake_vllm = tmp / "vllm"
            fake_vllm.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
            fake_vllm.chmod(fake_vllm.stat().st_mode | stat.S_IEXEC)
            expected = ["alpha beta", "gamma"]
            env = {**os.environ, "KV_CACHE_DTYPE": "nvfp4_ds_mla"}
            result = subprocess.run(
                [staged, fake_vllm, "serve", "/model", "--kv-cache-dtype", "nvfp4_ds_mla", *expected],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn('["serve", "/model", "--kv-cache-dtype", "nvfp4_ds_mla", "alpha beta", "gamma"]', result.stdout)
            self.assertTrue(marker.exists())

    def test_launcher_style_quoted_kv_flag_passes_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            staged, marker = staged_entrypoint(Path(raw))
            env = {**os.environ, "KV_CACHE_DTYPE": "nvfp4_ds_mla"}
            command = (
                'VLLM_BIN=/bin/true; exec "${VLLM_BIN}" serve /model '
                '--kv-cache-dtype "nvfp4_ds_mla"'
            )
            result = subprocess.run(
                [staged, "bash", "-lc", command],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(marker.exists())

    def test_audit_failure_blocks_child(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            staged, marker = staged_entrypoint(tmp, audit_exit=23)
            child = tmp / "child-ran"
            env = {**os.environ, "KV_CACHE_DTYPE": "nvfp4_ds_mla"}
            result = subprocess.run(
                [staged, "vllm", "serve", "/model", "--kv-cache-dtype", "nvfp4_ds_mla"],
                env=env,
                check=False,
            )
            self.assertEqual(result.returncode, 23)
            self.assertTrue(marker.exists())
            self.assertFalse(child.exists())

    def test_non_nvfp4_cache_is_rejected_before_audit(self) -> None:
        for argv, env_value in (
            (["vllm", "serve", "/model", "--kv-cache-dtype", "fp8"], "nvfp4_ds_mla"),
            (["vllm", "serve", "/model", "--kv-cache-dtype", "nvfp4_ds_mla"], "fp8"),
        ):
            with self.subTest(argv=argv, env=env_value), tempfile.TemporaryDirectory() as raw:
                staged, marker = staged_entrypoint(Path(raw))
                env = os.environ.copy()
                if env_value is not None:
                    env["KV_CACHE_DTYPE"] = env_value
                result = subprocess.run([staged, *argv], env=env, capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 64)
                self.assertIn("nvfp4_ds_mla", result.stderr)
                self.assertFalse(marker.exists())

    def test_missing_explicit_kv_cache_flag_is_rejected_before_audit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            staged, marker = staged_entrypoint(Path(raw))
            env = {**os.environ, "KV_CACHE_DTYPE": "nvfp4_ds_mla"}
            result = subprocess.run(
                [staged, "vllm", "serve", "/model"],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 64)
            self.assertIn("explicit --kv-cache-dtype", result.stderr)
            self.assertFalse(marker.exists())

    def test_duplicate_kv_cache_flags_are_rejected_before_audit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            staged, marker = staged_entrypoint(Path(raw))
            env = {**os.environ, "KV_CACHE_DTYPE": "nvfp4_ds_mla"}
            result = subprocess.run(
                [
                    staged,
                    "vllm",
                    "serve",
                    "/model",
                    "--kv-cache-dtype",
                    "nvfp4_ds_mla",
                    "--kv-cache-dtype=nvfp4_ds_mla",
                ],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 64)
            self.assertIn("exactly one", result.stderr)
            self.assertFalse(marker.exists())

    def test_unknown_child_command_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            staged, marker = staged_entrypoint(Path(raw))
            env = {**os.environ, "KV_CACHE_DTYPE": "nvfp4_ds_mla"}
            result = subprocess.run(
                [staged, "python", "serve", "--kv-cache-dtype", "nvfp4_ds_mla"],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 64)
            self.assertFalse(marker.exists())

    def test_zero_args_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            staged, marker = staged_entrypoint(Path(raw))
            result = subprocess.run([staged], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 64)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
