import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "deployments/private-smoke/scripts"
SCHEMA = ROOT / "deployments/private-smoke/schemas/qwen-manifest.schema.json"


class QwenLifecycleTest(unittest.TestCase):
    def test_inventory_is_sanitized_and_records_stable_metadata(self):
        text = (SCRIPTS / "inventory-qwen.sh").read_text()
        for required in ("--check-only", "qwen_manifest.py", "urbanplan-qwen"):
            self.assertIn(required, text)
        helper = (ROOT / "scripts/qwen_manifest.py").read_text()
        for required in (
            "container_id", "image_id", "config_hash", "restart_policy",
            "filesystem_targets", "litellm_success_30d", "st_dev", "st_ino",
        ):
            self.assertIn(required, helper)
        for forbidden in ("Config.Env", "VLLM_API_KEY", "secret_value"):
            self.assertNotIn(forbidden, helper)
        schema = json.loads(SCHEMA.read_text())
        self.assertIn("filesystem_targets", schema["required"])

    def test_stop_requires_fresh_accepted_gate_and_typed_confirmation(self):
        text = (SCRIPTS / "stop-qwen.sh").read_text()
        for required in (
            "--verify-only", "accepted", "manifest_sha256", "fabric",
            "artifacts", "STOP urbanplan-qwen", "STOP_WAIT_SECONDS",
            "docker compose", "8000",
        ):
            self.assertIn(required, text)
        self.assertNotIn("docker start", text)

    def test_purge_is_hash_bound_inode_safe_and_noninteractive_fail_closed(self):
        text = (SCRIPTS / "purge-qwen.sh").read_text()
        for required in (
            "manifest_sha256", "accepted", "st_dev", "st_ino", "realpath",
            "-t 0", "PURGE QWEN", "DELETE QWEN", "quarantine",
        ):
            self.assertIn(required, text)
        self.assertNotIn("rm -rf", text)
        self.assertNotIn("urbanplan-qwen up", text)

    def test_target_verifier_detects_replaced_inode_and_symlink(self):
        helper = ROOT / "scripts/qwen_manifest.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "service"
            target.mkdir()
            target_stat = target.stat()
            manifest = {
                "schema_version": 1,
                "run_id": "unit-run",
                "created_at": "2026-08-02T12:00:00Z",
                "filesystem_targets": [{
                    "kind": "service_root",
                    "path": str(target),
                    "realpath": str(target.resolve()),
                    "st_dev": target_stat.st_dev,
                    "st_ino": target_stat.st_ino,
                    "st_uid": target_stat.st_uid,
                    "file_type": "directory",
                    "exists": True,
                }],
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            good = subprocess.run(
                ["python3", str(helper), "verify-targets", "--manifest", str(path)],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(good.returncode, 0, good.stderr)
            manifest["filesystem_targets"][0]["st_ino"] += 1
            path.write_text(json.dumps(manifest))
            replaced = subprocess.run(
                ["python3", str(helper), "verify-targets", "--manifest", str(path)],
                text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(replaced.returncode, 0)
            manifest["filesystem_targets"][0]["st_ino"] = target_stat.st_ino
            path.write_text(json.dumps(manifest))
            target.rmdir()
            target.symlink_to(root)
            symlinked = subprocess.run(
                ["python3", str(helper), "verify-targets", "--manifest", str(path)],
                text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(symlinked.returncode, 0)


if __name__ == "__main__":
    unittest.main()
