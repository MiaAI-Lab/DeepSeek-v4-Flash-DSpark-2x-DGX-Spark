import json
from datetime import datetime, timezone
import hashlib
import os
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
            "-t 0", "PURGE QWEN $run_id $manifest_hash",
            "DELETE QWEN $run_id $manifest_hash", "quarantine",
            "verify-no-supervisors", "status-deepseek-v4-flash-dspark.sh",
            "litellm/smoke.sh",
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

    def test_stop_and_purge_reject_evidence_before_any_docker_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            report = root / "report.json"
            manifest.write_text("{}")
            report.write_text("{}")
            fake_bin = root / "bin"
            fake_bin.mkdir()
            marker = root / "docker-called"
            python = fake_bin / "python3"
            python.write_text("#!/bin/sh\nexit 9\n")
            python.chmod(0o755)
            docker = fake_bin / "docker"
            docker.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit 0\n")
            docker.chmod(0o755)
            env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}
            for script in ("stop-qwen.sh", "purge-qwen.sh"):
                with self.subTest(script=script):
                    marker.unlink(missing_ok=True)
                    result = subprocess.run(
                        [str(SCRIPTS / script), "--manifest", str(manifest),
                         "--gate-report", str(report)],
                        env=env, text=True, capture_output=True, check=False,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(marker.exists())

    def test_purge_report_rehashes_every_bound_evidence_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "qwen-manifest.json"
            artifact = root / "direct.json"
            manifest.write_text(json.dumps({
                "schema_version": 1,
                "run_id": "unit-run",
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "filesystem_targets": [],
            }))
            artifact.write_text('{"accepted":true}\n')
            artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
            pins = {"model_revision": "b" * 40}
            pin_hash = hashlib.sha256(
                json.dumps(pins, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            previous = "0" * 64
            entry = hashlib.sha256(
                f"{previous}:direct:{artifact_hash}:{pin_hash}".encode()
            ).hexdigest()
            report = root / "accepted.json"
            report.write_text(json.dumps({
                "accepted": True,
                "purge_eligible": True,
                "run_id": "unit-run",
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "pin_set_sha256": pin_hash,
                "chain_head_sha256": entry,
                "pins": pins,
                "gates": {"fabric": True},
                "evidence_chain": [{
                    "name": "direct", "artifact_path": artifact.name,
                    "artifact_sha256": artifact_hash, "previous_sha256": previous,
                    "entry_sha256": entry,
                }],
            }))
            command = [
                "python3", str(ROOT / "scripts/qwen_manifest.py"), "verify-report",
                "--manifest", str(manifest), "--report", str(report),
                "--required-gate", "fabric",
            ]
            good = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(good.returncode, 0, good.stderr)
            artifact.write_text('{"accepted":false}\n')
            tampered = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertNotEqual(tampered.returncode, 0)
            self.assertIn("artifact hash mismatch", tampered.stderr)


if __name__ == "__main__":
    unittest.main()
