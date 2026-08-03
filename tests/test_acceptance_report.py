import json
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deployments/private-smoke"
SANITIZER = DEPLOY / "scripts/sanitize-evidence.py"


class AcceptanceReportTest(unittest.TestCase):
    def run_sanitizer(self, payload):
        return subprocess.run(
            ["python3", str(SANITIZER), "--scan-only"],
            input=json.dumps(payload), text=True, capture_output=True, check=False,
        )

    def test_schema_is_strict_and_requires_release_equivalent_evidence(self):
        schema = json.loads((DEPLOY / "schemas/acceptance.schema.json").read_text())
        self.assertFalse(schema["additionalProperties"])
        self.assertTrue({
            "schema_version", "run_id", "created_at", "accepted",
            "manifest_sha256", "fixture_sha256", "pin_set_sha256",
            "chain_head_sha256", "pins", "gates", "functional_runs",
            "performance", "soak", "evidence_chain", "purge_eligible",
        }.issubset(schema["required"]))

    def test_sanitizer_rejects_credentials_private_addresses_and_host_paths(self):
        planted = (
            {"note": "sk-abcdefghijklmnopqrstuvwxyz012345"},
            {"node": "100.64.10.20"},
            {"path": "/home/plexiz/private/model"},
            {"authorization": "Bearer synthetic-value"},
        )
        for payload in planted:
            with self.subTest(payload=payload):
                result = self.run_sanitizer(payload)
                self.assertNotEqual(result.returncode, 0, result.stdout)
        clean = self.run_sanitizer({"accepted": False, "sha256": "a" * 64, "count": 2})
        self.assertEqual(clean.returncode, 0, clean.stderr)

    def test_suite_fixture_pins_workload_and_full_soak(self):
        fixture = json.loads((DEPLOY / "fixtures/suite.json").read_text())
        self.assertEqual(fixture["schema_version"], 1)
        self.assertEqual(fixture["soak"]["duration_seconds"], 1800)
        self.assertEqual(fixture["soak"]["sample_interval_seconds"], 5)
        self.assertEqual(fixture["soak"]["concurrency"], 1)
        self.assertEqual(fixture["performance"]["samples"], 20)
        self.assertEqual(fixture["performance"]["max_tokens"], 512)
        self.assertEqual(fixture["synthetic_expenses"]["expected_grand_total_centavos"], 374025)

    def test_orchestrator_hash_chains_all_gates_and_fails_closed(self):
        text = (DEPLOY / "run-acceptance.sh").read_text()
        for required in (
            "--validate-fixtures", "--live", "SOAK_DURATION_SECONDS",
            "1800", "SOAK_SAMPLE_INTERVAL_SECONDS", "5", "benchmark-direct.json",
            "benchmark-litellm.json", "hermes", "exactly two", "evidence_chain",
            "previous_sha256", "pin_set_sha256", "sanitize-evidence.py",
            "cleanup_failed_acceptance", "stop-deepseek-v4-flash-dspark.sh",
            "egress-policy.sh", "public_gateway_unchanged", "purge_eligible",
        ):
            self.assertIn(required, text)
        for forbidden in ("docker start urbanplan-qwen", "compose start qwen", "purge-qwen.sh --gate-report"):
            self.assertNotIn(forbidden, text)

    def test_purge_has_nondestructive_verify_mode(self):
        text = (DEPLOY / "scripts/purge-qwen.sh").read_text()
        self.assertIn("--verify-only", text)
        verify_prefix = text.split("if [ \"$VERIFY_ONLY\"", 1)[1]
        self.assertIn("exit 0", verify_prefix)

    def test_fixture_validation_entrypoint_passes_offline(self):
        result = subprocess.run(
            [str(DEPLOY / "run-acceptance.sh"), "--validate-fixtures"],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
