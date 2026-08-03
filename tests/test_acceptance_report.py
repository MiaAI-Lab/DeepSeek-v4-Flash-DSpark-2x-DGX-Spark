import json
import os
from pathlib import Path
import subprocess
import tempfile
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
        functional = schema["properties"]["functional_runs"]["items"]
        self.assertIn("gateway_attested", functional["required"])
        self.assertIn("sample_error_count", schema["properties"]["soak"]["required"])
        self.assertEqual(schema["properties"]["soak"]["properties"]["sample_error_count"]["maximum"], 3)

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
            "cleanup_failed_acceptance", "cleanup-acceptance.sh",
            "public_gateway_unchanged", "purge_eligible", "sample_error_limit",
            "BatchMode=yes", "ConnectTimeout=3", "ConnectionAttempts=1",
            "benchmark_spend_count", "wait_for_benchmark_spend", "client_request_id",
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

    def test_cleanup_runs_every_teardown_step_and_reports_rank_failure(self):
        cleanup = DEPLOY / "scripts/cleanup-acceptance.sh"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "calls.log"
            commands = {}
            for name, status in (("stop", 1), ("status", 1), ("docker", 0), ("egress", 0)):
                path = root / name
                path.write_text(f"#!/bin/sh\nprintf '%s\\n' \"{name} $*\" >>'{log}'\nexit {status}\n")
                path.chmod(0o755)
                commands[name] = path
            env_file = root / "dspark.env"
            litellm_env = root / "litellm.env"
            compose = root / "compose.yml"
            for path in (env_file, litellm_env, compose):
                path.write_text("\n")
            result = subprocess.run(
                [str(cleanup)],
                env={
                    **os.environ,
                    "ENV_FILE": str(env_file),
                    "LITELLM_ENV_FILE": str(litellm_env),
                    "STOP_DSPARK_BIN": str(commands["stop"]),
                    "STATUS_DSPARK_BIN": str(commands["status"]),
                    "DOCKER_BIN": str(commands["docker"]),
                    "EGRESS_POLICY_BIN": str(commands["egress"]),
                    "LITELLM_COMPOSE_FILE": str(compose),
                },
                text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            calls = log.read_text()
            for expected in ("stop", "status --expect stopped", "docker compose", "egress --remove"):
                self.assertIn(expected, calls)
            self.assertNotIn("qwen", calls.lower())
            self.assertIn("cleanup was incomplete", result.stderr)

    def test_egress_cleanup_has_scoped_noninteractive_override(self):
        policy = (DEPLOY / "litellm/egress-policy.sh").read_text()
        cleanup = (DEPLOY / "scripts/cleanup-acceptance.sh").read_text()
        self.assertIn("DSPARK_EGRESS_NONINTERACTIVE_REMOVE", policy)
        self.assertIn("DSPARK_EGRESS_NONINTERACTIVE_REMOVE=1", cleanup)
        install = policy.split("--install)", 1)[1].split("--check)", 1)[0]
        self.assertNotIn("DSPARK_EGRESS_NONINTERACTIVE_REMOVE", install)


if __name__ == "__main__":
    unittest.main()
