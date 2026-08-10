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

    def test_breaking_evidence_contracts_are_versioned_and_v1_is_retained(self):
        latest = json.loads((DEPLOY / "schemas/acceptance.schema.json").read_text())
        historical = json.loads((DEPLOY / "schemas/acceptance-v1.schema.json").read_text())
        node_latest = json.loads((DEPLOY / "schemas/node-evidence.schema.json").read_text())
        node_historical = json.loads((DEPLOY / "schemas/node-evidence-v1.schema.json").read_text())
        self.assertEqual(latest["properties"]["schema_version"]["const"], 2)
        self.assertEqual(historical["properties"]["schema_version"]["const"], 1)
        self.assertEqual(node_latest["properties"]["schema_version"]["const"], 2)
        self.assertEqual(node_historical["properties"]["schema_version"]["const"], 1)
        self.assertTrue(latest["$id"].endswith("acceptance-v2.json"))

    def test_schema_is_strict_and_requires_release_equivalent_evidence(self):
        schema = json.loads((DEPLOY / "schemas/acceptance.schema.json").read_text())
        self.assertFalse(schema["additionalProperties"])
        self.assertTrue({
            "schema_version", "run_id", "created_at", "accepted",
            "manifest_sha256", "fixture_sha256", "pin_set_sha256",
            "chain_head_sha256", "pins", "gates", "functional_runs",
            "performance", "semantic_readiness", "soak", "evidence_chain", "purge_eligible",
        }.issubset(schema["required"]))
        functional = schema["properties"]["functional_runs"]["items"]
        self.assertIn("gateway_attested", functional["required"])
        self.assertIn("sample_error_count", schema["properties"]["soak"]["required"])
        self.assertEqual(schema["properties"]["soak"]["properties"]["sample_error_count"]["maximum"], 3)

    def test_soak_waits_for_litellm_spend_counter_to_settle(self):
        text = (DEPLOY / "run-acceptance.sh").read_text()
        self.assertIn("def settled_spend_count():", text)
        self.assertIn("before_gateway = settled_spend_count()", text)
        self.assertIn("spend counter did not settle before soak", text)

    def test_soak_uses_multiple_shared_cache_blocks_before_the_nonce(self):
        text = (DEPLOY / "run-acceptance.sh").read_text()
        self.assertIn("SOAK_SHARED_PREFIX", text)
        self.assertIn(") * 160", text)
        self.assertIn('SOAK_SHARED_PREFIX\n            + f"\\nRequest nonce', text)

    def test_soak_requires_prometheus_metrics_instead_of_defaulting_to_zero(self):
        text = (DEPLOY / "run-acceptance.sh").read_text()
        self.assertIn("required Prometheus metric is missing", text)
        self.assertNotIn("def metric(text, name, default=0.0)", text)

    def test_speculative_acceptance_uses_exact_counter_deltas(self):
        schema = json.loads((DEPLOY / "schemas/acceptance.schema.json").read_text())
        soak = schema["properties"]["soak"]
        required = set(soak["required"])
        self.assertTrue({
            "speculative_accepted_tokens_delta",
            "speculative_draft_tokens_delta",
            "speculative_acceptance_ratio",
            "speculative_acceptance_observation",
        }.issubset(required))
        self.assertNotIn("speculative_acceptance_mean", required)
        self.assertEqual(
            soak["properties"]["speculative_acceptance_observation"]["enum"],
            ["observed", "not-observed"],
        )
        text = (DEPLOY / "run-acceptance.sh").read_text()
        self.assertIn("vllm:spec_decode_num_accepted_tokens_total", text)
        self.assertIn("vllm:spec_decode_num_draft_tokens_total", text)
        self.assertIn("accepted_delta / draft_delta", text)
        self.assertNotIn("speculative_acceptance_mean", text)
        self.assertNotIn("statistics.mean(spec", text)

    def test_capacity_resume_requires_an_accepted_full_soak_and_preserves_attempt(self):
        text = (DEPLOY / "run-acceptance.sh").read_text()
        self.assertIn("--resume-capacity", text)
        self.assertIn("resume requires an accepted full-duration soak", text)
        self.assertIn('previous_attempt="$FULL_CONTEXT_EVIDENCE.attempt-$attempt"', text)
        self.assertIn('mv "$FULL_CONTEXT_EVIDENCE" "$previous_attempt"', text)
        self.assertIn("reuse_full_context=1", text)
        self.assertIn("age > 24 * 3600", text)
        self.assertIn("Missing pre-capacity acceptance evidence", text)
        self.assertIn("Capacity resume requires exactly two Hermes result files", text)

    def test_live_acceptance_requires_full_context_and_scheduler_gates(self):
        text = (DEPLOY / "run-acceptance.sh").read_text()
        self.assertIn('probe-full-context.py"', text)
        self.assertIn('benchmark-scheduler.py"', text)
        self.assertIn('"full-context", full_context_path', text)
        self.assertIn('"scheduler", scheduler_path', text)
        self.assertIn('--required-gate full_context --required-gate scheduler', text)

    def test_direct_and_private_litellm_semantic_evidence_are_distinct(self):
        schema = json.loads((DEPLOY / "schemas/acceptance.schema.json").read_text())
        readiness = schema["properties"]["semantic_readiness"]
        self.assertEqual(
            set(readiness["required"]), {"direct_origin", "private_litellm"}
        )
        text = (DEPLOY / "run-acceptance.sh").read_text()
        for required in (
            "semantic-direct-origin.json", "semantic-private-litellm.json",
            "direct_origin", "private_litellm", "--semantic-canary",
        ):
            self.assertIn(required, text)

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
        self.assertEqual(fixture["soak"]["speculative_metrics"], {
            "accepted": "vllm:spec_decode_num_accepted_tokens_total",
            "draft": "vllm:spec_decode_num_draft_tokens_total",
        })
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
            "benchmark_spend_count", "wait_for_benchmark_spend", "origin_request_id",
            "WHERE t.request_id IN", "origin_response_ids",
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


    def test_capacity_evidence_is_scoped_to_current_container_start(self):
        text = (DEPLOY / "run-acceptance.sh").read_text()
        self.assertIn("{{.State.StartedAt}}", text)
        self.assertIn('["docker", "logs", "--since", started_at', text)
        self.assertIn("reported_token_capacity = min(reported_capacities)", text)

    def test_rollout_evidence_covers_runtime_and_minefield_dimensions(self):
        schema = json.loads((DEPLOY / "schemas/acceptance.schema.json").read_text())
        self.assertIn("rollout_evidence", schema["required"])
        rollout = schema["properties"]["rollout_evidence"]
        self.assertEqual(set(rollout["required"]), {
            "process_readiness", "api_readiness", "semantic_readiness", "kv_cache",
            "rank_participation", "memory", "prefix_cache", "speculative_decode",
            "minefield", "external_gateway", "prompt_reasoning_canaries_absent",
            "message_logging_disabled",
        })
        minefield = rollout["properties"]["minefield"]
        self.assertEqual(set(minefield["required"]), {
            "commit", "executed", "problem", "inconclusive", "unimplemented",
        })
        node = json.loads((DEPLOY / "schemas/node-evidence.schema.json").read_text())
        item = node["properties"]["nodes"]["items"]
        self.assertIn("memory_psi_full_avg10", item["required"])
        self.assertIn("memory_psi_full_avg10", item["properties"])

    def test_minefield_runner_is_pinned_isolated_and_summarizes_exact_counts(self):
        text = (ROOT / "scripts/run-minefield-pinned.py").read_text()
        for required in (
            "2b453b8a69dbaf8dc9d521dc2d6212cdaceb8169", "venv", "pip", "--api-key",
            "sys.stdin.read", "not_implemented_count", '"executed"', '"problem"',
            '"inconclusive"', '"unimplemented"', "os.O_EXCL", "os.fdopen",
            "--setup-timeout", "--doctor-timeout", "subprocess.TimeoutExpired",
        ):
            self.assertIn(required, text)
        self.assertNotIn("key = args.", text)

    def test_acceptance_checks_prompt_reasoning_canary_absence_and_logging_off(self):
        text = (DEPLOY / "run-acceptance.sh").read_text()
        for required in (
            "PROMPT_LOG_CANARY", "REASONING_LOG_CANARY", "check_canary_absence",
            "turn_off_message_logging: true", "minefield.json", "prefix_cache_queries_delta",
            "reported_token_capacity", "memory_psi_full_avg10",
        ):
            self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
