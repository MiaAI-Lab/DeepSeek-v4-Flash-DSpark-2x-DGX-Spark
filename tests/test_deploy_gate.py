from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deployments/private-smoke"


class DeployGateTest(unittest.TestCase):
    def test_preflight_is_fail_closed_and_checks_all_live_gates(self):
        text = (DEPLOY / "scripts/preflight.sh").read_text()
        for required in (
            "--all", "validate-dspark-config.sh", "verify-fabric.sh",
            "verify-artifact-manifest.py", "docker image inspect", "MemAvailable",
            "df -", "VLLM_ORIGIN_KEY_FILE", "ss -ltn", "WORKER_HOST",
        ):
            self.assertIn(required, text)

    def test_deploy_never_restarts_qwen_and_cleans_both_ranks_on_failure(self):
        text = (DEPLOY / "scripts/deploy-dspark.sh").read_text()
        for required in (
            "--direct-gate", "inventory-qwen.sh", "stop-qwen.sh",
            "start-deepseek-v4-flash-dspark.sh", "smoke-openai-compat.py",
            "benchmark.py", "collect-node-evidence.sh", "cleanup_failed_deploy",
            "--resume-direct", "stop-qwen.sh\" --verify-only", "verify-report",
        ):
            self.assertIn(required, text)
        self.assertNotIn("docker start urbanplan-qwen", text)
        self.assertNotIn("compose start qwen", text)

    def test_reasoning_smoke_uses_deepseek_v4_chat_template_contract(self):
        text = (ROOT / "scripts/smoke-openai-compat.py").read_text()
        self.assertIn(
            '"chat_template_kwargs": {"thinking": True, "reasoning_effort": "low"}',
            text,
        )

    def test_prepare_transfers_one_exact_nccl_image_to_worker(self):
        text = (DEPLOY / "scripts/deploy-dspark.sh").read_text()
        self.assertEqual(text.count('docker build -t "$nccl_image"'), 1)
        self.assertIn('docker save "$nccl_image"', text)
        self.assertIn('ssh -o BatchMode=yes "$WORKER_HOST" docker load', text)
        self.assertIn('if [ "$head_nccl" != "$worker_nccl" ]', text)
        self.assertIn('NCCL test image IDs differ between ranks.', text)

    def test_benchmark_enforces_workload_timing_usage_and_correlation(self):
        text = (DEPLOY / "scripts/benchmark.py").read_text()
        for required in (
            "--warmups", "--samples", "--concurrency", "512",
            "temperature", "0.6", "top_p", "0.95", "stream_options",
            "completion_tokens", "first_token", "last_token", "finish_reason",
            "median", "p95", "50.0", "5.0", "X-Request-Id",
            "request_success_total", "metric_delta", 'delta.get("reasoning")',
            'delta.get("reasoning_content")',
            '"chat_template_kwargs": {"thinking": False}',
        ):
            self.assertIn(required, text)

    def test_node_evidence_is_allowlisted_and_secret_free(self):
        schema = (DEPLOY / "schemas/node-evidence.schema.json").read_text()
        collector = (DEPLOY / "scripts/collect-node-evidence.sh").read_text()
        self.assertIn('"additionalProperties": false', schema)
        self.assertIn("node-evidence", collector)
        for forbidden in ("VLLM_API_KEY", "origin.key", "Authorization"):
            self.assertNotIn(forbidden, collector)


if __name__ == "__main__":
    unittest.main()
