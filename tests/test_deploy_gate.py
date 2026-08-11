from pathlib import Path
import json
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

    def test_all_advertised_reasoning_modes_use_effective_template_controls(self):
        compose = (ROOT / "docker-compose.dspark.yml").read_text()
        expected = {
            "off": '{"reasoning_effort":"none","drop_thinking":false}',
            "low": '{"thinking":true,"reasoning_effort":"low","drop_thinking":false}',
            "high": '{"thinking":true,"reasoning_effort":"high","drop_thinking":false}',
            "max": '{"thinking":true,"reasoning_effort":"max","drop_thinking":false}',
        }
        for mode, kwargs in expected.items():
            with self.subTest(mode=mode):
                self.assertIn(
                    f"{mode}) DEFAULT_CHAT_TEMPLATE_KWARGS='{kwargs}'",
                    compose,
                )
        # Characterize the audit finding: thinking=false is accepted but is not
        # an effective off switch for the pinned encoder.
        self.assertNotIn('DEFAULT_CHAT_TEMPLATE_KWARGS=\'{"thinking":false}\'', compose)

        pi_model = json.loads((ROOT / "pi-models.dspark.example.json").read_text())
        model = pi_model["providers"]["local-dspark"]["models"][0]
        self.assertEqual(
            {level for level, value in model["thinkingLevelMap"].items() if value is not None},
            {"off", "low", "high", "max"},
        )
        self.assertEqual(model["thinkingLevelMap"]["off"], "none")
        kwargs = model["compat"]["chatTemplateKwargs"]
        self.assertTrue(kwargs["thinking"]["omitWhenOff"])
        self.assertNotIn("omitWhenOff", kwargs["reasoning_effort"])
        self.assertFalse(kwargs["drop_thinking"])

    def test_private_litellm_smoke_skips_origin_only_routes(self):
        smoke = (ROOT / "scripts/smoke-openai-compat.py").read_text()
        self.assertIn('profile in {"direct", "direct-origin"}', smoke)
        self.assertIn("not public LiteLLM routes", smoke)

    def test_semantic_smoke_enumerates_modes_and_history_field_variants(self):
        text = (ROOT / "scripts/smoke-openai-compat.py").read_text()
        for required in (
            'THINKING_MODE_KWARGS', '"off": {"reasoning_effort": "none"',
            '"low": {"thinking": True, "reasoning_effort": "low"',
            '"high": {"thinking": True, "reasoning_effort": "high"',
            '"max": {"thinking": True, "reasoning_effort": "max"',
            'HISTORY_REASONING_FIELDS = ("reasoning", "reasoning_content")',
            'LIVE_HISTORY_REASONING_FIELDS = ("reasoning",)',
            '"drop_thinking": False', 'return_token_strs',
            "run_tool_history_render", "run_multiturn_tool",
            '"tool_call_id": first_call.get',
        ):
            self.assertIn(required, text)

    def test_prepare_transfers_one_exact_nccl_image_to_worker(self):
        text = (DEPLOY / "scripts/deploy-dspark.sh").read_text()
        self.assertEqual(text.count('docker build -t "$nccl_image"'), 1)
        self.assertIn('docker save "$nccl_image"', text)
        self.assertIn('ssh -o BatchMode=yes "$WORKER_HOST" docker load', text)
        self.assertIn('if [ "$head_nccl" != "$worker_nccl" ]', text)
        self.assertIn('NCCL test image IDs differ between ranks.', text)
        self.assertEqual(text.count('build-anemll-runtime-hotfixes.sh"'), 1)
        self.assertIn('docker load <"$runtime_archive"', text)
        self.assertIn('Runtime hotfix image IDs differ between ranks.', text)
        self.assertIn('scripts/verify-runtime-hotfixes.py', text)

    def test_benchmark_enforces_workload_timing_usage_and_correlation(self):
        text = (DEPLOY / "scripts/benchmark.py").read_text()
        for required in (
            "--warmups", "--samples", "--concurrency", "512",
            "temperature", "0.6", "top_p", "0.95", "stream_options",
            "completion_tokens", "first_token", "last_token", "finish_reason",
            "median", "p95", "50.0", "5.0", "X-Request-Id",
            "request_success_total", "metric_delta", 'delta.get("reasoning")',
            'delta.get("reasoning_content")',
            '"chat_template_kwargs": {"thinking": True, "reasoning_effort": "low"}',
            "Return exactly 128 numbered",
        ):
            self.assertIn(required, text)

    def test_stream_path_diagnostic_pairs_identical_payloads_and_alternates_order(self):
        text = (DEPLOY / "scripts/diagnose-stream-path.py").read_text()
        for required in (
            "--direct-base-url", "--proxy-base-url", "deepcopy(payload)",
            "list(reversed(paths))", '"paired_payloads": True',
            '"proxy_is_primary_bottleneck"', 'delta.get("reasoning")',
            'delta.get("reasoning_content")', '"seed": seed',
            'choices=("off", "low")', '"deterministic_paired_seed": True',
            '"paired_samples": paired_comparisons', "args.samples % 2",
        ):
            self.assertIn(required, text)

    def test_deploy_uses_bounded_semantic_status_before_full_lifecycle_smoke(self):
        deploy = (DEPLOY / "scripts/deploy-dspark.sh").read_text()
        start_at = deploy.index("start-deepseek-v4-flash-dspark.sh")
        semantic_at = deploy.index("status-deepseek-v4-flash-dspark.sh", start_at)
        full_smoke_at = deploy.index("smoke-openai-compat.py", semantic_at)
        self.assertLess(semantic_at, full_smoke_at)
        self.assertIn("--semantic", deploy[semantic_at:full_smoke_at])

    def test_node_evidence_is_allowlisted_and_secret_free(self):
        schema = (DEPLOY / "schemas/node-evidence.schema.json").read_text()
        collector = (DEPLOY / "scripts/collect-node-evidence.sh").read_text()
        self.assertIn('"additionalProperties": false', schema)
        self.assertIn("node-evidence", collector)
        self.assertIn("memory_psi_full_avg10=%s", collector)
        self.assertIn("/proc/pressure/memory", collector)
        for forbidden in ("VLLM_API_KEY", "origin.key", "Authorization"):
            self.assertNotIn(forbidden, collector)


if __name__ == "__main__":
    unittest.main()
