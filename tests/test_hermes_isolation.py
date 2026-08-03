import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
HERMES = ROOT / "deployments/private-smoke/hermes"
SCHEMA = ROOT / "deployments/private-smoke/schemas/hermes-result.schema.json"


class HermesIsolationTest(unittest.TestCase):
    def test_config_has_one_provider_one_model_and_no_fallback(self):
        text = (HERMES / "config.yaml").read_text()
        self.assertEqual(text.count("deepseek-smoke:"), 1)
        self.assertEqual(text.count("deepseek-v4-flash-0731-smoke:"), 1)
        for required in (
            "provider: custom:deepseek-smoke",
            "default: deepseek-v4-flash-0731-smoke",
            "key_env: DEEPSEEK_SMOKE_API_KEY",
            "transport: chat_completions",
            "discover_models: false",
            "fallback_providers: []",
            "cli: [terminal]",
            'backend: "docker"',
            'cwd: "/workspace"',
            'home_mode: "profile"',
            "container_persistent: false",
            "docker_persist_across_processes: false",
            "docker_volumes: []",
            "docker_mount_cwd_to_workspace: false",
            "docker_forward_env: []",
            "docker_network: false",
            "python:3.12-alpine@sha256:6d43704baacd1bfbe7c295d7f13079d5d8104ed33568873133f8fc69980419df",
        ):
            self.assertIn(required, text)
        for forbidden in (
            "telegram:", "slack:", "gmail", "linear", "mcp_servers",
            "docker.sock", "docker_extra_args", "api_key:",
        ):
            self.assertNotIn(forbidden, text.lower())

    def test_profile_creation_uses_external_private_home_and_key_file(self):
        text = (HERMES / "create-profile.sh").read_text()
        for required in (
            "HERMES_HOME", "chmod 0700", "chmod 0600", ".env",
            "DEEPSEEK_SMOKE_API_KEY", "--verify-only", "config.yaml",
        ):
            self.assertIn(required, text)
        for forbidden in ("profile use", "active_profile", "~/.hermes/profiles"):
            self.assertNotIn(forbidden, text)

    def test_suite_pins_invocation_and_dedicated_colima_socket(self):
        text = (HERMES / "run-suite.sh").read_text()
        for required in (
            "DOCKER_HOST", "DOCKER_CONFIG", "colima", "dspark-hermes-smoke", "--runtime docker",
            "HERMES_HOME", "--ignore-rules", "--usage-file", "-z",
            "--provider", "custom:deepseek-smoke", "--model",
            "deepseek-v4-flash-0731-smoke", "--toolsets", "terminal",
            "--repeat", "docker.sock", "--network=none", '"$DOCKER_BIN" inspect',
        ):
            self.assertIn(required, text)
        for forbidden in ("--safe-mode", "--ignore-user-config", "profile use"):
            self.assertNotIn(forbidden, text)

    def test_suite_guards_shared_state_and_cleans_ephemeral_state(self):
        text = (HERMES / "run-suite.sh").read_text()
        for required in (
            "default", "hermesia", "active_profile", "LaunchAgents",
            "readlink", "stat", "sha256", "before", "after", "trap",
            "rm -rf --", "hermes-agent=1", "hermes-profile",
        ):
            self.assertIn(required, text)

    def test_fixtures_are_synthetic_and_cover_positive_and_negative_contracts(self):
        transform = json.loads((HERMES / "fixtures/transform-input.json").read_text())
        contract = json.loads((HERMES / "fixtures/tool-contract.json").read_text())
        self.assertEqual(transform["dataset"], "synthetic-expenses")
        self.assertEqual({row["scope"] for row in transform["rows"]}, {"personal", "plexiz"})
        self.assertEqual(contract["allowed_toolsets"], ["terminal"])
        self.assertEqual(contract["network"], "none")
        self.assertEqual(contract["host_mounts"], [])
        self.assertEqual(contract["fallback_models"], [])
        for surface in ("skills", "mcp", "memory", "gateway", "host_paths", "network"):
            self.assertIn(surface, contract["negative_checks"])

    def test_result_schema_requires_isolation_and_request_evidence(self):
        schema = json.loads(SCHEMA.read_text())
        required = set(schema["required"])
        self.assertTrue({
            "schema_version", "run_id", "accepted", "provider", "model",
            "tasks", "negative_checks", "shared_state", "usage", "request_ids",
            "suite_pin_sha256",
        }.issubset(required))
        self.assertFalse(schema["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
