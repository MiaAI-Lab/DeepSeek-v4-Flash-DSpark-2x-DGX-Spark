import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
HERMES = ROOT / "deployments/private-smoke/hermes"
SCHEMA = ROOT / "deployments/private-smoke/schemas/hermes-result.schema.json"


class HermesIsolationTest(unittest.TestCase):
    def test_shell_entrypoints_parse(self):
        for script in sorted(HERMES.glob("*.sh")):
            with self.subTest(script=script.name):
                subprocess.run(["bash", "-n", str(script)], check=True)

    def test_config_has_one_provider_one_model_and_no_fallback(self):
        text = (HERMES / "config.yaml").read_text()
        self.assertEqual(text.count("deepseek-smoke:"), 1)
        self.assertEqual(text.count("deepseek-v4-flash-0731-smoke:"), 1)
        for required in (
            "provider: custom:deepseek-smoke",
            "default: deepseek-v4-flash-0731-smoke",
            "key_env: DEEPSEEK_SMOKE_API_KEY",
            "transport: chat_completions",
            "extra_body:", "temperature: 0",
            "discover_models: false",
            "fallback_providers: []",
            "cli: [terminal]",
            'backend: "docker"',
            'cwd: "/workspace"',
            'home_mode: "profile"',
            "api_max_retries: 1",
            "container_persistent: false",
            "docker_persist_across_processes: true",
            "docker_volumes: []",
            "docker_mount_cwd_to_workspace: false",
            "docker_forward_env: []",
            "docker_network: false",
            "python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de",
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

    def test_profile_accepts_named_failure_probe_request_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            key = root / "key"
            key.write_text("sk-synthetic\n")
            key.chmod(0o600)
            home = root / "profile"
            environment = dict(os.environ)
            environment["ALLOW_LOOPBACK_PROVIDER"] = "1"
            result = subprocess.run(
                [
                    str(HERMES / "create-profile.sh"),
                    "--home", str(home),
                    "--base-url", "http://127.0.0.1:4001/v1",
                    "--key-file", str(key),
                    "--request-id", "hermes-smoke-deadbeef-invalid",
                ],
                env=environment,
                text=True,
                capture_output=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_suite_pins_invocation_and_dedicated_colima_socket(self):
        text = (HERMES / "run-suite.sh").read_text()
        for required in (
            "DOCKER_HOST", "DOCKER_CONFIG", "colima", "dspark-hermes-smoke", "--runtime docker",
            "HERMES_HOME", "HERMES_STREAM_RETRIES=0", "--ignore-rules", "--usage-file", "-z",
            "HERMES_SMOKE_PROCESS_TIMEOUT", "HERMES_SMOKE_DUMP_REQUESTS", "HERMES_DUMP_REQUESTS",
            "ps -p", "kill -TERM", "kill -KILL", "status=124",
            "started_at=$SECONDS", "sleep 0.1",
            "--provider", "custom:deepseek-smoke", "--model",
            "deepseek-v4-flash-0731-smoke", "--toolsets", "terminal",
            "--repeat", "docker.sock", "--network=none", '"$DOCKER_BIN" inspect',
            "capture_default_container_observation", "recover_tool_evidence",
            "merge_workspace_writer_observation",
            "workspace writer observation lacks a container ID",
            "_dspark_workspace_writer", "at most one rotated default container",
            "summarize_request_diagnostics", "nested_arguments", "content_prefix",
            "observation_candidate", "Configuration is immutable",
        ):
            self.assertIn(required, text)
        for forbidden in ("--safe-mode", "--ignore-user-config", "profile use"):
            self.assertNotIn(forbidden, text)

    def test_failure_probe_excludes_profile_verification_requests(self):
        text = (HERMES / "run-suite.sh").read_text()
        profile_creation = text.index('ALLOW_LOOPBACK_PROVIDER=1 "$CREATE_PROFILE"')
        reset = text.index(': >"$count_file"', profile_creation)
        invocation = text.index('"$HERMES_BIN" -z', reset)
        self.assertLess(profile_creation, reset)
        self.assertLess(reset, invocation)
        self.assertIn('endswith("/chat/completions")', text)

    def test_positive_prompt_requires_one_deterministic_terminal_call(self):
        text = (HERMES / "run-suite.sh").read_text()
        for required in (
            "Your FIRST action MUST be exactly one terminal call",
            "TERMINAL_EVIDENCE_JSON",
            "from decimal import Decimal",
            "probe.connect(('1.1.1.1', 53))",
            "assert not Path(host_path).exists()",
        ):
            self.assertIn(required, text)

    def test_suite_guards_shared_state_and_cleans_ephemeral_state(self):
        text = (HERMES / "run-suite.sh").read_text()
        for required in (
            "default", "hermesia", "active_profile", "LaunchAgents",
            "readlink", "stat", "sha256", "before", "after", "trap",
            "rm -rf --", "hermes-agent=1", "hermes-profile", "prompt-backend-probe",
            "remove_hermes_containers",
        ):
            self.assertIn(required, text)

    def test_fixtures_are_synthetic_and_cover_positive_and_negative_contracts(self):
        transform = json.loads((HERMES / "fixtures/transform-input.json").read_text())
        contract = json.loads((HERMES / "fixtures/tool-contract.json").read_text())
        self.assertEqual(transform["dataset"], "synthetic-expenses")
        self.assertEqual({row["scope"] for row in transform["rows"]}, {"personal", "plexiz"})
        self.assertEqual(contract["allowed_toolsets"], ["terminal"])
        self.assertEqual(contract["network"], "none")
        self.assertIn("default work container", contract["negative_checks"]["network"])
        self.assertIn("metadata probe", contract["negative_checks"]["network"])
        self.assertEqual(contract["host_mounts"], [
            {
                "source_scope": "isolated_profile_cache",
                "destination_prefix": "/root/.hermes/cache/",
                "read_only": True,
            },
            {
                "source_scope": "isolated_empty_skills",
                "destination": "/root/.hermes/skills",
                "read_only": True,
            },
        ])
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
            "tool_evidence_sha256",
        }.issubset(required))
        self.assertFalse(schema["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
