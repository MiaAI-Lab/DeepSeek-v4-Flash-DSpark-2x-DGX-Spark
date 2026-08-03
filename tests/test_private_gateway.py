from pathlib import Path
import os
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
GATEWAY = ROOT / "deployments/private-smoke/litellm"


class PrivateGatewayTest(unittest.TestCase):
    def test_compose_pins_images_and_hardens_both_containers(self):
        text = (GATEWAY / "docker-compose.yml").read_text()
        for required in (
            "ghcr.io/berriai/litellm-database@sha256:5fa5f99cd5576e359a0e50395ad14edbe922ef41c152f67c534e4f8b6238c5ec",
            "postgres@sha256:b797483593b82cbea9a7ee41c88f324a90d10d9c2504d40e755d91c75456366d",
            "cap_drop", "ALL", "read_only: true", "no-new-privileges:true",
            "seccomp.json", "tmpfs", "internal: true", "ipv4_address",
            "PRISMA_HOME_DIR", "PRISMA_QUERY_ENGINE_BINARY",
            "prisma-cache:/app/cache/.cache/prisma-python:ro",
            "dspark-private-litellm-prisma-cache", "external: true",
            "${HEAD_TAILSCALE_IP:?set HEAD_TAILSCALE_IP to the head tailnet address}:4001:4001",
        ):
            self.assertIn(required, text)
        for forbidden in ("docker.sock", "network_mode: host", "4000:4000"):
            self.assertNotIn(forbidden, text)

    def test_config_has_one_model_and_no_retries_or_fallbacks(self):
        text = (GATEWAY / "config.yaml").read_text()
        self.assertEqual(text.count("- model_name:"), 1)
        for required in (
            "deepseek-v4-flash-0731-smoke", "openai/deepseek-v4-flash-0731",
            "http://172.30.0.1:8888/v1", "os.environ/VLLM_ORIGIN_KEY",
            "num_retries: 0", "fallbacks: []", "default_fallbacks: []",
            "turn_off_message_logging: true",
        ):
            self.assertIn(required, text)

    def test_entrypoint_and_bootstrap_keep_secret_values_out_of_metadata(self):
        compose = (GATEWAY / "docker-compose.yml").read_text()
        entrypoint = (GATEWAY / "secret-entrypoint.sh").read_text()
        bootstrap = (GATEWAY / "bootstrap-virtual-key.sh").read_text()
        self.assertNotIn("LITELLM_MASTER_KEY=", compose)
        self.assertNotIn("VLLM_ORIGIN_KEY=", compose)
        for required in ("MASTER_KEY_FILE", "ORIGIN_KEY_FILE", "DATABASE_PASSWORD_FILE"):
            self.assertIn(required, entrypoint)
        self.assertIn("/key/generate", bootstrap)
        self.assertIn('"models": ["deepseek-v4-flash-0731-smoke"]', bootstrap)
        self.assertIn("chmod 0600", bootstrap)

    def test_deploy_populates_pinned_prisma_cache_without_network(self):
        text = (GATEWAY / "deploy.sh").read_text()
        for required in (
            "prepare_prisma_cache", "--network none", "--read-only",
            "--cap-drop ALL", "no-new-privileges:true", "--user 1000:1000",
            "/root/.cache/prisma-python", "PRISMA_CACHE_SENTINEL",
            "docker volume rm", "--entrypoint /usr/bin/test",
        ):
            self.assertIn(required, text)

    def test_smoke_proves_key_scope_interfaces_and_egress(self):
        text = (GATEWAY / "smoke.sh").read_text()
        for required in (
            "--all-interfaces", "/key/generate", "/config", "wrong-model",
            "HEAD_TAILSCALE_IP", "127.0.0.1", "4001", "public_catalog",
            "docker.sock", "1.1.1.1", "172.30.0.1", "4000", "no fallback",
        ):
            self.assertIn(required, text)

    def test_egress_policy_builds_rules_under_nounset(self):
        policy = (GATEWAY / "egress-policy.sh").read_text()
        self.assertIn("sudo -n true", policy)
        self.assertIn("--privileged --pid host --entrypoint nsenter", policy)
        self.assertIn("sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8", policy)
        with tempfile.TemporaryDirectory() as temporary:
            sudo = Path(temporary) / "sudo"
            sudo.write_text("#!/bin/sh\nexit 0\n")
            sudo.chmod(0o755)
            environment = dict(os.environ)
            environment["PATH"] = f"{temporary}:{environment['PATH']}"
            result = subprocess.run(
                [str(GATEWAY / "egress-policy.sh"), "--check"],
                env=environment,
                text=True,
                capture_output=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
