import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
REVISION = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
IMAGE = (
    "ghcr.io/anemll/dspark-vllm-gx10@sha256:"
    "a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8"
)


def run(*args, env=None):
    return subprocess.run(
        [str(arg) for arg in args],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


class ArtifactContractTest(unittest.TestCase):
    def write_env(self, directory, *, revision=REVISION, image=IMAGE, key_file=True):
        secret = directory / "origin.key"
        if key_file:
            secret.write_text("test-secret-value\n")
            secret.chmod(stat.S_IRUSR | stat.S_IWUSR)
        elif secret.exists():
            secret.unlink()
        env_file = directory / ".env.dspark"
        env_file.write_text(
            "\n".join(
                [
                    "WORKER_HOST=worker.example",
                    "MASTER_ADDR=10.77.77.1",
                    "MASTER_PORT=25000",
                    "NODE_RANK=0",
                    "HEADLESS=",
                    "NCCL_IB_HCA=rocep1s0f0",
                    "NCCL_SOCKET_IFNAME=enp1s0f0np0",
                    "TP_SOCKET_IFNAME=enp1s0f0np0",
                    "GLOO_SOCKET_IFNAME=enp1s0f0np0",
                    "WORKER_NCCL_IB_HCA=rocep1s0f0",
                    "WORKER_NCCL_SOCKET_IFNAME=enp1s0f0np0",
                    "WORKER_TP_SOCKET_IFNAME=enp1s0f0np0",
                    "WORKER_GLOO_SOCKET_IFNAME=enp1s0f0np0",
                    "VLLM_HOST_IP=10.77.77.1",
                    "WORKER_VLLM_HOST_IP=10.77.77.2",
                    "DSPARK_MODEL=deepseek-ai/DeepSeek-V4-Flash-0731",
                    f"DSPARK_MODEL_REVISION={revision}",
                    f"DSPARK_VLLM_IMAGE={image}",
                    f"VLLM_ORIGIN_KEY_FILE={secret}",
                    "SERVED_MODEL_NAME=deepseek-v4-flash-0731",
                    "HF_CACHE=/tmp/head-hf",
                    "WORKER_HF_CACHE=/tmp/worker-hf",
                    "SECRET_CANARY=must-never-copy",
                    "VLLM_ORIGIN_KEY_HASH=must-never-copy",
                    "VLLM_API_KEY=must-never-copy",
                    "",
                ]
            )
        )
        return env_file, secret

    def test_example_uses_exact_immutable_pins_and_head_secret_file_contract(self):
        example = (ROOT / ".env.dspark.example").read_text()
        compose = (ROOT / "docker-compose.dspark.yml").read_text()
        self.assertIn(f"DSPARK_MODEL_REVISION={REVISION}", example)
        self.assertIn(f"DSPARK_VLLM_IMAGE={IMAGE}", example)
        self.assertIn("VLLM_ORIGIN_KEY_FILE=", example)
        self.assertIn("--revision ${DSPARK_MODEL_REVISION}", compose)
        self.assertNotIn("VLLM_API_KEY", compose)
        self.assertNotIn("VLLM_ORIGIN_KEY_FILE", compose)

    def test_validator_rejects_missing_revision_digest_and_secret_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            cases = [
                ("", IMAGE, True, "DSPARK_MODEL_REVISION"),
                (REVISION, "ghcr.io/anemll/dspark-vllm-gx10:0.1.1", True, "digest"),
                (REVISION, IMAGE, False, "VLLM_ORIGIN_KEY_FILE"),
            ]
            for revision, image, key_file, expected in cases:
                with self.subTest(expected=expected):
                    env_file, _ = self.write_env(
                        directory, revision=revision, image=image, key_file=key_file
                    )
                    result = run(
                        "bash",
                        ROOT / "validate-dspark-config.sh",
                        env={**os.environ, "ENV_FILE": str(env_file), "VALIDATE_RENDER": "0"},
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(expected, result.stderr)

    def test_prepare_passes_revision_to_online_and_offline_snapshot_resolution(self):
        script = (ROOT / "prepare-dspark-model-cache.sh").read_text()
        self.assertGreaterEqual(script.count('revision=os.environ["DSPARK_MODEL_REVISION"]'), 2)
        self.assertIn("local_files_only=True", script)
        offline_section = script.split("verify_cache()", 1)[1]
        self.assertIn("HF_HUB_OFFLINE=1", offline_section)

    def test_manifest_is_deterministic_full_snapshot_and_detects_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            snapshot = directory / "snapshot"
            (snapshot / "encoding").mkdir(parents=True)
            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"x": "model-00001-of-00001.safetensors"}})
            )
            (snapshot / "model-00001-of-00001.safetensors").write_bytes(b"weights")
            (snapshot / "config.json").write_text("{}")
            (snapshot / "tokenizer.json").write_text("{}")
            executable = snapshot / "encoding" / "encoding_dsv4.py"
            executable.write_text("print('ok')\n")
            executable.chmod(0o755)
            first = directory / "first.json"
            second = directory / "second.json"
            create = [
                "python3",
                ROOT / "scripts/verify-artifact-manifest.py",
                "create",
                "--snapshot",
                snapshot,
                "--revision",
                REVISION,
                "--image",
                IMAGE,
            ]
            self.assertEqual(run(*create, "--output", first).returncode, 0)
            self.assertEqual(run(*create, "--output", second).returncode, 0)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            manifest = json.loads(first.read_text())
            self.assertEqual(
                [entry["path"] for entry in manifest["files"]],
                sorted(
                    [
                        "config.json",
                        "encoding/encoding_dsv4.py",
                        "model-00001-of-00001.safetensors",
                        "model.safetensors.index.json",
                        "tokenizer.json",
                    ]
                ),
            )
            self.assertTrue(
                next(e for e in manifest["files"] if e["path"].endswith("encoding_dsv4.py"))[
                    "executable"
                ]
            )
            (snapshot / "model-00001-of-00001.safetensors").write_bytes(b"changed")
            self.assertEqual(run(*create, "--output", second).returncode, 0)
            mismatch = run(
                "python3",
                ROOT / "scripts/verify-artifact-manifest.py",
                "compare",
                "--left",
                first,
                "--right",
                second,
            )
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn("manifest mismatch", mismatch.stderr.lower())

    def test_worker_env_is_allowlisted_and_diagnostics_redact_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            env_file, secret = self.write_env(directory)
            head = directory / "head.env"
            worker = directory / "worker.env"
            generated = run(
                "python3",
                ROOT / "scripts/generate-node-env.py",
                "--source",
                env_file,
                "--head-output",
                head,
                "--worker-output",
                worker,
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)
            head_text = head.read_text()
            worker_text = worker.read_text()
            self.assertIn(f"VLLM_ORIGIN_KEY_FILE={secret}", head_text)
            self.assertIn("NODE_RANK=1", worker_text)
            self.assertIn("HEADLESS=1", worker_text)
            self.assertIn("HF_CACHE=/tmp/worker-hf", worker_text)
            self.assertNotIn("VLLM_ORIGIN_KEY", worker_text)
            self.assertNotIn("VLLM_API_KEY", worker_text)
            self.assertNotIn("SECRET_CANARY", worker_text)
            self.assertNotIn("must-never-copy", worker_text)
            self.assertNotIn("test-secret-value", worker_text)
            self.assertEqual(stat.S_IMODE(worker.stat().st_mode), 0o600)

            fake_bin = directory / "bin"
            fake_bin.mkdir()
            docker = fake_bin / "docker"
            docker.write_text("#!/bin/sh\nprintf 'rendered API_KEY=test-secret-value\\n'\n")
            docker.chmod(0o755)
            validated = run(
                "bash",
                ROOT / "validate-dspark-config.sh",
                env={
                    **os.environ,
                    "PATH": f"{fake_bin}:{os.environ['PATH']}",
                    "ENV_FILE": str(env_file),
                    "HEAD_ENV_FILE": str(head),
                    "WORKER_ENV_FILE": str(worker),
                },
            )
            combined = validated.stdout + validated.stderr
            self.assertNotIn("test-secret-value", combined)
            self.assertNotIn("must-never-copy", combined)


if __name__ == "__main__":
    unittest.main()
