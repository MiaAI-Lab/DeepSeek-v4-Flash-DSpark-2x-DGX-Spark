import json
import os
from pathlib import Path
import re
import stat
import shutil
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
        vllm_service = compose.split("origin-auth-proxy:", 1)[0]
        self.assertNotIn("VLLM_ORIGIN_KEY_FILE", vllm_service)

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

    def test_validator_accepts_an_immutable_local_derived_image_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            image_id = "sha256:" + "a" * 64
            env_file, _ = self.write_env(directory, image=image_id)
            result = run(
                "bash",
                ROOT / "validate-dspark-config.sh",
                env={**os.environ, "ENV_FILE": str(env_file), "VALIDATE_RENDER": "0"},
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_prepare_passes_revision_to_online_and_offline_snapshot_resolution(self):
        script = (ROOT / "prepare-dspark-model-cache.sh").read_text()
        self.assertGreaterEqual(script.count('revision=os.environ["DSPARK_MODEL_REVISION"]'), 2)
        self.assertIn("local_files_only=True", script)
        self.assertIn('HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-0}"', script)
        offline_section = script.split("verify_cache()", 1)[1]
        self.assertIn("HF_HUB_OFFLINE=1", offline_section)
        self.assertIn("rsync -a --partial --safe-links", script)
        self.assertIn("--exclude='*.incomplete' --exclude='trees/'", script)
        self.assertIn("PREPARE_DOWNLOAD=0", script)
        self.assertIn('${HF_DOWNLOAD_WORKERS:=4}', script)
        self.assertNotIn("rsync -a --delete", script)

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

    def test_manifest_hashes_huggingface_cache_symlinks_and_rejects_escapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_root = Path(tmp) / "models--deepseek-ai--DeepSeek-V4-Flash-0731"
            blobs = model_root / "blobs"
            snapshot = model_root / "snapshots" / REVISION
            blobs.mkdir(parents=True)
            snapshot.mkdir(parents=True)
            blob = blobs / "model-blob"
            blob.write_bytes(b"weights")
            shard = snapshot / "model-00001-of-00001.safetensors"
            shard.symlink_to(Path("../../blobs") / blob.name)
            output = Path(tmp) / "manifest.json"
            create = run(
                "python3", ROOT / "scripts/verify-artifact-manifest.py", "create",
                "--snapshot", snapshot, "--revision", REVISION, "--image", IMAGE,
                "--output", output,
            )
            self.assertEqual(create.returncode, 0, create.stderr)
            entry = json.loads(output.read_text())["files"][0]
            self.assertEqual(entry["link_target"], "../../blobs/model-blob")
            self.assertEqual(entry["size"], len(b"weights"))

            shard.unlink()
            outside = Path(tmp) / "outside"
            outside.write_bytes(b"escape")
            shard.symlink_to(outside)
            rejected = run(
                "python3", ROOT / "scripts/verify-artifact-manifest.py", "create",
                "--snapshot", snapshot, "--revision", REVISION, "--image", IMAGE,
                "--output", output,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("escapes model cache", rejected.stderr)

    def test_resolved_profile_reports_memory_sizing_and_guard_separately(self):
        start = (ROOT / "start-deepseek-v4-flash-dspark.sh").read_text()
        self.assertIn("memory control: kv-cache-memory-bytes", start)
        self.assertIn("startup guard gpu-memory-utilization", start)
        self.assertIn("memory control: gpu-memory-utilization", start)

    def test_default_memory_scheduler_profile_is_explicit_and_deterministic(self):
        example = (ROOT / ".env.dspark.example").read_text()
        compose = (ROOT / "docker-compose.dspark.yml").read_text()
        self.assertIn("MEMORY_CONTROL=kv-cache-memory-bytes", example)
        self.assertIn("KV_CACHE_MEMORY_BYTES=12884901888", example)
        self.assertRegex(example, r"(?m)^GPU_MEMORY_UTILIZATION=0\.80$")
        self.assertIn("MAX_NUM_BATCHED_TOKENS=8216", example)
        self.assertIn("MAX_CUDAGRAPH_CAPTURE_SIZE=32", example)
        self.assertIn("ENFORCE_EAGER=0", example)
        self.assertIn("--gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.80}", compose)
        self.assertIn("--kv-cache-memory-bytes ${KV_CACHE_MEMORY_BYTES:-12884901888}", compose)
        self.assertIn("MEMORY_ARGS", compose)
        self.assertIn("--max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS:-8216}", compose)
        self.assertIn("--max-cudagraph-capture-size ${MAX_CUDAGRAPH_CAPTURE_SIZE:-32}", compose)

    def test_compose_renders_one_kv_sizing_control_plus_admission_guard(self):
        compose_bin = shutil.which("docker-compose")
        if not compose_bin:
            self.skipTest("docker-compose is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env.dspark"
            default = (ROOT / ".env.dspark.example").read_text()
            env_file.write_text(default)
            rendered = run(
                compose_bin, "--env-file", env_file, "-f", ROOT / "docker-compose.dspark.yml", "config"
            )
            self.assertEqual(rendered.returncode, 0, rendered.stderr)
            self.assertIn("--kv-cache-memory-bytes 12884901888", rendered.stdout)
            self.assertIn("--gpu-memory-utilization 0.80", rendered.stdout)

            compatibility = default.replace(
                "MEMORY_CONTROL=kv-cache-memory-bytes", "MEMORY_CONTROL=gpu-memory-utilization", 1
            ).replace("KV_CACHE_MEMORY_BYTES=12884901888", "KV_CACHE_MEMORY_BYTES=", 1)
            env_file.write_text(compatibility)
            rendered = run(
                compose_bin, "--env-file", env_file, "-f", ROOT / "docker-compose.dspark.yml", "config"
            )
            self.assertEqual(rendered.returncode, 0, rendered.stderr)
            self.assertIn(r'case \"gpu-memory-utilization\"', rendered.stdout)
            self.assertIn(r'gpu-memory-utilization) MEMORY_ARGS=\"--gpu-memory-utilization 0.80\"', rendered.stdout)

    def test_validator_rejects_invalid_or_conflicting_memory_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            cases = (
                ({"KV_CACHE_MEMORY_BYTES": "nope"}, "positive integer"),
                ({"KV_CACHE_MEMORY_BYTES": "0"}, "positive integer"),
                ({"KV_CACHE_MEMORY_BYTES": "-1"}, "positive integer"),
                ({"GPU_MEMORY_UTILIZATION": "0"}, "greater than 0"),
                ({"GPU_MEMORY_UTILIZATION": "1.2"}, "no greater than 1"),
                ({"MEMORY_CONTROL": "gpu-memory-utilization"}, "cannot use KV_CACHE_MEMORY_BYTES"),
            )
            for overrides, expected in cases:
                with self.subTest(overrides=overrides):
                    env_file, _ = self.write_env(directory)
                    with env_file.open("a") as handle:
                        handle.write("MEMORY_CONTROL=kv-cache-memory-bytes\n")
                        handle.write("KV_CACHE_MEMORY_BYTES=12884901888\n")
                        handle.write("GPU_MEMORY_UTILIZATION=0.80\n")
                        for key, value in overrides.items():
                            handle.write(f"{key}={value}\n")
                    result = run(
                        "bash", ROOT / "validate-dspark-config.sh",
                        env={**os.environ, "ENV_FILE": str(env_file), "VALIDATE_RENDER": "0"},
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(expected, result.stderr)

    def test_fractional_compatibility_profile_is_explicit_and_projected_to_both_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            env_file, _ = self.write_env(directory)
            with env_file.open("a") as handle:
                handle.write("MEMORY_CONTROL=gpu-memory-utilization\n")
                handle.write("KV_CACHE_MEMORY_BYTES=\n")
                handle.write("GPU_MEMORY_UTILIZATION=0.80\n")
                handle.write("MAX_NUM_BATCHED_TOKENS=8216\n")
                handle.write("MAX_CUDAGRAPH_CAPTURE_SIZE=32\n")
                handle.write("ENFORCE_EAGER=0\n")
            validated = run(
                "bash", ROOT / "validate-dspark-config.sh",
                env={**os.environ, "ENV_FILE": str(env_file), "VALIDATE_RENDER": "0"},
            )
            self.assertEqual(validated.returncode, 0, validated.stderr)
            head = directory / "head.env"
            worker = directory / "worker.env"
            generated = run(
                "python3", ROOT / "scripts/generate-node-env.py",
                "--source", env_file, "--head-output", head, "--worker-output", worker,
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)
            expected = {
                "MEMORY_CONTROL=gpu-memory-utilization",
                "KV_CACHE_MEMORY_BYTES=",
                "GPU_MEMORY_UTILIZATION=0.80",
                "MAX_NUM_BATCHED_TOKENS=8216",
                "MAX_CUDAGRAPH_CAPTURE_SIZE=32",
                "ENFORCE_EAGER=0",
            }
            for output in (head, worker):
                self.assertTrue(expected.issubset(set(output.read_text().splitlines())))

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


    def test_restart_policies_distinguish_long_running_and_temporary_services(self):
        dspark = (ROOT / "docker-compose.dspark.yml").read_text()
        gateway = (ROOT / "deployments/private-smoke/litellm/docker-compose.yml").read_text()
        self.assertEqual(
            {value.strip().strip("'\"") for value in re.findall(r"^\s*restart:\s*(.+)$", dspark, re.MULTILINE)},
            {"no"},
        )
        self.assertEqual(
            {value.strip().strip("'\"") for value in re.findall(r"^\s*restart:\s*(.+)$", gateway, re.MULTILINE)},
            {"unless-stopped", "no"},
        )
        self.assertRegex(gateway, r"(?s)litellm:.*?restart: \"no\"")
        self.assertRegex(gateway, r"(?s)postgres:.*?restart: unless-stopped")
        self.assertIn("dspark-private-litellm-pgdata", gateway)
        self.assertNotIn("/var/lib/postgresql/data:size=", gateway)

    def test_scheduler_supports_authoritative_prechange_baseline_capture(self):
        script = (ROOT / "scripts/benchmark-scheduler.py").read_text()
        self.assertIn("--capture-baseline", script)
        self.assertIn("dspark-scheduler-baseline/v1", script)
        self.assertIn("baseline_capture", script)
        self.assertIn('pop("target_scheduler_tokens"', script)
        self.assertIn("p95_decode_latency_s", script)
        self.assertIn("p95_ttft_s", script)

    def test_rollback_tooling_is_manifest_driven_and_fail_closed(self):
        tool = (ROOT / "deployments/private-smoke/litellm/rollback.sh").read_text()
        for required in (
            "umask 077", "SHA256SUMS", "sha256sum -c", "pg_dump", "--entrypoint pg_restore", "--list",
            "chmod 0600", "docker compose", "migrate", "restore", "verify-existing-key",
            "stop-deepseek-v4-flash-dspark.sh", "start-deepseek-v4-flash-dspark.sh",
        ):
            self.assertIn(required, tool)
        self.assertLess(tool.index("stop-deepseek-v4-flash-dspark.sh"), tool.index("start-deepseek-v4-flash-dspark.sh"))
        self.assertIn("worker-first", tool)
        self.assertIn("LITELLM_VIRTUAL_KEY_FILE", tool)


if __name__ == "__main__":
    unittest.main()
