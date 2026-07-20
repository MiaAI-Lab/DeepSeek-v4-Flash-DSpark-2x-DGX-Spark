#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReleaseContractTests(unittest.TestCase):
    def test_production_manifest_preserves_nvfp4_ds_mla(self) -> None:
        manifest = json.loads((ROOT / "docker/runtime-manifest.production.json").read_text())
        self.assertEqual(manifest["model_id"], "deepseek-ai/DeepSeek-V4-Flash-DSpark")
        self.assertEqual(manifest["model_revision"], "913f0657a874f76844e2e91cbe706dbcaceeb6d7")
        self.assertEqual(manifest["kv_cache_dtype"], "nvfp4_ds_mla")
        self.assertEqual(manifest["speculative_method"], "dspark")
        self.assertEqual(manifest["production_profile"]["max_num_seqs"], 16)
        self.assertEqual(manifest["capacity_profile_384k"]["runtime_lane"], "legacy-stage-c")
        self.assertEqual(manifest["capacity_profile_384k"]["max_model_len"], 384000)
        self.assertEqual(manifest["long_context_profile"]["max_model_len"], 1048576)
        self.assertEqual(manifest["long_context_profile"]["runtime_lane"], "legacy-stage-c")
        self.assertEqual(manifest["cuda_version"], "13.0")

    def test_production_image_is_pinned_and_audited(self) -> None:
        text = (ROOT / "recipe/Dockerfile.production").read_text()
        self.assertIn("ghcr.io/anemll/dspark-vllm-gx10@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8", text)
        self.assertIn("runtime-manifest.production.json", text)
        self.assertIn('ENTRYPOINT ["/usr/local/bin/dspark-entrypoint.sh"]', text)
        self.assertIn('io.r0b0tlab.kv-cache.dtype="nvfp4_ds_mla"', text)
        self.assertNotIn("Marlin", text)

    def test_production_build_refuses_dirty_or_unpinned_release_inputs(self) -> None:
        text = (ROOT / "scripts/build-production-image.sh").read_text()
        self.assertIn("git -C \"$ROOT\" status --porcelain", text)
        self.assertIn("ALLOW_DIRTY_BUILD", text)
        self.assertIn("ALLOW_UNPINNED_BASE", text)
        self.assertIn("PINNED_BASE_IMAGE", text)

    def test_public_safety_scan_rejects_private_hostnames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            leak = Path(tmp) / "evidence.txt"
            leak.write_text("NCCL banner from gn100-2eea and r0b0t-dgx\n")
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/public_safety_scan.py"), tmp],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("private-hostname", result.stdout)

    def test_launch_profile_is_native_nvfp4_and_no_marlin(self) -> None:
        text = (ROOT / "run-dspark-dual-gb10.sh").read_text()
        self.assertIn('KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-nvfp4_ds_mla}"', text)
        self.assertIn("--moe-backend flashinfer_b12x", text)
        self.assertIn("--device=/dev/infiniband", text)
        self.assertIn("--shm-size=64g", text)
        self.assertNotIn("marlin", text.lower())

    def test_profiles_keep_throughput_and_one_million_lanes(self) -> None:
        production = (ROOT / "profiles/dspark-r0b0tlab-production.env").read_text()
        long_context = (ROOT / "profiles/dspark-r0b0tlab-1m.env").read_text()
        for needle in (
            "KV_CACHE_DTYPE=nvfp4_ds_mla",
            "MAX_MODEL_LEN=200000",
            "MAX_NUM_SEQS=16",
            "MAX_NUM_BATCHED_TOKENS=16384",
        ):
            self.assertIn(needle, production)
        self.assertIn("MAX_MODEL_LEN=1048576", long_context)
        self.assertIn("MAX_NUM_SEQS=2", long_context)
        self.assertIn("KV_CACHE_DTYPE=nvfp4_ds_mla", long_context)
        self.assertIn("DSPARK_RUNTIME_LANE=legacy-stage-c", long_context)
        self.assertIn("DSPARK_VLLM_IMAGE=vllm-dspark-runtime:dspark-nvfp4-stage-c", long_context)

    def test_production_runtime_source_contract_is_v025_native(self) -> None:
        audit = (ROOT / "scripts/audit_runtime.py").read_text()
        manifest = json.loads((ROOT / "docker/runtime-manifest.production.json").read_text())
        self.assertIn("v1/worker/gpu/spec_decode/dspark/speculator.py", audit)
        self.assertIn("DSparkSpeculator", audit)
        self.assertIn("DSparkDeepseekV4", audit)
        self.assertEqual(manifest["vllm_version"], "0.25.2.dev0+g752a3a504.d20260714")
        self.assertEqual(manifest["flashinfer_version"], "0.6.15")


if __name__ == "__main__":
    unittest.main(verbosity=2)
