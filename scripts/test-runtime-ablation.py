#!/usr/bin/env python3
"""CPU/source guards for the optional DSV4 runtime-ablation path."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOTFIX = ROOT / "patches/hotfix-dsv4-runtime-ablation.py"
COMPOSE = ROOT / "docker-compose.dspark.yml"
START = ROOT / "start-deepseek-v4-flash-dspark.sh"
ENV_EXAMPLE = ROOT / ".env.dspark.example"
DIRECTION = ROOT / "files/direction_r1.pt"
OVERLAY_MODEL = ROOT / "recipe/overlay/vllm/models/deepseek_v4/nvidia/model.py"
EXPECTED_DIRECTION_SHA = (
    "6e4d8a8f3aa9e21795faab2c5b14d29b019acdf2ddbfbd8238430458a5837fe0"
)


def load_hotfix():
    spec = importlib.util.spec_from_file_location("runtime_ablation_hotfix", HOTFIX)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANEMLL_FIXTURE = '''# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import regex as re
import torch
import torch.nn as nn

logger = None

class DeepseekV4DecoderLayer(nn.Module):
    def __init__(self, vllm_config, prefix):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size

        self.rms_norm_eps = config.rms_norm_eps

    def forward(self, x, positions, input_ids, post_mix=None, res_mix=None, residual=None):
        x = self.attn(positions, x, None)
        return x, residual, post_mix, res_mix

class DeepseekV4Model(nn.Module):
    pass
'''


class RuntimeAblationPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hotfix = load_hotfix()

    def test_bundled_direction_digest(self):
        self.assertEqual(hashlib.sha256(DIRECTION.read_bytes()).hexdigest(), EXPECTED_DIRECTION_SHA)

    def test_pinned_anemll_shape_gets_one_hook(self):
        updated, sites = self.hotfix.patch_text(ANEMLL_FIXTURE)
        self.assertEqual(sites, 1)
        self.assertEqual(updated.count(self.hotfix.MARK), 1)
        self.assertEqual(updated.count("x = self._ablate_refusal_direction(x)"), 1)
        compile(updated, "anemll_model.py", "exec")
        again, again_sites = self.hotfix.patch_text(updated)
        self.assertEqual(again, updated)
        self.assertEqual(again_sites, 1)

    def test_current_stage_c_source_gets_all_three_hooks(self):
        updated, sites = self.hotfix.patch_text(OVERLAY_MODEL.read_text())
        self.assertEqual(sites, 3)
        self.assertEqual(updated.count("x = self._ablate_refusal_direction(x)"), 3)
        compile(updated, str(OVERLAY_MODEL), "exec")

    def test_anchor_drift_fails_closed(self):
        broken = ANEMLL_FIXTURE.replace("self.hidden_size = config.hidden_size", "self.hidden = config.hidden_size")
        with self.assertRaisesRegex(ValueError, "decoder init"):
            self.hotfix.patch_text(broken)

    def test_cache_stamp_preserves_first_stock_cache_then_wipes_on_enable(self):
        old_root = os.environ.get("VLLM_CACHE_ROOT")
        old_lam = os.environ.get("DSV4_ABLATE_LAMBDA")
        old_layers = os.environ.get("DSV4_ABLATE_LAYERS")
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                cache = root / "torch_compile_cache"
                cache.mkdir()
                sentinel = cache / "stock"
                sentinel.write_text("ok")
                os.environ["VLLM_CACHE_ROOT"] = td
                self.hotfix.sync_compile_cache_stamp(False)
                self.assertTrue(sentinel.exists())
                os.environ["DSV4_ABLATE_LAMBDA"] = "3.5"
                os.environ["DSV4_ABLATE_LAYERS"] = "10-42"
                self.hotfix.sync_compile_cache_stamp(True, EXPECTED_DIRECTION_SHA)
                self.assertFalse(cache.exists())
                stamp = (root / ".dsv4_ablate_stamp").read_text()
                self.assertIn("enabled=1", stamp)
                self.assertIn(EXPECTED_DIRECTION_SHA, stamp)
        finally:
            for key, value in (
                ("VLLM_CACHE_ROOT", old_root),
                ("DSV4_ABLATE_LAMBDA", old_lam),
                ("DSV4_ABLATE_LAYERS", old_layers),
            ):
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class RuntimeAblationWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compose = COMPOSE.read_text()
        cls.start = START.read_text()
        cls.env = ENV_EXAMPLE.read_text()

    def test_compose_is_off_by_default_and_fail_closed(self):
        self.assertIn('ABLATE: "${ABLATE:-0}"', self.compose)
        self.assertIn("python3 /opt/hotfix-dsv4-runtime-ablation.py || exit 1", self.compose)
        self.assertIn("hotfix-dsv4-runtime-ablation.py:ro", self.compose)
        self.assertIn("unset DSV4_ABLATE_FILE", self.compose)
        self.assertIn(
            "export DSV4_ABLATE_FILE=/cache/huggingface/dspark-ablation/direction_r1.pt",
            self.compose,
        )

    def test_launcher_syncs_patch_and_direction_to_worker(self):
        self.assertIn("stage_ablation_direction", self.start)
        self.assertIn("worker ablation direction SHA-256 mismatch", self.start)
        self.assertIn(
            'scp "$DSPARK_ABLATION_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-dsv4-runtime-ablation.py"',
            self.start,
        )

    def test_double_ablation_is_rejected(self):
        self.assertIn('if [ "$ABLATE" = "1" ] && [ "${ABLITERATED:-0}" = "1" ]', self.start)

    def test_one_shot_shell_override_wins_over_env_file(self):
        source_pos = self.start.index('source "$_dspark_env_clean"')
        restore_pos = self.start.index('ABLATE="$_dspark_ambient_ablate"')
        resolve_pos = self.start.index('ABLATE="${ABLATE:-0}"')
        self.assertLess(source_pos, restore_pos)
        self.assertLess(restore_pos, resolve_pos)
        self.assertIn("DSV4_ABLATE_LAMBDA=", self.start)
        self.assertIn("DSV4_ABLATE_LAYERS=", self.start)

    def test_example_documents_off_default(self):
        self.assertIn("ABLATE=0", self.env)
        self.assertIn("DSV4_ABLATE_LAMBDA=3.5", self.env)
        self.assertIn("DSV4_ABLATE_LAYERS=10-42", self.env)


if __name__ == "__main__":
    unittest.main()
