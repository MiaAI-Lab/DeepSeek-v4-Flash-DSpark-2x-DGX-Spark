#!/usr/bin/env python3
"""CPU tests for DSpark checkpoint-to-stacked-parameter mapping."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "recipe" / "overlay" / "vllm" / "v1" / "spec_decode" / "dspark.py"
sys.modules["torch"] = types.ModuleType("torch")
SPEC = importlib.util.spec_from_file_location("dspark_mapping", SOURCE)
dspark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = dspark
SPEC.loader.exec_module(dspark)


class DSparkStackedMappingTest(unittest.TestCase):
    def test_maps_shared_expert_gate_and_up_shards(self):
        cases = (
            ("model.layers.43.ffn.shared_experts.w1.weight", "model.layers.43.ffn.shared_experts.gate_up_proj.weight", 0),
            ("model.layers.45.ffn.shared_experts.w3.weight_scale_inv", "model.layers.45.ffn.shared_experts.gate_up_proj.weight_scale_inv", 1),
        )
        for source, target, shard in cases:
            with self.subTest(source=source):
                self.assertEqual(dspark.map_dspark_stacked_param_name(source), (target, shard))

    def test_does_not_capture_routed_experts_or_markov_head(self):
        names = (
            "model.layers.43.ffn.experts.0.w1.weight",
            "model.layers.45.markov_head.markov_w1.weight",
        )
        for name in names:
            with self.subTest(name=name):
                self.assertIsNone(dspark.map_dspark_stacked_param_name(name))

    def test_keeps_attention_mapping(self):
        source = "model.layers.43.attn.wkv.weight"
        self.assertEqual(
            dspark.map_dspark_stacked_param_name(source),
            ("model.layers.43.attn.fused_wqa_wkv.weight", 1),
        )


if __name__ == "__main__":
    unittest.main()
