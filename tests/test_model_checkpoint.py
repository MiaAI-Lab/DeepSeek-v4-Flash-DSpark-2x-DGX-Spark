#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_model_checkpoint", ROOT / "scripts/verify_model_checkpoint.py"
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class ModelCheckpointContractTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> None:
        (root / ".r0b0tlab-model-revision").write_text(module.EXPECTED_REVISION + "\n")
        config = {
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_v4",
            "expert_dtype": "fp4",
            "max_position_embeddings": 1048576,
            "n_routed_experts": 256,
            "num_experts_per_tok": 6,
            "num_hidden_layers": 43,
            "num_nextn_predict_layers": 1,
        }
        (root / "config.json").write_text(json.dumps(config))
        weight_map = {}
        for index, name in enumerate(sorted(module.EXPECTED_SHARDS), 1):
            (root / name).write_bytes(b"x")
            weight_map[f"tensor.{index}"] = name
        index = {"metadata": {"total_size": module.EXPECTED_TOTAL_SIZE}, "weight_map": weight_map}
        (root / "model.safetensors.index.json").write_text(json.dumps(index))

    def test_integrated_48_shard_fixture_passes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.make_fixture(root)
            result = module.verify(root, module.EXPECTED_REVISION)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["shards"], 48)

    def test_missing_revision_marker_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.make_fixture(root)
            (root / ".r0b0tlab-model-revision").unlink()
            with self.assertRaisesRegex(ValueError, "revision marker"):
                module.verify(root, module.EXPECTED_REVISION)

    def test_missing_integrated_shard_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.make_fixture(root)
            (root / "model-00048-of-00048.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "missing or empty model shards"):
                module.verify(root, module.EXPECTED_REVISION)


if __name__ == "__main__":
    unittest.main(verbosity=2)
