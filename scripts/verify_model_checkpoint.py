#!/usr/bin/env python3
"""Verify the integrated 48-shard DSpark checkpoint inventory before launch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

EXPECTED_REVISION = "913f0657a874f76844e2e91cbe706dbcaceeb6d7"
EXPECTED_TOTAL_SIZE = 166878536440
EXPECTED_SHARDS = {f"model-{index:05d}-of-00048.safetensors" for index in range(1, 49)}


def fail(message: str) -> None:
    raise ValueError(message)


def verify(root: Path, expected_revision: str) -> dict[str, object]:
    if not root.is_dir():
        fail(f"model directory not found: {root}")
    revision_path = root / ".r0b0tlab-model-revision"
    if not revision_path.is_file():
        fail(f"missing model revision marker: {revision_path}")
    revision = revision_path.read_text(encoding="utf-8").strip()
    if revision != expected_revision:
        fail(f"model revision marker is {revision!r}, expected {expected_revision!r}")

    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    required_config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "expert_dtype": "fp4",
        "max_position_embeddings": 1048576,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "num_hidden_layers": 43,
        "num_nextn_predict_layers": 1,
    }
    for key, expected in required_config.items():
        if config.get(key) != expected:
            fail(f"config.{key} is {config.get(key)!r}, expected {expected!r}")

    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    if index.get("metadata", {}).get("total_size") != EXPECTED_TOTAL_SIZE:
        fail("integrated checkpoint total_size does not match the admitted snapshot")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        fail("weight_map is missing or empty")
    shards = set(weight_map.values())
    if shards != EXPECTED_SHARDS:
        missing = sorted(EXPECTED_SHARDS - shards)
        extra = sorted(shards - EXPECTED_SHARDS)
        fail(f"48-shard inventory mismatch: missing={missing}, extra={extra}")
    malformed = sorted(name for name in shards if not re.fullmatch(r"model-\d{5}-of-00048\.safetensors", name))
    if malformed:
        fail(f"malformed shard names: {malformed}")
    absent = sorted(name for name in shards if not (root / name).is_file() or (root / name).stat().st_size == 0)
    if absent:
        fail(f"missing or empty model shards: {absent}")

    return {
        "status": "PASS",
        "revision": revision,
        "model_type": config["model_type"],
        "architecture": config["architectures"][0],
        "index_total_size": EXPECTED_TOTAL_SIZE,
        "shards": len(shards),
        "weight_entries": len(weight_map),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--expected-revision", default=EXPECTED_REVISION)
    args = parser.parse_args()
    try:
        result = verify(args.model_dir, args.expected_revision)
    except Exception as exc:
        print(f"MODEL_CHECKPOINT_VERIFY_FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    print("DSPARK_MODEL_CHECKPOINT_VERIFY_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
