#!/usr/bin/env python3
"""Extract the real 0731 embedding table (and lm_head) from checkpoint shards.

Reads only the two tensors via safetensors lazy loading — no full model load.
Used by train-projector-v3.py so the projector can be aligned against the
LM's actual input embedding space instead of a random stand-in.

Usage (inside the Anemll container):
    python3 /tmp/extract-embed-table.py \
        --model-dir /cache/huggingface/dsv4-0731-moonvit \
        --out /cache/huggingface/webbrain-0731-moonvit-src/embed_table.safetensors
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WANTED = ("embed.weight", "head.weight")  # head = lm_head (logit-side diagnostic)


def _norm_stats(t: torch.Tensor, name: str) -> dict[str, float]:
    t32 = t.float()
    norms = t32.norm(dim=-1)
    stats = {
        "name": name,
        "shape": list(t.shape),
        "row_norm_mean": round(float(norms.mean()), 4),
        "row_norm_std": round(float(norms.std()), 4),
        "row_norm_p05": round(float(norms.quantile(0.05)), 4),
        "row_norm_p95": round(float(norms.quantile(0.95)), 4),
    }
    logger.info(
        "%s shape=%s row-norm mean=%.3f std=%.3f p05=%.3f p95=%.3f",
        name, tuple(t.shape),
        stats["row_norm_mean"], stats["row_norm_std"],
        stats["row_norm_p05"], stats["row_norm_p95"],
    )
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default="/cache/huggingface/dsv4-0731-moonvit")
    ap.add_argument(
        "--out",
        default="/cache/huggingface/webbrain-0731-moonvit-src/embed_table.safetensors",
    )
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]

    found: dict[str, torch.Tensor] = {}
    for key in WANTED:
        shard = weight_map.get(key)
        if shard is None:
            logger.warning("%s not in weight map; skipping", key)
            continue
        shard_path = model_dir / shard
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            found[key] = f.get_tensor(key)
        logger.info("read %s from %s", key, shard)

    if "embed.weight" not in found:
        raise RuntimeError("embed.weight not found in checkpoint index")

    embed = found["embed.weight"]
    if embed.shape != (129280, 4096):
        raise RuntimeError(f"unexpected embed.weight shape {tuple(embed.shape)}")

    stats = [_norm_stats(embed, "embed.weight")]
    if "head.weight" in found:
        stats.append(_norm_stats(found["head.weight"], "head.weight"))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in found.items()}, str(out_path))
    logger.info("saved %s tensors to %s", sorted(found), out_path)

    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info("stats -> %s", stats_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
