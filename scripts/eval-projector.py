#!/usr/bin/env python3
"""Evaluate fine-tuned projector vs original on color QA.

Compares the original WebBrain projector with the fine-tuned version
by running the color gate offline (no server restart needed).

Usage (inside Docker container):
    python scripts/eval-projector.py \\
        --model-dir /cache/huggingface/dsv4-0731-moonvit \\
        --tower-path /cache/huggingface/webbrain-0731-moonvit-src/vision_tower.safetensors \\
        --original-projector /cache/huggingface/webbrain-0731-moonvit-src/mm_projector.safetensors \\
        --finetuned-projector /cache/huggingface/mm_projector-finetuned-0731.safetensors \\
        --trials 10 \\
        --out results/projector-finetuned-colors.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image

# Add plugin to path for imports
ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "dsv4_moonvit_vllm"
sys.path.insert(0, str(PLUGIN))

from dsv4_moonvit_vllm.moonvit import load_tower_and_projector
from dsv4_moonvit_vllm.preprocess import pil_to_pixel_values_and_grid
from dsv4_moonvit_vllm.projector import PatchMerger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Color fixtures and matching (same as smoke-moonvit-colors.py)
COLOR_PROMPT = (
    "What color is this solid image? One word: red/green/blue/black/white."
)

FIXTURES: dict[str, tuple[int, int, int]] = {
    "red": (255, 0, 0),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
}

SYNONYMS: dict[str, tuple[str, ...]] = {
    "red": ("red", "crimson", "scarlet"),
    "black": ("black",),
    "white": ("white",),
    "green": ("green",),
    "blue": ("blue",),
}


def answer_matches(color: str, text: str) -> bool:
    lower = text.lower()
    return any(syn in lower for syn in SYNONYMS[color])


def create_color_image(rgb: tuple[int, int, int], size: int = 256) -> Image.Image:
    return Image.new("RGB", (size, size), color=rgb)


def load_model(
    model_dir: str,
    tower_path: str,
    projector_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, Any, Any]:
    """Load tower, projector, and LM for inference."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load projector
    projector = PatchMerger()
    projector.load_webbrain_safetensors(projector_path, device=device, dtype=dtype)

    # Read vision config
    config_path = Path(model_dir) / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    vision_dict = config.get("vision_config", {})

    # Load tower
    tower, _, meta = load_tower_and_projector(
        vision_dict=vision_dict,
        tower_path=tower_path,
        projector_path=projector_path,
        device=device,
        dtype=dtype,
    )

    if tower is None:
        raise RuntimeError("Tower failed to load")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # Load LM (full precision for accurate comparison)
    lm = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    lm.eval()

    return tower, projector, lm, tokenizer


@torch.no_grad()
def generate_caption(
    tower: nn.Module,
    projector: PatchMerger,
    lm: Any,
    tokenizer: Any,
    image: Image.Image,
    device: torch.device,
    dtype: torch.dtype,
    max_tokens: int = 32,
) -> str:
    """Generate caption for an image using tower + projector + LM."""
    # Encode image
    pixel_values, grid, ntok = pil_to_pixel_values_and_grid(image, max_image_tokens=512)
    pixel_values = pixel_values.unsqueeze(0).to(device=device, dtype=dtype)
    grid_tensor = torch.tensor([grid], dtype=torch.long, device=device)

    tower_out = tower(pixel_values, grid_tensor)
    if isinstance(tower_out, (list, tuple)):
        tower_out = tower_out[0]

    projected = projector(tower_out.to(dtype=dtype))

    # Build prompt
    messages = [
        {"role": "user", "content": "<image>\n" + COLOR_PROMPT},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Get text embeddings
    prompt_embeds = lm.get_input_embeddings()(prompt_ids)

    # Replace image tokens with projected embeddings
    img_mask = (prompt_ids == 129280)
    if img_mask.any():
        text_embeds = prompt_embeds.clone()
        img_positions = img_mask[0].nonzero(as_tuple=True)[0]
        n_replace = min(len(img_positions), projected.shape[0])
        text_embeds[0, img_positions[:n_replace]] = projected[:n_replace]

        # Generate
        outputs = lm.generate(
            inputs_embeds=text_embeds,
            max_new_tokens=max_tokens,
            temperature=0.0,
            do_sample=False,
        )
    else:
        # Fallback: no image tokens found
        outputs = lm.generate(
            input_ids=prompt_ids,
            max_new_tokens=max_tokens,
            temperature=0.0,
            do_sample=False,
        )

    # Decode only new tokens
    generated = outputs[0, prompt_ids.shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip()


def run_color_trials(
    tower: nn.Module,
    projector: PatchMerger,
    lm: Any,
    tokenizer: Any,
    color: str,
    trials: int,
    device: torch.device,
    dtype: torch.dtype,
    image_size: int = 256,
) -> dict:
    """Run color QA trials for a specific color."""
    image = create_color_image(FIXTURES[color], image_size)

    answers = []
    hits = 0
    errors = 0

    for i in range(trials):
        try:
            text = generate_caption(
                tower, projector, lm, tokenizer, image, device, dtype
            )
            answers.append(text)
            hits += int(answer_matches(color, text))
        except Exception as e:
            errors += 1
            answers.append(f"ERROR: {e}")

    rate = hits / max(1, trials)
    return {
        "trials": trials,
        "hits": hits,
        "errors": errors,
        "pass_rate": round(rate, 3),
        "answers": answers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default="/cache/huggingface/dsv4-0731-moonvit",
    )
    parser.add_argument(
        "--tower-path",
        default="/cache/huggingface/webbrain-0731-moonvit-src/vision_tower.safetensors",
    )
    parser.add_argument(
        "--original-projector",
        default="/cache/huggingface/webbrain-0731-moonvit-src/mm_projector.safetensors",
    )
    parser.add_argument(
        "--finetuned-projector",
        default="/cache/huggingface/mm_projector-finetuned-0731.safetensors",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--colors",
        default="red,black,white,green,blue",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--out",
        default="results/projector-finetuned-colors.json",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only compare if both projectors exist",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    colors = [c.strip() for c in args.colors.split(",")]

    # Check if projectors exist
    original_exists = Path(args.original_projector).is_file()
    finetuned_exists = Path(args.finetuned_projector).is_file()

    if args.compare_only and not (original_exists and finetuned_exists):
        logger.error("Both projectors must exist for comparison")
        return 1

    if not original_exists:
        logger.error("Original projector not found: %s", args.original_projector)
        return 1

    results = {"config": vars(args)}

    # Load tower (shared between both)
    logger.info("Loading tower from %s", args.tower_path)
    config_path = Path(args.model_dir) / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    vision_dict = config.get("vision_config", {})

    tower, _, _ = load_tower_and_projector(
        vision_dict=vision_dict,
        tower_path=args.tower_path,
        projector_path=args.original_projector,
        device=device,
        dtype=dtype,
    )
    if tower is None:
        logger.error("Tower failed to load")
        return 1

    # Load tokenizer (shared)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    # Load LM (shared)
    from transformers import AutoModelForCausalLM
    logger.info("Loading LM from %s", args.model_dir)
    lm = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    lm.eval()

    # Test original projector
    logger.info("Testing ORIGINAL projector: %s", args.original_projector)
    original_projector = PatchMerger()
    original_projector.load_webbrain_safetensors(args.original_projector, device=device, dtype=dtype)

    results["original"] = {}
    for color in colors:
        logger.info("  Color: %s", color)
        r = run_color_trials(
            tower, original_projector, lm, tokenizer,
            color, args.trials, device, dtype, args.image_size,
        )
        results["original"][color] = r
        logger.info("    %s: %d/%d (%.2f)", color, r["hits"], r["trials"], r["pass_rate"])

    # Test fine-tuned projector if it exists
    if finetuned_exists:
        logger.info("Testing FINE-TUNED projector: %s", args.finetuned_projector)
        finetuned_projector = PatchMerger()
        finetuned_projector.load_webbrain_safetensors(args.finetuned_projector, device=device, dtype=dtype)

        results["finetuned"] = {}
        for color in colors:
            logger.info("  Color: %s", color)
            r = run_color_trials(
                tower, finetuned_projector, lm, tokenizer,
                color, args.trials, device, dtype, args.image_size,
            )
            results["finetuned"][color] = r
            logger.info("    %s: %d/%d (%.2f)", color, r["hits"], r["trials"], r["pass_rate"])

        # Compare results
        logger.info("\n=== COMPARISON ===")
        for color in colors:
            orig_rate = results["original"][color]["pass_rate"]
            ft_rate = results["finetuned"][color]["pass_rate"]
            delta = ft_rate - orig_rate
            logger.info(
                "%s: original=%.2f finetuned=%.2f delta=%+.2f",
                color, orig_rate, ft_rate, delta,
            )
    else:
        logger.warning("Fine-tuned projector not found, skipping comparison")
        results["finetuned"] = None

    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to %s", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
