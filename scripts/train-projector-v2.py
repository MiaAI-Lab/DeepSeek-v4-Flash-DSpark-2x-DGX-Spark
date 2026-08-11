#!/usr/bin/env python3
"""Fine-tune PatchMerger projector for DeepSeek-V4-Flash-0731.

V2: Uses color discrimination loss instead of contrastive learning.
Train projector so that different colors produce distinguishable embeddings.

Usage:
    python3 scripts/train-projector-v2.py \
        --num-samples 5000 \
        --epochs 10 \
        --output-dir /tmp/projector-v2
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "dsv4_moonvit_vllm"
sys.path.insert(0, str(PLUGIN))

from dsv4_moonvit_vllm.preprocess import pil_to_pixel_values_and_grid
from dsv4_moonvit_vllm.projector import PatchMerger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Color dataset with known labels
COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
}
COLOR_NAMES = list(COLORS.keys())
COLOR_TO_IDX = {name: i for i, name in enumerate(COLOR_NAMES)}


class StandaloneTower(nn.Module):
    """Simplified tower for training."""
    
    def __init__(self, hidden_size: int = 1152):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_proj = nn.Linear(3 * 14 * 14, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=16, dim_feedforward=4304, batch_first=True)
            for _ in range(4)
        ])
        
    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        n_patches = pixel_values.shape[0]
        grid_h = grid_thw[0, 1].item()
        grid_w = grid_thw[0, 2].item()
        
        x = pixel_values.flatten(1)  # (N, 588)
        x = self.patch_proj(x)  # (N, hidden)
        x = x.view(grid_h, grid_w, self.hidden_size)
        
        # Merge 2x2
        merged_h, merged_w = grid_h // 2, grid_w // 2
        x = x.reshape(merged_h, 2, merged_w, 2, self.hidden_size)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(merged_h * merged_w, 4 * self.hidden_size)
        x = x.view(merged_h * merged_w, 4, self.hidden_size)
        
        return x
    
    def load_safetensors(self, path: str, device: torch.device, dtype: torch.dtype):
        from safetensors.torch import load_file
        state = load_file(path, device="cpu")
        own_state = self.state_dict()
        loaded = 0
        for key, tensor in state.items():
            clean_key = key.removeprefix("vision_tower.")
            if clean_key in own_state and own_state[clean_key].shape == tensor.shape:
                own_state[clean_key] = tensor.to(device=device, dtype=dtype)
                loaded += 1
        self.load_state_dict(own_state)
        self.to(device=device, dtype=dtype)
        self.eval()
        return loaded


def save_projector(projector: PatchMerger, path: str):
    from safetensors.torch import save_file
    state = projector.state_dict()
    save_state = {}
    mapping = {
        "pre_norm.weight": "pre_norm.weight",
        "pre_norm.bias": "pre_norm.bias",
        "linear_1.weight": "proj.0.weight",
        "linear_1.bias": "proj.0.bias",
        "linear_2.weight": "proj.2.weight",
        "linear_2.bias": "proj.2.bias",
    }
    for torch_key, save_key in mapping.items():
        if torch_key in state:
            save_state[save_key] = state[torch_key]
    save_file(save_state, path)
    logger.info("Saved projector to %s", path)


def create_color_batch(batch_size: int, image_size: int = 256):
    """Create a batch of color images with labels."""
    import random
    
    images = []
    labels = []
    
    for _ in range(batch_size):
        color_name = random.choice(COLOR_NAMES)
        rgb = COLORS[color_name]
        # Add slight variation
        varied = tuple(max(0, min(255, c + random.randint(-20, 20))) for c in rgb)
        img = Image.new("RGB", (image_size, image_size), color=varied)
        images.append(img)
        labels.append(COLOR_TO_IDX[color_name])
    
    return images, labels


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    
    logger.info("Device: %s, dtype: %s", device, dtype)
    
    # Load projector (trainable)
    projector = PatchMerger()
    projector.load_webbrain_safetensors(args.projector_path, device="cpu", dtype=dtype)
    projector = projector.to(device=device)
    projector.train()
    projector.requires_grad_(True)
    
    # Load tower (frozen)
    tower = StandaloneTower(hidden_size=1152)
    tower.load_safetensors(args.tower_path, device, dtype)
    tower.eval()
    tower.requires_grad_(False)
    
    # Color classifier head (trainable) - maps embeddings to color classes
    classifier = nn.Linear(4096, len(COLOR_NAMES)).to(device=device, dtype=dtype)
    classifier.train()
    
    # Optimizer for both projector and classifier
    optimizer = torch.optim.AdamW(
        list(projector.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    
    total_steps = (args.num_samples // args.batch_size) * args.epochs
    
    logger.info("Starting color discrimination training for %d steps", total_steps)
    
    global_step = 0
    losses = []
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_idx in range(args.num_samples // args.batch_size):
            try:
                # Create batch of color images
                images, labels = create_color_batch(args.batch_size)
                labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
                
                # Process each image through tower + projector
                embeddings = []
                for img in images:
                    pv, grid, _ = pil_to_pixel_values_and_grid(img, max_image_tokens=512)
                    pv = pv.to(device=device, dtype=dtype)
                    grid_t = torch.tensor([grid], dtype=torch.long, device=device)
                    
                    with torch.no_grad():
                        tower_out = tower(pv, grid_t)
                    
                    proj_out = projector(tower_out.to(dtype=dtype))
                    # Pool over tokens
                    emb = proj_out.mean(dim=0)  # (4096,)
                    embeddings.append(emb)
                
                embeddings = torch.stack(embeddings)  # (batch, 4096)
                
                # Classify colors
                logits = classifier(embeddings)  # (batch, 10)
                
                # Cross-entropy loss
                loss = F.cross_entropy(logits, labels_tensor)
                
                # Backprop
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                
                optimizer.step()
                
                # Stats
                preds = logits.argmax(dim=-1)
                correct = (preds == labels_tensor).sum().item()
                epoch_correct += correct
                epoch_total += args.batch_size
                epoch_loss += loss.item()
                global_step += 1
                
                if global_step % args.log_every == 0:
                    acc = epoch_correct / epoch_total * 100
                    avg_loss = epoch_loss / (batch_idx + 1)
                    logger.info(
                        "Step %d/%d | Loss: %.4f | Acc: %.1f%%",
                        global_step, total_steps, avg_loss, acc,
                    )
                    losses.append({"step": global_step, "loss": avg_loss, "acc": acc})
                
                if global_step % args.save_every == 0:
                    save_projector(projector, f"{args.output_dir}/mm_projector-step{global_step}.safetensors")
                    
            except Exception as e:
                logger.warning("Error: %s", e)
                continue
        
        acc = epoch_correct / epoch_total * 100
        avg_loss = epoch_loss / max(1, args.num_samples // args.batch_size)
        logger.info("Epoch %d: Loss=%.4f, Acc=%.1f%%", epoch + 1, avg_loss, acc)
    
    # Save final
    save_projector(projector, f"{args.output_dir}/mm_projector-finetuned-0731.safetensors")
    
    # Save loss curve
    with open(f"{args.output_dir}/projector-training-loss.json", "w") as f:
        json.dump({"losses": losses, "config": vars(args)}, f, indent=2)
    
    logger.info("Training complete!")
    return projector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tower-path", default="/cache/huggingface/webbrain-0731-moonvit-src/vision_tower.safetensors")
    parser.add_argument("--projector-path", default="/cache/huggingface/webbrain-0731-moonvit-src/mm_projector.safetensors")
    parser.add_argument("--output-dir", default="/tmp/projector-v2")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    train(args)


if __name__ == "__main__":
    main()
