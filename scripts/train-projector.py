#!/usr/bin/env python3
"""Fine-tune PatchMerger projector for DeepSeek-V4-Flash-0731.

Trains ONLY the projector on image-caption data using contrastive learning.
Tower and projector are trained to align image features with text embeddings.

Usage (inside Docker container):
    python scripts/train-projector.py \\
        --tower-path /cache/huggingface/webbrain-0731-moonvit-src/vision_tower.safetensors \\
        --projector-path /cache/huggingface/webbrain-0731-moonvit-src/mm_projector.safetensors \\
        --output-dir /cache/huggingface \\
        --num-samples 10000 \\
        --epochs 5 \\
        --batch-size 16 \\
        --lr 1e-4 \\
        --bf16
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add plugin to path for imports
ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "dsv4_moonvit_vllm"
sys.path.insert(0, str(PLUGIN))

from dsv4_moonvit_vllm.preprocess import pil_to_pixel_values_and_grid
from dsv4_moonvit_vllm.projector import PatchMerger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simple Text Encoder (frozen, uses word embeddings)
# ---------------------------------------------------------------------------

class SimpleTextEncoder(nn.Module):
    """Simple text encoder that uses word frequency to create embeddings.
    
    This is a lightweight alternative to the full LM for contrastive learning.
    """
    
    def __init__(self, vocab_size: int = 129280, embed_dim: int = 4096, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.embed_dim = embed_dim
        # Create a simple embedding that maps token IDs to vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Initialize with random but normalized vectors
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.embedding.weight.requires_grad = False  # Frozen
        # Store dtype for later
        self._dtype = dtype
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode token IDs to embeddings and average pool.
        
        Args:
            token_ids: (batch, seq_len) token IDs
            
        Returns:
            (batch, embed_dim) pooled embeddings (same dtype as embedding weights)
        """
        # Get embeddings
        embeds = self.embedding(token_ids)  # (batch, seq_len, embed_dim)
        
        # Average pool (mask out padding if needed)
        mask = (token_ids != 0).float().unsqueeze(-1)  # (batch, seq_len, 1)
        mask = mask.to(dtype=embeds.dtype)  # Match dtype
        embeds = embeds * mask
        sum_embeds = embeds.sum(dim=1)  # (batch, embed_dim)
        count = mask.sum(dim=1).clamp(min=1)  # (batch, 1)
        
        return sum_embeds / count


# ---------------------------------------------------------------------------
# Dataset: Synthetic color images with captions
# ---------------------------------------------------------------------------

class SyntheticColorDataset:
    """Synthetic dataset of solid color images with captions."""
    
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
    
    CAPTION_TEMPLATES = [
        "This is a solid {color} image.",
        "The image shows the color {color}.",
        "A solid {color} colored square.",
        "{color} color fills the entire image.",
        "This is a {color} picture.",
    ]
    
    def __init__(
        self,
        num_samples: int = 10000,
        image_size: int = 256,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.color_names = list(self.COLORS.keys())
        
    def __iter__(self):
        from PIL import Image
        import random
        
        color_idx = 0
        for i in range(self.num_samples):
            # Cycle through colors with some randomness
            color_name = self.color_names[color_idx % len(self.color_names)]
            rgb = self.COLORS[color_name]
            
            # Add slight color variation (±10)
            varied_rgb = tuple(max(0, min(255, c + random.randint(-10, 10))) for c in rgb)
            
            # Create image
            img = Image.new("RGB", (self.image_size, self.image_size), color=varied_rgb)
            
            # Create caption
            template = random.choice(self.CAPTION_TEMPLATES)
            caption = template.format(color=color_name)
            
            yield img, caption, color_name
            color_idx += 1


# ---------------------------------------------------------------------------
# Save projector
# ---------------------------------------------------------------------------

def save_projector_safetensors(projector: PatchMerger, path: str | Path) -> None:
    """Save projector weights as safetensors."""
    from safetensors.torch import save_file
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    state = projector.state_dict()
    webbrain_state = {}
    mapping = {
        "pre_norm.weight": "pre_norm.weight",
        "pre_norm.bias": "pre_norm.bias",
        "linear_1.weight": "proj.0.weight",
        "linear_1.bias": "proj.0.bias",
        "linear_2.weight": "proj.2.weight",
        "linear_2.bias": "proj.2.bias",
    }
    
    for torch_key, webbrain_key in mapping.items():
        if torch_key in state:
            webbrain_state[webbrain_key] = state[torch_key]
    
    save_file(webbrain_state, str(path))
    logger.info("Saved projector to %s (%d tensors)", path, len(webbrain_state))


# ---------------------------------------------------------------------------
# Standalone Tower
# ---------------------------------------------------------------------------

class StandaloneTower(nn.Module):
    """Standalone MoonViT tower for training.
    
    Input from pil_to_pixel_values_and_grid is already (N, 3, 14, 14) patches.
    We just need to project to hidden_size and apply transformer blocks.
    """
    
    def __init__(self, hidden_size: int = 1152, patch_size: int = 14):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        
        # Project patches to hidden_size (input is already 3x14x14 patches)
        self.patch_proj = nn.Linear(3 * patch_size * patch_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        
        # A few transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=16,
                dim_feedforward=4304,
                batch_first=True,
            )
            for _ in range(4)  # Use 4 blocks for speed
        ])
        
    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            pixel_values: (N, 3, 14, 14) pre-extracted patches where N = grid_h * grid_w
            grid_thw: (1, 3) grid [t, h, w]
        """
        n_patches = pixel_values.shape[0]
        grid_h = grid_thw[0, 1].item()
        grid_w = grid_thw[0, 2].item()
        
        # Verify patch count matches grid
        assert n_patches == grid_h * grid_w, f"Expected {grid_h * grid_w} patches, got {n_patches}"
        
        # Flatten each patch: (N, 3, 14, 14) -> (N, 588)
        x = pixel_values.flatten(1)  # (N, 588)
        
        # Project to hidden_size: (N, 588) -> (N, 1152)
        x = self.patch_proj(x)  # (N, hidden_size)
        
        # Reshape to spatial grid: (N, hidden) -> (grid_h, grid_w, hidden)
        x = x.view(grid_h, grid_w, self.hidden_size)
        
        # Merge 2x2 patches: (grid_h, grid_w, hidden) -> (grid_h//2, grid_w//2, 4*hidden)
        merged_h = grid_h // 2
        merged_w = grid_w // 2
        x = x.reshape(merged_h, 2, merged_w, 2, self.hidden_size)
        x = x.permute(0, 2, 1, 3, 4)  # (merged_h, merged_w, 2, 2, hidden)
        x = x.reshape(merged_h * merged_w, 4 * self.hidden_size)  # (merged_h*merged_w, 4*hidden)
        
        # Reshape to (N, 4, hidden_size) for the projector
        x = x.view(merged_h * merged_w, 4, self.hidden_size)
        
        return x  # (merged_h*merged_w, 4, hidden_size)
    
    def load_safetensors(self, path: str | Path, device: torch.device, dtype: torch.dtype):
        """Load weights from safetensors."""
        from safetensors.torch import load_file
        
        path = Path(path)
        if not path.is_file():
            logger.warning("Tower weights not found: %s", path)
            return 0
        
        state = load_file(str(path), device="cpu")
        own_state = self.state_dict()
        
        loaded = 0
        for key, tensor in state.items():
            clean_key = key.removeprefix("vision_tower.")
            if clean_key in own_state:
                if own_state[clean_key].shape == tensor.shape:
                    own_state[clean_key] = tensor.to(device=device, dtype=dtype)
                    loaded += 1
        
        self.load_state_dict(own_state)
        self.to(device=device, dtype=dtype)
        self.eval()
        
        return loaded


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    """Main training loop using contrastive learning."""
    from PIL import Image
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    
    logger.info("Device: %s, dtype: %s", device, dtype)
    
    # Load projector (trainable)
    logger.info("Loading projector from %s", args.projector_path)
    projector = PatchMerger()
    projector.load_webbrain_safetensors(args.projector_path, device="cpu", dtype=dtype)
    projector = projector.to(device=device)
    projector.train()
    projector.requires_grad_(True)
    
    # Load tower (frozen)
    logger.info("Loading tower from %s", args.tower_path)
    tower = StandaloneTower(hidden_size=1152, patch_size=14)
    tower.load_safetensors(args.tower_path, device, dtype)
    tower.eval()
    tower.requires_grad_(False)
    
    # Simple text encoder (frozen)
    text_encoder = SimpleTextEncoder(vocab_size=129280, embed_dim=4096, dtype=dtype)
    text_encoder = text_encoder.to(device=device, dtype=dtype)
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Learning rate scheduler
    total_steps = (args.num_samples // args.batch_size) * args.epochs
    warmup_steps = min(total_steps // 10, 100)
    
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Dataset
    dataset = SyntheticColorDataset(
        num_samples=args.num_samples,
        image_size=256,
    )
    
    # Simple tokenizer (just split by spaces)
    def simple_tokenize(text: str) -> torch.Tensor:
        """Simple tokenization for contrastive learning."""
        # Just use character-level encoding for simplicity
        tokens = [ord(c) % 129280 for c in text.lower()]
        # Pad or truncate to fixed length
        tokens = tokens[:64]
        tokens = tokens + [0] * (64 - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)
    
    # Training loop
    logger.info("Starting contrastive training for %d epochs", args.epochs)
    
    global_step = 0
    epoch_losses = []
    step_losses = []
    
    for epoch in range(args.epochs):
        logger.info("Epoch %d/%d", epoch + 1, args.epochs)
        
        epoch_loss = 0.0
        batch_count = 0
        batch_images = []
        batch_captions = []
        
        for img_idx, (image, caption, color_name) in enumerate(dataset):
            if img_idx >= args.num_samples:
                break
            
            try:
                # Process image
                pixel_values, grid, _ = pil_to_pixel_values_and_grid(image, max_image_tokens=512)
                batch_images.append((pixel_values, grid))
                batch_captions.append(caption)
                
                # Process batch
                if len(batch_images) >= args.batch_size:
                    # Encode images through tower + projector
                    image_embeds = []
                    for pv, grid in batch_images:
                        # pv is already (N, 3, 14, 14) - no need to unsqueeze
                        pv = pv.to(device=device, dtype=dtype)
                        grid_t = torch.tensor([grid], dtype=torch.long, device=device)
                        
                        with torch.no_grad():
                            tower_out = tower(pv, grid_t)
                        
                        proj_out = projector(tower_out.to(dtype=dtype))
                        # Pool over tokens
                        img_embed = proj_out.mean(dim=0)  # (hidden_dim,)
                        image_embeds.append(img_embed)
                    
                    image_embeds = torch.stack(image_embeds)  # (batch, hidden_dim)
                    
                    # Encode captions
                    token_ids = torch.stack([simple_tokenize(c) for c in batch_captions])
                    token_ids = token_ids.to(device)
                    
                    with torch.no_grad():
                        text_embeds = text_encoder(token_ids)  # (batch, hidden_dim)
                    
                    # Contrastive loss (InfoNCE)
                    # Normalize embeddings
                    image_embeds = F.normalize(image_embeds, dim=-1)
                    text_embeds = F.normalize(text_embeds, dim=-1)
                    
                    # Similarity matrix
                    logits = torch.mm(image_embeds, text_embeds.t()) * args.temperature
                    
                    # Labels: diagonal (each image matches its caption)
                    labels = torch.arange(logits.shape[0], device=device)
                    
                    # Cross entropy loss (symmetric)
                    loss_i2t = F.cross_entropy(logits, labels)
                    loss_t2i = F.cross_entropy(logits.t(), labels)
                    loss = (loss_i2t + loss_t2i) / 2
                    
                    # Backprop
                    loss.backward()
                    
                    # Optimizer step
                    torch.nn.utils.clip_grad_norm_(projector.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    epoch_loss += loss.item()
                    batch_count += 1
                    step_losses.append({"step": global_step, "loss": loss.item()})
                    
                    if global_step % args.log_every == 0:
                        lr = scheduler.get_last_lr()[0]
                        logger.info(
                            "Step %d/%d | Loss: %.4f | LR: %.2e",
                            global_step, total_steps, loss.item(), lr,
                        )
                    
                    # Save checkpoint
                    if global_step % args.save_every == 0:
                        ckpt_path = Path(args.output_dir) / f"mm_projector-step{global_step}.safetensors"
                        save_projector_safetensors(projector, ckpt_path)
                    
                    # Reset batch
                    batch_images = []
                    batch_captions = []
                    
                    # Periodic cleanup
                    if global_step % 50 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()
                        
            except Exception as e:
                logger.warning("Error at sample %d: %s", img_idx, e, exc_info=True)
                optimizer.zero_grad()
                batch_images = []
                batch_captions = []
                continue
        
        avg_epoch_loss = epoch_loss / max(1, batch_count)
        epoch_losses.append({"epoch": epoch + 1, "loss": avg_epoch_loss})
        logger.info("Epoch %d avg loss: %.4f", epoch + 1, avg_epoch_loss)
    
    # Save final projector
    final_path = Path(args.output_dir) / "mm_projector-finetuned-0731.safetensors"
    save_projector_safetensors(projector, final_path)
    
    # Save loss curve
    loss_path = Path(args.output_dir) / "projector-training-loss.json"
    loss_data = {
        "epoch_losses": epoch_losses,
        "step_losses": step_losses,
        "config": vars(args),
    }
    with open(loss_path, "w") as f:
        json.dump(loss_data, f, indent=2)
    logger.info("Saved loss curve: %s", loss_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tower-path",
        default="/cache/huggingface/webbrain-0731-moonvit-src/vision_tower.safetensors",
        help="Path to vision tower weights",
    )
    parser.add_argument(
        "--projector-path",
        default="/cache/huggingface/webbrain-0731-moonvit-src/mm_projector.safetensors",
        help="Path to original projector weights",
    )
    parser.add_argument(
        "--output-dir",
        default="/cache/huggingface",
        help="Directory to save fine-tuned projector",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of image-caption pairs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Max gradient norm",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature for contrastive loss",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use BF16 training",
    )
    parser.add_argument(
        "--no-bf16",
        dest="bf16",
        action="store_false",
        help="Disable BF16",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log loss every N steps",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save checkpoint every N steps",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
