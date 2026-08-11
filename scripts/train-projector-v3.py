#!/usr/bin/env python3
"""V3 projector fine-tune: align PatchMerger outputs with the REAL 0731 LM embedding space.

Why v3 exists
-------------
v1/v2 trained against a random stand-in tower and a throwaway classifier head,
so the projector was optimized for a signal that does not exist at serving time.
v3 uses only real components:

- **Real MoonViT tower** (frozen) via the plugin's ``load_tower_and_projector``
  (single-process vLLM distributed init; fails closed on missing weights).
- **Real 0731 ``embed.weight`` table** (frozen) extracted by
  ``scripts/extract-embed-table.py`` — caption anchors live in the exact input
  space the frozen LM consumes.
- Trainable: **PatchMerger only**, initialized from the original WebBrain weights.

Loss = symmetric InfoNCE(image ↔ caption embeddings)
     + color-word CE over the 10 color anchors
     + log-norm anchor toward the embed-table row-norm scale
       (0731 token rows are ~7.3 while raw projector rows are ~135 — an ~18x
       scale mismatch that is likely a first-order cause of the LM misreading
       image tokens).

Usage (inside the Anemll container):
    python3 /tmp/train-projector-v3.py \
        --model-dir /cache/huggingface/dsv4-0731-moonvit \
        --tower-path /cache/huggingface/webbrain-0731-moonvit-src/vision_tower.safetensors \
        --init-projector /cache/huggingface/webbrain-0731-moonvit-src/mm_projector.safetensors \
        --embed-table /cache/huggingface/webbrain-0731-moonvit-src/embed_table.safetensors \
        --coco-dir /cache/huggingface/datasets/coco \
        --steps 3000 --output-dir /cache/huggingface/projector-v3

Eval-only (offline gate, no training):
    python3 /tmp/train-projector-v3.py --eval-only \
        --projector-path <candidate.safetensors> --out <metrics.json>
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- colors ----
COLORS: dict[str, tuple[int, int, int]] = {
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
    # v3.1: extended vocabulary — open-ended color naming needs anchors for
    # common real-world colors (pic2.jpg hot-pink sweater was read as "blue"
    # partly because "pink" had no anchor at all).
    "pink": (255, 105, 180),
    "brown": (139, 69, 19),
    "gray": (128, 128, 128),
    "beige": (245, 245, 220),
    "navy": (0, 0, 128),
    "olive": (128, 128, 0),
    "teal": (0, 128, 128),
    "maroon": (128, 0, 0),
}
COLOR_NAMES = list(COLORS.keys())
GATE_COLORS = ["red", "black", "white", "green", "blue"]

# extra surface variants for anchor building (spelling aliases)
COLOR_ALIASES: dict[str, list[str]] = {"gray": ["grey", " grey", "Grey"]}

COLOR_CAPTION_TEMPLATES = [
    "a solid {c} image",
    "the image is {c}",
    "a {c} square",
    "a plain {c} background",
    "this picture is entirely {c}",
]
COLOR_SIZES = [224, 256, 320, 384, 448]

# v3.2: object-context anchors. The 0731 LM has strong text priors for garment
# questions ("what color is the sweater?" -> "Blue" even with NO image), which
# overrode the image signal. Bind hues to color words in object contexts with
# simple object-like renderings (colored shape on neutral background).
OBJECTS = [
    "sweater", "shirt", "dress", "hat", "bag", "car", "flower", "cup",
    "chair", "ball", "jacket", "skirt", "scarf", "shoe", "bird", "house",
]
OBJECT_TEMPLATES = [
    "a {c} {o}",
    "the {o} is {c}",
    "a photo of a {c} {o}",
    "this {o} is {c}",
]
NEUTRAL_BG = [(128, 128, 128), (220, 220, 220), (40, 40, 40), (245, 245, 240)]

SAVE_MAPPING = {
    "pre_norm.weight": "pre_norm.weight",
    "pre_norm.bias": "pre_norm.bias",
    "linear_1.weight": "proj.0.weight",
    "linear_1.bias": "proj.0.bias",
    "linear_2.weight": "proj.2.weight",
    "linear_2.bias": "proj.2.bias",
}


def save_projector(projector, path: str) -> None:
    from safetensors.torch import save_file

    state = projector.state_dict()
    out = {SAVE_MAPPING[k]: v.to(torch.bfloat16).contiguous() for k, v in state.items() if k in SAVE_MAPPING}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(out, path)
    logger.info("saved projector -> %s", path)


# ------------------------------------------------------------------ data ----
def make_color_sample(rng: random.Random) -> tuple[Image.Image, str, int]:
    name = rng.choice(COLOR_NAMES)
    rgb = COLORS[name]
    varied = tuple(max(0, min(255, c + rng.randint(-20, 20))) for c in rgb)
    size = rng.choice(COLOR_SIZES)

    if rng.random() < 0.5:
        # object-context: colored shape on a neutral background
        from PIL import ImageDraw

        bg = rng.choice(NEUTRAL_BG)
        img = Image.new("RGB", (size, size), color=bg)
        draw = ImageDraw.Draw(img)
        m = size // 5
        x0, y0 = m + rng.randint(0, m // 2), m + rng.randint(0, m // 2)
        x1, y1 = size - m + rng.randint(-m // 2, 0), size - m + rng.randint(-m // 2, 0)
        if rng.random() < 0.5:
            draw.rectangle([x0, y0, x1, y1], fill=varied)
        else:
            draw.ellipse([x0, y0, x1, y1], fill=varied)
        caption = rng.choice(OBJECT_TEMPLATES).format(c=name, o=rng.choice(OBJECTS))
    else:
        img = Image.new("RGB", (size, size), color=varied)
        caption = rng.choice(COLOR_CAPTION_TEMPLATES).format(c=name)
    return img, caption, COLOR_NAMES.index(name)


def load_coco_pairs(coco_dir: str, max_samples: int, rng: random.Random):
    """Return list of (image_path, caption) from COCO val2017 captions."""
    ann = Path(coco_dir) / "annotations" / "captions_val2017.json"
    img_root = Path(coco_dir) / "val2017"
    if not ann.is_file() or not img_root.is_dir():
        logger.warning("COCO not found under %s — synthetic colors only", coco_dir)
        return []
    data = json.loads(ann.read_text())
    id_to_file = {im["id"]: im["file_name"] for im in data["images"]}
    pairs = []
    for a in data["annotations"]:
        fn = id_to_file.get(a["image_id"])
        if fn:
            pairs.append((str(img_root / fn), a["caption"].strip().replace("\n", " ")))
    rng.shuffle(pairs)
    pairs = pairs[:max_samples]
    logger.info("COCO pairs loaded: %d", len(pairs))
    return pairs


# ----------------------------------------------------------------- model ----
def init_offline_vllm(port: int = 29781):
    """Single-process distributed env so vLLM's MoonViT linears construct."""
    from vllm.config import VllmConfig
    from vllm.config.vllm import set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    init_distributed_environment(
        world_size=1, rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}", local_rank=0,
    )
    vcfg = VllmConfig()
    ctx = set_current_vllm_config(vcfg)
    ctx.__enter__()
    initialize_model_parallel(tensor_model_parallel_size=1)
    return ctx


def encode_images(tower, projector, images, device, max_image_tokens=512):
    """images -> (list of (T_i, 4096) projected tensors). Tower frozen/no_grad."""
    from dsv4_moonvit_vllm.preprocess import pil_to_pixel_values_and_grid

    pvs, grids = [], []
    for img in images:
        pv, grid, _ = pil_to_pixel_values_and_grid(img, max_image_tokens=max_image_tokens)
        pvs.append(pv)
        grids.append(grid)
    pv_all = torch.cat(pvs, dim=0).to(device=device, dtype=torch.bfloat16)
    grid_t = torch.tensor(grids, dtype=torch.long, device=device)

    with torch.no_grad():
        vt_out = tower(pv_all, grid_t)
    chunks = list(vt_out) if isinstance(vt_out, (list, tuple)) else [vt_out]
    if len(chunks) != len(images):  # packed single tensor: fall back per-image
        if len(images) == 1:
            chunks = [vt_out]
        else:
            raise RuntimeError(f"tower returned {len(chunks)} chunks for {len(images)} images")

    proj_dtype = projector.pre_norm.weight.dtype
    return [projector(c.to(device=device, dtype=proj_dtype)) for c in chunks]


# ----------------------------------------------------------------- eval -----
@torch.no_grad()
def offline_eval(tower, projector, embed_table, color_anchor_vecs, device) -> dict:
    """Hue separation + color-word retrieval for the gate colors."""
    projector.eval()
    z = {}
    for name in COLOR_NAMES:
        img = Image.new("RGB", (256, 256), COLORS[name])
        emb = encode_images(tower, projector, [img], device)[0].float()
        z[name] = F.normalize(emb.mean(dim=0), dim=0)

    # rel_l2 between gate color pairs
    def rel_l2(a, b):
        return float(2 * (a - b).norm() / (a.norm() + b.norm() + 1e-8))

    pairs = {}
    for i, a in enumerate(GATE_COLORS):
        for b in GATE_COLORS[i + 1:]:
            pairs[f"{a}/{b}"] = round(rel_l2(z[a], z[b]), 4)

    anchors = F.normalize(color_anchor_vecs.float(), dim=1)  # (n_colors, 4096)
    retrieval = {}
    hits = 0
    for name in COLOR_NAMES:
        sims = anchors @ z[name]
        top = COLOR_NAMES[int(sims.argmax())]
        retrieval[name] = {"top1": top, "correct": top == name,
                           "margin": round(float(sims[COLOR_NAMES.index(name)] - sims.max()), 4)
                           if top != name else round(float(sims.max() - sims.topk(2).values[1]), 4)}
        hits += int(top == name)

    # output scale vs embed table scale
    sample = encode_images(tower, projector, [Image.new("RGB", (256, 256), COLORS["red"])], device)[0].float()
    row_norm = float(sample.norm(dim=-1).mean())

    projector.train()
    return {
        "rel_l2": pairs,
        "min_hue_rel_l2": min(v for k, v in pairs.items() if not ("black" in k or "white" in k)),
        "retrieval": retrieval,
        "retrieval_acc": round(hits / len(COLOR_NAMES), 3),
        "proj_row_norm": round(row_norm, 3),
    }


# ---------------------------------------------------------------- train -----
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default="/cache/huggingface/dsv4-0731-moonvit")
    ap.add_argument("--tower-path", default="/cache/huggingface/webbrain-0731-moonvit-src/vision_tower.safetensors")
    ap.add_argument("--init-projector", default="/cache/huggingface/webbrain-0731-moonvit-src/mm_projector.safetensors")
    ap.add_argument("--embed-table", default="/cache/huggingface/webbrain-0731-moonvit-src/embed_table.safetensors")
    ap.add_argument("--projector-path", default=None, help="eval-only: projector to score")
    ap.add_argument("--coco-dir", default="/cache/huggingface/datasets/coco")
    ap.add_argument("--coco-samples", type=int, default=10000)
    ap.add_argument("--color-fraction", type=float, default=0.4, help="fraction of each batch that is synthetic colors")
    ap.add_argument("--output-dir", default="/cache/huggingface/projector-v3")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--lambda-color", type=float, default=1.0)
    ap.add_argument("--lambda-norm", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=29781)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--out", default=None, help="eval metrics json (default: <output-dir>/v3-metrics.json)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ctx = init_offline_vllm(args.port)
    try:
        cfg = json.loads((Path(args.model_dir) / "config.json").read_text())

        from dsv4_moonvit_vllm.moonvit import load_tower_and_projector

        proj_for_eval = args.projector_path or args.init_projector
        tower, projector, meta = load_tower_and_projector(
            vision_dict=cfg["vision_config"],
            tower_path=args.tower_path,
            projector_path=proj_for_eval,
            device=device,
            dtype=torch.bfloat16,
        )
        if tower is None:
            raise RuntimeError(f"tower failed to load: {meta.get('tower_error')}")
        logger.info("tower loaded: %s", meta.get("tower"))

        from safetensors.torch import load_file
        embed_table = load_file(args.embed_table)["embed.weight"].to(device=device, dtype=torch.float32)
        embed_table.requires_grad_(False)
        target_log_norm = math.log(float(embed_table.norm(dim=-1).mean()))
        logger.info("embed table loaded; target log row-norm %.4f", target_log_norm)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

        def text_anchor_ids(text: str) -> list[int]:
            return tokenizer.encode(text, add_special_tokens=False)

        # color-word anchors: mean over surface variants; a variant may be
        # multi-token (e.g. "beige") — mean-pool its token embeddings.
        color_anchor_vecs = []
        for name in COLOR_NAMES:
            vecs = []
            variants = [name, f" {name}", name.capitalize(), f" {name.capitalize()}"]
            variants += COLOR_ALIASES.get(name, [])
            for variant in variants:
                v = text_anchor_ids(variant)
                if v:
                    vecs.append(embed_table[torch.tensor(v, device=device)].mean(dim=0))
            if not vecs:
                raise RuntimeError(f"no tokenizable variant for color {name}")
            color_anchor_vecs.append(F.normalize(torch.stack(vecs).mean(dim=0), dim=0))
        color_anchor_vecs = torch.stack(color_anchor_vecs)  # (n_colors, 4096)

        if args.eval_only:
            metrics = offline_eval(tower, projector.to(torch.float32), embed_table, color_anchor_vecs, device)
            out = args.out or str(Path(args.output_dir) / "v3-metrics.json")
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(json.dumps({"projector": proj_for_eval, "metrics": metrics}, indent=2))
            logger.info("eval metrics -> %s\n%s", out, json.dumps(metrics, indent=2))
            return 0

        # ---------------------------------------------------------- training
        projector = projector.to(torch.float32)
        projector.train()
        projector.requires_grad_(True)

        coco_pairs = load_coco_pairs(args.coco_dir, args.coco_samples, rng)

        optimizer = torch.optim.AdamW(projector.parameters(), lr=args.lr, weight_decay=0.01)

        def lr_at(step: int) -> float:
            if step < args.warmup:
                return args.lr * (step + 1) / args.warmup
            p = (step - args.warmup) / max(1, args.steps - args.warmup)
            return args.lr * 0.5 * (1 + math.cos(math.pi * p))

        history = []
        t0 = time.time()
        for step in range(args.steps):
            n_color = max(2, int(args.batch_size * args.color_fraction))
            n_coco = args.batch_size - n_color

            images, captions, color_labels = [], [], []
            for _ in range(n_color):
                img, cap, lab = make_color_sample(rng)
                images.append(img)
                captions.append(cap)
                color_labels.append(lab)
            coco_used = 0
            if coco_pairs and n_coco > 0:
                for _ in range(n_coco):
                    path, cap = coco_pairs[rng.randrange(len(coco_pairs))]
                    try:
                        img = Image.open(path).convert("RGB")
                    except Exception:
                        continue
                    images.append(img)
                    captions.append(cap)
                    coco_used += 1

            proj_outs = encode_images(tower, projector, images, device)
            z_img = torch.stack([F.normalize(o.float().mean(dim=0), dim=0) for o in proj_outs])

            cap_ids = [torch.tensor(text_anchor_ids(c)[:48], device=device) for c in captions]
            z_txt = torch.stack([
                F.normalize(embed_table[ids].mean(dim=0), dim=0) for ids in cap_ids
            ])

            # symmetric InfoNCE
            logits = z_img @ z_txt.T / args.tau
            labels = torch.arange(len(images), device=device)
            loss_nce = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

            # color-word CE on the color samples
            color_idx = torch.arange(n_color, device=device)
            color_logits = z_img[color_idx] @ color_anchor_vecs.T / args.tau
            loss_color = F.cross_entropy(color_logits, torch.tensor(color_labels, device=device))

            # log-norm anchor toward embed-table scale
            row_norms = torch.cat([o.float().norm(dim=-1) for o in proj_outs])
            loss_norm = ((row_norms.log() - target_log_norm) ** 2).mean()

            loss = loss_nce + args.lambda_color * loss_color + args.lambda_norm * loss_norm

            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
            optimizer.step()

            if (step + 1) % args.log_every == 0:
                with torch.no_grad():
                    acc = (color_logits.argmax(-1) == torch.tensor(color_labels, device=device)).float().mean()
                rec = {
                    "step": step + 1,
                    "loss": round(float(loss), 4),
                    "nce": round(float(loss_nce), 4),
                    "color": round(float(loss_color), 4),
                    "norm": round(float(loss_norm), 4),
                    "color_acc": round(float(acc), 3),
                    "mean_row_norm": round(float(row_norms.mean()), 2),
                    "lr": round(lr_at(step), 6),
                    "coco_in_batch": coco_used,
                }
                history.append(rec)
                logger.info("step %d/%d %s", step + 1, args.steps, rec)

            if (step + 1) % args.save_every == 0:
                save_projector(projector, str(Path(args.output_dir) / f"mm_projector-v3-step{step + 1}.safetensors"))

        final_path = str(Path(args.output_dir) / "mm_projector-v3-0731.safetensors")
        save_projector(projector, final_path)

        metrics = offline_eval(tower, projector, embed_table, color_anchor_vecs, device)
        out = args.out or str(Path(args.output_dir) / "v3-metrics.json")
        payload = {
            "config": vars(args),
            "final_projector": final_path,
            "train_seconds": round(time.time() - t0, 1),
            "history": history,
            "metrics": metrics,
        }
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(payload, indent=2))
        logger.info("done in %.1fs; metrics -> %s\n%s", time.time() - t0, out, json.dumps(metrics, indent=2))
        return 0
    finally:
        ctx.__exit__(None, None, None)


if __name__ == "__main__":
    raise SystemExit(main())
