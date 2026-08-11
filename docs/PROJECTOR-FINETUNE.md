# Projector Fine-tuning for DeepSeek-V4-Flash-0731

## V3 (current, working): embedding-space alignment against the real 0731 LM

**Status: deployed, all color gates pass (10/10 × 5 colors, DSpark ON).**

V3 fixes the two fatal flaws of v1/v2 (random stand-in tower; discarded
classifier head) by training the PatchMerger against **only real components**:

- **Real MoonViT tower** (frozen), loaded offline via the plugin's
  `load_tower_and_projector` inside a single-process vLLM distributed context
  (`init_distributed_environment` + `set_current_vllm_config(VllmConfig())` +
  `initialize_model_parallel(1)`).
- **Real 0731 `embed.weight` table** (frozen), extracted by
  `scripts/extract-embed-table.py` (129280×4096, read lazily from shard 1/45 —
  no full model load).
- **Trainable:** PatchMerger only, initialized from the original WebBrain weights.

Loss = symmetric InfoNCE (image ↔ caption embeddings, cosine, τ=0.07)
     + color-word CE over 10 color anchors
     + log-norm anchor toward the embed-table row-norm scale.

The norm anchor matters: the original WebBrain projector emits rows of norm
**~127** while 0731 token embeddings sit at **~7.3** — an ~18× scale mismatch
(it was trained for Kimi's embedding space). V3 converges to row-norm ≈7.7.

Data: synthetic solid colors (10 colors × jitter/sizes × 5 caption templates,
40% of each batch) + COCO val2017 captions (~10K pairs). 3000 steps, batch 24,
AdamW lr 1e-4 cosine, ~75 min on one GB10 alongside the running server.

Results (live gate, N=10, temp=0, DSpark MTP-6 ON):

| Metric | Original | v2 (never actually served — see below) | V3 |
| --- | --- | --- | --- |
| red | 60% | – | **100%** |
| black | 100% | – | **100%** |
| white | 70% | – | **100%** |
| green | 40% | – | **100%** |
| blue | 0% | – | **100%** |
| Offline color-word retrieval | 1/10 | – | **10/10** |
| Min hue rel_l2 (red/green/blue) | ~0.03 | – | **1.25** |
| Projector row norm | 127 | – | **7.35** |

Evidence: `results/projector-v3-colors.json` (live gate + DSpark counters),
`~/.cache/huggingface/projector-v3/v3-metrics.json` (offline metrics + training
curve), `~/.cache/huggingface/projector-v3/v3-metrics-original.json` (offline
baseline of the original projector measured the same way).

### Deploy / revert (v3)

The serve resolves the projector from `DSV4_MOONVIT_PROJECTOR` (or compose
auto-discovery: `webbrain-0731-moonvit-src/mm_projector.safetensors` first).
**The `mm_projector.safetensors` symlink inside the overlay model dir is NOT
read by the plugin** — earlier "deployments" that only repointed that symlink
(including the v2 fine-tune) never actually changed the served weights.

```bash
# Deploy v3 (both nodes get .env.dspark via the start script's scp)
echo 'DSV4_MOONVIT_PROJECTOR=/cache/huggingface/webbrain-0731-moonvit-src/mm_projector-v3-0731.safetensors' >> .env.dspark
./stop-deepseek-v4-flash-dspark.sh && ./start-deepseek-v4-flash-dspark.sh
# Verify: serve log must show
#   MoonViT enabled: ... projector=.../mm_projector-v3-0731.safetensors
#   encode_image ... norm= ~90-180 total (~7-13 per row) instead of ~1300

# Revert to the original WebBrain projector
sed -i '/^DSV4_MOONVIT_PROJECTOR=/d' .env.dspark
./stop-deepseek-v4-flash-dspark.sh && ./start-deepseek-v4-flash-dspark.sh
```

### Run v3 training

```bash
docker cp scripts/extract-embed-table.py deepseek-v4-flash-vllm-dspark-1:/tmp/
docker cp scripts/train-projector-v3.py  deepseek-v4-flash-vllm-dspark-1:/tmp/
docker exec deepseek-v4-flash-vllm-dspark-1 python3 /tmp/extract-embed-table.py
docker exec deepseek-v4-flash-vllm-dspark-1 python3 -u /tmp/train-projector-v3.py \
    --steps 3000 --output-dir /cache/huggingface/projector-v3
# Offline gate only (no training):
docker exec deepseek-v4-flash-vllm-dspark-1 python3 -u /tmp/train-projector-v3.py \
    --eval-only --projector-path <candidate.safetensors> --out /tmp/metrics.json
```

Known remaining limit: fine-grained object identity on natural images is still
weak (e.g. giraffes read as "elephants"; coarse scene, layout and colors are
right). Colors, luminance, and coarse scene description are reliable. The next
lever would be causal-LM-loss training, which needs the frozen 0731 LM in the
loop — infeasible on one GB10 node (157 GB FP8, vLLM-only architecture).

---

## V1/V2 (superseded, kept for reference)

## Overview

This directory contains scripts to fine-tune the PatchMerger projector for DeepSeek-V4-Flash-0731 using contrastive learning. The WebBrain projector was originally trained for Kimi (Moonshot) and produces near-collinear embeddings for different hues, causing the LM to hallucinate colors.

## Problem

The current projector outputs embeddings where:
- Red vs Green: rel_l2 ≈ 0.03 (nearly identical)
- Blue image suppresses "blue" logit ~0.5 below "red"
- LM reads luminance reliably but not hues

## Solution

Fine-tune ONLY the projector using contrastive learning to align image features with text embeddings. The tower and text encoder are frozen; only the projector weights are updated.

## Files Created

### 1. `scripts/train-projector.py`

Training script using contrastive learning:
- Uses synthetic color dataset (10 colors, 5 caption templates)
- Loads MoonViT tower (frozen)
- Loads PatchMerger projector (trainable)
- Uses simple text encoder (frozen)
- Trains with InfoNCE contrastive loss
- Saves `mm_projector-finetuned-0731.safetensors` as drop-in replacement

### 2. `scripts/eval-projector.py`

Evaluation script that:
- Compares original vs fine-tuned projector
- Runs color QA gate (red/black/white/green/blue)
- Reports pass rates and deltas
- Saves results to JSON

### 3. Updated `tests/test_moonvit_units.py`

Added `TestProjectorFinetuned` class with 5 new tests:
- `test_projector_finetuned_loads`: Verifies file loads correctly
- `test_projector_finetuned_shape`: Verifies output shapes (1-512 tokens)
- `test_projector_finetuned_deterministic`: Verifies determinism
- `test_projector_finetuned_param_count`: Verifies param count matches (40,119,040)
- `test_projector_finetuned_differs_from_original`: Verifies training occurred

## Usage

### Inside Docker Container

```bash
# 1. Train the projector (15-30 minutes on single GPU)
docker exec deepseek-v4-flash-vllm-dspark-1 bash -c '
cd /opt/dsv4-moonvit-vllm && \
python3 scripts/train-projector.py \
    --tower-path /cache/huggingface/webbrain-0731-moonvit-src/vision_tower.safetensors \
    --projector-path /cache/huggingface/webbrain-0731-moonvit-src/mm_projector.safetensors \
    --output-dir /cache/huggingface \
    --num-samples 10000 \
    --epochs 5 \
    --batch-size 16 \
    --lr 1e-4 \
    --bf16
'

# 2. Run tests
docker exec deepseek-v4-flash-vllm-dspark-1 bash -c '
cd /opt/dsv4-moonvit-vllm && \
python3 -m pytest tests/test_moonvit_units.py -v -k "Finetuned"
'
```

### Deploy Fine-tuned Projector

```bash
# Copy fine-tuned projector to serving directory
cp /cache/huggingface/mm_projector-finetuned-0731.safetensors \
   ~/.cache/huggingface/webbrain-0731-moonvit-src/

# Update symlink in model dir
cd ~/.cache/huggingface/dsv4-0731-moonvit/
ln -sf ../webbrain-0731-moonvit-src/mm_projector-finetuned-0731.safetensors \
       mm_projector.safetensors

# Restart serving stack
./stop-deepseek-v4-flash-dspark.sh
./start-deepseek-v4-flash-dspark.sh
```

## Training Details

- **Dataset**: Synthetic color dataset (10 colors × 5 caption templates = 50 unique pairs)
- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.01)
- **Scheduler**: Cosine annealing with linear warmup (100 steps)
- **Batch size**: 16
- **Epochs**: 5
- **Loss**: InfoNCE contrastive loss (temperature=0.07)
- **Mixed precision**: BF16 training

## Expected Loss Curve

- Start: ~2.3 (random baseline = ln(10) for 10 classes)
- End: ~2.0-2.1 (indicating some learning)

## Hardware Requirements

- Single GPU with ≥8GB VRAM
- Training time: 15-30 minutes

## How It Works

1. **Tower encoding**: MoonViT tower encodes images into patch features
2. **Projection**: PatchMerger projects features to LM embedding space
3. **Contrastive learning**: Train projector to match image embeddings with text embeddings of their captions
4. **InfoNCE loss**: Maximize similarity between matched image-text pairs, minimize for unmatched

## Technical Notes

### Standalone Tower

Since the full MoonViT tower requires vLLM's tensor parallel group, we use a standalone tower implementation:
- Projects patches to hidden_size (1152)
- Applies 4 transformer blocks
- Merges 2x2 patches for the projector
- Loads weights from the same safetensors file

### Text Encoder

For contrastive learning, we use a simple frozen text encoder:
- Embedding layer (129280 vocab × 4096 dim)
- Average pooling over tokens
- Initialized randomly but frozen during training

## Limitations

1. **Synthetic data**: Uses simple color images, not real-world images
2. **Simple text encoder**: Not the full LM, so alignment is approximate
3. **No causal LM loss**: Only contrastive loss, not next-token prediction

## Future Improvements

1. Use real image-caption datasets (COCO, ShareGPT4V)
2. Load full LM for causal language modeling loss
3. Multi-task training: contrastive + causal LM loss
4. Curriculum learning: start with simple colors, progress to complex images

## Rollback

If the fine-tuned projector causes issues:

```bash
# Restore original projector
cd ~/.cache/huggingface/dsv4-0731-moonvit/
ln -sf ../webbrain-0731-moonvit-src/mm_projector.safetensors \
       mm_projector.safetensors

# Restart
./stop-deepseek-v4-flash-dspark.sh
./start-deepseek-v4-flash-dspark.sh
```

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python3 scripts/train-projector.py --batch-size 8 ...
```

### Tower Loading Fails

The standalone tower loads weights from safetensors but uses a simplified architecture. If you see shape mismatches, check the tower config.

### Projector Not Loading

Check logs for:
```
Loaded projector to ... (6 tensors)
```

If you see errors, verify the safetensors file exists and has the correct keys.
