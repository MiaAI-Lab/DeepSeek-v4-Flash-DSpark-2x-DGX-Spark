# Goal: Fine-tune MoonViT PatchMerger projector for DeepSeek-V4-Flash-0731

## Context

Repo: `/home/mia/models/deepSeek-v4-Flash-DSpark`  
Model: `DeepseekV4MoonVitForCausalLM` — native MoonViT vision on 0731  
Plugin: `plugins/dsv4_moonvit_vllm/`  
Runtime: 2× DGX Spark, TP=2, Anemll `dspark-vllm-gx10:0.1.1`

The vision tower (`vision_tower.safetensors`, 833MB) encodes images fine — embeddings are distinguishable. The problem is the **projector** (`mm_projector.safetensors`, 80MB, ~14M params): it was trained by WebBrain for Kimi (Moonshot), not for DeepSeek-V4-Flash-0731. The LM gets near-collinear signals for different hues (red↔green rel_l2 0.03) and hallucinates/guesses.

## Goal

Build a training script that fine-tunes **only the projector** on image-caption data, with the tower and 0731 LM frozen. Save a new `mm_projector-finetuned-0731.safetensors` that can be dropped into the existing serving pipeline.

## Read first

- `plugins/dsv4_moonvit_vllm/dsv4_moonvit_vllm/projector.py` — PatchMerger architecture (LN→2×2→Linear×2)
- `plugins/dsv4_moonvit_vllm/dsv4_moonvit_vllm/moonvit.py` — tower load + encode
- `plugins/dsv4_moonvit_vllm/dsv4_moonvit_vllm/preprocess.py` — NaViT resize/pad/normalize
- `plugins/dsv4_moonvit_vllm/dsv4_moonvit_vllm/model.py` — `embed_multimodal` (tower→projector→embeddings)
- `tests/test_moonvit_units.py` — existing projector tests
- `docs/HANDOFF-VISION.md` §15 — root cause analysis

## What to build

### 1. `scripts/train-projector.py` — standalone training script

**Data**: Download a small image-caption dataset. Options (in order of preference):

- [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) captions subset (just the short captions, not full instructions)
- [ShareGPT4V](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V)
- Fallback: COCO captions 2017 train (~118K images, standard HF dataset)
- Use only 10–20K samples to keep training fast (1–2 hours on 1 GPU)

**Preprocessing**: Use the existing `pil_to_pixel_values_and_grid` from `preprocess.py` for images. For text, use the 0731 tokenizer with chat template (image token + caption).

**Training loop**:

```python
for image, caption in dataset:
    # 1. Encode image through frozen tower
    pixel_values, grid, _ = pil_to_pixel_values_and_grid(image)
    with torch.no_grad():
        tower_out = vision_tower(pixel_values, grid)  # frozen

    # 2. Project through trainable projector
    projected = projector(tower_out)  # trainable

    # 3. Build input: <image> tokens (replaced by projected embeddings) + caption
    # 4. Forward through frozen 0731 LM with inputs_embeds
    # 5. Cross-entropy loss on caption tokens only (not image positions)
    # 6. Backprop through projector only
    # 7. AdamW, lr=1e-4, cosine schedule, ~3 epochs
```

**Key constraints**:

- Tower MUST be on GPU, frozen (`requires_grad_(False)`)
- LM MUST be on GPU, frozen
- Only projector params have `requires_grad_(True)`
- Use existing `load_tower_and_projector()` from `moonvit.py`
- Use existing `PatchMerger` from `projector.py`
- Use `pil_to_pixel_values_and_grid()` from `preprocess.py` (same resize/normalize as serving)
- BF16 training
- Gradient accumulation if batch size 1–2 per GPU
- Log loss every 100 steps
- Save checkpoint every 1000 steps + final checkpoint

**Model loading**: Load 0731 the same way the serving pipeline does. The model dir is `/cache/huggingface/dsv4-0731-moonvit` (overlay with symlinks). Use `transformers.AutoModelForCausalLM` or the vLLM model class — either works as long as you freeze it.

### 2. Validation script — `scripts/eval-projector.py`

- Load original projector and fine-tuned projector
- Run the same 10-trial color gate as `smoke-moonvit-colors.py` but offline (encode images, run through projector, feed to LM, compare logits)
- OR just run against the live server: swap `mm_projector.safetensors` → restart → `python3 scripts/smoke-moonvit-colors.py --trials 10`
- Save results to `results/projector-finetuned-colors.json`

### 3. Tests — add to `tests/test_moonvit_units.py`

- `test_projector_finetuned_loads`: new projector file loads, shapes match, missing=0
- `test_projector_finetuned_shape`: forward pass produces correct (T, 4096) output

## Deliverables

1. `scripts/train-projector.py` — runnable training script
2. `scripts/eval-projector.py` — evaluation script
3. `mm_projector-finetuned-0731.safetensors` — new projector weights (save alongside originals, not overwriting)
4. `results/projector-training-loss.json` — loss curve
5. `results/projector-finetuned-colors.json` — before/after color comparison
6. Updated `tests/test_moonvit_units.py` with new tests
7. Updated `docs/VISION.md` and `results/moonvit-native-vision.md` with results

## What NOT to do

- Do NOT fine-tune the tower or the LM — projector only
- Do NOT use a different tokenizer or chat template than 0731
- Do NOT change the serving pipeline code (processor.py, model.py, etc.) — the new projector must be a drop-in replacement
- Do NOT overwrite `mm_projector.safetensors` — save as `mm_projector-finetuned-0731.safetensors`
- Do NOT commit/push to git
- Do NOT restart the serving stack (user is AFK, service should stay up)
- Do NOT `rm -rf` anything
- Do NOT download datasets to `/tmp` (use `/cache/huggingface` or the repo dir)

## Success criteria

- Training runs without errors for ≥1000 steps
- Loss decreases (should start ~2–4 for causal LM, converge to ~1.5–2.5)
- Fine-tuned projector loads in the serving pipeline (swap file → restart → API works)
- Color QA pass rates improve vs baseline (red ≥60% would be a meaningful gain over current 40%)
- Single-image and multi-image serving still works (no regression in pipeline)

## Hardware notes

- Single GPU is fine for training (don't need TP=2)
- ~8GB VRAM for projector + tower + small LM forward (use 4-bit quantized LM if OOM)
- Training should take 1–3 hours on a single A100/L40S/DGX Spark GPU
- Dataset download: ~2–5GB for COCO captions subset
