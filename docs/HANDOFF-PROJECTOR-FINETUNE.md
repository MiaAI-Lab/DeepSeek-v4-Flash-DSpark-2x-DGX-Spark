# Handoff: Projector Fine-tuning for DeepSeek-V4-Flash-0731

**Date:** 2026-08-10 (updated, same day — v3)  
**Repo:** `deepSeek-v4-Flash-DSpark`  
**Status:** **FIXED (v3).** Embedding-aligned projector deployed; live color gate 10/10 on all five colors with DSpark ON. Method doc: `docs/PROJECTOR-FINETUNE.md`.

---

## 0. V3 outcome (2026-08-10, supersedes the rest of this document)

**Live color gate (N=10, temp=0, DSpark MTP-6 ON): red 100%, black 100%, white 100%, green 100%, blue 100%; text-only `VISION_TEXT_OK`; spec-decode acceptance non-zero.** Evidence: `results/projector-v3-colors.json` (v3), `results/projector-v32-colors.json` (v3.2, currently deployed).

**Deployed: v3.2** (`mm_projector-v32-0731.safetensors`). v3.1 added 8 color
anchors (pink/brown/gray/beige/navy/olive/teal/maroon) — open-ended color
naming works for solids and neutral phrasings. v3.2 added object-context
anchors ("a pink sweater"); it did **not** fix garment-phrasing priors (the LM
answers "Blue" to "what color is the sweater?" even with no image — a text
prior, not a vision failure). See `docs/VISION.md` §Open-ended color naming.

What was actually wrong (three compounding bugs found this session):

1. **v2 trained on a fake tower.** `StandaloneTower` in `train-projector-v2.py`
   name/shape-matches almost none of the real MoonViT keys → the tower in
   training was effectively random-init. The classifier head it optimized is
   discarded at serving. Hence "fine-tuned" made things arbitrary/worse.
2. **The v2 fine-tune was never actually served.** The plugin resolves the
   projector from `DSV4_MOONVIT_PROJECTOR` / compose auto-discovery
   (`webbrain-0731-moonvit-src/mm_projector.safetensors`), **not** from the
   `mm_projector.safetensors` symlink in the overlay model dir that §5 below
   repointed. The v2 A/B numbers were baseline flakiness of the original
   projector. Head/worker had also diverged (head symlink → v2 file, worker →
   original), i.e. the two TP ranks ran *different* intended projectors.
3. **~18× embedding scale mismatch.** Original projector output row-norm ≈127
   vs 0731 `embed.weight` row-norm ≈7.3 — the WebBrain adapter is scaled for
   Kimi's input space, not 0731's. This (plus hue direction collapse) is why
   the LM misread image tokens.

**Fix (v3, `scripts/train-projector-v3.py`):** real frozen MoonViT tower
(loaded offline via single-process vLLM distributed init — recipe in
`docs/PROJECTOR-FINETUNE.md`), PatchMerger init from WebBrain weights, trained
with symmetric InfoNCE against **real 0731 `embed.weight`** caption anchors
(extracted by `scripts/extract-embed-table.py`) + color-word CE + log-norm
anchor. Data: synthetic colors (40%) + COCO val2017 captions (~10K). 3000
steps, ~75 min, trains alongside the live server.

Offline gate (original → v3): color-word retrieval 1/10 → 10/10; min hue
rel_l2 0.03 → 1.25; row-norm 127 → 7.35.

**Deployment (the real mechanism):**

```bash
# .env.dspark (synced to worker by the start script)
DSV4_MOONVIT_PROJECTOR=/cache/huggingface/webbrain-0731-moonvit-src/mm_projector-v3-0731.safetensors
# then ./stop-deepseek-v4-flash-dspark.sh && ./start-deepseek-v4-flash-dspark.sh
# verify in logs: MoonViT enabled: ... projector=.../mm_projector-v3-0731.safetensors
# and encode_image norm ~90-180 total per 100 tokens (not ~1300)
```

Files: candidate + checkpoints at `~/.cache/huggingface/projector-v3/`;
deployed copy at `webbrain-0731-moonvit-src/mm_projector-v3-0731.safetensors`
(both nodes). Originals untouched. Unit tests: 25 passed / 3 skipped
(env-overridable path resolution in `tests/test_moonvit_units.py`).

Remaining limits (2026-08-10, measured): fine-grained object identity on
natural images is still weak (giraffes → "elephants"; colors/scene/layout
correct). And **`reasoning_effort=max`/`high` over image tokens is unstable**:
the abliterated LM falls into scene-vocabulary repetition loops
(nondeterministic, penalties don't help; text-only max reasoning is stable).
Use `thinking=false` or `low` for image turns; for deep reasoning about image
content use the two-pass helper `scripts/vision-reason.py` (extract with
thinking off → reason at max over the description text). A true fix for
native max-effort image reasoning would need LM-in-the-loop training —
infeasible on one GB10 (157 GB FP8, vLLM-only arch).

Everything below is the pre-v3 handoff, kept for history.

---

## 1. What Was Done

### Created Files

| File | Purpose |
|------|---------|
| `scripts/train-projector.py` | Contrastive learning training (v1) - didn't work well |
| `scripts/train-projector-v2.py` | Color discrimination training (v2) - **working** |
| `scripts/eval-projector.py` | Evaluation script for comparing projectors |
| `tests/test_moonvit_units.py` | Updated with 5 new tests for fine-tuned projector |
| `docs/PROJECTOR-FINETUNE.md` | Documentation for the training pipeline |

### Training Results

**V2 Training (Color Discrimination):**
- Dataset: 10 synthetic colors × 5 caption templates
- Training: 5000 samples, 10 epochs, batch size 16
- Loss: 0.26 → 0.0 (converged)
- Accuracy: 93% → 100% (on training set)
- Time: ~4 minutes on single GPU

### Deployment Status

- ✅ Fine-tuned projector saved to: `/cache/huggingface/webbrain-0731-moonvit-src/mm_projector-finetuned-0731.safetensors`
- ✅ Symlink updated: `mm_projector.safetensors → mm_projector-finetuned-0731.safetensors`
- ✅ Server restarted with fine-tuned projector
- ✅ Vision pipeline confirmed working (images processed, responses generated)

---

## 2. Current State

### Server Status
- **Running:** `deepseek-v4-flash-vllm-dspark-1` on port 8888
- **Model:** `deepseek-v4-flash-0731-vision`
- **Vision:** Enabled, tower loaded (missing=0)
- **Projector:** Fine-tuned version active

### Color QA Results (with fine-tuned projector)

| Color | Baseline | Fine-tuned | Change |
|-------|----------|------------|--------|
| Red | 40% | **60%** | +20% ✅ |
| Black | 100% | **100%** | same |
| White | 80% | **80%** | same |
| Green | 40% | 20% | -20% ❌ |
| Blue | 0% | 0% | same |

### Known Issues

1. **Green got worse** - overfitting to synthetic training data
2. **Blue still 0%** - projector doesn't distinguish blue from other colors
3. **Red improved but not enough** - needs to reach ≥90% for success

---

## 3. Root Cause Analysis

### Why Training Worked But Results Are Mixed

1. **Synthetic data is too simple** - 10 solid colors don't represent real images
2. **Tower is a simplified copy** - not the real MoonViT, just a stand-in for gradient flow
3. **No real text encoder** - trained to classify colors, not to produce LM-friendly embeddings
4. **The real bottleneck** - the projector needs to produce embeddings that the **frozen 0731 LM** can distinguish, not just a classifier head

### What The Original Goal Document Says

From `docs/HANDOFF-VISION.md` §15:
> The only real fixes for hue QA: a retrained/fine-tuned projector for 0731, a different vision adapter, or WebBrain publishing a 0731-validated update.

The projector was trained by WebBrain for **Kimi (Moonshot)**, not DeepSeek-V4-Flash-0731. The LM reads luminance reliably but not hues because the embeddings are near-collinear.

---

## 4. What Needs To Be Done Next

### Priority 1: Better Training Data

The synthetic color dataset is insufficient. Need:

```bash
# Option A: LLaVA-Instruct-150K (captions subset)
# Option B: ShareGPT4V  
# Option C: COCO Captions 2017
```

Requirements:
- 10K-20K image-caption pairs
- Real images, not solid colors
- Short captions (1-2 sentences)

### Priority 2: Proper Tower

The standalone tower used for training is simplified. Need to either:
1. Load the real MoonViT tower (requires vLLM TP group)
2. Or pre-compute tower outputs and cache them for training

### Priority 3: Causal LM Loss

Instead of color classification, train with:
1. Image → tower → projector → embeddings
2. Feed embeddings + caption tokens to frozen LM
3. Compute cross-entropy loss on caption tokens
4. Backprop through projector only

This requires loading the full LM (155GB), which is the hard part.

---

## 5. Quick Commands

### Check Server Status
```bash
# Is server running?
docker ps | grep deepseek

# Check vision logs
docker logs deepseek-v4-flash-vllm-dspark-1 2>&1 | grep -E "MoonViT|projector"

# Test vision
python3 scripts/smoke-moonvit-colors.py --trials 3 --colors red,green,blue
```

### Run Training (V2)
```bash
docker exec deepseek-v4-flash-vllm-dspark-1 bash -c "
cd /opt/dsv4-moonvit-vllm && \
python3 /tmp/train-projector-v2.py \
    --tower-path /cache/huggingface/webbrain-0731-moonvit-src/vision_tower.safetensors \
    --projector-path /cache/huggingface/webbrain-0731-moonvit-src/mm_projector.safetensors \
    --output-dir /tmp/projector-v2 \
    --num-samples 5000 \
    --epochs 10
"
```

### Deploy Fine-tuned Projector
```bash
# Copy from container
docker cp deepseek-v4-flash-vllm-dspark-1:/tmp/projector-v2/mm_projector-finetuned-0731.safetensors \
    ~/.cache/huggingface/webbrain-0731-moonvit-src/

# Update symlink
cd ~/.cache/huggingface/dsv4-0731-moonvit/
ln -sf ../webbrain-0731-moonvit-src/mm_projector-finetuned-0731.safetensors \
       mm_projector.safetensors

# Restart server
cd ~/models/deepSeek-v4-Flash-DSpark
./stop-deepseek-v4-flash-dspark.sh
./start-deepseek-v4-flash-dspark.sh
```

### Revert to Original Projector
```bash
cd ~/.cache/huggingface/dsv4-0731-moonvit/
ln -sf ../webbrain-0731-moonvit-src/mm_projector.safetensors \
       mm_projector.safetensors

./stop-deepseek-v4-flash-dspark.sh
./start-deepseek-v4-flash-dspark.sh
```

---

## 6. Files Locations

### In Container
- Training script: `/tmp/train-projector-v2.py`
- Fine-tuned projector: `/tmp/projector-v2/mm_projector-finetuned-0731.safetensors`
- Loss curve: `/tmp/projector-v2/projector-training-loss.json`

### On Host
- Original projector: `~/.cache/huggingface/webbrain-0731-moonvit-src/mm_projector.safetensors`
- Fine-tuned projector: `~/.cache/huggingface/webbrain-0731-moonvit-src/mm_projector-finetuned-0731.safetensors`
- Active symlink: `~/.cache/huggingface/dsv4-0731-moonvit/mm_projector.safetensors`
- Training scripts: `~/models/deepSeek-v4-Flash-DSpark/scripts/train-projector*.py`
- Test results: `~/models/deepSeek-v4-Flash-DSpark/results/projector-*.json`

---

## 7. Success Criteria (from original goal)

| Criterion | Status |
|-----------|--------|
| Training runs ≥1000 steps | ✅ Done (3120 steps) |
| Loss decreases | ✅ 0.26 → 0.0 |
| Fine-tuned projector loads in serving | ✅ Deployed |
| Color QA improves | ⚠️ Red +20%, Green -20% |
| No regression in serving | ✅ Single/multi-image works |

**Overall:** Partial success. Training pipeline works, but needs better data and possibly the full LM for proper alignment.

---

## 8. Next Steps for New Owner

1. **Download real dataset** (COCO captions or LLaVA-Instruct)
2. **Pre-compute tower outputs** to avoid loading tower during training
3. **Load 4-bit LM** for causal language modeling loss
4. **Train with LM loss** instead of color classification
5. **Evaluate on full color gate** (red/black/white/green/blue × 10 trials)
6. **A/B test** original vs fine-tuned on real images (not just solid colors)

---

## 9. Contacts / Paths

| Role | Path |
|------|------|
| Training scripts | `~/models/deepSeek-v4-Flash-DSpark/scripts/train-projector*.py` |
| Plugin code | `~/models/deepSeek-v4-Flash-DSpark/plugins/dsv4_moonvit_vllm/` |
| Model overlay | `~/.cache/huggingface/dsv4-0731-moonvit/` |
| WebBrain artifacts | `~/.cache/huggingface/webbrain-0731-moonvit-src/` |
| Server logs | `docker logs deepseek-v4-flash-vllm-dspark-1` |
