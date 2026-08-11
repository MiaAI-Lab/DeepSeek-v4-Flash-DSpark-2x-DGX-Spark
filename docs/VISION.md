# Native MoonViT vision (DeepSeek-V4-Flash-0731) — RETIRED

> **2026-08-11: this lane is retired.** The production setup is now a local
> **Qwen3-VL-4B sidecar** (`docker-compose.vl-sidecar.yml`, port 8889) with the
> 0731 serve text-only; see `README.md` §Vision and `scripts/vision-reason.py`.
> Reasons: even after the v3.x projector fine-tunes (color gate 10/10), the
> adapter ceiling remained — weak fine-grained recognition, LM text priors
> overriding image evidence on some phrasings, and unstable max-effort
> reasoning over image tokens. The sidecar has none of these issues and stays
> fully on-prem. This document is kept for reference; the content below
> describes the retired native path.

# Native MoonViT vision (DeepSeek-V4-Flash-0731)

When **enabled**, the same DSpark vLLM process serves **text + images** for 0731 using WebBrain’s **MoonViT + PatchMerger** in-process. This is **not** a caption sidecar and **not** the SGLang production path.

## Enable (2× DGX Spark TP=2)

### 1. Download + SHA-verify tower/projector

```bash
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
hf download webbrain-one/DeepSeek-V4-Flash-0731-Vision-BF16 \
  vision_tower.safetensors mm_projector.safetensors config.json \
  --local-dir "$HF_HOME/webbrain-0731-moonvit-src"

./scripts/verify-moonvit-artifacts.sh
# Expected SHAs (PLAN-VISION.md §2.1):
#   tower:      1382c41f1a4afc91791ade630e2b1e1cef68cc5a1e09668a45970a5d5e1b8f15
#   projector:  7024d9d5c9714c7abbc09abda015f083b7d7b107745eb78879f019bf4721577a
```

Repeat on the **worker** (`WORKER_HF_CACHE`) so both ranks see the files.

### 2. Stage overlay model directory

Symlinks official 0731 shards + vision weights + merged `config.json` (`architectures: DeepseekV4MoonVitForCausalLM`):

```bash
python3 scripts/prepare-moonvit-model-dir.py \
  --output "$HF_HOME/dsv4-0731-moonvit"
# Worker: same under WORKER_HF_CACHE
```

### 3. Serve with MoonViT

In `.env.dspark` (or export before `./start-deepseek-v4-flash-dspark.sh`):

```bash
ENABLE_MOONVIT=1
DSV4_MOONVIT_ENABLED=1
DSPARK_MODEL=/cache/huggingface/dsv4-0731-moonvit   # path inside container
SERVED_MODEL_NAME=deepseek-v4-flash-0731-vision
DSV4_MOONVIT_TOWER=/cache/huggingface/webbrain-0731-moonvit-src/vision_tower.safetensors
DSV4_MOONVIT_PROJECTOR=/cache/huggingface/webbrain-0731-moonvit-src/mm_projector.safetensors
# Optional: leave KV room for ~0.9 GB BF16 tower+projector
GPU_MEMORY_UTILIZATION=0.78
```

`docker-compose.dspark.yml` installs `plugins/dsv4_moonvit_vllm` when `ENABLE_MOONVIT=1`, sets `VLLM_PLUGINS=dsv4_moonvit`, and passes `--limit-mm-per-prompt '{"image":1}'`. **DSpark MTP-5 stays on.**

Optional compose overlay: `docker-compose.moonvit.override.yml`.

### 4. Smoke

```bash
# Multimodal (OpenAI image_url)
python3 scripts/smoke-moonvit-chat.py \
  --base-url http://127.0.0.1:8888 \
  --model deepseek-v4-flash-0731-vision \
  --out results/smoke-mm-chat.json

# Text-only on the same endpoint
python3 scripts/smoke-moonvit-chat.py --text-only --out results/smoke-text-only.json

# Multi-turn text → image → follow-up
python3 scripts/smoke-moonvit-chat.py --multiturn --out results/smoke-multiturn.json
```

## Client API

### Multimodal chat

```bash
curl -s http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash-0731-vision",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What color is this? One word."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }],
    "max_tokens": 64,
    "temperature": 0,
    "chat_template_kwargs": {"thinking": false}
  }'
```

HTTPS `image_url` may work if the server can fetch URLs; **data URIs** are the reliable path on offline clusters.

### Thinking / reasoning

Image answers can be trapped in unclosed `<think>` if reasoning is on. For agent tools and smokes prefer:

```json
"chat_template_kwargs": {"thinking": false}
```

or set `DEFAULT_THINKING=off` for the vision lane. 0731 encoding / tools / parsers remain the same as text-only.

**`reasoning_effort=max` with images is unstable (measured 2026-08-10):** with
image tokens in the prompt, max/high-effort reasoning nondeterministically
falls into repetition attractors (scene-vocabulary loops, e.g. 狗/猫/电视…),
returning empty `content` (`finish_reason=length`) roughly half the time.
Text-only max reasoning is stable; `low` effort with images works for short
answers; repetition/presence penalties do **not** fix it. Root cause is in the
abliterated LM's long-horizon reasoning over image embeddings, not the vision
input (the loop vocabulary is scene-relevant).

For deep reasoning about image content use the **two-pass pattern** —
extract with thinking off (stable), then reason at max over the description
(text-only, stable):

```bash
python3 scripts/vision-reason.py --image photo.jpg \
  --question "Is this home pet-friendly? Reason step by step." \
  --show-description
```

### Text-only

Omit image parts; same `model` id and endpoint. Encoding, tool-call parser, and reasoning parser are unchanged.

### Multiturn: multi-image and re-attached images

The server accepts up to **4 images per request** (`--limit-mm-per-prompt {"image":4}`). Each
`<image>` placeholder in the prompt expands independently with palette phase restarting
at 0 per span.  This covers two common patterns:

**Pattern A — image in every turn (most clients):**
Many chat clients re-attach the image(s) in each turn so the model always sees them.
This now works without HTTP 400:

```json
"messages": [
  {"role": "user", "content": [
    {"type": "text", "text": "What color is this?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  ]},
  {"role": "assistant", "content": "Red."},
  {"role": "user", "content": [
    {"type": "text", "text": "Confirm the color."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  ]}
]
```

**Pattern B — genuine multi-image:**
Two to four images in a single prompt.  Each image gets its own MoonViT encode and
placeholder expansion.  Budget: 4×512 = 2048 merged tokens, well within
`max_num_batched_tokens` 8192.

```json
"messages": [{"role": "user", "content": [
  {"type": "text", "text": "Compare these images."},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
]}]
```

**Working pattern (still valid) — attach once, reference with text:**
The image stays in context from the first turn; follow-ups can be text-only.
You do **not** need to re-attach the image — but doing so no longer causes errors.

## Limits (v2)

| Limit | Value |
| --- | --- |
| Images per request | **4** (experimental) |
| Max merged image tokens per image | **512** |
| Max merged image tokens total | 2048 (4×512) |
| Tower / projector dtype | BF16 |
| Multi-image quality | **Experimental / unvalidated** (adapter trained 1 img/prompt) |
| Video | **Not supported** (non-goal) |
| Quality | Experimental (GUI/OCR not production-claimed; hue QA adapter-limited) |

## Architecture (in-process)

```
image_url → NaViT preprocess → MoonViT tower → PatchMerger (LN→2×2→MLP)
  → ≤512 × 4096 embeddings injected at token id 129280 (<image>)
  → palette_cycle route IDs on image positions only (text IDs unchanged)
  → DeepseekV4ForCausalLM + DSpark MTP-5
```

Plugin package: `plugins/dsv4_moonvit_vllm` (`vllm.general_plugins` entry `dsv4_moonvit`).

## DSpark

The multimodal class is a **transparent** wrapper: `forward(..., **kwargs)` delegates, `lm_head` and other attrs proxy to the language model. If SpecDecoding logs show **0.0x** per-position acceptance or the known **~1–15%** collapse band, treat it as a ship blocker (opaque wrapper).

## Gaps vs full multi-image/video

Tracked as non-blocking once single-image native chat works:

- Multi-image (`limit-mm-per-prompt` > 1)
- Video chunks
- Prefix-cache scale for image prompts
- Encoder-only-on-rank-0 memory optimization
- Production OCR/GUI quality

## Unit tests

```bash
# Host (needs torch) or inside Anemll container:
pip install -e plugins/dsv4_moonvit_vllm --no-deps
pytest tests/test_moonvit_units.py -q
```

## Color QA tip

Solid-color checks work best with pure RGB fixtures (≥256²) and a short forced-choice prompt, e.g.:

`What color is this solid image? One word: red/green/blue/black/white.`

MoonViT tower must load with WebBrain key names (`encoder.blocks.*.wqkv`, not `attn.qkv_proj`). Enable `--mm-encoder-tp-mode data` (model sets `supports_encoder_tp_data`).

### Color QA reliability (measured 2026-08-10, DSpark ON, temp=0, N=10, v3 projector)

| Fixture | Pass rate | Notes |
| --- | --- | --- |
| red, black, white, green, blue | **10/10 each** | stable across identical requests |

This requires the **v3 embedding-aligned projector**
(`webbrain-0731-moonvit-src/mm_projector-v3-0731.safetensors`, deployed via
`DSV4_MOONVIT_PROJECTOR` in `.env.dspark`). The original WebBrain projector is
adapter-limited for 0731 (red ~40–60%, green ~40%, blue 0%): it was trained for
Kimi's embedding space, and its output rows are ~18x the 0731 token-embedding
scale (norm ~127 vs ~7.3). The v3 projector is fine-tuned against the real
MoonViT tower + the real 0731 `embed.weight` table (InfoNCE caption alignment +
color CE + norm anchor); see `docs/PROJECTOR-FINETUNE.md`.

Deployment gotcha: the serve resolves the projector from
`DSV4_MOONVIT_PROJECTOR` / compose auto-discovery
(`webbrain-0731-moonvit-src/mm_projector.safetensors` first candidate) — **not**
from the `mm_projector.safetensors` symlink inside the overlay model dir.
Verify with the serve log: `encode_image ... norm=` should be ~10–20 per-row
with v3, ~127 with the original.

N-trial gate (exits non-zero when pass rates miss thresholds, captures DSpark metrics):

```bash
python3 scripts/smoke-moonvit-colors.py --trials 10 --out results/smoke-mm-status.json
```

Status thresholds: red ≥0.90, black/white ≥0.80, green/blue ≥0.50 (v3 passes all).

### Open-ended color naming and LM text priors (v3.1/v3.2, 2026-08-10)

The v3.1+ projectors add anchors for pink/brown/gray/beige/navy/olive/teal/maroon,
so open-ended color naming works for solid fields and neutral phrasings
("what color is the top?" → pink). One residual ceiling is **not** fixable at
the adapter level: the abliterated LM has strong text priors for certain
phrasings — it answers "Blue" to *"What color is the person's sweater?"* **even
with no image attached**, and that prior can override correct image evidence.
Object-context anchor training (v3.2) did not move it; image-first ordering
only nudges it. Guidance for clients:

- Prefer neutral phrasings: "main color", "what color is the top/clothing item",
  or forced-choice lists ("pink/blue/yellow") — all verified reliable.
- Avoid garment-noun questions ("sweater", "dress") when the answer matters;
  or use `scripts/vision-reason.py`, whose extraction prompt is neutral.
- Occasional open-ended descriptions may drift to Chinese; re-ask or pin
  "Answer in English."
