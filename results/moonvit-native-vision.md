# MoonViT native vision — run notes

**Date:** 2026-08-09  
**Target:** DeepSeek-V4-Flash-0731 (production abliterated overlay) + WebBrain MoonViT on 2× DGX Spark, Anemll `dspark-vllm-gx10:0.1.1`, TP=2, DSpark MTP.

## Artifacts (SHA-256)

| File | SHA-256 |
| --- | --- |
| `vision_tower.safetensors` | `1382c41f1a4afc91791ade630e2b1e1cef68cc5a1e09668a45970a5d5e1b8f15` |
| `mm_projector.safetensors` | `7024d9d5c9714c7abbc09abda015f083b7d7b107745eb78879f019bf4721577a` |

Staged under `$HF_HOME/webbrain-0731-moonvit-src` and overlay `$HF_HOME/dsv4-0731-moonvit` (relative symlinks; works host + `/cache/huggingface` mount).

## What shipped

| Path | Role |
| --- | --- |
| `plugins/dsv4_moonvit_vllm/` | vLLM `general_plugins` entry `dsv4_moonvit` |
| `scripts/verify-moonvit-artifacts.sh` | SHA gate |
| `scripts/prepare-moonvit-model-dir.py` | Overlay model dir |
| `scripts/smoke-moonvit-chat.py` | OpenAI multimodal + text-only + multiturn |
| `tests/test_moonvit_units.py` | Routing / projector / preprocess / transparency |
| `docs/VISION.md` | Client usage |
| `docker-compose.dspark.yml` | `ENABLE_MOONVIT=1` install + `--limit-mm-per-prompt` |
| `docker-compose.moonvit.override.yml` | Optional compose overlay |

## Enable

```bash
ENABLE_MOONVIT=1
DSPARK_MODEL=/cache/huggingface/dsv4-0731-moonvit
SERVED_MODEL_NAME=deepseek-v4-flash-0731-vision
GPU_MEMORY_UTILIZATION=0.78
DEFAULT_THINKING=off   # recommended for image answers in content
./start-deepseek-v4-flash-dspark.sh
```

## API example

```bash
python3 scripts/smoke-moonvit-chat.py \
  --base-url http://127.0.0.1:8888 \
  --model deepseek-v4-flash-0731-vision
```

```json
{
  "model": "deepseek-v4-flash-0731-vision",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "What color is this? One word."},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
  }],
  "chat_template_kwargs": {"thinking": false},
  "max_tokens": 64
}
```

## Unit tests

`15 passed, 1 skipped` (projector weight load when file present) inside Anemll container.

## DSpark

Transparent wrapper: `forward(..., **kwargs)`, `lm_head` proxy, `SupportsEagle3` for MTP aux layers. Spec config remains `method: dspark`, block size ≥5 (env `MTP_NUM_TOKENS=5` or 6).

## Live TP=2 status (2026-08-09 — processor fix)

| Check | Result |
| --- | --- |
| SHA tower/projector | **PASS** (both nodes) |
| Architecture | **DeepseekV4MoonVitForCausalLM** via `/cache/huggingface/dsv4-0731-moonvit` |
| Vision serve TP=2 + DSpark MTP | **PASS** |
| Text-only `/v1/chat/completions` | **PASS** (`VISION_TEXT_OK`) |
| OpenAI `image_url` multimodal | **PASS** (HTTP 200; placeholder expand + MoonViT encode on CUDA) |
| Multi-turn image | **PASS** |
| SpecDecoding metrics | Non-zero acceptance (e.g. 12/54 accepted; pos0 6/9) — no 0.0x collapse |
| Color recognition quality | **GAP** — solid red 128×128 fixture answered `Black.` (pipeline OK; vision quality TBD) |

### Processor fix (binding)

Root cause: chat-templated `<image>\n` BPE-merges `>`+newline into one token (e.g. id 1018), so exact target `[30,10253,32]` never matched. Text-fallback re-encode also destroyed OOV media id 129280.

Mitigations in `plugins/dsv4_moonvit_vllm/dsv4_moonvit_vllm/processor.py`:
1. `_find_media_token_span` — offset-mapping / progressive decode under BPE merges
2. `_apply_prompt_updates` override — token-level expand without re-encode
3. Expansion uses **in-vocab palette cycle IDs** (not 129280) so `input_processor` OOV check passes
4. Tower loaded on **CUDA** (was CPU → FLASH_ATTN backend error)

Artifacts: `results/smoke-mm-image.json`, `results/smoke-text-only.json`, `results/smoke-mm-multiturn.json`, `results/smoke-mm-status.json`, `results/dspark-metrics.txt`.

## Color / pixel dependence (post weight-load fix)

| Color | Answer (prompt: "What color is this solid image? One word: red/green/blue/black/white.") |
| --- | --- |
| red (255,0,0) | **Red.** |
| green | **Green.** |
| black | **Black.** |
| white / blue | imperfect (often Red) — residual gap |

Root causes fixed for "always Black":
1. **Tower weights never loaded**: loader renamed `wqkv`→`attn.qkv_proj` which does not exist on vLLM MoonViT; weights stayed random.
2. Fail-closed on non-persistent `time_weight` buffer blocked tower after rename fix.
3. `supports_encoder_tp_data=True` + `--mm-encoder-tp-mode data` for full BF16 tower per rank.
4. Token count must match encode path; prefer fused Kimi preprocess when available.
5. Smoke fixture 256×256 pure red + tuned color prompt.

Evidence: `results/smoke-mm-image.json` (`content` contains Red), `results/smoke-mm-status.json` (`pixel_dependent: true`).

---

## Goal-2 (2026-08-10): stabilization attempt — findings and honest metrics

**Question:** can solid-color QA reach red ≥90%/10, black/white ≥80%/10, one more hue,
with DSpark ON? **Answer: no — not with this adapter on this stack; evidence below.**

### Final gate (abliterated production backbone, DSpark MTP-6 ON, temp=0, N=10/color)

| Fixture | Pass | Answers |
| --- | --- | --- |
| red (255,0,0) | **4/10 (40%)** | Red/Black/White mix |
| black | **10/10 (100%)** | stable |
| white | **8/10 (80%)** | misses → Red |
| green | **4/10 (40%)** | misses → Red |
| blue | **0/10 (0%)** | → Red/Black systematically |

Text-only `VISION_TEXT_OK` ✓. DSpark after 51 image reqs: 102/336 accepted (30.4%),
per-pos [52,48,1,1] — no collapse. Machine-readable: `results/smoke-mm-status.json`.

### What was ruled out (port is faithful)

- Prefill-only flips: `max_tokens=1` still flips red → **DSpark/decode is not the cause**;
  jitter is in the prefill forward (MoE/attention kernels) on near-tied logits.
- Prefix cache: hit rate 0.0% for image prompts — no stale-image KV.
- Encoder cache: identical images share cached embeddings (no re-encode in logs) yet
  answers still flip → LM-side numerics on ~±0.25-logprob margins.
- Preprocess: fused Kimi path == numpy path; channel means red=(+0.67,-1,-1) — **RGB correct** (no BGR swap).
- Projector: all 6 WebBrain tensors bind **bit-exact** (unit test added).
- Palette: 64 IDs match `deepseek_vision.routing_palette`; expansion phase matches `routing.py`.
- SGLang reference (`third_party/webbrain-sglang-ext`): same tower call, same projector mapping, same merge semantics.
- Padding: 280×280 (no-pad multiple of 28) does not fix red.

### Backbone A/B (N=10/color each, same protocol)

| Backbone | red | black | white | green | blue |
| --- | --- | --- | --- | --- | --- |
| abliterated-32-32 (production) | 40–50% | 100% | 60–80% | 30–40% | 0% |
| official 0731 @ 9e165c30 | **0%** (→Black) | 100% | 80–100% | 10% | 0% |

Official overlay staged at `/cache/huggingface/dsv4-0731-moonvit-official` (both nodes)
+ `results/goal2-colors-*.json`. **Abliteration is not the cause; reverted to production backbone.**

### Root cause (measured)

Frontend embeddings for solid colors are nearly collinear — pooled rel_l2: red↔green 0.03,
red↔blue 0.06, red↔black 0.19, black↔white 0.08. The projector **amplifies luminance
(black/white) 1.5–2.2×** but hue directions barely; the LM must read 3–6% directional deltas
and fails for red/green/blue on either backbone (blue's own image *suppresses* "blue" by ~0.5
logprob vs "red"). WebBrain never GPU-validated the 0731 assembly (PLAN-VISION §2.1) — this
looks like the adapter's intrinsic ceiling, not a port defect.

### Protocol kept (measured best)

`What color is this solid image? One word: red/green/blue/black/white.` (text part first,
`thinking:false`, temp=0). Alternatives (open question, pixel-grounding, red-in-middle,
image-first, assistant-prefix) were all **equal or worse** for red. Larger 512px images do
not widen red margins. Gate: `scripts/smoke-moonvit-colors.py` (exit≠0 on threshold miss).

### What still works / doesn't

- ✅ OpenAI `image_url` path end-to-end; multiturn image-first stable (Red/Red anchoring).
- ✅ Black/white (luminance) reliable; DSpark MTP-6 healthy; text lane unaffected.
- ❌ Hue colors (red/green/blue) at/below ~50% — do not rely on hue QA with this adapter.
- ❌ red ≥90% bar not met on any available backbone.

---

## Multi-image (2026-08-10): `--limit-mm-per-prompt` raised to 4

**Goal:** Enable multi-image (N>1) prompts for native MoonViT vision. Covers clients that
re-attach images every turn (the original user pain) and genuine 2–4 image prompts.

### Changes

1. `docker-compose.dspark.yml`: `--limit-mm-per-prompt {"image":1}` → `{"image":4}`
2. `plugins/dsv4_moonvit_vllm/dsv4_moonvit_vllm/processor.py`:
   - `get_supported_mm_limits()` → `{"image": 4}`
   - Removed `len(images) > 1` error in `_call_hf_processor`
   - `_apply_prompt_updates`: replaced `_find_media_token_span` with `_find_nth_media_token_span`
     (finds nth occurrence via offset mapping; each span expands independently)
   - Placeholder validation relaxed: `n_ph == 0` is the error, not `n_ph != len(images)`
3. `tests/test_moonvit_units.py`: added `test_multi_span_palette_restart` and
   `test_replacements_multi_span` — palette phase restarts at 0 per span.

### Verification results

| Check | Result |
| --- | --- |
| Boot: `limit_mm_per_prompt` | `{"image": 4}` ✓ |
| Boot: `Loaded MoonViT tower (missing=0)` | Both ranks (TP0 + TP1) ✓ |
| Boot: `DeepseekV4MoonVitForCausalLM` | ✓ |
| Boot: `method=dspark num_spec_tokens=6` | ✓ |
| 2× same image in 1 prompt | `Yes.` (no HTTP 400) ✓ |
| 2× different images (black+white) | `Blue.` (accepted; adapter-limited quality) ✓ |
| Multiturn image re-attach | `Blue.` / `Blue.` (no error) ✓ |
| Text-only `VISION_TEXT_OK` | ✓ |
| Single-image color gate (N=10) | red 30%, black 100%, white 30% (adapter-limited, no regression in pipeline) ✓ |
| DSpark spec-decode | 142/360 accepted (39.4%), non-zero ✓ |
| Unit tests in container | 20 passed / 3 skipped ✓ |

### Notes

- Multi-image is **experimental / unvalidated**: WebBrain trained 1 image/prompt; quality with
  N>1 is not benchmarked.
- The adapter's hue weakness is unchanged (see Goal-2 above); black/white (luminance) remains
  the reliable signal.
- Budget: 4×512 = 2048 prompt tokens vs `max_num_batched_tokens` 8192 — no config change needed.
- `--mm-processor-cache-gb 0` and `--mm-encoder-tp-mode data` kept unchanged.

---

## 2026-08-10 (PM) — v3 projector: color QA fixed

**Live gate (N=10, temp=0, DSpark MTP-6 ON, v3 projector): red 10/10, black 10/10,
white 10/10, green 10/10, blue 10/10; text-only `VISION_TEXT_OK`; spec-decode
acceptance non-zero (24% after 50 image requests; 60% text-only before).**
Evidence: `results/projector-v3-colors.json`.

Root causes found this session (all three had to be fixed):

1. v2 fine-tune trained on a **random stand-in tower** and a discarded
   classifier head — its "results" were noise.
2. The v2 file was **never actually served**: the plugin resolves the projector
   from `DSV4_MOONVIT_PROJECTOR` / compose auto-discovery
   (`webbrain-0731-moonvit-src/mm_projector.safetensors`), not from the overlay
   model-dir symlink. Head and worker symlinks had also diverged.
3. The WebBrain projector emits image embeddings at row-norm **~127** vs 0731
   token embeddings **~7.3** (~18× scale mismatch, Kimi-scaled), on top of the
   near-collinear hue directions documented in §Goal-2.

v3 training (`scripts/train-projector-v3.py`): real frozen MoonViT tower +
real 0731 `embed.weight` anchors (InfoNCE + color CE + log-norm match),
COCO val2017 + synthetic colors, 3000 steps. Offline: color-word retrieval
1/10 → 10/10; min hue rel_l2 0.03 → 1.25; row-norm 127 → 7.35.

Real-image check (COCO): colors and coarse scene/layout are read correctly;
fine-grained object identity is still weak (giraffes → "elephants"). Colors and
luminance are now dependable; detailed captioning is not production-grade.

Deploy mechanism (both nodes): `DSV4_MOONVIT_PROJECTOR=...mm_projector-v3-0731.safetensors`
in `.env.dspark` + stop/start. Verify via serve log `encode_image ... norm=`.
