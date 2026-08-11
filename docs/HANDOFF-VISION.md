# Handoff: Native MoonViT vision on DeepSeek-V4-Flash DSpark (2× DGX Spark)

**Date:** 2026-08-10  
**Repo:** `deepSeek-v4-Flash-DSpark`  
**Goal:** One vLLM process serves 0731 text + WebBrain MoonViT + PatchMerger; OpenAI multimodal `/v1/chat/completions` with `image_url`; DSpark MTP stays on; **not** caption sidecar / SGLang production / raw-`<image>`-only API.

---

## 1. Executive status

| Area | Status | Notes |
| --- | --- | --- |
| Dual-node TP=2 vision serve | **UP** (as of handoff) | `deepseek-v4-flash-0731-vision` on `:8888` |
| Architecture | **Native** | `DeepseekV4MoonVitForCausalLM` via overlay model dir |
| OpenAI `image_url` path | **Wired end-to-end** | Processor expands placeholders; tower encodes; embeds merge |
| Text-only on same endpoint | **PASS** | `VISION_TEXT_OK` |
| DSpark MTP | **PASS (no collapse)** | Non-zero acceptance observed in `/metrics` |
| Color / pixel dependence | **PARTIAL / FLAKY** | Solid red often works with tuned prompt; not stable across runs; blue/white imperfect |
| Multi-image / video | **Experimental** (N≤4 images; video out of scope) | Limit 4 images, ≤512 tokens each |

**Bottom line for the next owner:**  
The **native multimodal pipeline is real** (not a stub). The main quality issue was **tower weights not loading** (wrong key rename). After that fix, vision **can** answer red correctly, but solid-color QA is **prompt-sensitive and non-deterministic** under the current stack (likely MTP + dual-rank + model/alignment). Do **not** treat “always Black” as still true—that was the broken-weight regime.

---

## 2. What “success” was defined as

From `VISION-GOAL.md` / implementer bar:

1. Single vLLM process: 0731 LM + WebBrain MoonViT + PatchMerger  
2. Client: OpenAI multimodal chat (`image_url` content parts)  
3. Same endpoint: text-only and image turns, 0731 encoding/tools  
4. DSpark MTP-5/6 on; no acceptance collapse (transparent wrapper)  
5. Not caption sidecar, not SGLang as production path, not raw-`<image>`-only API  

---

## 3. Topology and runtime

### Hardware / process model

- **2× DGX Spark**, tensor parallel **TP=2**, `nnodes=2`  
- Head: `10.0.0.1` (this machine), worker: `zurih@10.0.0.2`  
- Compose project: `deepseek-v4-flash`  
- Image: `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (vLLM ~0.25.2)  
- API: `http://127.0.0.1:8888` (bind `0.0.0.0:8888`)

### How to start / stop

```bash
# From repo root
./stop-deepseek-v4-flash-dspark.sh
./start-deepseek-v4-flash-dspark.sh   # sources .env.dspark
```

**Critical:** CLI env vars set *before* the start script are often **overwritten** by `source .env.dspark`. Put vision settings **in `.env.dspark`**, not only on the command line.

### Current `.env.dspark` vision settings (active)

```bash
DSPARK_MODEL=/cache/huggingface/dsv4-0731-moonvit
ENABLE_MOONVIT=1
DSV4_MOONVIT_ENABLED=1          # if set in file / compose
SERVED_MODEL_NAME=deepseek-v4-flash-0731-vision
DEFAULT_THINKING=off            # image answers land in content more reliably
GPU_MEMORY_UTILIZATION=0.78
# MTP: MTP_NUM_TOKENS=6 (or 5); method dspark
```

Text-only abliterated path (backup) was:

```text
drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32
SERVED_MODEL_NAME=deepseek-v4-flash-0731
```

A host backup of pre-vision env may exist as `.env.dspark.bak-text` if still present.

### Compose vision knobs (`docker-compose.dspark.yml`)

When `ENABLE_MOONVIT=1`:

- `pip install -e /opt/dsv4-moonvit-vllm`
- `VLLM_PLUGINS=…,dsv4_moonvit`
- Auto-discover tower/projector under `/cache/huggingface/webbrain-0731-moonvit-src/`
- Extra args:

```text
--limit-mm-per-prompt {"image":1}
--mm-processor-cache-gb 0
--mm-encoder-tp-mode data
```

Plugin is bind-mounted:

```text
./plugins/dsv4_moonvit_vllm → /opt/dsv4-moonvit-vllm
```

**Worker sync:** start script scp’s `.env.dspark` and some files; **plugin code must be rsynced to worker** after edits:

```bash
rsync -a --exclude '__pycache__' --exclude '*.egg-info' \
  plugins/dsv4_moonvit_vllm/ \
  zurih@10.0.0.2:/home/zurih/models/deepSeek-v4-Flash-DSpark/plugins/dsv4_moonvit_vllm/
```

Then full stop/start (module is imported at process start; bind-mount alone does not reload Python).

---

## 4. Model layout (overlay)

### Overlay directory (both nodes)

Host path (mounted as `/cache/huggingface` in container):

```text
~/.cache/huggingface/dsv4-0731-moonvit/
```

Contents:

- Symlinks to **abliterated 0731** weight shards + tokenizer/encoding  
- Local `config.json` with:
  - `"architectures": ["DeepseekV4MoonVitForCausalLM"]`
  - `vision_config` (Kimi/MoonViT-shaped)
  - `deepseek_vision` adapter (placeholder, max tokens 512, routing palette, SHAs metadata)
  - `image_token_id` / `media_placeholder_token_id` = **129280** (OOV vs vocab_size 129280)
- Symlinks: `mm_projector.safetensors`, tower lives under `webbrain-0731-moonvit-src/`

### WebBrain artifacts

```text
/cache/huggingface/webbrain-0731-moonvit-src/vision_tower.safetensors
/cache/huggingface/webbrain-0731-moonvit-src/mm_projector.safetensors
```

SHAs / plan references: `PLAN-VISION.md`. Reference SGLang port: `third_party/webbrain-sglang-ext/`.

### Scripts

| Script | Role |
| --- | --- |
| `scripts/verify-moonvit-artifacts.sh` | SHA gate |
| `scripts/prepare-moonvit-model-dir.py` | Build overlay (relative symlinks) |
| `scripts/smoke-moonvit-chat.py` | OpenAI text / image / multiturn smoke |

---

## 5. Plugin architecture

**Package:** `plugins/dsv4_moonvit_vllm/`  
**Entry:** `dsv4_moonvit` (`VLLM_PLUGINS`)

| Module | Role |
| --- | --- |
| `model.py` | `DeepseekV4MoonVitForCausalLM` — LM + tower + projector; `embed_multimodal`; palette routing; DSpark-transparent proxy |
| `wrapper.py` | Transparent LM attribute proxy (Eagle3 / MTP) |
| `moonvit.py` | Load MoonViT3d + PatchMerger; **weight key mapping**; encode |
| `projector.py` | Pure `PatchMerger` (LN → 2×2 → Linear×2) WebBrain keys `proj.0`/`proj.2` |
| `preprocess.py` | NaViT resize/pad/normalize/patchify; optional fused Kimi processor |
| `processor.py` | vLLM `BaseMultiModalProcessor`: HF call, **BPE-safe placeholder expand**, palette token expansion |
| `routing.py` | 64-ID palette cycle for hash MoE on image positions |
| `config.py` | `deepseek_vision`, `image_token_id`, palette, max tokens |

### Data path (request)

```text
OpenAI messages[content: text + image_url]
  → vLLM chat template injects literal "<image>" (and often "\n")
  → tokenizer encodes chat → prompt_ids (NOT special media id)
  → Dsv4MoonVitMultiModalProcessor:
       preprocess image → pixel_values (N,3,14,14) + grid_thws
       _apply_prompt_updates: find <image> span under BPE merges
       replace span with in-vocab palette cycle IDs (length = merged tokens)
  → Worker embed_multimodal:
       MoonViT3d(pixel, grid) → tpool merge → PatchMerger → (T, 4096)
  → merge multimodal embeds at placeholder ranges
  → DeepseekV4 LM generate (+ DSpark draft)
```

### Media id design

- Config logical id: **129280** (OOV; training/SGLang sentinel).  
- **Cannot** put 129280 in prompt_ids: vLLM `input_processor` rejects `Token id 129280 is out of vocabulary`.  
- **Mitigation:** expand placeholders to **routing palette IDs** (in-vocab). MoE routes correctly; multimodal mask still marks embed positions via PlaceholderRange.  
- `image_token_id` still used if any path still emits 129280; palette_cycle rewrites those for MoE.

---

## 6. Bugs fixed (chronological, important)

### 6.1 Placeholder never matched (HTTP 400 / “0 placeholders”)

**Symptom:**  
`Expected there to be 1 prompt placeholders corresponding to 1 image items, but instead found 0`

**Cause:**  
Bare `"<image>"` → tokens `[30, 10253, 32]` (`<`, `image`, `>`).  
In chat, `"<image>\n"` merges `>` + newline → e.g. `[30, 10253, 1018]` (`1018` ≈ `>\n`). Exact 3-id target never matches.  
vLLM text-fallback re-encode would also destroy OOV media ids.

**Fix (`processor.py`):**

- `_find_media_token_span`: offset mapping / progressive decode on full prompt  
- Override `_apply_prompt_updates`: replace span in token space (no re-encode)  
- `--mm-processor-cache-gb 0` (already) so cache path doesn’t re-tokenize wrongly  

### 6.2 OOV 129280 after expand (HTTP 400)

**Symptom:** `Token id 129280 is out of vocabulary`

**Fix:** Expand to **palette cycle** IDs, not `[129280]*n`.

### 6.3 MoonViT on CPU → FLASH_ATTN crash (HTTP 500)

**Symptom:** `flash_attn_maxseqlen_wrapper` with CPU backend  

**Fix (`model.py`):** Load tower/projector on `cuda:{current_device}`, not `"cpu"`.  
Also keep `grid_thws` on GPU with pixels (not `keep_on_cpu=True`).

### 6.4 Tower weights never applied (always “Black” / color-blind)

**This was the main quality root cause.**

**Symptom:** Pipeline 200 OK; multimodal token counts > 0; answers ignore hue (often Black/White only).

**Cause (`moonvit.py` `_load_tower_weights`):**  
Loader rewrote checkpoint keys:

```text
wqkv → attn.qkv_proj
wo   → attn.proj
```

vLLM `MoonViTEncoderLayer` **keeps** parameter names `wqkv` / `wo`.  
Almost all encoder weights failed to bind → **random tower**.

**Fix:** Prefer raw WebBrain key names; only use rename as last-resort alias.  
Fail closed if required params still missing after load.  
Ignore non-persistent buffer `patch_embed.pos_emb.time_weight` (sincos, not in ckpt).

**Log you want on every healthy boot (worker):**

```text
[dsv4_moonvit] Loaded MoonViT tower from ... (missing=0 unexpected_file_keys=0)
```

If you see `Could not load vLLM MoonViT tower` or `missing>0`, vision quality is invalid.

### 6.5 mm-encoder-tp-mode data rejected

**Symptom:**  
`This model does not support --mm-encoder-tp-mode data. Falling back to weights.`

**Fix:** `supports_encoder_tp_data = True` on `DeepseekV4MoonVitForCausalLM`.  
Compose already passes `--mm-encoder-tp-mode data` so each rank loads a **full** BF16 tower (disable_tp on vit linears), avoiding broken TP shard of vision weights under nnodes=2.

### 6.6 Token count vs encode mismatch

**Risk:** `image_token_count` used one NaViT path; encode used another (e.g. fused) → placeholder length ≠ embedding length.

**Fix:** `image_token_count` calls `pil_to_pixel_values_and_grid` (same path as encode).

### 6.7 Preprocess

- Prefer **vLLM fused** `KimiK25FusedVisionProcessor` when importable (numba).  
- Fallback pure numpy NaViT; mean/std **0.5/0.5/0.5**; patch 14; merge 2.  
- Optional `DSV4_MOONVIT_CHANNEL_ORDER=rgb|bgr` (default **rgb**). BGR experiment did **not** fix color QA.

---

## 7. Verification commands

### Health

```bash
curl -s http://127.0.0.1:8888/v1/models | python3 -m json.tool
docker logs deepseek-v4-flash-vllm-dspark-1 2>&1 | grep -E "Loaded MoonViT|DeepseekV4MoonVit|mm_encoder_tp"
```

### Smokes

```bash
# Image (solid red fixture 256², tuned prompt)
python3 scripts/smoke-moonvit-chat.py \
  --base-url http://127.0.0.1:8888 \
  --model deepseek-v4-flash-0731-vision \
  --out results/smoke-mm-image.json

# Text-only
python3 scripts/smoke-moonvit-chat.py \
  --base-url http://127.0.0.1:8888 \
  --model deepseek-v4-flash-0731-vision \
  --text-only \
  --out results/smoke-text-only.json

# Multiturn (image-first, then confirm color)
python3 scripts/smoke-moonvit-chat.py \
  --base-url http://127.0.0.1:8888 \
  --model deepseek-v4-flash-0731-vision \
  --multiturn \
  --out results/smoke-mm-multiturn.json
```

**Tuned color prompt (in smoke script now):**

```text
What color is this solid image? One word: red/green/blue/black/white.
```

Other phrasings (“dominant color…”) often push the model to Black even when embeddings differ.

### Metrics (DSpark)

```bash
curl -s http://127.0.0.1:8888/metrics | grep -E 'spec_decode_num_accepted|num_draft'
```

Expect **non-zero** accepted tokens / drafts after traffic (not all zeros).

### Unit tests (in container)

```bash
docker exec deepseek-v4-flash-vllm-dspark-1 \
  bash -c 'cd /opt/dsv4-moonvit-vllm && python3 -m pytest /path/if/mounted/tests -q'
# Host: tests/test_moonvit_units.py was 15 passed / 1 skipped historically
```

---

## 8. Known gaps and flaky behavior

1. **Solid-color accuracy is flaky across identical requests**  
   Observed sequence for pure red smoke: `Red / Red / Black / White / White`.  
   Multiturn (image-first) got stable `Red / Red` in one run while single-shot flaked.  
   Suspected contributors: DSpark speculative decode, dual-rank floating point, residual vision–LM alignment (abliterated LM + WebBrain tower/projector).

2. **Blue / white imperfect** even when red/green/black work under the tuned prompt (blue sometimes → red; white sometimes → red).

3. **History before image hurts color QA**  
   Text-only first turn then image → more Black answers. Smoke multiturn was changed to **image first**.

4. **129280 remains OOV**  
   transformers warns at load; intentional. Do not put 129280 in prompt_ids without raising vocab/input validation.

5. **Multi-image experimental**
   `limit-mm-per-prompt image=4`, max merged tokens 512 per image. Multi-span placeholder
   expansion with palette phase restart per span. Quality unvalidated (WebBrain trained 1 img/prompt).

6. **Plugin not auto-synced to worker** on every edit—manual rsync + restart.

7. **`.env.dspark` is machine-local** and was switched to vision overlay; restoring text-only serve requires reverting `DSPARK_MODEL` / `SERVED_MODEL_NAME` / possibly `ENABLE_MOONVIT=0`.

8. **Scratch dirs** for goal implementer (`/tmp/grok-goal-…`) are ephemeral; real evidence is under `results/` and `docs/`.

---

## 9. Evidence files (repo)

| Path | Content |
| --- | --- |
| `results/moonvit-native-vision.md` | Run notes, checklist, color table |
| `results/smoke-mm-image.json` | Last single-image response |
| `results/smoke-text-only.json` | Text smoke |
| `results/smoke-mm-multiturn.json` | Multiturn |
| `results/smoke-mm-status.json` | Machine-readable status (may lag last flake) |
| `results/dspark-metrics.txt` | Spec-decode metric sample |
| `docs/VISION.md` | Client usage + color QA tip |
| `docs/HANDOFF-VISION.md` | This document |
| `PLAN-VISION.md` / `VISION-GOAL.md` | Plan / success bar |

**Honest last status snapshot:** single-image smoke was **flaky** (status file may show `White` / `pixel_dependent: false` from a bad trial); multiturn showed **Red/Red**; text-only and DSpark metrics looked healthy. Re-run smokes after restart before claiming green.

---

## 10. Key code locations (edit map)

```text
plugins/dsv4_moonvit_vllm/dsv4_moonvit_vllm/
  processor.py     # BPE span find + expand to palette; mm field config
  moonvit.py       # weight load (DO NOT reintroduce wqkv→attn.qkv_proj default)
  model.py         # CUDA tower, supports_encoder_tp_data, encode_image debug prints
  preprocess.py    # fused-or-numpy NaViT; channel order env
  routing.py       # palette 64
docker-compose.dspark.yml   # ENABLE_MOONVIT, VLLM_MM_ARGS
.env.dspark                 # DSPARK_MODEL overlay + SERVED_MODEL_NAME
scripts/smoke-moonvit-chat.py
```

Debug prints still present (useful):

```text
[dsv4_moonvit] apply_updates ... span=[...]
[dsv4_moonvit] encode_image grid=... emb=... mean=... std=... norm=...
[dsv4_moonvit] Loaded MoonViT tower ... (missing=0 ...)
```

---

## 11. Recommended next steps (priority)

1. **Stabilize color QA**  
   - A/B with `enforce_eager` / disable DSpark for MM-only request path  
   - Fix seed if API supports it  
   - Compare emb cosine red vs black on worker (was hard without TP init; with `mm-encoder-tp-mode data` + loaded weights, re-try offline encode)  
   - Validate abliterated vs official 0731 text backbone for vision alignment  

2. **Harden weight load tests**  
   - Unit test: after load, `missing==0` and sample `encoder.blocks.0.wqkv.weight` L2 ≠ random init  
   - Fail container start if load incomplete (already raises; ensure both TP ranks log success)

3. **Clean debug noise**  
   - Gate `print(...)` behind env `DSV4_MOONVIT_DEBUG=1`

4. **Sync & docs**  
   - Commit plugin + compose + smoke + docs (`.env.dspark` usually stays local/untracked secrets-style)  
   - Document rsync worker requirement in README if not already  

5. **Optional product polish**  
   - Chat template / content-part order guidelines for clients (image with short forced-choice color prompt)  
   - Multi-image still non-goal  

---

## 12. Quick “is vision healthy?” checklist

After any restart:

- [ ] `curl /v1/models` → `deepseek-v4-flash-0731-vision`  
- [ ] Logs: `Resolved architecture: DeepseekV4MoonVitForCausalLM`  
- [ ] Logs: `Loaded MoonViT tower ... (missing=0 ...)` on **Worker_TP0** (and ideally TP1)  
- [ ] Logs: **no** `Falling back to --mm-encoder-tp-mode weights` (unless intentional)  
- [ ] Text smoke → `VISION_TEXT_OK`  
- [ ] Image smoke with tuned prompt → often `Red.` (re-run 3–5×; note flake rate)  
- [ ] Metrics → accepted draft tokens > 0 after a few requests  
- [ ] Plugin md5 on head == worker  

---

## 13. Contacts / paths (ops)

| Role | Path / host |
| --- | --- |
| Head workspace | `/home/mia/models/deepSeek-v4-Flash-DSpark` |
| Worker dir | `zurih@10.0.0.2:/home/zurih/models/deepSeek-v4-Flash-DSpark` |
| HF cache host | `~/.cache/huggingface` → container `/cache/huggingface` |
| Plugin mount | `plugins/dsv4_moonvit_vllm` → `/opt/dsv4-moonvit-vllm` |

---

## 14. One-paragraph summary for async chat

Native MoonViT on dual-node vLLM+DSpark is **running**: overlay model `dsv4-0731-moonvit`, plugin `dsv4_moonvit`, OpenAI `image_url` works through BPE-safe placeholder expansion and CUDA MoonViT encode. The big quality bug was **wrong tower weight key remapping** (`wqkv`→`attn.qkv_proj`); with correct load (`missing=0`) and `mm-encoder-tp-mode data`, solid red can answer **Red** and is distinguishable from black under a short forced-choice prompt, but answers are **still flaky** across identical requests and blue/white are weak—next work is determinism (MTP/eager), alignment, and automated load/color regression tests. Always rsync the plugin to the worker and restart both ranks after code changes; put vision env in `.env.dspark`.

---

## 15. Goal-2 addendum (2026-08-10): flakiness root-caused; hue ceiling documented

> **Superseded later the same day:** the "not reachable" verdict applied to the
> *original* WebBrain adapter. A projector fine-tuned against the real 0731
> embedding table (`mm_projector-v3-0731.safetensors`, see
> `docs/PROJECTOR-FINETUNE.md` and `docs/HANDOFF-PROJECTOR-FINETUNE.md`) passes
> the full color gate 10/10 × 5 colors with DSpark ON. The analysis below remains
> the correct diagnosis of *why the original adapter* fails (plus two serving
> bugs it missed: the fine-tune was never actually loaded — the plugin reads
> `DSV4_MOONVIT_PROJECTOR`, not the model-dir symlink — and an ~18× embedding
> scale mismatch vs 0731's token embeddings).

**TL;DR:** The flakiness is **not** DSpark, not prefix cache, not the encoder, not the
backbone, not the port. The WebBrain adapter's hue signal into 0731 is intrinsically weak
(frontend embeddings near-collinear: red↔green rel_l2 0.03); the LM reads luminance
(black/white) reliably but not hues. Prefill-kernel numerics then flip the near-tied color
logits run-to-run. The red ≥90% bar is **not reachable** on this adapter; honest metrics
landed in `results/smoke-mm-status.json` and `results/moonvit-native-vision.md` §Goal-2.

### Evidence chain (all verified on the live dual-node serve)

1. `max_tokens=1` (prefill-only, no spec decode) still flips red → **DSpark exonerated**;
   acceptance stayed healthy throughout (102/336 = 30.4% after 51 image reqs).
2. Prefix cache hit rate 0.0% on image prompts; identical images reuse the **encoder cache**
   (no re-encode in worker logs) yet answers still flip → noise is in the LM prefill forward.
3. Preprocess fused==numpy, **RGB order correct** (red→(+0.67,-1,-1)); projector binds
   **bit-exact** (new unit test); palette/phase match WebBrain config; SGLang reference
   uses identical tower/projector call semantics.
4. No-pad image size (280×280) doesn't fix red; 512px doesn't widen red margins; prompt
   variants (open/grounded/red-in-middle/image-first/assistant-prefix) all ≤ the current
   forced-choice prompt.
5. **Backbone A/B** (official overlay staged both nodes at
   `/cache/huggingface/dsv4-0731-moonvit-official`, 156 GB rsync to worker):
   official 0731 red **0/10** (always Black), abliterated red 4–5/10 → **abliteration is
   not the cause**; reverted to the production abliterated overlay.
6. Frontend measurement (offline, same weights): projector **amplifies luminance
   1.5–2.2×**, hue deltas stay ≤0.06 rel_l2; blue's own image suppresses "blue" ~0.5
   logprob below "red" in the LM.

### Final state

- Serve: `DSPARK_MODEL=/cache/huggingface/dsv4-0731-moonvit` (abliterated, production),
  DSpark MTP-6 ON, tower `missing=0` both ranks, `mm-encoder-tp-mode data`.
- Pass rates (N=10, temp=0): red 40%, black 100%, white 80%, green 40%, blue 0%;
  text-only `VISION_TEXT_OK`; multiturn image-first Red/Red.
- New: `scripts/smoke-moonvit-colors.py` = N-trial gate (exit≠0 on miss; thresholds
  red≥0.90, black/white≥0.80, hue≥0.50; captures spec-decode counters). Current adapter
  **fails** red/green/blue by design — the gate documents that honestly.
- Unit tests: 20 passed / 1 skipped in container (added projector bit-exact binding,
  RGB channel order, no-pad math, smoke answer matching).

### If someone picks this up again

- The only real fixes for hue QA: a retrained/fine-tuned projector for 0731 (out of repo
  scope), a different vision adapter, or WebBrain publishing a 0731-validated update.
- Official overlay + snapshot remain on both nodes if a future adapter needs them
  (worker disk was 97% full after the 156 GB transfer — check before adding more).
- Do not delete `results/goal2-*.json` — they are the A/B evidence.
