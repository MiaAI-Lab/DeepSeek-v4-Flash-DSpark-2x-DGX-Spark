# PLAN-VISION — MoonViT for DeepSeek-V4-Flash-0731 on this vLLM / DSpark stack

**Status:** implemented (experimental dual lane)  
**Date:** 2026-08-09  
**Target stack:** this repo — 2× DGX Spark, Anemll `dspark-vllm-gx10:0.1.1`, TP=2, DSpark MTP-5, `nvfp4_ds_mla` KV, official `deepseek-ai/DeepSeek-V4-Flash-0731` @ `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`  
**Vision source of truth:** [webbrain-one/DeepSeek-V4-Flash-0731-Vision-BF16](https://huggingface.co/webbrain-one/DeepSeek-V4-Flash-0731-Vision-BF16) (MoonViT + PatchMerger + routing bridge)

> Note: `https://huggingface.co/webbrain-one/DeepSeek-V4-Flash-0` is a 404. The 0731 vision packages are under `webbrain-one/DeepSeek-V4-Flash-0731-Vision-*`. The older preview overlay is `webbrain-one/DeepSeek-V4-Flash-Vision-BF16` and points at 0731 for the current backbone.

---

## 1. Executive summary

Upstream **DeepSeek-V4-Flash-0731 is text-only**. WebBrain attaches a **frozen MoonViT-3d tower** (from Kimi-K2.6) and a **trained ~40.1M PatchMerger projector** that maps vision features into DeepSeek’s **4096-d** token space, with a **64-ID routing palette** so image positions get deterministic hash-route IDs without changing text routing.

That package ships **SGLang integration only** (`sglang_ext/`, pinned SGLang commit, source patch). **Stock vLLM does not load MoonViT into DeepSeek V4.** This plan is how to port the same weights and semantics into **this vLLM + DSpark** recipe while keeping the validated text agent lane.

**Recommended path for this repo:** keep the existing 0731 text checkpoint and cache layout; add an **overlay model dir** (symlinks + vision files + config metadata) and a **vLLM general plugin** that:

1. Loads MoonViT + PatchMerger (BF16).
2. Implements multimodal embedding injection + palette routing on prefill/extend.
3. Wraps `DeepseekV4ForCausalLM` in a **DSpark-transparent** way (critical).
4. Exposes OpenAI-style image inputs (or a staged `/generate`-like path first).

**Do not** switch the production serve line to full WebBrain SGLang unless you deliberately leave DSpark / Anemll / NVFP4-MLA / this compose recipe behind.

---

## 2. Artifact analysis (WebBrain 0731-Vision-BF16)

### 2.1 What the package actually is

| Piece | Detail |
| --- | --- |
| Package kind | Complete **copy** of official 0731 text checkpoint **plus** vision overlay (not “BF16 of the whole MoE”) |
| Text backbone | `deepseek-ai/DeepSeek-V4-Flash-0731` @ `9e165c30…` — **same revision this recipe already pins** |
| Text tensors | 48 shards, **unchanged**; mixed storage (config `torch_dtype: bfloat16`, `quant_method: fp8`, etc.) |
| Architecture string | Still `DeepseekV4ForCausalLM` (vision is external glue, not a new HF class in the card) |
| Vision tower | `vision_tower.safetensors` — MoonViT-3d from `moonshotai/Kimi-K2.6` @ `7eb5002f…`, **329 BF16 tensors**, ~834 MB |
| SHA-256 tower | `1382c41f1a4afc91791ade630e2b1e1cef68cc5a1e09668a45970a5d5e1b8f15` |
| Projector | `mm_projector.safetensors` — **6 BF16 tensors**, **40,119,040** params, ~80 MB |
| SHA-256 projector | `7024d9d5c9714c7abbc09abda015f083b7d7b107745eb78879f019bf4721577a` |
| DSpark fields | Preserved: `dspark_block_size: 5`, noise token, target layers, markov rank |
| Serving claim | Custom **SGLang** external model/processor + source patch; **not** stock Transformers/SGLang/vLLM |
| GPU validation (0731 assembly) | **Not** claimed for the assembled 0731 BF16 package; earlier adapter validation was on other text packs / B200+SGLang |

Sibling repos (same tower/projector hashes):

- [webbrain-one/DeepSeek-V4-Flash-0731-Vision-NVFP4](https://huggingface.co/webbrain-one/DeepSeek-V4-Flash-0731-Vision-NVFP4) — same vision files on MJPansa NVFP4 text shards  
- [webbrain-one/DeepSeek-V4-Flash-Vision-BF16](https://huggingface.co/webbrain-one/DeepSeek-V4-Flash-Vision-BF16) — **overlay-only** (tower+projector+glue) for the older preview text ref  

For **this** stack, prefer either:

- **A.** Download only vision artifacts (~914 MB) and stage onto the existing 0731 cache (best disk reuse), or  
- **B.** Full 0731-Vision-BF16 snapshot (~167 GB text + vision) if you want a single path and mirror identity.

### 2.2 Projector and token contract

```text
LayerNorm → 2×2 patch merge → Linear(4608, 4608) → GELU → Linear(4608, 4096)
```

| Constraint | Value |
| --- | --- |
| Tower patch features | 1152-d (`vision_config.hidden_size` / `mm_hidden_size`) |
| Merge | `merge_kernel_size: [2, 2]` → 4×1152 = **4608** before MLP |
| LLM hidden | **4096** |
| Image placeholder | literal `<image>` |
| Placeholder token id | **129280** (`image_token_id` / `media_placeholder_token_id`) |
| Max merged image tokens | **512** (training envelope / integration limit) |
| Images per request (WebBrain) | **1** (current deliberate limit) |
| Routing policy | `palette_cycle` over fixed **64** IDs in `config.deepseek_vision.routing_palette` |
| Text routing IDs | **unchanged** |
| Image routing IDs | cycle palette only on image positions during **prefill/extend** (not decode) |

`vision_config.model_type` is `kimi_k25` (MoonViT / Kimi processor path). Processor expectations: Kimi-K2.6-style resize, normalize, patchify (WebBrain uses the pinned Kimi processor).

### 2.3 Shipped integration layout (SGLang)

Under the Hub repo:

```text
sglang_ext/deepseek_vision_sglang/
  __init__.py
  patch.py          # registers / monkey-patches pinned SGLang
  routing.py        # palette_cycle + text-id preservation
  models/           # external multimodal model wrapper
  processors/       # <image> + image_data preprocessing
scripts/launch_sglang_moonvit.sh
scripts/smoke_sglang_moonvit.py
docs/SGLANG_DEPLOYMENT.md
configs/routing/...palette64.json
VISION_ADAPTER_MANIFEST.json
```

Pinned SGLang commit: `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1`  
`requires_sglang_source_patch: true` — **stock engines will not work by config alone.**

First-party smoke API (SGLang): native `/generate` with `"text": "...<image>..."`, `"image_data": "data:image/..."`. OpenAI chat image parts are **explicitly unsupported** in the source package today.

### 2.4 Memory delta (2× Spark)

Rough order-of-magnitude **extra** vs text-only:

| Item | Size |
| --- | --- |
| MoonViT tower (BF16 weights) | ~0.83 GB on disk; on-GPU similar if resident full BF16 |
| Projector | ~0.08 GB |
| Working buffers (patches, merged tokens ≤512×4096) | small vs MoE; still competes with **KV pool** |
| KV impact | each image costs up to **512** sequence tokens (plus any tiling if you later change layout) |

At `GPU_MEMORY_UTILIZATION≈0.80–0.835` and ~2.5M-token KV on this cluster, expect **KV pool shrinkage** after loading the tower on both ranks (or after encoder TP policy). Plan for a modest util drop or max-seq/image budget when measuring.

### 2.5 Licenses (compliance gate)

| Component | License file / note |
| --- | --- |
| DeepSeek text | `LICENSE` / `LICENSE_DEEPSEEK_V4_FLASH` (upstream) |
| MoonViT tower | `LICENSE_KIMI_K2.6` |
| WebBrain projector + glue | package `LICENSE` |
| Downstream | must satisfy **all three** |

---

## 3. Current repo baseline (what we must not break)

| Area | Today |
| --- | --- |
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731` text-only (`docs/DEEPSEEK_V4_FLASH_0731.md`) |
| Image | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` |
| Serve | `docker-compose.dspark.yml` → `vllm serve` TP=2, nnodes=2, DSpark speculative, NVFP4 MLA KV, chunked prefill, prefix cache |
| Encoding | 0731 `encoding/encoding_dsv4.py` installed into vLLM tokenizer path; reasoning-effort `low` fix |
| Registry | `DeepseekV4ForCausalLM` → `vllm.models.deepseek_v4` |
| Plugins | Optional `vllm.general_plugins` example: `vllm_patch_gb10` (`gb10_hybrid_nvfp4`) |
| README stance | “0731 is text-only; pair with a multimodal sidecar when image input is required” |
| Spec decode | Overlay proposer accepts `mm_embed_inputs` in signature — multimodal-aware **draft path exists in API shape**, but language model is not multimodal |

Any vision wrapper that **hides** the backbone (no `**kwargs` through `forward`, no `lm_head` property) **kills DSpark acceptance** (~1–15% vs 50–64%). This is proven on Spark with the FlyCockpit DeepEncoderV2 plugin and will apply equally to a MoonViT wrapper.

---

## 4. Landscape: three vision approaches

| Approach | Tower / projector | Engine | DSpark on this stack | Quality / notes |
| --- | --- | --- | --- | --- |
| **WebBrain MoonViT (this plan)** | Kimi MoonViT ~834 MB + PatchMerger ~80 MB | Designed for **SGLang**; **vLLM port required** | Keep vLLM if ported carefully | Strong product story for browser/GUI; 0731 GPU smoke **not** claimed by publisher |
| **FlyCockpit DeepEncoderV2** | ~865 MB tower + ~40 MB adapter | **vLLM plugin already** (`dsv4_vision_vllm`) | Proven on 2× Spark with transparency fix | Screenshot-strong; different tower; ~50 tps after reboot; **not** MoonViT |
| **Sidecar VLM** | Separate small VLM | No change to DSpark text lane | Zero risk to text throughput | Extra hop; agent must stitch captions |

**Decision for “I want MoonViT in this vLLM stack”:** implement **Approach A** below. Optionally keep FlyCockpit as a short-term **spike** only if you need OpenAI image parts *before* MoonViT is ready — do not mix towers in one process.

---

## 5. Goals and non-goals

### Goals

1. Serve **0731 + MoonViT + WebBrain projector** through **this** compose/vLLM path on 2× GB10.
2. Preserve **DSpark MTP-5** acceptance on text-only and image requests (acceptance collapse is a ship blocker).
3. Preserve **0731 encoding** (reasoning effort, tools, `<think>`).
4. Fingerprint-verify tower/projector SHAs before first load.
5. Text-only parity smoke (same prompts as current lane) after enabling vision weights.
6. At least one live image smoke (describe / OCR screenshot).
7. Document agent contract (`thinking` flags, placeholder, limits).

### Non-goals (phase 1)

- Upstream merge into official vLLM / Anemll image.
- Video, multi-image, >512 merged tokens, OpenAI multi-part parity with every client.
- Replacing SGLang’s entire multimodal scheduler feature set.
- Claiming OCR/GUI grounding production quality (WebBrain also does not).
- Switching default production endpoint away from text-only without a dual-lane flag.

---

## 6. Architecture for vLLM + MoonViT

### 6.1 High-level data path

```text
  Client (image + text)
           │
           ▼
  Processor (Kimi-style)
    pixels → MoonViT patches
           │
           ▼
  vision_tower.safetensors  (frozen BF16)
           │ 1152-d patches
           ▼
  mm_projector PatchMerger  (trained BF16)
           │ ≤512 × 4096 embeddings
           ▼
  Embed merge at <image> / token_id 129280 positions
           │
           ├─ routing: text IDs keep hash route;
           │           image positions ← palette_cycle(64)
           │           (apply on prefill/extend only)
           ▼
  DeepseekV4ForCausalLM  (existing Anemll / overlay path)
           │
           ├─ DSpark draft reads auxiliary HS / lm_head  ← must stay visible
           ▼
  Logits / OpenAI response
```

### 6.2 Recommended packaging shape (mirror proven Spark pattern)

```text
plugin/   dsv4_moonvit_vllm/          # new pip package in this monorepo
  pyproject.toml                      # entry-point vllm.general_plugins
  src/dsv4_moonvit_vllm/
    __init__.py                       # register()
    model.py                          # DeepseekV4MoonVitForCausalLM wrapper
    moonvit.py                        # tower load + forward (port from sglang_ext)
    projector.py                      # PatchMerger dims/weights
    routing.py                        # palette_cycle (port from sglang_ext/routing.py)
    processor.py                      # HF processor + <image> expansion
    config.py                         # read vision_config + deepseek_vision
scripts/
  prepare-moonvit-model-dir.py        # symlink 0731 + copy vision files + config patch
  smoke-moonvit-image.py
  verify-moonvit-artifacts.sh         # SHA-256 gate
docs/ or PLAN-VISION.md               # this file → later VISION.md
```

**Model directory strategy (zero-cost text shards):**

1. Resolve local 0731 snapshot (already in HF cache on head + worker).
2. Create e.g. `$HF_CACHE/dsv4-0731-moonvit/` with **symlinks** to all text shards + tokenizer + encoding.
3. Copy/symlink `vision_tower.safetensors`, `mm_projector.safetensors`.
4. Write `config.json` = upstream 0731 + WebBrain `vision_config` + `deepseek_vision` + `image_token_id`.
5. Optionally set `architectures` to the **wrapper class name** registered by the plugin (FlyCockpit style), **or** keep `DeepseekV4ForCausalLM` and register a multimodal subclass via plugin registry hooks — pick one mechanism and stick to it.
6. Augment `model.safetensors.index.json` only if the weight loader must discover vision tensors by name; if the plugin loads tower/projector from explicit paths/env, index augmentation is optional.

Env knobs (proposal):

| Env | Purpose |
| --- | --- |
| `DSV4_MOONVIT_TOWER` | path to `vision_tower.safetensors` |
| `DSV4_MOONVIT_PROJECTOR` | path to `mm_projector.safetensors` |
| `DSV4_MOONVIT_MAX_IMAGE_TOKENS` | default 512 |
| `DSV4_MOONVIT_ENABLED` | compose feature flag |
| `VLLM_PLUGINS` | include `dsv4_moonvit` (and existing plugins) |

### 6.3 Critical implementation details

1. **Wrapper transparency (DSpark)**  
   - `forward(..., **kwargs)` fully delegated to language model after embedding merge.  
   - `lm_head` property → backbone.  
   - Any DSpark-specific attributes used by `dspark_proposer` / model runner must remain reachable (inspect Anemll image + overlay proposer for attribute access).  
   - Log `SpecDecoding metrics` acceptance; **0.0x per-position = wrapper bug**.

2. **Embedding injection site**  
   Port SGLang’s “inject multimodal embeddings without replacing DeepSeek V4 attention/routing” idea:  
   - Prefer implementing `get_input_embeddings` / multimodal embedding merge APIs that Anemll’s vLLM version exposes for other VLMs.  
   - If 0.25.x-era APIs differ, follow the FlyCockpit plugin’s hook points on the **same** Anemll image as a reference for *where* to patch, while replacing the tower math with MoonViT.

3. **Routing bridge**  
   - Port `palette_cycle` and the exact 64-ID list from config (do not invent a new palette — projector was trained against this bridge).  
   - Apply **only** on image positions during prefill/extend.  
   - Never rewrite text token route IDs.

4. **Vocab / placeholder**  
   - Token id **129280** may sit at/above `vocab_size: 129280` — confirm whether it is an extended special token or an out-of-range media marker handled only by the processor (SGLang path treats it as media placeholder). vLLM must not embed it via normal `embed_tokens` without replacement.

5. **Processor & chat template**  
   - 0731 has **no** Jinja chat template; encoding is Python (`encoding_dsv4.py`). Vision must compose with that encoder: expand `<image>` → N placeholder tokens **before** or **inside** the same encode path.  
   - Phase 1 may accept a simplified raw-prompt path (`/v1/completions` or a custom smoke) if chat multimodal plumbing is incomplete.  
   - Agent gotcha (FlyCockpit, likely relevant): image requests may need `chat_template_kwargs: {"thinking": false}` (or equivalent) so answers are not trapped in unclosed `<think>`.

6. **TP=2 placement**  
   - Default: replicate vision tower on both ranks (simplest, more memory).  
   - Better later: encoder on rank 0 + broadcast embeddings (more engineering).  
   - Do not assume SGLang’s `TP=4/5` profiles; this cluster is **TP=2 across two nodes**.

7. **CUDA graphs / chunked prefill / prefix cache**  
   - WebBrain: no CUDA-graph validation.  
   - Phase 1: allow graphs for **text decode** if image prefill runs outside captured graphs (common VLM pattern).  
   - Prefix cache: image tokens must participate in correct hashing (`mm_hash` fields already appear in overlay outputs types) or disable prefix cache for image requests initially.

8. **Quantization**  
   - Tower/projector stay **BF16**; text path stays official FP8 MoE as today.  
   - Do not “NVFP4 the vision tower” in phase 1.

---

## 7. Implementation phases

### Phase 0 — Inventory and pin (0.5–1 day)

- [ ] Confirm head + worker HF cache has 0731 @ `9e165c30…`.
- [ ] Download vision artifacts only (or full Vision-BF16 if preferred):
  ```bash
  hf download webbrain-one/DeepSeek-V4-Flash-0731-Vision-BF16 \
    vision_tower.safetensors mm_projector.safetensors \
    config.json VISION_ADAPTER_MANIFEST.json \
    --local-dir "$HF_CACHE/webbrain-0731-moonvit-src"
  ```
- [ ] Optionally fetch `sglang_ext/` + `docs/SGLANG_DEPLOYMENT.md` as **reference source** (not to run SGLang in production).
- [ ] SHA-256 verify tower + projector against §2.1.
- [ ] Snapshot Anemll image id and vLLM version string into this plan’s “environment pin” section when executed.
- [ ] Read FlyCockpit plugin only as **DSpark transparency / plugin entrypoint** reference (different weights).

**Exit:** verified files on both nodes; no serve-line changes yet.

### Phase 1 — Offline module port (1–3 days)

- [ ] Extract MoonViT + PatchMerger + routing pure PyTorch modules from `sglang_ext` (strip SGLang runtime deps).
- [ ] Unit test on CPU/GPU: load safetensors → random or fixture image → output shape `[T, 4096]`, `T ≤ 512`.
- [ ] Unit test routing: text ids unchanged; image span cycles palette.
- [ ] Document exact preprocessor (image size, mean/std, patch grid) from Kimi processor.

**Exit:** `pytest` (or script) green without vLLM.

### Phase 2 — vLLM plugin skeleton (2–5 days)

- [ ] Package `dsv4_moonvit_vllm` with `vllm.general_plugins` entry point (same pattern as `vllm_patch_gb10`).
- [ ] Register multimodal model class.
- [ ] Implement embedding merge + placeholder handling.
- [ ] Wire weight load for tower/projector via env paths.
- [ ] `prepare-moonvit-model-dir.py` symlink tree.
- [ ] Compose delta (feature-flagged):
  - `pip install` plugin at container start (or bake later).
  - `DSPARK_MODEL` → moonvit model dir.
  - `--limit-mm-per-prompt '{"image":1}'` (phase 1).
  - Do **not** remove DSpark speculative config.

**Exit:** server starts on 2× Spark; text-only hello world works; DSpark metrics non-zero.

### Phase 3 — Image smoke + DSpark acceptance (1–2 days)

- [ ] Image request smoke (screenshot + natural photo).
- [ ] Compare SpecDecoding acceptance text-only vs image (target: no collapse; aim for ~50%+ as on text lane).
- [ ] TTFT / decode tok/s note (expect lower than pure text; record baseline).
- [ ] Thinking/`reasoning_content` behavior matrix (`off`/`low`/`high` × with/without image).

**Exit:** written results under `results/moonvit-…json` + short section in docs.

### Phase 4 — Agent / OpenAI surface (optional, 2–4 days)

- [ ] `image_url` content parts (base64 data URIs).
- [ ] History replay counting image budget.
- [ ] pi / harness notes; dual model ids: `deepseek-v4-flash-0731` (text) vs `deepseek-v4-flash-0731-vision`.
- [ ] Multi-image only if routing + memory allow (not required for v1).

### Phase 5 — Hardening (ongoing)

- [ ] Prefix cache correctness for image prompts.
- [ ] Encoder TP / single-rank tower to reclaim KV.
- [ ] Concurrency with mixed text/image.
- [ ] Quality eval: UI OCR, charts, small controls (WebBrain gaps).
- [ ] Consider Anemll image bake once stable.
- [ ] Track WebBrain if they publish a real vLLM path or updated projector.

---

## 8. Compose / launch delta (sketch)

Do **not** apply until Phase 2. Illustrative only:

```yaml
# env additions
DSV4_MOONVIT_ENABLED=1
DSV4_MOONVIT_TOWER=/cache/huggingface/webbrain-0731-moonvit-src/vision_tower.safetensors
DSV4_MOONVIT_PROJECTOR=/cache/huggingface/webbrain-0731-moonvit-src/mm_projector.safetensors
DSPARK_MODEL=/cache/huggingface/dsv4-0731-moonvit
SERVED_MODEL_NAME=deepseek-v4-flash-0731-vision
VLLM_PLUGINS=dsv4_moonvit   # compose with existing plugins if any
```

Serve extras (names may match Anemll’s CLI):

```text
--limit-mm-per-prompt '{"image":1}'
# keep: speculative-config dspark, kv-cache-dtype nvfp4_ds_mla, tokenizer-mode deepseek_v4, ...
```

Container start (alongside encoding install):

```bash
python3 -m pip install --quiet --no-deps /opt/dsv4-moonvit-vllm
# then existing encoding_dsv4.py install + vllm serve ...
```

Prefer a **compose override file** (`docker-compose.moonvit.override.yml`) so the default text lane stays one flag away.

---

## 9. Validation gates (must pass before any default-on)

| Gate | Pass criteria |
| --- | --- |
| G0 Artifact | SHA-256 match for tower + projector |
| G1 Load | 2-node start, weights load, no OOM at chosen util |
| G2 Text parity | Same coding/tool smoke as current 0731 lane; no encoding regression |
| G3 DSpark | Spec decode acceptance not collapsed (log check); mean accept length sane |
| G4 Image smoke | Correct description of fixture image; non-empty `content` under documented thinking flags |
| G5 Memory | Record KV tokens before/after; document new util recommendation |
| G6 Perf | C1 decode tok/s text-only within ~10–15% of pre-vision baseline (or document intentional tradeoff) |

---

## 10. Risks and mitigations

| Risk | Severity | Mitigation |
| --- | --- | --- |
| DSpark acceptance collapse | **P0** | Transparent wrapper; never ship without G3 |
| Anemll multimodal API gaps | **P0** | Spike against live image; fallback inspect FlyCockpit hook points on same image |
| OOM / KV shrink | **P1** | Replicated tower cost; lower util; encoder-on-one-rank later |
| Placeholder/vocab mismatch | **P0** | Match WebBrain token id + merge logic exactly |
| Wrong palette / routing | **P0** | Copy config palette; unit tests |
| Thinking block empty `content` | **P1** | Document `thinking: false` for image agents |
| Dual download of 167 GB | **P2** | Overlay-only staging on existing 0731 |
| Quality expectations | **P2** | Experimental; not sole sensor for safety-critical automation |
| License stack | **P1** | Ship NOTICE with DeepSeek + Kimi + WebBrain |
| Scope creep to SGLang | **P2** | Keep SGLang as reference only for this recipe |

---

## 11. Alternative: temporary FlyCockpit lane (not MoonViT)

If product needs **any** vision on Spark next week while MoonViT is in progress:

- Plugin: [FlyCockpit/DeepSeek-V4-Vision-2x-DGX-Sparks](https://github.com/FlyCockpit/DeepSeek-V4-Vision-2x-DGX-Sparks)  
- Weights: `FlyCockpit/DeepSeek-V4-Flash-0731-vision`  
- Community report: works on 2× Spark; **must** apply wrapper transparency fix; ~40–50 tps; screenshot-oriented; context management issues → not production-grade.

**Do not** present FlyCockpit and MoonViT as interchangeable checkpoints.

---

## 12. Alternative: run WebBrain’s SGLang path (out of scope for “this vLLM stack”)

Only if abandoning DSpark/Anemll for a vision experiment node:

```bash
export DEEPSEEK_VISION_MODEL_PATH=.../DeepSeek-V4-Flash-0731-Vision-BF16
export DEEPSEEK_VISION_PYTHONPATH=$DEEPSEEK_VISION_MODEL_PATH/sglang_ext
# pin SGLang fdebc938…, apply their patch, launch_sglang_moonvit.sh
```

This does **not** deliver the goals of this repository’s Spark recipe.

---

## 13. Suggested file ownership in this monorepo

| Path | Action |
| --- | --- |
| `PLAN-VISION.md` | This plan (done) |
| `plugins/dsv4_moonvit_vllm/` | New (phase 2) |
| `scripts/prepare-moonvit-model-dir.py` | New |
| `scripts/verify-moonvit-artifacts.sh` | New |
| `scripts/smoke-moonvit-image.py` | New |
| `docker-compose.moonvit.override.yml` | New (optional) |
| `.env.dspark.example` | Document vision env vars |
| `docs/DEEPSEEK_V4_FLASH_0731.md` | Cross-link vision plan; keep text-only default clear |
| `README.md` | Short “Vision (experimental)” pointer when Phase 3 passes |
| `recipe/overlay/...` | Touch only if core DeepSeek V4 forward must learn mm embeds (prefer plugin first) |

---

## 14. Decision log (fill as work proceeds)

| Date | Decision | Rationale |
| --- | --- | --- |
| 2026-08-09 | Target WebBrain **0731-Vision** MoonViT + projector on **vLLM/DSpark**, not SGLang production | Matches user request + existing stack |
| 2026-08-09 | Prefer **artifact overlay** on official 0731 cache over re-downloading full 167 GB | Same text revision already pinned |
| | Architecture name: wrapper vs keep `DeepseekV4ForCausalLM` | TBD after Anemll registry spike |
| | Default-on vs dual lane | Prefer **dual lane** until G2–G6 pass |
| 2026-08-09 | Architecture: `DeepseekV4MoonVitForCausalLM` + `vllm.general_plugins` `dsv4_moonvit` | Register multimodal wrapper; keep language path as DeepseekV4ForCausalLM |
| 2026-08-09 | Overlay relative symlinks under HF_HOME | Absolute host paths break `/cache/huggingface` mount |
| 2026-08-09 | Dual lane via `ENABLE_MOONVIT=1` | Text-only default preserved |

---

## 15. Immediate next actions

1. Run Phase 0 downloads + SHA verify on head and worker.  
2. Vendor `sglang_ext` sources into a `third_party/webbrain-sglang-ext/` tree (read-only reference).  
3. Spike: load MoonViT+projector in a one-off GPU script inside the Anemll container (shape test).  
4. Spike: empty multimodal plugin that only wraps `DeepseekV4ForCausalLM` transparently and confirms **DSpark acceptance unchanged** on text.  
5. Only then merge embedding injection + real weights.

---

## 16. References

- [webbrain-one/DeepSeek-V4-Flash-0731-Vision-BF16](https://huggingface.co/webbrain-one/DeepSeek-V4-Flash-0731-Vision-BF16)  
- [webbrain-one/DeepSeek-V4-Flash-0731-Vision-NVFP4](https://huggingface.co/webbrain-one/DeepSeek-V4-Flash-0731-Vision-NVFP4)  
- [webbrain-one/DeepSeek-V4-Flash-Vision-BF16](https://huggingface.co/webbrain-one/DeepSeek-V4-Flash-Vision-BF16) (preview overlay)  
- [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) @ `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`  
- [moonshotai/Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6) @ `7eb5002f…` (tower provenance)  
- Method inspiration: [baseten/GLM-5.2-Vision-NVFP4](https://huggingface.co/baseten/GLM-5.2-Vision-NVFP4)  
- Spark community vLLM vision (DeepEncoderV2): NVIDIA forum “DeepSeek V4 Flash with Vision” + FlyCockpit playbook  
- This repo: `docs/DEEPSEEK_V4_FLASH_0731.md`, `docker-compose.dspark.yml`, `recipe/overlay/vllm/…`, `vllm_patch_gb10/` (plugin pattern)

---

*End of PLAN-VISION.md — implementation should not start mid-document; begin at §15 / Phase 0.*
