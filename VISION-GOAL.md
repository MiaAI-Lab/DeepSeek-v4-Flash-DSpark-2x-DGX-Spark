# VISION-GOAL — Native MoonViT vision on this DeepSeek-V4-Flash-0731 vLLM / DSpark stack

**Use this file as the goal prompt for an implementer or coding agent.**  
**Authoritative plan:** follow [`PLAN-VISION.md`](PLAN-VISION.md) for architecture, weights, and DSpark hazards. When this goal and the plan conflict on **success criteria**, **this file wins** (native vision). When they conflict on **technical constraints** (weights, routing, DSpark transparency), **the plan wins**.

---

## Goal (one sentence)

Make **vision work natively** on this repository’s **2× DGX Spark vLLM + DSpark** stack: the same served DeepSeek-V4-Flash-0731 process understands images via WebBrain’s **MoonViT + PatchMerger + routing bridge**, through **standard multimodal APIs**, with DSpark still engaged—not a caption sidecar, not SGLang-only, not a text model plus external VLM.

---

## Success definition: “vision works natively”

**Native** means all of the following:

| Native means | Not native (failure even if “an image answer exists”) |
| --- | --- |
| One **vLLM** OpenAI-compatible endpoint serves **text + images** for this model | Separate vision service / caption sidecar piped into text 0731 |
| Images are consumed **in-process** (MoonViT → projector → embedding inject into DeepSeek) | Client-side describe-then-prompt, or multi-hop agent glue as the only path |
| Clients use **normal multimodal chat** (`/v1/chat/completions` with `image_url` / vision content parts) | Only a proprietary raw `/generate` + literal `<image>` string path |
| Same **served model id** handles text-only and image turns (history can mix) | Two different backends for “text mode” vs “vision mode” |
| MoonViT + projector + palette routing run inside the **DSpark-enabled** engine | Turning off DSpark or leaving the Anemll recipe to get images |
| 0731 **encoding / tools / reasoning** still apply on multimodal turns | Vision path that bypasses tokenizer/encoding and breaks agent contracts |

Quality may still be experimental; **nativeness of the integration is not optional**.

---

## Success looks like

A **first-class multimodal serve** of 0731 + MoonViT on the same cluster/image family as today.

| Surface | Required behavior |
| --- | --- |
| **Engine** | Single `vllm serve` (this compose/recipe) loads text backbone + MoonViT + projector |
| **API** | `POST /v1/chat/completions` with multimodal `messages` (text + `image_url` data URIs or equivalent OpenAI vision parts) returns grounded answers |
| **Text-only on same model** | Omitting images still works with full 0731 agent semantics |
| **Spec decode** | DSpark MTP-5 remains on for text and image requests (no acceptance collapse) |
| **Ops** | Documented enable path (env/compose); dual-lane *during rollout* is fine, but **success is not** “sidecar documented as the vision solution” |

**Hard acceptance criteria (all required):**

1. **Native API** — A client can send a real image via **OpenAI-compatible multimodal chat** to the DSpark vLLM endpoint and get a correct, non-empty answer that depends on the image pixels (not on a pre-supplied caption). No sidecar VLM required.
2. **Native weights path** — MoonViT tower + PatchMerger from [webbrain-one/DeepSeek-V4-Flash-0731-Vision-BF16](https://huggingface.co/webbrain-one/DeepSeek-V4-Flash-0731-Vision-BF16) are loaded in the same server process as 0731; SHA-256 matches `PLAN-VISION.md` §2.1.
3. **Native embedding contract** — Preprocess → tower → `LN → 2×2 merge → MLP → 4096-d` → inject at image placeholder positions (token id **129280** / `<image>` expansion as implemented); **palette_cycle** routing on image positions only; text routing IDs unchanged.
4. **Load** — Two-node TP=2 serve starts with vision on Anemll `dspark-vllm-gx10` (or documented equivalent) without OOM at a documented `gpu_memory_utilization`.
5. **Text parity on the same endpoint** — Text-only chat/completions still pass 0731 encoding, tools, and reasoning-effort behavior.
6. **DSpark intact** — Speculative decoding stays on (`method: dspark`, block size 5). Logs must **not** show collapsed acceptance (`0.0x` per-position) on text-only or image requests. Do not ship if acceptance falls to the known broken ~1–15% band from opaque wrappers.
7. **Multimodal agent usability** — Documented contract for `thinking` / `chat_template_kwargs` so image answers land in `content` (not stuck only in unclosed reasoning). At least one multi-turn smoke: text → image → follow-up text on the **same** conversation.
8. **Docs + scripts** — Download, SHA verify, model-dir stage, enable serve, and **native chat smoke** (curl or script using multimodal messages) are written down and runnable on head+worker.

**Soft / phase-later (not required to claim “native works”, but track them):**

- Multi-image per request (`limit-mm-per-prompt` &gt; 1)
- Prefix-cache correctness for image prompts at scale
- Encoder-only-on-one-rank memory optimization
- Broad quality eval (GUI grounding, charts, small controls)
- Baking the plugin into the Anemll/runtime image

---

## What to build (implementation target)

Follow `PLAN-VISION.md` technical phases, but **phase success is gated by native chat**, not by a raw-marker-only prototype:

1. **Overlay model directory** — Symlink existing official 0731 snapshot; add tower + projector; merge `vision_config` / `deepseek_vision` / `image_token_id` from WebBrain. Prefer not re-downloading ~167 GB text shards if 0731 is already at `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`.
2. **vLLM multimodal integration** (plugin and/or model registration — e.g. `dsv4_moonvit_vllm` via `vllm.general_plugins`):
   - Load MoonViT (BF16) + PatchMerger (BF16) in-process
   - Wire vLLM’s **native multimodal** input path (processor / `SupportsMultiModal`-style hooks as required by this image’s vLLM version)
   - Map OpenAI `image_url` parts → processor → embeddings → merge into DeepSeek inputs
   - **palette_cycle** routing; DSpark-**transparent** wrapper around `DeepseekV4ForCausalLM`
3. **Compose / launch** — Enable vision in the real serve line (`--limit-mm-per-prompt`, plugin install, model dir). Dual-lane override during development is OK; the **validated** path must be the multimodal vLLM server.
4. **Tooling** — SHA verify, prepare-model-dir, and **`smoke-moonvit-chat.py`** (or equivalent) that hits `/v1/chat/completions` with multimodal messages—not only an internal tensor unit test.

Port vision **math and semantics** from WebBrain’s `sglang_ext/` as reference. **Do not** declare success by running their SGLang launcher instead of this stack.

---

## Non-negotiable constraints

- **Native on this stack:** deliverable is **vLLM + DSpark + MoonViT in one serve process**.
- **Not a sidecar:** caption-then-LLM is not success.
- **Not SGLang production:** SGLang may be reverse-engineered for tensor/routing behavior only.
- **Weights:** WebBrain MoonViT + PatchMerger only; do not rebrand FlyCockpit DeepEncoderV2 as MoonViT (study it only for DSpark transparency patterns if useful).
- **Text backbone:** Official `deepseek-ai/DeepSeek-V4-Flash-0731` at this recipe’s revision unless the user explicitly changes it.
- **DSpark / profile:** Keep MTP-5, `nvfp4_ds_mla` KV, TP=2, nnodes=2, 0731 parsers/encoding unless a measured multimodal exception is documented.
- **Tower/projector dtype:** BF16 in v1.
- **Licenses:** DeepSeek + Kimi-K2.6 + WebBrain notices where applicable.

---

## Explicit non-goals

- Claiming production-grade OCR / computer-use reliability or safety-critical readiness.
- Full video support.
- Multi-image and heavy concurrency tuning (after native single-image chat works).
- Upstream merge into official vLLM or Anemll (document bake later).
- Mixing MoonViT and FlyCockpit towers in one process.
- Treating “opt-in flag exists” or “raw `<image>` string only” as done **without** native multimodal chat.

Rollout may still keep a text-only compose profile for A/B and memory; that is ops, not the success bar.

---

## Context the agent must read first

| Priority | Path |
| --- | --- |
| 1 | [`VISION-GOAL.md`](VISION-GOAL.md) — **this file** (success = native vision) |
| 2 | [`PLAN-VISION.md`](PLAN-VISION.md) — weights, routing, DSpark wrapper, phases |
| 3 | [`docs/DEEPSEEK_V4_FLASH_0731.md`](docs/DEEPSEEK_V4_FLASH_0731.md) — text checkpoint + encoding |
| 4 | [`docker-compose.dspark.yml`](docker-compose.dspark.yml) — serve command and env surface |
| 5 | [`README.md`](README.md) — cluster ops, Anemll image, dual-node launch |
| 6 | [`vllm_patch_gb10/`](vllm_patch_gb10/) — example `vllm.general_plugins` package layout |
| 7 | WebBrain Hub + `sglang_ext/` (reference only): `DeepSeek-V4-Flash-0731-Vision-BF16` |

Hub note: `webbrain-one/DeepSeek-V4-Flash-0` is a **404**. Use **`DeepSeek-V4-Flash-0731-Vision-BF16`**.

---

## Suggested execution order (agent checklist)

- [ ] **Phase 0** — Download tower + projector (+ `sglang_ext` reference); SHA-256 verify; confirm 0731 on head **and** worker.
- [ ] **Phase 1** — Pure-torch MoonViT + projector + routing unit tests (≤512×4096; palette).
- [ ] **Spike** — Transparent no-op wrapper; DSpark acceptance healthy on text.
- [ ] **Phase 2** — Plugin + model dir + multimodal registration; two-node start with vision weights loaded.
- [ ] **Phase 3 — native bar** — OpenAI multimodal `/v1/chat/completions` smoke with a real image; same endpoint text-only smoke; DSpark metrics on both; multi-turn text/image smoke.
- [ ] **Docs** — “Vision is native on this endpoint when enabled”; how clients send `image_url` parts; limits and thinking-flag notes.
- [ ] **Done only when** hard criteria 1–8 pass—not when a non-chat internal path merely encodes an image.

---

## Environment pins (fill when you start; do not invent)

| Item | Expected / fill-in |
| --- | --- |
| Runtime image | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (or pin actual digest) |
| Text model | `deepseek-ai/DeepSeek-V4-Flash-0731` @ `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| Vision package | `webbrain-one/DeepSeek-V4-Flash-0731-Vision-BF16` |
| Tower SHA-256 | `1382c41f1a4afc91791ade630e2b1e1cef68cc5a1e09668a45970a5d5e1b8f15` |
| Projector SHA-256 | `7024d9d5c9714c7abbc09abda015f083b7d7b107745eb78879f019bf4721577a` |
| Topology | 2× DGX Spark / GB10, TP=2, nnodes=2 |
| Spec decode | DSpark, MTP / `dspark_block_size` = 5 |
| API | OpenAI-compatible multimodal chat on the DSpark vLLM port |

---

## Definition of done (for PRs / handoff)

- [ ] MoonViT vision runs **in-process** on the DSpark vLLM serve path
- [ ] **Native** multimodal chat works (`image_url` / vision content parts) with correct image-dependent answers
- [ ] Text-only and image turns work on the **same** served model without a sidecar
- [ ] DSpark acceptance remains healthy; wrapper transparency verified
- [ ] SHA-verified artifacts; prepare + smoke scripts; docs for clients
- [ ] `PLAN-VISION.md` decision log updated (registration/hooks chosen)
- [ ] Known limits listed (e.g. image count, ≤512 tokens, quality still experimental)—**without** redefining success as non-native

---

## Prompt block (paste into an agent)

```text
You are adding NATIVE MoonViT vision to DeepSeek-V4-Flash-0731 on this
monorepo’s 2× DGX Spark vLLM + DSpark stack.

Read VISION-GOAL.md (success criteria) and PLAN-VISION.md (architecture).

SUCCESS means vision works natively:
- One vLLM process serves 0731 text + WebBrain MoonViT + PatchMerger
- Clients use OpenAI-compatible multimodal /v1/chat/completions (image_url
  content parts); answers must depend on image pixels
- Same endpoint handles text-only and image turns with 0731 encoding/tools
- DSpark MTP-5 stays on; no acceptance collapse (transparent wrapper required)
- NOT a caption sidecar, NOT SGLang as the production path, NOT
  “raw <image> string only” as the sole supported client API

Use WebBrain DeepSeek-V4-Flash-0731-Vision-BF16 tower/projector SHAs from
the plan. Port math/routing from sglang_ext as reference only.

Implement through native multimodal chat smoke on two-node TP=2. Document
client usage and limits. Report API examples, DSpark metrics, and any gaps
vs full multi-image/video (those are non-blocking once single-image native
chat works).
```

---

*Keep this success bar stable: **native multimodal vLLM vision**. Put design changes in `PLAN-VISION.md`; put run evidence under `results/` or PR notes.*
