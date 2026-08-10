# Goal: Stabilize native MoonViT vision under DSpark

You are continuing NATIVE MoonViT vision for DeepSeek-V4-Flash-0731 on this
repo’s 2× DGX Spark vLLM + DSpark stack (TP=2, nnodes=2).

## Context (read first, do not re-litigate)

Read in order:

1. `docs/HANDOFF-VISION.md` — full handoff of what works and what doesn’t
2. `VISION-GOAL.md` / `PLAN-VISION.md` — original architecture bar
3. `docs/VISION.md` — client usage
4. `results/moonvit-native-vision.md` + `results/smoke-mm-status.json`

The pipeline is already native and live when configured correctly:

- Overlay model: `/cache/huggingface/dsv4-0731-moonvit`
- Plugin: `plugins/dsv4_moonvit_vllm/` (entry `dsv4_moonvit`)
- Serve name: `deepseek-v4-flash-0731-vision`
- OpenAI multimodal `/v1/chat/completions` with `image_url` content parts
- Placeholder expand is BPE-safe; expand uses in-vocab palette IDs (not OOV 129280)
- Tower weight load must keep WebBrain keys (`encoder.blocks.*.wqkv` / `.wo`) —
  do **NOT** reintroduce `wqkv`→`attn.qkv_proj` as the primary mapping
- `supports_encoder_tp_data=True` + `--mm-encoder-tp-mode data`
- DSpark MTP must **STAY ON** (`method: dspark`, `num_speculative_tokens` ≥ 5)

Worker is `zurih@10.0.0.2`. After plugin edits: rsync plugin to worker, then
full stop/start. Put serve config in `.env.dspark` (start script overwrites bare CLI env).

## What is still broken (your job)

Native path works, but vision quality / reliability under DSpark is **not** done:

1. Solid-color QA is **FLAKY** under identical requests with DSpark enabled
   (observed red → Red / Black / White across runs at temperature=0).
2. Some colors remain weak (blue/white often wrong even when red/green/black work).
3. Multiturn with text-before-image confuses color answers (image-first is better).
4. Need durable, repeatable evidence that answers **depend on image pixels** while
   DSpark acceptance stays healthy (no 0.0x collapse).

Do not claim success on a single lucky Red. Do not disable DSpark to “pass”
unless you prove a regression and re-enable it for the success bar below.

## SUCCESS (all required)

Success means vision works properly **with DSpark enabled**:

### A. Serve (dual-node TP=2)

- `DeepseekV4MoonVitForCausalLM`
- DSpark speculative config still active (`method=dspark`, MTP block size ≥ 5)
- Logs: `Loaded MoonViT tower ... (missing=0 ...)` on workers
- No fall-back that leaves tower unloaded / Identity

### B. Pixel dependence (DSpark ON, temperature=0, fixed seeds if available)

- Solid pure-red 256×256 (or larger): answer must clearly indicate red
  (red/crimson/scarlet) in ≥ N independent requests (recommend N=10)
- Solid pure-black: answer black (not red)
- Solid pure-white: answer white (not red/black)
- At least one other hue (green or blue) correct on the same protocol
- Pass rate: ≥ 90% on red; ≥ 80% on black/white; document any remaining miss rate
- Same endpoint text-only still returns `VISION_TEXT_OK`

### C. DSpark health (same process, after MM + text traffic)

- `/metrics` shows non-zero `spec_decode` acceptance (not all zeros / collapse)
- Mean accept length / per-pos accepts are sane vs pre-vision baseline
- No requirement to turn off speculative decode for vision to work

### D. Client API

- OpenAI `image_url` still the supported path (not raw-`<image>`-only)
- Document stable prompt / content-part ordering if required for reliability
- Smoke scripts must use the stable protocol and fail (exit ≠ 0) if pass rates fail

### E. Evidence

- Update `results/smoke-mm-status.json` with trial counts, pass rates, DSpark metrics
- Update `results/moonvit-native-vision.md` honestly
- Keep `docs/VISION.md` client notes current
- Prefer committed tests under `tests/` for weight-load + any deterministic unit pieces

## Non-goals (unless free)

- Multi-image / video
- SGLang production path
- Caption sidecar
- Changing the production abliterated text backbone unless required to meet success
  (if you switch backbones, document and re-verify DSpark)

## Suggested investigation order

1. Confirm tower load `missing=0` and emb norms differ for red vs black vs white
   on the live dual-node path (`encode_image` logs / small worker diagnostic).
2. Isolate flakiness: DSpark on vs off, `enforce_eager`, seed, rank consistency,
   prefix caching, request order. Prefer fixing under DSpark **ON**, not permanent OFF.
3. Align preprocess with Kimi fused path; ensure placeholder token count == emb rows.
4. Check multimodal merge / `is_multimodal` / palette IDs for mis-routing under MoE.
5. Prompt/template: if a stable forced-choice prompt is required, bake into smoke
   and docs; still require true pixel dependence (black ≠ red, white ≠ red).
6. Harden: fail start if tower incomplete; debug prints behind `DSV4_MOONVIT_DEBUG=1`.

## Ops constraints

- Use existing start/stop scripts; dual-node only.
- Rsync `plugins/dsv4_moonvit_vllm` to worker after code changes.
- Do not `rm -rf` model caches or force-push.
- Scratch only for temp logs; durable proof under `results/` and `docs/`.
- Run verification yourself; leave honest metrics if short of 100%.

## Deliverable when done

A short report: what fixed flakiness, pass rates (N trials), DSpark metric snapshot,
any residual limitations, and exact restart + smoke commands for the next human.

---

## One-liner success bar (harness checklists)

PASS iff dual-node DSpark-on serve answers solid red as red ≥ 90%/10 trials,
black as black ≥ 80%, white as white ≥ 80%, text-only `VISION_TEXT_OK`, and
`spec_decode` acceptance remains non-zero (no MTP collapse).
