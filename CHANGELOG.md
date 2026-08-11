# Changelog

## Unreleased

### Changed
- **VL sidecar 4-bit KV (GB10)**: production dtype is now `VL_SIDECAR_KV_CACHE_DTYPE=int4_per_token_head` + `TRITON_ATTN` (≈½ fp8 KV bytes). True `--kv-cache-dtype nvfp4` is **blocked** on SM12.1 (FlashInfer requires SM100 trtllm-gen; Triton rejects `nvfp4`). Coexist profile: main `GPU_MEMORY_UTILIZATION=0.82`, `MAX_NUM_SEQS=4` → **~1.60M** 0731 GPU KV tokens with VL up @ 32k; ≥2M needs ~0.85 main alone and starves VL TP=2 on worker free memory. Evidence: `results/vl-nvfp4-coexist-2026-08-11.md`.
- **`docker-compose.vl-sidecar.yml`**: pass `VLLM_SKIP_INIT_MEMORY_CHECK` (free-memory util gate on this image may still apply).

### Added
- **VL sidecar TP=2**: `docker-compose.vl-sidecar.yml` now runs Qwen3-VL-4B with `--tensor-parallel-size 2 --nnodes 2` across head+worker (separate NCCL `VL_SIDECAR_MASTER_PORT=25100`). Per-GPU util default **0.03** so 0731 can keep a larger `nvfp4` KV pool. Start is worker-first; prepare caches VL on both nodes; stop tears down VL on both ranks.
- **Vision support (production path)**: document end-to-end VL sidecar + MCP fusion in [`plugins/dspark_vision_mcp/README.md`](plugins/dspark_vision_mcp/README.md) §Vision support (architecture, tools, start/install, harness table, env knobs, errors). Agents stay on text-only 0731 and call `describe_image` / `ocr_image` / `compare_images` instead of switching models.
- **Multi-harness vision MCP install**: `scripts/install-dspark-vision-mcp.sh` + `plugins/dspark_vision_mcp/install_harnesses.py` detect and register `dspark-vision` into **pi**, **OMP**, **Hermes**, **opencode**, **goose**, **Grok Build**, **OpenClaw**, **ZCode**, and **Prime Agent**. Idempotent upsert + skill copy. `start-deepseek-v4-flash-dspark.sh` waits for the VL sidecar then runs the installer when `INSTALL_VISION_MCP` is on (defaults to follow `ENABLE_VL_SIDECAR`). Knobs in `.env.dspark.example` (`VISION_MCP_HARNESSES=auto`).
- **`plugins/dspark_vision_mcp`**: local vision MCP server for 0731. Calls the Qwen3-VL sidecar on `:8889` with base64 data URIs (local paths + http(s), missing-file / sidecar-down errors, auto-downscale, max 4 images).
- **Local VL sidecar for vision** (replaces native MoonViT as the production vision path): `docker-compose.vl-sidecar.yml` serves `cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit` with `--kv-cache-dtype int4_per_token_head` + `TRITON_ATTN` **TP=2 across both Sparks** on `:8889` (served name `qwen3-vl-4b`); `./prepare-dspark-model-cache.sh` downloads `VL_SIDECAR_MODEL` into `HF_CACHE` on head **and** worker; start brings the sidecar up worker-first when `ENABLE_VL_SIDECAR=1` under `HF_HUB_OFFLINE=1`. 0731 serve is text-only; util co-tuned in `.env.dspark`. `scripts/vision-reason.py` extracts via the sidecar and reasons via 0731 at max effort.

### Changed
- **Native MoonViT lane retired** (2026-08-11): after the v3/v3.1/v3.2 projector fine-tunes (color gate 10/10 × 5 colors), the adapter ceiling held for fine-grained recognition, garment-phrasing text priors, and max-effort reasoning over image tokens. Plugin/scripts/docs kept for reference; `docs/VISION.md` marked retired. Restore with `.env.dspark.bak-vision-v32` if ever needed.

### Added
- **`scripts/extract-embed-table.py`**: lazily extracts the real 0731 `embed.weight` / `head.weight` (129280×4096) from checkpoint shards for offline training/eval — no full model load.
- **`scripts/train-projector-v3.py`**: projector fine-tune that actually works — real frozen MoonViT tower (offline single-process vLLM distributed init) + InfoNCE alignment against the real 0731 embedding table + color-word CE + log-norm anchor; COCO val2017 captions + synthetic colors; built-in offline gate (`--eval-only`). Trains alongside the live server (~75 min, 3000 steps).
- **v3 projector deployed** (`webbrain-0731-moonvit-src/mm_projector-v3-0731.safetensors`, via `DSV4_MOONVIT_PROJECTOR` in `.env.dspark`): live color gate **10/10 on red, black, white, green, and blue** (N=10, temp=0, DSpark MTP-6 ON; was red 60 / white 70 / green 40 / blue 0 on the original adapter). See `results/projector-v3-colors.json`, `docs/PROJECTOR-FINETUNE.md`.
- **v3.2 projector** (`mm_projector-v32-0731.safetensors`, currently deployed): 18-color vocabulary (adds pink/brown/gray/beige/navy/olive/teal/maroon) + object-context anchors; open-ended color naming works for solids and neutral phrasings. Documented ceiling: LM garment-noun text priors ("sweater"→"Blue" even without an image) are not adapter-fixable. See `docs/VISION.md`.
- **`scripts/vision-reason.py`**: two-pass vision-reasoning helper (extract with `thinking=false` → reason over the description with `reasoning_effort=max`). Works around the measured instability of native max/high-effort reasoning over image tokens (scene-vocabulary repetition loops, empty answers; text-only max reasoning is stable). See `docs/VISION.md` §Thinking/reasoning.

### Fixed (vision, 2026-08-10)
- **Projector deployment path**: the plugin resolves the projector from `DSV4_MOONVIT_PROJECTOR` / compose auto-discovery — the `mm_projector.safetensors` symlink in the overlay model dir is never read, so earlier fine-tune "deployments" (v2) never actually served the new weights; head/worker symlinks had also diverged (split-brain TP ranks). Deployment is now via `.env.dspark` + verified by the `encode_image ... norm=` log (~10–20 per-row with v3 vs ~127 original).
- **~18× image-embedding scale mismatch**: WebBrain projector rows norm ~127 vs 0731 token embeddings ~7.3 (Kimi-scaled adapter). v3 converges to ~7.7 and the LM reads hue reliably.

### Added
- **`scripts/smoke-moonvit-colors.py`**: N-trial solid-color QA gate for native MoonViT vision (DSpark ON, temp=0). Fails (exit≠0) when pass rates miss thresholds (red ≥0.90, black/white ≥0.80, green/blue ≥0.50), asserts text-only `VISION_TEXT_OK`, and captures spec-decode counters into `results/smoke-mm-status.json`.
- **Unit tests** (`tests/test_moonvit_units.py`): projector weight binding bit-exactness vs WebBrain safetensors, RGB channel-order check, no-pad NaViT math, smoke-gate answer matching. 20 passed / 1 skipped in the Anemll container.
- **Multi-image support (experimental)**: `--limit-mm-per-prompt` raised from `{"image":1}` to `{"image":4}`. Plugin processor accepts N>1 images per request; each `<image>` placeholder expands independently with palette phase restarting at 0 per span. Covers clients that re-attach the same image every turn (multiturn fix) and genuine 2–4 image prompts. **Experimental / unvalidated** — WebBrain trained 1 image/prompt; quality with N>1 is not benchmarked.

### Changed (vision findings, 2026-08-10)
- **MoonViT solid-color flakiness root-caused**: adapter-intrinsic hue weakness (frontend embeddings near-collinear, rel_l2 0.03–0.19; projector amplifies luminance 1.5–2.2×). DSpark exonerated (`max_tokens=1` prefill-only still flips); prefix/encoder caches exonerated; preprocess/projector/palette verified faithful; backbone A/B (official vs abliterated, both staged) shows red fails on both (0/10 vs 4–5/10) — abliteration is not the cause. Honest pass rates: red ~40–50%, black 100%, white 60–80%, green 30–40%, blue 0%. See `results/moonvit-native-vision.md` §Goal-2, `docs/HANDOFF-VISION.md` §15, `docs/VISION.md` reliability table.

### Fixed
- **`nvfp4_ds_mla` long-context decode regression ([Issue #22](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/22))**: `nvfp4_ds_mla` was dispatched to the slow `_forward_bf16_kv` kernel path instead of the fast `_forward_fp8_kv` path, causing ~16x decode slowdown at 600K+ context (1.0 tok/s vs 17.3 tok/s with `fp8_ds_mla`).  The 584-byte KV layout is identical for both dtypes on DSV4; only the kernel dispatch differed.

  **Root cause** (line 880 in `flashmla_sparse.py`):
  ```python
  use_fp8_cache = self.kv_cache_dtype == "fp8_ds_mla"
  # nvfp4_ds_mla → False → slow _forward_bf16_kv
  # fp8_ds_mla   → True  → fast _forward_fp8_kv
  ```

  **Fix**:
  ```python
  use_fp8_cache = self.kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")
  ```

### Added
- **`patches/hotfix-nvfp4-ds-mla-issue22.sh`**: standalone hotfix script that patches `flashmla_sparse.py` inside a running container.  Idempotent (skips if already applied).  Usage: `docker exec <container> bash hotfix-nvfp4-ds-mla-issue22.sh`
- **`patches/fix-nvfp4-ds-mla-long-context.patch`**: human-readable reference patch
- **Automatic hotfix on start** (`start-deepseek-v4-flash-dspark.sh`): the start script syncs the hotfix to the worker, applies it to both head and worker containers after `compose up`, and restarts them so vLLM starts with the patched file.  Opt out with `DSPARK_SKIP_HOTFIX=1`.
- **`DSPARK_SKIP_HOTFIX` env var** (`.env.dspark.example`): set to `1` to skip the automatic hotfix (e.g. when using a pre-patched image)
- **Hotfix status in profile print** (`start-deepseek-v4-flash-dspark.sh`): shows whether the hotfix will apply, was skipped, or was not found

### Changed
- **`docs/PATCHES.md`**: added Issue #22 section with root cause analysis and fix details

### Previously unreleased (carried forward)
- Raise `DEFAULT_THINKING` from `low` to `max` in `.env.dspark.example`, enabling full reasoning effort by default. Request-level overrides still take precedence.
- Make `deepseek-ai/DeepSeek-V4-Flash-0731` the default checkpoint for the two-Spark 1M profile.
- Document the 0731 encoding, parser, and vision boundaries.
- Add a streaming benchmark sweep that reports observed TTFT, output throughput, and aggregate throughput without imposing a server-side output cap.
- Expand README Result / Quick Start / Verify notes for PR #14 (0731 boot KV, sweep highlights, regular-graph opt-out).
- Add official 0731 decode-benchmark capture and numbers under README Benchmarks (`docs/benchmarks.png`).

### Added (earlier)
- **`docs/ENVS.md`**: matrix of compose/`.env` knobs vs Anemll `0.1.1` `vllm.envs` registration and Stage-C overlay (`recipe/overlay/vllm/envs.py`)
- **`docker-compose.stage-c.override.yml`**: optional injection of Stage-C-only `VLLM_DSPARK_*` / `VLLM_USE_B12X_WO_PROJECTION` / related knobs

### Changed (earlier)
- **`docker-compose.dspark.yml`**: default Anemll path no longer injects Stage-C-only `VLLM_*` keys that warn as unknown on `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`
- **`.env.dspark.example`**: split Anemll-safe defaults vs commented Stage-C-only block; document `CUTE_DSL_ARCH=sm_121a`
- **README**: 0731 is the documented current lane; preview Anemll results kept as historical

### Notes
- Missing env registration on Anemll does **not** imply missing baked-in DSpark/Keys code paths; it only means those kill-switches are no-ops on 0.1.1
- Re-audit after image tag bumps (snippet in `docs/ENVS.md`)


## 2026-07-29

### Added
- **Auto RoCEv2 GID resolution** (`start-deepseek-v4-flash-dspark.sh`):
  - `resolve_nccl_gid_indexes()` resolves per-node RoCEv2 GID index from sysfs at launch, avoiding NCCL init failures from stale/shared literal GID indexes
  - `iface_ipv4()`, `pick_gid_match_ip()`, `resolve_rocev2_gid_index()` helper functions
  - `NCCL_IB_GID_AUTO=1` is now the default; set `NCCL_IB_GID_AUTO=0` to pin indexes manually
  - `NCCL_IB_GID_MATCH_IP` / `WORKER_NCCL_IB_GID_MATCH_IP` for explicit RoCE IPv4 match when the fabric address differs from the socket ifname
- **Per-node worker NCCL overrides** (`.env.dspark.example`, `start-deepseek-v4-flash-dspark.sh`):
  - `WORKER_NCCL_IB_HCA`, `WORKER_NCCL_SOCKET_IFNAME`, `WORKER_TP_SOCKET_IFNAME`, `WORKER_GLOO_SOCKET_IFNAME` for QSFP rings where facing port names differ per node
  - `WORKER_NCCL_IB_GID_INDEX` for pinned worker-side GID index
  - `remote_nccl_env()` injects per-worker NCCL env vars into remote docker-compose commands

### Changed
- **MTP_NUM_TOKENS default raised from 3 to 5** across all config files:
  - `.env.dspark.example`: `MTP_NUM_TOKENS=3` → `MTP_NUM_TOKENS=5`
  - `docker-compose.dspark.yml`: default fallback `3` → `5` (both env and `--speculative-config`)
  - `validate-dspark-config.sh`: diagnostic output updated to reflect new default
  - `start-deepseek-v4-flash-dspark.sh`: profile print and cudagraph capture size updated
  - Rationale: DSpark checkpoint `dspark_block_size` is 5; k<5 silently truncates draft blocks on Anemll 0.25.2 and is rejected on stock vLLM 0.26+
- **GPU_MEMORY_UTILIZATION lowered from 0.845 to 0.80** (`.env.dspark.example`) to provide headroom for cudagraph capture at the larger capture size (`max_num_seqs * (MTP_NUM_TOKENS + 1)` = 6×6 = 36)
- **NCCL documentation expanded** in `.env.dspark.example` with comments explaining QSFP ring topology, per-node port naming, GID index drift after reboot, and auto-resolve workflow
- **Profile print** in `start-deepseek-v4-flash-dspark.sh` now includes NCCL HCA/socket ifname, GID indexes, and cudagraph capture size for both head and worker nodes

### Mode changes (100755 → 100644, no content diff)
- `build-dspark-vllm-runtime.sh`
- `logs-deepseek-v4-flash-dspark.sh`
- `prepare-dspark-model-cache.sh`
- `smoke-deepseek-v4-flash-dspark.sh`
- `scripts/verify-overlay-sources.sh`
- `recipe/overlay/vllm/envs.py`
- `vllm_patch_gb10/README.md`
- `vllm_patch_gb10/pyproject.toml`
- `vllm_patch_gb10/vllm_gb10_hybrid_nvfp4/__init__.py`
- `vllm_patch_gb10/vllm_gb10_hybrid_nvfp4/config.py`
- `vllm_patch_gb10/vllm_gb10_hybrid_nvfp4/kernel.py`
