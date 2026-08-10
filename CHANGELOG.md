# Changelog

## Unreleased

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
