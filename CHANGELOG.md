### Changed
- **Text-only ship (vision deferred)**: product default is `ENABLE_VL_SIDECAR=0` with `GPU_MEMORY_UTILIZATION_TEXT=0.835` (0731 on `:8888` only). README documents the text-only agent profile. Optional **Experimental: Vision** section covers `ENABLE_VL_SIDECAR=1` / VL sidecar / MCP for experimenters (not the supported default). `PREPARE_VL_SIDECAR_MODEL` defaults to **0** in prepare + example (set `1` only for vision experiments). `stop-deepseek-v4-flash-dspark.sh` still sweeps leftover VL containers but reports text-only when the flag is off. VL compose / `plugins/dspark_vision_mcp` remain in-tree.

### Removed
- **Native MoonViT vision lane**: deleted `plugins/dsv4_moonvit_vllm`, MoonViT compose override, projector train/eval/smoke scripts, unit tests, WebBrain SGLang ext, and related docs/results (`docs/VISION.md`, `PLAN-VISION.md`, handoffs, projector notes). Vision is **only** the Qwen3-VL sidecar + MCP path (deferred for product docs).

### Added
- **Factory + Command Code vision MCP**: `install_harnesses.py` registers `ds4f-vision` into [Factory Droid](https://factory.ai) (`~/.factory/mcp.json` + `~/.factory/skills/`) and [Command Code](https://commandcode.ai) (`~/.commandcode/mcp.json` + `~/.commandcode/skills/`).
- **Vision MCP gated on flag**: harness install runs only when `ENABLE_VL_SIDECAR=1` (start path + `scripts/install-ds4f-vision-mcp.sh`; use `--force` to override).

### Changed
- **Tool-call DSML dict args ([Issue #21](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/21))**: after installing checkpoint `encoding/encoding_dsv4.py`, compose runs `patches/hotfix-encoding-dsv4-issue21.py` so `encode_arguments_to_dsml` accepts dict `arguments` (not only JSON strings). Prevents multi-turn tool history corruption. Upstream bug is in HF `encoding_dsv4.py` (not this recipe’s weights). Test: `python3 scripts/test-encoding-dsv4-issue21.py`.
- **Checkpoint revision pin ([Issue #19](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/19))**: official prepare/serve default to `DSPARK_REVISION=9e165c30e2704aec5d9d593cce3eebd58bbef1cb`. `prepare-dspark-model-cache.sh` passes `revision=` to `snapshot_download` and writes `refs/main` → that commit; compose passes `vllm serve --revision`. Abliterated uses optional `DSPARK_REVISION_ABLITERATED` (default unpinned). Clear `DSPARK_REVISION=` to follow tip of `main`.
- **Vision MCP rename**: harness / FastMCP / skill id is now **`ds4f-vision`** (CLI entry `ds4f-vision-mcp`, install script `scripts/install-ds4f-vision-mcp.sh`). Installers remove the legacy `dspark-vision` MCP/skill entries on upsert. Package path remains `plugins/dspark_vision_mcp`.
- **`ABLITERATED` checkpoint flag**: `0` → official [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), `1` → [`drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32`](https://huggingface.co/drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32). Start resolves `DSPARK_MODEL` from the flag. `./prepare-dspark-model-cache.sh` interactively asks which to download (or `--official` / `--abliterated` / `--yes`) and writes `ABLITERATED=` back into `.env.dspark`. Encoder auto-discovery follows the selected HF hub snapshot.
- **One-flag serve mode**: `ENABLE_VL_SIDECAR` defaults to **`0`** (text-only). `1` enables vision and sets main util — `0` → `GPU_MEMORY_UTILIZATION_TEXT` (**0.835**, larger KV), `1` → `GPU_MEMORY_UTILIZATION_VISION` (**0.80**) + VL sidecar. Measured Available KV: text ~**18.08 GiB / ~2.49M** tokens; vision main **13.37 GiB / 1.37M** + VL **1.54 GiB / 84k**. Docs: `README.md` §Experimental: Vision.
- **VL sidecar 4-bit KV (GB10)**: production dtype is `VL_SIDECAR_KV_CACHE_DTYPE=int4_per_token_head` + `TRITON_ATTN`. True `--kv-cache-dtype nvfp4` is **blocked** on SM12.1. Evidence: `results/vl-nvfp4-coexist-2026-08-11.md`.
- **pi / ZCode skill collision**: ZCode installer no longer copies `ds4f-vision` into `~/.agents/skills`.

### Added
- **VL sidecar TP=2** + **`plugins/dspark_vision_mcp`**: Qwen3-VL-4B AWQ on `:8889` across both Sparks; MCP tools `describe_image` / `ocr_image` / `compare_images`; multi-harness install (pi, OMP, Hermes, opencode, goose, Grok, OpenClaw, ZCode, Prime, Factory, Command Code). `scripts/vision-reason.py` for CLI two-pass.

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
