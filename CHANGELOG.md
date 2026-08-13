## 2026-08-13

### Known issue

- **Output corruption in long agentic sessions with the Issue #26 coordinator hotfix applied**: on a 2x DGX Spark TP=2 deployment at recipe defaults (official 0731, `MTP_NUM_TOKENS=5`, `MAX_NUM_SEQS=6`, `DEFAULT_THINKING=low`), a real agentic-harness session (zcode) produced mid-generation corruption — a short garbage-token run (mixed-script fragments) followed by the model replaying **verbatim earlier-context text** (its own system prompt) as if it were the answer. Server stayed healthy; HTTP 200; no scheduler/proposer warnings; spec-decode acceptance normal. Bisect on the same host: with `patches/hotfix-dsv4-issue26-hybrid-swa-min.py` skipped (coordinator back to stock, everything else unchanged, including #27 and #31), a comparable fresh-session agentic workload ran clean. Suspected mechanism: with the SWA-min constraint relaxed, a prefix-cache hit can be backed by sliding-window KV blocks that were already freed/evicted, so the request resumes on wrong attention state. Not yet root-caused; single-lane and short-context smoke tests do not reproduce it — it needs a long multi-turn agentic session.

### New

- **`DSPARK_SKIP_ISSUE26_HOTFIX=1`** — escape hatch that skips the Issue #26 coordinator hotfix in the compose entrypoint (both nodes; `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` becomes inert). Default `0` (hotfix applies as before). Costs the 32K+ x8 warm prefix-cache restoration from #26; use it if you hit the corruption above. Resolved-profile output of `start-deepseek-v4-flash-dspark.sh` now reports the #26 hotfix status either way.

### Fixed

- **Blank turns on stock OpenAI clients ([Issue #34](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/34))**: #31 only applied `thinking_token_budget` when the request sent the field. pi / OMP / VS Code custom EP cannot, so `DEFAULT_THINKING=max` still returned `content: null`. Recipe now applies omit-field defaults: `DEFAULT_THINKING_TOKEN_BUDGET=32768` and `DEFAULT_MAX_TOKENS=131072` (request wins; empty/`0` restores the old path). Generous on purpose — this model thinks a lot; 32k think + 128k total leaves ~100k for the answer. A client that still sends `max_tokens: 256` can blank; the server does not raise an explicit cap.

### Docs

- README: client recommendations for `thinking_token_budget` and `max_tokens` (they share one generation pool; equal caps empty `content`; recipe omit-field defaults 32768 / 131072).

### Fixed

- **`thinking_token_budget` rejected on DSpark / V2 ([Issue #31](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/31))**: with `DEFAULT_THINKING=max`, a reply that hits `max_tokens` while still inside `<think>` returns `content: null` and `finish_reason: length`. Stock Anemll `0.1.1` rejects `thinking_token_budget` with HTTP 400 (`not yet supported by the V2 model runner`); `VLLM_USE_V2_MODEL_RUNNER=0` cannot be used because DSpark exists only on V2.

  **Fix:** `patches/hotfix-dsv4-issue31-v2-thinking-budget.py` — drop the V2 400, install a V2 sampler hook that counts tokens after the last `<think>` and forces `</think>` when the request budget is exhausted, thread `ReasoningConfig` into the V2 `Sampler`. Mounted RO and applied in the compose entrypoint (synced to the worker by `start-deepseek-v4-flash-dspark.sh`).

  Live (TP=2, thinking `max`): control `17+25` unchanged; hard #31 prompt + `max_tokens=512` + `thinking_token_budget=64` → HTTP 200, short reasoning, **non-null `content`**; same prompt with no budget still `content: null` (default path unchanged). Clients must send `thinking_token_budget` or the empty-content case remains.

- **Prefix-cache collapse at 32K+ x8 ([Issue #26](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/26))** (`1f9765e`): DSV4-Flash + DSpark on Anemll `0.1.1` runs four KV groups (1× `MLAAttentionSpec` + 3× `SlidingWindowMLASpec`). `HybridKVCacheCoordinator.find_longest_cache_hit` takes the min hit length across groups, so a sliding-window group that frees old blocks by design zeroes the common hit. Warm x8 32K/62K then fully re-prefills (`prefix_cache_hits_total +0`, warm wall == cold). Independent of #27.

  Coordinator-only is necessary but not sufficient: at 44K+ x8, dense SWA tails also evict MLA prefix blocks from the shared pool.

  **Fix (both required):**
  1. `patches/hotfix-dsv4-issue26-hybrid-swa-min.py` — SWA/`SlidingWindowSpec` lookups keep their hit blocks but must not shrink `curr_hit_length`. Mounted RO and applied in the compose entrypoint next to the #27 hotfix.
  2. `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096` — sparsify SWA prefix-cache checkpoints (one tail per 4096-token segment + replay boundary).

  Live (TP=2, `max_num_seqs=8`, #27 live): x8 ~22.8K / ~44.7K / ~88.4K warm **8/8** (ratios 0.9986 / 0.9973 / 0.9996; 32K warm wall ~9 s vs ~421 s); x1 262K control 5/5. Repro: `scripts/reproduce-issue26-live.py`, `scripts/reproduce-issue26-control.py`.

## 2026-08-12

### Fixed

- **Decode-lane starvation under concurrent long prefill ([Issue #27](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/27))** (`2f180e7`): stock vLLM 0.25.2.dev0 defines `SchedulerConfig.max_num_partial_prefills` (default 1) but the v1 `Scheduler.schedule` admission loop never reads it. With chunked prefill + async scheduling + `max_num_seqs>=8` and `long_prefill_token_threshold=0`, multiple already-admitted prefills at the front of `self.running` each consume `max_num_batched_tokens`; decode-active requests later get `num_new_tokens==0` and are skipped (`continue`, not preempted) — cold-only, zero-preemption starvation that grows with prompt length.

  **Fix (both required):**
  1. `patches/hotfix-dsv4-issue27-partial-prefill-concurrency.py` — break waiting-admission once `len(self._inflight_prefills)` reaches `max_num_partial_prefills`.
  2. `LONG_PREFILL_TOKEN_THRESHOLD=1024` — cap each prefill chunk so decode lanes keep leftover budget.

  Live: x8 8K/16K/32K worst decode ~15 tok/s (was 2.07 / 0.47 / 0.36), +0 preemptions, MTP 96–99%. Repro: `scripts/reproduce-issue27-live.py`.

### New

#### DSV4 v0.27 performance hotfix backports (6 scripts, all idempotent)

Backported onto the `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` image (vLLM 0.25.2.dev0+g752a3a504.d20260714) from upstream vLLM 0.27.0 DeepSeek-V4 PRs.  Each supports `--status`, `--before`/`--after` (host-side KV + prompt-histogram validation).  Full reference: `docs/vllm-027-new-patches.md`.

- **`patches/hotfix-dsv4-skip-topk-49486.sh`** — upstream #49486, verbatim.  In `models/deepseek_v4/attention.py` (3 hunks): imports `tl, triton`; adds `_fill_short_context_topk_indices` Triton kernel; early-returns from `DeepseekV4Indexer.forward` when `max_seq_len // compress_ratio <= topk_tokens`, still building K cache but writing all-candidate indices directly (skips wq_b→RoPE→quant→QK-logits→top-k).  Fires only ≤2048 tokens (~3.4% E2E TTFT upstream).
- **`patches/hotfix-dsv4-dense-prefill-indexer-48407.sh`** — upstream #48407 port, **12 hunks, deliberately dormant (Stage A)**.  Adds indexer skip machinery across `model_executor/layers/sparse_attn_indexer.py`, `models/deepseek_v4/sparse_mla.py`, `models/deepseek_v4/attention.py`, `models/deepseek_v32/nvidia/attention.py` (param threading, skip gate, `num_decode_tokens` metadata).  The gate is bound to `dense_mha_metadata_layer_name=""` because this fork has no dense-MHA route for sparse-MLA prefills — enabling it would silently drop valid KV selection.  **Zero perf effect.  Do NOT activate until the dense-MHA route lands.**
- **`patches/hotfix-dsv4-mtp-buffer-50312.sh`** — upstream #50312 + **2 None-guards upstream lacks**.  In `models/deepseek_v4/nvidia/model.py` (2 hunks): allocates `_mtp_hidden_buffer` only when a speculator needs it (`use_eagle()`/`uses_draft_model()`), else `None`; skips the per-step `copy_` when `None`.  In `v1/worker/gpu/model_runner.py` (1 hunk, expect=2): None-guards both `get_mtp_target_hidden_states()` feed sites.  Saves ~448 MiB/rank (256 MiB/rank here) — memory ROI, no TTFT gain.
- **`patches/hotfix-dsv4-adaptive-topk-50004.sh`** — upstream #50004, verbatim.  In `models/deepseek_v4/sparse_mla.py` (2 hunks): computes `active_topk_width` from live `cm.max_seq_len`; returns packed `[num_tokens, width]` views and passes live width as the C128A kernel stride instead of full ~1M width (~1.0% E2E TTFT upstream).
- **`patches/hotfix-dsv4-skip-empty-c128-48957.sh`** — upstream #48957, **new this session**.  In `models/deepseek_v4/compressor.py` (6 hunks): imports `CUDAGraphMode`; adds `_get_c128_boundary` helper; adds `CompressorMetadata.c128_boundary`; `build()` populates it (C128 layers only, `block_size == 8`); captures `forward_context`; skips the compress→KV kernel launch when no request crosses a 128-token boundary this step (state-cache write still runs).  **Disabled under `CUDAGraphMode.FULL`** — live server runs `FULL_AND_PIECEWISE`, so the FULL-graph prefill path silently skips it.
- **`patches/hotfix-dsv4-flashmla-workspace-50298.sh`** — upstream #50298, **new this session**.  Across `models/deepseek_v4/nvidia/flashmla.py` + `models/deepseek_v4/common/ops/cache_utils.py` (6 hunks): optional `out=` on `combine_topk_swa_indices`; dummy path `forward_mqa` reserves the combined-topk int32 buffers; `_forward_prefill` requests all three buffers in one `get_simultaneous` and slices per chunk; passes `out=(...)` (no per-chunk `torch.full`/`torch.empty`).  1.88x on the combined-topk+SWA kernel upstream.

- **`start-deepseek-v4-flash-dspark.sh`**: all six scripts added to `DSV4_HOTFIX_FILES`, the sync-to-worker loop (`scp`), the worker apply loop, and the profile/echo strings.  Next fresh start applies all six + Issue #22 idempotently, then restarts both containers once.  See opt-outs below.
- **`docs/vllm-027-new-patches.md`**: status table, #48407 Stage A/B rationale, un-backported PR list (#49236 workspace pool — needs C++ op rebuild, image-level; #46789 sequence parallelism; #48993, #48047).

#### Env opt-outs

- **`DSPARK_SKIP_HOTFIX=1`** skips the six v0.27 perf backports only; Issue #22 still applies.
- **`DSPARK_SKIP_ISSUE22_HOTFIX=1`** also skips Issue #22 (fully clean baseline).
- Pass as inline prefix or `export` — a bare `VAR=1` on its own line is not exported to the start script (DO NOT do `DSPARK_SKIP_HOTFIX=1` then run the script; it silently applies everything).

## Unreleased

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
- **Automatic hotfix on start** (`start-deepseek-v4-flash-dspark.sh`): the start script syncs the hotfix to the worker, applies it to both head and worker containers after `compose up`, and restarts them so vLLM starts with the patched file.  Issue #22 always applies (baseline fix for the recipe-default KV dtype); the v0.27 perf backports opt out with `DSPARK_SKIP_HOTFIX=1`, Issue #22 with `DSPARK_SKIP_ISSUE22_HOTFIX=1`.
- **`DSPARK_SKIP_HOTFIX` env var** (`.env.dspark.example`): set to `1` to skip the v0.27 perf hotfixes (e.g. when using a pre-patched image). Issue #22 still applies; skip it too with `DSPARK_SKIP_ISSUE22_HOTFIX=1` (fully clean baseline).
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
