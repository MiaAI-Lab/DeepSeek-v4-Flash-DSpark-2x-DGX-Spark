## 2026-08-19

### Fixed

- **RULER-lite never reached its advertised context lengths ([Issue #81](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/81))**: `pad_to_length` appended one haystack sentence per loop with `guard < 200`, so every cell capped at ~4.8k tokens while still exiting 0. It now bulk-pads and `run_case` fails if `/tokenize` is under 97% of the target.

## 2026-08-18

### Changed

- **Issue #52 / PR #53: assistant-final continuation hotfix is now opt-in (`DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX`, default `0` = stock)** (`patches/hotfix-dsv4-assistant-final-continuation.py`): a request whose `messages` array ends with an assistant message renders with a bare EOS and no generation header, so the model generates from a dead state — the reported agent-harness loop of empty no-op turns with hallucinated DSML markup fragments. At the previous PR head the compose entrypoint already ran this patcher unconditionally inside the encoder-copy chain; this change restores **default stock rendering** (patch file stays mounted/synced to the worker but is never invoked) and moves invocation behind an exactly-`1` gate in `docker-compose.dspark.yml`, chained with `|| exit 1` so an ON boot fails rather than serving an unpatched or unverified encoder. The patcher itself is now fail-closed: missing encoder file (prerequisite), missing anchor, or a failed post-write self-check (patched module must import and render a trailing-assistant transcript with a generation header) all exit nonzero, any failed self-check **restores the original file bytes**, and an already-patched encoder is re-validated instead of rewritten (idempotent). Documented in `.env.dspark.example` and `docs/PATCHES.md`; CPU gates in `scripts/test-assistant-final-continuation.py` use the checkpoint's real encoder signature and separate render/transition control flow, proving stock bytes plus one appended header for assistant-final input, byte identity for every non-assistant-final shape, idempotence, fail-closed/restore, and static OFF-default compose/start wiring. They are wired into `scripts/ci-validate.sh`.

  Evidence status, stated exactly: render and no-regression evidence is from prior head `f08cd6c` — a causal one-prompt A/B via `/v1/completions` where the trailing turn left open produced 183 tokens with a coherent continuation versus 400 tokens of raw `<|DSML|tool_calls>` markup when closed with EOS, plus the `wo_eos` comparison showing reopening a *complete* turn yields a 1-token empty generation. **No rescue claim**: the live no-op-turn defect did not reproduce in that session, so there is no measured stuck-harness recovery. A first gated-ON boot on `d4b31daf` failed closed before serving because a review-requested guard named the nonexistent checkpoint variable `add_generation_prompt`; the original encoder bytes were restored. Corrected code commit `0864014` passed serialized live OFF/ON proof on both ranks: OFF had effective flag `0`, no patch marker, and a stock EOS render; ON had effective flag `1`, both ranks patched and verified, preserved all 98 stock token IDs, appended exactly `<|Assistant|><think>`, and completed a live continuation with `alpha beta`. Re-running on both ranks verified idempotence. A deliberate anchor-drift boot failed with exit `1` on both ranks and never served the API.
### Fixed

- **vLLM shm spin-wait wasting Grace P-cores on TP=2 ([Issue #79](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/79))**: `patches/hotfix-gb10-spin-wait.sh` flips `SpinCondition.busy_loop_s` from `1` to `0.002` in-image before `exec vllm`. Decode IPC always lands inside the old 1s window, so `sched_yield` never fell through to sleep. Opt out with `DSPARK_SKIP_SPIN_WAIT_HOTFIX=1`. Not gated by `DSPARK_SKIP_HOTFIX`. Same one-line change as PRs #71/#74.

## 2026-08-17

### Fixed

- **Start script vs `unless-stopped` after reboot ([Issue #72](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/72))**: dockerd restores both ranks, then `./start-…` used to `exit 1` on the existing-container guard — the same code as a real failure, which turns a systemd `Restart=on-failure` unit into a retry storm against a serving cluster. If the **head** container is already up, start now **exits 3** with a hint that this is expected after reboot; real failures stay `1`. A leftover **worker** with head down still exits 1 (stale rank after a head-only reboot — rebuild, do not treat as success). Supervisors: `SuccessExitStatus=3` + `RemainAfterExit=yes`, or `DSPARK_RESTART_POLICY=no`. Documented next to `DSPARK_RESTART_POLICY` in `docs/ENVS.md` / `.env.dspark.example`. This is not an adopt/attach path and does not claim the TP group is healthy.

## 2026-08-14

### Fixed

- **Issue #55: tool-call truncation reports `finish_reason="length"` (was `"tool_calls"`), invalid JSON args dropped** (`patches/hotfix-dsv4-issue55-tool-truncation.py`): when `max_tokens` cuts a request mid-tool-call, stock vLLM's chat-serving layer reported `finish_reason="tool_calls"` even though the call never completed, and the streaming flush could deliver a `tool_calls[].function.arguments` that did not parse as JSON. A harness that replayed the broken call then saw its whole transcript rejected with HTTP 400 (the reporter's "poisoned conversation with 400s" chain). The hotfix monkey-patches `vllm/entrypoints/openai/chat_completion/serving.py` at boot: on `FinishReason.LENGTH`, both the streaming final chunk and the non-streaming final choice report `finish_reason="length"` (not `"tool_calls"`) and drop any tool_calls whose `arguments` are not a complete JSON value. Natural-stop tool calls keep `finish_reason="tool_calls"`. This is the reporter's suggested direction #1 ("report `length` even mid-call — clients can then discard or retry").

  Live A/B on 2× DGX Spark TP=2 (official 0731 @ `9e165c30`, Anemll `0.1.1`) after restart: pre-patch every truncation cell (`max_tokens` = 300/800/1500/600+high) reported `finish_reason="tool_calls"` and streaming delivered `'{"path": "/tmp/log.txt", "content": "'` as args (unterminated JSON). Post-patch: all truncation cells report `finish_reason="length"`; non-truncated (`mt=3000`, natural stop) still reports `finish_reason="tool_calls"` with valid JSON args. Poisons the transcript only if a client ignores `finish_reason`; a harness that reads `length` discards the in-flight call and the 400-chain collapses.

  Note: invalid JSON can still arrive in the incrementally-streamed args fragments (OpenAI stream protocol is append-only, no take-back). The protection is the correct `finish_reason` signal; clients following OpenAI semantics drop the in-progress call on `length`. Streaming the args is unchanged for normal tool calls.

  **No-regression audit (live, 2026-08-14)**: a 10-shape regression sweep on the running TP=2 serve confirmed the patch is monotone-safer than upstream — the verdict diverges from stock vLLM only when `FinishReason.LENGTH` fires; every other path (plain-chat stop, stop-string match, plain-chat length, reasoning-only length, natural tool-call stop, `tool_choice: required` stop, streaming plain/chat/tool natural stop, streaming tool truncated) returns the same as upstream or strictly correct (valid args preserved). Discovery while auditing: `FinishReason` is an `IntEnum`, not a `StrEnum`, so stock vLLM's `output.finish_reason == "stop"` string comparison at `serving.py:961` never matched (its `tool_choice="required"` natural-stop OR branch has been dead since upstream); this hotfix's `str(...)` on the streaming/`else` branch also fixes the latent serialization where `FinishReason.ABORT/REPETITION` would reach the client as `'2'/'4'` instead of `'abort'/'repetition'`. The streaming-args-buff-fragment limitation is protocol-level and documented in the README; no recipe-side patch can take them back.

### Fixed (earlier today)

- **Fast, hard `thinking_token_budget` for the V2 runner (Issue #31 replacement)**: explicit request budgets now use two small GPU-resident Triton kernels to force exactly the boundary `</think>` token and observe accepted MTP tokens. The hot path performs no per-step device-to-host copy, Python token-list scan, or omit-field default; requests without the field retain the existing sampler path. Natural reasoning termination and tool calls remain intact. The start smoke now exercises the budget so both kernels compile before the endpoint is handed to clients. CPU gates cover patch application/idempotence and forbid the old host-scan patterns; live TP=2 checks covered budgets 0/16/32/64/256, four concurrent requests, and a two-turn tool call.

  Live bounded 65K fresh-prefix comparison on this cluster: no-budget decode **37.2 tok/s** (1,024-token length stop) versus budgeted decode **59.3 tok/s** (256 reasoning + visible answer, 769 total), with fresh prefill **1,693–1,815 tok/s**. Short-context budgeted decode measured **67.2 tok/s**. The withdrawn implementation's ~4.5 tok/s cliff did not recur.

  **Re-verified live after merge (`2689b1f` on `main`, 2026-08-14)** on 2× DGX Spark TP=2 (official 0731 @ `9e165c30`, Anemll `0.1.1`): patch applied cleanly at boot (11/11 anchors, idempotent on re-run); boot smoke exercises the budget Triton kernels on GB10/SM12.1 with no CUDA/Triton errors. A/B c=1–6, `prompt=256`, `max_tokens=4096`, `budget=1024` vs no-budget: per-stream median decode rate **at or above baseline** at every concurrency (Δ −0.0% to +19.4%, never materially below); aggregate swings are 4096-cap-hit artifacts, not budget-kernel cost. Omit-field path unchanged (no server-side default injected). Closes #48, #31, and #34; #56 stays open.

### Docs

- **README: client-side `thinking_token_budget`**: new "Enabling the budget from a client" subsection under [Thinking-token budgets](#thinking-token-budgets) shows how to turn the opt-in on from (a) `curl`/any OpenAI-compatible client (add the field to the request body; `0` disables reasoning, `N>0` caps it at `N` reasoning tokens) and (b) pi, by setting `thinkingTokenBudget` on the model plus a `"thinking_token_budget": { "$var": "model.thinkingTokenBudget", "omitWhenUnset": true }` entry in `chatTemplateKwargs`, so the field is only attached when you set one. `pi-models.dspark.example.json` already advertises `supportsThinkingTokenBudget: true`.

- **README: `max` reasoning magnitude**: the _Thinking and `max_tokens`_ section now states that `DEFAULT_THINKING=max` produced **~50,000 reasoning chars (~12.5k tokens) on a moderate prompt** (the checkpoint's `reasoning_effort=max` directive is "do not stop until … no error remains undiscovered"), so sizing `max_tokens` for `max` means tens of thousands of tokens — and points clients at `thinking_token_budget` as the supported way to hard-cap reasoning. Documents issue #56 (closed as resolved-by-#48).

- **README rewrite for scanability**: numbered quick start at the top; default profile and "what speed to expect" in short tables; dated benches and historical lanes moved to [`results/RESULTS-2026-08-14.md`](results/RESULTS-2026-08-14.md) (includes the live 256–128K × c=1/2/4/6 matrix). Old README anchors for KV, checkpoint, `max_tokens`, and Experimental Vision are kept.

## 2026-08-13

### Changed

- **GitHub Actions `validate` workflow**: on every push/PR, `scripts/ci-validate.sh` runs CPU-only recipe gates (shell syntax, patch compile, #21/#26 v2 unit tests, #49486/#48407 equality gate, overlay COPY check) and refuses to re-ship the withdrawn #31/#34 thinking-budget hook or a #26 v1 coordinator. This does **not** replace a live 2× Spark decode or tool-eval run.

- **Reverted the #31/#34 `thinking_token_budget` patch (bug + hotfix)**: the V2 sampler hook, `ThinkingBudgetState` O(n) per-step scan, and omit-field defaults (`DEFAULT_THINKING_TOKEN_BUDGET=32768`, `DEFAULT_MAX_TOKENS=131072`) are **fully removed** from compose, start, and `patches/`. That path was the [#39](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/39) **~4.7× decode tok/s cliff** at long context (Python scan × MTP rows on every request after #34). It is not incremental-scanned and left in; it is gone. Stock Anemll V2 again rejects `thinking_token_budget` (HTTP 400). Size client `max_tokens` (or set `DEFAULT_THINKING` below `max`) so a long think cannot empty `content`.

- **Current tip should not produce those two regressions**: together with **#26 v2** (SWA may shrink the common prefix hit again; see #36 below), this recipe no longer ships a mechanism that should **drop decode tok/s** the way #31/#34 did, or **garble** the way #26 v1 did (warm 21k DSML/CJK salad, invented tool names, stale cross-turn KV). #27 stays. `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096` stays. Remaining `high`/`max`+tools model/template noise is not a leftover budget or v1-cache patch.

### Fixed

- **Client `stop` strings no longer fire inside `<think>`**: vLLM v1 matches harness stops (`Question:`, lm-eval `stop[:4]`, …) against the whole stream, so think-in-prompt requests die mid-reason with `content: null`. Port of [tonyd2wild Patch 5 / Capicua25x](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark/blob/main/patches/0005-suppress-stops-in-reasoning.patch) onto the **Anemll** detokenizer (`/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/detokenizer.py`), not the Stage-C `/opt/env/...` bind-mount. Guard only if the last prompt token is the reasoning start marker; speculative chunks that contain `</think>` only evaluate stops *after* the marker. Default on; `DSPARK_SUPPRESS_STOPS_IN_REASONING=0` (or `VLLM_SUPPRESS_STOPS_IN_REASONING=0`) disables the guard; `DSPARK_SKIP_SUPPRESS_STOPS_HOTFIX=1` skips the file. Unit: `scripts/test-suppress-stops-in-reasoning.py`. This is **not** the withdrawn `#31` thinking-budget hook.

  Live on 2× Spark TP=2 (official 0731, this hotfix applied in the running container): think + `stop: ["Question:"]` returned `content` (`17 + 25 = 42`, 99 completion tokens) instead of null; thinking-off still cut at `Question:`; `PING-OK-17` and `low`+tools `grep(/tmp, Clash)` unchanged. Restart both ranks.

- **Worker Exited (1) on every start ([Issue #38](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/38))**: the start script applied `.sh` hotfixes then `compose restart`/`stop` while vLLM was loading, which tore down head's TCPStore under rank1 (or hung the operator on `Stopping`). Those scripts now run in the compose entrypoint **before** `exec vllm`, so start no longer stops a fresh boot. Compose has `restart: unless-stopped` and `stop_grace_period: 10s`; `./stop-…` `docker rm -f`s first.

- **Warm shared-prefix DSML / CJK salad after #26 ([Issue #36](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/36))**: the v1 hybrid-SWA hotfix refused to let sliding-window groups shrink `curr_hit_length`. A 21k Hermes system prefix then reported a 100% MLA cache hit while SWA had no retained tail at that length (different user turns move the replay boundary; `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096` only keeps sparse checkpoints). Prefill was skipped and SWA KV was padded with nulls → leftover `</｜DSML｜parameter>` / Chinese / loops. v2 restores the min-across-groups length so the common hit stops at the last SWA tail. Warm hits stay large because of retention, not because we ignore a missing SWA window. Unit: `scripts/test-issue26-swa-min-v2.py`. Restart required.

- **#31/#34 thinking-budget hook scanned the full prefix every decode step** *(withdrawn later the same day)*: after #34 every omitted-field request had a budget, so the V2 sampler hook no longer early-returned. Incremental scan was a stopgap; the whole hook is now removed (see Changed above).

- **Blank turns on stock OpenAI clients ([Issue #34](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/34))** *(withdrawn later the same day)*: omit-field defaults `DEFAULT_THINKING_TOKEN_BUDGET=32768` / `DEFAULT_MAX_TOKENS=131072` were added, then removed with the hook.

### Docs

- README: `thinking_token_budget` is not supported on this V2 serve; size `max_tokens` instead (see Changed above).

### Fixed

- **`thinking_token_budget` rejected on DSpark / V2 ([Issue #31](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/31))** *(hotfix later withdrawn; see Changed above)*: stock Anemll `0.1.1` rejects the field on the V2 runner (HTTP 400). DSpark cannot use `VLLM_USE_V2_MODEL_RUNNER=0`. Clients must size `max_tokens` so a `DEFAULT_THINKING=max` think cannot empty `content`.

- **Prefix-cache collapse at 32K+ x8 ([Issue #26](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/26))** (`1f9765e`): DSV4-Flash + DSpark on Anemll `0.1.1` runs four KV groups (1× `MLAAttentionSpec` + 3× `SlidingWindowMLASpec`). `HybridKVCacheCoordinator.find_longest_cache_hit` takes the min hit length across groups, so a sliding-window group that frees old blocks by design zeroes the common hit. Warm x8 32K/62K then fully re-prefills (`prefix_cache_hits_total +0`, warm wall == cold). Independent of #27.

  Coordinator-only is necessary but not sufficient: at 44K+ x8, dense SWA tails also evict MLA prefix blocks from the shared pool.

  **Fix (both required):**
  1. `patches/hotfix-dsv4-issue26-hybrid-swa-min.py` — originally skipped SWA shrink of `curr_hit_length` (v1). **Superseded by v2** (issue #36): SWA may shrink again; retention (below) is what keeps warm hits.
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
