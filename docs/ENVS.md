# Environment variable matrix (Anemll 0.1.1 vs Stage-C overlay)

This recipe defaults to the prebuilt image:

```text
ghcr.io/anemll/dspark-vllm-gx10:0.1.1
```

A large set of `VLLM_DSPARK_*` / extra B12X knobs still appear in historical
Stage-C docs and in `recipe/overlay/vllm/envs.py`. **Those symbols are
registered in the Stage-C overlay build**, not necessarily in the Anemll
prebuilt image.

vLLM validates process environment keys that start with `VLLM_`. Unknown keys
log:

```text
Unknown vLLM environment variable detected: VLLM_…
```

and are **ignored** (warning only; serve still starts).

> **Important:** missing env registration does **not** mean DSpark or the Keys
> concurrency patches are absent from Anemll. Logic may be baked into the image
> without exposing every Stage-C kill-switch. Conversely, setting a Stage-C-only
> env on Anemll does **not** enable that kill-switch.

Audit date: **2026-07-29**, image tag **`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`**,
by inspecting `vllm.envs.environment_variables` inside the container and
comparing to `recipe/overlay/vllm/envs.py` in this repo.

Re-check after image bumps:

```bash
docker run --rm --entrypoint python3 ghcr.io/anemll/dspark-vllm-gx10:0.1.1 - <<'PY'
import pathlib, vllm
ns = {}
exec(compile((pathlib.Path(vllm.__file__).parent / "envs.py").read_text(), "envs.py", "exec"), ns)
keys = ns["environment_variables"]
for k in sorted(keys):
    if any(s in k for s in ("B12", "DSPARK", "DSV4", "SPARSE_INDEXER", "FLASHINFER_SAMPLER")):
        print(k)
PY
```

---

## Compose / `.env` knobs by lane

### A. Safe on Anemll 0.1.1 (registered `VLLM_*` or non-`VLLM_` runtime)

| Variable | Role |
|----------|------|
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | Allow long context configs |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` | Sparse indexer workspace cap |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | Profiler / capture estimate |
| `VLLM_USE_FLASHINFER_SAMPLER` | FlashInfer sampler |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | `sample_tokens` RPC deadline (compose default **1800**; stock vLLM is 300). Issue #65/#87: mid-serve CuTeDSL/TileLang JIT can exceed 300s and kill EngineCore on TP=2. |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | Set `0` to opt out of DS4's automatic breakable-graph mode and retain regular CUDA graphs |
| `VLLM_USE_B12X_MOE` | Enable B12X MoE path |
| `VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM` | Experimental W4A16 selector |
| `VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M` | Experimental W4A16 selector |
| `VLLM_B12X_W4A16_FORCE_TILE_CONFIG` | Experimental W4A16 selector |
| `VLLM_HOST_IP` | Distributed bind address |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | Issue #26: sparsify SWA prefix-cache checkpoints (default 4096). This is the warm-hit fix; the coordinator must still let SWA shrink the common hit (hotfix v2, issue #36). |
| `VLLM_CACHE_ROOT` | vLLM cache root (compose sets path) |
| `CUTE_DSL_ARCH` | **Not** `VLLM_*` — CuTeDSL/b12x compile target (`sm_121a` on GB10) |
| `TILELANG_CACHE_DIR` | **Not** `VLLM_*`. Compose default `/cache/huggingface/tilelang-cache` (HF volume). Issue #65: in-image `~/.tilelang/cache` dies on container recreate. |
| `TRITON_CACHE_DIR` | **Not** `VLLM_*`. Compose default `/cache/huggingface/triton-cache` (HF volume). Issue #117: in-image `~/.triton/cache` dies on container recreate, so known shapes re-JIT mid-serve after every restart — and a compiling rank can stall its TP peer past torch's 600s NCCL watchdog. |
| `DSPARK_BOOT_SHAPE_WARMUP` | Launcher-side (not passed to the container). `1` (default) runs `scripts/boot-shape-warmup.sh` after the smoke request. `_prepare_dflash_inputs_kernel` keys on `next_pow2(scheduled_tokens + 6)` only — request concurrency does not enter the key — so coverage comes from a deterministic ladder of exact-token plain completions (s = 1/6/20/45/100/200, each verified via an authenticated `/tokenize` before firing) hitting every live BLOCK key {8,16,32,64,128,256}. Chat arms C=1/2/4/6 up to the launcher's resolved `MAX_NUM_SEQS` cover both bounded longer prompts and ordinary short requests with client-default generation settings; medium/long-prefill and thinking-off cover other batch-keyed variants. `0` skips. Warmup failure is a WARN, never a boot failure. |
| `DSPARK_WARMUP_REQ_TIMEOUT` | Launcher-side (read by `scripts/boot-shape-warmup.sh`, not passed to the container). Per-request curl `--max-time`, seconds, default **240** — first-ever boots pay real Triton compiles per request. Sequential worst case is 35 × timeout at the shipped `MAX_NUM_SEQS=6` default (23 × at `MAX_NUM_SEQS=4`) before the sweep exits nonzero and the launcher WARNs (non-fatal); raise it rather than skipping the sweep if first-boot compiles exceed the default. |
| `TORCH_CUDA_ARCH_LIST` / `FLASHINFER_CUDA_ARCH_LIST` | Build/JIT arch lists |
| `NCCL_*` / `TP_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` | Fabric |
| `NCCL_IB_MERGE_NICS` | Passthrough, default **unset**. Contract for all four passthrough knobs below: a configured non-empty value passes through unchanged; an empty value is normalized to absent (the entrypoint unsets empty definitions before exec, so NCCL's built-in default and config-file values still apply and cannot be masked). NCCL's own default is `1`: it *permits* merging compatible dual-port NICs; it does not select HCAs or force arbitrary links (`NCCL_NET_MERGE_LEVEL`/`NCCL_NET_FORCE_MERGE` participate in that topology decision). `0` disables merging. |
| `NCCL_NET_GDR_LEVEL` | Passthrough, default **unset**. Upstream GPUDirect RDMA override; no effect demonstrated on the submitted GB10 stack (see `NCCL_DMABUF_ENABLE`). |
| `NCCL_NET_GDR_READ` | Passthrough, default **unset**. Upstream GPUDirect RDMA override; no effect demonstrated on the submitted GB10 stack. |
| `NCCL_DMABUF_ENABLE` | Passthrough, default **unset**. `0` disables DMA-BUF probing (workaround control). Contributor-reported observation on GB10 driver `580.173.02`, that stack only: the container reported `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED=0` and boot logs showed `via NET/IB/x` with no `/GDRDMA`; no GDR effect was demonstrated there, which is not a claim about GDR availability in general. |
| `HF_*` / `TRANSFORMERS_OFFLINE` | Hub cache behavior |
| `MTP_NUM_TOKENS` | Consumed by compose command line (not a vLLM env registry key) |
| `DRAFT_SAMPLE_METHOD` | DSpark `draft_sample_method` in `--speculative-config`. Compose default **`probabilistic`** (the previously hardcoded value). `greedy` is what the official model cards pair with `num_speculative_tokens=7` (issue #84). Consumed by the compose command line, not a vLLM env registry key. The entrypoint (and `validate-dspark-config.sh`) accept exactly `probabilistic`/`greedy` and exit nonzero on anything else, so the raw value never reaches the `--speculative-config` JSON. |
| `DSPARK_FLASHINFER_AUTOTUNE` | **Not** `VLLM_*` — reversible issue #32 diagnostic switch, consumed by the compose command line (not a vLLM env registry key). Unset/empty or `1` emits the shipped `--enable-flashinfer-autotune`; `0` emits the proven `--no-enable-flashinfer-autotune` form (the one `docker-compose.vl-sidecar.yml` already uses). Any other value exits 2 in the container entrypoint and in `validate-dspark-config.sh` **before** vLLM exec. Diagnostic and reversible: comment/unset the line and stop/start to return to the exact previous argv. |
| `DSPARK_DIAG_FULL_DECODE_ONLY` | **Not** `VLLM_*` — reversible issue #32 diagnostic switch, consumed by the compose command line. Unset/empty or `0` adds **no** `--compilation-config` argument (effective argv identical to current main); `1` emits exactly one additional argv value: `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` (compact JSON as a single token; argument-parser-verified offline against pinned image 0.1.1). Any other value exits 2 before vLLM exec. Purely diagnostic for issue #32 triage — not a fix and not a performance claim; revert by unsetting and restarting. |
| `DSPARK_SUPPRESS_STOPS_IN_REASONING` | `1` (default): after the detokenizer hotfix, client `stop` stays dormant until `</think>`. `0` restores stock matching. Also accepts Tony's `VLLM_SUPPRESS_STOPS_IN_REASONING` via compose interpolation (not added as a compose `VLLM_*` key, so Anemll does not warn). |
| `DSPARK_SKIP_SUPPRESS_STOPS_HOTFIX` | `1` skips applying `patches/hotfix-dsv4-suppress-stops-in-reasoning.py` |
| `DSPARK_SKIP_SPIN_WAIT_HOTFIX` | `1` skips `patches/hotfix-gb10-spin-wait.sh` (issue #79: `busy_loop_s` 1s→2ms) |
| `DSPARK_ENABLE_ISSUE31_GPU_HOTFIX` | **Not** `VLLM_*`. Default `0` = stock V2 (no thinking_token_budget). `1` applies the GPU budget hotfix at boot (fail-closed). Issue #66: default-on omit-field traffic can hit a decode cliff. |
| `VLLM_API_KEY` | **Optional single-key auth** for the OpenAI endpoint, consumed natively by vLLM (its `--api-key` env alias). Empty (default) = no auth. Exactly one key. Mutually exclusive with `DSPARK_API_KEYS`. |
| `DSPARK_API_KEYS` | **Optional multi-key auth**, enforced by vLLM itself. Single-line keys use literal space/tab separators and are flattened into **one** `--api-key` flag (nargs list; repeating the flag would overwrite). Empty or space/tab-only (default) adds no flag, preserving stock behavior. Parsing trims/collapses separators, preserves order, allows duplicates, rejects CR/LF/VT/FF before empty classification, rejects backslashes, and rejects tokens starting with `-` without echoing token bytes (exit 2); it must be set in `.env.dspark`, not the shell. **Mutually exclusive with `VLLM_API_KEY`**: if both are meaningful, the entrypoint and `start-`/`smoke-`/`status-*.sh` scripts exit 2 before patch/install work. Every route outside the guarded prefixes `/v1`, `/v2`, `/inference` is keyless. On the pinned runtime that includes `POST /invocations` and `POST /generative_scoring` (both run inference unauthenticated) and the `/tokenize` / `/detokenize` utility routes, besides `/health`, `/metrics`, `/version`, `/ping`; a keyed deployment still needs network-level access control on the server port. Keys remain container argv/env, so rotation needs a stop/start; vLLM provides revocation rather than per-key request attribution. |
| `patches/hotfix-vllm-redact-api-key-log.sh` | Key-log redaction hotfix, required whenever either key variable is configured; apply + `--status` must succeed or the entrypoint fails the container before exec vllm, and `--status` exits nonzero unless every check passes. Upstream `log_non_default_args()` prints every `--api-key` value verbatim; the patch redacts that logger for both entrypoints while preserving the count as `'api_key': ['<redacted:N value(s)>']`. This closes the log channel only; keys remain visible through `docker inspect` / host `ps`. |

#### Issue #32 GB10 memory observer — host-side only (`DSPARK_GB10_OBSERVER*`)

External, report-only memory/NVRM flight recorder run **by the launcher on
each host** (head and worker independently) — never inside Compose or the
containers, and never passed through the container environment. It appends
newline-delimited JSON records (MemAvailable, memory PSI, sanitized NVRM
0x51/Xid kernel facts) under the state directory and takes no action beyond
its own validated `stop`. Unset `DSPARK_GB10_OBSERVER` keeps current behavior
exactly: the launchers issue no observer command at all. Attach to an
already-running cluster at any time (`observer start` locally + remotely);
no model restart involved. See README §"GB10 memory observer" for usage.

| Variable | Default | Notes |
|----------|---------|-------|
| `DSPARK_GB10_OBSERVER` | unset (= off) | Opt-in master switch gating the start and stop hooks (`status`/`logs` report observer state opportunistically and are never gated on it). Exactly three states: unset/empty or `0` = off (no observer command issued), `1` = on. Any other value aborts `./start-deepseek-v4-flash-dspark.sh` with exit 2 **before** vLLM startup. |
| `DSPARK_GB10_OBSERVER_INTERVAL` | `2` | Seconds between samples. Valid: positive finite numbers. Anything else (including `inf`/`nan`) fails the observer command (exit 2) — serving is never affected. |
| `DSPARK_GB10_OBSERVER_STATE_DIR` | empty → `${XDG_STATE_HOME:-$HOME/.local/state}/dspark-observer` | Records live here (outside git/container mounts), rotated at 16 MiB × 4 files. Empty behaves like unset. Set it per node if the worker's home differs from the head's — each host resolves this default itself. |
| `DSPARK_GB10_OBSERVER_AUTOSTOP` | `1` | Only unset/empty, `0` or `1`. `1`: stop hooks terminate the observer after rank teardown (outside `STOP_FAILURES`). `0`: observer keeps recording across restarts. Any other value: with the observer enabled, `./stop-deepseek-v4-flash-dspark.sh` exits 2 **after** rank teardown. |
| `DSPARK_GB10_OBSERVER_JOURNALCTL` | unset → `journalctl` | Test seam overriding the journal binary/path; leave unset in production. |

Worker-local defaults: the start hook ships per-node config to the worker only
for variables that are actually set in the head's process environment
(`%q`-quoted into the remote command line). Anything left unset — including
the state directory — is resolved **by the worker itself**, so its records land
under the worker's own `$HOME`/`XDG_STATE_HOME`, never the head's. The same
applies to stop/status/logs: they pass an explicit state-dir override to the
remote side only when one is configured.

Failure semantics: those two exit-2 cases are the only launcher-level effects
of bad observer configuration — start aborts before vLLM comes up; stop exits
after rank teardown. Only once configuration is valid does a failing observer
*command* stay WARN-only: serving and the stop verdict are never affected.

### B. Stage-C / overlay-registered only (warn + no-op on Anemll 0.1.1)

These appear in `recipe/overlay/vllm/envs.py` and in older validated Stage-C
lanes. On Anemll **0.1.1** they are **not** in `environment_variables` and only
produce unknown-env warnings if injected.

| Variable | Stage-C intent (summary) |
|----------|---------------------------|
| `VLLM_USE_B12X_WO_PROJECTION` | B12X WO projection path |
| `VLLM_DSPARK_CONFIDENCE_THRESHOLD` | Draft confidence threshold |
| `VLLM_DSPARK_CONFIDENCE_SCHEDULER` | Confidence scheduler mode |
| `VLLM_DSPARK_LOCAL_ARGMAX` | Local argmax draft path |
| `VLLM_DSPARK_REPLICATE_MARKOV_W1` | Markov W1 replicate |
| `VLLM_DSPARK_FUSED_MARKOV_ARGMAX` | Fused Markov argmax |
| `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK` | GPU rejected-context mask (Keys ragged path switch in overlay) |
| `VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT` | Reference KV quant/dequant |
| `VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP` | Hardware scheduler early stop |
| `VLLM_DSV4_B12X_COMPRESSED_MLA` | Compressed MLA experiment |
| `VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE` | Defer target cudagraph capture |
| `VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE_EXACT` | Exact defer variant |

Default Anemll compose **does not** inject these. For Stage-C images, merge:

```bash
docker compose --env-file .env.dspark \
  -f docker-compose.dspark.yml \
  -f docker-compose.stage-c.override.yml \
  up -d
```

(see `docker-compose.stage-c.override.yml`).

### C. Not registered as `VLLM_*` on either lane (or host-only)

| Variable | Notes |
|----------|--------|
| `VLLM_TRITON_MLA_SPARSE` | Not in Anemll 0.1.1 registry; not found as overlay registration in the same form — avoid on Anemll |
| `VLLM_SKIP_INIT_MEMORY_CHECK` | Not in Anemll 0.1.1 registry — avoid on Anemll |
| `DSPARK_SLOT_CLAMP` | Non-`VLLM_` prefix (no unknown-`VLLM_` warning). Only meaningful if the image reads it; treat as Stage-C/overlay unless confirmed |
| `B12X_W4A16_TC_DECODE` | Non-`VLLM_` package/debug knob |
| `VLLM_HOST` / `VLLM_PORT` | Used by **compose command substitution** / start scripts, not as in-process vLLM config envs in the same way as registry keys |
| `DSPARK_MODEL`, `DSPARK_REVISION`, `DSPARK_VLLM_IMAGE`, `ENABLE_VLLM_GB10_PATCH`, … | Launcher / compose only |
| `DSPARK_RESTART_POLICY` | Compose `restart:` (default `unless-stopped`, issue #38). After a reboot, dockerd restores the ranks, so `./start-…` exits **3** (already running) rather than 1. Supervising the launcher: set systemd `SuccessExitStatus=3` + `RemainAfterExit=yes`, or set `DSPARK_RESTART_POLICY=no` if the unit owns start/stop. Exit 3 does **not** prove the TP group is healthy (head-only reboot can leave a stale worker). |
| `DSPARK_STOP_GRACE` | Compose `stop_grace_period` (default `10s`; do not use 180s — hangs stop) |


---

## Recommended defaults by image

### Anemll `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (repo default)

Keep the slim set in `.env.dspark.example` + `docker-compose.dspark.yml`:

- Serve profile: `MTP_NUM_TOKENS=5`, capture `max_num_seqs * (k+1)`, `GPU_MEMORY_UTILIZATION≈0.80`
- `VLLM_USE_BREAKABLE_CUDAGRAPH=0` (explicit opt-out; omission auto-enables the slower breakable path on DS4)
- `VLLM_USE_B12X_MOE=1`
- `CUTE_DSL_ARCH=sm_121a` (GB10 CuTeDSL target; prevents slower JIT fallbacks)
- Do **not** rely on Stage-C-only `VLLM_DSPARK_*` for behavior on this tag

### Stage-C `vllm-dspark-runtime:dspark-nvfp4-stage-c`

- Build via `./build-dspark-vllm-runtime.sh`
- Set `DSPARK_VLLM_IMAGE=vllm-dspark-runtime:dspark-nvfp4-stage-c`
- Enable the Stage-C override compose file and the Stage-C block in `.env.dspark.example`
- Then the Keys-oriented switches (e.g. `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`) are meaningful

---

## What this does *not* claim

- It does **not** invalidate published Anemll decode benches. Throughput can be
  real while unused envs only add log noise.
- It does **not** assert Anemll lacks concurrency fixes—only that several
  **env kill-switches** from the overlay are not exposed on 0.1.1.
- Image tags after 0.1.1 may register more keys; re-run the audit snippet above.
