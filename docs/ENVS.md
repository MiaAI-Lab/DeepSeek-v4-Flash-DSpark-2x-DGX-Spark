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
| `DSPARK_BOOT_SHAPE_WARMUP` | Launcher-side (not passed to the container). `1` (default) runs `scripts/boot-shape-warmup.sh` after the smoke request: a C=1/2/4/6 + medium/long-prefill + thinking-off sweep that burns the spec-decode/prefill Triton shape buckets before traffic (issue #117). `0` skips. Warmup failure is a WARN, never a boot failure. |
| `TORCH_CUDA_ARCH_LIST` / `FLASHINFER_CUDA_ARCH_LIST` | Build/JIT arch lists |
| `NCCL_*` / `TP_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` | Fabric |
| `NCCL_IB_MERGE_NICS` | Passthrough, default **unset**. Contract for all four passthrough knobs below: a configured non-empty value passes through unchanged; an empty value is normalized to absent (the entrypoint unsets empty definitions before exec, so NCCL's built-in default and config-file values still apply and cannot be masked). NCCL's own default is `1`: it *permits* merging compatible dual-port NICs; it does not select HCAs or force arbitrary links (`NCCL_NET_MERGE_LEVEL`/`NCCL_NET_FORCE_MERGE` participate in that topology decision). `0` disables merging. |
| `NCCL_NET_GDR_LEVEL` | Passthrough, default **unset**. Upstream GPUDirect RDMA override; no effect demonstrated on the submitted GB10 stack (see `NCCL_DMABUF_ENABLE`). |
| `NCCL_NET_GDR_READ` | Passthrough, default **unset**. Upstream GPUDirect RDMA override; no effect demonstrated on the submitted GB10 stack. |
| `NCCL_DMABUF_ENABLE` | Passthrough, default **unset**. `0` disables DMA-BUF probing (workaround control). Contributor-reported observation on GB10 driver `580.173.02`, that stack only: the container reported `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED=0` and boot logs showed `via NET/IB/x` with no `/GDRDMA`; no GDR effect was demonstrated there, which is not a claim about GDR availability in general. |
| `HF_*` / `TRANSFORMERS_OFFLINE` | Hub cache behavior |
| `MTP_NUM_TOKENS` | Consumed by compose command line (not a vLLM env registry key) |
| `DRAFT_SAMPLE_METHOD` | DSpark `draft_sample_method` in `--speculative-config`. Compose default **`probabilistic`** (the previously hardcoded value). `greedy` is what the official model cards pair with `num_speculative_tokens=7` (issue #84). Consumed by the compose command line, not a vLLM env registry key. The entrypoint (and `validate-dspark-config.sh`) accept exactly `probabilistic`/`greedy` and exit nonzero on anything else, so the raw value never reaches the `--speculative-config` JSON. |
| `DSPARK_SUPPRESS_STOPS_IN_REASONING` | `1` (default): after the detokenizer hotfix, client `stop` stays dormant until `</think>`. `0` restores stock matching. Also accepts Tony's `VLLM_SUPPRESS_STOPS_IN_REASONING` via compose interpolation (not added as a compose `VLLM_*` key, so Anemll does not warn). |
| `DSPARK_SKIP_SUPPRESS_STOPS_HOTFIX` | `1` skips applying `patches/hotfix-dsv4-suppress-stops-in-reasoning.py` |
| `DSPARK_SKIP_SPIN_WAIT_HOTFIX` | `1` skips `patches/hotfix-gb10-spin-wait.sh` (issue #79: `busy_loop_s` 1s→2ms) |
| `DSPARK_ENABLE_ISSUE31_GPU_HOTFIX` | **Not** `VLLM_*`. Default `0` = stock V2 (no thinking_token_budget). `1` applies the GPU budget hotfix at boot (fail-closed). Issue #66: default-on omit-field traffic can hit a decode cliff. |
| `VLLM_API_KEY` | **Optional single-key auth** for the OpenAI endpoint, consumed natively by vLLM (its `--api-key` env alias). Empty (default) = no auth. Exactly one key. Mutually exclusive with `DSPARK_API_KEYS`. |
| `DSPARK_API_KEYS` | **Optional multi-key auth**, enforced by vLLM itself. Single-line keys use literal space/tab separators and are flattened into **one** `--api-key` flag (nargs list; repeating the flag would overwrite). Empty or space/tab-only (default) adds no flag, preserving stock behavior. Parsing trims/collapses separators, preserves order, allows duplicates, rejects CR/LF/VT/FF before empty classification, rejects backslashes, and rejects tokens starting with `-` without echoing token bytes (exit 2); it must be set in `.env.dspark`, not the shell. **Mutually exclusive with `VLLM_API_KEY`**: if both are meaningful, the entrypoint and `start-`/`smoke-`/`status-*.sh` scripts exit 2 before patch/install work. Every route outside the guarded prefixes `/v1`, `/v2`, `/inference` is keyless. On the pinned runtime that includes `POST /invocations` and `POST /generative_scoring` (both run inference unauthenticated) and the `/tokenize` / `/detokenize` utility routes, besides `/health`, `/metrics`, `/version`, `/ping`; a keyed deployment still needs network-level access control on the server port. Keys remain container argv/env, so rotation needs a stop/start; vLLM provides revocation rather than per-key request attribution. |
| `patches/hotfix-vllm-redact-api-key-log.sh` | Key-log redaction hotfix, required whenever either key variable is configured; apply + `--status` must succeed or the entrypoint fails the container before exec vllm, and `--status` exits nonzero unless every check passes. Upstream `log_non_default_args()` prints every `--api-key` value verbatim; the patch redacts that logger for both entrypoints while preserving the count as `'api_key': ['<redacted:N value(s)>']`. This closes the log channel only; keys remain visible through `docker inspect` / host `ps`. |

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
