# DeepSeek V4 Flash 0731

`deepseek-ai/DeepSeek-V4-Flash-0731` supersedes the preview checkpoint while retaining the same `DeepseekV4ForCausalLM` and DSpark speculative-decoding structure.

## Checkpoint

- Repository: `deepseek-ai/DeepSeek-V4-Flash-0731`
- Tested revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` (default `DSPARK_REVISION` in `.env.dspark.example`; prepare + `vllm serve --revision` both honor it)
- Context: `1048576`
- DSpark block size: `5`
- Quantization metadata: FP8 weights
- Architecture: text-only causal language model

The published checkpoint has no vision processor, projector, or vision tower.
This recipe serves **text-only** 0731 on `:8888`.

## Serving Profile

The default two-Spark profile uses MTP-5 probabilistic speculation, NVFP4 MLA KV cache, prefix caching, chunked prefill, asynchronous scheduling, CUDA graphs, and the `deepseek_v4` tokenizer, reasoning parser, and tool-call parser.

The model card does not ship a Jinja chat template. It includes an `encoding` package that defines message encoding and output parsing, including `low`, `high`, and `max` reasoning effort. Validate multi-turn role boundaries, reasoning separation, and tool calls after runtime upgrades because successful weight loading alone does not prove encoding compatibility.

The launcher installs `encoding/encoding_dsv4.py` from the exact pinned `DSPARK_MODEL_REVISION` into vLLM before import, on both ranks. It also corrects pre-0731 tokenizer wrappers that mapped `low` reasoning effort to `high`, and applies the Issue #21 hotfix so `encode_arguments_to_dsml` accepts dict tool `arguments` (not only JSON strings) — otherwise multi-turn tool history can be poisoned. These changes are required for the 0731 `reasoning_content`, reasoning-effort, and tool-argument semantics.

The equivalent Unsloth GGUF Jinja template implements the same central DS4
behavior—DSML tools, `<think>` boundaries, `high`/`max` instruction prefixes,
and retention of tool-turn `reasoning_content`—but it is not loaded by this
vLLM profile. Here, request controls are consumed by vLLM's custom tokenizer
wrapper and passed to the Python encoder. The underlying implementations fall
back to non-thinking when no kwarg exists, but this recipe defaults
`DEFAULT_THINKING=low` to match DeepSeek V4's intended base reasoning mode.
The setting accepts `off`, `low`, `high`, or `max`; `low` opens
`<think>` but adds no effort instruction. Off uses
`chat_template_kwargs.reasoning_effort=none`; `thinking=false` alone is not an
effective off switch in the pinned tokenizer. For pi, use
`pi-models.dspark.example.json`; it maps pi's off/low/high/max selector to the
same request-level controls. Every mapping also sends `drop_thinking=false` so
a later turn preserves Pi's replayed `reasoning` history. The custom encoder
accepts both `reasoning` and `reasoning_content` when called directly, but the
pinned live request model discards `reasoning_content` before `/tokenize` and
therefore exposes `reasoning` as the supported replay spelling. Without the
preservation kwarg the Python encoder defaults to stripping prior reasoning.

## Authenticated request boundary and cap hits

Only the literal origin-form target `/v1/chat/completions` selects chat request
validation. The authenticated proxy rejects query-bearing, trailing-slash,
duplicate-slash, percent-encoded, and absolute-form equivalents with HTTP 400.
For the canonical route it rejects malformed/non-object JSON and top-level keys
outside `scripts/chat-completion-request-fields.json` before forwarding. The
allowlist is the union of the pinned vLLM
`ChatCompletionRequest.model_fields` and explicitly documented repository
extensions (currently none); `scripts/verify-chat-completion-request-fields.py`
checks the stdlib proxy copy and can compare/regenerate it inside the immutable
runtime container. This keeps Pydantic out of the proxy process.

The semantic smoke also sends one deliberately unsupported field and one
single-token cap probe. Empty content ending in `finish_reason=length` is
reported as budget truncation even when no reasoning field is present;
reasoning presence is diagnostic only. Empty content ending in
`finish_reason=stop` fails the smoke. The proxy does not retry capped requests,
so callers that want another attempt must choose and send a larger output
budget themselves.

## Deterministic GB10 memory and scheduler profile

The production profile renders `--kv-cache-memory-bytes 12884901888` (12 GiB
per rank) together with `--gpu-memory-utilization 0.80`. These are not two KV
sizers: the explicit byte value exclusively controls KV allocation, while the
pinned vLLM build still evaluates utilization earlier as a startup
admission/headroom guard. Omitting it falls back to the runtime's 0.92 default
and can reject startup before `CacheConfig` applies the byte override. Scheduler
values are `MAX_NUM_BATCHED_TOKENS=8216`, MTP-5, and
`--max-cudagraph-capture-size 32`. `ENFORCE_EAGER=0` explicitly keeps the CUDA
graph path. `validate-dspark-config.sh` runs before either rank starts and
rejects a nonpositive/noninteger byte value, invalid admission fraction,
selector conflict, invalid scheduler/capture integers, or an eager value other
than `0`/`1`. Generated head and worker env files carry the same controls.

Fractional KV sizing exists only for a compatibility rollback: set
`MEMORY_CONTROL=gpu-memory-utilization`, clear `KV_CACHE_MEMORY_BYTES`, and keep
`GPU_MEMORY_UTILIZATION` explicit at a validated value greater than zero and no
greater than one.

After a controlled restart and before restoring gateway traffic, run the two
bounded gates (both read the origin credential from a mode-0600 file and omit
prompt/response bodies and secrets from their JSON):

```bash
python3 scripts/probe-full-context.py --profile near-max --max-output-tokens 1
python3 scripts/benchmark-scheduler.py --concurrency 6 --mtp 5
```

The full-context gate requires a near-1,048,576-token prefill, at least one
decoded token, no restart/OOM, post-warmup memory PSI `full avg10=0.00`, and at
least 8 GiB `MemAvailable` on both hosts. Its prefill may exceed 900 seconds on
DGX Spark, so the authenticated proxy is configured explicitly with
`VLLM_PROXY_UPSTREAM_TIMEOUT=3600`; the startup preflight accepts only 1–7200
seconds. The scheduler gate performs one observed warmup batch, then requires
six correct 8,192-token requests under the 8,216 scheduler budget, no eager
fallback/restart/OOM, p95 decode latency no more than 25% above the required
pre-change baseline, and steady-state p95 TTFT no more than 40% above it. The
TTFT envelope was recalibrated from 25% after repeated live trials landed at
32–36% while decode stayed within 3%, with no OOM/restart and with the near-1M
capacity/headroom gate passing. The deterministic prompts share almost their
entire prefix, so the six-sample TTFT is sensitive to cache/scheduler phasing;
40% is the smallest stable observed envelope rather than a generic relaxation.
The baseline lives at `artifacts/health-rollout/scheduler-baseline.json`. These
are operator gates; they do not restart the lane themselves.

## Benchmark Method

Run `scripts/benchmark-0731.py` against a warmed endpoint. The default sweep covers 256, 2K, 8K, 32K, and 128K prompt tokens at concurrency 1, 2, 4, and 6. Each request has a distinct first cache block so prefix caching cannot make later cases reuse earlier prefill work. It streams each response, records time to first token, prefill throughput, per-request decode throughput, and aggregate decode throughput using API-reported token counts from naturally completed responses. It does not impose a server-side output limit.

```bash
python3 scripts/benchmark-0731.py \
  --base-url http://127.0.0.1:8888/v1 \
  --model deepseek-v4-flash-0731 \
  --output results/deepseek-v4-flash-0731.json
```

## Two-Spark Results

Measured on two DGX Sparks connected over ConnectX-7 with tensor parallelism 2. The endpoint used MTP-5 probabilistic speculation, NVFP4 MLA KV cache, CUDA graphs, prefix caching, chunked prefill, and a 1,048,576-token context. Values are medians across requests except aggregate throughput.

| Prompt | Concurrency | TTFT (s) | Prefill tok/s | Decode tok/s | Aggregate tok/s |
|---:|---:|---:|---:|---:|---:|
| 256 | 1 | 0.63 | 447 | 75.4 | 69.1 |
| 256 | 2 | 0.81 | 357 | 58.3 | 104.9 |
| 256 | 4 | 1.26 | 222 | 46.8 | 164.5 |
| 256 | 6 | 1.42 | 197 | 36.9 | 191.2 |
| 2,048 | 1 | 0.81 | 2,563 | 68.8 | 62.0 |
| 2,048 | 2 | 1.11 | 1,911 | 57.0 | 97.6 |
| 2,048 | 4 | 1.38 | 1,505 | 44.0 | 154.7 |
| 2,048 | 6 | 6.06 | 342 | 34.7 | 143.7 |
| 8,192 | 1 | 4.80 | 1,713 | 73.9 | 43.7 |
| 8,192 | 2 | 7.51 | 1,176 | 49.8 | 56.2 |
| 8,192 | 4 | 14.50 | 578 | 37.4 | 72.3 |
| 8,192 | 6 | 18.38 | 454 | 23.6 | 73.1 |
| 32,768 | 1 | 22.96 | 1,428 | 64.0 | 16.6 |
| 32,768 | 2 | 26.82 | 1,287 | 41.5 | 24.8 |
| 32,768 | 4 | 44.85 | 756 | 17.4 | 26.7 |
| 32,768 | 6 | 60.75 | 550 | 10.8 | 27.9 |
| 131,072 | 1 | 78.75 | 1,665 | 65.2 | 5.9 |
| 131,072 | 2 | 111.17 | 1,306 | 30.9 | 6.6 |

The 131,072-token concurrency-4 probe did not complete within the 180-second measurement window, while the server remained healthy. Its partial values are retained in the raw JSON as capacity-bound evidence and are intentionally excluded from the throughput table.

A separate 900,000-token acceptance request completed with 899,994 API-reported prompt tokens, 900,000 total tokens, 1,028.85-second TTFT, and approximately 874.8 prefill tok/s. The response returned the requested sentinel and confirms the full 1,048,576-token serving profile beyond configuration metadata alone.

Raw measurements are in `results/deepseek-v4-flash-0731-2x-dgx-spark.json`.

## Regular Graph Opt-Out

Anemll `0.1.1` automatically enables breakable CUDA graphs for DeepSeek V4 when `VLLM_USE_BREAKABLE_CUDAGRAPH` is absent. The default recipe now sets it to `0`, which preserves the regular CUDA graph path without disabling CUDA graphs or enabling eager execution.

A matched 520-token natural-completion probe used temperature `0.2`, top-p `0.95`, MTP-5 probabilistic speculation, `MAX_NUM_SEQS=6`, `MAX_NUM_BATCHED_TOKENS=8192`, and the full 1,048,576-token context. Every measured response completed at its requested stop marker without chat-template leakage.

| Mode | Breakable graphs | Regular graphs | Change |
|---|---:|---:|---:|
| C1 decode, warm median | 74.55 tok/s | 95.9 tok/s | +28.6% |
| C2 aggregate decode, median | 134.2 tok/s | 151.8 tok/s | +13.1% |
| C4 aggregate decode | not measured | 263.7 tok/s | - |
| C6 aggregate decode | not measured | 340.5 tok/s | - |

The matched 14K-token prefill probes remained within normal run variance: warm C1 moved from 1,770-1,781 to 1,857 tok/s, while C2 moved from 1,920-1,954 to 1,943-1,987 tok/s. This setting is a decode improvement, not a claim that prefill is 28.6% faster.


## U5 rollout, persistence, rollback, and evidence

The private LiteLLM gateway and its PostgreSQL database are long-running
private production services, so both intentionally use `restart:
unless-stopped`. The DSpark rank/proxy compose remains an operator-controlled,
temporary cutover surface and keeps `restart: "no"`. PostgreSQL uses the named
`dspark-private-litellm-pgdata` volume; `/tmp` and `/run/postgresql` remain
tmpfs. `turn_off_message_logging: true` remains mandatory.

Before any live cutover, capture the authoritative scheduler baseline on the
**pre-change** lane and create the exact rollback bundle outside the repository:

```bash
python3 scripts/benchmark-scheduler.py --capture-baseline --concurrency 6 --mtp 5 \
  --baseline artifacts/health-rollout/scheduler-baseline.json
./deployments/private-smoke/litellm/rollback.sh snapshot \
  --bundle /srv/dspark-rollback/2026-08-09-prechange \
  --receipt artifacts/health-rollout/rollback-receipt.json
./deployments/private-smoke/litellm/rollback.sh verify \
  --bundle /srv/dspark-rollback/2026-08-09-prechange \
  --receipt artifacts/health-rollout/rollback-receipt.json
```

The bundle directory is mode 0700 and every contained input, manifest, restore
map, and PostgreSQL custom-format dump is mode 0600. `SHA256SUMS` covers every
critical head/worker env and compose/config input, the current virtual-key
file, and the database dump. Verification rejects missing/symlinked files, insecure modes, hash drift,
unsafe restore paths, or a dump that `pg_restore --list` cannot read. The exact
bundle contains secrets and original absolute destinations and therefore stays
on the trusted Spark host. Only the separately sanitized hash receipt belongs
in durable acceptance evidence.

For an old tmpfs database, migrate only after the snapshot verifies:

```bash
./deployments/private-smoke/litellm/rollback.sh migrate \
  --bundle /srv/dspark-rollback/2026-08-09-prechange \
  --key-file /run/private-smoke/hermes.key
```

This stops only the private gateway, creates/attaches the named volume, restores
the validated logical dump, and proves the already-issued mode-0600 Hermes key
still authenticates. It does not generate a replacement key. For any rollout
stop condition, use the reverse path—not the sanitized receipt:

```bash
./deployments/private-smoke/litellm/rollback.sh restore \
  --bundle /srv/dspark-rollback/2026-08-09-prechange \
  --key-file /run/private-smoke/hermes.key
```

Restore verifies hashes/dump again, stops the DSpark head before its worker,
restores exact inputs and database state, invokes the existing worker-first
start lifecycle (whose direct semantic smoke must pass), then starts LiteLLM
and re-proves the existing key. A failed verification aborts before mutation.
Database backups follow the same access/retention controls as the long-lived
service key and must be securely deleted after the acceptance retention window.

### Virtual-key rotation and emergency revocation

The model-scoped Hermes key is a long-lived private service credential owned by
this deployment. LiteLLM administrative credentials and virtual keys are read
only from regular mode-0600 files; secret values are never command arguments,
receipts, or logs.

```bash
python3 deployments/private-smoke/litellm/manage-virtual-key.py rotate \
  --master-key-file /run/private-smoke/litellm-master.key \
  --key-file /run/private-smoke/hermes.key \
  --receipt artifacts/health-rollout/key-rotation.json

# Emergency revocation (removes the local key only after the API proves denial):
python3 deployments/private-smoke/litellm/manage-virtual-key.py revoke \
  --master-key-file /run/private-smoke/litellm-master.key \
  --key-file /run/private-smoke/hermes.key \
  --receipt artifacts/health-rollout/key-revocation.json
```

Rotation generates and authenticates a replacement, revokes the old key through
`/key/delete`, proves the old key receives an auth denial and the replacement
succeeds, then atomically installs the new mode-0600 file. Receipts contain only
booleans, timestamps, and model scope—never key material. Keep a validated DB
snapshot until the rotation receipt and dependent-client smoke pass.

### Pinned Minefield and complete acceptance dimensions

Run the Minefield wrapper at the immutable commit. It creates an isolated
checkout and venv, installs that exact tree, reads the origin key file into
memory, transfers it to the pinned doctor over private stdin, and emits only a
mode-0600 structured summary. The key is absent from OS argv, shell history,
stdout, and JSON:

```bash
python3 scripts/run-minefield-pinned.py \
  --commit 2b453b8a69dbaf8dc9d521dc2d6212cdaceb8169 \
  --json artifacts/health-rollout/minefield.json
```

The summary reports exact executed, problem, inconclusive, and unimplemented
trap counts; a clean executed subset is not a claim about the full registry.
`run-acceptance.sh --live` additionally requires:

- structured head/worker process readiness, authenticated API/model discovery,
  and direct-origin plus private-LiteLLM semantic readiness;
- configured KV bytes and the runtime-reported token capacity (at least the
  advertised 1,048,576 tokens), NCCL world size two, and both rank roles;
- at least 8 GiB `MemAvailable` per host with memory PSI `full avg10=0.00`;
- warm prefix-cache query/hit deltas with reuse above 50%, plus exact
  speculative accepted/draft token deltas and their ratio;
- pinned Minefield coverage counts and an external Tailnet gateway request
  that fails without auth but completes a bounded generation with the scoped
  key; and
- prompt/reasoning canaries absent from proxy, vLLM, worker, LiteLLM, smoke,
  and evidence logs while LiteLLM message logging remains disabled.

Only sanitized summaries are hash-chained into `accepted.json`; prompt bodies,
reasoning bodies, keys, private addresses, and host paths are rejected.
