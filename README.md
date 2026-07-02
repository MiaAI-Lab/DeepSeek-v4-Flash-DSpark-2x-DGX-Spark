# DeepSeek V4 Flash DSpark C12 NVFP4 KV on 2x DGX Spark

Self-contained two-node DGX Spark recipe for serving `DeepSeek-V4-Flash-DSpark`
with vLLM TP=2, DSpark speculative decoding, and a 1M-token max model length
using the experimental `nvfp4_ds_mla` KV-cache path.

This repo captures the validated Stage C NVFP4 runtime, the 2026-06-30
agent-stability refresh, and the 2026-07-02 Keys C12 checkpoint:

- `max_model_len=1000000`
- `max_num_seqs=12`
- `kv_cache_dtype=nvfp4_ds_mla`
- reported KV pool: `3,225,280 tokens`
- configured active sequence slots: `12`
- single-stream decode stayed above `50 tok/s`
- deterministic direct prompts completed with no Chinese drift or repeated junk
- 2/4/6/12 concurrent code-gate prompts completed cleanly
- DSpark in-server concurrency patch validated at `max_model_len=200000`,
  `max_num_seqs=16`, with static C16 at `315.1 tok/s` aggregate and
  staggered C16 at `205.0 tok/s` aggregate

If you already deployed an older copy and saw agent garble, loops, Chinese
drift, or prompt/tool XML leaking into replies, start with
[`AGENT_GARBLE_FIX.md`](AGENT_GARBLE_FIX.md). The fix path keeps the C12 NVFP4
profile; it does not switch production to fp8 or a smaller fallback model.

## Result

### 2026-07-02 Keys C12 NVFP4 Checkpoint

The current high-concurrency lane keeps Tony's known-good Stage C NVFP4 image
and applies Keys' C12 serving profile.

Runtime:

- endpoint tested: `http://100.90.25.78:8888/v1`
- served model: `deepseek-v4-flash-dspark`
- image: `vllm-dspark-runtime:dspark-nvfp4-stage-c`
- model path: `/cache/huggingface/fraserprice/DeepSeek-V4-Flash-DSpark`
- `kv_cache_dtype=nvfp4_ds_mla`
- `max_model_len=1000000`
- `max_num_seqs=12`
- `max_num_batched_tokens=8192`
- `gpu_memory_utilization=0.85`
- `MTP_NUM_TOKENS=5`
- `VLLM_USE_B12X_WO_PROJECTION=1`
- `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`
- `thinking=false`
- `--generation-config vllm`
- `--override-generation-config '{"temperature":0.0,"top_p":1.0,"top_k":40,"repetition_penalty":1.05}'`

Boot evidence:

```text
GPU KV cache size: 3,225,280 tokens
Maximum concurrency for 1,000,000 tokens per request: ~3.2x
Application startup complete.
```

Code-gate validation:

| concurrency | success | server generation tok/s | acceptance | bad outputs |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1/1 | 52.79 | 0.585 | 0 |
| 2 | 2/2 | 79.76 | 0.600 | 0 |
| 4 | 4/4 | 134.70 | 0.602 | 0 |
| 6 | 6/6 | 127.78 | 0.615 | 0 |
| 12 | 12/12 | 230.10 | 0.602 | 0 |

The upstream checkpoint note for this run was not imported into this checkout;
this repo keeps the runtime changes and validation summary without the upstream
benchmark artifact folder.

Do not enable `VLLM_USE_B12X_FP8_GEMM=1` on this Stage C image. That flag hit a
DeepGEMM layout assertion during DSpark drafter warmup in testing.

### 2026-06-30 Clean Agent-Serving Checkpoint

The prior conservative clean endpoint was reproduced on Asusi/Spark4 before
sending the model back through Hermes/OpenClaw-style harnesses.

Runtime:

- endpoint tested: `http://100.90.25.78:8888/v1`
- served model: `deepseek-v4-flash-dspark`
- image used on that lane: `vllm-dspark-runtime:mia-raf-pr1-nvfp4-keys-c`
- model path: `/cache/huggingface/fraserprice/DeepSeek-V4-Flash-DSpark`
- `kv_cache_dtype=nvfp4_ds_mla`
- `max_model_len=1048576`
- `max_num_seqs=6`
- `max_num_batched_tokens=8192`
- `gpu_memory_utilization=0.80`
- `MTP_NUM_TOKENS=5`
- `thinking=false`
- `--generation-config vllm`
- `--override-generation-config '{"temperature":0.0,"top_p":1.0}'`
- explicit per-node `VLLM_HOST_IP` values

Boot evidence:

```text
GPU KV cache size: 1,990,142 tokens
Maximum concurrency for 1,048,576 tokens per request: 1.90x
Application startup complete.
```

Direct validation:

- `/v1/models` reported `"max_model_len": 1048576`
- deterministic sanity prompt returned `NVFP4 DSPARK OK`
- five longer English prompts completed with no CJK drift and no repeated junk
- code-gate server decode mean: `54.22 tok/s`
- 2/4/6 concurrent direct prompts all succeeded cleanly

Concurrency:

| concurrency | success | aggregate tok/s | stability |
| ---: | ---: | ---: | --- |
| 2 | 2/2 | 60.95 | no CJK/repeat junk |
| 4 | 4/4 | 83.21 | no CJK/repeat junk |
| 6 | 6/6 | 104.11 | no CJK/repeat junk |

The upstream checkpoint note for this run was not imported into this checkout.

### 1M NVFP4 Profile

Validated on 2x DGX Spark, one GPU per node, TP=2, single stream.

| Case | server tok/s | TTFC | acceptance | accepted/draft |
| --- | ---: | ---: | ---: | ---: |
| p256/g64 | 54.46 | 0.506s | 0.667 | 3.33 |
| p256/g256 | 65.38 | 0.324s | 0.718 | 3.59 |
| p512/g64 | 56.26 | 2.738s | 0.625 | 3.13 |
| p512/g256 | 54.41 | 0.422s | 0.550 | 2.75 |
| p512/g256 warmup1 | 56.73 | 0.417s | 0.585 | 2.92 |

Boot logs reported:

```text
GPU KV cache size: 2,044,166 tokens
Maximum concurrency for 1,048,576 tokens per request: 1.95x
```

The API reported:

```json
{"max_model_len":1048576}
```

The upstream checkpoint note for this run was not imported into this checkout.

### DSpark Concurrency Profile

Validated on the same 2x DGX Spark TP=2 deployment using Keys' DSpark
concurrency patch, `kv_cache_dtype=nvfp4_ds_mla`, `max_model_len=200000`,
`max_num_seqs=16`, `MTP_NUM_TOKENS=5`, and
`VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`.

Patch source:

- [drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash](https://github.com/drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash)
- Tested patch commit: `7e4d94bbcec95223550517c0fa9244e59f9f6483`

The live fix documented here keeps `kv_cache_dtype=nvfp4_ds_mla` and refreshes
the repo's already-vendored Keys overlay with the path-adjusted Patch 2b update
from that commit. In Patch 2b, ragged `query_start_loc` detection no longer
depends on `num_rejected_tokens_gpu`. Treat the service as validated only after
the built-in OpenAI-compatible chat smoke request plus agent-client validation
pass on the live service.

Static simultaneous batch, one TP=2 replica:

| concurrency | best aggregate tok/s | per-stream tok/s | acceptance |
| ---: | ---: | ---: | ---: |
| 1 | 57.6 | 57.6 | 0.635 |
| 4 | 140.8 | 35.2 | 0.619 |
| 8 | 252.6 | 31.6 | 0.635 |
| 16 | 315.1 | 19.7 | 0.609 |

Staggered independent arrivals, one TP=2 replica:

| concurrency | success | aggregate tok/s | acceptance |
| ---: | ---: | ---: | ---: |
| 4 | 4/4 | 109.2 | 0.544 |
| 8 | 8/8 | 147.3 | 0.534 |
| 16 | 16/16 | 205.0 | 0.567 |

Correctness sanity check: deterministic victim output remained byte-identical
under churn. A medium-churn condense test measured `0.529` acceptance and
`99.7 tok/s` across the churn window.

The upstream checkpoint note for this run was not imported into this checkout.

### Historical 60 tok/s DSpark Baseline

The older ~60 tok/s number was reproduced, but it is a separate diagnostic
profile, not this repo's default 1M NVFP4 deployment:

- image rebuilt from `rafaelcaricio/vllm#1` commit `3519c3b88`
- `max_model_len=262144`
- `max_num_seqs=1`
- `kv_cache_dtype=fp8`
- `MTP_NUM_TOKENS=5`
- `thinking=false`
- `temperature=0.0`, `top_p=1.0`
- measured `63.97 tok/s` on the `code_completion` gate with `67.9%`
  DSpark acceptance

Use this to diagnose image/runtime drift. Do not confuse it with the production
1M NVFP4 path. The upstream checkpoint note for this run was not imported into
this checkout.

### 2026-06-29 Full-1M Concurrency Microbench

The 200K/16 profile above maximizes raw concurrency. For agent fleets that want
the **full 1M context ceiling AND concurrency**, run `max_model_len=1048576`
with `max_num_seqs=6`. Every request can still grow to 1M while up to 6 sessions
run at once, because the shared KV pool — not a per-slot reservation — is the
real limit (see [How the KV cache works](#how-the-kv-cache-works-why-1m--concurrency-is-safe)).

Validated on the 2026-06-29 code-completion microbench deployment (NVFP4,
`max_model_len=1048576`, `max_num_seqs=6`,
`VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`, `VLLM_USE_B12X_WO_PROJECTION=1`):

- Boot: `GPU KV cache size: 1,901,239 tokens`, `Maximum concurrency for 1,048,576 tokens per request: 1.81x`
- 6 concurrent requests: **6/6 success**, **~182 tok/s aggregate** (~30 tok/s per stream), no OOM / no preemption failures
- Single-stream decode on this same profile: ~67 tok/s (code)

This is the right shape when most sessions sit far below 1M (typical agent
turns) but you still want the 1M ceiling available. The newer 2026-06-30
agent-stability checkpoint above is the safer number to cite for Hermes/OpenClaw
harness validation.

> Higher concurrency is not free: under sustained pressure you can see added
> scheduler churn, prefill contention, and KV fragmentation. 1M/6 is validated
> for normal-length agent traffic; for guaranteed deep-context work under load,
> 1M/2 is conservative and 500K/4 is a balanced middle.

## How the KV cache works (why 1M + concurrency is safe)

Three independent knobs, often confused:

| knob | what it is | this build |
| --- | --- | --- |
| **KV cache pool** | total shared KV memory in tokens, sized from `gpu_memory_utilization` after weights load | ~1.9–2.04M tokens (NVFP4) |
| `max_model_len` | per-request **ceiling** — how long any one request may grow | 1,048,576 (1M) |
| `max_num_seqs` | **concurrency cap** — max active sequences the scheduler runs at once | 6 |

The pool is **shared and allocated on demand**: PagedAttention hands KV blocks
to each request as it generates tokens and frees them when it finishes.
`max_model_len` and `max_num_seqs` are **ceilings, not reservations** — vLLM does
NOT pre-allocate `max_num_seqs × max_model_len` of KV. So the real constraint is:

```
sum(live tokens across all active requests) <= KV pool (~1.9M)
```

Worked examples at 1M ceiling / 6 slots:

```
6 requests x  50k tokens =  300k   fits easily
6 requests x 200k tokens =  1.2M   fits
6 requests x 317k tokens =  1.9M   ~at the limit
2 requests x 1M   tokens =  2.0M   ~at the limit  (this is the "1.81x" boot number)
6 requests x 1M   tokens =  6.0M   impossible — excess requests queue/preempt
```

The boot log's `Maximum concurrency for 1,048,576 tokens per request: 1.81x`
only means ~1.8 *simultaneous full-1M* requests fit. But agent turns are almost
never near 1M, so 6 normal-length sessions share the pool comfortably while the
1M ceiling stays available for the rare long one. That is exactly why
`1M + max_num_seqs=6` is safe: you are not reserving 6×1M, you are sharing one
~1.9M pool across short requests under a high ceiling.

## Gotcha: gibberish, loops, Chinese drift, or prompt/XML leakage

If the model boots and basic prompts like `hi` work, but real agent traffic
randomly turns into repeated characters, Chinese drift, leaked tool/schema XML,
or Telegram-visible junk, do not assume the weights are bad.

On this deployment there are three checks to make before blaming the weights:

1. **Runtime concurrency safety:** make sure the Keys Patch 2b logic is present
   in `recipe/overlay/vllm/v1/spec_decode/dspark_proposer.py`. The important
   behavior is that ragged `query_start_loc` handling does not depend on
   `num_rejected_tokens_gpu`, and the no-rejection path creates a zero rejected
   token tensor instead of falling through to unsafe request reshaping. Without
   this, concurrent DSpark requests can mix context.
2. **Runtime image provenance:** make sure the image really contains the current
   DSpark overlay. A reused local tag named `vllm-dspark-runtime:clean` caused
   misleading failures even though a nearby PR-head image worked. Rebuild from
   the intended overlay commit when in doubt.
3. **Decode/fallback safety:** for long OpenAI-compatible agent prompts, avoid
   unstable sampling and hidden fallback transitions. The server default should
   ignore the model card's sampling defaults and apply a small sampling floor:

```json
{
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 40,
  "repetition_penalty": 1.05,
  "include_reasoning": false,
  "reasoning_effort": "none",
  "chat_template_kwargs": {
    "thinking": false,
    "enable_thinking": false
  }
}
```

The compose launcher now includes `--generation-config vllm`, builds
`--override-generation-config` from the `GENERATION_*` env values, and sets
`thinking=false` so default requests do not inherit unstable model-card sampling.
Explicit client request parameters still win. For exact deterministic curl
checks, send `temperature: 0` in the request body.

Also clear agent fallback lists during validation. A model that looks fixed in
direct vLLM tests can still appear poisoned if the orchestration layer silently
falls back, reboots a session, or replays a stale prompt/tool transcript into
the visible message stream. Keep OpenClaw/Hermes changes separate from model
runtime validation unless you are deliberately testing that harness.

Validation gates to run after a live fix:

```text
direct vLLM prompts: clean
direct concurrent vLLM prompts: clean
agent harness prompts: clean, DeepSeek, no fallback
MTP5 accepted-token positions 0..4 active
```

This keeps NVFP4 KV and MTP5. Do not switch to fp8 or drop to a smaller fallback
model just to hide the symptom unless you intentionally accept the context and
quality tradeoff.

## Important Caveat

This is the **Stage C padded NVFP4** path. It keeps DeepSeek V4's known-good
584-byte sparse-MLA cache envelope while routing the runtime through
`nvfp4_ds_mla`.

It is **not** the unresolved true-layout 416-byte NVFP4 kernel fix. The
true-layout experiments were useful for diagnosis but failed past roughly 411
real prompt tokens, so they are intentionally not presented here as the
reproducible recipe.

## Credits

See [`CREDITS.md`](CREDITS.md) for the full attribution and license notes.

This recipe stands on prior public work:

- Keys / drowzeys' DSpark in-server concurrency patch:
  [drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash](https://github.com/drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash).
  This patch fixes the request-stable DSpark main-KV slot mapping and the
  ragged `query_start_loc` path needed for real independent-arrival
  continuous batching. The concurrency results in this repo depend directly on
  that work.
- Rafael Caricio's DSpark vLLM integration:
  [rafaelcaricio/vllm#1](https://github.com/rafaelcaricio/vllm/pull/1)
- Rafael Caricio's DSpark deployment/runbook PR:
  [rafaelcaricio/spark_vllm_docker#1](https://github.com/rafaelcaricio/spark_vllm_docker/pull/1)
- Fraser Price's DeepSeek V4 Flash DSpark model/runtime work:
  [fraserprice/DeepSeek-V4-Flash-DSpark](https://huggingface.co/fraserprice/DeepSeek-V4-Flash-DSpark)
  and [fraserprice/dspark-vllm](https://github.com/fraserprice/dspark-vllm)
- MiaAI-Lab's two-node DGX Spark packaging and worker-first launch runbook:
  [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
- **drowzeys ("Keys")** — origin of wiring the `nvfp4_ds_mla` KV-cache dtype into a DGX Spark launch recipe
  ([Keys---Full-GLM-5.2-Quantrio…](https://github.com/drowzeys/Keys---Full-GLM-5.2-Quantrio-INT4-INT8-mixed-8bit-Attention-on-4-x-DGX-Spark-GB10-Cluster)).
  This build's 1M NVFP4 KV path descends from that `nvfp4_ds_mla` work.
- Upstream vLLM, FlashInfer, NVIDIA Blackwell/CUDA/NCCL tooling, and
  DeepSeek V4 Flash.
- DeepSeek-AI's DeepSpec work as the public DSpark/speculative decoding
  foundation.

Our contribution here is the 1M NVFP4-KV checkpoint recipe, the Stage A/B/C
runtime patches, sanitized two-node launch config, applying and validating
Keys' concurrency patch on the NVFP4 profile, and measured benchmark artifacts
from the validated runs.

## License Notes

Repo scripts and docs are published under this repo's `LICENSE`. The vLLM
overlay/runtime files are vLLM-derived and retain their Apache-2.0 lineage and
SPDX headers where present. Base images, FlashInfer/TileLang/Triton/CUDA/NCCL,
and model weights are separate upstream artifacts with their own licenses and
usage terms.

## Files

| path | purpose |
| --- | --- |
| `recipe/overlay/` | base DSpark vLLM overlay files |
| `recipe/Dockerfile.dspark-runtime-overlay` | builds the base DSpark runtime overlay |
| `recipe/nvfp4/Dockerfile.stage-a` | adds `nvfp4_ds_mla` dtype plumbing |
| `recipe/nvfp4/Dockerfile.stage-b` | enables DeepSeek V4 `nvfp4_ds_mla` probe path |
| `recipe/nvfp4/Dockerfile.stage-c` | switches DeepSeek V4 NVFP4 to the validated 584-byte padded envelope |
| `docker-compose.dspark.yml` | two-node vLLM/DSpark service |
| `.env.dspark.example` | sanitized cluster configuration template |
| `build-dspark-vllm-runtime.sh` | builds the Stage C image locally and on the worker |
| `prepare-dspark-model-cache.sh` | downloads/verifies the model cache |
| `start-deepseek-v4-flash-dspark.sh` | worker-first launch and smoke test; honors worker path/cache/IP overrides |
| `stop-deepseek-v4-flash-dspark.sh` | stops head and worker services |
| `status-deepseek-v4-flash-dspark.sh` | shows head/worker container state |
| `logs-deepseek-v4-flash-dspark.sh` | tails head/worker DSpark logs |
| `smoke-deepseek-v4-flash-dspark.sh` | direct concurrent OpenAI-compatible smoke test |
| `validate-dspark-config.sh` | renders and checks the local DSpark compose/env config |
| `patches/keys-concurrency.patch` | full path-adjusted Keys concurrency patch reference |
| `docs/PATCHES.md` | plain-English Patch 1 / Patch 2 / Patch 2b concurrency explanation |
| `UPSTREAM_V024_STATUS.md` | current vLLM v0.24.0 vs DSpark PR #46995 upgrade notes |
| `AGENT_GARBLE_FIX.md` | update path for older deployments that saw agent garble/drift/loops |
| `scripts/agent_sanity_bench.py` | direct OpenAI-compatible 1/2/4/6 concurrency and garble check |
| `scripts/capture_runtime.sh` | captures head/worker Docker inspect, ps, and log tails before/after changes |
| `benchmarks/` | local benchmark scripts retained from this checkout; upstream benchmark artifacts were intentionally not imported |

## Quick Start

Run from the head node.

```bash
cp .env.dspark.example .env.dspark
```

Edit these values for your cluster:

- `WORKER_HOST`
- `WORKER_SCRIPT_DIR` if the worker checkout/deployment path differs from the head
- `MASTER_ADDR`
- `NCCL_IB_HCA`
- `NCCL_SOCKET_IFNAME`
- `NCCL_IB_GID_INDEX`
- `HF_CACHE`
- `WORKER_HF_CACHE` if the worker cache path differs from the head
- `VLLM_HOST_IP` and `WORKER_VLLM_HOST_IP` for each node's fabric IP

For this local setup the key values are:

```env
WORKER_HOST=10.0.0.2
MASTER_ADDR=10.0.0.1
VLLM_HOST_IP=10.0.0.1
WORKER_VLLM_HOST_IP=10.0.0.2
MASTER_PORT=25000
NCCL_IB_HCA=rocep1s0f1
NCCL_SOCKET_IFNAME=enp1s0f1np1
```

Keep these agent-serving defaults unless you are deliberately experimenting:

- `VLLM_HOST=0.0.0.0` if Hermes/OpenClaw or another machine must reach the API
- `MAX_MODEL_LEN=1000000`
- `MAX_NUM_SEQS=12`
- `GPU_MEMORY_UTILIZATION=0.85`
- `MTP_NUM_TOKENS=5`
- `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`
- `VLLM_USE_B12X_WO_PROJECTION=1`
- `GENERATION_TEMPERATURE=0.0`
- `GENERATION_TOP_P=1.0`
- `GENERATION_TOP_K=40`
- `GENERATION_REPETITION_PENALTY=1.05`
- `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`

Build the base overlay and Stage C NVFP4 image:

```bash
./build-dspark-vllm-runtime.sh
```

Prepare the model cache:

```bash
./prepare-dspark-model-cache.sh
```

Start the service:

```bash
./start-deepseek-v4-flash-dspark.sh
```

The API serves at:

```text
http://HEAD_NODE_IP:8888/v1
```

For head-node-only tests, set `VLLM_HOST=127.0.0.1`. For Hermes/OpenClaw or
another machine to use the endpoint, keep `VLLM_HOST=0.0.0.0` and control
access at the network/firewall layer.

## Runtime Profile

### C12 Agent-Serving Profile

Core vLLM flags:

- `--tensor-parallel-size 2`
- `--distributed-executor-backend mp`
- `--nnodes 2`
- `--kv-cache-dtype nvfp4_ds_mla`
- `--block-size 256`
- `--max-model-len 1000000`
- `--max-num-seqs 12`
- `--max-num-batched-tokens 8192`
- `--gpu-memory-utilization 0.85`
- `--speculative-config '{"method":"dspark","num_speculative_tokens":${MTP_NUM_TOKENS:-5}}'`
- `--generation-config vllm`
- `--override-generation-config '{"temperature":0.0,"top_p":1.0,"top_k":40,"repetition_penalty":1.05}'`

Key runtime env:

- `VLLM_USE_B12X_MOE=1`
- `VLLM_USE_B12X_WO_PROJECTION=1`
- `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`
- `VLLM_DSPARK_CONFIDENCE_SCHEDULER=off`
- `VLLM_DSPARK_LOCAL_ARGMAX=1`
- `VLLM_DSPARK_REPLICATE_MARKOV_W1=1`
- `VLLM_DSPARK_FUSED_MARKOV_ARGMAX=0`
- `VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT=0`
- `VLLM_DSV4_B12X_COMPRESSED_MLA=0`
- `VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE=0`
- `B12X_W4A16_TC_DECODE=0`

### 200k Concurrency Profile

For DSpark concurrency, use the included overlay files with Keys'
concurrency patch and set:

- `MAX_MODEL_LEN=200000`
- `MAX_NUM_SEQS=16`
- `VLLM_USE_B12X_WO_PROJECTION=1`
- `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`

This checkout intentionally does not import upstream's `benchmarks/keys-concurrency/`
folder. Use the retained local benchmark scripts if you need ad hoc checks:

```bash
python3 benchmarks/bench_concurrent.py http://127.0.0.1:8888 1,4,8,16
python3 benchmarks/staggered_bench.py http://127.0.0.1:8888 16 0.4
python3 benchmarks/correctness_test.py http://127.0.0.1:8888
```

### 1M Single-Stream Legacy Profile

For conservative single-stream testing, set `MAX_NUM_SEQS=1` and
`VLLM_USE_B12X_WO_PROJECTION=0`. Keep `MTP_NUM_TOKENS=5` unless you are
deliberately running an experiment; upstream Mia and Keys both validate the
DSpark path at MTP5.

## Verify

After launch:

```bash
curl -fsS http://127.0.0.1:8888/v1/models
```

Confirm the returned model entry reports:

```json
"max_model_len": 1000000
```

Then check logs:

```bash
docker compose --env-file .env.dspark -f docker-compose.dspark.yml logs vllm-dspark \
  | grep -E "GPU KV cache size|Maximum concurrency"
```

Expected C12 checkpoint values are around:

```text
GPU KV cache size: 3.2M tokens
Maximum concurrency for 1,000,000 tokens per request: ~3.2x
```

Before pointing an agent harness at the endpoint, run the direct sanity bench:

```bash
DSPARK_BASE_URL=http://HEAD_NODE_IP:8888/v1 \
CONCURRENCY=1,2,4,6 \
python3 scripts/agent_sanity_bench.py
```

Every row should report `bad_outputs: 0`. If this direct test is clean but an
agent still garbles, investigate the agent session, fallback list, or harness
prompt replay before blaming the DSpark weights.

Capture runtime evidence before and after any fix:

```bash
scripts/capture_runtime.sh runtime-before-change
scripts/capture_runtime.sh runtime-after-change
```

## Notes

- The old speed checkpoint is single stream, not aggregate throughput.
- The high-concurrency benchmark is aggregate throughput and was validated at
  `max_model_len=200000`, not full 1M context.
- Full context and high concurrency compete for the same KV pool. The C12
  1M profile is intended for normal agent traffic where most sessions sit far
  below the 1M ceiling; it is not twelve simultaneous full-1M requests.
- To combine DSpark concurrency with longer context, pick a lower context
  target first, then raise concurrency slowly while watching boot logs, KV
  allocation, acceptance, and request errors.
- 1M was validated as booted/advertised `max_model_len` with KV headroom and
  short-prompt speed probes. This repo does not claim a full 1M-token retrieval
  or correctness benchmark.
- The measured probes were p256/p512 with g64/g256. Rebenchmark if you change
  sampling, batching, context length, WO projection, compressed MLA, or the
  confidence scheduler.
- The current configured agent-serving profile is `MAX_MODEL_LEN=1000000`,
  `MAX_NUM_SEQS=12`, `GPU_MEMORY_UTILIZATION=0.85`,
  `MTP_NUM_TOKENS=5`, `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`,
  `VLLM_USE_B12X_WO_PROJECTION=1`, deterministic generation overrides, and
  `VLLM_DSV4_B12X_COMPRESSED_MLA=0`.
- Worker-first startup avoids a race during multi-node `mp` initialization.
- Requires matching images on both nodes, correct NCCL/RoCE settings, and a
  two-node Blackwell-class/DGX Spark setup.
- The API binds to `127.0.0.1` by default; exposing it is a deliberate security
  choice.
- The next max-sequence ladder to try is approximately 1.25M, 1.5M, then
  1.75M, with the same boot/log/speed gates. Raw KV math alone is not enough
  because DeepSeek V4 sparse MLA also allocates max-length-dependent workspaces.
