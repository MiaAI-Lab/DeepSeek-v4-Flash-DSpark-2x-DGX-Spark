# DeepSeek V4 Flash DSpark NVFP4 KV on 2x DGX Spark

Self-contained two-node DGX Spark recipe for serving `DeepSeek-V4-Flash-DSpark`
with vLLM TP=2, DSpark speculative decoding, and the experimental
`nvfp4_ds_mla` KV-cache path.

This repo includes Keys' DSpark concurrency patch in the vLLM overlay. That
patch makes DSpark's persistent draft KV follow request identity instead of
condensed batch-row position, and adds ragged mixed prefill/decode handling for
real independent sessions.

<p>
<a href="https://x.com/MiaAI_lab" target="_blank">
  <img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" />
</a>
</p>
<p>
<a href='https://ko-fi.com/Z8Z3SPLOD' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi6.png?v=6' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>
</p>

The current local run profile is configured for:

- `max_model_len=500000`
- `max_num_seqs=12`
- `kv_cache_dtype=nvfp4_ds_mla`
- `gpu_memory_utilization=0.84`
- API bind address `0.0.0.0:8888`

This repository also includes the original validated 2026-06-29 1M checkpoint
evidence in [`benchmarks/20260629-dspark-nvfp4-1m-context-checkpoint.md`](benchmarks/20260629-dspark-nvfp4-1m-context-checkpoint.md).
It also includes the Keys concurrency checkpoint in
[`benchmarks/20260629-dspark-keys-concurrency-checkpoint.md`](benchmarks/20260629-dspark-keys-concurrency-checkpoint.md).

## Current Profile

The active local `.env.dspark` profile is a 500k-context, 12-sequence
configuration for the same Stage C NVFP4 runtime:

```env
MAX_MODEL_LEN=500000
MAX_NUM_SEQS=12
DSPARK_VLLM_IMAGE=vllm-dspark-runtime:dspark-nvfp4-stage-c
VLLM_USE_B12X_WO_PROJECTION=1
VLLM_HOST=0.0.0.0
```

The rendered vLLM command should include:

```text
--kv-cache-dtype nvfp4_ds_mla
--max-model-len 500000
--max-num-seqs 12
--master-port 25000
```

The 12-sequence profile should be treated as an agent-serving target. It keeps a
500k per-request ceiling while allowing twelve active sequences to share the KV
pool. If it is unstable under your real workload, use the 1M/2 conservative
profile below or reduce `MAX_NUM_SEQS` to `4`.

> **Important:** The current profile is meant for real deep-context operation:
> up to **500k tokens per separate session** with `MAX_NUM_SEQS=12`. The KV cache
> is a shared pool, so twelve sessions do not each reserve 500k tokens up front.
> Normal agent sessions can run concurrently while retaining the 500k ceiling for
> unusually long requests.

> **Reported 1M/6 test:** A live 6-way run on the same recipe line reported
> `6/6` streams succeeded, about `181.7 tok/s` aggregate, no OOM, and no
> request failures. Reproduce on your own nodes before treating those numbers as
> a formal benchmark for your deployment.

The validated NVFP4 run reported about **2.04M tokens of GPU KV cache**. If you
prefer a conservative 1M-context profile, set:

```env
MAX_MODEL_LEN=1048576
MAX_NUM_SEQS=2
```

For a balanced deep-context fallback, set:

```env
MAX_MODEL_LEN=500000
MAX_NUM_SEQS=4
```

That trades the current 500k/12 profile for lower per-request context and more KV
headroom per active session.

## Keys Concurrency Patch

The runtime overlay includes Keys' DSpark concurrency patch, vendored as
[`patches/keys-concurrency.patch`](patches/keys-concurrency.patch).

The patch fixes the DSpark behavior that matters for `MAX_NUM_SEQS > 1`:

- request-stable DSpark main-KV slots
- ragged `query_start_loc` handling for mixed prefill/decode scheduler steps
- passing request ids into the DSpark proposer so persistent draft KV follows
  request identity

The included measured checkpoint for that patch used:

```env
MAX_MODEL_LEN=200000
MAX_NUM_SEQS=16
VLLM_USE_B12X_WO_PROJECTION=1
```

Measured aggregate decode reached `315.1 tok/s` for static C16 and `205.0 tok/s`
for staggered C16. Those numbers are benchmark evidence for the 200k/16 Keys
profile, not for the current 500k/8 local profile.

## High-Concurrency Mode

For many shorter independent sessions, users can switch from the deep-context
profile to the Keys high-concurrency profile. Edit `.env.dspark`:

```env
MAX_MODEL_LEN=200000
MAX_NUM_SEQS=16
VLLM_USE_B12X_WO_PROJECTION=1
```

This enables up to **16 active sequences** while using a lower per-request
context ceiling. It is the profile used by the included Keys concurrency
benchmark.

What changes:

- `MAX_MODEL_LEN=200000` lowers the per-session context ceiling so more active
  sessions can share the KV pool safely.
- `MAX_NUM_SEQS=16` raises the scheduler cap to 16 concurrent active sequences.
- `VLLM_USE_B12X_WO_PROJECTION=1` enables the B12X optimized output-projection
  path used by the measured high-concurrency run.

Use this mode when aggregate concurrency matters more than 500k/1M context per
individual session. For deep-context agent work, keep the default 500k/12 profile
or use the 500k/4 fallback if your workload pushes the KV pool too hard.

After starting the server, you can run the included concurrency probes:

```bash
python3 benchmarks/keys-concurrency/bench_concurrent.py http://127.0.0.1:8888 1,4,8,16
python3 benchmarks/keys-concurrency/staggered_bench.py http://127.0.0.1:8888 16 0.4
python3 benchmarks/keys-concurrency/correctness_test.py http://127.0.0.1:8888
```

If `VLLM_USE_B12X_WO_PROJECTION=1` is unstable on your runtime, set it back to
`0` and retest. That is slower in some concurrency cases but usually safer for
long-context NVFP4 operation.

Changing `VLLM_USE_B12X_WO_PROJECTION` changes the runtime path. After changing
it in `.env.dspark`, rebuild the runtime image before starting:

```bash
./build-dspark-vllm-runtime.sh
./start-deepseek-v4-flash-dspark.sh
```

You do not need to re-download the model unless the Hugging Face cache is
missing. On a fresh machine, run `./prepare-dspark-model-cache.sh` before
starting.

## Original 1M Checkpoint

The original checkpoint was validated on 2x DGX Spark, one GPU per node, TP=2,
single stream:

- `max_model_len=1048576`
- `kv_cache_dtype=nvfp4_ds_mla`
- reported KV pool: `2,044,166 tokens`
- reported max concurrency for 1M requests: `1.95x`
- single-stream decode stayed above `50 tok/s`

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

## Important Caveat

This is the **Stage C padded NVFP4** path. It keeps DeepSeek V4's known-good
584-byte sparse-MLA cache envelope while routing the runtime through
`nvfp4_ds_mla`.

It is **not** the unresolved true-layout 416-byte NVFP4 kernel fix. The
true-layout experiments were useful for diagnosis but failed past roughly 411
real prompt tokens, so they are intentionally not presented here as the
reproducible recipe.

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
| `.env.dspark` | local cluster configuration, ignored by git |
| `build-dspark-vllm-runtime.sh` | builds the Stage C image locally and on the worker |
| `prepare-dspark-model-cache.sh` | downloads/verifies the model cache |
| `start-deepseek-v4-flash-dspark.sh` | preflight checks, worker-first launch, and smoke test |
| `stop-deepseek-v4-flash-dspark.sh` | stops/removes head and worker DSpark services |
| `validate-dspark-config.sh` | prints the active env profile and rendered vLLM command |
| `status-deepseek-v4-flash-dspark.sh` | shows head/worker Compose state, containers, images, port, and API status |
| `logs-deepseek-v4-flash-dspark.sh` | prints head and worker DSpark logs |
| `smoke-deepseek-v4-flash-dspark.sh` | runs a configurable concurrent API smoke test |
| `PLANS.md` | script-hardening scope and validation notes |
| `patches/keys-concurrency.patch` | vendored Keys DSpark concurrency patch |
| `benchmarks/keys-concurrency/` | Keys concurrency benchmark scripts |
| `benchmarks/` | measured 1M and Keys concurrency checkpoint evidence |
| `CREDITS.md` | attribution and license notes for upstream work |

## Quick Start

Run from the head node.

```bash
cp .env.dspark.example .env.dspark
```

Edit `.env.dspark` for your cluster. For this local setup the key values are:

```env
WORKER_HOST=spark2-cx7
MASTER_ADDR=169.254.109.196
MASTER_PORT=25000
NCCL_IB_HCA=rocep1s0f1
NCCL_SOCKET_IFNAME=enp1s0f1np1
NCCL_IB_GID_INDEX=0
HF_CACHE=/home/zurih/.cache/huggingface
MAX_MODEL_LEN=500000
MAX_NUM_SEQS=12
```

For high-concurrency serving, use the `200000 / 16` profile described in
[High-Concurrency Mode](#high-concurrency-mode).

For a balanced fallback, use:

```env
MAX_MODEL_LEN=500000
MAX_NUM_SEQS=4
```

Build the base overlay and Stage C NVFP4 image:

```bash
./build-dspark-vllm-runtime.sh
```

Check the active rendered configuration before launch:

```bash
./validate-dspark-config.sh
```

Prepare the model cache:

```bash
./prepare-dspark-model-cache.sh
```

Start the service:

```bash
./start-deepseek-v4-flash-dspark.sh
```

Stop the service:

```bash
./stop-deepseek-v4-flash-dspark.sh
```

Inspect status or logs:

```bash
./status-deepseek-v4-flash-dspark.sh
./logs-deepseek-v4-flash-dspark.sh
```

Run a short six-way API smoke test after the server is up:

```bash
./smoke-deepseek-v4-flash-dspark.sh
```

Override smoke-test size when needed:

```bash
CONCURRENCY=3 MAX_TOKENS=16 ./smoke-deepseek-v4-flash-dspark.sh
```

The API serves at:

```text
http://127.0.0.1:8888/v1
```

By default the service binds to `0.0.0.0`. Set `VLLM_HOST=127.0.0.1` only if
you intentionally want to keep the API loopback-only on the head node.

## Script Behavior

The helper scripts are intentionally defensive:

- `start-deepseek-v4-flash-dspark.sh` checks required files, Docker, SSH,
  local and worker image presence, existing DSpark containers, and port `8888`
  before starting.
- `start-deepseek-v4-flash-dspark.sh` uses the explicit Compose project
  `deepseek-v4-flash`, so folder names do not change container names.
- `start-deepseek-v4-flash-dspark.sh` passes `.env.dspark` values explicitly to
  Compose, avoiding inherited shell variables such as `MASTER_PORT=29500`.
- `prepare-dspark-model-cache.sh` checks the Stage C image before downloading
  and verifies the worker image before remote cache preparation.
- `stop-deepseek-v4-flash-dspark.sh` uses the same Compose project and falls
  back to removing Compose-labeled `vllm-dspark` containers on both nodes.
- `validate-dspark-config.sh`, `status-deepseek-v4-flash-dspark.sh`, and
  `logs-deepseek-v4-flash-dspark.sh` are read-only helpers.
- `smoke-deepseek-v4-flash-dspark.sh` only sends OpenAI-compatible test
  requests to the running API; it does not modify runtime configuration.

## Runtime Profile

Core vLLM flags for the current local profile:

- `--tensor-parallel-size 2`
- `--distributed-executor-backend mp`
- `--nnodes 2`
- `--kv-cache-dtype nvfp4_ds_mla`
- `--block-size 256`
- `--max-model-len 500000`
- `--max-num-seqs 12`
- `--max-num-batched-tokens 8192`
- `--gpu-memory-utilization 0.84`
- `--speculative-config '{"method":"dspark","num_speculative_tokens":5}'`

Key runtime env:

- `VLLM_USE_B12X_MOE=1`
- `VLLM_USE_B12X_WO_PROJECTION=1`
- `VLLM_DSPARK_CONFIDENCE_SCHEDULER=off`
- `VLLM_DSPARK_LOCAL_ARGMAX=1`
- `VLLM_DSPARK_REPLICATE_MARKOV_W1=1`
- `VLLM_DSPARK_FUSED_MARKOV_ARGMAX=0`
- `VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT=0`
- `VLLM_DSV4_B12X_COMPRESSED_MLA=0`
- `VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE=0`
- `B12X_W4A16_TC_DECODE=0`

## Verify

Render the Compose config without starting the service:

```bash
env -u MASTER_PORT -u NODE_RANK -u HEADLESS -u WORKER_HOST -u MASTER_ADDR \
  COMPOSE_DISABLE_ENV_FILE=1 \
  docker compose --env-file .env.dspark -f docker-compose.dspark.yml config \
  | grep -E -- '--max-model-len|--max-num-seqs|--master-port|--kv-cache-dtype|image:'
```

After launch:

```bash
curl -fsS http://127.0.0.1:8888/v1/models
```

Confirm the returned model entry reports:

```json
"max_model_len": 500000
```

Check logs:

```bash
docker compose -p deepseek-v4-flash --env-file .env.dspark -f docker-compose.dspark.yml logs vllm-dspark \
  | grep -E "GPU KV cache size|Maximum concurrency"
```

## Credits

See [`CREDITS.md`](CREDITS.md) for full attribution.

In short, this recipe combines Rafael Caricio's DSpark vLLM integration, Fraser
Price's DeepSeek V4 Flash DSpark work, MiaAI-Lab's two-node DGX Spark packaging,
Keys/drowzeys' DSpark concurrency patch, and upstream vLLM/FlashInfer/NVIDIA/
DeepSeek components.

This repo's contribution is the NVFP4-KV Stage A/B/C recipe, the two-node DGX
Spark packaging, the applied Keys concurrency overlay, hardened helper scripts,
and benchmark artifacts from the validated runs.

## License Notes

Repo scripts and docs are published under this repo's `LICENSE`. The vLLM
overlay/runtime files are vLLM-derived and retain their Apache-2.0 lineage and
SPDX headers where present. Base images, FlashInfer/TileLang/Triton/CUDA/NCCL,
and model weights are separate upstream artifacts with their own licenses and
usage terms.
