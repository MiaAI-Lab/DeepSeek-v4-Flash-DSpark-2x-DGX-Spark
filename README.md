# DeepSeek V4 Flash DSpark NVFP4 KV on 2x DGX Spark

Self-contained two-node DGX Spark recipe for serving `DeepSeek-V4-Flash-DSpark`
with vLLM TP=2, DSpark speculative decoding, and the experimental
`nvfp4_ds_mla` KV-cache path.

This repo includes Keys' DSpark concurrency patch in the vLLM overlay. That
patch makes DSpark's persistent draft KV follow request identity instead of
condensed batch-row position, and adds ragged mixed prefill/decode handling for
real independent sessions.

The current reproducible run profile is configured for:

- `max_model_len=1048576`
- `max_num_seqs=2`
- `kv_cache_dtype=nvfp4_ds_mla`
- `gpu_memory_utilization=0.88`
- API bind address `0.0.0.0:8888`

This repository includes validated DSpark-r0b0tlab 1M evidence, plus Keys' Patch 2B concurrency and quality
certification notes in [`RESULTS.md`](RESULTS.md) and [`docs/PATCHES.md`](docs/PATCHES.md).

## Current Profile

The active reproducible `.env.dspark` profile is the validated 1M-context,
2-sequence DSpark-r0b0tlab configuration for the same Stage C NVFP4 runtime:

```env
MAX_MODEL_LEN=1048576
MAX_NUM_SEQS=2
DSPARK_VLLM_IMAGE=vllm-dspark-runtime:dspark-nvfp4-stage-c
VLLM_USE_B12X_WO_PROJECTION=1
VLLM_HOST=0.0.0.0
```

The rendered vLLM command should include:

```text
--kv-cache-dtype nvfp4_ds_mla
--max-model-len 1048576
--max-num-seqs 2
--master-port 25000
```

The 2-sequence profile is the conservative 1M profile that produced the
validated ~903K actual-prompt strict retrieval evidence. Use it for reproducing
the published 1M result before experimenting with higher concurrency.

> **Important:** The current profile is meant for real deep-context operation:
> up to **1M tokens per separate session** with `MAX_NUM_SEQS=2`. The KV cache
> is a shared pool, so active sessions do not reserve a full 1M-token KV block
> up front. Reproduce this profile before increasing concurrency.

> **Validated 1M/2 test:** a strict 1M retrieval sweep passed 10/50/90% depths
> at ~903K actual prompt tokens. See [`docs/DSPARK_R0B0TLAB_1M.md`](docs/DSPARK_R0B0TLAB_1M.md).

The validated 1M/2 NVFP4 run reported **3,421,786 tokens of GPU KV cache** and
3.26x maximum concurrency for 1,048,576 tokens/request. The checked-in
conservative profile is:

```env
MAX_MODEL_LEN=1048576
MAX_NUM_SEQS=2
```

For a balanced deep-context fallback, set:

```env
MAX_MODEL_LEN=500000
MAX_NUM_SEQS=4
```

That trades the 1M profile for lower per-request context and more KV headroom
per active session.

## Keys Concurrency Patch

The runtime overlay includes Keys' DSpark concurrency patch, vendored as
[`patches/keys-concurrency.patch`](patches/keys-concurrency.patch).

The patch fixes the DSpark behavior that matters for `MAX_NUM_SEQS > 1`:

- request-stable DSpark main-KV slots
- ragged `query_start_loc` handling for mixed prefill/decode scheduler steps
- Patch 2B ragged detection independent of whether the step has rejected tokens
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
profile, not for the current 1M/2 reproducible profile.

## DSpark-r0b0tlab 1M Profile

The primary reproducible 1M profile is documented in
[`docs/DSPARK_R0B0TLAB_1M.md`](docs/DSPARK_R0B0TLAB_1M.md), captured as
[`profiles/dspark-r0b0tlab-1m.env`](profiles/dspark-r0b0tlab-1m.env), and
rendered in the static report
[`publication/DSpark-r0b0tlab-test-results.html`](publication/DSpark-r0b0tlab-test-results.html).

Use this profile after host prep (`gdm3` stopped/disabled on participating
GB10 hosts):

```env
MAX_MODEL_LEN=1048576
MAX_NUM_SEQS=2
MAX_NUM_BATCHED_TOKENS=8192
GPU_MEMORY_UTILIZATION=0.88
MTP_NUM_TOKENS=5
DSPARK_VLLM_IMAGE=vllm-dspark-runtime:dspark-nvfp4-stage-c
```

Evidence summary for that profile:

- `/v1/models` reports `max_model_len=1048576`.
- GPU KV cache size was reported as 3,421,786 tokens.
- Strict-code-only 1M needle sweep passed 10/50/90% depths at ~902.9K actual prompt tokens.
- c1 smoke reached 59.2 aggregate tok/s; staggered c2 passed 2/2 requests with no errors.

## DSpark-r0b0tlab 384K Profile

The current r0b0tlab publication candidate is documented in
[`docs/DSPARK_R0B0TLAB_384K.md`](docs/DSPARK_R0B0TLAB_384K.md) and captured as
[`profiles/dspark-r0b0tlab-384k.env`](profiles/dspark-r0b0tlab-384k.env).

This profile uses:

```env
MAX_MODEL_LEN=384000
MAX_NUM_SEQS=4
GPU_MEMORY_UTILIZATION=0.88
VLLM_DSPARK_CONFIDENCE_SCHEDULER=off
VLLM_DSPARK_FUSED_MARKOV_ARGMAX=0
```

Evidence summary for that profile:

- `/v1/models` reports `max_model_len=384000`.
- Needle target 300K produced ~270.97K actual prompt tokens and passed 10/50/90% depths.
- Static c4 repeated at 140.7–158.5 aggregate tok/s, above the earlier 134.2 tok/s initial DSpark c4 artifact.
- Staggered c4 passed 4/4 requests with no errors.
- GSM8K N=50 smoke: c1 50/50, c4 49/50, 98% prediction agreement.

Do not claim reliable >300K actual-prompt retrieval from the current evidence; the 340K target (~307K actual prompt) hit an early-depth exact-retrieval failure.

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
individual session. For deep-context agent work, keep the default 1M/2 profile
or use the 500k/4 fallback if your workload pushes the KV pool too hard.

After starting the server, you can run the included concurrency probes:

```bash
python3 benchmarks/bench_concurrent.py http://127.0.0.1:8888 1,4,8,16
python3 benchmarks/staggered_bench.py http://127.0.0.1:8888 16 0.4
python3 benchmarks/correctness_test.py http://127.0.0.1:8888
```

Patch 2B also adds an optional GSM8K quality-certification probe:

```bash
GSM8K_DATA=/path/to/gsm8k_test.jsonl python3 benchmarks/gsm8k_eval.py http://127.0.0.1:8888 200 8 /tmp/gsm8k_concurrent.json
```

Run it once with `CONC=1` and once with `CONC=8`, then compare accuracy and
per-question predictions. See [`RESULTS.md`](RESULTS.md) for the upstream
quality-neutral reference result.

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
| `AGENTS.md` | agent/reproducer operating guide and evidence boundaries |
| `patches/keys-concurrency.patch` | vendored Keys DSpark concurrency patch, including Patch 2B |
| `benchmarks/` | Keys concurrency and quality-certification benchmark scripts |
| `docs/` | setup, container reproducibility, Patch 1/2/2B, and DSpark-r0b0tlab profile notes |
| `RESULTS.md` | upstream concurrency, correctness, and GSM8K quality-certification results |
| `CREDITS.md` | attribution and license notes for upstream work |

## Quick Start

Run from the head node.

```bash
cp .env.dspark.example .env.dspark
```

Edit `.env.dspark` for your cluster. For a reproducible 1M setup the key values are:

```env
WORKER_HOST=worker-host-or-roce-ip
MASTER_ADDR=head-roce-ip
MASTER_PORT=25000
NCCL_IB_HCA=rocepXsYfZ
NCCL_SOCKET_IFNAME=enpXsYfZnpN
NCCL_IB_GID_INDEX=0
HF_CACHE=/path/to/huggingface-cache
MAX_MODEL_LEN=1048576
MAX_NUM_SEQS=2
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

Run a short API smoke test after the server is up:

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
- `--max-model-len 1048576`
- `--max-num-seqs 2`
- `--max-num-batched-tokens 8192`
- `--gpu-memory-utilization 0.88`
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

Canonical repository green for DSpark-r0b0tlab is:

```bash
./scripts/ci-verify.sh
```

That gate checks shell syntax, benchmark Python compilation, required reproducibility files, overlay source presence, `.env.dspark.example` render output, profile assertions, publication artifact checksums, sanitization, and generated-file cleanliness.

Render the Compose config manually without starting the service:

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
"max_model_len": 1048576
```

Check logs:

```bash
docker compose -p deepseek-v4-flash --env-file .env.dspark -f docker-compose.dspark.yml logs vllm-dspark \
  | grep -E "GPU KV cache size|Maximum concurrency"
```

## Credits

See [`CREDITS.md`](CREDITS.md) for full attribution.

In short, this recipe combines Rafael Caricio's DSpark vLLM integration, Fraser
Price's DeepSeek V4 Flash DSpark work, Keys/drowzeys' DSpark concurrency patch,
our DSpark-r0b0tlab draft-head/model implementation, and upstream vLLM/
FlashInfer/NVIDIA/DeepSeek components.

This repo's contribution is the NVFP4-KV Stage A/B/C recipe, the two-node DGX
Spark packaging, the applied Keys concurrency overlay, hardened helper scripts from the validated runs.

## License Notes

Repo scripts and docs are published under this repo's `LICENSE`. The vLLM
overlay/runtime files are vLLM-derived and retain their Apache-2.0 lineage and
SPDX headers where present. Base images, FlashInfer/TileLang/Triton/CUDA/NCCL,
and model weights are separate upstream artifacts with their own licenses and
usage terms.
