# DeepSeek-V4-Flash DSpark on 2× DGX Spark

Native SM121 deployment for `deepseek-ai/DeepSeek-V4-Flash-DSpark` on two
NVIDIA GB10 systems: vLLM 0.25 DSpark speculative decoding, FlashInfer B12X
MoE, RoCE tensor parallelism, and **NVFP4 MLA KV cache**.

This repository is the r0b0tlab production/reproducibility layer around the
pinned upstream runtime. It adds a fail-closed image contract, audited
entrypoint, explicit throughput and one-million-token profiles, regression
checks, benchmark tooling, and machine-readable release evidence.

## Release contract

| Component | Pinned value |
|---|---|
| Model | `deepseek-ai/DeepSeek-V4-Flash-DSpark` |
| Model revision | `913f0657a874f76844e2e91cbe706dbcaceeb6d7` |
| Platform | 2× NVIDIA GB10, Linux aarch64, SM121, TP=2 |
| Base image | `ghcr.io/anemll/dspark-vllm-gx10@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8` |
| Base source | `Anemll/dspark-vllm-gx10@47503f8e38dadd4dededca798150db2619594fce` |
| vLLM | `0.25.2.dev0+g752a3a504.d20260714` |
| PyTorch / CUDA | `2.11.0+cu130` / CUDA 13.0 |
| FlashInfer | `0.6.15` |
| MoE | native `flashinfer_b12x` |
| Speculator | native DSpark |
| KV cache | **`nvfp4_ds_mla`** |

The same values are enforced by
[`docker/runtime-manifest.production.json`](docker/runtime-manifest.production.json),
[`scripts/audit_runtime.py`](scripts/audit_runtime.py), and the container
entrypoint. Supplying any other `--kv-cache-dtype` is rejected before vLLM runs.

## What changed

- Promoted the native vLLM 0.25 DeepSeek-V4 DSpark runtime to the production lane.
- Pinned the base by immutable digest and upstream source revision.
- Preserved native NVFP4 MLA KV on every profile; there is no FP8/BF16 cache
  substitution in the production commands.
- Selected FlashInfer B12X explicitly; no Marlin or emulation backend is used.
- Added a native-v0.25 production profile with 200K ceiling, 16 request slots,
  and 16K batched-token budget. The admitted 1M configuration remains a
  separate legacy Stage-C capacity lane because v0.25 does not have enough
  measured KV capacity to serve one million tokens on this hardware.
- Added fail-closed static/runtime audit, exact-image metadata, semantic/tool/
  retrieval gates, native-log checks, and benchmark scaffold tests.
- Added a benchmark harness that separates client and server token accounting,
  TTFT, ITL, prompt/prefill rate, decode rate, request success, memory, power,
  and DSpark acceptance evidence.
- Kept the older Stage-A/B/C overlay and historical 384K/1M artifacts for
  reproduction. Both capacity profiles are explicitly lane-scoped to Stage-C;
  neither is the default native-v0.25 production runtime.

## Performance and quality

The release-candidate measurements are stored under
[`results/production-candidate/`](results/production-candidate/) and summarized
in [`RESULTS.md`](RESULTS.md). Published tables distinguish:

- decode from prefill;
- client SSE rate from vLLM server counters;
- static concurrency from staggered/ragged arrivals;
- speculative acceptance from raw throughput;
- production throughput from one-million-token capacity;
- measured facts from historical or upstream results.

A speedup is accepted only when semantic output, tool calls, retrieval,
NVFP4-KV markers, native B12X, request success, concurrency, and long-context
checks remain green.

## Prerequisites

Two GB10 systems with:

- NVIDIA driver compatible with CUDA 13.0;
- Docker + NVIDIA Container Toolkit;
- passwordless SSH from head to worker;
- RoCE/RDMA connectivity and `/dev/infiniband`;
- the exact model snapshot available at the same container path on both nodes
  (a read-only shared mount is also valid);
- enough memory and storage for the model and image.

Do not use model or cache precision substitutes to make an unsupported runtime
appear healthy.

## Build the audited production image

```bash
git clone https://github.com/r0b0tlab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark.git
cd DeepSeek-v4-Flash-DSpark-2x-DGX-Spark
git checkout v0.25.2-production-dspark-k5-r1
./scripts/ci-verify.sh
./scripts/build-production-image.sh

docker run --rm --gpus all dspark-r0b0tlab:production-candidate audit
```

Expected audit terminator:

```text
DSPARK_RUNTIME_AUDIT_PASS
```

The build uses a digest-pinned base. `MAX_JOBS=6`, `NVCC_THREADS=2`, and
`FLASHINFER_NVCC_THREADS=2` are fixed for GB10-safe build behavior.

## Configure the two nodes

```bash
cp .env.dspark.example .env.dspark
```

Set at least:

```dotenv
WORKER_HOST=worker-host-or-roce-ip
MASTER_ADDR=head-roce-ip
HEAD_ETH_IF=head-socket-interface
WORKER_ETH_IF=worker-socket-interface
HEAD_IB_HCA=head-roce-device
WORKER_IB_HCA=worker-roce-device
NCCL_IB_GID_INDEX=gid-index
DSPARK_MODEL_DIR=/absolute/model/path
DSPARK_VLLM_IMAGE=dspark-r0b0tlab:production-candidate
```

Download model revision `913f0657a874f76844e2e91cbe706dbcaceeb6d7`, then
record the immutable revision in the snapshot before mounting it read-only:

```bash
printf '%s\n' 913f0657a874f76844e2e91cbe706dbcaceeb6d7 \
  > "$DSPARK_MODEL_DIR/.r0b0tlab-model-revision"
python3 scripts/verify_model_checkpoint.py "$DSPARK_MODEL_DIR"
```

The launcher repeats the structural 48-shard/model-revision check on both
nodes, requires identical image IDs, and for the native lane verifies the
image's source/model/KV/speculative labels before it removes or starts any
container. It then starts the worker first and the head second; the image
entrypoint runs the runtime audit before vLLM.

## Launch profiles

Production throughput lane:

```bash
set -a
source .env.dspark
source profiles/dspark-r0b0tlab-production.env
set +a
DSPARK_PREFLIGHT_ONLY=1 ./run-dspark-dual-gb10.sh
./run-dspark-dual-gb10.sh
```

One-million-token compatibility/capacity lane (legacy Stage-C runtime):

```bash
set -a
source .env.dspark
source profiles/dspark-r0b0tlab-1m.env
set +a
DSPARK_PREFLIGHT_ONLY=1 ./run-dspark-dual-gb10.sh
./run-dspark-dual-gb10.sh
```

All profiles keep `KV_CACHE_DTYPE=nvfp4_ds_mla`. The v0.25 production
entrypoint rejects a missing, conflicting, or duplicate cache-dtype argument.
The native launcher accepts the qualified 200K/16/16K/0.84/K5 profile by
default; non-release tuning requires explicit `ALLOW_EXPERIMENTAL_PROFILE=1`.
The two-node
launcher scopes Stage-C-only environment variables to
`DSPARK_RUNTIME_LANE=legacy-stage-c`; they are never exported into the native
v0.25 lane. `docker-compose.dspark.yml` represents the native production lane
only. Do not mix 1M Stage-C results into the v0.25 production performance table.

## Verify the live server

```bash
python3 scripts/runtime_gate.py \
  --base-url http://127.0.0.1:8888 \
  --worker-host "$WORKER_HOST" \
  --output results/production-candidate/runtime-gate.json
```

This gate checks:

1. exact served model identity;
2. deterministic semantic output;
3. forced tool-call parsing;
4. long-prompt retrieval;
5. DSpark, `nvfp4_ds_mla`, B12X, and NCCL IB markers from both ranks;
6. absence of active Marlin, emulation, or fallback markers.

## Benchmark decode and concurrency

```bash
python3 scripts/benchmark_dspark.py \
  --base-url http://127.0.0.1:8888 \
  --model deepseek-v4-flash-dspark \
  --container-name dspark_vllm \
  --concurrency 1 2 4 6 8 12 16 \
  --repeats 3 \
  --max-tokens 512 \
  --output results/production-candidate/decode.json
```

The harness requests streaming usage, rejects missing/inconsistent token counts,
captures TTFT/ITL, scrapes vLLM counters, and samples GPU/host telemetry. For an
uncached prefill-focused run, use deterministic synthetic input and a unique
prefix for every repeat:

```bash
python3 scripts/benchmark_dspark.py \
  --base-url http://127.0.0.1:8888 \
  --model deepseek-v4-flash-dspark \
  --container-name dspark_vllm \
  --concurrency 1 \
  --repeats 3 \
  --max-tokens 1 \
  --synthetic-prompt-words 16000 \
  --unique-prefix-per-repeat \
  --output results/production-candidate/prefill.json
```

Long-context retrieval has a separate uncached gate:

```bash
python3 scripts/long_context_gate.py \
  --base-url http://127.0.0.1:8888 \
  --words 90000 \
  --minimum-prompt-tokens 80000 \
  --nonce "$(date -u +%Y%m%dT%H%M%SZ)" \
  --output results/production-candidate/long-context.json
```

For independent-arrival/ragged behavior, also run:

```bash
python3 benchmarks/staggered_bench.py
```

## Repository map

| Path | Purpose |
|---|---|
| `recipe/Dockerfile.production` | digest-pinned audited production wrapper |
| `docker/runtime-manifest.production.json` | machine-readable runtime contract |
| `scripts/audit_runtime.py` | SM121, version, DSpark-source, and cache audit |
| `scripts/entrypoint.sh` | fail-closed audit and NVFP4-KV enforcement |
| `run-dspark-dual-gb10.sh` | worker-first two-node launch |
| `profiles/` | production, 1M, and historical profiles |
| `scripts/benchmark_dspark.py` | evidence-focused benchmark harness |
| `scripts/long_context_gate.py` | uncached deterministic long-context retrieval gate |
| `scripts/runtime_gate.py` | semantic/tool/retrieval/native live gate |
| `tests/` | release, launch, and benchmark scaffold contracts |
| `results/` | machine-readable accepted candidate evidence |
| `recipe/nvfp4/` | retained legacy Stage-A/B/C packaging |
| `recipe/overlay/` | retained historical vLLM overlay |

## Canonical verification

```bash
./scripts/ci-verify.sh
python3 scripts/verify_release.py --image dspark-r0b0tlab:production-candidate
python3 scripts/public_safety_scan.py .
git diff --check
```

Synthetic tests validate the harness and contracts; they are not hardware
performance evidence. Public claims require live dual-GB10 artifacts from the
exact release candidate.

## Privacy and redistribution

No model weights are included. Do not commit credentials, private prompts,
hostnames, LAN addresses, cache paths, or raw private logs. See
[`PRIVACY.md`](PRIVACY.md). The runtime includes no r0b0tlab telemetry.

## Credits and license

See [`CREDITS.md`](CREDITS.md) for DeepSeek, vLLM, Anemll, FlashInfer, NVIDIA,
Fraser Price, Rafael Caricio, Keys/drowzeys, and prior integration credits.
Repository scripts/docs are MIT licensed; upstream-derived vLLM code retains
its Apache-2.0 lineage. Model weights, images, CUDA, NCCL, FlashInfer, and other
upstream artifacts keep their own terms.
