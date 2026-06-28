# DeepSeek V4 Flash DSpark - Dual DGX Spark

Deploy [DeepSeek-V4-Flash-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark) across **two NVIDIA DGX Spark** nodes with vLLM, tensor parallelism, FP8 KV cache, InfiniBand/RoCE networking, and DSpark speculative decoding.

This repository contains a working two-node launch path for the Rafael Caricio DSpark vLLM integration:

- vLLM PR: <https://github.com/rafaelcaricio/vllm/pull/1>
- Deployment/runbook PR: <https://github.com/rafaelcaricio/spark_vllm_docker/pull/1>

The validated DSpark profile is optimized for **single-stream decode throughput** on TP=2. It uses a 262k context window and `max_num_seqs=1`. The older plain DeepSeek V4 Flash MTP/1M-context compose path is preserved as a fallback.

## Overview

The DSpark path uses:

- **2 x DGX Spark** nodes, one GPU per node
- **Tensor parallelism 2** across both nodes
- **vLLM multiprocess distributed executor** (`--distributed-executor-backend mp`)
- **DSpark speculative decoding** (`method=dspark`, 5 draft tokens)
- **DeepSeek V4 FP8 KV cache** (`--kv-cache-dtype fp8`)
- **B12X MoE kernels** and verifier output projection optimization
- **Prefix caching**, FlashInfer autotune, DeepSeek V4 tool parser, and DeepSeek V4 reasoning parser

Validated runtime endpoint:

```text
http://127.0.0.1:8888/v1
```

Served model name:

```text
deepseek-v4-flash-dspark
```

## Requirements

### Hardware

| Component | Requirement |
|-----------|-------------|
| Nodes | 2 x NVIDIA DGX Spark / GB10 |
| GPUs | 1 GPU per node |
| Interconnect | InfiniBand/RoCE between nodes |
| Storage | About 170 GB per node for `DeepSeek-V4-Flash-DSpark`, plus Docker images and caches |

### Software

- Docker with `docker compose`
- NVIDIA Container Toolkit
- Git
- curl
- Passwordless SSH from the head node to the worker node
- Hugging Face access to `deepseek-ai/DeepSeek-V4-Flash-DSpark`

The scripts assume both nodes use the same repository path.

## Files

| File | Purpose |
|------|---------|
| `.env.dspark.example` | DSpark environment template |
| `.env.dspark` | Local DSpark environment file created from the template and ignored by git |
| `docker-compose.dspark.yml` | DSpark vLLM service |
| `build-dspark-vllm-runtime.sh` | Fetch Rafael's vLLM PR and build `vllm-dspark-runtime:clean` on both nodes |
| `prepare-dspark-model-cache.sh` | Download and verify DSpark model shards on both nodes |
| `start-deepseek-v4-flash-dspark.sh` | Start the DSpark TP=2 server |
| `stop-deepseek-v4-flash-dspark.sh` | Stop the DSpark TP=2 server |
| `docker-compose.yml` | Legacy/plain DeepSeek V4 Flash MTP fallback |
| `start-deepseek-v4-flash.sh` | Legacy/plain MTP fallback start script |
| `stop-deepseek-v4-flash.sh` | Legacy/plain MTP fallback stop script |

## Quick Start

Run these commands from the **head node**.

### 1. Configure DSpark Environment

```bash
cp .env.dspark.example .env.dspark
```

Edit `.env.dspark` for your cluster:

| Variable | Description | Example |
|----------|-------------|---------|
| `WORKER_HOST` | SSH hostname or IP for the worker | `spark2-cx7` |
| `MASTER_ADDR` | Head-node RoCE/IP address used by torch distributed | `169.254.109.196` |
| `MASTER_PORT` | Distributed init port | `25000` |
| `HF_CACHE` | Host Hugging Face cache path | `/home/user/.cache/huggingface` |
| `NCCL_IB_HCA` | RDMA HCA name | `rocep1s0f1` |
| `NCCL_SOCKET_IFNAME` | Socket interface for NCCL/control traffic | `enp1s0f1np1` |
| `NCCL_IB_GID_INDEX` | RoCE GID index | `0` |

Useful discovery commands:

```bash
ibdev2netdev -v
ip addr show
```

### 2. Build the DSpark vLLM Runtime

```bash
./build-dspark-vllm-runtime.sh
```

This script:

1. Clones or updates Rafael's vLLM fork at `~/models/spark/vllm-dspark`.
2. Checks out `codex/dspark-harness-integration`.
3. Pulls `ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready`.
4. Builds the thin overlay image `vllm-dspark-runtime:clean`.
5. Verifies DSpark imports inside the image.
6. Repeats the same build and import verification on `WORKER_HOST`.

The image is a source overlay, not a full vLLM CUDA rebuild.

### 3. Download and Verify Model Weights

```bash
./prepare-dspark-model-cache.sh
```

This downloads `deepseek-ai/DeepSeek-V4-Flash-DSpark` into `HF_CACHE` on both nodes and verifies the safetensor shard set.

Expected successful verification:

```text
safetensor_shards=48
missing_shards=0
```

The script sets `HF_HUB_DISABLE_XET=1` by default because the Xet transfer path was observed to stall during large shard downloads on this setup.

### 4. Start DSpark

```bash
./start-deepseek-v4-flash-dspark.sh
```

The script:

1. Syncs `docker-compose.dspark.yml` and `.env.dspark` to the worker.
2. Starts the worker with `NODE_RANK=1` and `HEADLESS=1`.
3. Starts the head with `NODE_RANK=0`.
4. Waits for `http://127.0.0.1:8888/v1/models`.
5. Runs a minimal OpenAI-compatible chat request.

### 5. Stop DSpark

```bash
./stop-deepseek-v4-flash-dspark.sh
```

## API Usage

### List Models

```bash
curl http://127.0.0.1:8888/v1/models
```

Expected model id:

```text
deepseek-v4-flash-dspark
```

### Chat Completion

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash-dspark",
    "messages": [
      {"role": "user", "content": "Reply with OK."}
    ],
    "max_tokens": 8,
    "temperature": 0.0
  }'
```

### Streaming

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash-dspark",
    "messages": [
      {"role": "user", "content": "Write a short note about distributed inference."}
    ],
    "stream": true,
    "max_tokens": 128
  }'
```

## Architecture

```text
┌─────────────────────────┐       InfiniBand/RoCE       ┌─────────────────────────┐
│     spark1 / head       │◄────────────────────────────►│    spark2 / worker      │
│  NODE_RANK=0            │    torch.distributed/NCCL     │  NODE_RANK=1            │
│                         │                               │  HEADLESS=1             │
│  vLLM DSpark service    │                               │  vLLM DSpark service    │
│  TP rank 0              │                               │  TP rank 1              │
│  API port 8888          │                               │  headless worker        │
└─────────────────────────┘                               └─────────────────────────┘
```

## DSpark Runtime Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--tensor-parallel-size` | `2` | Split model across both DGX Spark nodes |
| `--pipeline-parallel-size` | `1` | No pipeline parallelism |
| `--distributed-executor-backend` | `mp` | vLLM multiprocess distributed executor |
| `--nnodes` | `2` | Two-node launch |
| `--kv-cache-dtype` | `fp8` | FP8 KV cache |
| `--max-model-len` | `262144` | Validated DSpark context length |
| `--max-num-seqs` | `1` | Single-stream decode profile |
| `--max-num-batched-tokens` | `8192` | Prefill/batch token cap |
| `--gpu-memory-utilization` | `0.80` | Validated memory budget |
| `--speculative-config` | `{"method":"dspark","num_speculative_tokens":5}` | DSpark 5-token draft block |
| `--served-model-name` | `deepseek-v4-flash-dspark` | OpenAI API model id |

Key DSpark/B12X environment defaults:

| Variable | Value |
|----------|-------|
| `VLLM_USE_B12X_MOE` | `1` |
| `VLLM_USE_B12X_WO_PROJECTION` | `1` |
| `VLLM_DSPARK_CONFIDENCE_SCHEDULER` | `off` |
| `VLLM_DSPARK_LOCAL_ARGMAX` | `1` |
| `VLLM_DSPARK_REPLICATE_MARKOV_W1` | `1` |
| `VLLM_DSPARK_FUSED_MARKOV_ARGMAX` | `0` |
| `VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT` | `0` |
| `VLLM_DSV4_B12X_COMPRESSED_MLA` | `0` |
| `VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE` | `0` |
| `B12X_W4A16_TC_DECODE` | `0` |

## Legacy MTP Fallback

The original plain DeepSeek V4 Flash MTP launch remains available:

```bash
./start-deepseek-v4-flash.sh
./stop-deepseek-v4-flash.sh
```

That path uses `docker-compose.yml`, serves `deepseek-ai/DeepSeek-V4-Flash`, and is separate from the DSpark compose file. Do not run both paths on port `8888` at the same time.

## Troubleshooting

### JSON/API Sanity Check

```bash
curl -fsS http://127.0.0.1:8888/v1/models
```

### Logs

Head:

```bash
COMPOSE_DISABLE_ENV_FILE=1 docker compose --env-file .env.dspark -f docker-compose.dspark.yml logs --tail=120 vllm-dspark
```

Worker:

```bash
ssh "$WORKER_HOST" "cd '$PWD' && env -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK -u HEADLESS COMPOSE_DISABLE_ENV_FILE=1 docker compose --env-file .env.dspark -f docker-compose.dspark.yml logs --tail=120 vllm-dspark"
```

### Model Cache Incomplete

Run:

```bash
./prepare-dspark-model-cache.sh
```

A good cache reports:

```text
safetensor_shards=48
missing_shards=0
```

If Hugging Face downloads stall, keep `HF_HUB_DISABLE_XET=1` in `.env.dspark`.

### Runtime Image Missing

Run:

```bash
./build-dspark-vllm-runtime.sh
```

Successful image verification prints:

```text
dspark overlay ok vllm.v1.spec_decode.dspark vllm.v1.spec_decode.dspark_proposer
```

### NCCL / RoCE

- Confirm IB link state with `ibstat`.
- Confirm HCA and interface names with `ibdev2netdev -v`.
- Confirm passwordless SSH with `ssh "$WORKER_HOST" hostname`.
- Check that both nodes use the same `MASTER_ADDR`, `MASTER_PORT`, `NCCL_IB_HCA`, and `NCCL_SOCKET_IFNAME`.

### Port Conflict

DSpark serves on port `8888`. Stop the legacy MTP path or any other service using that port before starting DSpark.

## Validation Snapshot

This repo's DSpark path was validated on a 2-node DGX Spark cluster with:

- `vllm-dspark-runtime:clean` built on both nodes
- `DeepSeek-V4-Flash-DSpark` cache verified on both nodes: 48 safetensor shards, 0 missing
- TP=2 world initialized over NCCL
- `/v1/models` returning `deepseek-v4-flash-dspark`
- Minimal `/v1/chat/completions` request succeeding

## License

Repository scripts and configuration are provided under the MIT License. Model weights are governed by the corresponding DeepSeek Hugging Face model license. The DSpark runtime image is built from Rafael Caricio's vLLM fork and the referenced base image; review those upstream licenses before redistribution.
