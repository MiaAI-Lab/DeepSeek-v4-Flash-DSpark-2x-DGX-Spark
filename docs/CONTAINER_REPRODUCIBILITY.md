# Container reproducibility

This document records the DSpark-r0b0tlab container build and launch path used for the reproducible DGX Spark / GB10 runtime.

## Final image

```text
vllm-dspark-runtime:dspark-nvfp4-stage-c
```

The image is built locally on the head node and, by default, rebuilt on the worker via SSH/rsync by:

```bash
./build-dspark-vllm-runtime.sh
```

## Build stages

The build script runs these stages in order:

| Stage | Dockerfile | Output tag |
|---|---|---|
| DSpark overlay | `recipe/Dockerfile.dspark-runtime-overlay` | `vllm-dspark-runtime:dspark-r0b0tlab-overlay` |
| NVFP4 A | `recipe/nvfp4/Dockerfile.stage-a` | `vllm-dspark-runtime:dspark-r0b0tlab-overlay-nvfp4-a` |
| NVFP4 B | `recipe/nvfp4/Dockerfile.stage-b` | `vllm-dspark-runtime:dspark-r0b0tlab-overlay-nvfp4-b` |
| NVFP4 C | `recipe/nvfp4/Dockerfile.stage-c` | `vllm-dspark-runtime:dspark-nvfp4-stage-c` |

Stage C is the publication/runtime target. It uses the validated padded NVFP4 sparse-MLA envelope and the `nvfp4_ds_mla` KV-cache path.

## Required source inputs

The build context is the repository root plus the vLLM overlay under:

```text
recipe/overlay/
recipe/nvfp4/
patches/keys-concurrency.patch
```

Before Docker build, the script runs:

```bash
scripts/verify-overlay-sources.sh
```

This catches missing overlay files before producing an image.

## Reproducible 1M launch profile

Use:

```text
profiles/dspark-r0b0tlab-1m.env
```

Core values:

```env
MAX_MODEL_LEN=1048576
MAX_NUM_SEQS=2
MAX_NUM_BATCHED_TOKENS=8192
GPU_MEMORY_UTILIZATION=0.88
MTP_NUM_TOKENS=5
DSPARK_VLLM_IMAGE=vllm-dspark-runtime:dspark-nvfp4-stage-c
```

Expected vLLM flags after rendering:

```text
--kv-cache-dtype nvfp4_ds_mla
--max-model-len 1048576
--max-num-seqs 2
--max-num-batched-tokens 8192
--gpu-memory-utilization 0.88
--speculative-config {"method":"dspark","num_speculative_tokens":5}
```

## Host prep for maximum context

Before 1M runs, stop desktop display managers on the involved GB10 hosts:

```bash
sudo systemctl stop gdm3 || true
sudo systemctl disable gdm3 || true
systemctl is-active gdm3 || true
```

The validated 1M run had `gdm3 inactive` on participating hosts before launch.

## Post-launch checks

```bash
curl -fsS http://127.0.0.1:8888/v1/models
```

Expected model metadata:

```json
{
  "id": "deepseek-v4-flash-dspark",
  "max_model_len": 1048576
}
```

Expected log markers:

```text
DeepSeekV4DSparkModel
kv_cache_dtype=nvfp4_ds_mla
Using probe DeepSeek V4 nvfp4_ds_mla KV cache format
Using 'B12X' Mxfp4 MoE backend
DeepGEMM E8M0 enabled
DSpark GPU rejected-context mask enabled
NCCL INFO Using network IB
```

Expected capacity from the validated 1M/2 run:

```text
GPU KV cache size: 3,421,786 tokens
Maximum concurrency for 1,048,576 tokens/request: 3.26x
```

## Verification commands

Static checks:

```bash
./validate-dspark-config.sh
python3 -m py_compile \
  benchmarks/bench_concurrent.py \
  benchmarks/staggered_bench.py \
  benchmarks/correctness_test.py \
  benchmarks/gsm8k_eval.py \
  benchmarks/needle_acceptance_sweep.py
```

Live strict 1M retrieval check:

```bash
python3 benchmarks/needle_acceptance_sweep.py \
  --tokens 1000000 \
  --depths 0.1 0.5 0.9 \
  --chars-per-token 4.35 \
  --max-tokens 32 \
  --strict-code-only \
  --outdir benchmark-results/one_million_strict_sweep_<timestamp>
```

Validated result: 3/3 PASS at ~903K actual prompt tokens.
