# DSpark-r0b0tlab 1M profile

This document records the current DSpark-r0b0tlab 1M-context operating profile and the retrieval optimization found on 2026-06-30.

## Runtime profile

Use `profiles/dspark-r0b0tlab-1m.env` after host prep:

```text
MAX_MODEL_LEN=1048576
MAX_NUM_SEQS=2
MAX_NUM_BATCHED_TOKENS=8192
GPU_MEMORY_UTILIZATION=0.88
MTP_NUM_TOKENS=5
VLLM_DSPARK_CONFIDENCE_SCHEDULER=off
VLLM_DSPARK_CONFIDENCE_THRESHOLD=0.0
VLLM_DSPARK_FUSED_MARKOV_ARGMAX=0
VLLM_DSPARK_LOCAL_ARGMAX=1
VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1
VLLM_DSPARK_REPLICATE_MARKOV_W1=1
VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT=0
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
```

## Host prep

Before launching 1M, stop and disable `gdm3` on the involved GB10 hosts and verify no display-manager/GNOME/Xorg processes remain. The validated run had `gdm3 inactive` on the head, worker, and node3 before relaunch.

## Live capacity evidence

The validated 1M launch reported:

```text
max_model_len=1048576
GPU KV cache size: 3,421,786 tokens
Maximum concurrency for 1,048,576 tokens per request: 3.26x
```

Native path markers were present:

```text
DeepSeekV4DSparkModel
nvfp4_ds_mla
B12X Mxfp4 MoE
DeepGEMM E8M0
NCCL INFO Using network IB
DSpark GPU rejected-context mask enabled
```

## Retrieval optimization

The previous long-context needle prompt asked the model to return the code and then explain how it found it. At ~903K actual prompt tokens, depth 50% failed by continuing nearby document records instead of emitting the code, even though DSpark acceptance did not collapse.

The fix is to use the strict-code-only retrieval prompt (`--strict-code-only`):

```bash
python3 benchmarks/needle_acceptance_sweep.py \
  --tokens 1000000 \
  --depths 0.1 0.5 0.9 \
  --chars-per-token 4.35 \
  --max-tokens 32 \
  --strict-code-only \
  --outdir benchmark-results/one_million_strict_sweep_<timestamp>
```

Evidence artifact:

```text
benchmark-results/one_million_strict_sweep_20260630_124058/
```

Result:

| Target | Actual prompt | Depth | Retrieval | Acceptance | Completion tokens | Elapsed |
|---:|---:|---:|---|---:|---:|---:|
| 1,000,000 | 902,922 | 10% | PASS | 0.40 | 14 | 888.290s |
| 1,000,000 | 902,920 | 50% | PASS | 0.36 | 14 | 882.799s |
| 1,000,000 | 902,923 | 90% | PASS | 0.40 | 14 | 882.239s |

This supports the current claim: **1M server profile with ~903K actual-prompt exact-code retrieval passing 10/50/90% depths under the strict-code-only harness.**

## Caveat

The strict-code-only result validates exact retrieval with a focused prompt and short completion. It should not be conflated with unconstrained long-form generation quality at 900K context. Keep separate quality, chat, and benchmark claims explicit.
