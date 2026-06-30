# DSpark-r0b0tlab 384K profile

This document records the current DSpark-r0b0tlab publication candidate for two DGX Spark / GB10 nodes running `DeepSeek-V4-Flash-DSpark` with vLLM TP=2.

## Profile

Use `profiles/dspark-r0b0tlab-384k.env` after setting node-local cluster/model paths.

```env
MAX_MODEL_LEN=384000
MAX_NUM_SEQS=4
MAX_NUM_BATCHED_TOKENS=8192
GPU_MEMORY_UTILIZATION=0.88
MTP_NUM_TOKENS=5
VLLM_DSPARK_CONFIDENCE_SCHEDULER=off
VLLM_DSPARK_CONFIDENCE_THRESHOLD=0.0
VLLM_DSPARK_FUSED_MARKOV_ARGMAX=0
VLLM_DSPARK_LOCAL_ARGMAX=1
```

Do not set `MTP_NUM_TOKENS` / speculative tokens above `5` for this checkpoint. The model config uses `dspark_block_size=5`.

## Native-path requirements

Publication evidence must show these log markers:

- `DeepSeekV4DSparkModel`
- `kv_cache_dtype=nvfp4_ds_mla`
- `Using probe DeepSeek V4 nvfp4_ds_mla KV cache format`
- `Using 'B12X' Mxfp4 MoE backend`
- `DeepGEMM E8M0 enabled on current platform`
- `NCCL INFO Using network IB`
- `DSpark local vocab-parallel argmax is enabled`

## Current evidence summary

All paths below are local benchmark artifact directories from the validation host and are intentionally not required to reproduce the profile.

### Baseline freeze

`benchmark-results/baseline_freeze_20260630_073654/`

Captured `/v1/models`, `/metrics`, head/worker container and image inspect, server logs, model config/index, git state, and a SHA256 manifest before sweeps.

### Throughput ladder

`benchmark-results/publication_ladder_384k_20260630_093518/`

Three repeated runs of `bench_concurrent.py http://127.0.0.1:8888 1,2,4` produced these best-of-two curves:

| Run | c1 agg tok/s | c2 agg tok/s | c4 agg tok/s |
|---:|---:|---:|---:|
| 1 | 57.8 | 88.2 | **158.5** |
| 2 | 57.8 | **89.5** | 144.1 |
| 3 | 55.9 | 84.6 | 140.7 |

The candidate therefore repeats at c4 between **140.7 and 158.5 tok/s**. This beats the earlier initial DSpark c4 artifact of **134.2 tok/s** while running `max_model_len=384000` instead of the older 200K profile.

Staggered c4 at 0.5s arrival spacing:

```text
requests: 4/4 ok, 0 errors
server aggregate gen tok/s over window: 87.3
draft acceptance: 0.564
VERDICT: PASS
```

### Higher-context needle retrieval

Primary artifact:

`benchmark-results/ctx384_needle_300k_20260630_090011/`

Repeat artifact:

`benchmark-results/publication_quality_384k_20260630_093950/needle_300k_repeat/`

The primary 300K target generated ~270.97K actual prompt tokens and passed 10/50/90% depths:

| Depth | Actual prompt tokens | Retrieval | Acceptance |
|---:|---:|---|---:|
| 10% | 270,974 | PASS | 0.5096 |
| 50% | 270,973 | PASS | 0.4816 |
| 90% | 270,975 | PASS | 0.5429 |

A repeat run also passed all depths, but completed earlier after retrieving the needle, so its completion lengths and acceptance rates are not directly comparable to the 256-token primary run.

A 340K target (~307K actual prompt tokens) is near the current reliability boundary: the 50% case passed, the 10% case failed exact retrieval, and the 90% case did not complete before the tool timeout. Do not publish a >300K usable-context claim from the current evidence.

### Correctness / quality

`benchmark-results/publication_ladder_384k_20260630_093518/condense_correctness.log` showed byte-identical deterministic victim output under churn, but the legacy threshold classified the run as FAIL because aggregate acceptance under short churn was 0.333. The failure was not output corruption.

`benchmark-results/publication_quality_384k_20260630_093950/condense_correctness_churn3.log` repeated with churn matched to the 4-sequence profile. Output remained byte-identical; acceptance was 0.375. Treat this as correctness-pass / acceptance-watch, not as a high-concurrency quality certification.

GSM8K smoke:

`benchmark-results/publication_gsm8k_384k_20260630_094051/`

| Run | N | Accuracy | Agreement |
|---|---:|---:|---:|
| c1 | 50 | 50/50 = 1.000 | — |
| c4 | 50 | 49/50 = 0.980 | 49/50 = 0.980 vs c1 |

This is a smoke test, not the full 200-question quality certification in `RESULTS.md`.

## Recommended public claim

Supported by current evidence:

- Two-node DGX Spark / GB10 TP=2 DSpark profile with `max_model_len=384000`.
- Native `nvfp4_ds_mla` KV, B12X/MXFP4 MoE, DeepGEMM, NCCL/RoCE path.
- Needle retrieval passes at ~271K actual prompt tokens across 10/50/90% depths.
- c4 static aggregate repeats at 140.7–158.5 tok/s, above the earlier initial DSpark c4 artifact of 134.2 tok/s.
- c4 staggered requests pass 4/4 with no errors.
- GSM8K N=50 smoke shows 98% c4-vs-c1 prediction agreement.

Not yet supported:

- A reliable 307K+ actual-prompt usable-context claim.
- A static-confidence-threshold scheduler speedup.
- Fused Markov argmax as a default.
- Full 200-question GSM8K quality certification for this exact 384K profile.
