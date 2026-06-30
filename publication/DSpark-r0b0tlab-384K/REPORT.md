# DSpark-r0b0tlab 384K publication bundle

This bundle summarizes the evidence for the DSpark-r0b0tlab 384K candidate without raw host logs, local usernames, local paths, or private IP addresses.

## Configuration

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
VLLM_DSPARK_REPLICATE_MARKOV_W1=1
VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1
```

The served model endpoint reported `max_model_len=384000`.

## Native acceleration path

The captured startup logs showed these native markers:

- `DeepSeekV4DSparkModel`
- `nvfp4_ds_mla` KV cache
- DeepSeek V4 `nvfp4_ds_mla` probe path
- `B12X` MXFP4 MoE backend
- `DeepGEMM E8M0`
- NCCL over IB/RoCE
- DSpark local vocab-parallel argmax

## Static throughput ladder

Three clean repeated ladders were run with the same harness at c1/c2/c4. Throughput is server-side aggregate generation tokens/second from vLLM metrics deltas.

| Run | c1 agg tok/s | c2 agg tok/s | c4 agg tok/s |
|---:|---:|---:|---:|
| 1 | 57.8 | 88.2 | 158.5 |
| 2 | 57.8 | 89.5 | 144.1 |
| 3 | 55.9 | 84.6 | 140.7 |

Earlier initial DSpark c4 reference: 134.2 aggregate tok/s. The 384K candidate repeats above that c4 reference while using a higher context ceiling.

## Staggered/ragged c4

```text
requests: 4/4 ok, 0 errors
server aggregate gen tok/s: 87.3
draft acceptance: 0.564
verdict: PASS
```

## Higher-context needle retrieval

Primary target: 300K synthetic-token prompt construction, producing about 270.97K actual API prompt tokens.

| Depth | Actual prompt tokens | Retrieval | Acceptance |
|---:|---:|---|---:|
| 10% | 270,974 | PASS | 0.5096 |
| 50% | 270,973 | PASS | 0.4816 |
| 90% | 270,975 | PASS | 0.5429 |

A repeat run passed all 10/50/90% depths as well, but completed after finding the needle and therefore generated fewer completion tokens; its acceptance rates are not directly comparable to the primary fixed-length run.

Boundary finding: a 340K construction produced ~307K actual prompt tokens. The 50% case passed, the 10% case failed exact retrieval, and the 90% case did not complete before the run timeout. Do not claim reliable 307K+ usable context from this evidence.

## Correctness and quality smoke

- Deterministic victim output under churn was byte-identical to the alone reference.
- The legacy condense script's acceptance threshold is too strict for this 4-sequence deep-context profile with very short churn prompts: acceptance was 0.333–0.375, not a zero-collapse, and no output corruption was observed.
- GSM8K N=50 smoke:

| Mode | Accuracy | Agreement |
|---|---:|---:|
| c1 | 50/50 = 1.000 | — |
| c4 | 49/50 = 0.980 | 49/50 = 0.980 vs c1 |

This is a smoke test, not a replacement for the existing full 200-question quality certification.

## Supported public claims

- Two-node DGX Spark / GB10 TP=2 DeepSeek-V4-Flash-DSpark profile with `max_model_len=384000`.
- Native `nvfp4_ds_mla` KV, B12X/MXFP4 MoE, DeepGEMM, and NCCL/RoCE path.
- Needle retrieval passes at ~271K actual prompt tokens at 10/50/90% depths.
- Static c4 aggregate repeats at 140.7–158.5 tok/s, above the earlier initial DSpark c4 result of 134.2 tok/s.
- Staggered c4 passes 4/4 with no errors.
- GSM8K N=50 smoke shows 98% c4-vs-c1 agreement.

## Not supported yet

- Reliable >300K actual-prompt retrieval.
- A static confidence-threshold scheduler speedup.
- Fused Markov argmax as default.
- Full 200-question GSM8K certification for this exact 384K profile.
