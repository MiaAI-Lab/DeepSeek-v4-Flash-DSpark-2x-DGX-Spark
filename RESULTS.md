# Results — native vLLM 0.25 production candidate

All promoted measurements in this section are from one two-node stack:

- 2× NVIDIA GB10 / SM121, one GPU per node;
- tensor parallel size 2 over NCCL IB/RoCE;
- `deepseek-ai/DeepSeek-V4-Flash-DSpark` revision
  `913f0657a874f76844e2e91cbe706dbcaceeb6d7`;
- vLLM `0.25.2.dev0+g752a3a504.d20260714`;
- native DSpark speculative decoding, FlashInfer B12X, DeepGEMM E8M0;
- **`nvfp4_ds_mla` KV cache**;
- production profile: 200K maximum context, 16 request slots, 16K batched
  tokens, GPU-memory utilization 0.84, DSpark K=5.

The source artifacts and checksums are under
[`results/production-candidate/`](results/production-candidate/). The canonical
aggregate is [`summary.json`](results/production-candidate/summary.json).

## Release gates

| Gate | Result |
|---|---|
| Exact model identity | PASS |
| Deterministic semantic probes | PASS |
| Forced tool-call parsing | PASS |
| Native DSpark speculative counters | PASS |
| `nvfp4_ds_mla` startup/runtime markers | PASS |
| FlashInfer B12X / DeepGEMM E8M0 markers | PASS |
| NCCL IB/RoCE on both ranks | PASS |
| Active Marlin/emulation/fallback markers | none |
| Static concurrency through c16 | PASS |
| Staggered/ragged c16, 250 ms arrivals | 16/16 PASS |
| Uncached 101K-token retrieval | PASS |

The live runtime gate is
[`runtime-gate-final.json`](results/production-candidate/runtime-gate-final.json).
Sanitized startup evidence is in
[`startup-evidence.txt`](results/production-candidate/startup-evidence.txt).

## Decode and static concurrency

The table reports medians from the K=5 production candidate. c1 and c16 use a
five-repeat confirmation run; c2/c4/c8 use the three-repeat sweep. Client usage
and server generation-counter deltas agreed in every promoted row.

| Concurrent requests | Aggregate decode tok/s, median | Observed range | TTFT median | DSpark draft acceptance, median | Success |
|---:|---:|---:|---:|---:|---:|
| 1 | **72.33** | 68.83–72.99 | 0.226 s | 79.4% | 5/5 |
| 2 | **105.99** | 105.52–114.19 | 0.471 s | 82.0% | 6/6 |
| 4 | **165.42** | 128.74–167.40 | 0.443 s | 81.2% | 12/12 |
| 8 | **239.57** | 218.01–242.67 | 0.951 s | 78.7% | 24/24 |
| 16 | **342.70** | 237.87–375.25 | 0.411 s | 78.6% | 80/80 |

The c16 range includes one low outlier; the promoted value is the predeclared
median, not the maximum. Full rows, TTFT, ITL, telemetry, and counter snapshots
are retained in `decode-k5.json` and `decode-k5-confirm.json`.

## DSpark speculative-depth sweep

Each K used the same model, production context/batch profile, prompts,
concurrency levels, request length, and collection method.

| K | c1 median tok/s | c2 | c4 | c8 | c16 | Decision |
|---:|---:|---:|---:|---:|---:|---|
| 3 | 62.21 | 97.14 | 137.92 | 231.78 | 337.00 | rejected |
| 4 | 67.54 | 99.62 | 150.96 | 228.21 | 241.84 | rejected |
| **5** | **70.57** | **105.99** | **165.42** | **239.57** | 310.95 | selected; five-repeat c16 confirmation reached 342.70 |

K=3 had higher acceptance but lower decode throughput at c1/c2/c4/c8. K=5
won those matched three-repeat levels and was then independently stable in its
five-repeat c16 confirmation. Because K=3/K=4 did not receive matched
five-repeat c16 confirmations, that confirmation is not described as a
five-repeat cross-K win. The production default remains K=5.

## Uncached prefill

The prefill harness prepends a different deterministic nonce to every warmup
and measured repeat. This changes the first prefix block and prevents reuse of
completed prefix-cache chains.

| Prompt tokens | Repeats | Prompt tok/s median | Range | Success |
|---:|---:|---:|---:|---:|
| 18,015 | 3 | **1,793.55** | 1,240.09–1,836.17 | 3/3 |

This is a prefill-focused run with `max_tokens=1`; it is not mixed with decode
throughput. The artifact is
[`prefill-16kwords-k5-unique.json`](results/production-candidate/prefill-16kwords-k5-unique.json).

## Long context and NVFP4 capacity

The final production startup reported:

- 462,426 NVFP4 KV tokens;
- 16.25 GiB KV allocation;
- reported 2.31× concurrency at 200,000 tokens/request;
- Python cache dtype and runtime cache marker both `nvfp4_ds_mla`.

An uncached retrieval request used 101,305 actual prompt tokens. The code at the
beginning was returned exactly after 54.43 seconds. Evidence:
[`long-context-100k.json`](results/production-candidate/long-context-100k.json).

The production profile is therefore qualified at 200K. It does **not** claim
one-million-token capacity on vLLM 0.25.

## Staggered/ragged arrivals

Sixteen requests launched 250 ms apart all completed: 16/16, zero HTTP errors.
This exercises overlapping prefill/decode with request slots at different
phases. The legacy staggered script's numerical Prometheus parser is not
label-strict, so its printed throughput and acceptance are retained in the raw
artifact but are deliberately **not promoted as release numbers**. Only the
request-success result is used as a gate.

## Rejected experiments

### 32K batched-token budget

`MAX_NUM_BATCHED_TOKENS=32768` failed closed at startup. After model allocation,
12.59 GiB remained for KV while one 200K request required 14.29 GiB. The runtime
estimated only 30,312 tokens of admissible maximum context. The candidate was
reverted to 16K batched tokens and requalified.

### vLLM 0.25 one-million-token profile

The vLLM 0.25 production lane has fewer than one million measured KV tokens and
cannot truthfully admit a 1,048,576-token request. The prior Stage-C lane remains
available as a separate compatibility/capacity profile in
`profiles/dspark-r0b0tlab-1m.env`. Its results are not mixed into production
throughput tables. The historical 1M concurrency aggregate remains provisional
because the publication and parser disagreed; it is not cited here.

## Historical results and comparison limits

The previous repository report measured a different profile: legacy patched
vLLM, FP8 KV cache, 262K maximum context, GPU-memory utilization 0.80, and
Stage-C-specific scheduler/proposer controls. It reported roughly 49 tok/s at
c1 and 290 tok/s at static c16, plus 191 tok/s for staggered c16.

Those figures are retained in Git history for provenance, but they are **not a
matched baseline** for this NVFP4/200K/vLLM-0.25 release. No formal percentage
speedup is claimed across the profile change. The current release instead
requires its own semantic, native-backend, cache, concurrency, and retrieval
gates to pass.

The earlier 200-question GSM8K FP8 study is likewise historical evidence, not a
new NVFP4 quality result. No new GSM8K score is claimed because the exact public
dataset fixture was not present in this checkout; the release uses deterministic
semantic, tool, and retrieval gates rather than silently substituting a dataset.

## Evidence integrity

Verify all accepted artifacts with:

```bash
sha256sum -c results/production-candidate/SHA256SUMS
```

Synthetic unit tests validate harness behavior only. They are never presented
as GPU performance evidence.
