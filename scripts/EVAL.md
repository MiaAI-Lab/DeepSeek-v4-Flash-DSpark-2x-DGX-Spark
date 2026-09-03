# Evaluation methodology (DS4 DSpark serving audit)

Why these tests, and how they were designed. Sources: NVIDIA RULER
(arXiv:2404.06654), vLLM speculative-decoding docs, Snowflake + Red Hat spec-decode
benchmarks, and MiaAI-Lab's own `scripts/stability-quick.py` + `scripts/benchmark-0731.py`.

## Principles (learned the hard way — all verified on-cluster 2026-08-16)

1. **Tokenize-verified context lengths, never word-count estimates.**
   The naive "1.3 words/token" filler heuristic under-pads ~1.2x on the DS4
   tokenizer (this English filler runs ~1.56 tok/word). A "--lengths 524288" sweep
   actually sent ~815K tokens, and 700K+ blew past the 1M ceiling — the server's
   *correct* 400 got scored as a *false* "GARBLE DETECTED". Every length here is
   confirmed via `/tokenize` before the request is sent.

2. **Cold prefill only.** Garble, tool-call truncation, and long-context collapse
   only reproduce on cold prefills (unique nonce busts the prefix cache). Warm
   requests passing proves nothing.

3. **Pathological prompts lie.** A repeated-word prompt ("distributed distributed…")
   triggers degenerate repetition that collapses the DSpark drafter (per-position
   acceptance 0.93→0.33 on good prompts vs 0.72→0.11 on the bad one) and reads as a
   fake ~45 tok/s regression. Use unique cold prefixes + real task instructions.

4. **Decode rate excludes TTFT.** Wall-clock tok/s includes prefill time. Report
   decode tok/s after first token (MiaAI's `median_output_tok_s`), plus TTFT
   separately. First request after a restart has a 6-10s cold-autotune TTFT that is
   NOT a regression — warm ≥2 requests per prompt length first.

5. **SSE undercounts.** DSpark packs multiple accepted draft tokens per stream
   chunk. Count via `usage.completion_tokens` (stream with
   `stream_options={"include_usage": true}`) or non-streaming usage — never
   per-chunk `delta.content` events (~4.5x undercount).

## Phases

| Phase | Tool | What it tests | Why |
|---|---|---|---|
| 1. Throughput | `bench-miaai.py` | C1/C6 decode tok/s (excl TTFT), TTFT | Matches MiaAI's 08-14 matrix; C1 ≈ 75.5 official / 73 Keys on this recipe |
| 2. Spec-decode health | `spec-acceptance.py` | Overall + per-position draft acceptance from /metrics | THE lever for decode speed; abliteration drains it (0.585 vs 0.718); drift = red flag |
| 3. RULER-lite quality | `ruler-lite.py` | S/MK-NIAH retrieval, variable tracking (multi-hop), common-words aggregation at 8k→262k | Beyond shallow NIAH: tests real long-context reasoning, not just recall (RULER's core finding: NIAH-perfect models fail tracing/aggregation at depth) |
| 4. Tool calling | `tool-battery.py` | Single/complex/parallel/multi-turn calls + issue55 truncation | The agent-serving path; issue55 = truncated calls must report `finish=length`, never broken tool JSON |
| 5. Deep-context tools | `deepctx-tool-battery.py` | Same battery at 32k/131k+ | Tool calls at depth (the Hermes/agent case) |
| 6. Garble | `context-garble-sweep.py` | Cold-prefill prompt/schema/secret echo at depth | The classic DS4 failure mode; only visible cold |

## Expected values (this recipe, 2x DGX Spark, Anemll 0.1.1, verified 2026-08-16)

- C1 decode: **73-76 tok/s** (Keys abliterated 73.1, official 75.5); C6 aggregate 168-180
- Acceptance: **~68-75%** overall, pos0 ~0.93 → pos4 ~0.33 (normal DSpark collapse)
- RULER-lite: retrieval 100%, tracing/aggregation pass at all tested lengths
- Tool battery: 7/7 + deep-context 8/8; issue55 truncation always `finish=length`
- Garble: CLEAN at every length up to the 1M ceiling

If a phase regresses, fix the recipe, not the test.
