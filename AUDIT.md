# Serving audit suite (verified on 2x DGX Spark / Anemll 0.1.1, 2026-08-16)

This repo includes a **serving audit suite** — tests that answer "is this deployment
actually good?" beyond "does it respond?" They were designed from first
principles after a false-alarm hunt (a naive benchmark read 45 tok/s where the
truth was 73+; a garble sweep scored the server's *correct* context-limit 400
as a model failure). Methodology and rationale: [`scripts/EVAL.md`](scripts/EVAL.md).

## Run everything

```bash
bash scripts/run-audit.sh --base-url http://127.0.0.1:8888/v1 --model deepseek-v4-flash-0731
```

Or each phase individually:

| Phase | Tool | Checks |
|---|---|---|
| Throughput | `scripts/bench-miaai.py` | C1/C6 decode tok/s (excl TTFT), TTFT |
| Spec-decode health | `scripts/spec-acceptance.py` | Draft acceptance rate + per-position curve |
| Long-context quality | `scripts/ruler-lite.py` | RULER-style retrieval / multi-hop tracing / aggregation at 8k→262k |
| Tool calling | `scripts/tool-battery.py` | Agent tool calls incl. truncation safety (issue55) |
| Deep-context tools | `scripts/deepctx-tool-battery.py` | Tool battery at 32k/131k+ |
| Garble | `scripts/context-garble-sweep.py` | Cold-prefill prompt/schema/secret echo at depth |

## Expected values (this recipe, 2x DGX Spark, Anemll 0.1.1)

- C1 decode: **73-76 tok/s** (Keys abliterated 73.1, official 75.5); C6 aggregate 168-180
- Acceptance: **~68-75%** overall; per-position pos0 ~0.93 → pos4 ~0.33 (normal DSpark curve)
- RULER-lite: **8/8** at 8k/32k (retrieval, variable tracking, common-words)
- Tool battery: **7/7**; deep-context **8/8**; issue55 truncation always `finish=length`
- Garble: **CLEAN** at every length through ~900k tokens (cold prefill)

## Design notes (the hard-won lessons)

1. **Tokenize-verified lengths** — word-count filler estimates lie (~1.56 tok/word
   here, not 1.3). Verify via `/tokenize` or the 1M ceiling gets mis-scored.
2. **Cold prefill only** — garble/truncation/collapse only reproduce on cold
   prefixes (unique nonce busts prefix cache).
3. **Pathological prompts lie** — repeated-word prompts collapse the DSpark
   drafter and read as a fake ~40% regression. Use real task prompts.
4. **Decode rate excludes TTFT**; warm ≥2 requests per prompt length first
   (first request after restart is 6-10s cold autotune).
5. **SSE undercounts** (~4.5x) — always count via `usage.completion_tokens`.
6. **The 1M ceiling is a 400, not a bug** — over-length prompts are correctly
   rejected; that's the server protecting itself.
