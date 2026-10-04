# Agent clients and the prefix cache: operator notes from six days of production

Scope: one pair of DGX Sparks serving this recipe (DeepSeek-V4-Flash Vision-Exp, the abliterated (Keys) build, recipe `957890a`, TP=2,
`MTP_NUM_TOKENS=5` + `DSPARK_ENABLE_DSPARK_BLOCK_K=1`, `DEFAULT_THINKING=high`, `DSPARK_ENABLE_ISSUE144_EFFORT_ALIGN=1`,
`DSPARK_ENABLE_ISSUE191_TOOLCALL_FAILCLOSED=1`, `DSPARK_ENABLE_DSML_RECOVERY=1`) under real agent traffic for six days:
Claude Code sessions through an Anthropic-protocol shim, OpenAI-compatible chat and desktop clients, and an IDE plugin.
Nothing below changes the recipe. It is about what the clients send, what that does to the prefix cache, and what the
engine's own counters say the pair spends its time on. Numbers come from vLLM `/metrics` (cumulative since the last
restart) and from single-stream probes on an idle pair; the scripts are in `scripts/metrics-*.py` and
`scripts/context-step-probe.py`.

The short version: with the client fixes in section 2 the prefix cache runs at 96%, prefill becomes a small share of
the time, and the pair is mostly a single-stream decode machine whose speed is tokens-per-step divided by a step time that does
not depend on context length.

## 1. What six days of agent traffic look like (4,288 requests)

| Measure (quantiles are interpolated inside histogram buckets) | Value |
|---|---|
| Prompt tokens read / tokens generated | 411 M / 6.0 M |
| Prefix-cache hit rate | 96.3 % (47 % across all clients before the fixes below) |
| Summed per-request time: prefill / decode (overlapping requests double-count) | 3.9 h / 39.1 h (decode 91 %) |
| Sequences per decode step (`inter_token_latency` observations ÷ decode steps) | 1.11 on average; single-sequence steps between 88 % and 96 % |
| Queue wait, preemptions | ~0, 0 |
| Step time (`inter_token_latency_seconds`, one observation per sequence per decode step): p50 / mean / p90 | 69.5 / 82.6 / 124 ms |
| Per-request time per output token: p50 | 21.3 ms |
| Prompt tokens per request: p50 / p90 / p99 | 75 k / 209 k / ~470 k (p99 interpolated inside the 200 k–500 k bucket, which holds 442 requests) |
| Prompt tokens actually computed per request (cache misses): p50 / p90 | 1.3 k / 8.8 k |
| Generated tokens per request: p50 / mean | 491 / 1,380 |
| TTFT: p50 / p90 / p99 | 1.7 / 7.8 / 34 s |
| Requests finished by `max_tokens` | 408 (9.5 %) |
| Drafter acceptance (k=5): overall; per position 0..4 | 49.4 %; 81.6 / 62.3 / 46.0 / 33.2 / 24.0 % |

Reading it: agent transcripts are long (about two thirds are over 50 k tokens) but almost entirely cached, so prefill
is 9 % of the summed request time. Concurrency is low (about one decode step in nine carries a second sequence), so single-stream
step latency dominates and aggregate throughput is secondary for this workload. The conditional
acceptance per draft position is flat (76 / 74 / 72 / 72 %), i.e. the drafter's quality does not decay along the
draft and the fifth position still earns its keep at k=5.

Run `python3 scripts/metrics-decode-report.py` against any engine to get the same table.

## 2. Client behaviours that silently defeat the prefix cache

Each of these was found by diffing consecutive requests captured at the proxy. The symptom is always the same:
`prefix_cache_hits_total` grows far slower than `prefix_cache_queries_total`, and the engine re-reads the whole
transcript every turn (7–18 k tokens per turn here, 5–11 s at ~1.6 k tok/s) while the client thinks nothing changed.

### 2.1 Claude Code's rotating attribution block

Claude Code injects a small rotating block (`...cch=<hash>`) into the system prompt. It is tokenized, sits near the
front of the prefix, and changes per request, so every turn misses from that point on. Fix, in the settings file's
`env` (a shell export is not read):

```json
{ "env": { "CLAUDE_CODE_ATTRIBUTION_HEADER": "0" } }
```

Measured on one session: recent hit rate from ~5 % to 53–78 %, and the episodes in which requests queued on KV
capacity behind a heavy prefill (`num_requests_waiting` > 0 with the KV cache near full, stalling the interactive
request) from ~120 s to ~4 s.

### 2.2 Claude Code's per-turn budget marker (the one that hides behind a shim)

Every turn Claude Code appends a system-role message `<total_tokens>N tokens left</total_tokens>` and keeps the old
ones in the transcript. If an Anthropic→OpenAI shim hoists **every** system-role message into the top-level system
prompt (the obvious implementation), the system text gains one changing line per turn and everything rendered after
it, i.e. the whole history, is re-read. With 27 tool schemas rendered first, the cached-token count pins at exactly
the schema size while prompts grow.

Fix that worked: strip the marker everywhere; hoist only the **leading** system-role messages into `system`; fold any
later system-role message into the next user message as a leading text block (append-only history). Result on a
three-turn probe: consecutive system prompts byte-identical, request 4 re-read 182 tokens with 30,208 cached. Stripping
the marker *before* the hoist does nothing (the hoist re-adds the copies from the message list).

### 2.3 Anything volatile near the top of the prompt

A chat UI that printed the wall clock in its system prompt re-read the whole thread every turn; moving the clock line
to the tail of the newest user message restored 98 % cached tokens on two threads a minute apart. The same applies to
per-turn context injections (open file, errors, reminders): send them in an ephemeral trailing block that is not
stored in the history, so the stored prefix stays byte-identical.

### 2.4 Reasoning-effort flips mid-thread

With `DSPARK_ENABLE_ISSUE144_EFFORT_ALIGN=1` the effort directive renders after the system block instead of right after
BOS, so a client that flips `reasoning_effort` mid-conversation keeps the cached prefix: 1,536 of 1,726 prefix tokens
stayed cached on a flip (stock: 0). With no system message the rendered prompt is byte-identical to stock.

### 2.5 Attaching project files

If a client attaches several files each turn, serialize them byte-identically and order them by volatility: stable
files first, manifests at the end of the stable block, the file being edited last. Measured on a 39 k-token project
attachment: an edit to the hottest file re-read 253 tokens instead of 39,165, and a repeat turn was 99.6 % cached.

## 3. Images in long agent transcripts

`LIMIT_MM_PER_PROMPT` counts every image in the **resent conversation**, not "open" images. A screenshot-heavy agent
session (an image-read tool call on each step) crosses the cap after that many screenshots and every later turn fails with
HTTP 400 `At most N image(s) may be provided in one prompt`. Raising the cap (`image=16`) only moves the wall
(the 17th image returns HTTP 400, verified).

The client-side rule that works, mirroring what the vendor's own harness does: keep the newest N images, replace older
ones with a text placeholder that carries the file path so the model can re-read one on demand, and offload in batches
(here: budget 12, batches of 4) so the prefix changes rarely rather than every turn. Verified on a 20-image session:
never more than 12 images in flight, placeholders fixed in batches, the model reported correctly which images had been
offloaded. 12 images cost 1,438 prompt tokens and 4.1 s.

## 4. Reasoning effort, temperature, and bare clients

Measured on an 18-problem set with ground truth (AIME-style plus arithmetic), thinking on, temperature 0, `max_tokens`
32 k, three-way concurrency:

| `reasoning_effort` | Correct | Cap hits (32 k) | Mean output tokens, finished items only | Mean output tokens, cap hits counted at 32 k | Mean wall time |
|---|---|---|---|---|---|
| low (empty directive) | 16 / 18 | 2 | 1,976 | 5,312 | 143 s |
| high | 18 / 18 | 0 | 2,463 | 2,463 | 109 s |
| max | 14 / 18 | 4 | 1,146 | 8,002 | 279 s |

Output tokens are `usage.completion_tokens` (reasoning plus answer). At temperature 0 `high` was the only level that
finished every item; per finished item it spent more tokens than `low` or `max`, but counting the runaways at their
cap it was the cheapest level overall and the fastest by wall time. Every miss at `low` and `max` was a cap hit, never
a wrong answer (`high` solved both of `low`'s runaways in 9.1 k and 4.8 k output tokens). The levels were not re-run
at 0.7.

Two things to know before benchmarking this yourself:

- Reasoning length is strongly a temperature effect. At temperature 0 the model takes a long greedy reasoning path
  (~10 k tokens on a code prompt); at 0.7 the same prompt reasons in 0.7–2.3 k. Do not conclude from a temperature-0
  run that a level is broken.
- `thinking_token_budget` / `thinking_budget` is ignored on this build (Model Runner V2; the server logs
  `thinking_token_budget unsupported`). Clients that rely on it get no cap; use `max_tokens`.

Bare clients (no `chat_template_kwargs`) get `DEFAULT_THINKING`. With `high` as the server default, that includes an
agent harness's small side requests (titles, summaries, classifiers, typically `max_tokens` ≤ 5 k): they reason at high
against a small cap and end on `length`, sometimes with empty content. 9.5 % of the requests in section 1 finished on
`max_tokens`; the attribution to clients is pending a per-request log, but a proxy in front of such clients should send
`{"thinking": false}` explicitly for requests that carry no thinking configuration.

## 5. What sets tokens per second (single-stream probes on the idle pair)

`scripts/metrics-delta-probe.py` measures one request at a time from the spec-decode counter deltas, so the
per-position acceptance is exact for that request. Same checkpoint and flags as section 1; short prompts (36 tokens
for the prose case, 68 for code, 2,563 for the copy case); thinking off unless stated; one sample per cell. Cases that
overlapped with other traffic are omitted.

| Case | Temp | Acceptance | Tokens / step | tok/s | Step (ms) | Per-position acceptance |
|---|---|---|---|---|---|---|
| Prose, 400-word essay | 0.3 | 20.7 % | 2.03 | 31.8 | 63.9 | 62 / 30 / 10 / 2 / 0 |
| Prose | 0.7 | 24.7 % | 2.23 | 33.9 | 66.0 | 69 / 35 / 13 / 4 / 2 |
| Prose | 1.0 | 21.6 % | 2.08 | 32.3 | 64.6 | 62 / 30 / 12 / 4 / 1 |
| Prose, thinking on (high) | 0.7 | 21.9 % | 2.10 | 34.4 | 60.9 | 62 / 26 / 15 / 5 / 2 |
| Code, Three.js scene | 0.0 | 70.8 % | 4.54 | 71.3 | 63.5 | 95 / 82 / 73 / 59 / 45 |
| Code | 0.3 | 76.2 % | 4.81 | 75.5 | 63.7 | 95 / 88 / 78 / 68 / 51 |
| Code | 1.0 | 68.0 % | 4.40 | 69.1 | 63.6 | 95 / 84 / 66 / 54 / 41 |
| Verbatim copy of a 120-line snippet with one rename | 1.0 | 99.7 % | 5.98 | 84.1 | 71.2 | 100 / 100 / 100 / 99 / 99 |

And step time versus context (`scripts/context-step-probe.py`, prose, temperature 0.7, 300 tokens, nested prefixes so
the shorter contexts are cache hits; one sample per row):

| Prompt tokens | Newly computed | TTFT | Prefill (new tok/s) | Step (ms) | tok/s |
|---|---|---|---|---|---|
| 1.9 k | 1.9 k | 1.1 s | 1,700 | 68.6 | 31.2 |
| 25 k | 23 k | 12.8 s | 1,800 | 67.1 | 35.2 |
| 75 k | 50 k | 33.4 s | 1,500 | 67.1 | 32.4 |
| 150 k | 75 k | 61.2 s | 1,230 | 68.9 | 29.0 |

What follows from the two tables:

- **Step time is flat with context** (67–69 ms from 2 k to 150 k tokens) and within about 10 % across prose, code
  and copy (61–71 ms). Speed differences between workloads are therefore tokens-per-step, i.e. acceptance, and
  acceptance is set by what the model is writing: about 2 tokens per step on prose, 4.5–4.8 on code, 6 (the k=5
  ceiling) when it re-emits text that is in the context. The "prose 30 tok/s, code 75 tok/s" experience on this pair
  is this table.
- **Temperature barely moves prose acceptance.** On code the cells differ by up to 9 % (0.3 highest, 0.0 below it),
  which one sample per cell cannot resolve.
- **Prefill does degrade with context**: newly computed tokens go from ~1.8 k tok/s at 25 k to ~1.2 k at 150 k. That is
  the TTFT tail in section 1 (p90 7.8 s, p99 34 s: compactions and large file reads), not a decode cost.
- **On the draft depth**: positions 3–4 contribute almost nothing on prose and a lot on code and copy. With the
  trained `dspark_block_size=5` under block-K the five draft tokens come from one parallel pass, so k=5 is the right
  depth for a mixed workload; the way to make a re-emit-the-whole-file workflow faster is to emit less (patches
  instead of full documents), not a longer draft.
- **Step floor, an estimate**: with roughly 13 B active parameters at 4 bits, one step streams about 6.5 GB of
  weights, half per box under TP=2, which is about 12 ms at a GB10's ~273 GB/s. The measured 67 ms is several times
  that, so the single-stream step is dominated by per-layer synchronization and launch overhead rather than by weight
  bandwidth. That is where kernel work pays; nothing on the client side changes it.

## 6. Reproducing

```bash
# what the engine has been doing since its last restart (any vLLM, spec decode or not)
python3 scripts/metrics-decode-report.py http://127.0.0.1:8888/metrics

# acceptance by content type and temperature; waits for an idle engine before each case, flags overlaps
python3 scripts/metrics-delta-probe.py --base-url http://127.0.0.1:8888 --model deepseek-v4-flash-vision-exp

# step time and prefill throughput at 2k / 25k / 75k / 150k tokens of context
python3 scripts/context-step-probe.py --base-url http://127.0.0.1:8888 --model deepseek-v4-flash-vision-exp
```

Both probes write their JSON to `results/` and take `--api-key` when `DSPARK_API_KEYS` guards the chat endpoint.

Method notes. Quantiles in section 1 are linear interpolations inside Prometheus histogram buckets, so a p99 is only
as precise as its bucket. `inter_token_latency_seconds` is observed once per sequence per decode step (its count
equals the draft count), which is what makes it a step-time distribution here, and the sequences-per-step figure is
its count divided by the engine's decode steps. `--base-url` on the probes accepts either `http://host:8888` or the
`http://host:8888/v1` form the other scripts use. Per-request acceptance is not exposed by vLLM. Deltas of `spec_decode_num_drafts_total`,
`spec_decode_num_draft_tokens_total`, `spec_decode_num_accepted_tokens_total` and
`spec_decode_num_accepted_tokens_per_pos_total` around a single request on an idle engine give it exactly; the probes
mark a case `contaminated` when the engine counted more generated tokens than the request's `usage` reports, which
means another request overlapped.

Caveats: one pair, one checkpoint (the Keys abliterated Vision-Exp build), one recipe revision; production traffic is
not a controlled workload; the probe rows are single samples unless stated. Prompts used by the probes are in the
scripts.
