# GLM-NEW-REPORT — Improving decode tok/s and (accuracy + tool-calling) quality for DeepSeek-V4-Flash-0731

> Scope: this report was produced by reading this repository
> (`DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`) at `HEAD` (commit `b131b2a`).
> It targets **only** the default DeepSeek-V4-Flash-0731 DSpark lane (Anemll
> image, NVFP4 DS-MLA KV, MTP-5 probabilistic spec decode, `deepseek_v4`
> tokenizer + tool/reasoning parser). It does **not** cover the experimental
> GLM-5.2 lane under `vLLM-Moet/`. No new benchmarks were run; the findings
> below consolidate the live config, the published sweep, and the repo's own
> docs/patches into a single, actionable improvement plan for DeepSeek V4
> Flash decode tok/s and tool-calling quality.

## 1. The production serve profile (what's actually running)

Defaults come from `docker-compose.dspark.yml` (the compose command) and
`.env.dspark.example`. The shipped serve command is:

```
--tensor-parallel-size 2 --pipeline-parallel-size 1
--kv-cache-dtype nvfp4_ds_mla --block-size 256
--max-model-len 1048576 --max-num-seqs 6 --max-num-batched-tokens 8192
--max-cudagraph-capture-size <seqs*(k+1)>
--gpu-memory-utilization 0.80
--enable-prefix-caching --enable-prompt-tokens-details
--async-scheduling --enable-chunked-prefill
--speculative-config {"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}
--moe-backend flashinfer_b12x
--tokenizer-mode deepseek_v4
--tool-call-parser deepseek_v4 --enable-auto-tool-choice
--reasoning-parser deepseek_v4
--reasoning-config {"reasoning_parser":"deepseek_v4",...}
--default-chat-template-kwargs {"thinking":true,"reasoning_effort":"low"}
--generation-config vllm
--enable-flashinfer-autotune --nnodes 2
```

Runtime env that is wired in by default (`.env.dspark.example`, compose):

- `MTP_NUM_TOKENS=5` (checkpoint `dspark_block_size` floor; k<5 silently truncates)
- `DEFAULT_THINKING=low` (DeepSeek V4 base reasoning mode)
- `VLLM_USE_BREAKABLE_CUDAGRAPH=0` (regular CUDA graphs — the *faster* path)
- `VLLM_USE_B12X_MOE=1`
- `VLLM_USE_FLASHINFER_SAMPLER=1`
- `CUTE_DSL_ARCH=sm_121a` / `TORCH_CUDA_ARCH_LIST=12.1a`
- `MAX_NUM_BATCHED_TOKENS=8192`, `MAX_NUM_SEQS=6`, `GPU_MEMORY_UTILIZATION=0.80`

Two structural facts to keep in mind for the rest of the report:

1. **The Anemll `0.1.1` image does not register the Stage-C `VLLM_DSPARK_*`
   kill-switches** (`docs/ENVS.md`). Setting them on the default image only
   emits "Unknown vLLM environment variable" warnings and is a no-op. They
   become live only when you build/run the Stage-C image
   (`docker-compose.stage-c.override.yml`).
2. **`max_model_len`/`max_num_seqs` are ceilings, not reservations.** The KV
   pool is shared and allocated on demand; the real constraint is
   `sum(live tokens) <= KV pool`, not `seqs × max_len` (README "How the KV
   cache works"). So raising concurrency is *free* up to the pool size.

## 2. Decode tok/s — current numbers and levers

### 2.1 Current measured baseline

From the published 0731 sweep on 2× DGX Spark, TP=2, NVFP4 DS-MLA, MTP-5
(`results/deepseek-v4-flash-0731-2x-dgx-spark.json` and the README table).
Per-stream decode tok/s and aggregate tok/s:

| Prompt | C1 decode | C1 agg | C6 decode | C6 agg |
|---:|---:|---:|---:|---:|
| 256 | 75.4 | 69.1 | 36.9 | 191.2 |
| 2,048 | 68.8 | 62.0 | 34.7 | 143.7 |
| 8,192 | 73.9 | 43.7 | 23.6 | 73.1 |
| 32,768 | 64.0 | 16.6 | 10.8 | 27.9 |
| 131,072 | 65.2 | 5.9 | — | 6.6 |

A separate published capture (2048-token completions, concurrency 1–6) showed
peak aggregate **134.6 tok/s at x3**, dropping to 120.4 tok/s at x4 where TTFT
spikes to 5.36 s. The README also documents a 200K/16 high-concurrency lane
that reaches **315 tok/s static / 205 tok/s staggered** aggregate (see §2.3).

Three observations that point the levers:

- **Per-stream decode tok/s is high at C1 (~65–75 tok/s) and degrades as
  concurrency rises** — classic memory-bandwidth sharing under the
  shared KV pool. Aggregate tok/s keeps climbing, which is the whole point of
  concurrency, but each user feels the drop.
- **TTFT spikes hard under concurrency at long context** (8K prompt × 6 ≈
  24 s TTFT; 32K × 6 ≈ 85–98 s TTFT). Prefill is being starved by decode
  traffic in the shared KV pool — a scheduler/memory headroom issue, not a
  weights issue.
- **The cheapest decode-tok/s wins in this repo are config knobs that are
  already documented but not on by default for every workload class.**

### 2.2 Lever 1 — Pick the right concurrency profile per workload class

The repo ships three validated profiles; choosing the right one is the single
biggest decode-tok/s lever for free (no quality cost):

| Workload | Profile | Result |
|---|---|---|
| Deep-context agent traffic (most sessions ≪ 1M, occasional long one) | `MAX_MODEL_LEN=1048576`, `MAX_NUM_SEQS=6` (default) | 1M ceiling + 6 concurrent; ~182 tok/s aggregate at 6× full-1M-microbench |
| Short-context, high aggregate rate | `MAX_MODEL_LEN=200000`, `MAX_NUM_SEQS=16` + Keys patch + `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1` | 315 tok/s static / 205 tok/s staggered aggregate |
| Big-prompt coding / long single prompts | `MAX_NUM_BATCHED_TOKENS=16384`, `MAX_NUM_SEQS=4`, `GPU_MEMORY_UTILIZATION=0.87` | Lifts prefill throughput ceiling; capture size shrinks |

The 200K/16 lane is the documented path when raw aggregate tok/s matters more
than the 1M context ceiling. It is the only one of the three that needs the
Keys concurrency patch (Patch 1/2/2b in `docs/PATCHES.md`) and the
`VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1` ragged-context path.

### 2.3 Lever 2 — `MAX_NUM_BATCHED_TOKENS` and `GPU_MEMORY_UTILIZATION`

Both are pure .env.dspark knobs, documented but not the default for the
1M/6 profile. For long coding / big prompts the README explicitly recommends:

```env
MAX_MODEL_LEN=1048576
MAX_NUM_SEQS=4
MAX_NUM_BATCHED_TOKENS=16384
GPU_MEMORY_UTILIZATION=0.87
```

- Raising `MAX_NUM_BATCHED_TOKENS` 8192 → 16384 lifts the **prefill** throughput
  ceiling (more tokens per prefill step). Watch the capture size: it is
  `MAX_NUM_SEQS * (MTP_NUM_TOKENS + 1)`, so at k=5 / seqs=4 it is 24.
- Raising `GPU_MEMORY_UTILIZATION` 0.80 → 0.85–0.87 grows the KV pool. The 0731
  lane measured ~2.49M tokens at util 0.835 and ~2.8M at the prior 0.85 lane.
  More KV pool = more headroom for concurrent deep-context requests.
- If a graph capture OOMs at higher util / k, the repo's own guidance is to
  **lower util toward ~0.78** rather than drop `MTP_NUM_TOKENS` below 5 (k<5
  silently truncates the draft blocks).

### 2.4 Lever 3 — Keep regular CUDA graphs (`VLLM_USE_BREAKABLE_CUDAGRAPH=0`)

The README's natural-completion probe at full 1M context directly compares the
two graph modes on the 0731 lane:

| Mode | Breakable graphs | Regular graphs | Change |
| --- | ---: | ---: | ---: |
| C1 decode, warm median | 74.6 tok/s | 95.9 tok/s | +28.6% |
| C2 aggregate decode, median | 134.2 tok/s | 151.8 tok/s | +13.1% |

Regular graphs are strictly faster here, and the default `VLLM_USE_BREAKABLE_CUDAGRAPH=0`
already opts into them. **Do not regress** to the auto-breakable path that
Anemll enables when this env is unset — it costs ~13–28% of decode tok/s.
This is the single biggest "don't break what works" item in the repo.

### 2.5 Lever 4 — `MTP_NUM_TOKENS` tuning (carefully)

`MTP_NUM_TOKENS=5` is the floor (checkpoint `dspark_block_size=5`; k<5 silently
truncates draft blocks). The capture size is `MAX_NUM_SEQS * (k+1)` — at the
default k=5/seqs=6 that is 36. The README documents a live cluster running
k=6/seqs=4/util=0.835 successfully.

Higher k can improve effective tok/s via acceptance, but:
- It grows the cudagraph capture size and the KV-store draft footprint.
- It can OOM at capture time on the default util 0.80 — the safe response is
  to **lower util**, not drop k below 5.
- `--max-cudagraph-capture-size` is auto-derived from `seqs*(k+1)` in the
  compose command, so it tracks k for free.

This lever is workload-dependent and should be A/B'd against the published
benchmark protocol (§4), not trusted on a single cold run.

### 2.6 Lever 5 — Stage-C kill-switches (only with the Stage-C image)

The default Anemll image does not register `VLLM_DSPARK_*` / `VLLM_DSV4_*`
keys (`docs/ENVS.md`). To unlock the rest of the DeepSeek-V4-Flash-DSpark
concurrency machinery — the Keys concurrency patch and the ragged
`query_start_loc` context path — you must:

1. Build the Stage-C image: `./build-dspark-vllm-runtime.sh`
2. Set `DSPARK_VLLM_IMAGE=vllm-dspark-runtime:dspark-nvfp4-stage-c`
3. Merge `docker-compose.stage-c.override.yml`
4. Enable the documented block (these become real, not warnings):

```env
VLLM_USE_B12X_WO_PROJECTION=1
VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1
VLLM_DSPARK_LOCAL_ARGMAX=1
VLLM_DSPARK_REPLICATE_MARKOV_W1=1
DSPARK_SLOT_CLAMP=1
# ...see .env.dspark.example Stage-C block
```

`VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1` is the one that makes the ragged
`query_start_loc` path active — it is required for the 200K/16 high-concurrency
profile to be correct under real (staggered, mixed prefill/decode) arrivals.
The Anemll default silently runs the uniform-row path, which 500s on
non-uniform batches (`docs/PATCHES.md` Patch 2/2b).

### 2.7 Lever 6 — `--enable-prefix-caching` for effective throughput

The default compose ships with `--enable-prefix-caching` and
`--async-scheduling` on. For agent workloads that repeat a long system prompt
or tool schema, prefix caching is a major effective-throughput win — the
cached prompt blocks don't just save prefill FLOPs, they leave more KV pool
for the actual conversation. Keep it on; do not tune it off.

### 2.8 Lever 7 — `--block-size 256` and KV-pool sizing

`--block-size 256` is the documented PagedAttention block size. It is not a
free .env knob in this repo, but it is worth re-tuning in tandem with
`GPU_MEMORY_UTILIZATION` if the resident-tier settle pattern in the live
benchmarks (§4) becomes unstable.

### 2.9 Lever 8 — Disabling earlyoom on the host

README "Notes": disable earlyoom on the DGX Spark hosts to avoid spurious OOM
kills of vLLM worker/head processes under transient GPU memory pressure during
deep-context concurrency. This is a host-level setting, not a vLLM knob:

```bash
sudo systemctl stop earlyoom && sudo systemctl disable earlyoom
```

It is free insurance against a decode-tok/s cliff caused by a killed worker
under sustained load — keep it in the runbook.

## 3. Accuracy + tool-calling quality levers

### 3.1 The core tool-calling issue: DSML syntax temperature asymmetry

Fully diagnosed in `docs/DSML_SYNTAX_TEMP_ASYMMETRY.md`. This is the
single most important quality finding for the DeepSeek-V4-Flash-0731 lane and
is worth restating as a lever:

- DeepSeek-V4 tool calls are DSML markup: `<｜DSML｜tool_calls>`,
  `<｜DSML｜invoke name=…>`, `<｜DSML｜parameter …>`. The upstream **ds4** engine
  forces `temperature=0` for **syntax tokens** (tags, invoke headers, JSON
  punctuation) and only samples *payload* tokens at request temperature.
  Structural tokens can never be derailed by temperature.
- **vLLM (this recipe)** samples **every** token at request temperature —
  DSML headers, parameter tags, braces, and payload alike. At temp 1.0 each
  structural token is a full-temperature sample that can come out malformed.
- `tool-eval-bench` reads tool calls **only** from the structured
  `message.tool_calls` field. A call whose DSML is malformed enough that
  `--tool-call-parser deepseek_v4` cannot parse it (the strict
  `parse_tool_calls` in `encoding/encoding_dsv4.py` raises `ValueError` on any
  malformed structural token) is scored as **"no tool call"
  (`missing_step` failure)**. So at temp > 0, vLLM loses points for a
  *mechanical* serving artifact, not a model-capability gap.

**Mitigations, cheapest first:**

1. **Run the tool-eval-bench at temp 0.** vLLM is greedy everywhere, so syntax
   cannot be derailed — ds4's robustness for free, at the cost of also making
   payloads greedy. This is the clean diagnostic that separates "syntax
   corruption" from "genuine capability gap."
2. **Structured outputs with xgrammar structural tags.** Author a DSML
   structural-tag grammar and verify it composes with the `deepseek_v4` tool
   parser and streaming. Upstream RFC `vllm-project/vllm#32142` is still open
   for structural-tag function calling — this is the principled, robust fix.
3. **Custom logits processor.** Track DSML parse state per request and force
   argmax inside syntax regions — effectively a port of ds4's hand-written C
   `try_repair_dsml` logic. Heaviest to build, but gives ds4-grade robustness
   without a grammar.

### 3.2 Encoder install correctness (re-verify after any vLLM bump)

The 0731 checkpoint ships no Jinja `chat_template`; compose installs
`encoding/encoding_dsv4.py` into vLLM on both ranks and patches a
`low/high/max` reasoning-effort bug in the vLLM tokenizer wrapper (see the
compose command block). Failure modes if this drifts:

- `DEFAULT_THINKING` maps incorrectly (`off/low/high/max` → wrong kwargs).
- Multi-turn tool semantics break silently: the bench strips
  `reasoning_content` from history; a mismatch in the encoder degrades
  multi-turn tool scenarios specifically (`docs/DSML_SYNTAX_TEMP_ASYMMETRY.md`).
- The README "Gotcha" explicitly warns: if direct vLLM prompts are clean but
  an agent harness garbles, check the harness session replay / fallback list /
  prompt-tool-XML handling *before* blaming DSpark weights.

Keep `DEFAULT_THINKING=low` (DeepSeek V4 base reasoning mode; opens `未成年`
adds no effort prefix). Validate `off`/`low`/`high`/`max` after any image bump.

### 3.3 Do not override generation config server-side

The compose uses `--generation-config vllm` and intentionally does **not**
install a server-side `--override-generation-config`. Explicit client request
parameters still win. The "Gotcha" README section warns that unstable sampling
and hidden fallback transitions are a harness artifact — keep the server
deterministic and let the client control sampling.

### 3.4 Use pi's specialized DeepSeek mapping, not the generic one

`pi-models.dspark.example.json` maps pi's `off/low/high/max` selector to
request-level `chat_template_kwargs.thinking` + `chat_template_kwargs.reasoning_effort`.
vLLM returns generated reasoning in its `reasoning` stream field; pi recognizes
it, stores it as a thinking block, and replays it as `reasoning`; vLLM
normalizes to `reasoning_content` before the custom encoder runs, so tool-call
reasoning is not lost. **Do not use pi's generic top-level OpenAI
`reasoning_effort`** — the custom encoder only reads these values from
`chat_template_kwargs`.

### 3.5 Tool-turn `reasoning_content` retention

The Unsloth GGUF Jinja template that the README references implements the same
central DS4 behavior — DSML tools, `未成年` boundaries, `high`/`max` effort
prefixes, and retention of tool-turn `reasoning_content`. The custom encoder
used here is the source of truth; validate multi-turn role boundaries and
tool-turn reasoning retention after every encoder install.

## 4. Measurement discipline (applies to every lever above)

The repo's own benchmark methodology lives in `benchmarks/bench_decode_only.py`
and `scripts/benchmark-0731.py`. The non-negotiable rules for trusting any
DeepSeek-V4-Flash-0731 number:

1. **Measure tokens via API `usage.completion_tokens`, not SSE chunk counts.**
   MTP emits multi-token chunks; chunk counts skew the rate
   (`benchmarks/bench_decode_only.py`).
2. **Use the decode-only window.** Per-stream decode tok/s =
   `(completion_tokens − 1) / (t_last − t_first)`; aggregate =
   `sum(completion_tokens − 1) / (max t_last − min t_first)` — prefill and the
   first token are excluded so MTP multi-token chunks don't skew the rate.
3. **Use differentials, not first-token-inclusive runs.** e.g.
   `(tokens_384 − tokens_32)` time — prefill contaminates the first window.
4. **Distinct first cache block per request** in the sweep harness so prefix
   caching cannot make later cases reuse earlier prefill work (the 0731 sweep
   in `scripts/benchmark-0731.py` already does this).
5. **Do not trust a single run.** The published 0731 sweep takes the median of
   several trials; the historical "60 tok/s baseline" was a separate
   diagnostic, not the production path — do not confuse the two.
6. **Re-run the OpenAI smoke + agent-client validation after any change.** The
   README "Gotcha" and `docs/PATCHES.md` both say: validate the direct
   OpenAI-compatible API path first, then test the agent harness; treat the
   service as validated only after the built-in smoke request plus
   agent-client validation pass on the live service.

## 5. Prioritized next steps (DeepSeek-V4-Flash-0731 only)

### Immediate (free, no quality risk)
1. **Pick the concurrency profile by workload class** (§2.2): keep 1M/6 for
   deep-context agents; switch to 200K/16 (+ Keys patch +
   `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`) for short-context high-aggregate
   traffic; switch to `MAX_NUM_BATCHED_TOKENS=16384` + `MAX_NUM_SEQS=4` +
   `GPU_MEMORY_UTILIZATION=0.87` for big-prompt coding.
2. **Keep `VLLM_USE_BREAKABLE_CUDAGRAPH=0`** — regular graphs are +13–28% on
   this lane. Never let it auto-enable.
3. **Run the tool-eval-bench regression at temp 0** to isolate DSML
   syntax-corruption from any real capability gap (§3.1).
4. **Disable earlyoom on both DGX Spark hosts** to remove a decode-tok/s cliff
   under sustained deep-context load (§2.9).

### Short term (medium effort, modest risk)
5. **Bump `MAX_NUM_BATCHED_TOKENS` 8192 → 16384 and
   `GPU_MEMORY_UTILIZATION` 0.80 → 0.85–0.87** for the big-prompt lane. If
   graph capture OOMs, lower util toward ~0.78, not k below 5 (§2.3, §2.5).
6. **Validate the Stage-C high-concurrency path** end-to-end on the Stage-C
   image: `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1` is required for the
   ragged-context path that makes `max_num_seqs>1` correct under real arrivals
   (§2.6, `docs/PATCHES.md`).
7. **Author a DSML structural-tag grammar for xgrammar** (or port ds4's
   `try_repair_dsml` as a logits processor) to close the tool-eval-bench gap
   at temp > 0 without forcing every tool-bench run to temp 0 (§3.1).

### Medium term (validate, don't assume)
8. **Re-verify the encoder install** (`encoding_dsv4.py` + the
   `low/high/max` patch) after any Anemll image bump — multi-turn tool
   semantics depend on it (§3.2, §3.5).
9. **A/B `MTP_NUM_TOKENS=5` vs `=6`** at the target concurrency on the standard
   decode-only protocol, watching both tok/s and the KV pool headroom
   (§2.5). Capture size auto-tracks k via the compose command.
10. **Re-run the 0731 sweep** (`scripts/benchmark-0731.py`) after any of the
    above, at the documented 256/2K/8K/32K/128K prompt × C1/C2/C4/C6 grid, and
    compare to the published medians in `results/` before claiming a
    regression or improvement.

## 6. Source index (where each finding above came from)

| Topic | Primary source in this repo |
|---|---|
| Default serve flags | `docker-compose.dspark.yml`, `.env.dspark.example` |
| Anemll vs Stage-C env matrix | `docs/ENVS.md` |
| Concurrency patches (Patch 1/2/2b) | `docs/PATCHES.md` |
| DSML temperature asymmetry | `docs/DSML_SYNTAX_TEMP_ASYMMETRY.md` |
| 0731 checkpoint + serving profile | `docs/DEEPSEEK_V4_FLASH_0731.md` |
| Breakable vs regular graph numbers | `README.md` (PR #14 table) |
| 200K/16 high-concurrency profile | `README.md` ("DSpark Concurrency Profile") |
| Big-prompt lane, KV math | `README.md` (".env.dspark.example" + "How the KV cache works") |
| Published 0731 sweep medians | `results/deepseek-v4-flash-0731-2x-dgx-spark.json` |
| Benchmark methodology (decode-only) | `benchmarks/bench_decode_only.py`, `scripts/benchmark-0731.py` |
| earlyoom host note | `README.md` ("Notes") |

---

*Authored by reading the repository at `HEAD` (commit `b131b2a`,
"Restore --enable-prompt-tokens-details"). No live experiments were run; this
report aggregates the existing measured evidence and open action items
already documented in the repo's own docs and the live compose/.env config.*
