# Patch 1 & Patch 2 — detailed reference

Both patches live in the DSpark vLLM overlay and together make `--max-num-seqs > 1`
**correct** under vLLM-v1 continuous batching. Single-stream and uniform-static
batches keep the original code path (byte-identical).

Files changed:

| file | + | − | role |
|---|---:|---:|---|
| `vllm/v1/spec_decode/dspark_proposer.py` | 158 | 10 | draft loop, slot map, ragged context (Patch 1+2+2b) |
| `vllm/models/deepseek_v4/nvidia/dspark.py` | 110 | 12 | persistent KV store (`store_main_kv`), `prefill_main` |
| `vllm/v1/worker/gpu_model_runner.py` | 10 | 0 | thread `req_ids` into `propose()` |

---

## Patch 1 — request-stable KV slot

### Symptom
At `max_num_seqs>1`, draft acceptance collapsed toward 0 (garbage drafts), even
though nothing crashed — the engine silently degraded to single-stream quality.

### Root cause
DSpark's draft keeps one persistent cross-step tensor per attention module —
`DeepSeekV4DSparkAttention.main_kv_cache`, shape `[max_num_seqs, window, head_dim]`
— a per-row **ring buffer** holding each sequence's sliding-window KV history. It
was read/written by **batch-row position** (`main_kv_cache[:batch_size]`). The
draft proposer carried **no request identity**.

Under vLLM-v1 continuous batching the running set is *condensed* whenever a request
finishes (a later request is moved into the freed row). The model's persistent
`main_kv_cache` row is **not** moved with it, so after a condense a request reads a
ring buffer that belongs to a **different** request → corrupted draft context →
acceptance collapse. (Single-stream never condenses row 0, which is why it worked.)

### Fix
Key the persistent cache by a **stable per-request slot** instead of batch row:

- `dspark_proposer.py`: add `self._req_id_to_slot: dict[str,int]` and
  `self._free_slots`. `_row_to_slot(req_ids)` reclaims slots of finished requests,
  assigns a free slot (lowest-first) to new ones, and returns the slot per row in
  `req_ids` order. A persistent, cudagraph-captured `_draft_slot_index_buffer`
  carries the slots into the graphed draft read path.
- `dspark.py`: `store_main_kv` and `forward_dspark` index the cache by
  `slot_index` (gather `index_select` on read, scatter `index_copy_` on write)
  instead of `[:batch_size]`.
- `gpu_model_runner.py`: pass `req_ids=self.input_batch.req_ids` into `propose()`
  (only for the DSpark proposer).

### Why it's safe
The math is unchanged — it only re-routes which physical row a request uses. When
the computed permutation is identity (a genuine single-request-at-a-time server
always gets slot 0), the code takes the **original in-place write path,
byte-for-byte**. Gating is on the *permutation identity*, not on `batch==1`, so the
"batch condenses to one surviving request holding a non-zero slot" case stays
correct.

---

## Patch 2 — ragged context path

### Symptom
Under real (independent / staggered) arrivals at `max_num_seqs>1`, the server
returned HTTP 500:

```
ValueError: DSpark currently requires uniform flattened per-request inputs;
got 41 rows for batch_size=2.   (dspark_proposer.py: _view_by_request)
```

### Root cause
`prepare_context` reshaped the flat target hidden states into a **rectangular**
`[batch, seq, H]` via `_view_by_request` / `_positions_by_request`, asserting every
request contributed the **same** number of rows. With chunked prefill (required —
disabling it needs `max_num_batched_tokens >= max_model_len`, infeasible at long
context) a single step **mixes prefill and decode** rows, so per-request row counts
differ (e.g. "41 rows for batch_size=2" = one request prefilling alongside one
decoding). Rectangular reshape is impossible → crash. The static benchmark passed
only because all prompts were identical length (uniform).

### Fix
Make the context path **ragged** using `query_start_loc` (per-request segment
offsets) — the same mechanism `_trim_rejected_target_context` already used:

- `dspark_proposer.py` `prepare_context`: detect non-uniform segment lengths
  (`ragged = len(set(seg_lengths)) != 1`). In the ragged branch, skip the
  rectangular view; compute each request's draft anchor with a flat index
  `anchor_idx = starts + clamp(len - rejected - 1, 0, len-1)` and
  `index_select` the per-request last hidden/positions. Pass the flat hidden +
  `query_start_loc` + `slot_index` to `prefill_main`.
- `dspark.py`: `store_main_kv(..., query_start_loc=...)` dispatches to a new
  `_store_main_kv_ragged` that loops requests via `query_start_loc`, truncates each
  segment to the last `window_size` rows, computes `slots = positions % window`,
  applies the rejected-suffix mask, and `index_copy_`s into that request's slot.
  `prefill_main` threads `query_start_loc` through and skips the rectangular view in
  ragged mode.

### Why it's safe
Storage is **position-addressed** (`positions % window`), so it never needed
uniform lengths — only the intermediate rectangular view did. When lengths are
uniform (`query_start_loc is None` / static / single-stream) the original
rectangular fast-path runs unchanged. Ragged/mixed steps run **eager** (mixed steps
are never cudagraph-captured), so dynamic Python loops / variable shapes are safe;
the uniform decode-only graphed path is untouched.

### Scope
Only the `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1` path was made ragged (the path
used in serving). The legacy `_trim_rejected_target_context` path still assumes
uniform. **Run with `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`.**

---

## Patch 2b — ragged detection independent of rejection

### Symptom (found by the GSM8K quality eval)
After Patch 2, a prefill-heavy step with **no rejection** still 500'd:
`ValueError: ... got 166 rows for batch_size=3` at `_view_by_request`. Earlier
staggered tests (uniform-ish prompts) missed it; GSM8K's varied prompt lengths hit
it.

### Root cause
Patch 2 computed `ragged` **only inside** `if gpu_mask and num_rejected_tokens_gpu
is not None`. On steps with no rejection (`num_rejected=None`, e.g. fresh requests
prefilling), detection was skipped and the code fell through to the rectangular
`_view_by_request` → crash. Raggedness depends on `query_start_loc` segment lengths,
**not** on rejection.

### Fix
- Enter the detection/ragged branch whenever `_gpu_rejected_context_mask` is on,
  **regardless of `num_rejected_tokens_gpu`** (which may be `None`).
- In the ragged anchor, default `rejected` to zeros when `num_rejected_tokens_gpu is
  None`. `_store_main_kv_ragged` already handled `None` (no masking).

### Validation
GSM8K N=8 (200 Q) — the load that crashed pre-fix — now completes with **0 errors**,
93.5% accuracy vs 95.0% sequential, **97.5% per-question agreement** (quality-neutral
within batch FP-nondeterminism).

---

## Issue #21 — `encode_arguments_to_dsml` corrupts dict tool arguments

### Symptom
Multi-turn tool calling fails after the first successful tool turn: prior assistant
tool calls are re-encoded into the prompt with a single wrapped `arguments`
parameter instead of the real keys. The model then imitates that corrupt history.

### Root cause
HF checkpoint `encoding/encoding_dsv4.py` always does `json.loads(tool_call["arguments"])`.
When `arguments` is already a `dict` (common in OpenAI-compatible replay),
`json.loads` raises and the `except` path wraps it as `{"arguments": <dict>}`.

Upstream: `deepseek-ai/DeepSeek-V4-Flash-0731` `encoding/encoding_dsv4.py` (same
bug in the Keys abliterated snapshot). Not a vLLM recipe weights bug.

### Fix
Dispatch on type before parsing; keep the wrap only for non-JSON strings.
Applied at container boot after encoder install via
`patches/hotfix-encoding-dsv4-issue21.py` (mounted at
`/opt/hotfix-encoding-dsv4-issue21.py`).

### Test
```bash
python3 scripts/test-encoding-dsv4-issue21.py
```

---

## Issue #22 — `nvfp4_ds_mla` long-context decode regression

### Symptom
With `--kv-cache-dtype nvfp4_ds_mla` (the recipe default), decode throughput
drops to ~1 tok/s at 600K+ context, while `fp8_ds_mla` maintains ~17 tok/s at
the same context length. Short-context throughput (~66 tok/s) is unaffected.

### Root cause
`flashmla_sparse.py` line 880 dispatches `nvfp4_ds_mla` to the slow
`_forward_bf16_kv` kernel path instead of the fast `_forward_fp8_kv` path.
The584-byte KV layout is identical for both dtypes on DSV4; only the kernel
dispatch differs.

```python
# Line 880 in flashmla_sparse.py
use_fp8_cache = self.kv_cache_dtype == "fp8_ds_mla"
# nvfp4_ds_mla → False → slow _forward_bf16_kv (~1 tok/s at 600K)
# fp8_ds_mla   → True  → fast _forward_fp8_kv (~17 tok/s at 600K)
```

### Fix
```python
use_fp8_cache = self.kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")
```

### Hotfix for running containers
```bash
docker exec <container> bash /path/to/hotfix-nvfp4-ds-mla-issue22.sh
# Then restart the vLLM process inside the container.
```

### File changed
| file | change |
|---|---|
| `v1/attention/backends/mla/flashmla_sparse.py` | `use_fp8_cache` check: include `nvfp4_ds_mla` |

---

## Issue #27 — partial-prefill admission cap (running-derived count)

`patches/hotfix-dsv4-issue27-partial-prefill-concurrency.py` breaks the
waiting-admission loop once in-flight partial prefills reach
`DSPARK_MAX_INFLIGHT_PREFILLS` (1–3; fallback
`SchedulerConfig.max_num_partial_prefills`). The count is derived from
`self.running` in exact parity with the stock `_inflight_prefills` set:
requests admitted in earlier steps (list prefix) count while
`num_computed_tokens < num_prompt_tokens` (the set's own membership, discard at
the end of the last-chunk step), so release timing and admission decisions
match the set whenever it is intact; decoders never qualify
(`num_computed_tokens >= num_prompt_tokens`; an unscheduled decoder sits at
`num_tokens + num_output_placeholders - 1`). Requests admitted this step (list
suffix, sized by `scheduled_new_reqs` + `scheduled_resumed_reqs`) count by the
set's add predicate (`num_computed_tokens + num_scheduled_tokens[request_id]
< num_tokens`), so single-chunk same-step bursts are not throttled. The set is
retained only for `_inflight_prefill_reserved_blocks` (issue #154 defense).
Each `Scheduler` construction logs `[issue27-hotfix] in-flight prefill cap=N
env=<raw>` once; a bounded tripwire (≤16 warnings per process) logs
`in-flight prefill undercount` only when the set truly loses a running partial
prefill; verbose `[issue27-adm]` admission lines require the existing
`DSPARK_ISSUE43_SCHED_DIAG=1` knob. Any older `[issue27-hotfix]` application
without the `[issue27-r3]` marker is refused (exit 1); `--status` reports
`APPLIED (r3)` / `APPLIED (r2, stale)` / `APPLIED (pre-r2, stale)` /
`NOT APPLIED`.

---

## Issue #52 — trailing assistant turn closes with EOS (no-op loop)

### Symptom
An agent harness gets stuck emitting empty turns: 1-2 generated tokens with
`finish_reason: "stop"`, no tool call, fragments of hallucinated markup
(`<result observation="no-op"></content>`, stray `</parameter>`). During a live
incident 6 of 37 requests generated ≤10 tokens, all `stop`, zero `length`.

### Root cause
`render_message()` (HF checkpoint `encoding/encoding_dsv4.py`, installed as
`vllm/tokenizers/deepseek_v4_encoding.py`) appends the generation header only
when the trailing message is `user` or `developer`. A request whose `messages`
ends with an **assistant** message is closed with EOS and gets no header, so
the prompt ends on a bare EOS and the model generates from a dead state.
Self-sustaining: the harness records the empty turn, so the next request is
also assistant-final.

### Fix
Widen the separate generation-header transition condition to also match only
the final assistant message
(`patches/hotfix-dsv4-assistant-final-continuation.py`). The checkpoint encoder
has no `add_generation_prompt` input; the patch preserves the closed assistant
turn, then appends a fresh generation header. Reopening the turn with `wo_eos`
was measured worse (1-token empty generation on a complete turn) and is not
used. Runs after the entrypoint copies the encoder into place.

### Extension — trailing `latest_reminder` annotation (Issue #120)
A harness retry can append a trailing `latest_reminder` message after the
re-sent partial assistant turn. The reminder defeats the fix above: stock
closes the assistant turn with EOS, renders the bare reminder after it, and
the prompt still ends with no generation header — a dead state the model
escapes with immediate EOS or hallucinated markup. Verified on the real
checkpoint encoder (`encoding_dsv4.py`, snapshot `9e165c30`): a
reminder tail directly after `user`/`developer` already ends inside the
pending generation slot — the checkpoint emits
`ASSISTANT_SP_TOKEN` + thinking token *before* such a reminder — so those
tails are correct as-is.

The transition condition is therefore widened by exactly one more clause: a
**final** `latest_reminder` whose immediate predecessor is an **assistant**
message gains one fresh generation header appended after the reminder content
(thinking mode `<｜Assistant｜><think>`, chat mode `<｜Assistant｜></think>`).
Reminder tails after user/developer, reminders mid-transcript, task-precedence
rendering, and every assistant-final shape are byte-identical to the
pre-extension hotfix behavior; the post-write self-check additionally fails
closed if a patched encoder double-headers a user→reminder tail.

### Flag (default OFF = stock)
| value | behavior |
|---|---|
| `0` / unset / anything ≠ `1` | stock renderer; patcher mounted/synced to the worker but **never invoked** |
| `1` | patcher runs at container boot, chained with `\|\| exit 1` |

Fail-closed when ON: missing encoder file, missing anchor, or a failed
post-write self-check (patched module must import and render a
trailing-assistant transcript with a generation header and an
assistant-plus-trailing-`latest_reminder` transcript with one fresh header,
without appending a second header to a user→reminder tail) → nonzero exit,
boot aborts; a failed self-check **restores the original file bytes** first. An
already-patched encoder is re-validated (idempotent), never double-patched.

### Evidence status
Render/no-regression evidence is from prior head `f08cd6c`. The measured
positive-path evidence is a causal one-prompt A/B via `/v1/completions`:
trailing turn left open → 183 tokens, coherent continuation; closed with EOS →
400 tokens of raw `<|DSML|tool_calls>` markup emitted as text. **No rescue
claim**: the live no-op-loop defect did not reproduce in that session, so no
measured stuck-harness recovery exists.

A first gated-ON boot on `d4b31daf` failed closed before serving because a
review-requested guard named the nonexistent checkpoint variable
`add_generation_prompt`; the original encoder bytes were restored. Corrected
code commit `0864014` then passed serialized live proof on both ranks. OFF:
effective flag `0`, no patch marker, and the assistant-final render ended on
EOS. ON: effective flag `1`, both ranks logged `patched and verified`, and the
same 98 stock token IDs were preserved with exactly
`<|Assistant|><think>` appended. A live continuation completed with
`alpha beta`. Re-running the patcher on both ranks reported
`already applied and verified`. A deliberate anchor-drift boot exited `1` on
both ranks, entered restart/failure state, and never served the API.

Extension evidence is CPU-only so far: the patched patcher was applied to a
copy of the real checkpoint encoder (snapshot `9e165c30`) and a 16-case
render matrix confirmed the fixed shape gains exactly one fresh header while
every other shape stays byte-identical to the pre-extension hotfix. **No live
serving validation of the reminder-tail rescue exists yet** — run the same
gated-ON/OFF boot proof on both ranks before relying on it in production.

### Test
```bash
python3 scripts/test-assistant-final-continuation.py
```

## Bounded Responses API store

`VLLM_ENABLE_RESPONSES_API_STORE=1` enables the pinned vLLM process-local
response store. The stock implementation never evicts. The launcher therefore
checks and applies `patches/hotfix-dsv4-responses-store.py` on every rank before
engine startup; missing, drifted, invalid, or failed patching aborts the start.
Default `0` does not invoke the patcher and leaves `serving.py` byte-identical.

The target is pinned vLLM `752a3a504`:

```text
/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/responses/serving.py
stock SHA-256  fe3a48ab09c516835ce6dd1471c06cc784ae7504eaa7af7f10574704106830d8
patched SHA-256 1b0033131a34e03a2e129743258f5da81b3e60e979072920153f6d09bf4e5d8f
```

`DSPARK_RESPONSES_STORE_MAX_ENTRIES` is a positive terminal-entry cap
(default `256`). Response, rendered-message, and background-event state is one
eviction bundle. Retrieval and `previous_response_id` continuation refresh LRU
recency. Continuation preprocessing pins its bundle against concurrent
eviction; tracked background producers are retained until their synchronous
completion callback terminalizes status, signals waiting streams, and prunes.
Background event state is published before the lazy reader is returned, and
readers capture that state so later dictionary eviction cannot truncate replay.
Foreground stream messages are retained only after iteration begins and are
removed on error or early close unless a terminal response was stored.

Queued, in-progress, pinned, and tracked-producer entries can temporarily exceed
the terminal cap. The setting bounds entry count, not bytes or concurrent
request admission. Stored state remains memory-only and is lost on any process
restart. Recreate every rank when changing either setting; a Docker restart
preserves the patched writable layer, not stored Responses state.

The patcher accepts only the exact stock or patched full-file hash, compiles the
postimage, preserves file mode, publishes through a same-directory atomic
rename, verifies the result, and rolls back on failed post-publication
verification. `--check`/`--status` are non-mutating.

## Issue #138 — type-less assistant `output_text` history replay

### Scope and source identity

The Anemll 0.1.1 image contains vanilla vLLM commit
`752a3a504485790a2e8491cacbb35c137339ad34` at this serving boundary. The
pinned file is:

```text
/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/responses/protocol.py
Git blob ba8bc5a40f1bcffe8073cfdb4f0a8995da5e02e4
ResponsesRequest.input_item_parsing old-method SHA-256 2412484a81e8679cedf1934287f1b4187a72bf6e8c910c8ecad463b29b79d9d7
expected new-method SHA-256 536f3a305821445328c1f2131b898bef8a8f0c7d278cef4ba29701501eaf3d78
```

`patches/hotfix-vllm-issue138-responses-history.py` locks the complete validator
method plus the surrounding `ResponseInputOutputItem` alias and request `input`
field. A version string alone is not accepted.

### Compatibility transformation

With `DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT=1`, and only then, this
reported replay item:

```json
{"role":"assistant","content":[{"type":"output_text","text":"hello"}]}
```

receives only the missing item-level `"type":"message"`. It is then handed to
the pinned validator's existing assistant-output branch, which supplies only
missing `id`, `status`, and `annotations`. Supplied `id`, `status`, `phase`,
`annotations`, `logprobs`, text, optional model fields, list position, and all
other input items are preserved.

The exception requires a **missing** `type` key, assistant role, a content list
of length exactly one, and one dictionary part with `type=output_text` and a
string `text`. The singleton rule is required: pinned downstream conversion
reads `content[0]`, so newly accepting type-less multipart content could drop
parts. Explicit null/empty/unknown types, multipart or mixed content, missing or
non-string text, refusals, non-assistant roles, malformed id/status/annotations,
tools, reasoning, and other typed items retain stock acceptance or rejection.
Canonical replay remains the complete output item:

```json
{
  "type": "message",
  "id": "msg_...",
  "status": "completed",
  "role": "assistant",
  "content": [{"type":"output_text","text":"hello","annotations":[]}]
}
```

Clients that can emit this canonical form should continue to do so. The hotfix
is compatibility behavior, not a new canonical schema, and it does not alter
Chat Completions, response storage, `previous_response_id`, tools, reasoning,
tokenization, scheduling, or model output.

### Gate, transaction, and removal

| value | behavior |
|---|---|
| unset / `0` / anything other than exact `1` | stock bytes and stock `/v1/responses` validation; patcher not invoked |
| exact `1` | both TP containers atomically apply or idempotently verify the exact postimage before engine exec |

The patcher stages in the target directory, preserves mode, flushes and fsyncs,
uses `os.replace`, re-reads the published bytes, rechecks the exact source
state, and compiles again. Missing targets, source drift, duplicate/mixed/partial
states, invalid UTF-8, compile failures, or publication errors exit nonzero. A
post-publication failure atomically restores and verifies the original exact
bytes and mode. Compose uses `|| exit 1`; neither rank has a masked path to
engine exec. Recreate both containers when changing the flag in either
direction because restart cannot unpatch an existing writable layer.

Remove the flag, mount, patcher, focused tests, verifier, and this compatibility
text together when a future pinned image contains a merged upstream fix that
accepts the exact raw singleton fixture with the same rejection boundary.

CPU contracts:

```bash
python3 scripts/test-issue138-responses-history-hotfix.py
python3 scripts/test-issue138-responses-history-live.py
```

Live stock/enabled commands are in the README. No live A/B result is claimed by
this implementation commit; the mode-strict two-turn run is the release gate.

---

## Codex `agent_message` Responses history compatibility

`patches/hotfix-vllm-codex-agent-message.py` source-locks the complete pinned
`ResponsesRequest.input_item_parsing` method by SHA-256, plus the surrounding
input union guards. It accepts either the stock method or the exact issue #138
postimage and is invoked after issue138, so either opt-in can run alone. The
issue138 patcher also recognizes the exact combined postimage; rerunning the
configured issue138-then-Codex order on one writable layer is byte-idempotent.

With `DSPARK_ENABLE_CODEX_AGENT_MESSAGE_COMPAT=1`, the patch converts only an
`agent_message` with non-empty string `id`, `author`, and `recipient`, exactly
one content part containing string `type=input_text` and `text`, and either no
internal metadata or the evidenced `turn_id` / numeric `create_time`
dictionary. The result contains only `type=message`, `role=assistant`, and the
original one-part content list. Extra or missing keys, empty, multipart, or
malformed content, altered metadata, and all unknown types remain unchanged for
stock Pydantic validation and therefore retain the prior rejection behavior.

The conversion irreversibly drops `id`, `author`, `recipient`, and internal
chat metadata. The model sees the text as assistant conversation history, but
cannot recover routing or attribution. Do not enable this compatibility layer
when those fields must be available to the model for audit or routing logic.

The patcher uses the same atomic publish, mode preservation, fsync,
post-publication verification, atomic rollback, and `--status` behavior as the
issue138 patch, without importing vLLM or GPU dependencies. CPU verification:

```bash
python3 scripts/test-codex-agent-message-compat.py
python3 scripts/test-python-hotfix-failclosed.py
```

Recreate both containers after toggling the flag; a restart does not revert a
patched writable layer.

---

## Issue #141 — sparse-MLA verify-decode chunking workaround (default OFF)

### Evidence and scope

Issue #141 is a **stochastic** TP=2 engine death sampled inside FlashInfer's
SM120 sparse-MLA paged-attention fallback; the pinned revision (`0472b9b3`)
routes calls of at most 64 rows to its standalone DSv4 decode kernel instead.
On one reporting pair, splitting at 64 survived 3,145,728 generated tokens
while a 65-row split failed in the first comparison round; the second failing
pair has **not** repeated that A/B, and the live TP/stream mechanism remains
unknown. This is an opt-in path-avoidance **workaround**, not a root-cause
fix: the fixed 64 is the pinned kernel dispatch boundary, not a deployment
concurrency threshold.

### Target and semantics

`patches/hotfix-dsv4-issue141-sparse-mla-decode-chunk.py` runs before vLLM is
imported and edits only the installed Anemll adapter method:

```
/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py
DeepseekV4FlashInferSM120Attention._forward_decode
```

Calls with at most 64 rows retain one call with the original unsliced objects.
Larger calls run sequential, monotonically increasing `slice` views of at most
64 rows. Exactly six row-coupled arguments are sliced together: `query`,
`sparse_indices`, `out`, `swa_topk_lens`, and the optional
`extra_sparse_indices` / `extra_sparse_topk_lens`. Both KV-cache objects,
workspace, sinks, scale, and layout remain shared by identity. Each inner call
writes a disjoint view of the existing output; no `clone`, `cat`, contiguous
copy, output replacement, retry, stream change, or generic FlashInfer wrapper
is introduced.

The patcher locks the **entire** pinned Anemll `_forward_decode` method,
including its overlay-only `_pad_decode_sparse_indices` call. It also requires
the load-bearing fragments of the pinned FlashInfer `_core.py` and
`_sparse_mla_sm120.py`: the 64-row workspace cutoff, the sliced SM120 call
signature, `_DECODE_MAX_TOKENS = 64`, the DSv4 decode dispatch predicate, and
the custom-op mutation contract that excludes both caches. An exact old method
applies once; an exact new method is recompiled and reverified without a
write. Missing, duplicate, mixed, partial-marker, SM100-like, or drifted
source is rejected.

Before publication the complete updated adapter source is compiled. Publication
uses a mode-preserving same-directory temporary file and `os.replace`; committed
bytes, mode, marker state, and syntax are checked afterward. Any failure after
the rename atomically restores and verifies the original bytes before returning
nonzero. The enabled Compose gate is chained with `|| exit 1`, so incompatible
source never reaches `exec vllm`.

### Enablement and rollback

| value | behavior |
|---|---|
| unset, `0`, or anything other than exact `1` | patcher is not invoked; installed adapter bytes remain stock |
| exact `1` | validate pinned sources, atomically apply or reverify, and fail boot closed on any error |

Enable in the authoritative `.env.dspark`:

```bash
DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK=1
./stop-deepseek-v4-flash-dspark.sh
./start-deepseek-v4-flash-dspark.sh
```

The launcher reports the resolved 0/1 state and syncs the selected patch source
to the worker; both rank entrypoints receive the same normalized flag and run
the same source preflight. Check both rank logs for `applied and verified` (or
`already applied and verified`) and then make a real generation request with a
terminal `finish_reason`. `/health` alone is insufficient: it remained 200 in
reported stalled/silently truncated incidents.

Rollback by setting `0` (or removing the variable), then run the same paired
stop/start flow. A process restart or `docker compose restart` is **not** a
rollback: the modified site-packages file remains in that container's writable
layer. Recreating both containers restores immutable image bytes.

### Validation status and remaining gates

The committed CPU suite freezes the pinned method and guard digests, exercises
exact apply/idempotence/drift/atomic rollback, and executes the injected block
against shared-backing fake tensors across the 1–576 boundary row matrix in
SWA-only and compressed-cache shapes. It also checks exact-1 fail-closed
Compose ordering and worker wiring.

The initial live TP=2 campaign at pre-trim head `890d9de` covered
pinned-image apply/boot on both ranks, short concurrency-16 generation, and
post-run smoke and restore. Before relying on the workaround, close the
outstanding gates: disposable pinned-image
extraction plus SM121a numerical/CUDA-graph tests, a two-rank OFF/ON/drift boot
proof, and repeated stochastic generation soaks verifying terminal
`finish_reason`, rank stability, throughput, and peak scratch memory. The
second failing pair still needs the 64/65 comparison; one clean burst cannot
close a stochastic issue.

### Test

```bash
python3 scripts/test-issue141-sparse-mla-decode-chunk.py
```

---

## Issues #136 + #210 — XGrammar termination and post-reasoning FSM chain

### Symptom and source fix

With DSpark MTP, `structural_tag` constraints, async scheduling, and TP=2,
one accepted draft batch can contain a terminating token followed by speculative
tokens. Pinned vLLM `752a3a504` passes those trailing tokens to an already
terminated XGrammar matcher. The characteristic warning is `The matcher has
terminated ... but is trying to accept new token`; the affected request can
then stop making progress and eventually end in the generic 1,800-second
`sample_tokens` RPC timeout. Raising or lowering that deadline does not repair
the grammar state machine.

`patches/hotfix-vllm-issue136-xgrammar-termination.py` backports only the three
`XgrammarGrammar` method hunks from upstream vLLM PR
[#52805](https://github.com/vllm-project/vllm/pull/52805), merge
`12f64b39d29282437e35be9aa5db432fb2a1a6e6`:

- `accept_tokens` stops at the terminating token, counts it, caches termination,
  ignores the rest of that batch, and treats a later acceptance as a successful
  no-op;
- `validate_tokens` stops at termination, rolls back only the accepted prefix,
  and returns no speculative drafts after cached termination;
- `reset` clears the matcher, counter, and cached termination flag.

The same flag then applies the single-hunk vLLM PR
[#53046](https://github.com/vllm-project/vllm/pull/53046) (issue #210) to
`v1/structured_output/__init__.py`: in `grammar_bitmask`'s speculative window,
a draft after the reasoning-end marker is checked with `validate_tokens`
before `accept_tokens`, so a grammar-invalid draft that predates the bitmask
is skipped instead of tripping the spurious `Failed to advance FSM` error
path. No output corruption was demonstrated for the prior code, but the FSM
state path is correctness-sensitive; the upstream fix removes the desync risk
class. The reporter measured their best tool-evaluation result on this recipe
with both backports active.

One flag, one transaction: both candidates are built and compiled before
either file is written; publication is per-file atomic in chain order
(backend first), and a second-file failure rolls the first file back to its
exact original bytes (refusing to clobber a concurrent change). A pre-chain
#136-only state (backend patched, manager stock) is completed by publishing
the manager file only; the inverse mix is refused as invalid.

Relationship to the #44993 grammar-advance train: both patchers now touch
`v1/structured_output/__init__.py`, on non-overlapping regions (proven
byte-exact in both application orders by the test suite). Compose order is
fixed — the default train runs first — so the chain normally sees the
post-#44993 file; with `DSPARK_SKIP_HOTFIX=1` it sees the pristine pinned
image. The patcher pins BOTH stock identities (post-#44993 and pristine) with
their respective post-images; neither is a prerequisite of the other.

### Compatibility and exact identities

Enabled mode accepts only all of the following:

- image `ghcr.io/anemll/dspark-vllm-gx10:0.1.1@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8`
  — the registry manifest digest (the repo's pin). It resolves directly (no
  index layer) to config/image ID `sha256:3430d6614a8e2925f34d059af6caf05aff42387326db4d05639a60f10f2654d8`
  on a pulled host (`docker image inspect .Id`); both names refer to the same
  image, and the pristine manager fixture was extracted from a fresh
  `docker create` of it.
- installed metadata `vllm==0.25.2.dev0+g752a3a504.d20260714` and
  `xgrammar==0.2.3`;
- stock target SHA-256
  `231f6b9d7dab5e8d68aba486fa5912db99f8bdd3f9d8842ee3e0bb12bdb7cb67`
  (12,699 bytes), or exact post-image SHA-256
  `6c7e23c0ae5c6836d0d56862c6e825c49727fa2409b881b44ea2526f1fd03f04`
  (12,983 bytes);
- manager target `v1/structured_output/__init__.py`, two legitimate stock
  identities with their post-images: post-#44993
  `e782163b8a83d58e61a655df042d3126cde8c913a2eeaf9d4a061148cd8e5c77`
  (21,979 bytes) →
  `3dff0e1e35f04f35e8c50c17d9efa65cd5fc8db1f25d4eb5d536b6e61114a616`
  (22,271 bytes), or pristine
  `fd23813a4e0d8cdc93fa1e6687e5a4f4e514b0ae37dec707d50d840771390818`
  (22,076 bytes) →
  `53186ccf86e3d620a9aa91af8c541516f0b45a3f640d937607a252bc42f376e6`
  (22,368 bytes).

Anything else—including another vLLM/xgrammar version, a symlink, partial
application, or drift before/inside/after the method region—is incompatible.
No source is written. An exact stock file is completely constructed and
compiled in memory, staged beside the target, and published with one atomic
rename. Mode/owner/group are retained and the file plus directory are fsynced.
Any post-publication read/hash/metadata/compile failure atomically restores and
verifies the original bytes. An exact post-image is reverified without a write.

### Flag and status

| value | behavior |
|---|---|
| `0` / unset / anything other than exact `1` | patcher is not invoked; installed vLLM bytes remain stock |
| `1` | worker and head compatibility checks must both pass before either rank starts; each container then applies/reverifies fail-closed before `exec vllm` |

The supported launcher syncs the patcher to the worker's canonical `patches/`
path. Direct Compose starts retain the per-container fail-before-exec gate but
do not provide the launcher's cluster-wide two-rank preflight.

Nonmutating check (stock or patched exits 0; incompatible exits 2):

```bash
docker compose --env-file .env.dspark -f docker-compose.dspark.yml run \
  --rm --no-deps --entrypoint python3 vllm-dspark \
  /opt/hotfix-vllm-issue136-xgrammar-termination.py --check
```

Running-container status (`patched` exits 0, `stock` or `partial-invalid` exits 1,
`incompatible` exits 2):

```bash
docker compose --env-file .env.dspark -f docker-compose.dspark.yml exec \
  vllm-dspark python3 \
  /opt/hotfix-vllm-issue136-xgrammar-termination.py --status
```

### CPU and live acceptance

The hermetic fixture/transaction/startup suite requires no vLLM, xgrammar,
torch, Docker, GPU, or network:

```bash
python3 scripts/test-issue136-xgrammar-termination.py
```

Issue closure still requires a maintenance-window canary on the exact two-node
async/TP=2/MTP5 lane. After enabling only
`DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX=1` and recreating both ranks, run:

```bash
VLLM_API_KEY='...' python3 scripts/verify-issue136-xgrammar-live.py \
  --base-url http://127.0.0.1:8888/v1 \
  --model deepseek-v4-flash-dspark \
  --output /tmp/issue136-xgrammar-live.json
```

The verifier uses a 120-second deadline per request and records no credentials,
headers, request bodies, or response bodies. It runs 20 sequential strict tool
requests; 100 at concurrency four alternating required/named tool choice; five
bounded strict-JSON `ignore_eos` diagnostics; ten ordinary-tool controls; and
ten plain-chat controls. Require all 145 cases plus pre/post `/health` to pass.
From the same saved UTC start time, both rank logs must contain zero matcher
termination warnings, `Failed to advance FSM`/`grammar rejected tokens`, shared
memory broadcast-block waits, `sample_tokens` timeouts, and
`EngineDeadError`/`EngineCore encountered an issue`; restart counts must remain
unchanged and no request may remain running with frozen generation progress.
The JSON request report alone is not the complete live gate.

### Rollback and upgrade

Changing the flag requires a real two-node stop/removal and start. To roll back,
set the flag to `0`, require `stop-deepseek-v4-flash-dspark.sh` to remove both
service containers, then start normally. `docker restart` or restarting the
process reuses the patched writable layer and is **not** rollback. On an image
upgrade, leave the flag off: enabled mode intentionally rejects even a newer
upstream file. Once the image incorporates PR #52805, remove this patcher,
flag, fixtures/tests, sync/preflight, and documentation together.

Evidence currently checked in is CPU/source-exact only. Do not claim the live
incident closed until the two-rank canary and log/health gate above pass.

---

## Issue #191 — fail-closed named/required `tool_choice` contract (default OFF)

**Symptom.** With Vision-Exp (`n_predict=3`, `MTP_NUM_TOKENS=6`), async
scheduling and TP=2, the shipped 145-case `scripts/verify-issue136-xgrammar-live.py`
gate scored `142/145` twice at concurrency 4: HTTP 200 responses with zero
`tool_calls` (named and `required` lanes) or arguments that violate the
`strict` schema. Failing labels changed between runs and every failed case
replayed `18/18` clean, so this is a concurrency-dependent engine race, not a
prompt problem. See MIA issue #191.

**Where the contract leaks.** `_create_chat_completion` returns
`chat_completion_full_generator(...)` directly; that path serialises
`tool_calls or []` for named/required choices with no terminal check.

**What the engine actually does (measured 2026-09-03, async on and off).**
The DeepSeek-V4 named/required structural tag is a strict *sequence*
(`\n\n<｜DSML｜tool_calls>\n` … `</｜DSML｜tool_calls>`, `deepseek_xml` schema
style) and XGrammar 0.2.3 enforces `required`, property order and
`additionalProperties` on it (verified on CPU with the model tokenizer). The
scheduler never logged `Unexpected: grammar rejected tokens`; every
`Failed to advance FSM` line came from the *tolerated* branch of
`StructuredOutputManager.grammar_bitmask` — drafts proposed after a mid-window
`</think>` are checked against the fresh grammar and rejected (they predate the
mask), which is expected and harmless but logged at ERROR. With
`chat_template_kwargs.thinking=false` the same 145-case gate produces zero such
lines. The real residual failure is reasoning length: see hunk 3 above. The
`-1` placeholder / single-slot draft hand-off of async scheduling (vLLM #49694 /
#54437) remains a code-level fail-open hazard, but it was not the observed cause
(same violation rate with `DSPARK_ASYNC_SCHEDULING=0`).

**What the patcher does.** `patches/hotfix-vllm-issue191-toolcall-failclosed.py`
(source-exact, post-issue55 identity `08ddb5f3…`, patched identity `873ac9c6…`)
adds two hunks to `entrypoints/openai/chat_completion/serving.py`:

1. helper block after `_dsml_issue55_json_ok`: `_issue191_tool_contract_violation(request, response)`
   returns `None` or a short reason (`tool-call-cardinality:N`, `tool-call-name`,
   `tool-arguments-json`, `tool-arguments-type`, `tool-arguments-schema:<path>:<keyword>`,
   `tool-call-truncated`, `no-choices`). Schema checks use the image's
   `jsonschema` (4.26.0) and fall back to a required/type/additionalProperties
   checker; a malformed schema never fails the request.
2. tail of `_create_chat_completion`: on a violation log one WARNING
   `[issue191-toolcall] contract violation request=… attempt=… mode=… reason=…`,
   then (mode `failclosed`) regenerate the same engine input with a fresh engine
   request id (`<id>-issue191r<n>`, client-visible id unchanged) up to
   `DSPARK_ISSUE191_TOOLCALL_RETRIES` times and finally answer HTTP 500; mode
   `log` returns the response unchanged. Beam search is never retried; a
   `length` finish counts as a violation only when it left no tool call. Streaming is out of scope (chunks are already sent).
3. **thinking-off fallback on the last retry** (`DSPARK_ISSUE191_TOOLCALL_THINKOFF_FALLBACK`,
   default `1`). The 2026-09-03 measurement found the residual violations are
   not grammar desync at all: with `thinking=true, reasoning_effort=low` a
   fraction of strict requests reason for 300–500 tokens (non-deterministic
   across batches even at temperature 0), so `max_tokens=512` cuts the reply
   before or inside the DSML call (`finish_reason=length`, zero or a salvaged
   partial call). Replaying the identical engine input mostly replays the
   problem. The last `failclosed` attempt therefore swaps the prompt's trailing
   `<think>` marker (id taken from the request's reasoning parser) for
   `</think>` — byte-identical to rendering the chat with `thinking=false` —
   passes `reasoning_ended=True` and thinking-off `chat_template_kwargs` to the
   engine, and parses the reply with a thinking-off parser. The grammar then
   constrains the reply from its first token and the call fits the client's
   budget. The fallback only fires when the prompt ends with `<think>` (otherwise
   the retry is identical); the log line `[issue191-toolcall] regenerating … fallback=thinkoff`
   marks it. `0` keeps every retry identical.

**Gates.** Default `0` changes no bytes. `1` requires the exact pinned identity
on both ranks (`--check` preflight worker then head, apply at container start,
post-apply digest verification, atomic same-directory replace). CPU suite:
`python3 scripts/test-issue191-toolcall-failclosed.py`. Live acceptance: the
145-case gate must reach `145/145` with the hotfix on, and the WARNING count
in `docker logs` is the measured raw violation rate.

**Companion knob.** `DSPARK_ASYNC_SCHEDULING=0` removes `--async-scheduling` on
both ranks so the grammar bitmask rows are built from real draft tokens; it is
the single-variable A/B for the engine-side trigger and costs decode throughput.
## DSpark block-k unlock — `num_speculative_tokens` follows `dspark_block_size` (default OFF)

**Symptom.** Vision-Exp ships `num_nextn_predict_layers=3` and
`dspark_block_size=5`. The pinned `SpeculativeConfig.__post_init__` maps
`num_nextn_predict_layers` to `n_predict` and rejects any
`num_speculative_tokens > n_predict` that is not a multiple of it ("Ensure
divisibility for MTP module reuse"), so the recipe runs k=6 (0731 has one stage
and boots k=5). The launcher mirrors that rule. Measured against 0731 on the
same `bench_quick` (2×GB10, TP=2): prefill identical, single-stream decode
−15–20 % (greedy) and −20–30 % (temp 0.6); per-position draft acceptance
0.89/0.73/0.49/0.34/0.23/0.15 at k=6 versus 0731's 0.93/0.75/0.66/0.58/0.47 at k=5.

**Why the rule does not apply.** The DSpark drafter (`models/deepseek_v4/nvidia/dspark.py`,
`v1/worker/gpu/spec_decode/dspark/speculator.py`) *stacks* the `mtp.{0,1,2}`
stages into one non-causal backbone and predicts every position of the block in
one parallel pass (anchor + k−1 noise queries), then samples left-to-right with
the Markov head; no stage is re-run per step. The checkpoint's own
`inference/model.py::DSparkBlock` drafts exactly `dspark_block_size` tokens, so
k=5 is the trained shape.

**What the patcher does.** `patches/hotfix-vllm-dspark-block-k.py`
(source-exact, stock identity `3f1abd1c…`, patched identity `7fffe035…`) adds
`self.method != "dspark"` to that single condition in `config/speculative.py`
and nothing else. `DSPARK_ENABLE_DSPARK_BLOCK_K=1` gates it (mount, `--check`
preflight worker then head, apply at container start, atomic replace) and
relaxes the launcher's `MTP_NUM_TOKENS` rule to `>= 1`; the CPU suite is
`python3 scripts/test-dspark-block-k.py`. Pair it with `MTP_NUM_TOKENS=5`.
Capture size follows (`MAX_NUM_SEQS * (k + 1)` rounded up to 8 → 40 at 6×5).

**Measured (2026-09-03, Vision-Exp, k=5 vs k=6, same machine):** greedy 8K
decode 56.6 vs 55.3 tok/s, greedy 32K 56.6 vs 53.6, temp 0.6 49.9–56.9 vs
48.6–53.1, single-stream mini-bench 56–64 vs 51–55, concurrency-4 aggregate
unchanged (110 vs 106–111), TTFT unchanged. Per-position acceptance is the same
at either k (0.88/0.74/0.53/0.36/0.24), so the gain is the cheaper step.

---

## RoPE SWA fix — sparse-SWA layers use plain RoPE, not YaRN (default OFF)

**Symptom.** `models/deepseek_v4/common/rope.py::build_deepseek_v4_rope`
promotes the checkpoint's rope type to `deepseek_yarn` whenever it is not
`"default"` — for every layer. The served Vision-Exp abliterated checkpoint
ships a flat `rope_scaling = {type: yarn, factor: 16,
original_max_position_embeddings: 65536}` with `sliding_window=128`, so its
sparse-SWA layers — 0 and 1 (`compress_ratios[i]=0` → `compress_ratio=1`) plus
the three DSpark drafter layers past `num_hidden_layers` — run YaRN factor=16
over a 128-token window. Per the DeepSeek-V4 reference (`inference/model.py`
L481-485) and transformers#45892, YaRN belongs only to compressor (CSA/HCA)
layers; sliding-window layers must use plain RoPE.

**What the patcher does.** `patches/hotfix-vllm-rope-swa-fix.py` ports merged
upstream [vllm#54815](https://github.com/vllm-project/vllm/pull/54815)
source-exact (stock identity `0074271a…` → patched `6452ce2e…`; the patched
bytes minus the one mark comment equal the upstream post-image byte-for-byte):
each call works on a per-layer dict copy (nested `{"main","compress"}`
checkpoints route by layer type), the YaRN promotion additionally requires
`compress_ratio > 1`, and every other layer takes `deepseek_yarn` with
`factor=1.0` over `max_position_embeddings` — identity scaling, i.e. plain
RoPE on the same kernel path. Compressor layers resolve byte-identical
parameters to stock (`deepseek_yarn`, factor=16, theta=160000), and the
shared `config.rope_parameters` dict is no longer mutated across layers.
`DSPARK_ENABLE_ROPE_SWA_FIX=1` gates it (mount, `--check` preflight worker
then head, apply at container start, atomic replace); the CPU suite is
`python3 scripts/test-rope-swa-fix.py` (pins, idempotency, fail-closed CLI,
and stubbed-`get_rope` routing for compress_ratio 1/4/128 with the served
checkpoint's real rope values).

**Live gate before defaulting on.** Positions ≤128 sit where the YaRN ramp is
near-identity, so short-context output should be indistinguishable; the
abliterated checkpoint may nevertheless have adapted to the served embedding.
Run the 128K+ long-context quality A/B vs control (same seeds, gate26
harness) before flipping the default.
## DSpark draft SWA prefix fix — prefix-cache hits recompute the last draft window (default OFF)

**Symptom.** With `--enable-prefix-caching`, re-sending an identical prompt
(retries, cached tool-call prefixes, agent loops) returns a degenerate,
truncated response (e.g. a 5-token `["json` + stop) instead of the full
output; deterministic at temperature 0. Upstream report and fix:
Anemll/dspark-vllm-gx10#2 (`4afc5e7eeb`).

**Root cause.** The DSpark draft model attends over a sliding window of
`sliding_window` (128) tokens populated from the target's hidden states via
`precompute_and_store_context_kv`. On a prefix-cache hit the target skips
recomputing the cached prefix, so only the non-cached suffix reaches the
draft: its window cache is missing the prefix, the draft degenerates, and the
verifier accepts the truncated output.

**What the patcher does.** `patches/hotfix-vllm-dspark-swa-prefix.py` ports
the upstream overlay verbatim (plus one `# [dspark-swa-prefix]` mark line per
file): `v1/core/kv_cache_manager.py` (stock identity `be9c5091…`, patched
`09f0e990…`, whole-file pinned — no other recipe hotfix touches it) gains a
`dspark_window_size` parameter capping `max_cache_hit_length` to
`num_tokens - 1 - dspark_window_size` in `get_computed_blocks`;
`v1/core/sched/scheduler.py` (co-owned at boot by grammar-advance,
empty-encoder-output and issue #27, so held to source-exact regions that must
each occur exactly once; pure stock `e25d4c9a…` -> patched `69fc8118…` proven
against fixtures) reads the draft's `hf_config.sliding_window` under
`use_dspark()` and passes it to the `KVCacheManager`. Without DSpark the
window stays `None` and cache-hit arithmetic is stock. Cost when active: a
128-token recompute per prefix-cache hit. `DSPARK_ENABLE_DSPARK_SWA_PREFIX=1`
gates it (mount, `--check` preflight worker then head, apply at container
start, both targets preflighted before either atomic replace); the CPU suite
is `python3 scripts/test-dspark-swa-prefix.py`.

**Upstream validation (2×GB10, TP=2, 0731, K6 + probabilistic):** repeated
json60 prompt 12.1 tok/s broken -> 86.5 fixed; count300/mult12/bst/story and
8K/32K/100K prefill unchanged within noise. Local live gate still to run on
the Vision-Exp abliterated lane: repeated-prompt output-quality A/B.

---

## DSML recovery — malformed-wrapper DeepSeek V4 tool calls recover instead of leaking (default OFF)

**Symptom.** DeepSeek V4 intermittently emits an otherwise complete DSML
`<invoke name="...">` block while the outer `tool_calls` opener is missing or
malformed — one observed DeepSeek-V4-Flash-0731 variant emits `toolcalls`
(upstream vllm#51914). The pinned parser only enters the tool-call state
machine on the exact outer opener, so the whole invoke leaks verbatim into
user-visible content (or stays in reasoning) and the structured tool call is
lost; agent traffic sees DSML markup as prose instead of a tool call.

**What the patcher does.** `patches/hotfix-vllm-dsml-recovery.py` ports open
upstream [vllm#52645](https://github.com/vllm-project/vllm/pull/52645) (head
`3df9776b0d`, the current-main DeepSeek V4 extraction of the #49117
orphan-invoke recovery direction) onto the pinned parser engine, adapted to
the pinned engine's pre-`token_count` API. Six files, all sole-owned by this
hotfix and pinned by whole-file stock+patched identity:

- `parser/engine/parser_engine_config.py` (stock `0854bd50…` → `76ed8f12…`):
  `Transition` gains opt-in `provisional_tool_call` /
  `commit_provisional_tool_call` markers; `ParserState` gains
  `FOREIGN_BLOCK` / `FOREIGN_REASONING_BLOCK`.
- `parser/engine/streaming_parser_engine.py` (`4ac9135e…` → `cd7d8778…`): a
  provisional transition buffers its semantic events and raw text; the
  completed name is validated through a parser-owned callback; only the
  configured `INVOKE_END` transition commits (returning to CONTENT and
  absorbing one optional outer closer); every other exit — truncation, an
  outer `TOOL_END` without `INVOKE_END`, a rejected name, `finish()` — puts
  the raw text back in its original content or reasoning state. Parser-level
  drop tokens (EOS) never enter the buffers; name buffering aborts past 256
  chars or a newline so quoted markers cannot stall a response.
- `parser/deepseek_v4.py` (`97d7cd3c…` → `2cc89a1b…`): provisional
  transitions for a bare `INVOKE_PREFIX` from CONTENT/REASONING; V3.2
  `function_calls` wrappers become verbatim passthrough states (their inner
  invokes are never recovered); the recovery validator accepts only names
  declared by the live request and nothing under `tool_choice="none"`.
- `parser/engine/adapters.py` (`dc1c1317…` → `9d743734…`),
  `parser/abstract_parser.py` (`fd4eb7a6…` → `e11c1b78…`),
  `parser/engine/parser_engine.py` (`886bf629…` → `f8f403ad…`): the request's
  tools and `tool_choice` are mirrored into the reasoning-side engine before
  recovery validation (non-streaming and per-delta), and a rolled-back
  candidate parked in deferred reasoning is flushed at stream end.

Misspelled wrappers are deliberately not normalized: recovery anchors on the
inner invoke structure, so missing and corrupted openers share one
conservative path and unrecognized wrapper text is preserved as content.
`DSPARK_ENABLE_DSML_RECOVERY=1` gates it (mount, `--check` preflight worker
then head, apply at container start; all six targets preflight before any
write, one atomic replace per file, files already written roll back to stock
if a later file fails). The CPU suite is `python3 scripts/test-dsml-recovery.py`:
fixture/transform pins, patcher fail-closed/idempotency/rollback, the
upstream #52645 regression matrix (16 engine + 5 serving-style delegating
scenarios) replayed against the pinned fixtures, and a 16-case stock/patched
parity matrix proving normal wrapped DSML, reasoning, streaming, and
`tool_choice` handling are byte-identical in behavior.

**Live gate before defaulting on.** Recovery only fires on traffic the stock
parser already fails to execute, but the lane contract is agent tool-call
acceptance parity vs the 42.3% C1 baseline on live agent traffic (existing
parser suite + issue #191 tool-call contract stay green in CI).

---

## Issue #117 — bounded SHM dispatch-ring reader recovery

### Scope and upstream fix

The mid-serve failure addressed here is a local `MessageQueue` reader parked in
`SpinCondition.wait()` after a PUB/SUB notification is missed. The dispatch is
already authoritative in shared memory, but an indefinite socket poll prevents
the reader from checking that slot again.

`patches/hotfix-vllm-issue117-shm-ring-buffer.py` backports both changes from
upstream vLLM PR #45224, merge
`10c75477b07c2f1a361f54b7357af1019bba5fd8`:

- `ReadTimeoutWithWarnings.timeout_ms()` is capped by the upstream
  `SHM_READER_RECHECK_INTERVAL_MS = 5000`, including indefinite/no-warning
  reads, so the authoritative written flag is checked again;
- `acquire_read()` releases the reader slot, advances the ring index, and
  records the read in `finally` even when the consumer raises.

This is not an orphaned-SHM lifecycle fix. It does not enumerate, unlink, or
reuse `/dev/shm/psm_*` objects and does not claim to fix the separate
stop/start API-readiness failure associated with ownerless segments.

### Compatibility and publication

The patcher accepts only
`vllm==0.25.2.dev0+g752a3a504.d20260714` and one of four complete-file
identities: exact issue-117 stock or patched bytes, each with either the exact
stock `busy_loop_s = 1` line or the independent issue #79
`busy_loop_s = 0.002` overlay. It never changes that issue #79 line.

Marker-only, partial, mixed, duplicated, independently drifted, symlinked, and
unsupported-version states are incompatible. Compatibility checks are
unconditional under `PYTHONOPTIMIZE=1`. A stock post-image is built and
compiled in memory; rollback and candidate images are staged beside the
target with retained mode/owner/group and file `fsync`, then one atomic rename
publishes the candidate. The directory and published bytes are fsynced and
re-read. Any post-publication failure atomically restores and verifies the
original bytes and metadata. An exact patched image is verified without a
write.

### Startup and rollback

The launcher syncs the dedicated patcher, checks worker then head without
mutation, and only then starts either service. Each container applies the
patcher and requires a successful `--status` before `exec vllm`.

The backport is default-on. Set
`DSPARK_SKIP_ISSUE117_RECHECK_HOTFIX=1`, then stop/remove and recreate both
service containers, to restore image stock for issue #117 without changing the
separate issue #79 spin-wait setting. A process or Docker restart reuses the
writable layer and is not rollback.

The hermetic behavior/source/transaction/startup suite is:

```bash
python3 scripts/test-issue117-shm-ring-buffer.py
```

---

## Item 6 — sequence-parallel Lightning indexer for long prefills (default OFF)

### Why

`DeepseekV4Indexer` is replicated across TP ranks (`wq_b` / `weights_proj` are
`ReplicatedLinear`), so every rank scores all 64 index heads against the whole
compressed key range of every prefill query. At long context that O(queries ×
keys) score is the dominant prefill cost (900K prefill ≈ 875 tok/s vs ≈ 2,500
tok/s at 2K) and it is computed once per rank. Report:
`docs/CLAUDE/fable5-1-report.md` §3.6.

### What the patch does

`patches/hotfix-dsv4-sp-indexer-prefill.py` edits
`vllm/model_executor/layers/sparse_attn_indexer.py`. In the prefill loop, for a
chunk with at least `DSPARK_SP_INDEXER_MIN_KEYS` compressed keys (default 8192
= 32K tokens at compress ratio 4), TP rank `r`:

1. splits every request's compressed key range into `tp` contiguous,
   page-aligned slices (`_sp_indexer_split`) and gathers only its own slice
   from the paged indexer cache (shifted block table, rank-local
   `cu_seq_lens`);
2. maps each query's global `[ks, ke)` onto its local rows
   (`_sp_indexer_local_bounds`, causal bound clamped into the slice);
3. runs the stock `fp8_fp4_mqa_logits` + `top_k_per_row_prefill` on the local
   slice (half the logits, half the K gather at TP=2);
4. packs `(score, global_id)` for its local top-k, all-gathers the candidates
   across the TP group (`tp × index_topk × 8 B` per query) and runs vLLM's
   DCP `stable_topk_from_gathered_candidates_cutedsl` into the shared
   `topk_indices_buffer`.

Exactness follows the DCP argument: a token in the global top-k is in its
owning rank's local top-k, so merging local top-k sets equals top-k over the
full row. Decode, chunks below the threshold, DCP>1, TP=1 and XPU keep the
stock replicated path byte-for-byte. Control flow is symmetric across ranks
(the decision uses CPU-side chunk metadata that is identical on all ranks), so
the collective cannot desynchronize.

### Flag (default OFF = stock)

| value | behavior |
|---|---|
| `0` / unset / anything ≠ `1` | patcher not invoked; stock bytes |
| `1` | apply at boot on both ranks, `|| exit 1` (fail-closed); `DSPARK_SP_INDEXER_MIN_KEYS` tunes the threshold at runtime |

Recreate both containers when flipping (a restart keeps the patched layer).

### Validation

* CPU: `python3 tests/test_sp_indexer_prefill.py` — applies to the real image
  file, idempotent, compiles; split/bounds math vs brute force for many request
  shapes at TP 2/3/4 (page alignment, disjoint cover, per-query local range).
* GPU (inside the image, one GPU, no process group): `scripts/test-sp-indexer-gpu.py`
  runs the real kernels for emulated TP=2 and TP=3 and compares the merged top-k
  against the stock full-range path (valid-candidate counts and score multisets).
  Needs the DeepGEMM SM121 header alias below when the JIT cache is cold.
* Live A/B still required: `scripts/bench-ttft.py` at 32K–900K prompts with
  `DSPARK_ENABLE_SP_INDEXER=0/1`, plus `scripts/ruler-lite.py` for quality.

---

## DeepGEMM SM121 indexer-logits header alias (default OFF)

The image's vendored DeepGEMM emits `sm121_fp8_mqa_logits<...>` and includes
`impls/sm121_fp8_mqa_logits.cuh` on GB10 (CC 12.1) but ships only `sm120_*`
headers. Production works because `VLLM_CACHE_ROOT/deep_gemm/cache/` already
holds cubins built from `sm120_fp8_mqa_logits.cuh` (Jul 16 / Jul 27). A cache
miss — fresh volume, new cache root, new index head count, the fp4 indexer path
— fails at first use with `Failed to open .../sm121_fp8_mqa_logits.cuh`.

`patches/hotfix-deepgemm-sm121-mqa-header-alias.sh` (gate
`DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS=1`) writes four alias headers
(`#include <deep_gemm/impls/sm120_X.cuh>` + `#define sm121_X sm120_X`) for the
fp8/fp4 × contiguous/paged mqa-logits kernels. Idempotent; `--status` reports.
Details: `docs/CLAUDE/item8-fp4-kv-design.md` §5.
