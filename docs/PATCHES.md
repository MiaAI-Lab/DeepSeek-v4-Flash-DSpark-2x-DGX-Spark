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
