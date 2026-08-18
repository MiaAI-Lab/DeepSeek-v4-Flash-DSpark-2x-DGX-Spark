# vLLM 0.27.0 backports — status and staging notes

Reference for the hotfixes in `patches/` that backport DeepSeek-V4 work from
upstream vLLM v0.27.0 onto the Anemll dspark-vllm-gx10 0.1.1 image
(vLLM 0.25.2.dev0+g752a3a504.d20260714).

| hotfix | upstream PR | effect | state on this fork |
|---|---|---|---|
| `hotfix-dsv4-skip-topk-49486.sh` | #49486 | 3.4% E2E TTFT: skip indexer topk/router when every candidate is selected | **live**; fires only at ≤2048 tokens (512 topk × 4 compress ratio) |
| `hotfix-dsv4-dense-prefill-indexer-48407.sh` | #48407 | skip indexer scoring on short dense prefills | **Stage A only — dormant by design** (see below) |
| `hotfix-dsv4-mtp-buffer-50312.sh` | #50312 | 448 MiB GPU memory saved in the PP buffer (256 MiB/rank here) | **live**; includes two `model_runner.py` None-guards upstream 0.27.0 lacks |
| `hotfix-dsv4-adaptive-topk-50004.sh` | #50004 | 1.0% E2E: adaptive C128A topk width | **live** |
| `hotfix-dsv4-skip-empty-c128-48957.sh` | #48957 | ~2x kernel: skip C128 compressor launch when no request crosses a 128-token boundary | **script verified, not yet applied**; fires only when cudagraph mode ≠ FULL |
| `hotfix-dsv4-flashmla-workspace-50298.sh` | #50298 | 1.88x kernel: workspace reuse for combined topk+SWA indices on the FlashMLA prefill path | **script verified, not yet applied** |
| `hotfix-dsv4-topk-compile-time-consts.py` | #51967 | promotes 5 args of `_compute_global_topk_indices_and_lens_kernel` (`global_topk_indices_stride`, `topk_indices_stride`, `topk`, `block_table_stride`, `block_size`) to `tl.constexpr`; annotation-only, numerically identical | **wired, default OFF** behind `DSPARK_ENABLE_TOPK_CONSTEXPR_HOTFIX=1` (exactly `1`, fail-closed `\|\| exit 1`); **unmeasured on this fork** — upstream measured +0.50% output throughput / −0.98% TPOT at TP=8 |

The `.sh` backports above are idempotent, apply on both nodes (each runs its
own container), and never restart the server themselves. Each supports
`--before` / `--after` (host-side KV-budget + prompt-histogram validation) and
`--status`. The `.py` backport (#51967) is idempotent and applies on both
nodes through the compose entrypoint once its gate is `1`; it supports
`--status` (read-only probe) and honours `VLLM_ROOT`, but has no
`--before`/`--after` host-side capture because it changes no KV budget.

---

## #48407 — why it ships dormant, and what Stage B is

### What upstream does
When the main MLA attention routes a short prefill to dense MHA
(`prefill_max_seq_len <= topk_tokens`) and the batch has zero decode tokens
(decode always consumes top-k), the indexer's compress+score+top-k work is
pure waste. Upstream makes the indexer op check the main MLA metadata
(`use_dense_mha && num_decode_tokens == 0`, not FULL-cudagraph, not stream
capturing) and return the untouched buffer early, after still writing the
K cache.

### Why Stage A is dormant here
This fork has **no dense-MHA route** for sparse-MLA prefills:
`use_dense_mha` / `dense_mha` / `num_mha_tokens` / `force_mqa` do not exist in
`vllm/models/deepseek_v4/` or its MLA backends. The upstream premise ("top-k
is not consumed this step") is therefore false here — every batch's top-k
indices ARE consumed. Shipping the skip live would silently drop valid KV
selection = wrong attention.

So the backport installs the machinery but binds
`DeepseekV4Indexer.indexer_op.dense_mha_metadata_layer_name = ""`
(`models/deepseek_v4/attention.py`). `_resolve_layer_name("")` is falsy, so
the gate can never fire. **Stage A has zero performance effect — that is
intentional.**

### Stage B — two options (do NOT enable from the hotfix script)

1. **Implement the dense-MHA route, then bind.** Port the upstream short-
   prefill dense path (`mla_attention.py` `use_dense_mha` decision +
   `sparse_mla_attention.py` populating it in the prefill metadata), then set
   the binding to the **main MLA cache prefix** — NOT the indexer's own
   `.k_cache.prefix`. Only then does the skip fire, and only when it is
   provably a no-op.
2. **Kill #48407 entirely.** If the dense route is never planned, revert the
   Stage A hunks (the hotfix is idempotent text replacement; reverse the
   old/new strings) to keep `sparse_attn_indexer.py` close to mainline.

Until one of these happens, leave the binding `""`.

---

## 0.27.0 DSV4 performance PRs NOT yet backported

Checked against the running container — these are absent from the fork base:

| upstream PR | effect | backport difficulty |
|---|---|---|
| #49236 | 3.9% E2E TTFT: `DeepseekV4EagerScratchPool` workspace reuse | **needs C++ op rebuild** (new `_out` kernel variant) — image-level change |
| #46789 | sequence parallelism | feature-scale change, not a hotfix |
| #48993 | compact MXFP4 indexer KV cache | unassessed |
| #48047 | sparse-MLA q-head padding removal | requires FlashInfer ≥ 0.6.14 |

---

## Known cosmetic nits (not worth a respin alone)

- The 50312 backport keeps the pre-existing "Only allocated on the last PP
  rank" comment above the now-conditional allocation (upstream deleted it).
- All four scripts print "Nothing was left half-applied" on anchor errors, but
  hunks are written per-file as they match — an earlier hunk would stay
  applied. Idempotency makes this recoverable; moot on the 0.1.1 image, where
  every anchor matches.
- The upstream unit tests for #50004
  (`test_deepseek_v4_c128a_dynamic_topk_packed_buffers`) and #48407
  (`test_mla_short_prefill_indexer.py`) are not ported.
