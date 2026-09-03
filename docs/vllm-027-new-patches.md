# vLLM 0.27.0 backports — status and staging notes

Reference for the hotfixes in `patches/` that backport DeepSeek-V4 work from
upstream vLLM v0.27.0 onto the Anemll dspark-vllm-gx10 0.1.1 image
(vLLM 0.25.2.dev0+g752a3a504.d20260714).

| hotfix | upstream PR | effect | state on this fork |
|---|---|---|---|
| `hotfix-dsv4-skip-topk-49486.sh` | #49486 + #52492 | 3.4% E2E TTFT: skip indexer topk/router when every candidate is selected; the #52492 guard keeps the shortcut out of CUDA graph capture (a captured shortcut replays against longer cached prefixes and selects candidates 0..topk-1 unscored) | **live**; fires only at ≤2048 tokens (512 topk × 4 compress ratio), eager steps only |
| `hotfix-dsv4-dense-prefill-indexer-48407.sh` | #48407 | skip indexer scoring on short dense prefills | **Stage A only — dormant by design** (see below) |
| `hotfix-dsv4-mtp-buffer-50312.sh` | #50312 | 448 MiB GPU memory saved in the PP buffer (256 MiB/rank here) | **live**; includes two `model_runner.py` None-guards upstream 0.27.0 lacks |
| ~~`hotfix-dsv4-adaptive-topk-50004.sh`~~ | #50004 | 1.0% E2E: adaptive C128A topk width | **removed** — upstream vLLM reverted #50004 in [#51318](https://github.com/vllm-project/vllm/pull/51318): the builder writes packed rows at the live batch's stride while FULL-graph consumers keep the capture-time stride, so rows ≥ 1 read stale slot ids (intermittent wrong-context attention / token corruption; see the NVIDIA forum report on 2× DGX Spark). The 0.1.1 image's stock code is the exact pre-#50004 state, so removal = upstream's post-revert state. Historical note: upstream re-landed adaptive top-k width capture-safely in [#52823](https://github.com/vllm-project/vllm/pull/52823) on 2026-08-21; this fork still removes the obsolete #50004 backport because its pinned image's stock code predates #50004 entirely — there is no capture-safe variant to port onto it |
| `hotfix-dsv4-skip-empty-c128-48957.sh` | #48957 | ~2x kernel: skip C128 compressor launch when no request crosses a 128-token boundary | **script verified, not yet applied**; fires only when cudagraph mode ≠ FULL |
| `hotfix-dsv4-flashmla-workspace-50298.sh` | #50298 | 1.88x kernel: workspace reuse for combined topk+SWA indices on the FlashMLA prefill path | **script verified, not yet applied** |

All four are idempotent, apply on both nodes (each runs its own container),
and never restart the server themselves. Each supports `--before` / `--after`
(host-side KV-budget + prompt-histogram validation) and `--status`.

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
- The upstream unit tests for #48407
  (`test_mla_short_prefill_indexer.py`) are not ported.
