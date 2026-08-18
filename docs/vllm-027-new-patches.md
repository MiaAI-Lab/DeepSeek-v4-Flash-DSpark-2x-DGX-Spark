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
| `hotfix-dsv4-qk-rmsnorm-split-blocks.py` | #49283 | per-task Q/KV RMSNorm block widths; on this checkpoint that is **KV reduction 1024→512, Q unchanged** | **wired, default OFF** (`DSPARK_ENABLE_QK_RMSNORM_SPLIT=1`, fail-closed); byte-equivalent to upstream `419d610a`, never applied to the image (see below) |

The six `.sh` backports are idempotent, apply on both nodes (each runs its own
container), and never restart the server themselves. Each supports `--before` /
`--after` (host-side KV-budget + prompt-histogram validation) and `--status`.
The Python backport (`#49283`) is idempotent and both-node too, but it is
gated OFF by default and exposes `--status` / `--dry-run` instead of
`--before` / `--after` — it changes no KV budget, so there is nothing for those
two to compare.

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

## #49283 — what it does on this checkpoint, and what is NOT measured

### The transform
`_fused_q_kv_rmsnorm_kernel` normalises two rows per token on a
`(num_tokens, 2)` grid. Stock code launches both tasks with one shared width,
`BLOCK_SIZE = next_power_of_2(max(q_size, kv_size))`. Upstream extracts a
reusable `_rmsnorm_row` helper and gives each task its own compile-time width.
Five literal anchors, one write, no allocation change, same grid, same launch
count.

### The premise is inverted here — check the config, not the upstream comment
The caller splits `qr_kv` into `[q_lora_rank, head_dim]`, so `q_size =
q_lora_rank` and `kv_size = head_dim`. `DeepSeek-V4-Flash-0731` sets
`q_lora_rank = 1024`, `head_dim = 512`:

| task | size | stock `BLOCK_SIZE` | patched | change |
|---|---:|---:|---:|---|
| Q | 1024 | `next_pow2(1024)` = 1024 | `Q_BLOCK` = 1024 | none |
| KV | 512 | 1024 | `KV_BLOCK` = 512 | 1024 → 512 |

So the **KV** row is the narrow one, `Q_BLOCK` is unchanged, and because 1024
is already a power of two the Q row never had a masked lane to waste. Upstream's
rationale (narrow Q rows paying for KV-wide reductions) and its 1.19–1.34×
kernel microbenchmark are for `q_size, kv_size = 192, 576` on an Intel B60 —
the opposite asymmetry on a different backend. This is the same class of
mistake #48407 documents above: **the upstream premise is false here**, so the
effect that remains is only the KV-side narrowing.

### Evidence status
**Verified, CPU, re-asserted on every CI run**
(`scripts/test-qk-rmsnorm-split-blocks.py`): applying the patcher to upstream's
pre-image `8688a06` (`tests/fixtures/vllm-49283/fused_qk_rmsnorm.base.py`,
sha256 `8ea5fd82ab09db66…`) reproduces upstream's post-image `419d610a`
(`…head.py`, sha256 `cb5262282376c5c4…`) **byte for byte**; all five anchors
occur exactly once; re-running re-validates instead of double-patching and
leaves the bytes stable; anchor drift is refused with nothing written; a
post-write self-check failure restores the original bytes.

**Not verified: any serving-level effect.** No end-to-end measurement is
claimed. Per token per layer the kernel touches 1024 + 512 bf16 inputs and as
many outputs (~0.5 MB/token across 43 layers); decode on this MoE is
weight-bandwidth bound by orders of magnitude, the launch count is unchanged,
and only one of two tasks narrows, so the expected effect is at or below the
noise floor of `probes/ttft-bench` (one sample per 2K/8K/32K band, median of
bands). Earlier revisions of the patch docstring claimed +4.0% decode and a
+12.2% stacked figure; neither was attributable to this patch and both were
removed rather than restated.

**Not verified: that the anchors match the deployed image.** They are proven
against upstream `8688a06` (2026-07-21); the pinned image tree is
`g752a3a504` (2026-07-14), and anchor 3 embeds the post-grid-fix
`pid_task = tl.program_id(1)` layout. Close the gap without mutating anything:
`docker exec <c> python3 /opt/hotfix-dsv4-qk-rmsnorm-split-blocks.py --dry-run`.
Upstream #49283 is still **open**, self-described as a clarity/flexibility
refactor, and the file was deleted from upstream `main` on 2026-07-30, so a
future image bump will (fail-closed) break the anchors.

### Numerics
Masked lanes load exact `0.0` and contribute exact `0.0` to `sum(x * x)`, so
the mathematical variance is unchanged and the Q row is bitwise unchanged. The
KV row's fp32 summation *order* can change with the block width, because Triton
derives the per-thread element layout from `BLOCK`. Bitwise equality on the KV
row is therefore **not** claimed; the bound is ~1 ulp fp32 in `variance`, hence
≤1 ulp in the bf16 store. Two consequences: do not build a bitwise A/B gate on
the KV row, and never apply this on one rank only — `fused_wqa_wkv` is built
with `disable_tp=True`, so the latent is TP-replicated and asymmetric
application would make the ranks disagree. `start-deepseek-v4-flash-dspark.sh`
scp's both the patcher and `.env.dspark` to the worker, so one flag value
governs both ranks.

### Operating it
| value of `DSPARK_ENABLE_QK_RMSNORM_SPLIT` | behavior |
|---|---|
| `0` / unset / anything ≠ `1` | stock kernel; patcher mounted and synced to the worker but **never invoked** |
| `1` | patcher runs at boot on each rank, chained with `\|\| exit 1` |

The rewrite lives in the container filesystem, which is not a volume, so it is
discarded by every `docker compose up` and re-applied by the next boot.
`--status` reports `applied` / `not-applied` / `drifted` with the target sha256
and the image's vLLM version and exits 0/1/2; `--dry-run` predicts the apply
outcome (anchors *and* the resulting self-check) and exits 0/2. Neither writes.

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
