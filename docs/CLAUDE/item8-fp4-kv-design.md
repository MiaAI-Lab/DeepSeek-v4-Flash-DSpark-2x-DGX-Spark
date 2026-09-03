# Item 8 — a real 4-bit compressed-KV layout for DeepSeek-V4-Flash on SM121 (design + feasibility)

**Status:** design and feasibility only (no runtime change). Companion to
`docs/CLAUDE/fable5-1-report.md` §3.8. Written 2026-09-02 against the pinned image
`ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (vLLM `0.25.2.dev0+g752a3a504`, FlashInfer 0.6.15).

## 1. What `nvfp4_ds_mla` is today

| Fact | Where |
|---|---|
| Per-token KV row is **584 B = 448 B NoPE (fp8 e4m3, seven UE8M0 per-64 scales) + 128 B RoPE (64 × bf16) + 8 B footer (7 scales + 1 pad)** for *both* `fp8_ds_mla` and `nvfp4_ds_mla`. | FlashInfer `include/flashinfer/attention/sparse_mla_sm120/model/kv_cache_traits.cuh` (`KVCacheTraits<DSV4>`: `KV_GMEM_STRIDE = 584`, `QUANT_TILE = 64`, `SCALE_FORMAT = UE8M0_BYTE`, `V_HAS_ROPE = true`); vLLM `v1/kv_cache_interface.py` (`storage_block_size * 584` for either dtype); `models/deepseek_v4/attention.py::get_kv_cache_spec` (`alignment = 584 if nvfp4_ds_mla else 576`). |
| The "nvfp4" string only changes kernel dispatch (issue #22 routes it to the fp8 kernel because the bytes are identical). | `patches/hotfix-nvfp4-ds-mla-issue22.sh`, `flashinfer_sparse.py::_as_sparse_cache` (`view(-1, 64, 1, 584)`). |
| Writers: compressed C4/C128 rows are produced by a **Python-side CuTeDSL kernel** (`nvidia/ops/sparse_attn_compress_cutedsl.py::SparseAttnCompressNormRopeStoreC4Kernel`, Triton fallback `common/ops/fused_compress_quant_cache.py`); the sliding-window (SWA) rows by the **C++ op** `torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` (not modifiable without rebuilding vLLM's `_C`). | `compressor.py::forward`, `attention.py::_fused_qnorm_rope_kv_insert` |
| Reader: one FlashInfer SM120 kernel family for decode *and* chunked prefill, **JIT-built from CUDA sources shipped in the container** (`flashinfer/data/csrc/sparse_mla_sm120_decode_dsv4.cu`, `..._prefill.cu`, headers under `sparse_mla_sm120/`), module name `sparse_mla_sm120` (`flashinfer/jit/mla.py::gen_sparse_mla_sm120_module`). A fork is therefore possible without a FlashInfer wheel rebuild; the JIT cache key must be busted by renaming the module. | |
| Hybrid allocator constraint: every KV group must have the **same page size in bytes**. Today all groups land on 64 rows × 584 B = 37,376 B (MLA C4: 256 tokens / 4; SWA: 64 tokens; draft SWA groups use block 4/8 with the same page bytes). | `v1/kv_cache_utils.py` (uniform page size), boot log KV groups (1× `MLAAttentionSpec` 256 + 3× `SlidingWindowMLASpec` 64/4/8) |

Per-token KV bytes actually consumed at long context (TP=2 replicates all of it on both ranks):
21 C4 layers × 584/4 = 3.07 KB, 20 C128 layers × 584/128 = 0.09 KB, indexer K cache 21 × 132/4 = 0.69 KB
(replicated), SWA windows are a per-sequence constant (46 layers × 128 rows × 584 B ≈ 3.4 MB/seq). The
headline "2.33 M tokens in 17.04 GiB" (7.8 KB/token) is allocator accounting across the four groups, not
the physical row cost — so a smaller compressed row moves capacity more than linearly with the row shrink.

## 2. Layout options that satisfy the page-size constraint

Only row sizes that divide the common page with a power-of-two row count are cheap for the allocator
(`37,376 = 584 × 64 → 292 × 128 → 146 × 256`). Candidates for the compressed (C4/C128) row:

| Option | Row layout | Bytes | Fits 292? | Numerics risk | Kernel work |
|---|---|---|---|---|---|
| **A — "half row", coarse NoPE scales** | 448 NoPE as e2m1 (224 B) + 4 UE8M0 scales per **128** dims (4 B) + 64 RoPE as fp8 e4m3 (64 B) | 292 | yes (exact) | NoPE block-128 scaling is coarse for e2m1 (1 mantissa bit); RoPE fp8 is fine | dequant fp4→fp8/bf16 in `d2_load_b.cuh`, scale path in `scale_convert.cuh`, RoPE tile now fp8 in `xv_rope_mma.cuh` |
| **B — "half row", fp4 RoPE** | 224 B NoPE e2m1 + 14 UE8M0 per-32 (MXFP4, matches the existing indexer MXFP4 kernels) + 32 B RoPE e2m1 + 2 B RoPE scales + 8 B footer, 12 B pad | 280/292 | yes | RoPE in fp4 degrades positional precision at 1M context — highest quality risk | same as A plus fp4 RoPE dequant |
| **C — 360 B compressed row, bf16 RoPE kept** | 224 B + 14 B + 128 B RoPE bf16 (+ footer) | 366 → 368 | **no** (37,376/368 not integral) | lowest risk | requires decoupling page sizes per group in the hybrid allocator (upstream vLLM design change) or moving SWA rows to the same 368 B (impossible: SWA rows are written by the C++ op) |
| **D — DCP-2 instead of fp4** | keep 584 B; shard *tokens* across the two ranks (`decode_context_parallel_size=2`) | — | n/a | none (exact) | DSV4 sparse backend has no DCP support in this image (`grep dcp models/deepseek_v4/` is empty); indexer op *does* (`_merge_dcp_topk_global`) |

Recommendation: prototype **A** first (exact fit, fp8 RoPE preserved, one quant-tile change) with an
offline numerics check before touching kernels; keep **B** as fallback if block-128 NoPE scaling is too
lossy; treat **D** as the zero-numerics-risk alternative for capacity if upstream lands DSV4 DCP.

Expected capacity at option A: compressed term 3.07 → 1.54 KB/token, indexer unchanged (0.69), C128
0.05 → ≈ 2.3 KB/token vs 3.85 today → **+65 % tokens at the physical level; +40–50 % in the allocator
headline** (SWA groups keep 584 B pages).

## 3. Step 0 — the cheap experiment: MXFP4 *indexer* cache

vLLM already ships an MXFP4 indexer K cache (`AttentionConfig.use_fp4_indexer_cache`, Triton insert
`_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn`, DeepGEMM `fp8_fp4_*_mqa_logits` consume
packed FP4 K), but the builder gates it to datacenter Blackwell:
`v1/attention/backends/mla/indexer.py:274-283` ("use_fp4_indexer_cache requires Blackwell datacenter
GPUs"). DeepGEMM does ship `sm120_fp4_mqa_logits.cuh` / `sm120_fp4_paged_mqa_logits.cuh`, so the gate is
probably conservative rather than a hard kernel limit. A one-line gate relaxation + `--attention-config
'{"use_fp4_indexer_cache": true}'` would cut the (replicated) indexer cache in half (−0.35 KB/token ≈ −9 %
KV bytes) and halve indexer K reads. Needs the SM121 DeepGEMM header alias (§5) because the fp4 logits
kernels are certainly not in the JIT cache yet.

## 4. Work breakdown for option A (kernel fork)

1. **Offline numerics** (no engine change): dump one layer's compressed KV rows from a real prompt
   (hook `compress_norm_rope_store_cutedsl`), simulate e2m1 + per-128 UE8M0 quant/dequant in numpy,
   measure attention-output error vs fp8 rows; the Stage-C knob
   `VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT` in `recipe/overlay` was built for this kind of check.
2. **Writer**: add an fp4 variant of `SparseAttnCompressNormRopeStoreC4Kernel` (values: two e2m1 per
   byte, scales: exponent per 128, RoPE quantized to e4m3), selected by a new cache dtype string
   (`nvfp4c_ds_mla` — keep `nvfp4_ds_mla` meaning what it means today to avoid breaking #22).
3. **Reader**: new `KVCacheTraits<DSV4_FP4>` (`KV_GMEM_STRIDE 292`, `QUANT_TILE 128`, `NUM_SCALES 4`,
   RoPE fp8) + dequant in the B-operand load; instantiate in `sparse_mla_sm120_decode_dsv4.cu` and the
   prefill orchestrator; rename the JIT module (e.g. `sparse_mla_sm120_fp4c`) so the cache is rebuilt.
   SWA cache stays `KVCacheTraits<DSV4>` (the dual-cache API already takes the two caches separately).
4. **vLLM plumbing**: `attention.py::get_kv_cache_spec` (alignment 292 for compressed layers only,
   block 512 tokens so the page stays 37,376 B), `kv_cache_interface.py` byte formulas,
   `flashinfer_sparse.py::_as_sparse_cache` (row width per cache), `#22`-style dispatch.
5. **Validation**: RULER-lite 8K→262K (`scripts/ruler-lite.py`), garble sweep to 900K
   (`scripts/context-garble-sweep.py`), spec-decode acceptance unchanged (`scripts/spec-acceptance.py`),
   `bench-miaai.py` c=1/c=6 (attention decode reads shrink, so a small tok/s gain at long context is
   expected, none at short).

Effort estimate: 2–3 engineer-weeks including validation. Risk: numerics at 1M context (mitigated by
step 1) and FlashInfer JIT build time (~minutes per boot on first use; persisted cache afterwards).

## 5. Hazard found while doing this: DeepGEMM SM121 header mismatch

The vendored DeepGEMM (`vllm/third_party/deep_gemm`) generates
`sm121_fp8_mqa_logits<...>` / `#include "impls/sm121_fp8_mqa_logits.cuh"` on GB10 but ships only
`sm120_*` headers. The live stack works because `VLLM_CACHE_ROOT/deep_gemm/cache/kernel.smxx_fp8_mqa_logits.*`
holds cubins compiled from `sm120_fp8_mqa_logits.cuh` (Jul 16 / Jul 27 builds). Any cache miss — fresh
volume, new `VLLM_CACHE_ROOT`, a different index head count, or the fp4 indexer path — fails at first use:

```
Failed to open .../deep_gemm/impls/sm121_fp8_mqa_logits.cuh
identifier "sm121_fp8_mqa_logits" is undefined
```

`patches/hotfix-deepgemm-sm121-mqa-header-alias.sh` (opt-in) writes four alias headers
(`#include "sm120_X.cuh"` + `#define sm121_X sm120_X`); it produced working kernels in the item-6 GPU
test (`scripts/test-sp-indexer-gpu.py`). Back up `VLLM_CACHE_ROOT/deep_gemm` until this is wired in.
