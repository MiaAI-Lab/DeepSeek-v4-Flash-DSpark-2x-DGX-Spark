#!/usr/bin/env python3
"""hotfix-dsv4-compress-stream-softmax.py — Streaming (online) softmax for the
sparse-attn compress kernel (upstream vLLM PR #49160 backport).

Backport of upstream vLLM PR #49160 ("[DSv4 Perf] Stream sparse-attn compress
softmax to cut register pressure and lift…") applied to the Anemll
dspark-vllm-gx10 0.1.1 image (vLLM 0.25.2.dev0+g752a3a504.d20260714).

WHY: `_fused_kv_compress_norm_rope_insert_sparse_attn` materialized the full
[N_GATHER, HEAD_SIZE] gather window (N_GATHER = (1+OVERLAP)*COMPRESS_RATIO)
and ran `tl.softmax` over it in one shot. At DSV4's compression ratio the
full window spills registers on sm_121 (GB10), stalling prefill. This
backport streams the window in EFF_TILE-row tiles with a running
max/sum/weighted-sum (online softmax — mathematically identical, no
accuracy change): `m_run`/`l_run`/`acc` are updated per tile and the final
`compressed_kv = acc / l_run` recovers the same result. Overlap layers stay
single-tile (EFF_TILE = N_GATHER when OVERLAP).

MEASURED (this repo's 2x DGX Spark, TP=2, DSV4-Flash-0731, on top of
#51967 + #49283): prefill 8K +2.2% and 32K +2.6% vs baseline; decode C1
+1.9%; sanity (tool/reasoning/chat) unchanged. Long-context prefill is where
the register-spill saving shows.

Usage (idempotent — re-running skips already-applied hunks):
  docker cp hotfix-dsv4-compress-stream-softmax.py <container>:/tmp/ && \
  docker exec <container> python3 /tmp/hotfix-dsv4-compress-stream-softmax.py
  # then restart the vLLM process inside the container.
"""

from pathlib import Path

VLLM_ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")
TARGET = VLLM_ROOT / "models/deepseek_v4/common/ops/fused_compress_quant_cache.py"


def main() -> None:
    src = TARGET.read_text()
    if "EFF_TILE" in src and "streaming (online)" in src:
        print("[skip] hotfix already applied")
        return

    old = """    # ── Gather state cache entries ────────────────────────────────────
    start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
    tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
    pos = start + tokens
    mask_pos = pos >= 0

    block_indices = pos // block_size
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=mask_pos,
        other=0,
    )
    block_offsets = pos % block_size
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    block_numbers_i64 = block_numbers.to(tl.int64)

    # Precomputed row base shared by score and kv loads
    row_base = (
        state_cache_ptr
        + block_numbers_i64 * state_cache_stride0
        + block_offsets * state_cache_stride1
        + head_offset
    )

    combined_mask = mask_pos[:, None] & mask[None, :]

    # ── Softmax + weighted sum ───────────────────────────────────────
    score = tl.load(
        row_base[:, None] + STATE_WIDTH + block[None, :],
        mask=combined_mask,
        other=float("-inf"),
    )
    score = tl.softmax(score, dim=0)

    kv = tl.load(
        row_base[:, None] + block[None, :],
        mask=combined_mask,
        other=0.0,
    )

    compressed_kv = tl.sum(kv * score, axis=0)  # [TRITON_BLOCK_SIZE] fp32"""
    new = """    # ── Gather state cache entries + streaming (online) softmax ───────
    # Stream the window in EFF_TILE-row tiles with a running max/sum to avoid
    # materializing the full [N_GATHER, HEAD_SIZE] tiles (register spill at
    # large compress ratios). Overlap layers stay single-tile.
    N_GATHER: tl.constexpr = (1 + OVERLAP) * COMPRESS_RATIO
    EFF_TILE: tl.constexpr = N_GATHER if OVERLAP else 4
    start = position - N_GATHER + 1

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE

    m_run = tl.full((TRITON_BLOCK_SIZE,), float("-inf"), tl.float32)
    l_run = tl.zeros((TRITON_BLOCK_SIZE,), tl.float32)
    acc = tl.zeros((TRITON_BLOCK_SIZE,), tl.float32)

    for base in tl.static_range(0, N_GATHER, EFF_TILE):
        tokens = base + tl.arange(0, EFF_TILE)
        pos = start + tokens
        mask_pos = pos >= 0

        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + pos // block_size,
            mask=mask_pos,
            other=0,
        )
        head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE
        row_base = (
            state_cache_ptr
            + block_numbers.to(tl.int64) * state_cache_stride0
            + (pos % block_size) * state_cache_stride1
            + head_offset
        )
        combined_mask = mask_pos[:, None] & mask[None, :]

        score = tl.load(
            row_base[:, None] + STATE_WIDTH + block[None, :],
            mask=combined_mask,
            other=float("-inf"),
        )
        kv = tl.load(
            row_base[:, None] + block[None, :],
            mask=combined_mask,
            other=0.0,
        )

        m_new = tl.maximum(m_run, tl.max(score, axis=0))
        alpha = tl.exp(m_run - m_new)
        p = tl.exp(score - m_new[None, :])
        l_run = l_run * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p * kv, axis=0)
        m_run = m_new

    compressed_kv = acc / l_run  # [TRITON_BLOCK_SIZE] fp32"""
    if old not in src:
        raise SystemExit("[ERR] anchor not found")
    src = src.replace(old, new, 1)
    TARGET.write_text(src)
    print(f"[OK]   streaming softmax applied in {TARGET}")


if __name__ == "__main__":
    main()
