#!/usr/bin/env python3
"""Sequence-parallel Lightning indexer for prefill (opt-in, TP>1).

Stock Anemll 0.1.1 replicates the DeepSeek-V4 Lightning indexer across TP
ranks: ``DeepseekV4Indexer`` uses ``ReplicatedLinear`` and every rank scores
all 64 index heads against the *whole* compressed key range. On long prompts
that O(queries x keys) score is the dominant prefill cost (900K prefill runs at
~875 tok/s vs ~2,500 tok/s at 2K), and it is computed twice.

This hotfix makes the prefill branch of ``sparse_attn_indexer`` sequence
parallel: TP rank ``r`` gathers and scores only a contiguous, page-aligned
slice ``[lo_r, hi_r)`` of every request's compressed keys, takes its local
top-k, and the ranks exchange only ``(score, global_id)`` candidates
(``tp x index_topk`` per query, a few MB per chunk) before an exact stable
top-k merge. The exchange reuses vLLM's DCP indexer merge kernel
(``stable_topk_from_gathered_candidates_cutedsl``), and exactness follows the
same argument as DCP: a token in the global top-k is in its owning rank's local
top-k, so merging the per-rank local top-k sets is equivalent to top-k over the
full score row.

Decode is untouched (per-step indexer work is small and the extra collective
would cost more than it saves). Only chunks with at least
``DSPARK_SP_INDEXER_MIN_KEYS`` compressed keys (default 8192 = 32K tokens of
context at compress ratio 4) take the parallel path; shorter chunks run the
stock replicated path. DCP>1 and TP=1 always use stock code.

Gate: compose runs this patcher only when ``DSPARK_ENABLE_SP_INDEXER=1``
(fail-closed). Idempotent. Patches
``vllm/model_executor/layers/sparse_attn_indexer.py``.
"""
from pathlib import Path
import sys

P = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "sparse_attn_indexer.py"
)
MARK = "# [dspark-sp-indexer]"

# The helper block is a module constant so the CPU/GPU tests can exec the exact
# code that lands in the image.
HELPER_SRC = '''
# [dspark-sp-indexer] sequence-parallel Lightning-indexer prefill: every TP
# rank scores a contiguous page-aligned slice of each request's compressed key
# range, then the ranks merge (score, global_id) candidates with the DCP
# stable-topk selector. Exact (same argument as the DCP merge).
_SP_INDEXER_MIN_KEYS = int(os.environ.get("DSPARK_SP_INDEXER_MIN_KEYS", "8192"))


def _sp_indexer_split(
    cu_seq_lens: torch.Tensor, block_rows: int, tp_size: int, tp_rank: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Page-aligned [lo, hi) of each request's compressed keys for this rank.

    Returns ``(lo, local_len, local_cu)``; ``local_cu`` is the rank-local
    cumulative length with the same shape/dtype as ``cu_seq_lens``. All
    device-side; no host sync.
    """
    lens = cu_seq_lens[1:] - cu_seq_lens[:-1]
    lo = (lens * tp_rank // tp_size) // block_rows * block_rows
    if tp_rank + 1 == tp_size:
        hi = lens
    else:
        hi = (lens * (tp_rank + 1) // tp_size) // block_rows * block_rows
    local_len = hi - lo
    local_cu = torch.zeros_like(cu_seq_lens)
    torch.cumsum(local_len, dim=0, out=local_cu[1:])
    return lo.to(torch.int32), local_len.to(torch.int32), local_cu


def _sp_indexer_local_bounds(
    cu_seq_lens: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    lo: torch.Tensor,
    local_len: torch.Tensor,
    local_cu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map global per-query [ks, ke) (ks == owning request start) onto this
    rank's local gathered rows. Returns ``(lo_per_query, ks_local, ke_local)``.
    """
    # ks == cu_seq_lens[req]. right=True picks the LAST request sharing that
    # cu value: zero-length requests duplicate their successor's start, and the
    # successor (the one that actually has keys) must win. The empty request's
    # own queries have rel_ke == 0, so mapping them onto the successor is
    # harmless (their local range stays empty).
    req = torch.searchsorted(cu_seq_lens, cu_seqlen_ks, right=True) - 1
    req = req.clamp_(min=0, max=lo.shape[0] - 1)
    rel_ke = cu_seqlen_ke - cu_seqlen_ks
    lo_q = lo[req]
    ks_l = local_cu[req]
    ke_l = ks_l + torch.clamp(rel_ke - lo_q, min=0).minimum(local_len[req])
    return lo_q, ks_l.to(torch.int32), ke_l.to(torch.int32)


def _sp_indexer_local_candidates(
    chunk,
    kv_cache: torch.Tensor,
    k_quant_full: torch.Tensor,
    k_scale_full: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    weights: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    use_fp4_cache: bool,
    tp_size: int,
    tp_rank: int,
) -> torch.Tensor:
    """Gather + score this rank's key slice and return packed
    ``[M, topk, 2]`` fp32 (score, global_id) candidates; -inf/-1 padding."""
    block_rows = kv_cache.shape[1]
    cu_seq_lens = chunk.cu_seq_lens
    lo, local_len, local_cu = _sp_indexer_split(cu_seq_lens, block_rows, tp_size, tp_rank)
    k_quant = k_quant_full[: chunk.max_local_total_seq_lens]
    k_scale = k_scale_full[: chunk.max_local_total_seq_lens]
    if not chunk.skip_kv_gather:
        max_blocks = chunk.block_table.shape[1]
        blk_idx = torch.arange(max_blocks, device=lo.device, dtype=torch.int32)
        blk_idx = (blk_idx + (lo // block_rows)[:, None]).clamp_(max=max_blocks - 1)
        local_block_table = torch.gather(chunk.block_table, 1, blk_idx.to(torch.int64))
        ops.cp_gather_indexer_k_quant_cache(
            kv_cache, k_quant, k_scale, local_block_table, local_cu
        )
    lo_q, ks_l, ke_l = _sp_indexer_local_bounds(
        cu_seq_lens, chunk.cu_seqlen_ks, chunk.cu_seqlen_ke, lo, local_len, local_cu
    )
    q_slice = q_quant[chunk.token_start : chunk.token_end]
    q_scale_slice = (
        q_scale[chunk.token_start : chunk.token_end] if q_scale is not None else None
    )
    if use_fp4_cache:
        q_cast = q_slice.view(torch.int8)
        k_cast = k_quant.view(torch.int8)
        k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
    else:
        q_cast = q_slice
        k_cast = k_quant
        k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
    logits = fp8_fp4_mqa_logits(
        (q_cast, q_scale_slice),
        (k_cast, k_scale_cast),
        weights[chunk.token_start : chunk.token_end],
        ks_l,
        ke_l,
        clean_logits=False,
    )
    ops.top_k_per_row_prefill(
        logits,
        ks_l,
        ke_l,
        topk_indices,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        topk_tokens,
    )
    valid = topk_indices >= 0
    local_idx = topk_indices.clamp(min=0)
    col = (local_idx + ks_l[:, None]).clamp_(max=max(logits.shape[1] - 1, 0))
    score = torch.gather(logits, 1, col.to(torch.int64))
    score = score.masked_fill_(~valid, float("-inf"))
    gid = torch.where(valid, local_idx + lo_q[:, None], torch.full_like(local_idx, -1))
    return torch.stack((score, gid.to(torch.float32)), dim=-1)


def _sp_indexer_prefill_chunk(
    chunk,
    kv_cache: torch.Tensor,
    k_quant_full: torch.Tensor,
    k_scale_full: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    weights: torch.Tensor,
    topk_indices_buffer: torch.Tensor,
    topk_tokens: int,
    use_fp4_cache: bool,
    dcp_world_size: int,
) -> bool:
    """Run one prefill chunk sequence-parallel across TP. Returns False (and
    does nothing) when the stock replicated path should run instead."""
    if dcp_world_size > 1 or _SP_INDEXER_MIN_KEYS <= 0:
        return False
    if chunk.total_seq_lens < _SP_INDEXER_MIN_KEYS:
        return False
    if topk_tokens not in (512, 1024, 2048) or not has_cutedsl():
        return False
    if current_platform.is_xpu():
        return False
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return False
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        stable_topk_from_gathered_candidates_cutedsl,
    )

    topk_indices = topk_indices_buffer[chunk.token_start : chunk.token_end, :topk_tokens]
    packed = _sp_indexer_local_candidates(
        chunk,
        kv_cache,
        k_quant_full,
        k_scale_full,
        q_quant,
        q_scale,
        weights,
        topk_indices,
        topk_tokens,
        use_fp4_cache,
        tp_size,
        get_tensor_model_parallel_rank(),
    )
    gathered = get_tp_group().all_gather(packed, dim=1)
    stable_topk_from_gathered_candidates_cutedsl(gathered, topk_tokens, out=topk_indices)
    return True
'''

IMPORT_OLD = "from vllm.distributed import get_dcp_group\n"
IMPORT_NEW = (
    "from vllm.distributed import (  # [dspark-sp-indexer]\n"
    "    get_dcp_group,\n"
    "    get_tensor_model_parallel_rank,\n"
    "    get_tensor_model_parallel_world_size,\n"
    ")\n"
    "from vllm.distributed.parallel_state import get_tp_group  # [dspark-sp-indexer]\n"
)
OS_OLD = "import torch\n\nimport vllm.envs as envs\n"
OS_NEW = "import os  # [dspark-sp-indexer]\nimport torch\n\nimport vllm.envs as envs\n"
LOGGER_OLD = "logger = init_logger(__name__)\n"
LOOP_OLD = (
    "        for chunk in prefill_metadata.chunks:\n"
    "            cu_seqlen_ks = chunk.cu_seqlen_ks\n"
    "            cu_seqlen_ke = chunk.cu_seqlen_ke\n"
)
LOOP_NEW = (
    "        for chunk in prefill_metadata.chunks:\n"
    "            # [dspark-sp-indexer] long chunks: score this rank's key slice\n"
    "            # only and merge candidates across TP (exact). Falls through\n"
    "            # to the stock replicated path otherwise.\n"
    "            if _sp_indexer_prefill_chunk(\n"
    "                chunk,\n"
    "                kv_cache,\n"
    "                k_quant_full,\n"
    "                k_scale_full,\n"
    "                q_quant,\n"
    "                q_scale,\n"
    "                weights,\n"
    "                topk_indices_buffer,\n"
    "                topk_tokens,\n"
    "                use_fp4_cache,\n"
    "                dcp_world_size,\n"
    "            ):\n"
    "                continue\n"
    "            cu_seqlen_ks = chunk.cu_seqlen_ks\n"
    "            cu_seqlen_ke = chunk.cu_seqlen_ke\n"
)


def apply(path: Path) -> bool:
    src = path.read_text()
    if MARK in src:
        print(f"[dspark-sp-indexer] already applied to {path}")
        return False
    for anchor, name in (
        (IMPORT_OLD, "distributed import"),
        (OS_OLD, "torch/envs import"),
        (LOGGER_OLD, "logger"),
        (LOOP_OLD, "prefill loop"),
    ):
        assert src.count(anchor) == 1, (
            f"dspark-sp-indexer: {name} anchor not found exactly once; refusing to patch"
        )
    src = src.replace(IMPORT_OLD, IMPORT_NEW, 1)
    src = src.replace(OS_OLD, OS_NEW, 1)
    src = src.replace(LOGGER_OLD, LOGGER_OLD + HELPER_SRC, 1)
    src = src.replace(LOOP_OLD, LOOP_NEW, 1)
    compile(src, "sparse_attn_indexer.py", "exec")
    path.write_text(src)
    print(f"[dspark-sp-indexer] patched {path}")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        status_src = P.read_text() if P.is_file() else ""
        print("dspark SP indexer prefill          :",
              "APPLIED" if MARK in status_src else "NOT APPLIED")
        raise SystemExit(0)
    apply(P)
