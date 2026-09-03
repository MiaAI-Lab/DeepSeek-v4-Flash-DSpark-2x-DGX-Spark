#!/usr/bin/env python3
"""GPU equivalence test for the sequence-parallel indexer prefill hotfix.

Runs INSIDE the Anemll image on one GPU (no TP process group needed): the
per-rank candidate step is executed once per emulated rank and the resulting
candidate tensors are concatenated exactly as the TP all-gather would, then
merged with the real CuTeDSL stable-topk selector. The result is compared to
the stock single-rank path (full gather, full-range logits, top_k_per_row_prefill)
on the same paged indexer cache. Real kernels throughout (DeepGEMM
fp8_fp4_mqa_logits, cp_gather_indexer_k_quant_cache, top_k_per_row_prefill,
stable_topk_from_gathered_candidates_cutedsl).

Usage (host, GPU idle; a private VLLM_CACHE_ROOT keeps the production JIT cache
untouched, and the DeepGEMM SM121 header alias is required on a cold cache —
see docs/CLAUDE/item8-fp4-kv-design.md §5):
  docker run --rm --gpus all --ipc host --entrypoint bash \
    -e VLLM_CACHE_ROOT=/vc -e TRITON_CACHE_DIR=/vc/triton -e DG_JIT_USE_NVRTC=0 \
    -e CUTE_DSL_ARCH=sm_121a -e TORCH_CUDA_ARCH_LIST=12.1a -v /tmp/sp-vc:/vc \
    -v $PWD/patches:/opt/dspark-patches:ro -v $PWD/scripts:/opt/scripts:ro \
    ghcr.io/anemll/dspark-vllm-gx10:0.1.1 -c \
    'bash /opt/dspark-patches/hotfix-deepgemm-sm121-mqa-header-alias.sh && \
     PATH=/usr/local/cuda/bin:$PATH python3 /opt/scripts/test-sp-indexer-gpu.py'
Verified 2026-09-02 on GB10: PASS for emulated TP=2 and TP=3 on three request
shape sets (identical valid-candidate counts and score multisets vs stock).
"""
from __future__ import annotations

import importlib.util
import sys
import types

import torch

PATCH = "/opt/dspark-patches/hotfix-dsv4-sp-indexer-prefill.py"
HEAD_DIM = 128
IDX_HEADS = 64
TOPK = 512
BLOCK_ROWS = 64  # 256-token pages / compress ratio 4


def load_helpers():
    spec = importlib.util.spec_from_file_location("sp_patch", PATCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from vllm import _custom_ops as ops
    from vllm.platforms import current_platform
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits
    from vllm.utils.import_utils import has_cutedsl

    ns: dict = {
        "torch": torch,
        "os": __import__("os"),
        "ops": ops,
        "fp8_fp4_mqa_logits": fp8_fp4_mqa_logits,
        "has_cutedsl": has_cutedsl,
        "current_platform": current_platform,
        "get_tensor_model_parallel_world_size": lambda: 2,
        "get_tensor_model_parallel_rank": lambda: 0,
        "get_tp_group": lambda: None,
    }
    exec(compile(mod.HELPER_SRC, "helper", "exec"), ns)
    return ns


def make_case(device, lens, query_pos):
    """Build a paged fp8 indexer cache + chunk metadata for `lens` compressed
    keys per request and queries at compressed positions `query_pos` (list of
    lists, one per request; each entry is the exclusive end ke-ks)."""
    num_reqs = len(lens)
    blocks_per_req = [(L + BLOCK_ROWS - 1) // BLOCK_ROWS for L in lens]
    max_blocks = max(blocks_per_req)
    num_blocks = sum(blocks_per_req) + 1
    row_bytes = HEAD_DIM + 4
    g = torch.Generator(device="cpu").manual_seed(0)
    # random fp8 payload (as raw bytes) and fp32 scale 1.0 per token
    raw = torch.randint(0, 255, (num_blocks, BLOCK_ROWS, row_bytes), generator=g, dtype=torch.uint8)
    # avoid NaN/Inf fp8 bit patterns (0x7f/0xff -> nan in e4m3fn)
    payload = raw[..., :HEAD_DIM]
    payload[payload == 0x7F] = 0x70
    payload[payload == 0xFF] = 0xF0
    scale = torch.tensor([1.0], dtype=torch.float32).view(torch.uint8)
    raw[..., HEAD_DIM:] = scale
    kv_cache = raw.to(device)
    block_table = torch.zeros((num_reqs, max_blocks), dtype=torch.int32)
    nxt = 1
    for i, nb in enumerate(blocks_per_req):
        block_table[i, :nb] = torch.arange(nxt, nxt + nb)
        nxt += nb
    block_table = block_table.to(device)
    cu = torch.zeros(num_reqs + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(lens, dtype=torch.int32), 0)
    ks, ke = [], []
    for i, qp in enumerate(query_pos):
        for e in qp:
            ks.append(int(cu[i]))
            ke.append(int(cu[i]) + int(e))
    M = len(ks)
    ks_t = torch.tensor(ks, dtype=torch.int32, device=device)
    ke_t = torch.tensor(ke, dtype=torch.int32, device=device)
    chunk = types.SimpleNamespace(
        cu_seq_lens=cu.to(device),
        cu_seqlen_ks=ks_t,
        cu_seqlen_ke=ke_t,
        block_table=block_table,
        total_seq_lens=int(cu[-1]),
        max_local_total_seq_lens=int(cu[-1]),
        local_cu_seq_lens=cu.to(device),
        local_total_seq_lens=int(cu[-1]),
        skip_kv_gather=False,
        token_start=0,
        token_end=M,
    )
    qraw = torch.randint(0, 255, (M, IDX_HEADS, HEAD_DIM), generator=g, dtype=torch.uint8)
    qraw[qraw == 0x7F] = 0x70
    qraw[qraw == 0xFF] = 0xF0
    q = qraw.to(device).view(torch.float8_e4m3fn)
    weights = torch.rand((M, IDX_HEADS), generator=g).to(device) * 0.05
    return kv_cache, chunk, q, weights, M


def reference(ns, kv_cache, chunk, q, weights, M, device):
    from vllm import _custom_ops as ops
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

    T = chunk.total_seq_lens
    k_quant = torch.empty((T, HEAD_DIM), dtype=torch.float8_e4m3fn, device=device)
    k_scale = torch.empty((T, 4), dtype=torch.uint8, device=device)
    ops.cp_gather_indexer_k_quant_cache(kv_cache, k_quant, k_scale, chunk.block_table, chunk.cu_seq_lens)
    logits = fp8_fp4_mqa_logits(
        (q, None), (k_quant, k_scale.view(torch.float32).squeeze(-1)), weights,
        chunk.cu_seqlen_ks, chunk.cu_seqlen_ke, clean_logits=False,
    )
    idx = torch.full((M, TOPK), -1, dtype=torch.int32, device=device)
    ops.top_k_per_row_prefill(logits, chunk.cu_seqlen_ks, chunk.cu_seqlen_ke, idx, M,
                              logits.stride(0), logits.stride(1), TOPK)
    return logits, idx


def sp_merged(ns, kv_cache, chunk, q, weights, M, device, tp=2):
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        stable_topk_from_gathered_candidates_cutedsl,
    )

    T = chunk.total_seq_lens
    packed_all = []
    for r in range(tp):
        k_quant = torch.empty((T, HEAD_DIM), dtype=torch.float8_e4m3fn, device=device)
        k_scale = torch.empty((T, 4), dtype=torch.uint8, device=device)
        idx = torch.full((M, TOPK), -1, dtype=torch.int32, device=device)
        packed = ns["_sp_indexer_local_candidates"](
            chunk, kv_cache, k_quant, k_scale, q, None, weights, idx, TOPK, False, tp, r
        )
        packed_all.append(packed)
    gathered = torch.cat(packed_all, dim=1)
    out = torch.full((M, TOPK), -1, dtype=torch.int32, device=device)
    stable_topk_from_gathered_candidates_cutedsl(gathered, TOPK, out=out)
    return out


def row_scores(logits, ks, idx):
    valid = idx >= 0
    col = (idx.clamp(min=0) + ks[:, None]).to(torch.int64)
    s = torch.gather(logits, 1, col)
    s[~valid] = float("-inf")
    return torch.sort(s, dim=1, descending=True).values, valid.sum(1)


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0
    device = torch.device("cuda")
    ns = load_helpers()
    cases = [
        # (compressed lens per request, per-request query ends (rel_ke))
        ([3000, 5000], [[3000, 2999, 700, 300], [5000, 4096, 2501, 64]]),
        ([1024], [[1024, 1000, 513, 512, 511, 1]]),
        ([65, 7000, 130], [[65], [7000, 3500, 3499, 3600], [130, 5]]),
    ]
    fails = 0
    for lens, qpos in cases:
        kv_cache, chunk, q, weights, M = make_case(device, lens, qpos)
        logits, ref_idx = reference(ns, kv_cache, chunk, q, weights, M, device)
        for tp in (2, 3):
            got = sp_merged(ns, kv_cache, chunk, q, weights, M, device, tp=tp)
            ref_s, ref_n = row_scores(logits, chunk.cu_seqlen_ks, ref_idx)
            got_s, got_n = row_scores(logits, chunk.cu_seqlen_ks, got)
            # count of valid candidates must match, and the sorted score
            # multisets must match (robust to tie ordering)
            same_n = bool(torch.equal(ref_n, got_n))
            finite = torch.isfinite(ref_s)
            same_s = bool(torch.allclose(ref_s[finite], got_s[finite], rtol=0, atol=0))
            # exact index equality is expected too when there are no ties
            same_idx = bool(torch.equal(torch.sort(ref_idx, 1).values, torch.sort(got, 1).values))
            ok = same_n and same_s
            fails += 0 if ok else 1
            print(f"lens={lens} tp={tp}: valid-count match={same_n} score-multiset match={same_s} "
                  f"exact-index match={same_idx} -> {'PASS' if ok else 'FAIL'}")
            torch.cuda.synchronize()
    print("RESULT:", "PASS" if fails == 0 else f"FAIL ({fails})")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
