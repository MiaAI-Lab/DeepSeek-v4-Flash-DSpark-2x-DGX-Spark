# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 TP padding for non-divisible tensor-parallel sizes (TP=3 on 3 Sparks).

WHY THIS EXISTS
---------------
DeepSeek-V4-Flash has num_attention_heads=64 and o_groups=8. Neither divides by 3,
so vLLM refuses to start at TP=3 (`attention.py:183`) -- and, worse, silently
mis-shards the group axis (`attention.py:191`, `8 // 3 == 2`, no assert).

This module supplies the padding so a 3-node tensor-parallel deployment is legal
AND numerically identical to TP=1.

THE GROUP SEMANTICS (measured, not assumed)
-------------------------------------------
V4's o_proj is a per-group BMM:

    z = einsum("bhr,hdr->bhd", o, wo_a)     # h=group, r=hpg*head_dim, d=o_lora_rank
    out = wo_b(z.flatten(1))

Two candidate resolutions were tested against the REAL fp8 checkpoint weights
(`layers.3.attn.wo_a.weight`, [8192, 4096]):

  R2  pad groups 8 -> 24 (24 % 3 == 0), heads -> 72, hpg = 3.
      *** REJECTED -- arithmetically impossible. ***
      With G=24, each group slab needs width hpg*head_dim = 3*512 = 1536, but the
      checkpoint's slabs are 4096 wide. You cannot pad to a NARROWER tensor.
      Independently: a real 8-head group cannot split 3 ways (8/3 = 2.67).

  R1  replicate groups (n_local_groups = n_groups = 8), shard heads WITHIN each
      group. Pad heads-per-group 8 -> 9, so total heads = 8 * 9 = 72, and each
      rank takes 3 heads per group.
      *** CORRECT -- reproduces TP=1 at rel 2.3e-06 (fp8 e4m3 quantization noise). ***

The critical consequence of R1: wo_a's out-features stay G*o_lora_rank (they are
NOT divided by tp_size), so wo_b's input dim is unchanged and the all-reduce sums
exact per-rank partials. The padded 9th head in each group is zero and therefore
contributes exactly nothing.

Note the head count 72 also satisfies the naive rule `heads % (tp * o_groups) == 0`,
but for a DIFFERENT structural reason (9 heads per replicated group, not 24 padded
groups). Same number, different mechanism -- do not conflate them.

WHAT NOT TO DO
--------------
* Do NOT reuse GLM's `glm_wrapper_pad_backend_ok()`. It resolves against
  FLASHINFER_MLA_SPARSE_SM120, a backend DSv4 explicitly REJECTS
  (`nvidia/model.py:769-777`). Setting the correct backend env var would silently
  flip the pad value.
* Do NOT route `attn_sink` through GLM's `copy_tp_shard_with_pad()`. That helper
  calls `dest.zero_()` first; a zeroed sink lane means exp(0)=1, which adds a
  phantom unit to the softmax denominator and ATTENUATES REAL HEADS. Sink pad
  lanes must stay -inf. See `pad_fill_value()` below.

Enabled by env VLLM_DSV4_TP_PAD (default "1").

Redmine: 
"""
from __future__ import annotations

import os

import torch

__all__ = [
    "dsv4_tp_pad_enabled",
    "pad_heads_for_groups",
    "local_heads_per_group",
    "pad_fill_value",
    "describe_geometry",
    "copy_head_shard_within_groups",
]


def dsv4_tp_pad_enabled() -> bool:
    return os.environ.get("VLLM_DSV4_TP_PAD", "1") not in ("", "0", "false", "False")


# FlashInfer DSv4 SM120 decode dispatch head counts. The backend zero-pads a
# smaller local head count up to the next entry around the kernel call, so any
# local <= 128 is runnable. (nvidia/flashinfer_sparse.py:592-603)
#
# NOTE: this is vLLM's CLAIM about the kernel, not the kernel itself. A decode
# smoke test at the chosen local head count is the only way to confirm it.
_DISPATCH_LOCAL_HEADS = (8, 16, 32, 64, 128)
_MAX_LOCAL_HEADS = 128


def pad_heads_for_groups(num_heads: int, tp_size: int, o_groups: int) -> int:
    """Smallest head count >= num_heads that is TP-legal under R1 group semantics.

    R1 keeps all `o_groups` groups on every rank and shards heads within each
    group, so the binding constraint is on heads-per-group:

        heads_per_group % tp_size == 0

    i.e. num_heads must be a multiple of `tp_size * o_groups`.

    For V4-Flash at TP=3: 64 -> 72 (8 groups x 9 heads, 3 heads/group/rank).

    Raises if the resulting local head count exceeds what the decode kernel can
    dispatch -- better a loud failure here than a silent mis-dispatch later.
    """
    if tp_size <= 1:
        return num_heads
    if o_groups <= 0:
        raise ValueError(f"o_groups must be positive, got {o_groups}")

    step = tp_size * o_groups
    padded = ((num_heads + step - 1) // step) * step

    local = padded // tp_size
    if local > _MAX_LOCAL_HEADS:
        raise ValueError(
            f"padded local head count {local} exceeds the max dispatchable "
            f"{_MAX_LOCAL_HEADS} (heads={num_heads} -> {padded}, tp={tp_size}, "
            f"o_groups={o_groups})"
        )
    return padded


def local_heads_per_group(padded_heads: int, tp_size: int, o_groups: int) -> int:
    """Heads per group held by ONE rank under R1.

    V4-Flash at TP=3: 72 / (3 * 8) = 3.

    This is the value that must satisfy the hard invariant at
    `fused_inv_rope_fp8_quant.py:183`:  num_heads == n_groups * heads_per_group
    where num_heads there is the rank-local head count.
    """
    step = tp_size * o_groups
    if padded_heads % step != 0:
        raise ValueError(
            f"padded_heads={padded_heads} is not a multiple of "
            f"tp_size*o_groups={step}; R1 sharding would be non-integral"
        )
    return padded_heads // step


def pad_fill_value(param_name: str) -> float:
    """The value a padded lane of `param_name` must be filled with.

    *** THE WHOLE POINT OF THIS FUNCTION IS THAT IT IS NOT ALWAYS ZERO. ***

    attn_sink is an extra logit appended to the softmax denominator:
        denom = sum_j exp(s_j) + exp(sink)

      sink = -inf  =>  exp(-inf) = 0  =>  denominator unchanged, sink DISABLED.
                       This is the identity element and the correct pad value.
      sink = 0     =>  exp(0) = 1     =>  adds 1.0 to the denominator, shrinking
                       every attention weight for that head. Silently wrong --
                       plausible output, wrong attention temperature.

    Every other padded tensor (wq_b rows, wo_a slices, wo_b rows) takes 0.0,
    which annihilates exactly through the fp8 einsum and the all-reduce.

    wo_b's pad rows do not strictly need zeroing (a zero `z` slice annihilates
    them), but they MUST NOT be left as uninitialized `torch.empty` memory:
    0 * NaN = NaN would poison every rank through the all-reduce.
    """
    if param_name.endswith("attn_sink") or ".attn_sink" in param_name:
        return -float("inf")
    return 0.0


def describe_geometry(num_heads: int, tp_size: int, o_groups: int) -> str:
    """One-line INFO log so a wrong shard is visible at startup, not at eval time."""
    padded = pad_heads_for_groups(num_heads, tp_size, o_groups)
    local = padded // tp_size
    hpg = local_heads_per_group(padded, tp_size, o_groups)
    return (
        f"DSv4 TP pad: heads {num_heads}->{padded} (tp={tp_size}), "
        f"local_heads={local}, o_groups={o_groups} (REPLICATED, not divided), "
        f"heads_per_group_per_rank={hpg}, real_heads={num_heads}"
    )


def copy_head_shard_within_groups(
    dest: torch.Tensor,
    src: torch.Tensor,
    tp_rank: int,
    tp_size: int,
    o_groups: int,
    real_heads: int,
    head_dim: int,
    fill: float = 0.0,
) -> None:
    """Load a rank's head-shard under R1 group-replicated sharding.

    Unlike a flat column shard, R1 takes a contiguous slice of heads FROM EACH
    GROUP, not a contiguous slice of the whole head axis. Rank k holds heads
    [k*hpg, (k+1)*hpg) within every one of the `o_groups` groups.

    src is the full checkpoint tensor whose leading axis is real_heads*head_dim.
    dest is this rank's parameter, sized o_groups*hpg*head_dim.

    Padded lanes (group slots >= real heads-per-group) get `fill`.
    """
    dest.fill_(fill)

    real_hpg = real_heads // o_groups                # 8
    hpg_local = dest.shape[0] // (o_groups * head_dim)
    start = tp_rank * hpg_local

    for g in range(o_groups):
        for j in range(hpg_local):
            h_in_group = start + j
            if h_in_group >= real_hpg:
                continue  # padded head: leave at `fill`
            src_head = g * real_hpg + h_in_group
            dst_off = (g * hpg_local + j) * head_dim
            src_off = src_head * head_dim
            dest[dst_off : dst_off + head_dim].copy_(
                src[src_off : src_off + head_dim]
            )
