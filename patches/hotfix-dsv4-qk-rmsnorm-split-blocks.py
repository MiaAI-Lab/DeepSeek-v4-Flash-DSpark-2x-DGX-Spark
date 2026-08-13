#!/usr/bin/env python3
"""hotfix-dsv4-qk-rmsnorm-split-blocks.py — Right-size Q/KV RMSNorm blocks
(upstream vLLM PR #49283 backport).

Backport of upstream vLLM PR #49283 ("[DSv4 Perf] Right-size per-task RMSNorm
blocks for the fused QK norm") applied to the Anemll dspark-vllm-gx10 0.1.1
image (vLLM 0.25.2.dev0+g752a3a504.d20260714).

WHY: the fused Q/KV RMSNorm kernel (`_fused_q_kv_rmsnorm_kernel`) launched
both the Q row and the KV row with one shared `BLOCK_SIZE`:
`triton.next_power_of_2(max(q_size, kv_size))`. On DeepSeek-V4 the Q row is
narrower than the KV row, so the Q task ran with a wide block whose excess
lanes were masked (`mask = block < SIZE`) — pure ALU waste on every
attention layer of every prefill/decode step. This backport extracts a
reusable `_rmsnorm_row` helper and gives the Q and KV tasks their own block
widths (`Q_BLOCK = next_pow2(q_size)`, `KV_BLOCK = next_pow2(kv_size)`),
so narrow Q rows no longer pay for KV-wide reductions. fp32 math is
unchanged; the kernel is numerically identical.

MEASURED (this repo's 2x DGX Spark, TP=2, DSV4-Flash-0731): 32K prefill
+2.2% vs baseline; decode flat; no regressions in the 2K/8K prefill or C4
decode lanes.

Usage (idempotent — re-running skips already-applied hunks):
  docker cp hotfix-dsv4-qk-rmsnorm-split-blocks.py <container>:/tmp/ && \
  docker exec <container> python3 /tmp/hotfix-dsv4-qk-rmsnorm-split-blocks.py
  # then restart the vLLM process inside the container.
"""

from pathlib import Path

VLLM_ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")
TARGET = VLLM_ROOT / "models/deepseek_v4/common/ops/fused_qk_rmsnorm.py"


def main() -> None:
    src = TARGET.read_text()
    if "Q_BLOCK" in src and "_rmsnorm_row" in src:
        print("[skip] hotfix already applied")
        return

    # 1. Insert the reusable _rmsnorm_row helper before the kernel.
    old1 = "@triton.jit\ndef _fused_q_kv_rmsnorm_kernel("
    new1 = """@triton.jit
def _rmsnorm_row(
    row_in,
    weight_ptr,
    row_out,
    eps,
    SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # fp32 throughout, single cast at store. BLOCK is next_pow2(SIZE) so narrow
    # Q rows avoid KV-wide reductions that waste ALU on masked lanes.
    block = tl.arange(0, BLOCK)
    mask = block < SIZE
    x = tl.load(row_in + block, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / SIZE
    rrms = tl.rsqrt(variance + eps)
    w = tl.load(weight_ptr + block, mask=mask, other=0.0).to(tl.float32)
    y = x * rrms * w
    tl.store(row_out + block, y.to(row_out.dtype.element_ty), mask=mask)


@triton.jit
def _fused_q_kv_rmsnorm_kernel("""
    if old1 not in src:
        raise SystemExit("[ERR] anchor1 not found")
    src = src.replace(old1, new1, 1)

    # 2. Split the single BLOCK_SIZE into Q_BLOCK/KV_BLOCK.
    old2 = "    Q_SIZE: tl.constexpr,\n    KV_SIZE: tl.constexpr,\n    BLOCK_SIZE: tl.constexpr,\n"
    new2 = "    Q_SIZE: tl.constexpr,\n    KV_SIZE: tl.constexpr,\n    Q_BLOCK: tl.constexpr,\n    KV_BLOCK: tl.constexpr,\n"
    if old2 not in src:
        raise SystemExit("[ERR] anchor2 not found")
    src = src.replace(old2, new2, 1)

    # 3. Replace the kernel body with per-task _rmsnorm_row calls.
    old3 = """    pid_task = tl.program_id(1)

    if pid_task == 0:
        SIZE = Q_SIZE
        row_in = q_ptr + token_idx * q_in_stride
        weight_ptr = q_weight_ptr
        row_out = q_out_ptr + token_idx * q_out_stride
    else:
        SIZE = KV_SIZE
        row_in = kv_ptr + token_idx * kv_in_stride
        weight_ptr = kv_weight_ptr
        row_out = kv_out_ptr + token_idx * kv_out_stride

    # RMSNorm in fp32 throughout — matches csrc/layernorm_kernels.cu's
    # `(scalar_t)(x * s_variance * w)` and DeepseekV4's compressor kernel, which
    # keep x, rrms, and w all in fp32 and perform a single cast at store.
    block = tl.arange(0, BLOCK_SIZE)
    mask = block < SIZE
    x = tl.load(row_in + block, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / SIZE
    rrms = tl.rsqrt(variance + eps)
    w = tl.load(weight_ptr + block, mask=mask, other=0.0).to(tl.float32)
    y = x * rrms * w
    tl.store(row_out + block, y.to(row_out.dtype.element_ty), mask=mask)"""
    new3 = """    pid_task = tl.program_id(1)

    if pid_task == 0:
        _rmsnorm_row(
            q_ptr + token_idx * q_in_stride,
            q_weight_ptr,
            q_out_ptr + token_idx * q_out_stride,
            eps,
            Q_SIZE,
            Q_BLOCK,
        )
    else:
        _rmsnorm_row(
            kv_ptr + token_idx * kv_in_stride,
            kv_weight_ptr,
            kv_out_ptr + token_idx * kv_out_stride,
            eps,
            KV_SIZE,
            KV_BLOCK,
        )"""
    if old3 not in src:
        raise SystemExit("[ERR] anchor3 not found")
    src = src.replace(old3, new3, 1)

    # 4. Host-side block split.
    old4 = "    block_size = triton.next_power_of_2(max(q_size, kv_size))"
    new4 = "    q_block = triton.next_power_of_2(q_size)\n    kv_block = triton.next_power_of_2(kv_size)"
    if old4 not in src:
        raise SystemExit("[ERR] anchor4 not found")
    src = src.replace(old4, new4, 1)

    # 5. Launch args.
    old5 = "        BLOCK_SIZE=block_size,"
    new5 = "        Q_BLOCK=q_block,\n        KV_BLOCK=kv_block,"
    if old5 not in src:
        raise SystemExit("[ERR] anchor5 not found")
    src = src.replace(old5, new5, 1)

    TARGET.write_text(src)
    print(f"[OK]   Q/KV split blocks applied in {TARGET}")


if __name__ == "__main__":
    main()
