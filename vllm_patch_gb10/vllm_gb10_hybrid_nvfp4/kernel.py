"""Experimental hybrid NVFP4 linear kernel for vLLM on GB10/sm_121.

Rationale:
  * Small M, usually decode: Marlin W4A16 often wins because the operation is
    latency/bandwidth-bound and avoids W4A4 activation quant overhead.
  * Large M, usually prefill: CUTLASS W4A4 can win if the stack exposes a fast
    native/FP4-capable path for sm_121.

This code composes stock vLLM kernels. It does not add CUDA kernels. If your
vLLM/CUDA/FlashInfer build does not support the CUTLASS NVFP4 path on GB10,
this plugin will fail fast rather than silently claiming a speedup.
"""

from __future__ import annotations

import os

import torch

from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
from vllm.model_executor.kernels.linear.nvfp4.base import (
    NvFp4LinearKernel,
    NvFp4LinearLayerConfig,
)
from vllm.model_executor.kernels.linear.nvfp4.cutlass import CutlassNvFp4LinearKernel
from vllm.model_executor.kernels.linear.nvfp4.marlin import MarlinNvFp4LinearKernel
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    slice_nvfp4_output,
)
from vllm.utils.torch_utils import direct_register_custom_op

# Default copied from the sm_120 crossover reported by community benchmarks.
# Re-benchmark on GB10. Try 64, 128, 256, 512.
M_THRESHOLD = int(os.environ.get("GB10_HYBRID_NVFP4_M_THRESHOLD", "128"))
ALLOW_NON_GB10 = os.environ.get("GB10_HYBRID_NVFP4_ALLOW_NON_GB10", "0") == "1"


def _check_device() -> tuple[bool, str | None]:
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    major, minor = torch.cuda.get_device_capability()
    cc = major * 10 + minor
    if cc == 121 or ALLOW_NON_GB10:
        return True, None
    return False, f"expected sm_121 / compute capability 12.1, got {major}.{minor}"


def _hybrid_nvfp4_gemm(
    x: torch.Tensor,
    marlin_weight: torch.Tensor,
    marlin_weight_scale: torch.Tensor,
    marlin_weight_global_scale: torch.Tensor,
    marlin_workspace: torch.Tensor,
    cutlass_weight: torch.Tensor,
    cutlass_weight_scale: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    alpha: torch.Tensor,
    size_n: int,
    size_k: int,
    weights_padding_bytes: int,
) -> torch.Tensor:
    # M is the flattened token dimension before the last hidden dim.
    num_tokens = x.numel() // x.shape[-1]

    if num_tokens < M_THRESHOLD:
        return apply_fp4_marlin_linear(
            input=x,
            weight=marlin_weight,
            weight_scale=marlin_weight_scale,
            weight_global_scale=marlin_weight_global_scale,
            workspace=marlin_workspace,
            size_n=size_n,
            size_k=size_k,
            bias=None,
        )

    output_shape = [*x.shape[:-1], size_n]
    x_fp4, x_blockscale = scaled_fp4_quant(
        x,
        input_global_scale_inv,
        is_sf_swizzled_layout=True,
        backend="cutlass",
        padded_n=x.shape[-1] + weights_padding_bytes * 2,
    )
    out = cutlass_scaled_fp4_mm(
        x_fp4,
        cutlass_weight,
        x_blockscale,
        cutlass_weight_scale,
        alpha,
        x.dtype,
    )
    out = slice_nvfp4_output(out, size_n)
    return out.view(*output_shape)


def _hybrid_nvfp4_gemm_fake(
    x: torch.Tensor,
    marlin_weight: torch.Tensor,
    marlin_weight_scale: torch.Tensor,
    marlin_weight_global_scale: torch.Tensor,
    marlin_workspace: torch.Tensor,
    cutlass_weight: torch.Tensor,
    cutlass_weight_scale: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    alpha: torch.Tensor,
    size_n: int,
    size_k: int,
    weights_padding_bytes: int,
) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], size_n), dtype=x.dtype, device=x.device)


direct_register_custom_op(
    op_name="gb10_hybrid_nvfp4_gemm",
    op_func=_hybrid_nvfp4_gemm,
    mutates_args=["marlin_workspace"],
    fake_impl=_hybrid_nvfp4_gemm_fake,
)


class Gb10HybridNvFp4LinearKernel(NvFp4LinearKernel):
    """Marlin W4A16 for small M, CUTLASS W4A4 for large M."""

    @classmethod
    def is_supported(
        cls,
        compute_capability: int | None = None,
    ) -> tuple[bool, str | None]:
        device_ok, device_why = _check_device()
        if not device_ok:
            return False, device_why

        ok_c, why_c = CutlassNvFp4LinearKernel.is_supported(compute_capability)
        ok_m, why_m = MarlinNvFp4LinearKernel.is_supported(compute_capability)
        if ok_c and ok_m:
            return True, None
        return False, f"cutlass: {why_c}; marlin: {why_m}"

    @classmethod
    def can_implement(
        cls,
        config: NvFp4LinearLayerConfig,
    ) -> tuple[bool, str | None]:
        return True, None

    def __init__(self, config: NvFp4LinearLayerConfig) -> None:
        super().__init__(config)
        self._cutlass = CutlassNvFp4LinearKernel(config)
        self._marlin = MarlinNvFp4LinearKernel(config)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Marlin repacking mutates weight tensors, so prepare a clone/shim for
        # the Marlin path, then let CUTLASS transform the real layer tensors.
        shim = torch.nn.Module()
        shim.weight = torch.nn.Parameter(layer.weight.data.clone(), requires_grad=False)
        shim.weight_scale = torch.nn.Parameter(
            layer.weight_scale.data.clone(), requires_grad=False
        )
        shim.weight_global_scale = torch.nn.Parameter(
            layer.weight_global_scale.data.clone(), requires_grad=False
        )
        shim.output_size_per_partition = layer.output_size_per_partition
        shim.input_size_per_partition = layer.input_size_per_partition
        shim.params_dtype = layer.params_dtype

        self._marlin.process_weights_after_loading(shim)
        layer.gb10_marlin_shim = shim

        self._cutlass.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shim = layer.gb10_marlin_shim
        out = torch.ops.vllm.gb10_hybrid_nvfp4_gemm(
            x,
            shim.weight,
            shim.weight_scale,
            shim.weight_global_scale,
            shim.workspace,
            layer.weight,
            layer.weight_scale,
            layer.input_global_scale_inv,
            layer.alpha,
            layer.output_size_per_partition,
            layer.input_size_per_partition,
            getattr(layer, "weights_padding_cols", 0),
        )
        if bias is not None:
            out = out + bias
        return out
