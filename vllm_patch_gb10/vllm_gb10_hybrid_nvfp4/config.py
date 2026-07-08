"""Quantization config for the GB10 hybrid NVFP4 linear kernel.

This subclasses vLLM's ModelOpt mixed precision quantization config and swaps
NVFP4 linear methods to a dispatcher that uses:
  * Marlin W4A16 for small-M decode-like calls
  * CUTLASS W4A4 for large-M prefill-like calls

It is opt-in only via `--quantization modelopt_gb10_hybrid`.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearLayerConfig
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4LinearMethod,
)

from .kernel import M_THRESHOLD, Gb10HybridNvFp4LinearKernel

logger = init_logger(__name__)


@register_quantization_config("modelopt_gb10_hybrid")
class ModelOptGb10HybridConfig(ModelOptMixedPrecisionConfig):
    """ModelOpt quantization config with hybrid NVFP4 linear dispatch."""

    def get_name(self) -> str:
        return "modelopt_gb10_hybrid"

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict,
        user_quant: str | None,
        hf_config=None,
    ) -> str | None:
        # Be conservative: never claim a model unless the user opts in.
        if user_quant != "modelopt_gb10_hybrid":
            return None

        base = ModelOptMixedPrecisionConfig.override_quantization_method(
            hf_quant_cfg,
            user_quant,
            hf_config=hf_config,
        )
        return "modelopt_gb10_hybrid" if base == "modelopt_mixed" else None

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        method = super().get_quant_method(layer, prefix)
        if isinstance(method, ModelOptNvFp4LinearMethod):
            ok, why = Gb10HybridNvFp4LinearKernel.is_supported()
            if not ok:
                raise ValueError(
                    "modelopt_gb10_hybrid was requested, but the hybrid NVFP4 "
                    f"linear kernel is unsupported in this vLLM/CUDA stack: {why}"
                )

            # This relies on vLLM internal kernel interfaces and is expected to
            # track vLLM closely. Test against your exact vLLM commit/container.
            method.kernel = Gb10HybridNvFp4LinearKernel(NvFp4LinearLayerConfig())
            logger.info_once(
                "Using Gb10HybridNvFp4LinearKernel: Marlin for M < %d tokens, "
                "CUTLASS W4A4 for M >= %d tokens",
                M_THRESHOLD,
                M_THRESHOLD,
            )
        return method
