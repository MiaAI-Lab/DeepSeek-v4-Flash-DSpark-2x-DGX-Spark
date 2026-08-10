"""dsv4_moonvit_vllm — native MoonViT vision plugin for DeepSeek-V4-Flash-0731."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_registered = False


def register() -> None:
    """vLLM general_plugins entry point. Re-entrant."""
    global _registered
    if _registered:
        return
    try:
        from vllm.model_executor.models.registry import ModelRegistry

        ModelRegistry.register_model(
            "DeepseekV4MoonVitForCausalLM",
            "dsv4_moonvit_vllm.model:DeepseekV4MoonVitForCausalLM",
        )
        # Import model module to attach MULTIMODAL_REGISTRY processor.
        from . import model as _model  # noqa: F401

        _registered = True
        logger.info("dsv4_moonvit plugin registered")
    except Exception as exc:
        logger.warning("dsv4_moonvit register failed: %s", exc)
        raise


__all__ = ["register"]
