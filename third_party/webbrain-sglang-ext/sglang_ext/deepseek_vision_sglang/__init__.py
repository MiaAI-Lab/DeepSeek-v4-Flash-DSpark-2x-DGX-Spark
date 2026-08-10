"""External SGLang package for the DeepSeek V4 + MoonViT adapter.

This package is intentionally standalone.  A staged Hugging Face model directory can
ship ``sglang_ext/deepseek_vision_sglang`` and expose it through ``PYTHONPATH`` without
installing the training project.
"""

SGLANG_SOURCE_COMMIT = "fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1"

__all__ = ["SGLANG_SOURCE_COMMIT"]
