"""vLLM general plugin for experimental hybrid NVFP4 linear kernels on GB10/sm_121.

The plugin registers the opt-in quantization method `modelopt_gb10_hybrid`.
It does not change normal vLLM behavior unless `--quantization modelopt_gb10_hybrid`
is passed.
"""

_registered = False


def register() -> None:
    """Entry point called by vLLM's plugin loader. Must be re-entrant."""
    global _registered
    if _registered:
        return

    # Importing config runs the @register_quantization_config decorator.
    from . import config  # noqa: F401

    _registered = True
