#!/usr/bin/env python3
"""In-container numerical probe for the patched DeepSeek V4 decoder layer."""
from __future__ import annotations

import importlib.util
import os

import torch
import torch.nn as nn

MODEL = "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/model.py"
DIRECTION = os.environ["DSV4_ABLATE_FILE"]

spec = importlib.util.spec_from_file_location("dsv4_ablation_selftest_model", MODEL)
assert spec and spec.loader
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)

assert model._DSV4_ABLATE_FILE == DIRECTION
assert model._DSV4_ABLATE_LAMBDA == 3.5
assert (model._DSV4_ABLATE_LAYER_LO, model._DSV4_ABLATE_LAYER_HI) == (10, 42)
layer_cls = model.DeepseekV4DecoderLayer


def make(prefix: str):
    layer = nn.Module()
    layer.hidden_size = 4096
    layer._ablate_lambda = 0.0
    layer.register_buffer(
        "_refusal_dir", torch.zeros(4096, dtype=torch.float32), persistent=False
    )
    layer_cls._init_refusal_ablation(layer, prefix)
    return layer


selected = make("model.layers.15")
direction = selected._refusal_dir
assert direction.shape == (4096,)
assert direction.dtype == torch.float32
assert abs(direction.norm().item() - 1.0) < 1e-5
assert selected._ablate_lambda == 3.5
assert "_refusal_dir" not in selected.state_dict()

for prefix, enabled in (
    ("model.layers.5", False),
    ("model.layers.42", True),
    ("model.layers.43", False),
):
    layer = make(prefix)
    actual = layer._ablate_lambda > 0 and layer._refusal_dir.norm().item() > 0.5
    assert actual is enabled, (prefix, actual, enabled)

x32 = torch.randn(7, 4096, dtype=torch.float32)
y32 = layer_cls._ablate_refusal_direction(selected, x32)
error32 = (y32 @ direction + 2.5 * (x32 @ direction)).abs().max().item()
assert error32 < 1e-5, error32

xbf16 = torch.randn(7, 4096, dtype=torch.bfloat16)
ybf16 = layer_cls._ablate_refusal_direction(selected, xbf16)
error_bf16 = (
    ybf16.float() @ direction + 2.5 * (xbf16.float() @ direction)
).abs().max().item()
assert error_bf16 < 5e-2, error_bf16

disabled = make("model.layers.5")
assert torch.equal(layer_cls._ablate_refusal_direction(disabled, xbf16), xbf16)
print(
    "runtime ablation selftest passed: "
    f"fp32_error={error32:.2e} bf16_error={error_bf16:.2e}"
)
