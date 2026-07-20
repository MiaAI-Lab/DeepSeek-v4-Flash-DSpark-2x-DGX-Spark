#!/usr/bin/env python3
"""Fail-closed static/runtime audit for the DSpark-r0b0tlab image."""
from __future__ import annotations

import importlib.metadata as metadata
import json
from pathlib import Path
import sys
from typing import NoReturn


def fail(message: str) -> NoReturn:
    print(f"DSPARK_RUNTIME_AUDIT_FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


manifest_path = Path("/opt/r0b0tlab/runtime-manifest.json")
if not manifest_path.is_file():
    fail("runtime manifest is missing")
try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except Exception as exc:
    fail(f"runtime manifest is invalid: {exc}")

if manifest.get("kv_cache_dtype") != "nvfp4_ds_mla":
    fail("production manifest must require nvfp4_ds_mla")
if manifest.get("speculative_method") != "dspark":
    fail("production manifest must require DSpark speculation")

try:
    import torch
    import vllm
except Exception as exc:
    fail(f"runtime imports failed: {exc}")

expected_cuda = manifest.get("cuda_version")
if torch.version.cuda != expected_cuda:
    fail(f"expected PyTorch CUDA {expected_cuda}, got {torch.version.cuda!r}")
if not str(torch.__version__).startswith("2.11.0"):
    fail(f"expected PyTorch 2.11.0, got {torch.__version__!r}")
try:
    flashinfer = metadata.version("flashinfer-python")
except metadata.PackageNotFoundError:
    fail("flashinfer-python is not installed")

root = Path(vllm.__file__).resolve().parent
required_sources = {
    "DSpark speculator": root / "v1/worker/gpu/spec_decode/dspark/speculator.py",
    "DeepSeek V4 draft model": root / "models/deepseek_v4/nvidia/dspark.py",
}
for label, path in required_sources.items():
    if not path.is_file():
        fail(f"{label} source is missing at {path}")

source_text = "\n".join(path.read_text(errors="replace") for path in required_sources.values())
for marker in ("DSparkSpeculator", "DSparkDeepseekV4", "context_slot_mappings"):
    if marker not in source_text:
        fail(f"required DSpark source marker is absent: {marker}")

expected_versions = {
    "vllm": manifest.get("vllm_version"),
    "torch": manifest.get("torch_version"),
    "flashinfer": manifest.get("flashinfer_version"),
}
actual_versions = {
    "vllm": vllm.__version__,
    "torch": torch.__version__,
    "flashinfer": flashinfer,
}
for component, expected in expected_versions.items():
    if actual_versions[component] != expected:
        fail(
            f"{component} version mismatch: expected {expected!r}, "
            f"got {actual_versions[component]!r}"
        )

if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability()
    if capability != (12, 1):
        fail(f"expected SM121 capability (12, 1), got {capability}")
    device = torch.cuda.get_device_name()
else:
    capability = None
    device = None

print(json.dumps({
    "status": "PASS",
    "vllm": vllm.__version__,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "flashinfer": flashinfer,
    "device": device,
    "capability": capability,
    "kv_cache_dtype": manifest["kv_cache_dtype"],
    "speculative_method": manifest["speculative_method"],
}, sort_keys=True))
print("DSPARK_RUNTIME_AUDIT_PASS")
