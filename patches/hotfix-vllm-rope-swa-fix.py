#!/usr/bin/env python3
"""RoPE SWA fix: sparse-SWA layers use plain RoPE, not YaRN (opt-in).

Port of merged upstream vllm-project/vllm#54815.  The pinned Anemll vLLM builds
every DeepseekV4 rotary embedding through ``build_deepseek_v4_rope`` in
``models/deepseek_v4/common/rope.py``, which promotes the checkpoint's rope
type to ``deepseek_yarn`` whenever it is not ``"default"`` — for *every* layer.
Per the DeepSeek-V4 reference implementation (``inference/model.py`` L481-485)
and transformers#45892, YaRN long-context scaling belongs only to compressor
(CSA/HCA) layers; sliding-window layers must use plain RoPE.

The served Vision-Exp abliterated checkpoint ships a flat
``rope_scaling = {type: yarn, factor: 16, original_max_position_embeddings:
65536}`` with ``sliding_window=128``, so its sparse-SWA layers — layers 0 and 1
(``compress_ratios[i] = 0`` → ``compress_ratio = 1``) plus the three DSpark
drafter layers past ``num_hidden_layers`` — get YaRN factor=16 today over a
128-token window.

The transform replaces the rope-parameter block with the upstream post-image:
each call now works on a per-layer dict copy (nested ``{"main", "compress"}``
checkpoints route by layer type; flat ones are copied), the YaRN promotion
additionally requires ``compress_ratio > 1``, and every other layer takes
``deepseek_yarn`` with ``factor=1.0`` over ``max_position_embeddings`` — the
identity scaling, i.e. plain RoPE on the same kernel path.  Compressor layers
keep byte-for-byte the parameters they resolve today.

The opt-in Compose gate runs this before ``vllm serve``.  It accepts only the
pinned Anemll 0.1.1 vLLM version and the exact stock identity of
``models/deepseek_v4/common/rope.py`` (no other recipe hotfix touches that
file).  Applying is one same-directory atomic replace; an already-patched
target is verified but never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
import stat
import sys
import tempfile
from pathlib import Path

PRODUCTION_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/common/rope.py"
)
EXPECTED_VLLM_VERSION = "0.25.2.dev0+g752a3a504.d20260714"

STOCK_SHA256 = "0074271a6f26ce1538e0df3ae66eae30ff61d4d50ad0aa73ce0c53c4182670e7"
STOCK_SIZE = 1_242
PATCHED_SHA256 = "6452ce2eb94d69d2b24344dfdad8bb55d13dca1969067f9a26fb83c4cb066482"
PATCHED_SIZE = 2_111
MARK = "# [dspark-rope-swa] upstream vllm#54815: sparse-SWA layers use plain RoPE."

REGION_OLD = b'''    rope_parameters = config.rope_parameters
    rope_parameters["rope_theta"] = (
        config.compress_rope_theta if compress_ratio > 1 else config.rope_theta
    )
    if rope_parameters["rope_type"] != "default":
        rope_parameters["rope_type"] = (
            "deepseek_yarn"
            if rope_parameters.get("apply_yarn_scaling", True)
            else "deepseek_llama_scaling"
        )
'''
REGION_NEW = b'''    rope_parameters = config.rope_parameters
    # [dspark-rope-swa] upstream vllm#54815: sparse-SWA layers use plain RoPE.
    # Newer checkpoints nest per-layer-type rope dicts ({"main", "compress"});
    # older ones ship a single flat dict shared by all layer types.
    if isinstance(rope_parameters.get("main"), dict) and isinstance(
        rope_parameters.get("compress"), dict
    ):
        key = "compress" if compress_ratio > 1 else "main"
        rope_parameters = dict(rope_parameters[key])
    else:
        rope_parameters = dict(rope_parameters)

    rope_parameters["rope_theta"] = (
        config.compress_rope_theta if compress_ratio > 1 else config.rope_theta
    )
    if compress_ratio > 1 and rope_parameters["rope_type"] != "default":
        # YaRN applies only to compressor (CSA/HCA) layers.
        rope_parameters["rope_type"] = (
            "deepseek_yarn"
            if rope_parameters.get("apply_yarn_scaling", True)
            else "deepseek_llama_scaling"
        )
    else:
        # Sliding-window layers use plain RoPE (theta=rope_theta, no YaRN).
        rope_parameters["rope_type"] = "deepseek_yarn"
        rope_parameters["factor"] = 1.0
        rope_parameters["original_max_position_embeddings"] = max_position_embeddings
'''


class HotfixError(RuntimeError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def transform(stock: bytes) -> bytes:
    """Stock bytes -> patched bytes; refuses anything but exactly one site."""
    if stock.count(REGION_OLD) != 1:
        raise HotfixError("rope-parameter region not found exactly once")
    if MARK.encode() in stock:
        raise HotfixError("target already carries the rope-swa mark")
    patched = stock.replace(REGION_OLD, REGION_NEW, 1)
    compile(patched, "rope.py", "exec")
    return patched


def _vllm_version(provider=importlib.metadata.version) -> str:
    try:
        version = provider("vllm")
    except importlib.metadata.PackageNotFoundError as error:
        raise HotfixError("vllm is not installed") from error
    if version != EXPECTED_VLLM_VERSION:
        raise HotfixError(
            f"unsupported vllm version {version!r}; expected {EXPECTED_VLLM_VERSION!r}"
        )
    return version


def inspect(target: Path, *, provider=importlib.metadata.version) -> tuple[str, bytes]:
    _vllm_version(provider)
    try:
        st = target.lstat()
    except FileNotFoundError:
        raise HotfixError("target is missing")
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise HotfixError("target is not a regular file")
    data = target.read_bytes()
    digest = _sha256(data)
    if digest == PATCHED_SHA256 and len(data) == PATCHED_SIZE:
        return "patched", data
    if digest == STOCK_SHA256 and len(data) == STOCK_SIZE:
        return "stock", data
    raise HotfixError(
        f"unsupported target bytes sha256={digest} size={len(data)}; "
        "expected the pinned stock or patched identity"
    )


def apply(target: Path, *, provider=importlib.metadata.version) -> str:
    state, data = inspect(target, provider=provider)
    if state == "patched":
        return "already-patched"
    patched = transform(data)
    if _sha256(patched) != PATCHED_SHA256 or len(patched) != PATCHED_SIZE:
        raise HotfixError("transformed bytes do not match the pinned patched identity")
    fd, tmp_name = tempfile.mkstemp(prefix=".dspark-rope-swa-", dir=str(target.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(patched)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    verify_state, _ = inspect(target, provider=provider)
    if verify_state != "patched":
        raise HotfixError("post-apply verification failed")
    return "applied"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify compatibility only")
    parser.add_argument("--status", action="store_true", help="print the target state")
    parser.add_argument("--target", type=Path, default=PRODUCTION_TARGET)
    args = parser.parse_args(argv)
    try:
        if args.check or args.status:
            state, _ = inspect(args.target)
            print(f"dspark-rope-swa: {state} ({args.target})")
            return 0
        outcome = apply(args.target)
        print(f"dspark-rope-swa: {outcome} ({args.target})")
        return 0
    except HotfixError as error:
        print(f"dspark-rope-swa: FAIL-CLOSED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
