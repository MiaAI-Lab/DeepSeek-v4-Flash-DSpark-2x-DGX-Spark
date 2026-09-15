#!/usr/bin/env python3
"""MXFP4 indexer K cache on GB10: relax the datacenter-only gate (opt-in).

The pinned Anemll vLLM ships a complete MXFP4 Lightning-indexer K cache
(``AttentionConfig.use_fp4_indexer_cache``: Triton insert writes packed FP4 K,
DeepGEMM ``fp8_fp4_*_mqa_logits`` consume it), but the metadata builder in
``v1/attention/backends/mla/indexer.py`` asserts Blackwell *datacenter* only
("use_fp4_indexer_cache requires Blackwell datacenter GPUs (sm_10x)").  The
vendored DeepGEMM ships ``sm120_fp4_mqa_logits.cuh`` and
``sm120_fp4_paged_mqa_logits.cuh``, and the decode-path flattening right below
the gate already handles every non-SM100 family via the shared
``smxx_fp8_fp4_paged_mqa_logits`` contract, so the gate is conservative rather
than a kernel limit (docs/CLAUDE/item8-fp4-kv-design.md §3).

The transform widens that one assert to also accept the consumer-Blackwell
family (``is_device_capability_family(120)`` — GB10 is SM121) and nothing
else.  Everything downstream simply follows the flag: the indexer allocates
the same 132 B/row K cache (the pinned writer keeps FP8-size allocation and
uses the first half for FP4), writes packed MXFP4, and the logits kernels
read half the bytes per key.  Enablement is a separate launch arg
(``--attention-config '{"use_fp4_indexer_cache":true}'``) emitted by the
Compose gate only when ``DSPARK_ENABLE_MXFP4_INDEXER_CACHE=1``.

The fp4 logits kernels are not in the persisted DeepGEMM JIT cache, and the
vendored DeepGEMM names kernels ``sm121_*`` on GB10 while shipping ``sm120_*``
headers only, so the launcher requires ``DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS=1``
(item8 design §5) whenever this hotfix is enabled.

The opt-in Compose gate runs this before ``vllm serve``.  It accepts only the
pinned Anemll 0.1.1 vLLM version and the exact stock identity of
``v1/attention/backends/mla/indexer.py`` (no other recipe hotfix touches that
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
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py"
)
EXPECTED_VLLM_VERSION = "0.25.2.dev0+g752a3a504.d20260714"

STOCK_SHA256 = "02505c6c29505aa0b0607be0675ed11612f600b1caa6daa8d98f2a49937abc1b"
STOCK_SIZE = 37_907
PATCHED_SHA256 = "bfb0376d8f2b4d2864281ee34d94163c96bb806caf3b8529af6dd8aceb69e9b5"
PATCHED_SIZE = 38_129
MARK = "# [dspark-mxfp4-indexer] DeepGEMM ships sm120 fp4 (paged) mqa-logits"

REGION_OLD = b'''        assert (
            current_platform.is_device_capability_family(100)
            or not self.use_fp4_indexer_cache
        ), (
            "use_fp4_indexer_cache requires Blackwell datacenter GPUs "
            "(sm_10x, e.g. B200/GB200); sm_120 (consumer Blackwell) and "
            "earlier architectures are not supported."
        )
'''
REGION_NEW = b'''        assert (
            current_platform.is_device_capability_family(100)
            # [dspark-mxfp4-indexer] DeepGEMM ships sm120 fp4 (paged) mqa-logits
            # kernels; consumer Blackwell (sm_12x, e.g. GB10) runs them too.
            or current_platform.is_device_capability_family(120)
            or not self.use_fp4_indexer_cache
        ), (
            "use_fp4_indexer_cache requires Blackwell GPUs (sm_10x "
            "datacenter, e.g. B200/GB200, or sm_12x consumer, e.g. GB10); "
            "earlier architectures are not supported."
        )
'''


class HotfixError(RuntimeError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def transform(stock: bytes) -> bytes:
    """Stock bytes -> patched bytes; refuses anything but exactly one site."""
    if stock.count(REGION_OLD) != 1:
        raise HotfixError("fp4 indexer gate region not found exactly once")
    if MARK.encode() in stock:
        raise HotfixError("target already carries the mxfp4-indexer mark")
    patched = stock.replace(REGION_OLD, REGION_NEW, 1)
    compile(patched, "indexer.py", "exec")
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
    fd, tmp_name = tempfile.mkstemp(
        prefix=".dspark-mxfp4-indexer-", dir=str(target.parent)
    )
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
            print(f"dspark-mxfp4-indexer: {state} ({args.target})")
            return 0
        outcome = apply(args.target)
        print(f"dspark-mxfp4-indexer: {outcome} ({args.target})")
        return 0
    except HotfixError as error:
        print(f"dspark-mxfp4-indexer: FAIL-CLOSED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
