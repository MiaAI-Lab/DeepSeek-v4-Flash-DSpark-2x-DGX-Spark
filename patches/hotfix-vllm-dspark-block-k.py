#!/usr/bin/env python3
"""DSpark block-k unlock: let ``num_speculative_tokens`` follow the trained
DSpark block size instead of the MTP module-reuse divisibility rule (opt-in).

``SpeculativeConfig.__post_init__`` in the pinned Anemll vLLM rejects
``num_speculative_tokens > n_predict`` unless it is a multiple of ``n_predict``
("Ensure divisibility for MTP module reuse").  That rule exists for stock
DeepSeek MTP, where one MTP layer is re-run per speculative step.  DSpark is a
different drafter: its ``n_predict`` (= ``num_nextn_predict_layers``) stages are
*stacked* into one non-causal draft backbone that predicts every position of a
``dspark_block_size`` block in a single parallel pass (see the checkpoint's
``inference/model.py::DSparkBlock``), so the number of stages says nothing about
how many tokens may be drafted.  DeepSeek-V4-Flash-Vision-Exp ships
``num_nextn_predict_layers=3`` with ``dspark_block_size=5``; the rule forces
k=6 there (0731 had one stage and booted k=5), i.e. one more noise query than
the block was trained with.

The transform adds ``self.method != "dspark"`` to that one condition.  Nothing
else changes: k must still be >= 1, DSpark's own speculator sizes its buffers
from ``num_speculative_tokens`` exactly as before, and the launcher keeps its
own ``MTP_NUM_TOKENS`` sanity check.

The opt-in Compose gate runs this before ``vllm serve``.  It accepts only the
pinned Anemll 0.1.1 vLLM version and the exact stock identity of
``config/speculative.py`` (no other recipe hotfix touches that file).  Applying
is one same-directory atomic replace; an already-patched target is verified but
never rewritten.
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
    "/usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py"
)
EXPECTED_VLLM_VERSION = "0.25.2.dev0+g752a3a504.d20260714"

STOCK_SHA256 = "3f1abd1ca3042fba239e7bf98b08f645f3e950c16ab510fbc99a49c5c507721f"
STOCK_SIZE = 56_845
PATCHED_SHA256 = "7fffe035bf28f30fb66d6cb38e30b759730ae1da111f2c183558e233b67bc235"
PATCHED_SIZE = 56987
MARK = "# [dspark-block-k] stacked DSpark stages are not re-used per step"

REGION_OLD = b'''                    elif (
                        self.num_speculative_tokens > n_predict
                        and self.num_speculative_tokens % n_predict != 0
                    ):
                        # Ensure divisibility for MTP module reuse.
                        raise ValueError(
'''
REGION_NEW = b'''                    elif (
                        self.method != "dspark"
                        # [dspark-block-k] stacked DSpark stages are not re-used per step
                        and self.num_speculative_tokens > n_predict
                        and self.num_speculative_tokens % n_predict != 0
                    ):
                        # Ensure divisibility for MTP module reuse.
                        raise ValueError(
'''


class HotfixError(RuntimeError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def transform(stock: bytes) -> bytes:
    """Stock bytes -> patched bytes; refuses anything but exactly one site."""
    if stock.count(REGION_OLD) != 1:
        raise HotfixError("divisibility region not found exactly once")
    if MARK.encode() in stock:
        raise HotfixError("target already carries the block-k mark")
    patched = stock.replace(REGION_OLD, REGION_NEW, 1)
    compile(patched, "speculative.py", "exec")
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
    fd, tmp_name = tempfile.mkstemp(prefix=".dspark-block-k-", dir=str(target.parent))
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
            print(f"dspark-block-k: {state} ({args.target})")
            return 0
        outcome = apply(args.target)
        print(f"dspark-block-k: {outcome} ({args.target})")
        return 0
    except HotfixError as error:
        print(f"dspark-block-k: FAIL-CLOSED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
