#!/usr/bin/env python3
"""DSpark draft SWA prefix fix: always recompute the draft's sliding window on
a prefix-cache hit (opt-in port of Anemll/dspark-vllm-gx10#2, ``4afc5e7eeb``).

The DSpark draft model attends over a sliding window of ``sliding_window``
(128) tokens, populated from the target model's hidden states via
``precompute_and_store_context_kv``.  On a prefix-cache hit the target skips
recomputing the cached prefix, so the draft's sliding-window cache is missing
the prefix, the draft degenerates, and the verifier accepts a truncated
response on repeated identical prompts (deterministic at temperature 0).

The port is upstream-verbatim plus one ``# [dspark-swa-prefix]`` mark line per
file: ``kv_cache_manager.py`` gains a ``dspark_window_size`` parameter that
caps ``max_cache_hit_length`` to ``num_tokens - 1 - dspark_window_size`` in
``get_computed_blocks`` so the target always recomputes the last window, and
``sched/scheduler.py`` reads the draft model's ``hf_config.sliding_window``
when ``use_dspark()`` and passes it to the ``KVCacheManager``.  Without DSpark
(or with the gate off) ``dspark_window_size`` stays ``None`` and the cache-hit
arithmetic is unchanged.

``kv_cache_manager.py`` is touched by no other recipe hotfix, so it is pinned
by whole-file identity (stock and patched sha256, block-k style).  The
scheduler is co-owned at boot by ``hotfix-dsv4-grammar-advance.sh``,
``hotfix-vllm-empty-encoder-output.py``, issue #27 and the opt-in scheduler
patchers, so — like ``hotfix-vllm-empty-encoder-output.py`` — it is held to
source-exact regions that must each occur exactly once, and its whole-file
identities are asserted only for the pure stock->patched transform (proven
against the fixtures by ``scripts/test-dspark-swa-prefix.py``).  Both targets
preflight before either is written; publication is one same-directory atomic
replace per file, and an already-patched target is verified, never rewritten.
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
from typing import NamedTuple

PRODUCTION_KV_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_manager.py"
)
PRODUCTION_SCHED_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py"
)
EXPECTED_VLLM_VERSION = "0.25.2.dev0+g752a3a504.d20260714"
MARK = "# [dspark-swa-prefix]"

KV_SIG_OLD = b"""        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
"""
KV_SIG_NEW = b"""        enable_caching: bool = True,
        use_eagle: bool = False,
        dspark_window_size: int | None = None,
        log_stats: bool = False,
"""
KV_ATTR_OLD = b"""        self.enable_caching = enable_caching
        self.use_eagle = use_eagle
        self.log_stats = log_stats
"""
KV_ATTR_NEW = b"""        self.enable_caching = enable_caching
        self.use_eagle = use_eagle
        self.dspark_window_size = dspark_window_size
        self.log_stats = log_stats
"""
KV_HIT_OLD = b"""        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens = (
"""
KV_HIT_NEW = b"""        max_cache_hit_length = request.num_tokens - 1
        # [dspark-swa-prefix]
        # DSpark fix: the draft model attends over a sliding window of
        # `dspark_window_size` tokens, populated from the target's hidden
        # states via `precompute_and_store_context_kv`. On a prefix-cache hit
        # the target skips computing the cached prefix, so the draft's SWA
        # cache would be missing the window and the draft degenerates. Force
        # the target to always recompute the last `dspark_window_size` tokens
        # so the draft always has its full window populated. The cost is a
        # small recompute (128 tokens) relative to the cached prefix.
        if self.dspark_window_size is not None and self.dspark_window_size > 0:
            max_cache_hit_length = max(
                request.num_tokens - 1 - self.dspark_window_size, 0
            )
        computed_blocks, num_new_computed_tokens = (
"""
SCHED_WINDOW_OLD = b"""            if speculative_config.use_dspark():
                # DSpark drafts a block of num_spec_tokens query tokens in which the
                # anchor itself is the first prediction position (no separate bonus
                # query), so it needs exactly num_spec_tokens lookahead slots.
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
"""
SCHED_WINDOW_NEW = b"""            if speculative_config.use_dspark():
                # DSpark drafts a block of num_spec_tokens query tokens in which the
                # anchor itself is the first prediction position (no separate bonus
                # query), so it needs exactly num_spec_tokens lookahead slots.
                self.num_lookahead_tokens = self.num_spec_tokens

        # [dspark-swa-prefix]
        # DSpark fix: the draft model attends over a sliding window of
        # `dspark_window_size` tokens. On a prefix-cache hit the target skips
        # computing the cached prefix, so the draft's SWA cache would be
        # missing the window and the draft degenerates. We force the target to
        # always recompute the last `dspark_window_size` tokens (see
        # KVCacheManager.get_computed_blocks). Read the window size from the
        # draft model's HF config.
        self.dspark_window_size: int | None = None
        if speculative_config is not None and speculative_config.use_dspark():
            draft_cfg = getattr(speculative_config, "draft_model_config", None)
            hf_cfg = getattr(draft_cfg, "hf_config", None)
            window = getattr(hf_cfg, "sliding_window", None)
            if window is not None and window > 0:
                self.dspark_window_size = int(window)

        # Create the KV cache manager.
"""
SCHED_CTOR_OLD = b"""            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
"""
SCHED_CTOR_NEW = b"""            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            dspark_window_size=self.dspark_window_size,
            log_stats=self.log_stats,
"""


class Spec(NamedTuple):
    """One patch target: its regions, identity pins, and pinning policy."""

    label: str
    regions: tuple[tuple[bytes, bytes], ...]
    stock_sha256: str
    stock_size: int
    patched_sha256: str
    patched_size: int
    identity_pinned: bool  # sole-owned file: refuse any non-pinned identity


KV = Spec(
    label="kv_cache_manager.py",
    regions=(
        (KV_SIG_OLD, KV_SIG_NEW),
        (KV_ATTR_OLD, KV_ATTR_NEW),
        (KV_HIT_OLD, KV_HIT_NEW),
    ),
    stock_sha256="be9c50918c8cd01736102de4020e2a9a8650675bcd4eb4227d40d2bede6bb853",
    stock_size=27_482,
    patched_sha256="09f0e9905ccd1d93e086caea8df1990b82afda30e2ba9d3d5059b921b9637af1",
    patched_size=28_412,
    identity_pinned=True,
)
SCHED = Spec(
    label="scheduler.py",
    regions=(
        (SCHED_WINDOW_OLD, SCHED_WINDOW_NEW),
        (SCHED_CTOR_OLD, SCHED_CTOR_NEW),
    ),
    stock_sha256="e25d4c9a95abdbe8e516714ed02574d929ca0d5e8c11c4cc73b84d3a3b905443",
    stock_size=125_101,
    patched_sha256="69fc81185d7970b5e7cbfda856e23b0aa911dada96d4cab050b1213c3c86dcdc",
    patched_size=126_104,
    identity_pinned=False,
)


class HotfixError(RuntimeError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def classify(spec: Spec, data: bytes) -> str:
    """Return "stock" or "patched" from region counts; anything else fails."""
    old_counts = tuple(data.count(old) for old, _ in spec.regions)
    new_counts = tuple(data.count(new) for _, new in spec.regions)
    if old_counts == (0,) * len(spec.regions) and new_counts == (1,) * len(
        spec.regions
    ):
        return "patched"
    if old_counts == (1,) * len(spec.regions) and new_counts == (0,) * len(
        spec.regions
    ):
        return "stock"
    raise HotfixError(
        f"{spec.label}: region drift old={old_counts} new={new_counts}; "
        "expected every region exactly once"
    )


def transform(spec: Spec, stock: bytes) -> bytes:
    """Stock bytes -> patched bytes; refuses anything but exactly one site each."""
    if MARK.encode() in stock:
        raise HotfixError(f"{spec.label}: target already carries the SWA-prefix mark")
    patched = stock
    for old, new in spec.regions:
        if patched.count(old) != 1:
            raise HotfixError(f"{spec.label}: region not found exactly once")
        patched = patched.replace(old, new, 1)
    compile(patched, spec.label, "exec")
    if classify(spec, patched) != "patched":
        raise HotfixError(f"{spec.label}: transformed bytes fail classification")
    if _sha256(stock) == spec.stock_sha256:
        # Pure stock preimage: the postimage must match the pinned identity.
        if _sha256(patched) != spec.patched_sha256 or len(patched) != spec.patched_size:
            raise HotfixError(
                f"{spec.label}: transform of pinned stock does not match the "
                "pinned patched identity"
            )
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


def inspect(spec: Spec, target: Path, *, provider=importlib.metadata.version) -> tuple[str, bytes]:
    _vllm_version(provider)
    try:
        st = target.lstat()
    except FileNotFoundError:
        raise HotfixError(f"{spec.label}: target is missing")
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise HotfixError(f"{spec.label}: target is not a regular file")
    data = target.read_bytes()
    if spec.identity_pinned:
        digest = _sha256(data)
        if digest == spec.patched_sha256 and len(data) == spec.patched_size:
            return "patched", data
        if digest == spec.stock_sha256 and len(data) == spec.stock_size:
            return "stock", data
        raise HotfixError(
            f"{spec.label}: unsupported target bytes sha256={digest} "
            f"size={len(data)}; expected the pinned stock or patched identity"
        )
    return classify(spec, data), data


def _publish(target: Path, patched: bytes) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=".dspark-swa-prefix-", dir=str(target.parent))
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


def apply(
    kv_target: Path, sched_target: Path, *, provider=importlib.metadata.version
) -> dict[str, str]:
    """Preflight both targets, then write only the ones still stock."""
    pairs = ((KV, kv_target), (SCHED, sched_target))
    states = {spec.label: inspect(spec, target, provider=provider) for spec, target in pairs}
    outcomes: dict[str, str] = {}
    for spec, target in pairs:
        state, data = states[spec.label]
        if state == "patched":
            outcomes[spec.label] = "already-patched"
            continue
        _publish(target, transform(spec, data))
        verify_state, _ = inspect(spec, target, provider=provider)
        if verify_state != "patched":
            raise HotfixError(f"{spec.label}: post-apply verification failed")
        outcomes[spec.label] = "applied"
    return outcomes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify compatibility only")
    parser.add_argument("--status", action="store_true", help="print the target states")
    parser.add_argument("--kv-target", type=Path, default=PRODUCTION_KV_TARGET)
    parser.add_argument("--sched-target", type=Path, default=PRODUCTION_SCHED_TARGET)
    args = parser.parse_args(argv)
    pairs = ((KV, args.kv_target), (SCHED, args.sched_target))
    try:
        if args.check or args.status:
            report = ", ".join(
                f"{spec.label}={inspect(spec, target)[0]} ({target})"
                for spec, target in pairs
            )
        else:
            outcomes = apply(args.kv_target, args.sched_target)
            report = ", ".join(
                f"{spec.label}={outcomes[spec.label]} ({target})"
                for spec, target in pairs
            )
        print(f"dspark-swa-prefix: {report}")
        return 0
    except HotfixError as error:
        print(f"dspark-swa-prefix: FAIL-CLOSED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
