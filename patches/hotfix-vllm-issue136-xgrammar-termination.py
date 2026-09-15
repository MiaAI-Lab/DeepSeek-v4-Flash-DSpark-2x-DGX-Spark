#!/usr/bin/env python3
"""Source-exact XGrammar backport chain for issues #136 + #210.

The opt-in Compose gate runs this before ``vllm serve``.  One flag applies the
chain in upstream order across TWO pinned files:

1. ``vllm/v1/structured_output/backend_xgrammar.py`` — the three vLLM #52805
   XgrammarGrammar hunks (issue #136: tokens after termination no longer
   desync the cached grammar state).
2. ``vllm/v1/structured_output/__init__.py`` — the vLLM #53046 hunk (issue
   #210: a post-reasoning-end draft window validates each speculative token
   with ``validate_tokens`` before ``accept_tokens``, so grammar-invalid
   drafts that predate the bitmask no longer trip spurious "Failed to advance
   FSM" errors).  No output corruption was demonstrated for the prior code;
   the FSM state path is correctness-sensitive and the upstream fix removes
   the desync risk class.

It accepts only the pinned Anemll 0.1.1 package versions and the exact
stock/post-patch source identities of both targets.  Both candidates are
built and compiled before either file is written; publication is a
recoverable same-directory atomic rename per file, and a failure publishing
the second file rolls the first back to its exact original bytes and
metadata.  An already-fully-patched chain is verified but never rewritten.

Chain states: ``stock`` (both files stock), ``patched`` (both files patched),
``partial`` (exactly one file patched).  A pre-chain #136-only application
(backend patched, manager stock) is a valid legacy partial: apply completes
the chain by patching the manager file only.  The reverse mix (manager
patched, backend stock) is never legitimate and is refused as incompatible.
``--status`` prints ``stock`` / ``patched`` / ``partial-invalid`` and exits 0
only when fully patched.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

EXPECTED_VLLM_VERSION = "0.25.2.dev0+g752a3a504.d20260714"
EXPECTED_XGRAMMAR_VERSION = "0.2.3"
MARK = "[issue136-xgrammar]"

BACKEND_OLD_REGION = b'''    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the FSM.

        Returns True if the FSM was advanced successfully.
        Returns False if the FSM failed to advance.
        """
        if self._is_terminated:
            return False
        for token in tokens:
            if not self.matcher.accept_token(token):
                logger.error(
                    "Failed to advance FSM for request %s "
                    "for tokens %s. Please file an issue.",
                    request_id,
                    token,
                )
                return False
            self.num_processed_tokens += 1
        self._is_terminated = self.matcher.is_terminated()
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the FSM in sequence.
        Will not advance the FSM.

        Returns the prefix list of tokens that are accepted by the FSM.
        """
        accepted_tokens = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted_tokens.append(token)
            else:
                break
        if len(accepted_tokens) > 0:
            # Rollback the FSM to the initial state
            self.matcher.rollback(len(accepted_tokens))
        return accepted_tokens

    def rollback(self, num_tokens: int) -> None:
        self.matcher.rollback(num_tokens)
        self.num_processed_tokens -= num_tokens
        self._is_terminated = self.matcher.is_terminated()

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        self.matcher.fill_next_token_bitmask(bitmask, idx)

    def is_terminated(self) -> bool:
        return self._is_terminated

    def reset(self):
        self.num_processed_tokens = 0
        self.matcher.reset()


# cf https://github.com/mlc-ai/xgrammar/blob/a32ac892676d2eedc0327416105b9b06edfb94b2/cpp/json_schema_converter.cc
'''
BACKEND_NEW_REGION = b'''    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the FSM.

        Returns True if all grammar-constrained tokens were accepted.
        Tokens after termination are ignored. Returns False if the FSM
        failed to advance.
        """
        if self._is_terminated:
            return True
        for token in tokens:
            if not self.matcher.accept_token(token):
                logger.error(
                    "Failed to advance FSM for request %s "
                    "for tokens %s. Please file an issue.",
                    request_id,
                    token,
                )
                return False
            self.num_processed_tokens += 1
            self._is_terminated = self.matcher.is_terminated()
            if self._is_terminated:
                break
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the FSM in sequence.
        Will not advance the FSM.

        Returns the prefix list of tokens that are accepted by the FSM.
        """
        if self._is_terminated:
            return []

        accepted_tokens = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted_tokens.append(token)
                if self.matcher.is_terminated():
                    break
            else:
                break
        if len(accepted_tokens) > 0:
            # Rollback the FSM to the initial state
            self.matcher.rollback(len(accepted_tokens))
        return accepted_tokens

    def rollback(self, num_tokens: int) -> None:
        self.matcher.rollback(num_tokens)
        self.num_processed_tokens -= num_tokens
        self._is_terminated = self.matcher.is_terminated()

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        self.matcher.fill_next_token_bitmask(bitmask, idx)

    def is_terminated(self) -> bool:
        return self._is_terminated

    def reset(self):
        self.matcher.reset()
        self.num_processed_tokens = 0
        self._is_terminated = False


# cf https://github.com/mlc-ai/xgrammar/blob/a32ac892676d2eedc0327416105b9b06edfb94b2/cpp/json_schema_converter.cc
'''

MANAGER_OLD_REGION = (
    b"                    if advance_grammar and not grammar.is_terminated():\n"
    b"                        accepted = grammar.accept_tokens(req_id, [token])\n"
    b"                        if accepted:\n"
    b"                            state_advancements += 1\n"
    b"                        elif not post_reasoning_end_in_window:\n"
    b"                            raise AssertionError(\n"
    b"                                (token, req_id, scheduled_spec_decode_tokens)\n"
    b"                            )\n"
)
MANAGER_NEW_REGION = (
    b"                    if advance_grammar and not grammar.is_terminated():\n"
    b"                        if post_reasoning_end_in_window:\n"
    b"                            accepted = bool(grammar.validate_tokens([token]))\n"
    b"                            if accepted:\n"
    b"                                accepted = grammar.accept_tokens(req_id, [token])\n"
    b"                        else:\n"
    b"                            accepted = grammar.accept_tokens(req_id, [token])\n"
    b"                        if accepted:\n"
    b"                            state_advancements += 1\n"
    b"                        elif not post_reasoning_end_in_window:\n"
    b"                            raise AssertionError(\n"
    b"                                (token, req_id, scheduled_spec_decode_tokens)\n"
    b"                            )\n"
)


@dataclass(frozen=True)
class SourceVariant:
    """One exact stock/patched identity pair for a target file."""

    name: str
    stock_sha256: str
    stock_size: int
    patched_sha256: str
    patched_size: int


@dataclass(frozen=True)
class TargetSpec:
    """One chain member: paired stock/patched identities plus the hunk pair."""

    name: str
    production_path: Path
    variants: tuple[SourceVariant, ...]
    stock_region_sha256: str
    patched_region_sha256: str
    old_region: bytes
    new_region: bytes


# The manager file carries two legitimate stock identities.  The default boot
# train applies the #44993 grammar-advance backport before this chain's gate
# (compose order is fixed), so the usual pre-image is the post-#44993 file.
# With DSPARK_SKIP_HOTFIX=1 the chain instead sees the pristine pinned image.
# Neither is a prerequisite of the other: the #53046 anchor exists exactly once
# in both, and the #44993 anchors are intact in the #53046 post-image, so both
# application orders converge.
TARGETS: tuple[TargetSpec, ...] = (
    TargetSpec(
        name="backend_xgrammar",
        production_path=Path(
            "/usr/local/lib/python3.12/dist-packages/vllm/v1/structured_output/"
            "backend_xgrammar.py"
        ),
        variants=(
            SourceVariant(
                name="pinned",
                stock_sha256="231f6b9d7dab5e8d68aba486fa5912db99f8bdd3f9d8842ee3e0bb12bdb7cb67",
                stock_size=12_699,
                patched_sha256="6c7e23c0ae5c6836d0d56862c6e825c49727fa2409b881b44ea2526f1fd03f04",
                patched_size=12_983,
            ),
        ),
        stock_region_sha256="9677073da0986c345f8fa36c787248ff5b3a1b0fbe999da31a91491f3267a149",
        patched_region_sha256="2a7417bbe9e32179c3de8a5750358339320bec672b388fc0ede978e2270b72f4",
        old_region=BACKEND_OLD_REGION,
        new_region=BACKEND_NEW_REGION,
    ),
    TargetSpec(
        name="structured_output_init",
        production_path=Path(
            "/usr/local/lib/python3.12/dist-packages/vllm/v1/structured_output/"
            "__init__.py"
        ),
        variants=(
            SourceVariant(
                name="post-44993",
                stock_sha256="e782163b8a83d58e61a655df042d3126cde8c913a2eeaf9d4a061148cd8e5c77",
                stock_size=21_979,
                patched_sha256="3dff0e1e35f04f35e8c50c17d9efa65cd5fc8db1f25d4eb5d536b6e61114a616",
                patched_size=22_271,
            ),
            SourceVariant(
                name="pristine",
                stock_sha256="fd23813a4e0d8cdc93fa1e6687e5a4f4e514b0ae37dec707d50d840771390818",
                stock_size=22_076,
                patched_sha256="53186ccf86e3d620a9aa91af8c541516f0b45a3f640d937607a252bc42f376e6",
                patched_size=22_368,
            ),
        ),
        stock_region_sha256="e74d20b6eb1ae8d2faff1af624c3756fa40bcabd9fb99dcead5e6ff311231c78",
        patched_region_sha256="d8a63f046f5685bae87b29a15f62912621c3f460fca3c158e9abca1018bb8d37",
        old_region=MANAGER_OLD_REGION,
        new_region=MANAGER_NEW_REGION,
    ),
)

MetadataProvider = Callable[[str], str]
Mode = Literal["apply", "check", "status"]
FileState = Literal["stock", "patched"]
ChainState = Literal["stock", "patched", "partial-legacy", "partial-invalid"]


class HotfixError(RuntimeError):
    """Expected compatibility or transaction failure."""


class CompatibilityError(HotfixError):
    """The installed packages or target bytes are outside the supported pin."""


class RollbackError(HotfixError):
    """Publishing failed and exact restoration also failed."""


@dataclass(frozen=True)
class Inspection:
    state: FileState
    data: bytes
    file_stat: os.stat_result
    digest: str
    variant: SourceVariant


@dataclass(frozen=True)
class ApplyResult:
    outcome: Literal["applied", "already-patched"]
    pre_sha256: str
    post_sha256: str
    vllm_version: str
    xgrammar_version: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _compile_source(data: bytes, label: str) -> None:
    try:
        text = data.decode("utf-8", "strict")
        compile(text, label, "exec")
    except (UnicodeDecodeError, SyntaxError) as error:
        raise CompatibilityError(
            f"source is not valid UTF-8 Python ({type(error).__name__})"
        ) from error


def _load_versions(provider: MetadataProvider) -> tuple[str, str]:
    values: dict[str, str] = {}
    for package in ("vllm", "xgrammar"):
        try:
            value = provider(package)
        except Exception as error:
            raise CompatibilityError(
                f"{package} package metadata unavailable ({type(error).__name__})"
            ) from error
        if not isinstance(value, str):
            raise CompatibilityError(f"{package} package metadata is not a string")
        values[package] = value

    if values["vllm"] != EXPECTED_VLLM_VERSION:
        raise CompatibilityError(
            f"unsupported vllm version {values['vllm']!r}; "
            f"expected {EXPECTED_VLLM_VERSION!r}"
        )
    if values["xgrammar"] != EXPECTED_XGRAMMAR_VERSION:
        raise CompatibilityError(
            f"unsupported xgrammar version {values['xgrammar']!r}; "
            f"expected {EXPECTED_XGRAMMAR_VERSION!r}"
        )
    return values["vllm"], values["xgrammar"]


def _lstat_regular(target: Path) -> os.stat_result:
    try:
        file_stat = target.lstat()
    except FileNotFoundError as error:
        raise CompatibilityError(f"target is missing ({target.name})") from error
    if stat.S_ISLNK(file_stat.st_mode):
        raise CompatibilityError(f"target is a symbolic link ({target.name})")
    if not stat.S_ISREG(file_stat.st_mode):
        raise CompatibilityError(f"target is not a regular file ({target.name})")
    return file_stat


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _same_metadata(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        stat.S_IMODE(left.st_mode),
        left.st_uid,
        left.st_gid,
    ) == (
        stat.S_IMODE(right.st_mode),
        right.st_uid,
        right.st_gid,
    )


def _read_file(target: Path) -> bytes:
    try:
        return target.read_bytes()
    except OSError as error:
        raise CompatibilityError(
            f"cannot read target {target.name} ({type(error).__name__})"
        ) from error


def inspect_target(
    spec: TargetSpec, target: Path
) -> Inspection:
    """Classify an exact stock or exact patched regular file without mutation."""
    before = _lstat_regular(target)
    data = _read_file(target)
    after = _lstat_regular(target)
    if not _same_identity(before, after) or not _same_metadata(before, after):
        raise CompatibilityError(f"target changed while it was being inspected ({target.name})")

    digest = _sha256(data)
    old_count = data.count(spec.old_region)
    new_count = data.count(spec.new_region)
    matched: SourceVariant | None = None
    state: FileState | None = None
    for variant in spec.variants:
        if (
            len(data) == variant.stock_size
            and digest == variant.stock_sha256
            and (old_count, new_count) == (1, 0)
        ):
            matched, state = variant, "stock"
            break
        if (
            len(data) == variant.patched_size
            and digest == variant.patched_sha256
            and (old_count, new_count) == (0, 1)
        ):
            matched, state = variant, "patched"
            break
    if matched is None or state is None:
        raise CompatibilityError(
            f"source identity mismatch ({spec.name}): "
            f"sha256={digest}, bytes={len(data)}, "
            f"regions(old={old_count},new={new_count})"
        )
    _compile_source(data, target.name)
    return Inspection(state, data, after, digest, matched)


def build_candidate(spec: TargetSpec, variant: SourceVariant, stock: bytes) -> bytes:
    """Build and fully validate the exact derived post-image in memory."""
    if (
        len(stock) != variant.stock_size
        or _sha256(stock) != variant.stock_sha256
        or stock.count(spec.old_region) != 1
        or stock.count(spec.new_region) != 0
        or _sha256(spec.old_region) != spec.stock_region_sha256
        or _sha256(spec.new_region) != spec.patched_region_sha256
    ):
        raise CompatibilityError(f"candidate input is not the exact stock source ({spec.name})")

    offset = stock.index(spec.old_region)
    prefix = stock[:offset]
    suffix = stock[offset + len(spec.old_region) :]
    candidate = prefix + spec.new_region + suffix
    if (
        len(candidate) != variant.patched_size
        or _sha256(candidate) != variant.patched_sha256
        or candidate.count(spec.old_region) != 0
        or candidate.count(spec.new_region) != 1
    ):
        raise CompatibilityError(f"constructed post-image failed exact validation ({spec.name})")
    _compile_source(candidate, spec.production_path.name)
    return candidate


def _write_all(fd: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(fd, remaining)
        if written <= 0:
            raise OSError("short write while staging hotfix")
        remaining = remaining[written:]


def _stage_temp(target: Path, data: bytes, original: os.stat_result) -> Path:
    fd = -1
    temp_path: Path | None = None
    try:
        fd, name = tempfile.mkstemp(
            prefix=f".{target.name}.issue136-", suffix=".tmp", dir=target.parent
        )
        temp_path = Path(name)
        try:
            os.fchown(fd, original.st_uid, original.st_gid)
        except OSError:
            staged = os.fstat(fd)
            if (staged.st_uid, staged.st_gid) != (
                original.st_uid,
                original.st_gid,
            ):
                raise
        os.fchmod(fd, stat.S_IMODE(original.st_mode))
        _write_all(fd, data)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        return temp_path
    except BaseException:
        if fd >= 0:
            os.close(fd)
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
        raise


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(directory, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _assert_original_unchanged(
    target: Path, original_data: bytes, original_stat: os.stat_result
) -> None:
    current_stat = _lstat_regular(target)
    if (
        not _same_identity(current_stat, original_stat)
        or not _same_metadata(current_stat, original_stat)
        or _read_file(target) != original_data
    ):
        raise CompatibilityError(f"target changed before atomic publication ({target.name})")


def _verify_published(
    spec: TargetSpec,
    variant: SourceVariant,
    target: Path,
    candidate: bytes,
    original_stat: os.stat_result,
) -> None:
    current_stat = _lstat_regular(target)
    data = _read_file(target)
    if not _same_metadata(current_stat, original_stat):
        raise HotfixError(f"published target metadata changed ({spec.name})")
    if (
        data != candidate
        or len(data) != variant.patched_size
        or _sha256(data) != variant.patched_sha256
        or data.count(spec.old_region) != 0
        or data.count(spec.new_region) != 1
    ):
        raise HotfixError(f"published target failed exact post-image verification ({spec.name})")
    _compile_source(data, target.name)


def _verify_restored(
    target: Path, original_data: bytes, original_stat: os.stat_result
) -> None:
    restored_stat = _lstat_regular(target)
    restored = _read_file(target)
    if restored != original_data or not _same_metadata(restored_stat, original_stat):
        raise RollbackError(f"rollback did not restore exact bytes and metadata ({target.name})")
    _compile_source(restored, target.name)


def _unlink_temp(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _classify_chain(inspections: list[Inspection]) -> ChainState:
    states = [i.state for i in inspections]
    if states == ["stock"] * len(TARGETS):
        return "stock"
    if states == ["patched"] * len(TARGETS):
        return "patched"
    # backend patched + manager stock: a pre-chain #136 application, which
    # apply() completes by publishing the manager file only.
    if states == ["patched", "stock"]:
        return "partial-legacy"
    # manager patched + backend stock is never a legitimate provenance.
    return "partial-invalid"


def _status_word(chain: ChainState) -> str:
    # Status vocabulary is fixed: stock / patched / partial-invalid.
    return "partial-invalid" if chain.startswith("partial") else chain


def _publish_one(
    spec: TargetSpec,
    target: Path,
    inspection: Inspection,
    candidate: bytes,
) -> None:
    """Atomically replace one target, restoring it exactly on any failure."""
    rollback_temp: Path | None = None
    candidate_temp: Path | None = None
    published = False
    try:
        rollback_temp = _stage_temp(target, inspection.data, inspection.file_stat)
        candidate_temp = _stage_temp(target, candidate, inspection.file_stat)
        _assert_original_unchanged(target, inspection.data, inspection.file_stat)
        try:
            os.replace(candidate_temp, target)
        except BaseException:
            # A testable but real possibility: a wrapper/interruption raises
            # after rename completed.  The vanished source temp distinguishes
            # it from a replace that failed before changing the target.
            if not os.path.lexists(candidate_temp):
                published = True
            raise
        else:
            candidate_temp = None
            published = True

        _fsync_directory(target.parent)
        _verify_published(spec, inspection.variant, target, candidate, inspection.file_stat)
    except BaseException as primary_error:
        if published:
            try:
                if rollback_temp is None:
                    raise RollbackError(f"rollback image is unavailable ({spec.name})")
                os.replace(rollback_temp, target)
                rollback_temp = None
                _fsync_directory(target.parent)
                _verify_restored(target, inspection.data, inspection.file_stat)
            except BaseException as rollback_error:
                raise RollbackError(
                    f"hotfix publication failed and rollback failed ({spec.name}; "
                    f"{type(primary_error).__name__}; "
                    f"{type(rollback_error).__name__})"
                ) from rollback_error
        raise
    finally:
        _unlink_temp(candidate_temp)
        _unlink_temp(rollback_temp)


def apply(
    targets: dict[str, Path], metadata_provider: MetadataProvider
) -> ApplyResult:
    """Apply the chain transactionally, or verify an exact full post-image.

    ``targets`` maps each ``TargetSpec.name`` to the path to inspect/patch;
    ``main`` always passes the fixed production targets and installed package
    metadata.  Tests pass temporary fixture paths and a hermetic provider.

    Both candidates are built and compiled before any write.  Files publish
    in chain order; a failure publishing a later file rolls every earlier
    publication back to its exact original bytes and metadata.
    """
    versions = _load_versions(metadata_provider)
    specs: list[TargetSpec] = []
    inspections: list[Inspection] = []
    for spec in TARGETS:
        if spec.name not in targets:
            raise CompatibilityError(f"missing target path for {spec.name}")
        specs.append(spec)
        inspections.append(inspect_target(spec, targets[spec.name]))

    chain = _classify_chain(inspections)
    if chain == "patched":
        joined = ",".join(i.digest for i in inspections)
        return ApplyResult("already-patched", joined, joined, *versions)

    if chain == "partial-invalid":
        raise CompatibilityError(
            "invalid partial chain: structured_output_init patched while "
            "backend_xgrammar is stock; refusing to guess provenance"
        )
    to_publish = [
        (spec, insp) for spec, insp in zip(specs, inspections) if insp.state == "stock"
    ]

    # Build and compile every candidate before the first write.
    candidates = [build_candidate(spec, insp.variant, insp.data) for spec, insp in to_publish]

    published_so_far: list[tuple[TargetSpec, Path, Inspection, bytes]] = []
    try:
        for (spec, insp), candidate in zip(to_publish, candidates):
            path = targets[spec.name]
            _publish_one(spec, path, insp, candidate)
            published_so_far.append((spec, path, insp, candidate))
    except BaseException as primary_error:
        # Roll earlier publications back in reverse order.  _publish_one
        # already restored its own target when its publication failed, so only
        # fully completed earlier writes need rollback here.
        rollback_failures: list[str] = []
        for spec, path, insp, candidate in reversed(published_so_far):
            try:
                _publish_one_restore(spec, path, insp, candidate)
            except BaseException as rollback_error:
                rollback_failures.append(
                    f"{spec.name}:{type(rollback_error).__name__}"
                )
        if rollback_failures:
            raise RollbackError(
                f"chain publication failed ({type(primary_error).__name__}) and "
                f"rollback failed for {', '.join(rollback_failures)}"
            ) from primary_error
        raise

    pre = ",".join(i.digest for i in inspections)
    post = ",".join(i.variant.patched_sha256 for i in inspections)
    return ApplyResult("applied", pre, post, *versions)


def _publish_one_restore(
    spec: TargetSpec, target: Path, inspection: Inspection, candidate: bytes
) -> None:
    """Restore one already-published target to its inspected original bytes.

    Refuses to clobber a concurrent change: the target must still hold exactly
    the candidate this chain just published, byte for byte.
    """
    try:
        current_stat = _lstat_regular(target)
    except CompatibilityError as error:
        raise RollbackError(
            f"refusing rollback: {spec.name} is no longer the published "
            f"regular file ({error})"
        ) from error
    current = _read_file(target)
    if current != candidate or not _same_metadata(current_stat, inspection.file_stat):
        raise RollbackError(
            f"refusing rollback: {spec.name} no longer holds the bytes or "
            "metadata this chain published (concurrent modification)"
        )
    rollback_temp: Path | None = None
    try:
        rollback_temp = _stage_temp(target, inspection.data, inspection.file_stat)
        os.replace(rollback_temp, target)
        rollback_temp = None
        _fsync_directory(target.parent)
        _verify_restored(target, inspection.data, inspection.file_stat)
    finally:
        _unlink_temp(rollback_temp)


def _display_versions() -> tuple[str, str]:
    displayed: list[str] = []
    for package in ("vllm", "xgrammar"):
        try:
            value = importlib.metadata.version(package)
        except Exception as error:
            value = f"unavailable:{type(error).__name__}"
        displayed.append(value)
    return displayed[0], displayed[1]


def _display_digests(targets: dict[str, Path]) -> str:
    digests: list[str] = []
    for spec in TARGETS:
        target = targets[spec.name]
        try:
            file_stat = target.lstat()
            if stat.S_ISREG(file_stat.st_mode):
                digests.append(_sha256(target.read_bytes()))
                continue
        except OSError:
            pass
        digests.append("unavailable")
    return ",".join(digests)


def _log(
    mode: Mode,
    outcome: str,
    vllm_version: str,
    xgrammar_version: str,
    pre_sha256: str,
    post_sha256: str,
) -> None:
    print(
        f"{MARK} mode={mode} vllm={vllm_version} "
        f"xgrammar={xgrammar_version} pre_sha256={pre_sha256} "
        f"post_sha256={post_sha256} outcome={outcome}",
        file=sys.stderr,
    )


def _production_targets() -> dict[str, Path]:
    return {spec.name: spec.production_path for spec in TARGETS}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="check compatibility without writing")
    modes.add_argument("--status", action="store_true", help="report stock/patched/partial-invalid/incompatible")
    args = parser.parse_args(argv)
    mode: Mode = "status" if args.status else "check" if args.check else "apply"
    shown_versions = _display_versions()
    targets = _production_targets()

    try:
        if mode in {"check", "status"}:
            versions = _load_versions(importlib.metadata.version)
            inspections = [
                inspect_target(spec, targets[spec.name]) for spec in TARGETS
            ]
            chain = _classify_chain(inspections)
            joined = ",".join(i.digest for i in inspections)
            _log(mode, chain, versions[0], versions[1], joined, joined)
            if mode == "check":
                if chain == "partial-invalid":
                    print(f"incompatible: {chain}")
                    return 2
                print(f"compatible: {chain}")
                return 0
            word = _status_word(chain)
            if chain == "partial-legacy":
                print(
                    f"{MARK} detail: backend_xgrammar=patched "
                    "structured_output_init=stock (pre-chain #136 application; "
                    "apply completes the chain)",
                    file=sys.stderr,
                )
            print(word)
            return 0 if chain == "patched" else 1

        result = apply(targets, importlib.metadata.version)
        _log(
            mode,
            result.outcome,
            result.vllm_version,
            result.xgrammar_version,
            result.pre_sha256,
            result.post_sha256,
        )
        print(result.outcome)
        return 0
    except CompatibilityError as error:
        digests = _display_digests(targets)
        _log(mode, "incompatible", shown_versions[0], shown_versions[1], digests, digests)
        print(f"{MARK} incompatible: {error}", file=sys.stderr)
        if mode == "status":
            print("incompatible")
        return 2
    except BaseException as error:
        digests = _display_digests(targets)
        _log(mode, "failed", shown_versions[0], shown_versions[1], digests, digests)
        print(
            f"{MARK} failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
