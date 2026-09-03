#!/usr/bin/env python3
"""Source-exact vLLM #52805 backport for issue #136.

The opt-in Compose gate runs this before ``vllm serve``.  It accepts only the
pinned Anemll 0.1.1 package versions and the exact stock/post-patch source
identities.  Applying the three XgrammarGrammar hunks is one recoverable,
same-directory atomic publication; an already-patched target is verified but
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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

PRODUCTION_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/structured_output/"
    "backend_xgrammar.py"
)
EXPECTED_VLLM_VERSION = "0.25.2.dev0+g752a3a504.d20260714"
EXPECTED_XGRAMMAR_VERSION = "0.2.3"
STOCK_SHA256 = "231f6b9d7dab5e8d68aba486fa5912db99f8bdd3f9d8842ee3e0bb12bdb7cb67"
PATCHED_SHA256 = "6c7e23c0ae5c6836d0d56862c6e825c49727fa2409b881b44ea2526f1fd03f04"
STOCK_SIZE = 12_699
PATCHED_SIZE = 12_983
STOCK_REGION_SHA256 = "9677073da0986c345f8fa36c787248ff5b3a1b0fbe999da31a91491f3267a149"
PATCHED_REGION_SHA256 = "2a7417bbe9e32179c3de8a5750358339320bec672b388fc0ede978e2270b72f4"
MARK = "[issue136-xgrammar]"

OLD_REGION = b'''    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
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
NEW_REGION = b'''    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
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

MetadataProvider = Callable[[str], str]
Mode = Literal["apply", "check", "status"]
State = Literal["stock-compatible", "patched"]


class HotfixError(RuntimeError):
    """Expected compatibility or transaction failure."""


class CompatibilityError(HotfixError):
    """The installed packages or target bytes are outside the supported pin."""


class RollbackError(HotfixError):
    """Publishing failed and exact restoration also failed."""


@dataclass(frozen=True)
class Inspection:
    state: State
    data: bytes
    file_stat: os.stat_result
    digest: str
    vllm_version: str
    xgrammar_version: str


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
        raise CompatibilityError("target is missing") from error
    if stat.S_ISLNK(file_stat.st_mode):
        raise CompatibilityError("target is a symbolic link")
    if not stat.S_ISREG(file_stat.st_mode):
        raise CompatibilityError("target is not a regular file")
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
            f"cannot read target ({type(error).__name__})"
        ) from error


def inspect_target(
    target: Path, metadata_provider: MetadataProvider
) -> Inspection:
    """Classify an exact stock or exact patched regular file without mutation."""
    before = _lstat_regular(target)
    vllm_version, xgrammar_version = _load_versions(metadata_provider)
    data = _read_file(target)
    after = _lstat_regular(target)
    if not _same_identity(before, after) or not _same_metadata(before, after):
        raise CompatibilityError("target changed while it was being inspected")

    digest = _sha256(data)
    old_count = data.count(OLD_REGION)
    new_count = data.count(NEW_REGION)
    if (
        len(data) == STOCK_SIZE
        and digest == STOCK_SHA256
        and (old_count, new_count) == (1, 0)
    ):
        state: State = "stock-compatible"
    elif (
        len(data) == PATCHED_SIZE
        and digest == PATCHED_SHA256
        and (old_count, new_count) == (0, 1)
    ):
        state = "patched"
    else:
        raise CompatibilityError(
            "source identity mismatch: "
            f"sha256={digest}, bytes={len(data)}, "
            f"regions(old={old_count},new={new_count})"
        )
    _compile_source(data, target.name)
    return Inspection(
        state,
        data,
        after,
        digest,
        vllm_version,
        xgrammar_version,
    )


def build_candidate(stock: bytes) -> bytes:
    """Build and fully validate the exact derived post-image in memory."""
    if (
        len(stock) != STOCK_SIZE
        or _sha256(stock) != STOCK_SHA256
        or stock.count(OLD_REGION) != 1
        or stock.count(NEW_REGION) != 0
        or _sha256(OLD_REGION) != STOCK_REGION_SHA256
        or _sha256(NEW_REGION) != PATCHED_REGION_SHA256
    ):
        raise CompatibilityError("candidate input is not the exact stock source")

    offset = stock.index(OLD_REGION)
    prefix = stock[:offset]
    suffix = stock[offset + len(OLD_REGION) :]
    candidate = prefix + NEW_REGION + suffix
    if (
        len(candidate) != PATCHED_SIZE
        or _sha256(candidate) != PATCHED_SHA256
        or candidate.count(OLD_REGION) != 0
        or candidate.count(NEW_REGION) != 1
        or candidate[:offset] != prefix
        or candidate[offset + len(NEW_REGION) :] != suffix
    ):
        raise CompatibilityError("constructed post-image failed exact validation")
    _compile_source(candidate, PRODUCTION_TARGET.name)
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
        raise CompatibilityError("target changed before atomic publication")


def _verify_published(
    target: Path,
    candidate: bytes,
    original_stat: os.stat_result,
    metadata_provider: MetadataProvider,
) -> None:
    current_stat = _lstat_regular(target)
    data = _read_file(target)
    if not _same_metadata(current_stat, original_stat):
        raise HotfixError("published target metadata changed")
    if (
        data != candidate
        or len(data) != PATCHED_SIZE
        or _sha256(data) != PATCHED_SHA256
        or data.count(OLD_REGION) != 0
        or data.count(NEW_REGION) != 1
    ):
        raise HotfixError("published target failed exact post-image verification")
    _load_versions(metadata_provider)
    _compile_source(data, target.name)


def _verify_restored(
    target: Path, original_data: bytes, original_stat: os.stat_result
) -> None:
    restored_stat = _lstat_regular(target)
    restored = _read_file(target)
    if restored != original_data or not _same_metadata(restored_stat, original_stat):
        raise RollbackError("rollback did not restore exact bytes and metadata")
    _compile_source(restored, target.name)


def _unlink_temp(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def apply(target: Path, metadata_provider: MetadataProvider) -> ApplyResult:
    """Apply to ``target`` transactionally, or verify an exact post-image.

    ``main`` always passes the fixed production target and installed package
    metadata.  Tests pass temporary fixture paths and a hermetic provider.
    """
    inspection = inspect_target(target, metadata_provider)
    if inspection.state == "patched":
        return ApplyResult(
            "already-patched",
            inspection.digest,
            inspection.digest,
            inspection.vllm_version,
            inspection.xgrammar_version,
        )

    candidate = build_candidate(inspection.data)
    rollback_temp: Path | None = None
    candidate_temp: Path | None = None
    published = False
    try:
        # Prepare a durable restoration image before publication.  Neither
        # temporary path is visible at the target until one atomic replace.
        rollback_temp = _stage_temp(
            target, inspection.data, inspection.file_stat
        )
        candidate_temp = _stage_temp(target, candidate, inspection.file_stat)
        _assert_original_unchanged(
            target, inspection.data, inspection.file_stat
        )
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
        _verify_published(
            target, candidate, inspection.file_stat, metadata_provider
        )
    except BaseException as primary_error:
        if published:
            try:
                if rollback_temp is None:
                    raise RollbackError("rollback image is unavailable")
                os.replace(rollback_temp, target)
                rollback_temp = None
                _fsync_directory(target.parent)
                _verify_restored(
                    target, inspection.data, inspection.file_stat
                )
            except BaseException as rollback_error:
                raise RollbackError(
                    "hotfix publication failed and rollback failed "
                    f"({type(primary_error).__name__}; "
                    f"{type(rollback_error).__name__})"
                ) from rollback_error
        raise
    finally:
        _unlink_temp(candidate_temp)
        _unlink_temp(rollback_temp)

    return ApplyResult(
        "applied",
        inspection.digest,
        PATCHED_SHA256,
        inspection.vllm_version,
        inspection.xgrammar_version,
    )


def _display_versions() -> tuple[str, str]:
    displayed: list[str] = []
    for package in ("vllm", "xgrammar"):
        try:
            value = importlib.metadata.version(package)
        except Exception as error:
            value = f"unavailable:{type(error).__name__}"
        displayed.append(value)
    return displayed[0], displayed[1]


def _display_digest() -> str:
    try:
        file_stat = PRODUCTION_TARGET.lstat()
        if stat.S_ISREG(file_stat.st_mode) and not stat.S_ISLNK(file_stat.st_mode):
            return _sha256(PRODUCTION_TARGET.read_bytes())
    except OSError:
        pass
    return "unavailable"


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="check compatibility without writing")
    modes.add_argument("--status", action="store_true", help="report stock-compatible/patched/incompatible")
    args = parser.parse_args(argv)
    mode: Mode = "status" if args.status else "check" if args.check else "apply"
    shown_versions = _display_versions()

    try:
        if mode in {"check", "status"}:
            inspection = inspect_target(
                PRODUCTION_TARGET, importlib.metadata.version
            )
            _log(
                mode,
                inspection.state,
                inspection.vllm_version,
                inspection.xgrammar_version,
                inspection.digest,
                inspection.digest,
            )
            if mode == "check":
                print(f"compatible: {inspection.state}")
                return 0
            print(inspection.state)
            return 0 if inspection.state == "patched" else 1

        result = apply(PRODUCTION_TARGET, importlib.metadata.version)
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
        digest = _display_digest()
        _log(
            mode,
            "incompatible",
            shown_versions[0],
            shown_versions[1],
            digest,
            digest,
        )
        print(f"{MARK} incompatible: {error}", file=sys.stderr)
        if mode == "status":
            print("incompatible")
        return 2
    except BaseException as error:
        digest = _display_digest()
        _log(
            mode,
            "failed",
            shown_versions[0],
            shown_versions[1],
            digest,
            digest,
        )
        print(
            f"{MARK} failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
