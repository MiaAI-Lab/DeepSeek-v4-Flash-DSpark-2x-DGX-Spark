#!/usr/bin/env python3
"""Source-exact vLLM #45224 backport for issue #117.

The pinned Anemll image can carry the independent issue #79 busy-loop change,
so this patcher recognizes that one exact companion overlay without changing
it.  Only complete pinned stock or complete issue-117 post-images are accepted.
Publication is staged beside the target, durable, atomic, and recoverable.
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
    "/usr/local/lib/python3.12/dist-packages/vllm/distributed/"
    "device_communicators/shm_broadcast.py"
)
EXPECTED_VLLM_VERSION = "0.25.2.dev0+g752a3a504.d20260714"
UPSTREAM_MERGE = "10c75477b07c2f1a361f54b7357af1019bba5fd8"
MARK = "[issue117-shm-ring]"

SPIN_STOCK = b"busy_loop_s: float = 1"
SPIN_ISSUE79 = b"busy_loop_s: float = 0.002"

OLD_CONSTANT = (
    b"VLLM_RINGBUFFER_WARNING_INTERVAL = envs.VLLM_RINGBUFFER_WARNING_INTERVAL\n\n"
)
NEW_CONSTANT = b'''VLLM_RINGBUFFER_WARNING_INTERVAL = envs.VLLM_RINGBUFFER_WARNING_INTERVAL
# Cap on how long an idle reader parks before re-reading the authoritative SHM
# written-flag. Bounds lost-notify recovery latency to ~5s while the periodic
# wakeup stays negligible (one flag check per reader every 5s).
SHM_READER_RECHECK_INTERVAL_MS = 5000


'''
OLD_TIMEOUT = b'''        def timeout_ms(self) -> int | None:
            """Returns a timeout that is:
            - min(time to deadline, time to next warning) if we're logging warnings
            - time to deadline, if we're not logging warnings
            - None if the timeout is None and we're not logging warnings
            - raise TimeoutError if we are past the deadline
            """
            warning_wait_time = self.warning_wait_time_ms
            if self.timeout is None:
                return warning_wait_time

            time_left_ms = int((self.deadline - time.monotonic()) * 1000)
            if time_left_ms <= 0:
                raise TimeoutError

            if warning_wait_time and warning_wait_time < time_left_ms:
                return warning_wait_time

            return time_left_ms
'''
NEW_TIMEOUT = b'''        def timeout_ms(self) -> int:
            """Returns a timeout, capped at the recheck interval, that is:
            - min(time to deadline, time to next warning) if we're logging warnings
            - time to deadline, if we're not logging warnings
            - recheck interval if the timeout is None and we're not logging warnings
            - raise TimeoutError if we are past the deadline
            """
            wait_ms = SHM_READER_RECHECK_INTERVAL_MS
            if self.warning_wait_time_ms is not None:
                wait_ms = min(wait_ms, self.warning_wait_time_ms)
            if self.timeout is None:
                return wait_ms
            time_left_ms = int((self.deadline - time.monotonic()) * 1000)
            if time_left_ms <= 0:
                raise TimeoutError
            return min(wait_ms, time_left_ms)
'''
OLD_RELEASE = b'''                with self.buffer.get_data(self.current_idx) as buf:
                    yield buf

                # caller has read from the buffer
                # set the read flag
                metadata_buffer[self.local_reader_rank + 1] = 1
                # Memory fence ensures the read flag is visible to the writer.
                # Without this, writer may not see our read completion and
                # could wait indefinitely for all readers to finish.
                memory_fence()
                self.current_idx = (self.current_idx + 1) % self.buffer.max_chunks

                self._spin_condition.record_read()
'''
NEW_RELEASE = b'''                with self.buffer.get_data(self.current_idx) as buf:
                    try:
                        yield buf
                    finally:
                        # caller has read from the buffer; set the read flag.
                        metadata_buffer[self.local_reader_rank + 1] = 1
                        # Memory fence ensures the read flag is visible to the writer.
                        # Without this, writer may not see our read completion and
                        # could wait indefinitely for all readers to finish.
                        memory_fence()
                        next_idx = self.current_idx + 1
                        self.current_idx = next_idx % self.buffer.max_chunks
                        self._spin_condition.record_read()
'''

# Complete-file identities derived from vLLM commit 752a3a504485790a2e8491cacbb35c137339ad34.
# Git blob 43e066c44b08453a781098fb04c04d37d8c1a429; bytes are not normalized.
# The issue #79 variants differ only in SPIN_STOCK versus SPIN_ISSUE79.
STOCK_SHA256 = "7ff67c2ef6b8a33a13b11aa3cb202da7887d1d44eed27c6a02d817ea24807d61"
STOCK_ISSUE79_SHA256 = "423234a203429b4d74aa48021a3ea02f3811be7d6a1938369feed298254fd51f"
PATCHED_SHA256 = "911e0dd65e0a0c6346e4f8f2120d2417fefe431ce5f2618ba9e9c9e1986faf23"
PATCHED_ISSUE79_SHA256 = "30d8b62817adab4fabde8ddc6ce9a0f4b71899b80f70e8a50c688f5e63a46b0f"
SOURCE_IDENTITIES: dict[str, tuple[str, bool, int]] = {
    STOCK_SHA256: ("stock-compatible", False, 39_864),
    STOCK_ISSUE79_SHA256: ("stock-compatible", True, 39_868),
    PATCHED_SHA256: ("patched", False, 40_312),
    PATCHED_ISSUE79_SHA256: ("patched", True, 40_316),
}
CANDIDATE_DIGESTS = {
    STOCK_SHA256: PATCHED_SHA256,
    STOCK_ISSUE79_SHA256: PATCHED_ISSUE79_SHA256,
}
PATCHES = (
    (OLD_CONSTANT, NEW_CONSTANT),
    (OLD_TIMEOUT, NEW_TIMEOUT),
    (OLD_RELEASE, NEW_RELEASE),
)

MetadataProvider = Callable[[str], str]
Mode = Literal["apply", "check", "status"]
State = Literal["stock-compatible", "patched"]


class HotfixError(RuntimeError):
    """Expected compatibility or transaction failure."""


class CompatibilityError(HotfixError):
    """The installed package or target bytes are outside the supported pin."""


class RollbackError(HotfixError):
    """Publication failed and exact restoration also failed."""


@dataclass(frozen=True)
class Inspection:
    state: State
    issue79: bool
    data: bytes
    file_stat: os.stat_result
    digest: str
    vllm_version: str


@dataclass(frozen=True)
class ApplyResult:
    outcome: Literal["applied", "already-patched"]
    issue79: bool
    pre_sha256: str
    post_sha256: str
    vllm_version: str


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


def _load_version(provider: MetadataProvider) -> str:
    try:
        version = provider("vllm")
    except Exception as error:
        raise CompatibilityError(
            f"vllm package metadata unavailable ({type(error).__name__})"
        ) from error
    if not isinstance(version, str):
        raise CompatibilityError("vllm package metadata is not a string")
    if version != EXPECTED_VLLM_VERSION:
        raise CompatibilityError(
            f"unsupported vllm version {version!r}; expected {EXPECTED_VLLM_VERSION!r}"
        )
    return version


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


def _region_counts(data: bytes) -> tuple[int, ...]:
    return (
        data.count(OLD_CONSTANT),
        data.count(NEW_CONSTANT),
        data.count(OLD_TIMEOUT),
        data.count(NEW_TIMEOUT),
        data.count(OLD_RELEASE),
        data.count(NEW_RELEASE),
        data.count(SPIN_STOCK),
        data.count(SPIN_ISSUE79),
    )


def _classify_data(data: bytes) -> tuple[State, bool, str]:
    digest = _sha256(data)
    identity = SOURCE_IDENTITIES.get(digest)
    counts = _region_counts(data)
    if identity is None or len(data) != identity[2]:
        raise CompatibilityError(
            "source identity mismatch: "
            f"sha256={digest}, bytes={len(data)}, regions={counts}"
        )

    state = identity[0]
    issue79 = identity[1]
    expected_regions = (1, 0, 1, 0, 1, 0) if state == "stock-compatible" else (0, 1, 0, 1, 0, 1)
    expected_spin = (0, 1) if issue79 else (1, 0)
    if counts[:6] != expected_regions or counts[6:] != expected_spin:
        raise CompatibilityError(
            f"complete {state} source inventory mismatch: regions={counts}"
        )
    return state, issue79, digest


def inspect_target(
    target: Path, metadata_provider: MetadataProvider
) -> Inspection:
    """Classify an exact regular stock/post-image without mutation."""
    before = _lstat_regular(target)
    vllm_version = _load_version(metadata_provider)
    data = _read_file(target)
    after = _lstat_regular(target)
    if (
        not _same_identity(before, after)
        or not _same_metadata(before, after)
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise CompatibilityError("target changed while it was being inspected")
    state, issue79, digest = _classify_data(data)
    _compile_source(data, target.name)
    return Inspection(state, issue79, data, after, digest, vllm_version)


def build_candidate(stock: bytes) -> bytes:
    """Build and fully validate the exact upstream-derived post-image in memory."""
    state, issue79, stock_digest = _classify_data(stock)
    if state != "stock-compatible":
        raise CompatibilityError("candidate input is not a complete stock source")

    candidate = stock
    for old, new in PATCHES:
        if candidate.count(old) != 1 or candidate.count(new) != 0:
            raise CompatibilityError("candidate input region inventory changed")
        candidate = candidate.replace(old, new, 1)

    candidate_state, candidate_issue79, candidate_digest = _classify_data(candidate)
    if (
        candidate_state != "patched"
        or candidate_issue79 != issue79
        or candidate_digest != CANDIDATE_DIGESTS[stock_digest]
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


def _stage_temp(
    target: Path, data: bytes, original: os.stat_result, tag: str
) -> Path:
    fd = -1
    temp_path: Path | None = None
    try:
        fd, name = tempfile.mkstemp(
            prefix=f".{target.name}.issue117-{tag}-",
            suffix=".tmp",
            dir=str(target.parent),
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


def _require_original_unchanged(
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
    issue79: bool,
    metadata_provider: MetadataProvider,
) -> None:
    current_stat = _lstat_regular(target)
    data = _read_file(target)
    if not _same_metadata(current_stat, original_stat):
        raise HotfixError("published target metadata changed")
    state, published_issue79, _digest = _classify_data(data)
    if data != candidate or state != "patched" or published_issue79 != issue79:
        raise HotfixError("published target failed exact post-image verification")
    _load_version(metadata_provider)
    _compile_source(data, target.name)


def _verify_restored(
    target: Path,
    original_data: bytes,
    original_stat: os.stat_result,
    original_state: State,
    issue79: bool,
) -> None:
    restored_stat = _lstat_regular(target)
    restored = _read_file(target)
    if restored != original_data or not _same_metadata(restored_stat, original_stat):
        raise RollbackError("rollback did not restore exact bytes and metadata")
    state, restored_issue79, _digest = _classify_data(restored)
    if state != original_state or restored_issue79 != issue79:
        raise RollbackError("rollback did not restore the original source state")
    _compile_source(restored, target.name)


def _unlink_temp(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except OSError:
        pass


def apply(target: Path, metadata_provider: MetadataProvider) -> ApplyResult:
    """Apply transactionally or verify an exact issue-117 post-image."""
    inspection = inspect_target(target, metadata_provider)
    if inspection.state == "patched":
        return ApplyResult(
            "already-patched",
            inspection.issue79,
            inspection.digest,
            inspection.digest,
            inspection.vllm_version,
        )

    candidate = build_candidate(inspection.data)
    candidate_digest = _sha256(candidate)
    rollback_temp: Path | None = None
    candidate_temp: Path | None = None
    published = False
    try:
        rollback_temp = _stage_temp(
            target, inspection.data, inspection.file_stat, "rollback"
        )
        candidate_temp = _stage_temp(
            target, candidate, inspection.file_stat, "candidate"
        )
        _require_original_unchanged(
            target, inspection.data, inspection.file_stat
        )
        try:
            os.replace(candidate_temp, target)
        except BaseException:
            if not os.path.lexists(candidate_temp):
                published = True
            raise
        else:
            candidate_temp = None
            published = True

        _fsync_directory(target.parent)
        _verify_published(
            target,
            candidate,
            inspection.file_stat,
            inspection.issue79,
            metadata_provider,
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
                    target,
                    inspection.data,
                    inspection.file_stat,
                    inspection.state,
                    inspection.issue79,
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
        inspection.issue79,
        inspection.digest,
        candidate_digest,
        inspection.vllm_version,
    )


def _display_version() -> str:
    try:
        return importlib.metadata.version("vllm")
    except Exception as error:
        return f"unavailable:{type(error).__name__}"


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
    pre_sha256: str,
    post_sha256: str,
    issue79: bool | None,
) -> None:
    issue79_state = "unknown" if issue79 is None else "patched" if issue79 else "stock"
    print(
        f"{MARK} mode={mode} vllm={vllm_version} upstream={UPSTREAM_MERGE} "
        f"issue79={issue79_state} pre_sha256={pre_sha256} "
        f"post_sha256={post_sha256} outcome={outcome}",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--check", action="store_true", help="check compatibility without writing"
    )
    modes.add_argument(
        "--status", action="store_true", help="report patched/stock/incompatible"
    )
    args = parser.parse_args(argv)
    mode: Mode = "status" if args.status else "check" if args.check else "apply"
    shown_version = _display_version()

    try:
        if mode in {"check", "status"}:
            inspection = inspect_target(
                PRODUCTION_TARGET, importlib.metadata.version
            )
            _log(
                mode,
                inspection.state,
                inspection.vllm_version,
                inspection.digest,
                inspection.digest,
                inspection.issue79,
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
            result.pre_sha256,
            result.post_sha256,
            result.issue79,
        )
        print(result.outcome)
        return 0
    except CompatibilityError as error:
        digest = _display_digest()
        _log(mode, "incompatible", shown_version, digest, digest, None)
        print(f"{MARK} incompatible: {error}", file=sys.stderr)
        if mode == "status":
            print("incompatible")
        return 2
    except BaseException as error:
        digest = _display_digest()
        _log(mode, "failed", shown_version, digest, digest, None)
        print(
            f"{MARK} failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
