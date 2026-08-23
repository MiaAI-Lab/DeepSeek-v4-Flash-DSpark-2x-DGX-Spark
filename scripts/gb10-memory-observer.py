#!/usr/bin/env python3
"""External, report-only GB10 memory/NVRM flight recorder (Issue #32).

The observer records host memory, memory PSI, bounded vLLM process-liveness
facts, and sanitized NVRM ``NV_ERR_NO_MEMORY (0x51)`` / Xid kernel facts.  It
never changes serving state, contacts an API, elevates privileges, or acts on
a threshold.  An isolated 0x51 is a fact, not a safety verdict.

Commands:

  run      Sample in the foreground.
  start    Detach ``run`` and return only after journal startup was attempted
           and a durable session_start plus first sample have been recorded.
  stop     Stop only the exact pidfile-recorded observer, using a pidfd.  If
           safe handle-based signaling is unavailable, refuse to signal.
  status   Report state without changing it.  Absence is success; malformed
           configuration still exits 2 before command dispatch.
  once     Append one durable session_start/sample/session_end session.

Exit codes are 0 for command success, 1 for an operational or durability
failure, and 2 for usage/configuration errors.  The five command names are a
stable interface.

Records are newline-delimited JSON, serialized by a separate advisory lock
shared by daemon and one-shot writers.  Every record is fully written,
fdatasync'd, and closed.  Interrupted appends are rolled back, and any
unterminated tail is durably repaired before another append.  Rotation is
bounded to 16 MiB x four files; a rotation failure stops recording rather
than appending past the limit.  Every attempt retries state-directory
metadata durability so an earlier directory-sync failure cannot be forgotten.

The journal follower is current-boot and future-only (``-n 0 -f``).  Events
while no follower exists are not replayed.  Each session_start therefore
carries ``resume_after_ts_utc``, the timestamp of the preceding durable
record (or null for the first session), as the explicit left edge of that
unobserved gap.  Any follower EOF, including exit status 0, is degraded.
Input lines, the filtered queue, counters, and work per sample tick are all
bounded; drop/backlog/gap facts are recorded in samples.

Every record carries a random ``session_id``.  The atomic pidfile additionally
stores boot ID, /proc starttime, and the exact script argv, allowing status
and stop to correlate a daemon even when overlapping ``once`` sessions append
their own markers.  Detached daemons and journal children receive independent
minimal allowlisted environments.  The daemon changes to ``/``, uses umask
077, and gives its Linux journal child a parent-death signal.
"""
from __future__ import annotations

import ctypes
import errno
import fcntl
import json
import math
import os
import queue
import re
import secrets
import select
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple

SCHEMA = "dspark-gb10-observer/v1"
DEFAULT_INTERVAL_S = 2.0
ROTATION_BYTES = 16 * 1024 * 1024
ROTATION_KEEP = 4  # active records.ndjson plus .1 ... .3
RECORD_LINE_LIMIT = 256 * 1024
READY_TIMEOUT_S = 5.0
STOP_GRACE_S = 5.0
STOP_KILL_WAIT_S = 2.0
JOURNAL_LINE_LIMIT = 64 * 1024
JOURNAL_QUEUE_MAX = 256
JOURNAL_ITEMS_PER_TICK = 16
JOURNAL_DRAIN_BUDGET_S = 0.050
JOURNAL_COUNTER_MAX = (1 << 63) - 1
PROC_SCAN_MAX = 8192
PROC_SCAN_BUDGET_S = 0.050
PROC_CMDLINE_LIMIT = 64 * 1024
EXCERPT_LIMIT = 240
LOCK_NAME = "observer.lock"
RECORD_LOCK_NAME = "records.lock"
PIDFILE_NAME = "observer.pid"
PIDFILE_TMP_NAME = ".observer.pid.tmp"
RECORDS_NAME = "records.ndjson"
DAEMON_FLAG = "--dspark-gb10-observer-daemon"
READY_FD_PREFIX = "--ready-fd="
PIDFILE_VERSION = 1
PR_SET_PDEATHSIG = 1

ENV_ENABLED = "DSPARK_GB10_OBSERVER"
ENV_INTERVAL = "DSPARK_GB10_OBSERVER_INTERVAL"
ENV_STATE_DIR = "DSPARK_GB10_OBSERVER_STATE_DIR"
ENV_AUTOSTOP = "DSPARK_GB10_OBSERVER_AUTOSTOP"
ENV_JOURNALCTL = "DSPARK_GB10_OBSERVER_JOURNALCTL"
_OBSERVER_ENV_NAMES = {
    ENV_ENABLED,
    ENV_INTERVAL,
    ENV_STATE_DIR,
    ENV_AUTOSTOP,
    ENV_JOURNALCTL,
}
_LOCALE_ENV_NAMES = {
    "PATH",
    "LANG",
    "LANGUAGE",
    "TZ",
    "LC_ALL",
    "LC_CTYPE",
    "LC_NUMERIC",
    "LC_TIME",
    "LC_COLLATE",
    "LC_MONETARY",
    "LC_MESSAGES",
    "LC_PAPER",
    "LC_NAME",
    "LC_ADDRESS",
    "LC_TELEPHONE",
    "LC_MEASUREMENT",
    "LC_IDENTIFICATION",
}

_NVRM_RE = re.compile(r"\bNVRM\b", re.IGNORECASE)
_NO_MEMORY_RE = re.compile(r"\bNV_ERR_NO_MEMORY\b|\(0x51\)", re.IGNORECASE)
_XID_CODE_RE = re.compile(r"(\d+)\s*,")
_CTRL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")
_JOURNAL_TS_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2}(?:\.\d+)?)(Z|[+-]\d{2}:?\d{2})?"
)

USAGE = """\
usage: gb10-memory-observer.py {run|start|stop|status|once}

  run      sample this node in the foreground
  start    detach and wait for a durable first sample
  stop     stop the exact pidfile-recorded instance through a pidfd
  status   report state/pidfile/record facts (absence is success)
  once     append one durable one-shot session

Configuration: DSPARK_GB10_OBSERVER_{INTERVAL,STATE_DIR,AUTOSTOP}; see the
module docstring.  Newline-delimited JSON records; strictly report-only.
"""


class ConfigError(Exception):
    """Invalid DSPARK_GB10_OBSERVER_* environment configuration."""


@dataclass(frozen=True)
class Config:
    interval_s: float
    state_dir: Path
    journal_command: Tuple[str, ...]


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    boot_id: str
    starttime: int
    argv: Tuple[str, ...]
    session_id: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _journal_ts_to_utc(line: str) -> Optional[str]:
    match = _JOURNAL_TS_RE.match(line)
    if not match:
        return None
    stamp = f"{match.group(1)}T{match.group(2)}"
    zone = match.group(3)
    if zone:
        if zone == "Z":
            stamp += "+00:00"
        elif ":" not in zone[1:]:
            stamp += f"{zone[:3]}:{zone[3:]}"
        else:
            stamp += zone
    try:
        parsed = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_state_dir(env) -> Path:
    xdg = env.get("XDG_STATE_HOME")
    if xdg:
        return Path(xdg) / "dspark-observer"
    home = env.get("HOME")
    base = Path(home) if home else Path.home()
    return base / ".local" / "state" / "dspark-observer"


def _validate_boolean(env, name: str) -> None:
    raw = env.get(name)
    if raw in (None, "", "0", "1"):
        return
    raise ConfigError(f"{name} must be 0 or 1, got {raw!r}")


def parse_config(env=None) -> Config:
    env = os.environ if env is None else env
    _validate_boolean(env, ENV_ENABLED)
    _validate_boolean(env, ENV_AUTOSTOP)

    raw_interval = env.get(ENV_INTERVAL)
    if raw_interval in (None, ""):
        interval_s = DEFAULT_INTERVAL_S
    else:
        try:
            interval_s = float(raw_interval)
        except (ValueError, OverflowError):
            raise ConfigError(
                f"{ENV_INTERVAL} must be a finite positive number of seconds, got {raw_interval!r}"
            ) from None
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ConfigError(
                f"{ENV_INTERVAL} must be a finite positive number of seconds, got {raw_interval!r}"
            )

    raw_state = env.get(ENV_STATE_DIR)
    state_dir = Path(raw_state).expanduser() if raw_state else _default_state_dir(env)
    journal_bin = env.get(ENV_JOURNALCTL) or "journalctl"
    journal_command = (journal_bin, "-b", "0", "-k", "-f", "-o", "short-iso", "-n", "0")
    return Config(interval_s=interval_s, state_dir=state_dir, journal_command=journal_command)


def _minimal_environment(source, *, include_observer: bool) -> dict:
    allowed = set(_LOCALE_ENV_NAMES)
    if include_observer:
        allowed.update(_OBSERVER_ENV_NAMES)
        allowed.update({"HOME", "XDG_STATE_HOME"})
    result = {
        name: value
        for name, value in source.items()
        if name in allowed
    }
    result.setdefault("PATH", os.defpath)
    result.setdefault("LANG", "C.UTF-8")
    return result


def _minimal_daemon_environment(source=None) -> dict:
    return _minimal_environment(
        os.environ if source is None else source, include_observer=True
    )


def _minimal_journal_environment(source=None) -> dict:
    return _minimal_environment(
        os.environ if source is None else source, include_observer=False
    )


def _set_linux_pdeathsig(expected_parent_pid: int) -> None:
    """Make the soon-to-exec child die with its observer parent.

    Reset SIGTERM first because the sampling loop installs a Python handler.
    The getppid check closes the race in which the parent dies just before the
    prctl call.
    """
    if not sys.platform.startswith("linux"):
        raise OSError(errno.ENOTSUP, "parent-death signaling requires Linux")
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
        error_number = ctypes.get_errno() or errno.EINVAL
        raise OSError(error_number, os.strerror(error_number))
    if os.getppid() != expected_parent_pid:
        signal.raise_signal(signal.SIGTERM)


def sanitize_excerpt(line: str, limit: int = EXCERPT_LIMIT) -> str:
    return _CTRL_CHARS_RE.sub(" ", line).strip()[:limit]


def classify_kernel_line(line: str) -> Optional[dict]:
    """Return only sanitized NVRM 0x51 / Xid facts."""
    if not _NVRM_RE.search(line):
        return None
    types = []
    xid = None
    if _NO_MEMORY_RE.search(line):
        types.append("nvrm_err_no_memory_0x51")
    position = line.lower().find("xid")
    if position >= 0:
        types.append("xid")
        code = _XID_CODE_RE.search(line[position : position + 80])
        if code:
            xid = int(code.group(1))
    if not types:
        return None
    fact = {"types": sorted(types), "excerpt": sanitize_excerpt(line)}
    if xid is not None:
        fact["xid"] = xid
    stamp = _journal_ts_to_utc(line)
    if stamp:
        fact["journal_ts_utc"] = stamp
    return fact


def read_mem_available_kib(path: str = "/proc/meminfo") -> Tuple[Optional[int], Optional[str]]:
    try:
        with open(path, encoding="ascii") as handle:
            content = handle.read(1024 * 1024)
    except OSError as exc:
        return None, f"unreadable: errno {exc.errno}"
    for line in content.splitlines():
        key, sep, rest = line.partition(":")
        if sep and key.strip() == "MemAvailable":
            fields = rest.split()
            if fields:
                try:
                    return int(fields[0]), None
                except ValueError:
                    break
    return None, "MemAvailable missing or unparsable"


def read_memory_psi(path: str = "/proc/pressure/memory") -> dict:
    try:
        with open(path, encoding="ascii") as handle:
            content = handle.read(1024 * 1024)
    except FileNotFoundError:
        return {}
    except OSError as exc:
        return {"psi_error": f"unreadable: errno {exc.errno}"}
    fields = {}
    for line in content.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0] not in ("some", "full"):
            return {"psi_error": "unrecognized PSI line"}
        prefix = f"psi_{parts[0]}"
        for item in parts[1:]:
            name, eq, value = item.partition("=")
            if not eq or name not in ("avg10", "avg60", "avg300", "total"):
                return {"psi_error": "unrecognized PSI line"}
            if name == "total":
                continue
            try:
                fields[f"{prefix}_{name}"] = float(value)
            except ValueError:
                return {"psi_error": "unrecognized PSI line"}
    return fields


def count_vllm_serve_processes(
    proc_root: str = "/proc",
    max_entries: int = PROC_SCAN_MAX,
    budget_s: float = PROC_SCAN_BUDGET_S,
) -> Tuple[Optional[int], Optional[str]]:
    """Bounded process-liveness scan; truncated scans are explicit errors."""
    deadline = time.monotonic() + budget_s
    count = 0
    inspected = 0
    try:
        with os.scandir(proc_root) as entries:
            for entry in entries:
                if time.monotonic() >= deadline or inspected >= max_entries:
                    return None, "process scan budget exhausted"
                if not entry.name.isdigit():
                    continue
                inspected += 1
                try:
                    with open(os.path.join(proc_root, entry.name, "cmdline"), "rb") as handle:
                        raw = handle.read(PROC_CMDLINE_LIMIT + 1)
                except OSError:
                    continue
                if len(raw) > PROC_CMDLINE_LIMIT:
                    continue
                argv = [
                    part.decode("utf-8", "replace")
                    for part in raw.split(b"\x00")
                    if part
                ]
                if "serve" in argv and any(
                    arg == "vllm" or arg.endswith("/vllm") for arg in argv
                ):
                    count += 1
    except OSError as exc:
        return None, f"unreadable: errno {exc.errno}"
    return count, None


def _safe_error(prefix: str, exc: BaseException) -> str:
    number = getattr(exc, "errno", None)
    return f"{prefix}: errno {number}" if number is not None else prefix


class JournalSource:
    """Long-lived byte-bounded reader with a filtered bounded fact queue."""

    def __init__(self, command: Tuple[str, ...]):
        self.command = list(command)
        self.proc: Optional[subprocess.Popen] = None
        self.lines: "queue.Queue[dict]" = queue.Queue(maxsize=JOURNAL_QUEUE_MAX)
        self.error: Optional[str] = None
        self.eof = False
        self._thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()
        self._dropped_long_lines = 0
        self._dropped_queue_full = 0
        self._counter_saturated = False

    def _increment(self, attribute: str) -> None:
        with self._state_lock:
            value = getattr(self, attribute)
            if value >= JOURNAL_COUNTER_MAX:
                self._counter_saturated = True
            else:
                setattr(self, attribute, value + 1)

    def _set_error(self, detail: str) -> None:
        with self._state_lock:
            if self.error is None:
                self.error = detail

    def start(self) -> None:
        expected_parent = os.getpid()
        try:
            proc = subprocess.Popen(
                self.command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=JOURNAL_LINE_LIMIT + 1,
                close_fds=True,
                env=_minimal_journal_environment(),
                preexec_fn=lambda: _set_linux_pdeathsig(expected_parent),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self._set_error(_safe_error("spawn failed", exc))
            return
        self.proc = proc
        self._thread = threading.Thread(
            target=self._pump,
            args=(proc,),
            daemon=True,
            name="gb10-observer-journal",
        )
        self._thread.start()

    def _discard_line_remainder(self, stream, raw: bytes) -> None:
        while raw and not raw.endswith(b"\n"):
            raw = stream.readline(JOURNAL_LINE_LIMIT + 1)

    def _pump(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        stream = proc.stdout
        try:
            while True:
                raw = stream.readline(JOURNAL_LINE_LIMIT + 1)
                if not raw:
                    break
                if len(raw) > JOURNAL_LINE_LIMIT:
                    self._discard_line_remainder(stream, raw)
                    self._increment("_dropped_long_lines")
                    continue
                line = raw.rstrip(b"\r\n").decode("utf-8", "replace")
                fact = classify_kernel_line(line)
                if fact is None:
                    continue
                try:
                    self.lines.put_nowait(fact)
                except queue.Full:
                    self._increment("_dropped_queue_full")
        except (OSError, ValueError) as exc:
            self._set_error(_safe_error("read failed", exc))
        finally:
            try:
                stream.close()
            except OSError:
                pass
            status = proc.poll()
            if status is None:
                self._set_error("journal stream reached EOF")
            else:
                self._set_error(f"journalctl exited with status {status}")
            with self._state_lock:
                self.eof = True

    def health(self) -> Tuple[bool, Optional[str]]:
        with self._state_lock:
            error = self.error
            eof = self.eof
        if error:
            return False, error
        proc = self.proc
        if proc is None:
            return False, "not started"
        status = proc.poll()
        if status is not None:
            return False, f"journalctl exited with status {status}"
        if eof:
            return False, "journal stream reached EOF"
        return True, None

    def has_activity(self) -> bool:
        with self._state_lock:
            return bool(self.error or self.eof) or not self.lines.empty()

    def drain(
        self,
        on_fact: Callable[[dict], None],
        max_items: int = JOURNAL_ITEMS_PER_TICK,
        budget_s: float = JOURNAL_DRAIN_BUDGET_S,
    ) -> int:
        deadline = time.monotonic() + max(0.0, budget_s)
        processed = 0
        while processed < max_items:
            if processed and time.monotonic() >= deadline:
                break
            try:
                fact = self.lines.get_nowait()
            except queue.Empty:
                break
            on_fact(fact)
            processed += 1
        return processed

    def snapshot(self) -> dict:
        with self._state_lock:
            long_lines = self._dropped_long_lines
            queue_full = self._dropped_queue_full
            saturated = self._counter_saturated
        backlog = self.lines.qsize()
        return {
            "journal_dropped_long_lines_total": long_lines,
            "journal_dropped_queue_full_total": queue_full,
            "journal_backlog_items": backlog,
            "journal_gap": bool(long_lines or queue_full),
            "journal_counter_saturated": saturated,
        }

    def close(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=2)
                else:
                    proc.wait()
            except (OSError, subprocess.TimeoutExpired):
                pass
            finally:
                for stream in (proc.stdout, proc.stderr, proc.stdin):
                    if stream is not None and not stream.closed:
                        try:
                            stream.close()
                        except OSError:
                            pass
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=2)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written is None or written <= 0:
            raise OSError(errno.EIO, "short write while recording")
        view = view[written:]


class Observer:
    def __init__(
        self,
        config: Config,
        *,
        proc_root: str = "/proc",
        journal: Optional[JournalSource] = None,
        ready_fd: Optional[int] = None,
    ):
        self.config = config
        self.state_dir = config.state_dir
        self.records_path = self.state_dir / RECORDS_NAME
        self.record_lock_path = self.state_dir / RECORD_LOCK_NAME
        self.lock_path = self.state_dir / LOCK_NAME
        self.pidfile_path = self.state_dir / PIDFILE_NAME
        self.proc_root = Path(proc_root)
        self.seq = 0
        self.session_id = secrets.token_hex(16)
        self.monotonic_origin = time.monotonic()
        self.boot_id = self._read_boot_id()
        self.hostname = socket.gethostname()
        self.journal = journal if journal is not None else JournalSource(config.journal_command)
        self.lock_fd: Optional[int] = None
        self._stop_requested = False
        self._journal_degraded_emitted = False
        self.ready_fd = ready_fd
        if self.ready_fd is not None:
            os.set_inheritable(self.ready_fd, False)

    def _read_boot_id(self) -> str:
        path = self.proc_root / "sys" / "kernel" / "random" / "boot_id"
        try:
            value = path.read_text(encoding="ascii").strip()
        except OSError:
            return "unknown-boot-id"
        return value or "unknown-boot-id"

    def _ensure_state_dir(self) -> None:
        self.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.state_dir, 0o700)

    def _sync_state_dir(self) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
        fd = os.open(self.state_dir, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _record_paths(self, *, oldest_first: bool) -> list:
        generations = [
            Path(f"{self.records_path}.{index}")
            for index in range(ROTATION_KEEP - 1, 0, -1)
        ]
        paths = generations + [self.records_path]
        return paths if oldest_first else list(reversed(paths))

    @staticmethod
    def _records_from_path(path: Path) -> Iterator[dict]:
        try:
            handle = open(path, "rb")
        except OSError:
            return
        with handle:
            while True:
                raw = handle.readline(RECORD_LINE_LIMIT + 1)
                if not raw:
                    break
                if len(raw) > RECORD_LINE_LIMIT:
                    while raw and not raw.endswith(b"\n"):
                        raw = handle.readline(RECORD_LINE_LIMIT + 1)
                    continue
                try:
                    record = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(record, dict):
                    yield record

    def _latest_record_ts_locked(self) -> Optional[str]:
        for path in self._record_paths(oldest_first=False):
            latest = None
            for record in self._records_from_path(path):
                stamp = record.get("ts_utc")
                if isinstance(stamp, str):
                    latest = stamp
            if latest is not None:
                return latest
        return None

    def _repair_unterminated_tail_locked(self) -> None:
        flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(self.records_path, flags)
        except FileNotFoundError:
            return
        try:
            size = os.lseek(fd, 0, os.SEEK_END)
            if size == 0 or os.pread(fd, 1, size - 1) == b"\n":
                return
            truncate_at = 0
            position = size
            while position > 0:
                start = max(0, position - JOURNAL_LINE_LIMIT)
                chunk = os.pread(fd, position - start, start)
                newline = chunk.rfind(b"\n")
                if newline >= 0:
                    truncate_at = start + newline + 1
                    break
                position = start
            os.ftruncate(fd, truncate_at)
            os.fdatasync(fd)
        finally:
            os.close(fd)

    def _rotate_if_needed_locked(self, incoming_bytes: int) -> None:
        if incoming_bytes > min(ROTATION_BYTES, RECORD_LINE_LIMIT):
            raise OSError(errno.EFBIG, "record exceeds bounded line/file size")
        try:
            size = self.records_path.stat().st_size
        except FileNotFoundError:
            return
        if size == 0 or size + incoming_bytes <= ROTATION_BYTES:
            return
        oldest = Path(f"{self.records_path}.{ROTATION_KEEP - 1}")
        if oldest.exists():
            oldest.unlink()
        for index in range(ROTATION_KEEP - 2, 0, -1):
            source = Path(f"{self.records_path}.{index}")
            if source.exists():
                os.replace(source, Path(f"{self.records_path}.{index + 1}"))
        os.replace(self.records_path, Path(f"{self.records_path}.1"))
        self._sync_state_dir()

    def _record(self, event: str, **fields) -> None:
        self._ensure_state_dir()
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        lock_fd = os.open(self.record_lock_path, flags, 0o600)
        try:
            os.fchmod(lock_fd, 0o600)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            # Always retry metadata durability. A failed directory fsync must
            # never be forgotten merely because the directory entry now exists.
            self._sync_state_dir()
            self._repair_unterminated_tail_locked()
            fields = dict(fields)
            if event == "session_start" and "resume_after_ts_utc" not in fields:
                fields["resume_after_ts_utc"] = self._latest_record_ts_locked()
            payload = dict(fields)
            payload.update(
                {
                    "schema": SCHEMA,
                    "event": event,
                    "ts_utc": _utc_now_iso(),
                    "monotonic_s": round(time.monotonic() - self.monotonic_origin, 6),
                    "boot_id": self.boot_id,
                    "hostname": self.hostname,
                    "seq": self.seq,
                    "session_id": self.session_id,
                }
            )
            encoded = (
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            self._rotate_if_needed_locked(len(encoded))
            created = not self.records_path.exists()
            record_fd = os.open(
                self.records_path,
                os.O_RDWR
                | os.O_CREAT
                | os.O_APPEND
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                os.fchmod(record_fd, 0o600)
                pre_append_size = os.lseek(record_fd, 0, os.SEEK_END)
                try:
                    _write_all(record_fd, encoded)
                    os.fdatasync(record_fd)
                except OSError as write_error:
                    try:
                        os.ftruncate(record_fd, pre_append_size)
                        os.fdatasync(record_fd)
                        self._sync_state_dir()
                    except OSError as rollback_error:
                        raise rollback_error from write_error
                    raise
            finally:
                os.close(record_fd)
            if created:
                self._sync_state_dir()
            self.seq += 1
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)

    def acquire_lock(self) -> bool:
        self._ensure_state_dir()
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        self.lock_fd = os.open(self.lock_path, flags, 0o600)
        try:
            os.fchmod(self.lock_fd, 0o600)
        except OSError:
            os.close(self.lock_fd)
            self.lock_fd = None
            raise
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(self.lock_fd)
            self.lock_fd = None
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                return False
            raise
        return True

    def release_lock(self) -> None:
        if self.lock_fd is None:
            return
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(self.lock_fd)
        self.lock_fd = None

    def _proc_stat(self, pid: int) -> Optional[Tuple[str, int]]:
        try:
            with open(self.proc_root / str(pid) / "stat", "rb") as handle:
                data = handle.read(8193)
        except OSError:
            return None
        if len(data) > 8192:
            return None
        close_paren = data.rfind(b")")
        if close_paren < 0:
            return None
        fields = data[close_paren + 1 :].split()
        if len(fields) <= 19:
            return None
        try:
            state = fields[0].decode("ascii")
            starttime = int(fields[19])
        except (UnicodeDecodeError, ValueError):
            return None
        return state, starttime

    def _proc_argv(self, pid: int) -> Optional[Tuple[str, ...]]:
        try:
            with open(self.proc_root / str(pid) / "cmdline", "rb") as handle:
                raw = handle.read(PROC_CMDLINE_LIMIT + 1)
        except OSError:
            return None
        if not raw or len(raw) > PROC_CMDLINE_LIMIT:
            return None
        return tuple(
            part.decode("utf-8", "surrogateescape")
            for part in raw.split(b"\x00")
            if part
        )

    @staticmethod
    def _is_observer_argv(argv: Tuple[str, ...]) -> bool:
        foreground = len(argv) == 3 and argv[2] == "run"
        daemon = (
            len(argv) == 4
            and argv[2] == DAEMON_FLAG
            and argv[3].startswith(READY_FD_PREFIX)
            and argv[3][len(READY_FD_PREFIX) :].isdigit()
        )
        if not foreground and not daemon:
            return False
        try:
            return Path(argv[1]).resolve() == Path(__file__).resolve()
        except OSError:
            return False

    def _self_identity(self) -> ProcessIdentity:
        pid = os.getpid()
        stat = self._proc_stat(pid)
        argv = self._proc_argv(pid)
        if (
            self.boot_id == "unknown-boot-id"
            or stat is None
            or stat[0] == "Z"
            or argv is None
            or not self._is_observer_argv(argv)
        ):
            raise OSError(errno.EINVAL, "cannot establish exact observer identity")
        return ProcessIdentity(pid, self.boot_id, stat[1], argv, self.session_id)

    def _write_pidfile(self) -> ProcessIdentity:
        identity = self._self_identity()
        payload = {
            "version": PIDFILE_VERSION,
            "pid": identity.pid,
            "boot_id": identity.boot_id,
            "starttime": identity.starttime,
            "argv": list(identity.argv),
            "session_id": identity.session_id,
        }
        data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        temporary = self.state_dir / PIDFILE_TMP_NAME
        fd = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_TRUNC
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            os.fchmod(fd, 0o600)
            _write_all(fd, data)
            os.fsync(fd)
        except BaseException:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise
        finally:
            os.close(fd)
        published = False
        try:
            os.replace(temporary, self.pidfile_path)
            published = True
            self._sync_state_dir()
        except BaseException:
            if published:
                try:
                    self.pidfile_path.unlink()
                except OSError:
                    pass
                try:
                    self._sync_state_dir()
                except OSError:
                    pass
            else:
                try:
                    temporary.unlink()
                except OSError:
                    pass
            raise
        return identity

    def _read_pidfile(self) -> Optional[ProcessIdentity]:
        try:
            with open(self.pidfile_path, "rb") as handle:
                raw = handle.read(65537)
        except OSError:
            return None
        if len(raw) > 65536:
            return None
        try:
            data = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        required = {"version", "pid", "boot_id", "starttime", "argv", "session_id"}
        if not isinstance(data, dict) or set(data) != required:
            return None
        if data.get("version") != PIDFILE_VERSION:
            return None
        pid = data.get("pid")
        starttime = data.get("starttime")
        boot_id = data.get("boot_id")
        session_id = data.get("session_id")
        argv = data.get("argv")
        if (
            not isinstance(pid, int)
            or isinstance(pid, bool)
            or pid <= 0
            or not isinstance(starttime, int)
            or isinstance(starttime, bool)
            or starttime < 0
            or not isinstance(boot_id, str)
            or not boot_id
            or not isinstance(session_id, str)
            or not re.fullmatch(r"[0-9a-f]{32}", session_id)
            or not isinstance(argv, list)
            or not argv
            or not all(isinstance(part, str) for part in argv)
        ):
            return None
        return ProcessIdentity(pid, boot_id, starttime, tuple(argv), session_id)

    def _process_matches(self, identity: ProcessIdentity) -> bool:
        if identity.boot_id != self.boot_id or not self._is_observer_argv(identity.argv):
            return False
        stat = self._proc_stat(identity.pid)
        argv = self._proc_argv(identity.pid)
        return bool(
            stat is not None
            and stat[0] != "Z"
            and stat[1] == identity.starttime
            and argv == identity.argv
        )

    def _live_identity(self) -> Optional[ProcessIdentity]:
        identity = self._read_pidfile()
        return identity if identity is not None and self._process_matches(identity) else None

    def _forget_pidfile(self, session_id: Optional[str] = None) -> bool:
        if session_id is not None:
            current = self._read_pidfile()
            if current is None:
                try:
                    self.pidfile_path.stat()
                except FileNotFoundError:
                    return True  # the exiting daemon durably removed its own file
                except OSError:
                    return False
                return False  # present but malformed: never unlink ambiguous identity
            if current.session_id != session_id:
                return False
        try:
            self.pidfile_path.unlink()
        except FileNotFoundError:
            return True
        except OSError:
            return False
        try:
            self._sync_state_dir()
        except OSError:
            return False
        return True

    def _kernel_record(self, fact: dict) -> None:
        self._record("kernel", **fact)

    def sample_tick(self) -> None:
        mem_value, mem_error = read_mem_available_kib(str(self.proc_root / "meminfo"))
        fields = {}
        if mem_value is not None:
            fields["mem_available_kib"] = mem_value
        if mem_error:
            fields["meminfo_error"] = mem_error
        fields.update(read_memory_psi(str(self.proc_root / "pressure" / "memory")))
        vllm_count, vllm_error = count_vllm_serve_processes(str(self.proc_root))
        if vllm_count is not None:
            fields["vllm_proc_count"] = vllm_count
        if vllm_error:
            fields["vllm_liveness_error"] = vllm_error

        processed = self.journal.drain(self._kernel_record)
        healthy, journal_error = self.journal.health()
        if not healthy:
            if not self._journal_degraded_emitted:
                self._record(
                    "degraded",
                    source="journal",
                    detail=journal_error or "journal unavailable",
                )
                self._journal_degraded_emitted = True
            fields["journal_ok"] = False
            fields["journal_error"] = journal_error or "journal unavailable"
        else:
            self._journal_degraded_emitted = False
            fields["journal_ok"] = True

        journal_facts = self.journal.snapshot()
        fields.update(journal_facts)
        fields["journal_gap"] = bool(journal_facts["journal_gap"] or not healthy)
        fields["journal_processed_this_tick"] = processed
        fields["journal_drain_limit_hit"] = journal_facts["journal_backlog_items"] > 0
        self._record("sample", **fields)

    def settle_journal(self, budget_s: float = 0.5) -> None:
        deadline = time.monotonic() + budget_s
        while time.monotonic() < deadline:
            if self.journal.has_activity():
                return
            time.sleep(0.01)

    def _start_fields(self, *, mode: Optional[str] = None) -> dict:
        fields = {
            "pid": os.getpid(),
            "interval_s": self.config.interval_s,
            "journal_mode": "current_boot_future_only",
        }
        if mode is not None:
            fields["mode"] = mode
        return fields

    def _notify_ready(self) -> None:
        if self.ready_fd is None:
            return
        fd = self.ready_fd
        _write_all(fd, b"1")
        self.ready_fd = None
        try:
            os.close(fd)
        except OSError:
            pass

    def _close_ready_fd(self) -> None:
        if self.ready_fd is None:
            return
        try:
            os.close(self.ready_fd)
        except OSError:
            pass
        self.ready_fd = None

    def sample_once(self) -> int:
        try:
            self._ensure_state_dir()
            self.journal.start()
            self._record("session_start", **self._start_fields(mode="once"))
            self.settle_journal()
            self.sample_tick()
            self._record("session_end", reason="once")
        except OSError as exc:
            print(f"error: required observer record was not durable: {_safe_error('write failed', exc)}", file=sys.stderr)
            return 1
        finally:
            self.journal.close()
        return 0

    def run_loop(self) -> int:
        try:
            if not self.acquire_lock():
                print("observer: another instance holds the lock; exiting", file=sys.stderr)
                self._close_ready_fd()
                return 0
        except OSError as exc:
            print(f"error: cannot use state dir: {_safe_error('state failure', exc)}", file=sys.stderr)
            self._close_ready_fd()
            return 1
        signal.signal(signal.SIGTERM, self._request_stop)
        signal.signal(signal.SIGINT, self._request_stop)
        completed = False
        published_identity: Optional[ProcessIdentity] = None
        result = 0
        try:
            self.journal.start()
            self._record("session_start", **self._start_fields())
            self.sample_tick()
            published_identity = self._write_pidfile()
            try:
                self._notify_ready()
            except OSError:
                self._forget_pidfile(published_identity.session_id)
                published_identity = None
                raise
            while not self._stop_requested:
                if not self._sleep_interval():
                    break
                self.sample_tick()
            self._record(
                "session_end",
                reason="signal" if self._stop_requested else "shutdown",
            )
            completed = True
        except OSError as exc:
            print(f"error: observer recording halted: {_safe_error('write failed', exc)}", file=sys.stderr)
            result = 1
        finally:
            self.journal.close()
            if completed and published_identity is not None:
                if not self._forget_pidfile(published_identity.session_id):
                    result = 1
            self.release_lock()
            self._close_ready_fd()
        return result

    def _request_stop(self, signum, frame) -> None:
        self._stop_requested = True

    def _sleep_interval(self) -> bool:
        deadline = time.monotonic() + self.config.interval_s
        while not self._stop_requested:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            time.sleep(min(0.2, remaining))
        return False

    @staticmethod
    def _wait_ready_pipe(fd: int, timeout_s: float) -> bool:
        poller = select.poll()
        poller.register(fd, select.POLLIN | select.POLLHUP | select.POLLERR)
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if not poller.poll(max(1, math.ceil(remaining * 1000))):
                return False
            try:
                return os.read(fd, 1) == b"1"
            except InterruptedError:
                continue
            except OSError:
                return False

    def start_detached(self) -> int:
        existing = self._live_identity()
        if existing is not None:
            print(f"observer: already running (pid {existing.pid})")
            return 0
        ready_read: Optional[int] = None
        ready_write: Optional[int] = None
        try:
            self._ensure_state_dir()
            script = str(Path(__file__).resolve())
            executable = str(Path(sys.executable).resolve())
            daemon_environment = _minimal_daemon_environment()
            daemon_environment[ENV_STATE_DIR] = str(self.state_dir.resolve())
            journal_binary = self.config.journal_command[0]
            if os.sep in journal_binary and not os.path.isabs(journal_binary):
                journal_binary = str(Path(journal_binary).resolve())
            daemon_environment[ENV_JOURNALCTL] = journal_binary
            ready_read, ready_write = os.pipe()
            os.set_inheritable(ready_write, True)
            child = os.fork()
        except OSError as exc:
            for fd in (ready_read, ready_write):
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            print(
                f"error: cannot start observer: {_safe_error('fork/state failure', exc)}",
                file=sys.stderr,
            )
            return 1
        if child == 0:
            assert ready_read is not None and ready_write is not None
            try:
                os.close(ready_read)
                os.setsid()
                if os.fork() != 0:
                    os.close(ready_write)
                    os._exit(0)
                os.chdir("/")
                os.umask(0o077)
                devnull = os.open(os.devnull, os.O_RDWR)
                os.dup2(devnull, 0)
                os.dup2(devnull, 1)
                os.dup2(devnull, 2)
                if devnull > 2:
                    os.close(devnull)
                argv = [
                    executable,
                    script,
                    DAEMON_FLAG,
                    f"{READY_FD_PREFIX}{ready_write}",
                ]
                os.execve(executable, argv, daemon_environment)
            except BaseException:
                try:
                    os.close(ready_write)
                except OSError:
                    pass
                os._exit(1)
        assert ready_read is not None and ready_write is not None
        os.close(ready_write)
        try:
            os.waitpid(child, 0)
        except ChildProcessError:
            pass
        try:
            ready = self._wait_ready_pipe(ready_read, READY_TIMEOUT_S)
        finally:
            os.close(ready_read)
        if ready:
            identity = self._live_identity()
            if identity is not None:
                print(
                    f"observer: started (pid {identity.pid}); records {self.records_path}"
                )
                return 0
        print("error: observer did not become durably ready; inspect the state dir", file=sys.stderr)
        return 1

    @staticmethod
    def _pidfd_exited(pidfd: int, timeout_s: float) -> bool:
        poller = select.poll()
        poller.register(pidfd, select.POLLIN | select.POLLHUP | select.POLLERR)
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            remaining = deadline - time.monotonic()
            timeout_ms = 0 if remaining <= 0 else max(1, math.ceil(remaining * 1000))
            if poller.poll(timeout_ms):
                return True
            if remaining <= 0 or time.monotonic() >= deadline:
                return False

    def _signal_and_wait_pidfd(self, pidfd: int) -> bool:
        sender = getattr(signal, "pidfd_send_signal", None)
        if not callable(sender):
            return False
        try:
            sender(pidfd, signal.SIGTERM, None, 0)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        if self._pidfd_exited(pidfd, STOP_GRACE_S):
            return True
        try:
            sender(pidfd, signal.SIGKILL, None, 0)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        return self._pidfd_exited(pidfd, STOP_KILL_WAIT_S)

    def stop_running(self) -> int:
        identity = self._read_pidfile()
        if identity is None:
            print("observer: no usable pidfile; nothing to stop")
            self._forget_pidfile()
            return 0
        opener = getattr(os, "pidfd_open", None)
        sender = getattr(signal, "pidfd_send_signal", None)
        if not callable(opener) or not callable(sender):
            if self.boot_id == "unknown-boot-id":
                print(
                    "error: current boot ID is unavailable; observer was not signaled",
                    file=sys.stderr,
                )
                return 1
            if identity.boot_id != self.boot_id:
                print(
                    f"observer: stale pidfile from previous boot (pid {identity.pid}); removed"
                )
                self._forget_pidfile(identity.session_id)
                return 0
            if not self._process_matches(identity):
                print(f"observer: stale or replaced pid {identity.pid}; not signaled, entry removed")
                self._forget_pidfile(identity.session_id)
                return 0
            print("error: safe pidfd signaling is unavailable; observer was not signaled", file=sys.stderr)
            return 1
        try:
            pidfd = opener(identity.pid, 0)
        except ProcessLookupError:
            print(f"observer: stale pidfile (pid {identity.pid} not running); removed")
            self._forget_pidfile(identity.session_id)
            return 0
        except OSError as exc:
            print(f"error: cannot open a safe process handle: {_safe_error('pidfd_open failed', exc)}", file=sys.stderr)
            return 1
        try:
            if self.boot_id == "unknown-boot-id":
                print(
                    "error: current boot ID is unavailable; observer was not signaled",
                    file=sys.stderr,
                )
                return 1
            if identity.boot_id != self.boot_id:
                print(
                    f"observer: stale pidfile from previous boot (pid {identity.pid}); removed"
                )
                self._forget_pidfile(identity.session_id)
                return 0
            if not self._process_matches(identity):
                print(
                    f"observer: pid {identity.pid} identity does not exactly match the pidfile; "
                    "not signaled, stale entry removed"
                )
                self._forget_pidfile(identity.session_id)
                return 0
            if self._pidfd_exited(pidfd, 0):
                self._forget_pidfile(identity.session_id)
                print(f"observer: pid {identity.pid} already exited")
                return 0
            if not self._signal_and_wait_pidfd(pidfd):
                print(
                    f"error: pid {identity.pid} did not exit through its bound pidfd; "
                    "pidfile retained",
                    file=sys.stderr,
                )
                return 1
        finally:
            os.close(pidfd)
        if not self._forget_pidfile(identity.session_id):
            print("error: observer exited but its pidfile could not be durably removed", file=sys.stderr)
            return 1
        print(f"observer: stopped pid {identity.pid}")
        return 0

    def _iter_records_locked(self) -> Iterator[dict]:
        if not self.record_lock_path.exists():
            for path in self._record_paths(oldest_first=True):
                yield from self._records_from_path(path)
            return
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        try:
            lock_fd = os.open(self.record_lock_path, flags)
        except OSError:
            return
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_SH)
            for path in self._record_paths(oldest_first=True):
                yield from self._records_from_path(path)
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)

    def print_status(self) -> int:
        print(f"state_dir: {self.state_dir}")
        pidfile_identity = self._read_pidfile()
        live = (
            pidfile_identity
            if pidfile_identity is not None and self._process_matches(pidfile_identity)
            else None
        )
        if live is not None:
            print("running: yes")
            print(f"pid: {live.pid}")
        else:
            print("running: no")
            if pidfile_identity is not None:
                print(
                    f"stale_pidfile: pid {pidfile_identity.pid} "
                    f"boot {pidfile_identity.boot_id[:12]}"
                )
        try:
            size = self.records_path.stat().st_size
            print(f"records: {self.records_path} ({size} bytes active)")
        except OSError:
            print("records: none yet")

        target_session = pidfile_identity.session_id if pidfile_identity is not None else None
        session_start = None
        session_end = None
        last_sample = None
        selected_session = target_session
        for record in self._iter_records_locked():
            record_session = record.get("session_id")
            event = record.get("event")
            if target_session is None and event == "session_start" and isinstance(record_session, str):
                selected_session = record_session
                session_start = record.get("ts_utc")
                session_end = None
                last_sample = None
                continue
            if not isinstance(record_session, str) or record_session != selected_session:
                continue
            if event == "session_start":
                session_start = record.get("ts_utc")
                session_end = None
            elif event == "sample":
                last_sample = record
            elif event == "session_end":
                session_end = record.get("ts_utc")
        print(f"session_id: {selected_session or 'missing'}")
        if last_sample is not None:
            print(
                f"last_sample: {last_sample.get('ts_utc')} "
                f"journal_ok={last_sample.get('journal_ok')}"
            )
            if last_sample.get("journal_error"):
                print(f"journal_degradation: {last_sample['journal_error']}")
        print(f"session_start: {session_start or 'missing'}")
        print(f"session_end: {session_end or 'missing'}")
        return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help"):
        print(USAGE, end="")
        return 0
    try:
        config = parse_config()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    command = argv[0] if argv else ""
    ready_fd = None
    if command == DAEMON_FLAG:
        if len(argv) != 2 or not argv[1].startswith(READY_FD_PREFIX):
            print("error: invalid private daemon readiness handle", file=sys.stderr)
            return 2
        raw_fd = argv[1][len(READY_FD_PREFIX) :]
        try:
            ready_fd = int(raw_fd)
            if ready_fd < 3:
                raise ValueError
            os.fstat(ready_fd)
        except (OSError, ValueError):
            print("error: invalid private daemon readiness handle", file=sys.stderr)
            return 2
    elif argv[1:]:
        print(f"error: unexpected arguments: {' '.join(argv[1:])}", file=sys.stderr)
        return 2
    observer = Observer(config, ready_fd=ready_fd)
    if command in ("run", DAEMON_FLAG):
        return observer.run_loop()
    if command == "start":
        return observer.start_detached()
    if command == "stop":
        return observer.stop_running()
    if command == "status":
        return observer.print_status()
    if command == "once":
        return observer.sample_once()
    print(f"error: unknown command {command!r}", file=sys.stderr)
    print(USAGE, end="", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
