#!/usr/bin/env python3
"""Deterministic standard-library regressions for the Issue #32 observer.

The suite uses private /proc fixtures, finite byte streams, injected syscall
failures, and observer-owned subprocesses only.  It performs no network,
Docker, GPU, service, or live-API work.
"""
from __future__ import annotations

import contextlib
import errno
import importlib.util
import io
import json
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
OBSERVER = ROOT / "scripts" / "gb10-memory-observer.py"

FIXTURE_BOOT_ID = "11111111-2222-3333-4444-555555555555"
FIXTURE_MEMINFO = """\
MemTotal:       131072000 kB
MemFree:         12345678 kB
MemAvailable:    8388608 kB
Buffers:           123456 kB
Cached:          12345678 kB
"""
FIXTURE_PSI = """\
some avg10=12.50 avg60=6.25 avg300=3.13 total=1000
full avg10=0.00 avg60=0.00 avg300=0.00 total=0
"""
JOURNAL_051 = (
    "2026-08-13T05:35:27.123456+08:00 spark-head kernel: NVRM: "
    "Xid(0000:01:00): NV_ERR_NO_MEMORY (0x51) _memdescAllocInternal"
)


def load_module():
    name = "gb10_memory_observer_test"
    spec = importlib.util.spec_from_file_location(name, OBSERVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_observer(args, env_overrides, cwd=None, timeout=30, observer_path=OBSERVER):
    env = dict(os.environ)
    for name in list(env):
        if name.startswith("DSPARK_GB10_OBSERVER"):
            del env[name]
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, str(observer_path), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        timeout=timeout,
    )


class FakeJournal:
    def __init__(self, *, healthy=True, error=None, facts=None, snapshot=None, order=None):
        self.healthy = healthy
        self.error = error
        self.facts = list(facts or [])
        self.order = order
        self.started = False
        self.closed = False
        self._snapshot = {
            "journal_dropped_long_lines_total": 0,
            "journal_dropped_queue_full_total": 0,
            "journal_backlog_items": 0,
            "journal_gap": False,
            "journal_counter_saturated": False,
        }
        if snapshot:
            self._snapshot.update(snapshot)

    def start(self):
        self.started = True
        if self.order is not None:
            self.order.append("journal_start")

    def health(self):
        return self.healthy, self.error

    def has_activity(self):
        return True

    def drain(self, callback):
        processed = 0
        while self.facts and processed < 16:
            callback(self.facts.pop(0))
            processed += 1
        self._snapshot["journal_backlog_items"] = len(self.facts)
        return processed

    def snapshot(self):
        return dict(self._snapshot)

    def close(self):
        self.closed = True
        if self.order is not None:
            self.order.append("journal_close")


class FixtureCase(unittest.TestCase):
    def setUp(self):
        self.work = Path(tempfile.mkdtemp(prefix="gb10-observer-test-"))
        self.state_dir = self.work / "state"
        self.proc_root = self.work / "proc"
        self.write_proc_fixtures()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.work, ignore_errors=True)

    def write_proc_fixtures(self, meminfo=FIXTURE_MEMINFO, psi=FIXTURE_PSI):
        self.proc_root.mkdir(parents=True, exist_ok=True)
        (self.proc_root / "meminfo").write_text(meminfo, encoding="ascii")
        if psi is not None:
            (self.proc_root / "pressure").mkdir(exist_ok=True)
            (self.proc_root / "pressure" / "memory").write_text(psi, encoding="ascii")
        boot = self.proc_root / "sys" / "kernel" / "random"
        boot.mkdir(parents=True, exist_ok=True)
        (boot / "boot_id").write_text(FIXTURE_BOOT_ID + "\n", encoding="ascii")

    def base_env(self, extra=None):
        env = {
            "DSPARK_GB10_OBSERVER_STATE_DIR": str(self.state_dir),
            "DSPARK_GB10_OBSERVER_JOURNALCTL": "/bin/false",
        }
        if extra:
            env.update(extra)
        return env

    def config(self, module):
        return module.parse_config(self.base_env())

    def observer(self, module, journal=None):
        return module.Observer(
            self.config(module),
            proc_root=str(self.proc_root),
            journal=journal if journal is not None else FakeJournal(),
        )

    def read_records(self):
        records = []
        active = self.state_dir / "records.ndjson"
        for index in range(3, 0, -1):
            path = Path(f"{active}.{index}")
            if path.exists():
                records.extend(
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line
                )
        if active.exists():
            records.extend(
                json.loads(line)
                for line in active.read_text(encoding="utf-8").splitlines()
                if line
            )
        return records

    def write_process(self, pid, starttime, argv, *, state="S"):
        process = self.proc_root / str(pid)
        process.mkdir(parents=True, exist_ok=True)
        after_comm = [state] + ["0"] * 18 + [str(starttime)] + ["0"] * 4
        (process / "stat").write_text(
            f"{pid} (observer process) " + " ".join(after_comm) + "\n",
            encoding="ascii",
        )
        (process / "cmdline").write_bytes(
            b"\x00".join(part.encode("utf-8") for part in argv) + b"\x00"
        )

    def write_pidfile(self, module, pid, starttime, argv, session_id="a" * 32):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": module.PIDFILE_VERSION,
            "pid": pid,
            "boot_id": FIXTURE_BOOT_ID,
            "starttime": starttime,
            "argv": list(argv),
            "session_id": session_id,
        }
        (self.state_dir / module.PIDFILE_NAME).write_text(
            json.dumps(payload) + "\n", encoding="utf-8"
        )
        return payload


class ParserAndConfigTests(FixtureCase):
    def test_proc_parsers_use_exact_fixtures(self):
        module = load_module()
        value, error = module.read_mem_available_kib(str(self.proc_root / "meminfo"))
        self.assertEqual((value, error), (8388608, None))
        psi = module.read_memory_psi(str(self.proc_root / "pressure" / "memory"))
        self.assertEqual(psi["psi_some_avg10"], 12.5)
        self.assertEqual(psi["psi_full_avg300"], 0.0)
        missing, detail = module.read_mem_available_kib(str(self.work / "missing"))
        self.assertIsNone(missing)
        self.assertIn("unreadable", detail)

    def test_kernel_classification_is_sanitized_factual_and_filtered(self):
        module = load_module()
        fact = module.classify_kernel_line(JOURNAL_051 + "\x07" + "x" * 500)
        self.assertEqual(fact["types"], ["nvrm_err_no_memory_0x51", "xid"])
        self.assertNotIn("xid", fact)
        self.assertEqual(fact["journal_ts_utc"], "2026-08-12T21:35:27Z")
        self.assertLessEqual(len(fact["excerpt"]), module.EXCERPT_LIMIT)
        self.assertNotIn("\x07", fact["excerpt"])
        self.assertIsNone(
            module.classify_kernel_line("userspace: NV_ERR_NO_MEMORY (0x51)")
        )
        xid = module.classify_kernel_line(
            "2026-08-13T14:19:03Z host kernel: NVRM: Xid(0000): 79, fallen off"
        )
        self.assertEqual(xid["xid"], 79)

    def test_journal_command_is_current_boot_future_only(self):
        module = load_module()
        command = module.parse_config({}).journal_command
        self.assertEqual(
            command,
            ("journalctl", "-b", "0", "-k", "-f", "-o", "short-iso", "-n", "0"),
        )
        overridden = module.parse_config(
            {module.ENV_JOURNALCTL: "/private/test-journal"}
        ).journal_command
        self.assertEqual(overridden[0], "/private/test-journal")
        self.assertEqual(overridden[1:], command[1:])

    def test_nonfinite_and_nonpositive_intervals_are_configuration_errors(self):
        module = load_module()
        for value in ("0", "-1", "nan", "NaN", "inf", "+inf", "-inf", "1e9999", "oops"):
            with self.subTest(value=value):
                with self.assertRaises(module.ConfigError):
                    module.parse_config({module.ENV_INTERVAL: value})
                result = run_observer(
                    ["status"], self.base_env({module.ENV_INTERVAL: value})
                )
                self.assertEqual(result.returncode, 2, (value, result.stderr))
        self.assertEqual(module.parse_config({module.ENV_INTERVAL: ""}).interval_s, 2.0)

    def test_boolean_validation_keeps_lifecycle_defaults_out_of_config(self):
        module = load_module()
        config = module.parse_config({})
        self.assertFalse(hasattr(config, "enabled"))
        self.assertFalse(hasattr(config, "autostop"))
        for name in (module.ENV_ENABLED, module.ENV_AUTOSTOP):
            with self.assertRaises(module.ConfigError):
                module.parse_config({name: "yes"})

    def test_process_liveness_scan_has_a_hard_entry_budget(self):
        module = load_module()
        for pid in range(100, 106):
            process = self.proc_root / str(pid)
            process.mkdir()
            (process / "cmdline").write_bytes(b"python\x00worker\x00")
        count, error = module.count_vllm_serve_processes(
            str(self.proc_root), max_entries=2, budget_s=60.0
        )
        self.assertIsNone(count)
        self.assertEqual(error, "process scan budget exhausted")

    def test_daemon_and_journal_environment_allowlists_drop_credentials(self):
        module = load_module()
        sentinel = "SENTINEL-CREDENTIAL-DO-NOT-INHERIT"
        source = {
            "PATH": "/usr/bin",
            "HOME": "/home/test",
            "LANG": "C",
            "LC_ALL": "C",
            module.ENV_INTERVAL: "3",
            module.ENV_STATE_DIR: "/tmp/state",
            "HF_TOKEN": sentinel,
            "OPENAI_API_KEY": sentinel,
            "AWS_SECRET_ACCESS_KEY": sentinel,
            "LC_FAKE_SECRET": sentinel,
        }
        daemon = module._minimal_daemon_environment(source)
        follower = module._minimal_journal_environment(source)
        for environment in (daemon, follower):
            self.assertNotIn(sentinel, environment.values())
            self.assertNotIn("HF_TOKEN", environment)
            self.assertNotIn("LC_FAKE_SECRET", environment)
        self.assertEqual(daemon[module.ENV_INTERVAL], "3")
        self.assertNotIn(module.ENV_INTERVAL, follower)
        self.assertNotIn("HOME", follower)


class TrackingBytesIO(io.BytesIO):
    def __init__(self, initial):
        super().__init__(initial)
        self.read_sizes = []

    def readline(self, size=-1):
        self.read_sizes.append(size)
        return super().readline(size)


class FakePopen:
    def __init__(self, stdout, status=0):
        self.stdout = stdout
        self.returncode = status

    def poll(self):
        return self.returncode


class JournalBoundednessTests(FixtureCase):
    def test_reader_bounds_bytes_filters_before_queue_and_counts_all_loss(self):
        module = load_module()
        nonmatches = b"ordinary kernel line\n" * 1000
        overlong = b"NVRM: Xid(0): 79, " + b"x" * module.JOURNAL_LINE_LIMIT + b"\n"
        matches = b"".join(
            f"NVRM: Xid(0): {index}, fact\n".encode()
            for index in range(module.JOURNAL_QUEUE_MAX + 11)
        )
        stream = TrackingBytesIO(nonmatches + overlong + matches)
        source = module.JournalSource(("unused",))
        source._pump(FakePopen(stream, status=0))
        facts = source.snapshot()
        self.assertEqual(source.lines.qsize(), module.JOURNAL_QUEUE_MAX)
        self.assertTrue(all(isinstance(item, dict) for item in list(source.lines.queue)))
        self.assertEqual(facts["journal_dropped_long_lines_total"], 1)
        self.assertEqual(facts["journal_dropped_queue_full_total"], 11)
        self.assertTrue(facts["journal_gap"])
        self.assertTrue(stream.read_sizes)
        self.assertLessEqual(max(stream.read_sizes), module.JOURNAL_LINE_LIMIT + 1)
        healthy, detail = source.health()
        self.assertFalse(healthy, "clean status-0 EOF must never remain healthy")
        self.assertIn("status 0", detail)

    def test_nonzero_and_stream_eof_are_degraded_too(self):
        module = load_module()
        for status in (1, 42):
            source = module.JournalSource(("unused",))
            source._pump(FakePopen(TrackingBytesIO(b""), status=status))
            self.assertEqual(source.health()[0], False)
            self.assertIn(str(status), source.health()[1])

    def test_drain_has_deterministic_count_and_elapsed_budgets(self):
        module = load_module()
        source = module.JournalSource(("unused",))
        for index in range(5):
            source.lines.put_nowait({"types": ["xid"], "excerpt": str(index)})
        seen = []
        with mock.patch.object(module.time, "monotonic", side_effect=[0.0, 0.0, 0.1]):
            processed = source.drain(seen.append, max_items=4, budget_s=0.05)
        self.assertEqual(processed, 2)
        self.assertEqual(len(seen), 2)
        self.assertEqual(source.lines.qsize(), 3)

    def test_sample_surfaces_drop_backlog_gap_and_tick_budget_facts(self):
        module = load_module()
        source = module.JournalSource(("unused",))
        source.proc = mock.Mock()
        source.proc.poll.return_value = None
        source._increment("_dropped_long_lines")
        source._increment("_dropped_queue_full")
        for index in range(module.JOURNAL_ITEMS_PER_TICK + 3):
            source.lines.put_nowait(
                {"types": ["xid"], "xid": index, "excerpt": f"NVRM {index}"}
            )
        observer = self.observer(module, source)
        with mock.patch.object(module.time, "monotonic", return_value=0.0):
            observer.sample_tick()
        sample = [r for r in self.read_records() if r["event"] == "sample"][-1]
        self.assertEqual(sample["journal_dropped_long_lines_total"], 1)
        self.assertEqual(sample["journal_dropped_queue_full_total"], 1)
        self.assertEqual(sample["journal_processed_this_tick"], module.JOURNAL_ITEMS_PER_TICK)
        self.assertEqual(sample["journal_backlog_items"], 3)
        self.assertTrue(sample["journal_drain_limit_hit"])
        self.assertTrue(sample["journal_gap"])

    def test_drop_counters_saturate_instead_of_growing_without_bound(self):
        module = load_module()
        source = module.JournalSource(("unused",))
        source._dropped_long_lines = module.JOURNAL_COUNTER_MAX
        source._increment("_dropped_long_lines")
        facts = source.snapshot()
        self.assertEqual(
            facts["journal_dropped_long_lines_total"], module.JOURNAL_COUNTER_MAX
        )
        self.assertTrue(facts["journal_counter_saturated"])


class RecordAndSessionTests(FixtureCase):
    def test_once_uses_proc_fixtures_and_all_records_share_session_id(self):
        module = load_module()
        observer = self.observer(module, FakeJournal(healthy=False, error="finite EOF"))
        self.assertEqual(observer.sample_once(), 0)
        records = self.read_records()
        self.assertEqual(
            [record["event"] for record in records],
            ["session_start", "degraded", "sample", "session_end"],
        )
        session_ids = {record["session_id"] for record in records}
        self.assertEqual(session_ids, {observer.session_id})
        self.assertRegex(observer.session_id, r"^[0-9a-f]{32}$")
        sample = next(record for record in records if record["event"] == "sample")
        self.assertEqual(sample["mem_available_kib"], 8388608)
        self.assertEqual(sample["psi_some_avg10"], 12.5)
        self.assertFalse(sample["journal_ok"])
        self.assertTrue(sample["journal_gap"])
        for record in records:
            for field in (
                "schema",
                "event",
                "ts_utc",
                "monotonic_s",
                "boot_id",
                "hostname",
                "seq",
                "session_id",
            ):
                self.assertIn(field, record)

    def test_restart_gap_anchor_points_to_preceding_durable_record(self):
        module = load_module()
        first = self.observer(module, FakeJournal())
        self.assertEqual(first.sample_once(), 0)
        first_records = self.read_records()
        preceding = first_records[-1]["ts_utc"]
        second = self.observer(module, FakeJournal())
        self.assertEqual(second.sample_once(), 0)
        starts = [r for r in self.read_records() if r["event"] == "session_start"]
        self.assertIsNone(starts[0]["resume_after_ts_utc"])
        self.assertEqual(starts[1]["resume_after_ts_utc"], preceding)
        self.assertEqual(starts[1]["journal_mode"], "current_boot_future_only")
        serialized = json.dumps(starts[1])
        self.assertNotIn("journal_command", serialized)
        self.assertNotIn("/bin/false", serialized)

    def test_every_required_once_record_failure_returns_one(self):
        module = load_module()
        for target in ("session_start", "sample", "session_end"):
            with self.subTest(target=target):
                state = self.work / target
                config = module.parse_config(
                    {
                        module.ENV_STATE_DIR: str(state),
                        module.ENV_JOURNALCTL: "/bin/false",
                    }
                )
                observer = module.Observer(
                    config, proc_root=str(self.proc_root), journal=FakeJournal()
                )
                real_record = observer._record

                def injected(event, _target=target, **fields):
                    if event == _target:
                        raise OSError(errno.EIO, "injected required-record failure")
                    return real_record(event, **fields)

                with mock.patch.object(observer, "_record", side_effect=injected):
                    self.assertEqual(observer.sample_once(), 1)

    def test_full_write_loop_handles_partial_writes_and_rejects_zero(self):
        module = load_module()
        observer = self.observer(module)
        real_write = os.write
        calls = []

        def partial(fd, data):
            chunk = bytes(data[: max(1, len(data) // 3)])
            calls.append(len(chunk))
            return real_write(fd, chunk)

        with mock.patch.object(module.os, "write", side_effect=partial):
            observer._record("sample", marker="partial-write")
        self.assertGreater(len(calls), 1)
        self.assertEqual(self.read_records()[-1]["marker"], "partial-write")

        failing = self.observer(module)
        with mock.patch.object(module.os, "write", return_value=0):
            with self.assertRaises(OSError):
                failing._record("sample", marker="zero-write")

    def test_records_permission_failure_is_operational_failure(self):
        module = load_module()
        observer = self.observer(module)
        real_open = os.open

        def deny_records(path, *args, **kwargs):
            if os.fspath(path) == os.fspath(observer.records_path):
                raise OSError(errno.EACCES, "injected records denial")
            return real_open(path, *args, **kwargs)

        with mock.patch.object(module.os, "open", side_effect=deny_records):
            self.assertEqual(observer.sample_once(), 1)

    def test_prefix_then_error_rolls_back_before_next_successful_once(self):
        module = load_module()
        failed = self.observer(module)
        real_write = os.write
        calls = 0

        def prefix_then_error(fd, data):
            nonlocal calls
            calls += 1
            if calls == 1:
                return real_write(fd, bytes(data[: max(1, len(data) // 2)]))
            raise OSError(errno.ENOSPC, "injected after prefix")

        with mock.patch.object(module.os, "write", side_effect=prefix_then_error):
            self.assertEqual(failed.sample_once(), 1)
        recovered = self.observer(module)
        self.assertEqual(recovered.sample_once(), 0)
        paths = [self.state_dir / module.RECORDS_NAME] + [
            Path(f"{self.state_dir / module.RECORDS_NAME}.{index}")
            for index in range(1, module.ROTATION_KEEP)
        ]
        parsed = []
        for path in paths:
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                parsed.append(json.loads(line))
        self.assertEqual(
            [record["event"] for record in parsed],
            ["session_start", "sample", "session_end"],
        )
        self.assertTrue(all(record["session_id"] == recovered.session_id for record in parsed))

    def test_unterminated_tail_is_durably_repaired_before_append(self):
        module = load_module()
        initial = self.observer(module)
        initial._record("sample", marker="valid-before-tail")
        active = self.state_dir / module.RECORDS_NAME
        with open(active, "ab") as handle:
            handle.write(b'{"unterminated":')
            handle.flush()
            os.fdatasync(handle.fileno())
        recovered = self.observer(module)
        self.assertEqual(recovered.sample_once(), 0)
        lines = active.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines]
        self.assertEqual(records[0]["marker"], "valid-before-tail")
        self.assertEqual(records[-1]["event"], "session_end")

    def test_directory_sync_failure_is_retried_until_one_succeeds(self):
        module = load_module()
        first = self.observer(module)
        real_sync = first._sync_state_dir
        sync_calls = 0

        def fail_creation_sync_once():
            nonlocal sync_calls
            sync_calls += 1
            if sync_calls == 2:
                raise OSError(errno.EIO, "post-creation directory sync failed")
            real_sync()

        with mock.patch.object(first, "_sync_state_dir", side_effect=fail_creation_sync_once):
            self.assertEqual(first.sample_once(), 1)
        self.assertTrue((self.state_dir / module.RECORDS_NAME).exists())
        with mock.patch.object(
            module.os, "fsync", side_effect=OSError(errno.EIO, "directory sync still failed")
        ):
            self.assertEqual(self.observer(module).sample_once(), 1)
        recovered = self.observer(module)
        self.assertEqual(recovered.sample_once(), 0)
        records = self.read_records()
        self.assertTrue(all(isinstance(record, dict) for record in records))
        self.assertEqual(
            [record["event"] for record in records[-3:]],
            ["session_start", "sample", "session_end"],
        )

    def test_fdatasync_failure_is_not_reported_as_once_success(self):
        module = load_module()
        observer = self.observer(module)
        with mock.patch.object(
            module.os, "fdatasync", side_effect=OSError(errno.EIO, "injected sync")
        ):
            self.assertEqual(observer.sample_once(), 1)

    def test_startup_order_publishes_ready_only_after_journal_start_and_sample(self):
        module = load_module()
        order = []
        journal = FakeJournal(order=order)
        observer = self.observer(module, journal)
        real_record = observer._record

        def recording(event, **fields):
            order.append(event)
            return real_record(event, **fields)

        identity = module.ProcessIdentity(
            123, FIXTURE_BOOT_ID, 456, (sys.executable, str(OBSERVER), module.DAEMON_FLAG, f"{module.READY_FD_PREFIX}9"), observer.session_id
        )

        def publish():
            order.append("ready_pidfile")
            return identity

        with (
            mock.patch.object(module.signal, "signal"),
            mock.patch.object(observer, "_record", side_effect=recording),
            mock.patch.object(observer, "_write_pidfile", side_effect=publish),
            mock.patch.object(observer, "_sleep_interval", return_value=False),
            mock.patch.object(observer, "_forget_pidfile", return_value=True),
        ):
            self.assertEqual(observer.run_loop(), 0)
        self.assertLess(order.index("journal_start"), order.index("session_start"))
        self.assertLess(order.index("session_start"), order.index("sample"))
        self.assertLess(order.index("sample"), order.index("ready_pidfile"))


    def test_failed_first_sync_never_publishes_startup_readiness(self):
        module = load_module()
        observer = self.observer(module, FakeJournal())
        with (
            mock.patch.object(module.signal, "signal"),
            mock.patch.object(
                module.os, "fdatasync", side_effect=OSError(errno.EIO, "sync failed")
            ),
            mock.patch.object(observer, "_write_pidfile") as publish,
        ):
            self.assertEqual(observer.run_loop(), 1)
        publish.assert_not_called()
        self.assertFalse((self.state_dir / module.PIDFILE_NAME).exists())


    def test_post_rename_sync_failure_never_signals_start_readiness(self):
        module = load_module()
        launcher = self.observer(module)
        window_marker = self.work / "pidfile-visible-before-sync-failure"

        def fake_exec(executable, argv, environment):
            ready_fd = int(argv[-1][len(module.READY_FD_PREFIX) :])
            daemon = module.Observer(
                launcher.config,
                proc_root=str(self.proc_root),
                journal=FakeJournal(),
                ready_fd=ready_fd,
            )
            self.write_process(os.getpid(), 123456, tuple(argv))
            identity = module.ProcessIdentity(
                os.getpid(),
                FIXTURE_BOOT_ID,
                123456,
                tuple(argv),
                daemon.session_id,
            )
            daemon._self_identity = lambda: identity
            real_sync = daemon._sync_state_dir

            def fail_only_while_published():
                if daemon.pidfile_path.exists():
                    window_marker.write_text(str(os.getpid()), encoding="ascii")
                    time.sleep(0.15)
                    raise OSError(errno.EIO, "post-rename directory sync failure")
                real_sync()

            daemon._sync_state_dir = fail_only_while_published
            result = daemon.run_loop()
            os._exit(result)

        with mock.patch.object(module.os, "execve", side_effect=fake_exec):
            self.assertEqual(launcher.start_detached(), 1)
        self.assertTrue(window_marker.exists())
        self.assertFalse((self.state_dir / module.PIDFILE_NAME).exists())
        self.assertIsNone(launcher._live_identity())
        daemon_pid = int(window_marker.read_text(encoding="ascii"))
        deadline = time.monotonic() + 5
        live = True
        while time.monotonic() < deadline:
            try:
                raw = Path(f"/proc/{daemon_pid}/stat").read_bytes()
            except OSError:
                live = False
                break
            state = raw.rsplit(b")", 1)[-1].split()[0]
            if state == b"Z":
                live = False
                break
            time.sleep(0.01)
        self.assertFalse(live, "failed readiness daemon must exit")


class RotationAndConcurrencyTests(FixtureCase):
    def test_record_creation_and_rotation_sync_the_state_directory(self):
        module = load_module()
        observer = self.observer(module)
        real_fsync = os.fsync
        directory_syncs = []

        def tracking_fsync(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                directory_syncs.append(fd)
            return real_fsync(fd)

        old_limit = module.ROTATION_BYTES
        try:
            with mock.patch.object(module.os, "fsync", side_effect=tracking_fsync):
                observer._record("sample", marker="created")
                creation_syncs = len(directory_syncs)
                module.ROTATION_BYTES = (self.state_dir / module.RECORDS_NAME).stat().st_size + 128
                observer._record("sample", marker="rotated")
            self.assertGreaterEqual(creation_syncs, 1)
            self.assertGreater(len(directory_syncs), creation_syncs)
        finally:
            module.ROTATION_BYTES = old_limit

    def test_rotation_failure_halts_before_oversized_append(self):
        module = load_module()
        old_limit = module.ROTATION_BYTES
        module.ROTATION_BYTES = 512
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            active = self.state_dir / module.RECORDS_NAME
            active.write_bytes(b"x" * 499 + b"\n")
            before = active.stat().st_size
            observer = self.observer(module)
            with mock.patch.object(
                module.os, "replace", side_effect=OSError(errno.EACCES, "denied")
            ):
                with self.assertRaises(OSError):
                    observer._record("sample", marker="must-not-append")
            self.assertEqual(active.stat().st_size, before)
            self.assertFalse(Path(f"{active}.1").exists())
        finally:
            module.ROTATION_BYTES = old_limit

    def test_concurrent_daemon_and_once_writers_remain_json_and_bounded(self):
        module = load_module()
        old_limit = module.ROTATION_BYTES
        module.ROTATION_BYTES = 2048
        errors = []
        barrier = threading.Barrier(4)

        def daemon_writer():
            try:
                observer = self.observer(module)
                barrier.wait(timeout=5)
                for index in range(24):
                    observer._record(
                        "sample", writer="daemon", index=index, padding="d" * 80
                    )
            except BaseException as exc:
                errors.append(exc)

        def once_writer(name):
            try:
                observer = self.observer(module, FakeJournal())
                barrier.wait(timeout=5)
                self.assertEqual(observer.sample_once(), 0)
                observer._record("sample", writer=name, padding="o" * 80)
            except BaseException as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=daemon_writer),
            threading.Thread(target=once_writer, args=("once-a",)),
            threading.Thread(target=once_writer, args=("once-b",)),
        ]
        try:
            for thread in threads:
                thread.start()
            barrier.wait(timeout=5)
            for thread in threads:
                thread.join(timeout=10)
            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(errors, [])
            final = self.observer(module)
            final._record("sample", writer="final-marker")

            active = self.state_dir / module.RECORDS_NAME
            paths = [active] + [Path(f"{active}.{index}") for index in range(1, 4)]
            self.assertTrue(active.exists())
            self.assertFalse(Path(f"{active}.4").exists())
            for path in paths:
                if not path.exists():
                    continue
                self.assertLessEqual(path.stat().st_size, module.ROTATION_BYTES)
                for line in path.read_text(encoding="utf-8").splitlines():
                    json.loads(line)
            self.assertIn("final-marker", active.read_text(encoding="utf-8"))
        finally:
            module.ROTATION_BYTES = old_limit


class PidfdIdentityTests(FixtureCase):
    def observer_argv(self, module):
        return (
            str(Path(sys.executable).resolve()),
            str(OBSERVER.resolve()),
            module.DAEMON_FLAG,
            f"{module.READY_FD_PREFIX}9",
        )

    def test_pidfile_is_atomic_json_with_all_exact_identity_fields(self):
        module = load_module()
        observer = self.observer(module)
        identity = module.ProcessIdentity(
            4321, FIXTURE_BOOT_ID, 987654, self.observer_argv(module), observer.session_id
        )
        observer._ensure_state_dir()
        with mock.patch.object(observer, "_self_identity", return_value=identity):
            self.assertEqual(observer._write_pidfile(), identity)
        data = json.loads((self.state_dir / module.PIDFILE_NAME).read_text())
        self.assertEqual(data["boot_id"], FIXTURE_BOOT_ID)
        self.assertEqual(data["starttime"], 987654)
        self.assertEqual(tuple(data["argv"]), identity.argv)
        self.assertEqual(data["session_id"], observer.session_id)
        self.assertEqual(list(self.state_dir.glob("*.tmp")), [])

    def test_safe_signaling_unavailable_refuses_and_retains_live_pidfile(self):
        module = load_module()
        observer = self.observer(module)
        pid, starttime = 42001, 111
        argv = self.observer_argv(module)
        self.write_process(pid, starttime, argv)
        self.write_pidfile(module, pid, starttime, argv)
        with (
            mock.patch.object(module.os, "pidfd_open", None, create=True),
            mock.patch.object(module.signal, "pidfd_send_signal", None, create=True),
        ):
            self.assertEqual(observer.stop_running(), 1)
        self.assertTrue((self.state_dir / module.PIDFILE_NAME).exists())

    def test_exact_starttime_and_argv_mismatch_are_never_signaled(self):
        module = load_module()
        observer = self.observer(module)
        pid, starttime = 42002, 222
        argv = self.observer_argv(module)
        self.write_process(pid, starttime + 1, argv)
        self.write_pidfile(module, pid, starttime, argv)
        fake_fd = os.open(os.devnull, os.O_RDONLY)
        calls = []
        with (
            mock.patch.object(module.os, "pidfd_open", return_value=fake_fd, create=True),
            mock.patch.object(
                module.signal, "pidfd_send_signal", side_effect=lambda *args: calls.append(args)
            ),
        ):
            self.assertEqual(observer.stop_running(), 0)
        self.assertEqual(calls, [])
        self.assertFalse((self.state_dir / module.PIDFILE_NAME).exists())

    def test_replacement_between_term_and_kill_can_only_receive_pidfd_signals(self):
        module = load_module()
        observer = self.observer(module)
        pid, starttime = 42003, 333
        argv = self.observer_argv(module)
        self.write_process(pid, starttime, argv)
        self.write_pidfile(module, pid, starttime, argv)
        fake_fd = os.open(os.devnull, os.O_RDONLY)
        sent = []

        def send(bound_fd, sig, siginfo, flags):
            sent.append((bound_fd, sig))
            if sig == signal.SIGTERM:
                self.write_process(pid, starttime + 99, ("/usr/bin/foreign", "replacement"))

        with (
            mock.patch.object(module.os, "pidfd_open", return_value=fake_fd, create=True),
            mock.patch.object(module.signal, "pidfd_send_signal", side_effect=send),
            mock.patch.object(observer, "_pidfd_exited", side_effect=[False, False, True]),
            mock.patch.object(module.os, "kill", side_effect=AssertionError("numeric signal"), create=True) as numeric,
        ):
            self.assertEqual(observer.stop_running(), 0)
        self.assertEqual(sent, [(fake_fd, signal.SIGTERM), (fake_fd, signal.SIGKILL)])
        numeric.assert_not_called()
        self.assertFalse((self.state_dir / module.PIDFILE_NAME).exists())

    def test_unconfirmed_pidfd_kill_is_failure_and_retains_identity(self):
        module = load_module()
        observer = self.observer(module)
        pid, starttime = 42004, 444
        argv = self.observer_argv(module)
        self.write_process(pid, starttime, argv)
        self.write_pidfile(module, pid, starttime, argv)
        fake_fd = os.open(os.devnull, os.O_RDONLY)
        with (
            mock.patch.object(module.os, "pidfd_open", return_value=fake_fd, create=True),
            mock.patch.object(module.signal, "pidfd_send_signal", create=True),
            mock.patch.object(observer, "_pidfd_exited", side_effect=[False, False, False]),
        ):
            self.assertEqual(observer.stop_running(), 1)
        self.assertTrue((self.state_dir / module.PIDFILE_NAME).exists())

    def test_pidfd_is_opened_before_live_identity_validation(self):
        module = load_module()
        observer = self.observer(module)
        pid, starttime = 42005, 555
        argv = self.observer_argv(module)
        self.write_process(pid, starttime, argv)
        self.write_pidfile(module, pid, starttime, argv)
        fake_fd = os.open(os.devnull, os.O_RDONLY)
        order = []
        real_matches = observer._process_matches

        def opened(*args):
            order.append("pidfd_open")
            return fake_fd

        def matched(identity):
            order.append("identity_validation")
            return real_matches(identity)

        with (
            mock.patch.object(module.os, "pidfd_open", side_effect=opened),
            mock.patch.object(module.signal, "pidfd_send_signal", create=True),
            mock.patch.object(observer, "_process_matches", side_effect=matched),
            mock.patch.object(observer, "_pidfd_exited", return_value=True),
        ):
            self.assertEqual(observer.stop_running(), 0)
        self.assertLess(order.index("pidfd_open"), order.index("identity_validation"))


class StatusCorrelationTests(FixtureCase):
    def test_overlapping_once_cannot_mask_live_or_crashed_daemon_session(self):
        module = load_module()
        old_limit = module.ROTATION_BYTES
        module.ROTATION_BYTES = 850
        self.addCleanup(setattr, module, "ROTATION_BYTES", old_limit)
        daemon = self.observer(module)
        daemon._record("session_start", pid=43001, journal_mode="current_boot_future_only")
        daemon._record("sample", journal_ok=True)
        once = self.observer(module)
        once._record("session_start", mode="once", journal_mode="current_boot_future_only")
        once._record("sample", journal_ok=False)
        once._record("session_end", reason="once")
        daemon._record("sample", journal_ok=True, marker="daemon-after-once")

        argv = (
            str(Path(sys.executable).resolve()),
            str(OBSERVER.resolve()),
            module.DAEMON_FLAG,
            f"{module.READY_FD_PREFIX}9",
        )
        self.write_process(43001, 1234, argv)
        self.write_pidfile(module, 43001, 1234, argv, daemon.session_id)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            daemon.print_status()
        live = output.getvalue()
        self.assertIn("running: yes", live)
        self.assertIn(f"session_id: {daemon.session_id}", live)
        self.assertIn("session_end: missing", live)
        self.assertNotIn(f"session_id: {once.session_id}", live)

        import shutil

        shutil.rmtree(self.proc_root / "43001")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            daemon.print_status()
        crashed = output.getvalue()
        self.assertIn("running: no", crashed)
        self.assertIn(f"session_id: {daemon.session_id}", crashed)
        self.assertIn("session_end: missing", crashed)
        self.assertIn("daemon-after-once", json.dumps(self.read_records()))


class DetachedProcessSafetyTests(FixtureCase):
    @staticmethod
    def process_exited(pid):
        try:
            raw = Path(f"/proc/{pid}/stat").read_bytes()
        except OSError:
            return True
        fields = raw.rsplit(b")", 1)[-1].split()
        return bool(fields) and fields[0] == b"Z"

    def wait_path(self, path, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.exists() and path.stat().st_size:
                return
            time.sleep(0.02)
        self.fail(f"timed out waiting for {path}")

    def wait_exit(self, pid, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process_exited(pid):
                return
            time.sleep(0.02)
        self.fail(f"pid {pid} did not exit")

    def test_detached_env_cwd_umask_pdeathsig_and_restart_cleanup(self):
        module = load_module()
        child_log = self.work / "journal-pids"
        env_log = self.work / "journal-env.json"
        cwd_log = self.work / "journal-cwd"
        stub = self.work / "journal-follower"
        stub.write_text(
            f"#!{sys.executable}\n"
            "import json, os, time\n"
            f"with open({str(child_log)!r}, 'a', encoding='ascii') as h:\n"
            "    h.write(str(os.getpid()) + '\\n'); h.flush(); os.fsync(h.fileno())\n"
            f"with open({str(env_log)!r}, 'w', encoding='utf-8') as h:\n"
            "    json.dump(dict(os.environ), h); h.flush(); os.fsync(h.fileno())\n"
            f"with open({str(cwd_log)!r}, 'w', encoding='utf-8') as h:\n"
            "    h.write(os.getcwd()); h.flush(); os.fsync(h.fileno())\n"
            "while True:\n"
            "    time.sleep(1)\n",
            encoding="utf-8",
        )
        stub.chmod(0o755)
        sentinel = "SENTINEL-OBSERVER-CREDENTIAL-32"
        env = self.base_env(
            {
                module.ENV_JOURNALCTL: str(stub),
                module.ENV_INTERVAL: "0.2",
                "HF_TOKEN": sentinel,
                "OPENAI_API_KEY": sentinel,
                "AWS_SECRET_ACCESS_KEY": sentinel,
            }
        )
        daemon_pids = []
        child_pids = []
        daemon_handles = []
        child_handles = []
        try:
            started = run_observer(["start"], env)
            self.assertEqual(started.returncode, 0, started.stderr)
            pidfile = json.loads((self.state_dir / module.PIDFILE_NAME).read_text())
            daemon_pid = pidfile["pid"]
            daemon_pids.append(daemon_pid)
            daemon_handles.append(os.pidfd_open(daemon_pid, 0))
            self.wait_path(child_log)
            self.wait_path(env_log)
            self.wait_path(cwd_log)
            child_pid = int(child_log.read_text().splitlines()[-1])
            child_pids.append(child_pid)
            child_handles.append(os.pidfd_open(child_pid, 0))

            daemon_environ = Path(f"/proc/{daemon_pid}/environ").read_bytes()
            self.assertNotIn(sentinel.encode(), daemon_environ)
            self.assertNotIn(b"HF_TOKEN=", daemon_environ)
            daemon_names = {
                item.partition(b"=")[0].decode("ascii")
                for item in daemon_environ.split(b"\x00")
                if item
            }
            allowed_daemon = set(module._LOCALE_ENV_NAMES) | {
                "HOME",
                "XDG_STATE_HOME",
            } | set(module._OBSERVER_ENV_NAMES)
            self.assertLessEqual(daemon_names, allowed_daemon)
            follower_env = json.loads(env_log.read_text())
            self.assertNotIn(sentinel, follower_env.values())
            self.assertNotIn("HF_TOKEN", follower_env)
            self.assertNotIn(module.ENV_STATE_DIR, follower_env)
            allowed_follower = {
                "PATH", "LANG", "LANGUAGE", "TZ", "LC_ALL", "LC_CTYPE",
                "LC_NUMERIC", "LC_TIME", "LC_COLLATE", "LC_MONETARY",
                "LC_MESSAGES", "LC_PAPER", "LC_NAME", "LC_ADDRESS",
                "LC_TELEPHONE", "LC_MEASUREMENT", "LC_IDENTIFICATION",
            }
            self.assertLessEqual(set(follower_env), allowed_follower)
            self.assertEqual(os.readlink(f"/proc/{daemon_pid}/cwd"), "/")
            status_text = Path(f"/proc/{daemon_pid}/status").read_text()
            self.assertRegex(status_text, r"(?m)^Umask:\s+0077$")
            self.assertEqual(cwd_log.read_text(), "/")
            records_blob = b"".join(
                path.read_bytes()
                for path in self.state_dir.glob("records.ndjson*")
                if path.is_file()
            )
            self.assertNotIn(sentinel.encode(), records_blob)

            signal.pidfd_send_signal(daemon_handles[-1], signal.SIGKILL, None, 0)
            self.wait_exit(daemon_pid)
            self.wait_exit(child_pid)

            env_log.unlink(missing_ok=True)
            cwd_log.unlink(missing_ok=True)
            restarted = run_observer(["start"], env)
            self.assertEqual(restarted.returncode, 0, restarted.stderr)
            second = json.loads((self.state_dir / module.PIDFILE_NAME).read_text())
            daemon_pids.append(second["pid"])
            daemon_handles.append(os.pidfd_open(second["pid"], 0))
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                lines = child_log.read_text().splitlines()
                if len(lines) >= 2:
                    break
                time.sleep(0.02)
            self.assertEqual(len(lines), 2)
            second_child = int(lines[-1])
            child_pids.append(second_child)
            child_handles.append(os.pidfd_open(second_child, 0))
            self.assertNotEqual(second_child, child_pid)
            self.assertTrue(self.process_exited(child_pid))
            self.assertFalse(self.process_exited(second_child))

            stopped = run_observer(["stop"], env)
            self.assertEqual(stopped.returncode, 0, stopped.stderr)
            self.wait_exit(second_child)
        finally:
            run_observer(["stop"], env)
            for pid, handle in list(zip(daemon_pids, daemon_handles)) + list(
                zip(child_pids, child_handles)
            ):
                try:
                    if not self.process_exited(pid):
                        signal.pidfd_send_signal(handle, signal.SIGKILL, None, 0)
                except OSError:
                    pass
                finally:
                    os.close(handle)


class CliAndSourceAuditTests(FixtureCase):
    def test_cli_names_exit_contract_and_durable_start_sample(self):
        module = load_module()
        unknown = run_observer(["unknown"], self.base_env())
        extra = run_observer(["status", "extra"], self.base_env())
        self.assertEqual(unknown.returncode, 2)
        self.assertEqual(extra.returncode, 2)
        absent_status = run_observer(["status"], self.base_env())
        absent_stop = run_observer(["stop"], self.base_env())
        self.assertEqual(absent_status.returncode, 0)
        self.assertEqual(absent_stop.returncode, 0)

        started = run_observer(
            ["start"],
            self.base_env({module.ENV_INTERVAL: "0.2"}),
            cwd=ROOT,
            observer_path=Path("scripts/gb10-memory-observer.py"),
        )
        try:
            self.assertEqual(started.returncode, 0, started.stderr)
            records = self.read_records()
            self.assertIn("session_start", [record["event"] for record in records])
            self.assertIn("sample", [record["event"] for record in records])
            identity = json.loads((self.state_dir / module.PIDFILE_NAME).read_text())
            session_records = [
                record
                for record in records
                if record["session_id"] == identity["session_id"]
            ]
            self.assertIn("sample", [record["event"] for record in session_records])
        finally:
            stopped = run_observer(["stop"], self.base_env())
        self.assertEqual(stopped.returncode, 0, stopped.stderr)

    def test_report_only_source_has_no_numeric_signal_or_remediation_path(self):
        source = OBSERVER.read_text(encoding="utf-8")
        self.assertNotIn("os.kill(", source)
        for pattern in (
            r"\bdocker(?:-compose)?\b",
            r"\bcurl\b|\bwget\b|https?://",
            r"\bssh\b|\bscp\b|\brsync\b",
            r"\bsudo\b|\bpkexec\b",
            r"nvidia-smi",
            r"kill_rank|abort_serve|restart_serv|threshold_action",
        ):
            self.assertIsNone(re.search(pattern, source, re.IGNORECASE), pattern)
        self.assertIn("pidfd_send_signal", source)
        self.assertIn("PR_SET_PDEATHSIG", source)
        journal_close = source[
            source.index("    def close(self) -> None:") : source.index("def _write_all")
        ]
        for child_signal in ("proc.terminate()", "proc.kill()"):
            self.assertEqual(source.count(child_signal), 1)
            self.assertIn(child_signal, journal_close)

    def test_ci_registration_remains(self):
        ci = (ROOT / "scripts" / "ci-validate.sh").read_text(encoding="utf-8")
        self.assertIn("scripts/gb10-memory-observer.py", ci)
        self.assertIn("scripts/test-gb10-memory-observer.py", ci)


if __name__ == "__main__":
    unittest.main()
