#!/usr/bin/env python3
"""Behavioral tests for the #27 partial-prefill admission-cap hotfix."""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
HOTFIX = ROOT / "patches" / "hotfix-dsv4-issue27-partial-prefill-concurrency.py"
ENV_NAME = "DSPARK_MAX_INFLIGHT_PREFILLS"

FIXTURE = """\
class Request:
    pass


class _Logger:
    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(message)


logger = _Logger()


class Scheduler:
    def __init__(self, config_cap=1):
        self.scheduler_config = type(
            "SchedulerConfig",
            (),
            {"max_num_partial_prefills": config_cap},
        )()
        self.max_num_running_reqs = 8
        self.num_waiting_for_streaming_input = 0
        self.running = []
        self.waiting = [object()]

        # In-flight requests still prefilling (prefill chunks + in-progress
        # async KV loads). Their remaining-block reservation gates async loads.
        self._inflight_prefills: set[Request] = set()

    def schedule(self):
        token_budget = 1
        admitted = 0
        can_schedule_waiting = True
        if can_schedule_waiting:
            while self.waiting and token_budget > 0:
                num_running = len(self.running) + self.num_waiting_for_streaming_input
                if num_running >= self.max_num_running_reqs:
                    break

                self.waiting.pop()
                admitted += 1
        return admitted
"""


def _apply_to(path: Path) -> None:
    txt = HOTFIX.read_text()
    marker = 'Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py")'
    txt = txt.replace(marker, f"Path({str(path)!r})")
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            exec(compile(txt, str(HOTFIX), "exec"), {})
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise


def _load_scheduler(raw: str | None, config_cap: int = 1):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "scheduler.py"
        path.write_text(FIXTURE)
        _apply_to(path)
        patched = path.read_text()

    namespace: dict = {}
    exec(compile(patched, str(path), "exec"), namespace)
    with mock.patch.dict(os.environ, {}, clear=False):
        if raw is None:
            os.environ.pop(ENV_NAME, None)
        else:
            os.environ[ENV_NAME] = raw
        scheduler = namespace["Scheduler"](config_cap)
    return scheduler, namespace["logger"], patched


class Issue27InflightCapTest(unittest.TestCase):
    def test_valid_and_fallback_values_are_cached(self):
        cases = (
            (None, 1),
            ("", 1),
            ("   ", 1),
            ("0", 1),
            ("-1", 1),
            ("1", 1),
            ("2", 2),
            ("3", 3),
            ("4", 3),
            (" 2 ", 2),
        )
        for raw, expected in cases:
            with self.subTest(raw=raw):
                scheduler, logger, _ = _load_scheduler(raw)
                self.assertEqual(
                    scheduler._dspark_max_inflight_prefills,
                    expected,
                )
                self.assertEqual(logger.warnings, [])

    def test_malformed_values_warn_and_use_config_fallback(self):
        for raw in ("two", "2.0", "1x"):
            with self.subTest(raw=raw):
                scheduler, logger, _ = _load_scheduler(raw, config_cap=2)
                self.assertEqual(scheduler._dspark_max_inflight_prefills, 2)
                self.assertEqual(len(logger.warnings), 1)
                self.assertIn(ENV_NAME, logger.warnings[0])

    def test_config_fallback_is_also_clamped(self):
        scheduler, _, _ = _load_scheduler(None, config_cap=4)
        self.assertEqual(scheduler._dspark_max_inflight_prefills, 3)

    def test_schedule_uses_cached_cap_without_rereading_environment(self):
        scheduler, _, _ = _load_scheduler("2")
        inflight = object()
        scheduler._inflight_prefills.add(inflight)
        with mock.patch.dict(os.environ, {ENV_NAME: "two"}):
            self.assertEqual(scheduler.schedule(), 1)
        self.assertIn(inflight, scheduler._inflight_prefills)

    def test_admission_stops_at_cached_cap(self):
        scheduler, _, _ = _load_scheduler("1")
        scheduler._inflight_prefills.add(object())
        self.assertEqual(scheduler.schedule(), 0)

    def test_apply_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scheduler.py"
            path.write_text(FIXTURE)
            _apply_to(path)
            patched = path.read_text()
            _apply_to(path)
            self.assertEqual(path.read_text(), patched)


if __name__ == "__main__":
    sys.exit(0 if unittest.main(verbosity=2) else 1)
