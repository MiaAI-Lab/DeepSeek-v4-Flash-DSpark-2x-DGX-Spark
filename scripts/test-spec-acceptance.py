#!/usr/bin/env python3
"""CPU regressions for spec-acceptance metric parsing and window reporting."""
import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "scripts" / "spec-acceptance.py"
FIXTURE = ROOT / "results" / "dspark-metrics.txt"
SPEC = importlib.util.spec_from_file_location("spec_acceptance", SRC)
sa = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(sa)


class _Response:
    def __init__(self, text: str):
        self.data = text.encode()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self):
        return self.data


class MetricParsingTest(unittest.TestCase):
    def parse(self, text: str) -> dict:
        original = sa.urllib.request.urlopen
        sa.urllib.request.urlopen = lambda *_a, **_k: _Response(text)
        try:
            return sa.get_metrics("http://unused/v1")
        finally:
            sa.urllib.request.urlopen = original

    def test_real_fixture_reads_last_position_label_and_ignores_created_gauges(self):
        metrics = self.parse(FIXTURE.read_text())
        self.assertEqual(metrics["drafts"], 105.0)
        self.assertEqual(metrics["per_pos"], {
            0: 84.0, 1: 63.0, 2: 1.0, 3: 1.0, 4: 0.0, 5: 0.0,
        })

    def test_position_label_order_does_not_matter(self):
        metrics = self.parse(
            'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="7",engine="0"} 9\n'
        )
        self.assertEqual(metrics["per_pos"], {7: 9.0})


class WindowReportingTest(unittest.TestCase):
    def run_main(self, before: dict, after: dict) -> str:
        snapshots = iter((before, after))
        original_metrics = sa.get_metrics
        original_run = sa.subprocess.run
        original_argv = sys.argv
        self.bench_calls = []
        sa.get_metrics = lambda _url: next(snapshots)
        sa.subprocess.run = lambda *_a, **_k: self.bench_calls.append(_a)
        sys.argv = ["spec-acceptance.py", "--trials", "2"]
        try:
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(sa.main(), 0)
            return output.getvalue()
        finally:
            sa.get_metrics = original_metrics
            sa.subprocess.run = original_run
            sys.argv = original_argv

    def test_reports_window_rates_from_counter_deltas(self):
        output = self.run_main(
            {"drafted": 500.0, "accepted": 300.0, "drafts": 100.0,
             "per_pos": {0: 90.0, 1: 70.0}},
            {"drafted": 550.0, "accepted": 334.0, "drafts": 110.0,
             "per_pos": {0: 99.0, 1: 77.0}},
        )
        self.assertIn("OVERALL ACCEPTANCE = 68.0%", output)
        self.assertIn("pos0: 0.900  (9/10)", output)
        self.assertIn("pos1: 0.700  (7/10)", output)

    def test_without_draft_count_falls_back_to_window_counts(self):
        output = self.run_main(
            {"drafted": 10.0, "accepted": 5.0, "drafts": None,
             "per_pos": {0: 4.0}},
            {"drafted": 15.0, "accepted": 8.0, "drafts": None,
             "per_pos": {0: 7.0}},
        )
        self.assertIn("pos0: 3 accepted", output)
        self.assertNotIn("pos0: 0.", output)

    def test_spec_decode_off_reports_without_running_benchmark(self):
        absent = {"drafted": None, "accepted": None, "drafts": None, "per_pos": {}}
        output = self.run_main(absent, absent)
        self.assertIn("NO draft counters", output)
        self.assertEqual(self.bench_calls, [])

    def test_counters_vanishing_after_burst_reports_instead_of_crashing(self):
        output = self.run_main(
            {"drafted": 500.0, "accepted": 300.0, "drafts": 100.0,
             "per_pos": {0: 90.0}},
            {"drafted": None, "accepted": None, "drafts": None, "per_pos": {}},
        )
        self.assertIn("NO draft counters", output)
        self.assertEqual(len(self.bench_calls), 1)


if __name__ == "__main__":
    unittest.main()
