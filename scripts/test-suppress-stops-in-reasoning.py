#!/usr/bin/env python3
"""Unit tests for suppress-stops-in-reasoning hotfix (no live serve)."""
from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
HOTFIX = ROOT / "patches" / "hotfix-dsv4-suppress-stops-in-reasoning.py"


def _load():
    spec = importlib.util.spec_from_file_location("hotfix_stops", HOTFIX)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SuppressStopsPatchTest(unittest.TestCase):
    def setUp(self):
        self.mod = _load()

    def test_applies_anemll_stock(self):
        src = (
            self.mod.IMPORT_OLD
            + "PAD\n"
            + self.mod.FACTORY_OLD
            + "PAD\n"
            + self.mod.INIT_OLD
            + "PAD\n"
            + self.mod.STOP_OLD
            + "                output_text=self.output_text,\n"
        )
        new, status = self.mod.apply_text(src)
        self.assertEqual(status, "applied")
        self.assertIn(self.mod.MARK, new)
        self.assertIn("import os", new)
        self.assertIn("_maybe_enable_reasoning_stop_guard", new)
        self.assertIn("stop_check_offset = max(stop_check_offset", new)
        self.assertIn("not self._reasoning_stop_guard or self._reasoning_closed", new)

    def test_idempotent(self):
        src = (
            self.mod.IMPORT_OLD
            + self.mod.FACTORY_OLD
            + self.mod.INIT_OLD
            + self.mod.STOP_OLD
        )
        once, _ = self.mod.apply_text(src)
        twice, status = self.mod.apply_text(once)
        self.assertEqual(status, "skipped")
        self.assertEqual(once, twice)

    def test_missing_anchor(self):
        new, status = self.mod.apply_text("not a detokenizer\n")
        self.assertTrue(status.startswith("missing"))
        self.assertEqual(new, "not a detokenizer\n")

    def test_cli_fails_when_target_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "detokenizer.py"
            self.assertEqual(self.mod.main(["hotfix", str(missing)]), 1)

    def test_apply_live_anemll_snapshot_if_present(self):
        snap = Path("/tmp/anemll-detokenizer.py")
        if not snap.is_file():
            self.skipTest("no /tmp/anemll-detokenizer.py")
        src = snap.read_text(encoding="utf-8")
        if self.mod.MARK in src:
            self.skipTest("live snapshot already patched")
        new, status = self.mod.apply_text(src)
        self.assertEqual(status, "applied", status)
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
            tmp.write(new)
            path = tmp.name
        try:
            compile(new, path, "exec")
        finally:
            os.unlink(path)


class GuardSemanticsTest(unittest.TestCase):
    """CPU stand-in for the update() offset / dormant logic."""

    def test_offset_skips_think_tail_in_same_chunk(self):
        output = "foo Question: bar</think> answer"
        marker = "</think>"
        stop_check_offset = 0
        idx = output.find(marker, max(0, stop_check_offset - (len(marker) - 1)))
        self.assertNotEqual(idx, -1)
        stop_check_offset = max(stop_check_offset, idx + len(marker))
        tail = output[stop_check_offset:]
        self.assertNotIn("Question:", tail)
        self.assertIn("answer", tail)

    def test_env_default_on(self):
        mod = _load()
        src = (
            mod.IMPORT_OLD + mod.FACTORY_OLD + mod.INIT_OLD + mod.STOP_OLD
        )
        new, status = mod.apply_text(src)
        self.assertEqual(status, "applied")
        self.assertIn('os.environ.get(key) != "0"', new)


if __name__ == "__main__":
    unittest.main()
