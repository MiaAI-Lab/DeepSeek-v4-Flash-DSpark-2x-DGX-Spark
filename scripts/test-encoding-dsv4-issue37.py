#!/usr/bin/env python3
"""CPU regression tests for the bounded high/max encoder hotfix."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOTFIX = ROOT / "patches" / "hotfix-encoding-dsv4-issue37.py"


def _load_hotfix():
    spec = importlib.util.spec_from_file_location("hotfix_issue37", HOTFIX)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Issue37EncodingTest(unittest.TestCase):
    def setUp(self):
        self.hotfix = _load_hotfix()
        self.stock = (
            "header\n"
            + self.hotfix.OLD_PROMPTS
            + "\nbody\n"
            + self.hotfix.OLD_GUARD
            + "\ntail\n"
        )

    def test_replaces_both_anchors_atomically(self):
        updated, status = self.hotfix.patch_text(self.stock)
        self.assertEqual(status, "applied")
        self.assertIn(self.hotfix.NEW_PROMPTS, updated)
        self.assertIn(self.hotfix.NEW_GUARD, updated)
        self.assertNotIn("Do not stop reasoning", updated)
        self.assertNotIn("entire deliberation process", updated)
        self.assertIn("reasoning near 768 tokens", updated)
        self.assertIn("reasoning near 1536 tokens", updated)
        self.assertIn("Stop when further thought is unlikely", updated)
        self.assertIn('thinking_mode == "thinking" and not tools', updated)

    def test_is_idempotent(self):
        once, _ = self.hotfix.patch_text(self.stock)
        twice, status = self.hotfix.patch_text(once)
        self.assertEqual(status, "skipped")
        self.assertEqual(twice, once)

    def test_refuses_partial_or_unknown_source(self):
        partial = self.stock.replace(self.hotfix.OLD_GUARD, "unknown guard")
        updated, status = self.hotfix.patch_text(partial)
        self.assertEqual(status, "missing")
        self.assertEqual(updated, partial)

    def test_file_patch_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "encoding.py"
            path.write_text(self.stock, encoding="utf-8")
            self.assertEqual(self.hotfix.patch_file(path), "applied")
            self.assertEqual(self.hotfix.patch_file(path), "skipped")


if __name__ == "__main__":
    unittest.main()
