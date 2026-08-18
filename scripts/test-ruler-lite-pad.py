#!/usr/bin/env python3
"""CPU tests for RULER-lite padding (issue #81). No live endpoint."""
import importlib.util
import random
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "ruler-lite.py"
_spec = importlib.util.spec_from_file_location("ruler_lite", _SRC)
rl = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(rl)


class FakeTok:
    """HAYSTACK_SENTENCE = 24 tokens; everything else is 40 + 24 * copies."""

    unit = 24
    base = 40

    def __init__(self):
        self.calls = 0

    def __call__(self, text: str) -> int:
        self.calls += 1
        if text == rl.HAYSTACK_SENTENCE:
            return self.unit
        n = text.count(rl.HAYSTACK_SENTENCE)
        return self.base + n * self.unit


class TestHaystackReps(unittest.TestCase):
    def test_closes_a_large_gap_in_one_chunk(self):
        # Old loop: 200 * 24 = 4800. New reps for a 32k gap at 24 tok/unit.
        self.assertGreater(rl.haystack_reps(32768 - 40, 24), 200)
        self.assertEqual(rl.haystack_reps(24, 24), 2)


class TestPadToLength(unittest.TestCase):
    def test_reaches_32768(self):
        tok = FakeTok()
        padded = rl.pad_to_length("http://unused/v1", "m", "base prompt", 32768,
                                  tokenize_fn=tok)
        n = tok(padded)
        self.assertGreaterEqual(n, 32768)
        self.assertLess(tok.calls, 20, "must bulk-pad, not one sentence per call")

    def test_reaches_262144(self):
        tok = FakeTok()
        padded = rl.pad_to_length("http://unused/v1", "m", "base prompt", 262144,
                                  tokenize_fn=tok)
        self.assertGreaterEqual(tok(padded), 262144)

    def test_old_200_cap_would_miss_32k(self):
        tok = FakeTok()
        text = "base prompt"
        for _ in range(200):
            text += " " + rl.HAYSTACK_SENTENCE
        self.assertLess(tok(text), 32768)
        self.assertEqual(tok(text), 40 + 200 * 24)


class TestShortPadFails(unittest.TestCase):
    def test_stuck_tokenizer_raises_in_run_case(self):
        def stuck(*_a, **_k) -> int:
            return 100

        def boom(*_a, **_k):
            raise AssertionError("chat must not run when padding missed")

        orig_tok, orig_chat = rl.tokenize, rl.chat
        rl.tokenize = stuck
        rl.chat = boom
        try:
            with self.assertRaises(RuntimeError) as ctx:
                rl.run_case("http://unused/v1", "m", 32768, rl.task_sniah,
                            random.Random(0))
            self.assertIn("padding fell short", str(ctx.exception))
        finally:
            rl.tokenize = orig_tok
            rl.chat = orig_chat


if __name__ == "__main__":
    unittest.main()
