#!/usr/bin/env python3
"""CPU tests for RULER-lite padding (issue #81) and the client HTTP timeout.

No live endpoint: /tokenize and /chat/completions are stubbed at urlopen.
"""
import argparse
import contextlib
import importlib.util
import io
import json
import random
import sys
import tempfile
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


class _FakeResponse(io.StringIO):
    """Context-manager stand-in for what urlopen returns."""

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class TestRequestTimeout(unittest.TestCase):
    """The 900 s default buys ~787k tokens at the recorded 874.8 prefill tok/s,
    so a ~900k case needs --request-timeout raised or the client hangs up
    mid-prefill and its own limit is scored as a model FAIL."""

    def _timeouts_seen(self, extra_args):
        """Run main() end to end with the socket stubbed; collect every timeout."""
        seen = []

        def fake_urlopen(req, timeout=None):
            seen.append(timeout)
            if req.full_url.endswith(rl.TOK_URL_SUFFIX):
                body = {"count": 10 ** 7}          # already past any target
            else:
                body = {"choices": [{"message": {"content": "unused"}}]}
            return _FakeResponse(json.dumps(body))

        orig_urlopen, orig_timeout, orig_argv = (
            rl.urllib.request.urlopen, rl.REQUEST_TIMEOUT, sys.argv)
        rl.urllib.request.urlopen = fake_urlopen
        try:
            with tempfile.TemporaryDirectory() as tmp:
                sys.argv = ["ruler-lite.py", "--lengths", "8192", "--tasks", "sniah",
                            "--output", str(Path(tmp) / "out.json")] + extra_args
                with contextlib.redirect_stdout(io.StringIO()):
                    rl.main()
        finally:
            rl.urllib.request.urlopen = orig_urlopen
            rl.REQUEST_TIMEOUT = orig_timeout
            sys.argv = orig_argv
        return seen

    def test_default_is_900_on_every_request(self):
        seen = self._timeouts_seen([])
        self.assertEqual(set(seen), {900.0})
        self.assertGreaterEqual(len(seen), 2, "tokenize and chat must both be covered")

    def test_flag_reaches_every_request(self):
        self.assertEqual(set(self._timeouts_seen(["--request-timeout", "3600"])),
                         {3600.0})


class TestPositiveTimeout(unittest.TestCase):
    def test_accepts_positive(self):
        self.assertEqual(rl.positive_timeout("3600"), 3600.0)
        self.assertEqual(rl.positive_timeout("0.5"), 0.5)

    def test_rejects_non_positive_and_non_finite(self):
        # A socket timeout trips instantly on 0/negative and OverflowErrors on
        # inf, which would resurface as a per-case FAIL rather than a usage error.
        for bad in ("0", "-1", "nan", "inf", "-inf"):
            with self.subTest(bad=bad):
                self.assertRaises(argparse.ArgumentTypeError, rl.positive_timeout, bad)

    def test_flag_is_wired_to_the_validator(self):
        orig_argv, orig_timeout = sys.argv, rl.REQUEST_TIMEOUT
        sys.argv = ["ruler-lite.py", "--request-timeout", "0"]
        try:
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as ctx:
                rl.main()
        finally:
            sys.argv, rl.REQUEST_TIMEOUT = orig_argv, orig_timeout
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("--request-timeout", err.getvalue())


if __name__ == "__main__":
    unittest.main()
