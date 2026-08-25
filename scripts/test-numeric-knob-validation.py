#!/usr/bin/env python3
"""Tests for the shared numeric-knob validation (dspark-numeric-knobs.sh).

Exercises the real sourced function — not an extracted copy — so both the launcher
and the validator, which source the same file, are covered. Checks: non-integer
rejection, empty/default passthrough, leading-zero decimal normalisation (010 -> 10,
never octal), overflow/oversize rejection (no 64-bit wrap to negative), real CRLF,
per-knob bounds, and the worker env-snapshot writeback.

    python3 scripts/test-numeric-knob-validation.py -q

CPU-only; no GPU, container, or network.
"""
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HELPER = os.path.join(ROOT, "dspark-numeric-knobs.sh")
LAUNCHER = os.path.join(ROOT, "start-deepseek-v4-flash-dspark.sh")
VALIDATOR = os.path.join(ROOT, "validate-dspark-config.sh")


def call(var, val, snapshot=None):
    """Set <var>=<val> (real bytes via env), source the helper, run the function."""
    env = dict(os.environ, **{var: val})
    snap = f'"{snapshot}"' if snapshot else ""
    script = f"""set -euo pipefail
source {HELPER!r}
dspark_validate_numeric_knobs {snap}
printf 'OK %s=%s cap=%s\\n' {var!r} "${{{var}:-<unset>}}" \
  "$(( ${{MAX_NUM_SEQS:-6}} * (${{MTP_NUM_TOKENS:-5}} + 1) ))"
"""
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    return p.returncode, p.stdout + p.stderr


class Accepts(unittest.TestCase):
    def test_valid(self):
        rc, out = call("MAX_NUM_SEQS", "6")
        self.assertEqual(rc, 0, out); self.assertIn("MAX_NUM_SEQS=6", out); self.assertIn("cap=36", out)

    def test_empty_uses_default(self):
        rc, out = call("MAX_NUM_SEQS", "")
        self.assertEqual(rc, 0, out); self.assertIn("cap=36", out)

    def test_leading_zero_is_decimal_not_octal(self):
        rc, out = call("MAX_NUM_SEQS", "010")
        self.assertEqual(rc, 0, out); self.assertIn("MAX_NUM_SEQS=10", out); self.assertIn("cap=60", out)

    def test_leading_zero_eight(self):
        rc, out = call("MAX_NUM_SEQS", "08")
        self.assertEqual(rc, 0, out); self.assertIn("MAX_NUM_SEQS=8", out); self.assertIn("cap=48", out)

    def test_upper_bound_ok(self):
        rc, out = call("MAX_NUM_SEQS", "4096")
        self.assertEqual(rc, 0, out); self.assertIn("MAX_NUM_SEQS=4096", out)


class Rejects(unittest.TestCase):
    def _reject(self, var, val, needle="must be"):
        rc, out = call(var, val)
        self.assertEqual(rc, 2, f"expected reject, got rc={rc}: {out}")
        self.assertIn(needle, out, out)
        # never leak a wrapped/negative value or continue
        self.assertNotIn("cap=-", out); self.assertNotIn("OK ", out)

    def test_decimal(self):        self._reject("MAX_NUM_SEQS", "6.5", "non-negative integer")
    def test_alnum(self):          self._reject("MAX_NUM_SEQS", "6x", "non-negative integer")
    def test_bareword(self):       self._reject("MAX_NUM_SEQS", "eight", "non-negative integer")
    def test_mtp_bareword(self):   self._reject("MTP_NUM_TOKENS", "five", "non-negative integer")
    def test_batched_bad(self):    self._reject("MAX_NUM_BATCHED_TOKENS", "bad", "non-negative integer")

    def test_real_carriage_return(self):
        # a genuine \r byte (not literal backslash-r) — CRLF from a Windows-edited env
        self._reject("MAX_NUM_SEQS", "6\r", "non-negative integer")

    def test_overflow_2p64_minus_1(self):
        # the documented case: must NOT wrap to -1 and exit 0
        self._reject("MAX_NUM_SEQS", "18446744073709551615", "must be between")

    def test_overflow_2p63(self):
        self._reject("MAX_NUM_SEQS", "9223372036854775808", "must be between")

    def test_over_per_knob_max(self):
        self._reject("MAX_NUM_SEQS", "5000", "must be between")     # > 4096

    def test_zero_seqs_rejected(self):
        self._reject("MAX_NUM_SEQS", "0", "must be between")        # min is 1


class SnapshotWriteback(unittest.TestCase):
    """The load-bearing worker sync: the passed snapshot must be normalised in place."""
    def _snapshot(self, line):
        fd, path = tempfile.mkstemp(suffix=".env")
        with os.fdopen(fd, "w") as f:
            f.write("WORKER_HOST=w\n" + line + "\nMASTER_ADDR=127.0.0.1\n")
        return path

    def test_snapshot_normalised_to_decimal(self):
        snap = self._snapshot("MAX_NUM_SEQS=010")
        try:
            rc, out = call("MAX_NUM_SEQS", "010", snapshot=snap)
            self.assertEqual(rc, 0, out)
            with open(snap) as f:
                body = f.read()
            self.assertIn("MAX_NUM_SEQS=10", body, "snapshot not normalised for the worker")
            self.assertNotIn("MAX_NUM_SEQS=010", body)
        finally:
            os.unlink(snap)

    def test_snapshot_untouched_when_valid(self):
        snap = self._snapshot("MAX_NUM_SEQS=6")
        try:
            rc, _ = call("MAX_NUM_SEQS", "6", snapshot=snap)
            self.assertEqual(rc, 0)
            with open(snap) as f:
                self.assertIn("MAX_NUM_SEQS=6", f.read())
        finally:
            os.unlink(snap)


class WiredIn(unittest.TestCase):
    """Guard against drift: both shipped scripts must source + call the helper."""
    def test_launcher_sources_and_passes_snapshot(self):
        with open(LAUNCHER) as f:
            s = f.read()
        self.assertIn("dspark-numeric-knobs.sh", s)
        self.assertIn('dspark_validate_numeric_knobs "$_dspark_env_clean"', s)

    def test_validator_sources_and_calls(self):
        with open(VALIDATOR) as f:
            s = f.read()
        self.assertIn("dspark-numeric-knobs.sh", s)
        self.assertIn("dspark_validate_numeric_knobs", s)


if __name__ == "__main__":
    unittest.main(verbosity=0 if "-q" in sys.argv else 2,
                  argv=[a for a in sys.argv if a != "-q"])
