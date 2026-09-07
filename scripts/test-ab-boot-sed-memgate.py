#!/usr/bin/env python3
"""CPU regressions for ab-boot.sh env-edit escaping and the memory gate.

Two defects in scripts/ab-boot.sh:

1. K went unescaped into a grep -E pattern and a sed -E s||| expression, and V
   went unescaped into the sed REPLACEMENT: an unescaped `&` expands to the
   whole match, an unescaped `|` ends the s||| expression early, and a
   newline splits the operator's live .env.dspark mid-edit. K is now
   validated as an env-var name (alnum+underscore is literal in ERE),
   multiline V is rejected, and & | \\ are escaped for the replacement.

2. The host-memory gate was fail-open: the script runs `set -u` WITHOUT
   `-e`, so when the worker ssh failed MW came back empty, `[ "" -lt GATE ]`
   errored inside the `||`, the gate evaluated false, and the benchmark ran
   against a possibly-broken boot. The gate now fails closed on unreadable
   MemAvailable from either node.

The tests extract the shipped blocks and run them against a temp .env.dspark
/ synthetic MA-MW-GATE values.
"""
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts" / "ab-boot.sh").read_text()

_kv_start = SOURCE.index('case "$K" in')
KV_BLOCK = SOURCE[_kv_start:SOURCE.index('echo "== env:', _kv_start)
                                        + len('echo "== env: $(grep -E "^${K}=" .env.dspark)"')]

_mg_start = SOURCE.index("GATE=${AB_MEM_GATE_KB")
_mg_end = SOURCE.index("fi", SOURCE.index('MEM_BELOW_GATE', _mg_start)) + 2
MEMGATE_BLOCK = SOURCE[_mg_start:_mg_end]


def run_kv(env_content: str, key: str, val: str):
    workdir = Path(tempfile.mkdtemp())
    env_file = workdir / ".env.dspark"
    env_file.write_text(env_content)
    script = f"""set -uo pipefail
K={shlex.quote(key)}; V={shlex.quote(val)}
{KV_BLOCK}
"""
    r = subprocess.run(["bash", "-c", script], capture_output=True,
                       text=True, cwd=workdir)
    r.env_text = env_file.read_text()  # type: ignore[attr-defined]
    shutil.rmtree(workdir)
    return r


def run_gate(ma: str, mw: str, gate: str = "3000000"):
    script = f"""set -uo pipefail
MA={shlex.quote(ma)}; MW={shlex.quote(mw)}
AB_MEM_GATE_KB={shlex.quote(gate)}
{MEMGATE_BLOCK}
"""
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


BASE_ENV = "MAX_NUM_SEQS=6\nVLLM_PORT=8888\n"


class EnvEdit(unittest.TestCase):
    def test_normal_edit(self):
        r = run_kv(BASE_ENV, "MAX_NUM_SEQS", "8")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.env_text, "MAX_NUM_SEQS=8\nVLLM_PORT=8888\n")

    def test_metachar_key_rejected(self):
        r = run_kv(BASE_ENV, "MAX.NUM", "8")
        self.assertEqual(r.returncode, 2)
        self.assertEqual(r.env_text, BASE_ENV)

    def test_ampersand_value_is_literal(self):
        r = run_kv(BASE_ENV, "MAX_NUM_SEQS", "A&B")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("MAX_NUM_SEQS=A&B\n", r.env_text)

    def test_pipe_value_is_literal(self):
        r = run_kv(BASE_ENV, "MAX_NUM_SEQS", "a|b")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("MAX_NUM_SEQS=a|b\n", r.env_text)

    def test_backslash_value_is_literal(self):
        r = run_kv(BASE_ENV, "MAX_NUM_SEQS", r"a\b")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("MAX_NUM_SEQS=a\\b\n", r.env_text)

    def test_multiline_value_rejected(self):
        r = run_kv(BASE_ENV, "MAX_NUM_SEQS", "8\nMALICIOUS=1")
        self.assertEqual(r.returncode, 2)
        self.assertEqual(r.env_text, BASE_ENV)

    def test_missing_key_exits_3(self):
        r = run_kv(BASE_ENV, "NO_SUCH_KNOB", "1")
        self.assertEqual(r.returncode, 3)
        self.assertEqual(r.env_text, BASE_ENV)


class MemoryGate(unittest.TestCase):
    def test_both_above_gate_passes(self):
        r = run_gate("8000000", "8000000")
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_below_gate_exits_2(self):
        r = run_gate("8000000", "1000000")
        self.assertEqual(r.returncode, 2)
        self.assertIn("MEM_BELOW_GATE", r.stdout)

    def test_empty_worker_value_fails_closed(self):
        # The original bug: worker ssh failed -> MW empty -> gate passed.
        r = run_gate("8000000", "")
        self.assertEqual(r.returncode, 2)
        self.assertIn("worker MemAvailable unreadable", r.stderr)

    def test_garbage_worker_value_fails_closed(self):
        r = run_gate("8000000", "ssh: connect timeout")
        self.assertEqual(r.returncode, 2)
        self.assertIn("worker MemAvailable unreadable", r.stderr)

    def test_empty_head_value_fails_closed(self):
        r = run_gate("", "8000000")
        self.assertEqual(r.returncode, 2)
        self.assertIn("head MemAvailable unreadable", r.stderr)


class SourceShape(unittest.TestCase):
    def test_worker_host_first_match_only(self):
        line = next(ln for ln in SOURCE.splitlines() if ln.startswith("WORKER="))
        self.assertIn("head -n1", line)

    def test_sed_replacement_uses_escaped_value(self):
        self.assertNotIn("s|^${K}=.*|${K}=${V}|", SOURCE)
        self.assertIn("V_SED=$(printf '%s' \"$V\" | sed 's/[&|\\\\]/\\\\&/g')", SOURCE)


if __name__ == "__main__":
    unittest.main()
