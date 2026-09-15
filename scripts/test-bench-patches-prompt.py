#!/usr/bin/env python3
"""Host regression for the shipped TTFT prompt assignment, including its fallback.

Only the assignment runs: no benchmark requests, containers, or result files.
"""
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "scripts" / "bench-patches.sh"
SOURCE = BENCH.read_text()

_m = re.search(r"^run_ttft\(\) \{(.*?)^\}", SOURCE, re.M | re.S)
assert _m, "run_ttft function not found in bench-patches.sh"
_assignment = re.search(r"^[ \t]*prompt=.*$", _m.group(1), re.M)
assert _assignment, "prompt assignment not found in run_ttft"
ASSIGNMENT = _assignment.group(0)
SIZES = sorted({int(n) for n in re.findall(r"^[ \t]*run_ttft\s+(\d+)\s", SOURCE, re.M)})
assert SIZES, "TTFT size declarations not found in bench-patches.sh"


class PromptGeneration(unittest.TestCase):
    def test_declared_sizes_generate_increasing_prompts_without_fallback(self):
        previous_length = 0
        for n in SIZES:
            with self.subTest(n=n):
                # Execute the shell assignment unchanged so a fallback is observed,
                # rather than reconstructing or evaluating the arithmetic in Python.
                out = subprocess.run(
                    ["bash", "-c", 'set -euo pipefail\nprompt_tokens=$1\n'
                     + ASSIGNMENT + '\nprintf \'%s\' "$prompt"', "prompt-check", str(n)],
                    capture_output=True, text=True,
                )
                self.assertEqual(out.returncode, 0, out.stderr)
                self.assertRegex(out.stdout, r"\A(?:hello )+\Z")
                self.assertGreater(len(out.stdout), previous_length)
                previous_length = len(out.stdout)


if __name__ == "__main__":
    unittest.main()
