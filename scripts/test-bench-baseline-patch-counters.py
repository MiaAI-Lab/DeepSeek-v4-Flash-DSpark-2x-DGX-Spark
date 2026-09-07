#!/usr/bin/env python3
"""CPU regressions for the bench-baseline patch-state counters.

The CHECKS block in both bench-baseline scripts counted patch markers with

    c=$(grep -c 'pattern' 'file' 2>/dev/null || echo 0)

On GNU grep, `grep -c` with zero matches prints "0" AND exits 1, so the
`|| echo 0` appended a second 0: c="0\\n0". The later `echo $c1 $c2 $c3 $c4`
word-split into up to 8 fields and the `cut -d' ' -fN` reporting read shifted
values (a real #49486 hit could be reported as 0 while #50312 showed #49486's
count). The counters now use `|| true` plus a `${c:-0}` default, which is
exactly one token in all three cases (match / no match / missing file).

These tests extract the shipped CHECKS payload from both scripts and run it
against a fake vLLM tree, asserting the emitted counters stay 1:1 with the
labelled positions.
"""
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {
    "bench-baseline-issue22-only.sh": (
        ("models/deepseek_v4/sparse_mla.py", "nvfp4_ds_mla"),
        ("models/deepseek_v4/attention.py", "PORT #49486"),
        ("models/deepseek_v4/nvidia/model.py", "needs_mtp_hidden_states"),
        ("model_executor/layers/sparse_attn_indexer.py", "dense_mha_metadata_layer_name"),
    ),
    "bench-baseline-no-patches.sh": (
        ("models/deepseek_v4/attention.py", "PORT #49486"),
        ("models/deepseek_v4/nvidia/model.py", "needs_mtp_hidden_states"),
        ("models/deepseek_v4/sparse_mla.py", "active_topk_width"),
        ("model_executor/layers/sparse_attn_indexer.py", "dense_mha_metadata_layer_name"),
    ),
}


def payload_of(script: Path) -> str:
    text = script.read_text()
    m = re.search(r'bash -c "\n(.*?)\n" 2>/dev/null\)', text, re.S)
    assert m, f"CHECKS block not found in {script.name}"
    # The payload is written for an outer double-quoted context; unescape to
    # the form the container-side bash actually receives.
    return m.group(1).replace("\\$", "$")


def run_payload(payload: str, tree: dict[str, str]) -> list[str]:
    workdir = Path(tempfile.mkdtemp())
    vllm = workdir / "vllm"
    for rel, content in tree.items():
        f = vllm / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content)
    # In production the payload runs inside an outer double-quoted string, so
    # $VLLM expands before the container-side bash sees it (the single quotes
    # then quote the concrete path). Reproduce that expansion here.
    payload = payload.replace("$VLLM", shlex.quote(str(vllm)))
    result = subprocess.run(["bash", "-c", payload], capture_output=True, text=True)
    shutil.rmtree(workdir)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().split()


class CounterValues(unittest.TestCase):
    def test_all_missing_files_report_zero(self):
        for name, _ in SCRIPTS.items():
            payload = payload_of(ROOT / "scripts" / name)
            with self.subTest(script=name):
                self.assertEqual(run_payload(payload, {}), ["0", "0", "0", "0"])

    def test_match_and_zero_match_positions_do_not_shift(self):
        # First counter=2, second counter=0 (file exists, no match), rest
        # missing. The old double-print turned this into 5+ fields and moved
        # later positions.
        for name, counters in SCRIPTS.items():
            tree = {counters[0][0]: f"{counters[0][1]}\n{counters[0][1]}\n",
                    counters[1][0]: "unrelated content\n"}
            payload = payload_of(ROOT / "scripts" / name)
            with self.subTest(script=name):
                self.assertEqual(run_payload(payload, tree), ["2", "0", "0", "0"])

    def test_every_field_is_a_single_token(self):
        for name, counters in SCRIPTS.items():
            # All files exist with zero matches — the exact double-print case.
            tree = {rel: "unrelated content\n" for rel, _ in counters}
            payload = payload_of(ROOT / "scripts" / name)
            with self.subTest(script=name):
                fields = run_payload(payload, tree)
                self.assertEqual(fields, ["0", "0", "0", "0"])
                self.assertEqual(len(fields), 4)


class SourceShape(unittest.TestCase):
    def test_grep_c_no_longer_double_prints(self):
        for name in SCRIPTS:
            text = (ROOT / "scripts" / name).read_text()
            with self.subTest(script=name):
                self.assertIsNone(
                    re.search(r"grep -c[^\n]*\|\| echo 0\)", text),
                    "grep -c … || echo 0 returned",
                )
                self.assertEqual(text.count("|| true); c"), 4)


if __name__ == "__main__":
    unittest.main()
