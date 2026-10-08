#!/usr/bin/env python3
"""CPU regressions for DSPARK_MODEL resolution.

The checkpoint id is DSPARK_MODEL_OFFICIAL. start, validate, and prepare
used to overwrite an explicit DSPARK_MODEL with no message, so a serve
could come up on the official checkpoint while the operator believed
another was loaded. The three scripts carry one identical block: a
non-empty disagreeing value exits 2 and names both ids; unset, empty,
or equal keeps the resolved official id.
"""
from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEGIN = "# DSPARK_MODEL resolution (begin)\n"
END = "# DSPARK_MODEL resolution (end)\n"
FILES = (
    ROOT / "start-deepseek-v4-flash-dspark.sh",
    ROOT / "validate-dspark-config.sh",
    ROOT / "prepare-dspark-model-cache.sh",
)
OFFICIAL = "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"


def extract(path: Path) -> str:
    text = path.read_text()
    start = text.index(BEGIN)
    stop = text.index(END, start)
    return text[start : stop + len(END)]


BLOCK = extract(FILES[0])


def run_block(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    script = "set -u\n" + BLOCK + (
        "printf 'MODEL=%s\\nOFFICIAL=%s\\n' \"$DSPARK_MODEL\" \"$DSPARK_MODEL_OFFICIAL\"\n"
    )
    clean = os.environ.copy()
    clean.pop("DSPARK_MODEL", None)
    clean.pop("DSPARK_MODEL_OFFICIAL", None)
    clean.update(env)
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env=clean,
    )


class ModelIdResolution(unittest.TestCase):
    def test_the_three_scripts_carry_the_same_block(self):
        blocks = [extract(path) for path in FILES]
        self.assertEqual(blocks[0], blocks[1])
        self.assertEqual(blocks[0], blocks[2])
        self.assertIn('DSPARK_MODEL="$DSPARK_MODEL_OFFICIAL"', blocks[0])
        self.assertIn("exit 2", blocks[0])

    def test_unset_resolves_to_the_official_default(self):
        result = run_block({})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            f"MODEL={OFFICIAL}\nOFFICIAL={OFFICIAL}\n",
        )
        self.assertEqual(result.stderr, "")

    def test_empty_model_is_treated_as_unset(self):
        result = run_block({"DSPARK_MODEL": ""})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"MODEL={OFFICIAL}\n", result.stdout)

    def test_official_override_is_the_resolved_id(self):
        custom = "nvidia/DeepSeek-V4-Flash-0731-NVFP4"
        result = run_block({"DSPARK_MODEL_OFFICIAL": custom})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            f"MODEL={custom}\nOFFICIAL={custom}\n",
        )

    def test_explicit_model_equal_to_official_is_kept(self):
        result = run_block({"DSPARK_MODEL": OFFICIAL})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"MODEL={OFFICIAL}\n", result.stdout)
        self.assertEqual(result.stderr, "")

    def test_disagreeing_model_exits_2_and_names_both_ids(self):
        custom = "nvidia/DeepSeek-V4-Flash-0731-NVFP4"
        chosen = "org/other-checkpoint"
        result = run_block(
            {"DSPARK_MODEL": custom, "DSPARK_MODEL_OFFICIAL": chosen}
        )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn(f"DSPARK_MODEL='{custom}'", result.stderr)
        self.assertIn(f"DSPARK_MODEL_OFFICIAL='{chosen}'", result.stderr)
        self.assertIn("not a switch", result.stderr)
        self.assertNotIn("MODEL=", result.stdout)

    def test_disagreeing_model_against_the_default_official_id(self):
        custom = "nvidia/DeepSeek-V4-Flash-0731-NVFP4"
        result = run_block({"DSPARK_MODEL": custom})
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn(f"DSPARK_MODEL='{custom}'", result.stderr)
        self.assertIn(f"DSPARK_MODEL_OFFICIAL='{OFFICIAL}'", result.stderr)


if __name__ == "__main__":
    unittest.main()
