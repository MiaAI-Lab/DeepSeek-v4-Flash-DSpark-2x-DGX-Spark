#!/usr/bin/env python3
"""CPU guards for the adaptive issue-27 scheduler hotfix."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches" / "hotfix-dsv4-issue27-partial-prefill-concurrency.py"


class AdaptivePrefillPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = PATCH.read_text()

    def test_detects_mtp_decode_from_output_not_transient_chunk_flag(self) -> None:
        self.assertIn('"            req.num_output_tokens > 0\\n"', self.source)
        self.assertIn('"            for req in self.requests.values()\\n"', self.source)
        self.assertIn('"        if decode_is_live:\\n"', self.source)

    def test_prompt_complete_fallback_covers_first_decode_step(self) -> None:
        self.assertIn(
            '"                req.num_computed_tokens >= req.num_prompt_tokens\\n"',
            self.source,
        )
        self.assertIn('"                and not req.is_prefill_chunk\\n"', self.source)

    def test_active_threshold_controls_running_and_waiting_prefills(self) -> None:
        self.assertIn('"                num_new_tokens = active_prefill_threshold\\n"', self.source)
        self.assertIn('"                    threshold = active_prefill_threshold\\n"', self.source)

    def test_patch_remains_idempotent_and_migrates_old_revision(self) -> None:
        self.assertIn("if ADAPTIVE_MARK not in src:", self.source)
        self.assertIn("old_registry_detector", self.source)
        self.assertIn('statuses.append("adaptive-output-state")', self.source)


if __name__ == "__main__":
    unittest.main()
