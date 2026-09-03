#!/usr/bin/env python3
"""CPU regression tests for complete encoder-only output semantics."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
HOTFIX = ROOT / "patches" / "hotfix-vllm-empty-encoder-output.py"
OVERLAY_OUTPUTS = ROOT / "recipe" / "overlay" / "vllm" / "v1" / "outputs.py"
OVERLAY_SCHEDULER = (
    ROOT / "recipe" / "overlay" / "vllm" / "v1" / "core" / "sched" / "scheduler.py"
)

OUTPUTS_FIXTURE = '''def make_empty_encoder_model_runner_output(req_ids):
    # No tokens generated yet ⇒ one empty list per request
    sampled_token_ids: list[list[int]] = [[0] for _ in req_ids]
    return sampled_token_ids
'''

SCHEDULER_FIXTURE = '''class RequestStatus:
    FINISHED_STOPPED = "finished"

class Scheduler:
    def __init__(self, vllm_config):
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder

    def _update_request_with_output(self, request, new_token_ids):
        raise AssertionError("encoder-only requests must not sample")

    def finish(self, request, new_token_ids, pooler_output=None):
        stopped = False
        for _ in [None]:
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids
                )
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

        return stopped
'''


def _load_hotfix():
    spec = importlib.util.spec_from_file_location("empty_encoder_output_hotfix", HOTFIX)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(source: str, req_ids: list[str]) -> list[list[int]]:
    namespace: dict[str, object] = {}
    exec(source, namespace)
    make_output = namespace["make_empty_encoder_model_runner_output"]
    return make_output(req_ids)


def _scheduler(source: str, *, encoder_only: bool):
    namespace: dict[str, object] = {}
    exec(source, namespace)
    model_config = SimpleNamespace(
        is_encoder_decoder=False,
        multimodal_config=SimpleNamespace(mm_encoder_only=encoder_only),
    )
    return namespace["Scheduler"](SimpleNamespace(model_config=model_config))


class EmptyEncoderOutputHotfixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hotfix = _load_hotfix()

    def test_patch_removes_phantom_tokens_and_keeps_distinct_rows(self):
        self.assertEqual(_rows(OUTPUTS_FIXTURE, ["a", "b"]), [[0], [0]])
        updated, status = self.hotfix.patch_outputs_text(OUTPUTS_FIXTURE)
        self.assertEqual(status, "applied")
        rows = _rows(updated, ["a", "b"])
        self.assertEqual(rows, [[], []])
        self.assertIsNot(rows[0], rows[1])

    def test_encoder_only_request_finishes_when_prompt_is_consumed(self):
        updated, status = self.hotfix.patch_scheduler_text(SCHEDULER_FIXTURE)
        self.assertEqual(status, "applied")
        scheduler = _scheduler(updated, encoder_only=True)
        request = SimpleNamespace(
            pooling_params=None,
            num_computed_tokens=16,
            num_prompt_tokens=16,
            status="running",
        )
        self.assertTrue(scheduler.finish(request, []))
        self.assertEqual(request.status, "finished")

    def test_encoder_only_request_does_not_finish_early(self):
        updated, _ = self.hotfix.patch_scheduler_text(SCHEDULER_FIXTURE)
        scheduler = _scheduler(updated, encoder_only=True)
        request = SimpleNamespace(
            pooling_params=None,
            num_computed_tokens=15,
            num_prompt_tokens=16,
            status="running",
        )
        self.assertFalse(scheduler.finish(request, []))
        self.assertEqual(request.status, "running")

    def test_non_encoder_model_does_not_take_completion_branch(self):
        updated, _ = self.hotfix.patch_scheduler_text(SCHEDULER_FIXTURE)
        scheduler = _scheduler(updated, encoder_only=False)
        request = SimpleNamespace(
            pooling_params=None,
            num_computed_tokens=16,
            num_prompt_tokens=16,
            status="running",
        )
        self.assertFalse(scheduler.finish(request, []))
        self.assertEqual(request.status, "running")

    def test_patch_is_idempotent(self):
        outputs, outputs_status = self.hotfix.patch_outputs_text(OUTPUTS_FIXTURE)
        scheduler, scheduler_status = self.hotfix.patch_scheduler_text(SCHEDULER_FIXTURE)
        self.assertEqual((outputs_status, scheduler_status), ("applied", "applied"))
        outputs_again, outputs_second = self.hotfix.patch_outputs_text(outputs)
        scheduler_again, scheduler_second = self.hotfix.patch_scheduler_text(scheduler)
        self.assertEqual((outputs_second, scheduler_second), ("skipped", "skipped"))
        self.assertEqual((outputs_again, scheduler_again), (outputs, scheduler))

    def test_source_drift_fails_before_mutating_either_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "outputs.py"
            scheduler = Path(tmp) / "scheduler.py"
            outputs.write_text(OUTPUTS_FIXTURE, encoding="utf-8")
            scheduler.write_text("# incompatible scheduler\n", encoding="utf-8")
            self.assertEqual(
                self.hotfix.main(["hotfix", str(outputs), str(scheduler)]),
                1,
            )
            self.assertEqual(outputs.read_text(encoding="utf-8"), OUTPUTS_FIXTURE)
            self.assertEqual(
                scheduler.read_text(encoding="utf-8"),
                "# incompatible scheduler\n",
            )

    def test_missing_target_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "outputs.py"
            scheduler = Path(tmp) / "missing.py"
            outputs.write_text(OUTPUTS_FIXTURE, encoding="utf-8")
            self.assertEqual(
                self.hotfix.main(["hotfix", str(outputs), str(scheduler)]),
                1,
            )

    def test_file_application_and_second_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "outputs.py"
            scheduler = Path(tmp) / "scheduler.py"
            outputs.write_text(OUTPUTS_FIXTURE, encoding="utf-8")
            scheduler.write_text(SCHEDULER_FIXTURE, encoding="utf-8")
            argv = ["hotfix", str(outputs), str(scheduler)]
            self.assertEqual(self.hotfix.main(argv), 0)
            first = (outputs.read_text(), scheduler.read_text())
            self.assertEqual(_rows(first[0], ["a"]), [[]])
            self.assertTrue("self.is_mm_encoder_only" in first[1])
            self.assertEqual(self.hotfix.main(argv), 0)
            self.assertEqual((outputs.read_text(), scheduler.read_text()), first)

    def test_repository_overlays_already_have_upstream_shape(self):
        outputs_source = OVERLAY_OUTPUTS.read_text(encoding="utf-8")
        scheduler_source = OVERLAY_SCHEDULER.read_text(encoding="utf-8")
        outputs_unchanged, outputs_status = self.hotfix.patch_outputs_text(outputs_source)
        scheduler_unchanged, scheduler_status = self.hotfix.patch_scheduler_text(
            scheduler_source
        )
        self.assertEqual((outputs_status, scheduler_status), ("skipped", "skipped"))
        self.assertEqual(outputs_unchanged, outputs_source)
        self.assertEqual(scheduler_unchanged, scheduler_source)


if __name__ == "__main__":
    unittest.main()
