#!/usr/bin/env python3
"""Regressions for issue #80's decode-active-only mixed prefill cap.

The production scheduler must retain the 1024-token global long-prefill
threshold for prefill-only throughput, but dynamically cap each prefill chunk
to 256 whenever any eligible decode-active request is present in ``running``.
The mixed cap must be applied before issue #43's decode-service floor, and the
same schedule-start decision must cap a new/resumed WAITING request's first
non-async chunk after RUNNING requests have been scheduled.
"""
from __future__ import annotations

import pathlib
import re
import unittest

from tests.sim.test_issue43_scheduler_sim import Req, production_order_step, step


ROOT = pathlib.Path(__file__).resolve().parent.parent
HOTFIX = ROOT / "patches/hotfix-dsv4-issue43-decode-fairness-and-diag.py"
COMPOSE = ROOT / "docker-compose.dspark.yml"
ENV_EXAMPLE = ROOT / ".env.dspark.example"

LONG_THRESHOLD = 1024
MIXED_PREFILL_CAP = 256
SPS = 6
TOKEN_BUDGET = 8192
CURRENT_STEP = 7


def _prefill(rid: str = "prefill") -> Req:
    request = Req(rid, prompt_tokens=4096, max_tokens=128)
    request.is_prefill_chunk = True
    return request


def _decode(rid: str = "decode") -> Req:
    request = Req(
        rid,
        prompt_tokens=32,
        max_tokens=128,
        sampled_tokens_per_step=SPS,
    )
    request.num_computed_tokens = request.num_prompt_tokens
    return request


def _run(running: list[Req], *, token_budget: int = TOKEN_BUDGET):
    diag: dict = {}
    scheduled, leftover = step(
        running,
        token_budget=token_budget,
        current_step=CURRENT_STEP,
        long_threshold=LONG_THRESHOLD,
        num_sampled_tokens_per_step=SPS,
        diag=diag,
        floor=True,
        mixed_prefill_cap=MIXED_PREFILL_CAP,
    )
    return {request.request_id: tokens for request, tokens in scheduled}, leftover, diag


class MixedPrefillCapBehaviorTests(unittest.TestCase):
    def test_prefill_only_retains_global_1024_threshold(self):
        scheduled, leftover, diag = _run([_prefill()])

        self.assertEqual(scheduled, {"prefill": LONG_THRESHOLD})
        self.assertEqual(leftover, TOKEN_BUDGET - LONG_THRESHOLD)
        self.assertEqual(
            diag,
            {
                "prefill": {"prefill": LONG_THRESHOLD},
                "decode": {},
                "skips": [],
            },
        )

    def test_mixed_prefill_then_decode_caps_prefill_and_preserves_sps(self):
        scheduled, leftover, diag = _run([_prefill(), _decode()])

        self.assertEqual(
            scheduled,
            {"prefill": MIXED_PREFILL_CAP, "decode": SPS},
        )
        self.assertEqual(leftover, TOKEN_BUDGET - MIXED_PREFILL_CAP - SPS)
        self.assertEqual(
            diag,
            {
                "prefill": {"prefill": MIXED_PREFILL_CAP},
                "decode": {"decode": SPS},
                "skips": [],
            },
        )

    def test_mixed_decode_then_prefill_caps_independent_of_running_order(self):
        scheduled, leftover, diag = _run([_decode(), _prefill()])

        self.assertEqual(
            scheduled,
            {"decode": SPS, "prefill": MIXED_PREFILL_CAP},
        )
        self.assertEqual(leftover, TOKEN_BUDGET - MIXED_PREFILL_CAP - SPS)
        self.assertEqual(
            diag,
            {
                "prefill": {"prefill": MIXED_PREFILL_CAP},
                "decode": {"decode": SPS},
                "skips": [],
            },
        )

    def _assert_noneligible_decode_does_not_cap(self, decode: Req):
        scheduled, leftover, diag = _run([_prefill(), decode])

        self.assertEqual(scheduled, {"prefill": LONG_THRESHOLD})
        self.assertEqual(leftover, TOKEN_BUDGET - LONG_THRESHOLD)
        self.assertEqual(
            diag,
            {
                "prefill": {"prefill": LONG_THRESHOLD},
                "decode": {},
                "skips": [],
            },
        )

    def test_ineligible_decode_does_not_trigger_mixed_cap(self):
        decode = _decode()
        decode.next_decode_eligible_step = CURRENT_STEP + 1

        self._assert_noneligible_decode_does_not_cap(decode)

    def test_finished_decode_does_not_trigger_mixed_cap(self):
        decode = _decode()
        decode.num_computed_tokens = decode.num_prompt_tokens + decode.max_tokens
        decode.num_output_placeholders = 1

        self._assert_noneligible_decode_does_not_cap(decode)

    def test_cap_precedes_issue43_floor_without_skip_and_diag_reconciles(self):
        exact_budget = MIXED_PREFILL_CAP + SPS
        scheduled, leftover, diag = _run(
            [_prefill(), _decode()],
            token_budget=exact_budget,
        )

        self.assertEqual(
            scheduled,
            {"prefill": MIXED_PREFILL_CAP, "decode": SPS},
        )
        self.assertEqual(diag["skips"], [])
        diagnostic_total = sum(diag["prefill"].values()) + sum(
            diag["decode"].values()
        )
        self.assertEqual(diagnostic_total, sum(scheduled.values()))
        self.assertEqual(leftover, exact_budget - sum(scheduled.values()))
        self.assertEqual(leftover, 0)


class WaitingFirstChunkCapBehaviorTests(unittest.TestCase):
    def _production_step(self, running: list[Req], waiting: list[Req]):
        return production_order_step(
            running,
            waiting,
            token_budget=TOKEN_BUDGET,
            current_step=CURRENT_STEP,
            long_threshold=LONG_THRESHOLD,
            num_sampled_tokens_per_step=SPS,
            mixed_prefill_cap=MIXED_PREFILL_CAP,
        )

    def test_running_decode_is_scheduled_before_cold_waiting_prefill_cap(self):
        decode = _decode()
        scheduled, leftover, diag = self._production_step([decode], [_prefill()])

        self.assertEqual(
            [(request.request_id, tokens) for request, tokens in scheduled],
            [("decode", SPS), ("prefill", MIXED_PREFILL_CAP)],
        )
        self.assertEqual(leftover, TOKEN_BUDGET - SPS - MIXED_PREFILL_CAP)
        self.assertEqual(diag["decode"], {"decode": SPS})
        self.assertEqual(diag["prefill"], {})
        self.assertEqual(diag["skips"], [])
        self.assertEqual(
            sum(tokens for _, tokens in scheduled) + leftover,
            TOKEN_BUDGET,
        )
        self.assertEqual(
            sum(diag["prefill"].values()) + sum(diag["decode"].values()),
            scheduled[0][1],
        )

    def test_waiting_prefill_only_retains_global_threshold(self):
        scheduled, leftover, diag = self._production_step([], [_prefill()])

        self.assertEqual(
            [(request.request_id, tokens) for request, tokens in scheduled],
            [("prefill", LONG_THRESHOLD)],
        )
        self.assertEqual(leftover, TOKEN_BUDGET - LONG_THRESHOLD)
        self.assertEqual(diag, {"prefill": {}, "decode": {}, "skips": []})

    def test_ineligible_and_finished_running_decodes_do_not_cap_waiting(self):
        ineligible = _decode("ineligible")
        ineligible.next_decode_eligible_step = CURRENT_STEP + 1
        finished = _decode("finished")
        finished.num_computed_tokens = finished.num_prompt_tokens + finished.max_tokens
        finished.num_output_placeholders = 1

        for decode in (ineligible, finished):
            with self.subTest(decode=decode.request_id):
                scheduled, leftover, diag = self._production_step(
                    [decode], [_prefill()]
                )
                self.assertEqual(
                    [(request.request_id, tokens) for request, tokens in scheduled],
                    [("prefill", LONG_THRESHOLD)],
                )
                self.assertEqual(leftover, TOKEN_BUDGET - LONG_THRESHOLD)
                self.assertEqual(
                    diag, {"prefill": {}, "decode": {}, "skips": []}
                )

    def test_resumed_waiting_request_is_capped_but_async_load_zero_is_not_scheduled(self):
        resumed = _prefill("resumed")
        resumed.num_computed_tokens = 512
        async_load = _prefill("async-load")
        async_load.load_kv_async = True

        scheduled, leftover, _ = self._production_step(
            [_decode()], [resumed, async_load]
        )

        self.assertEqual(
            [(request.request_id, tokens) for request, tokens in scheduled],
            [("decode", SPS), ("resumed", MIXED_PREFILL_CAP)],
        )
        self.assertEqual(leftover, TOKEN_BUDGET - SPS - MIXED_PREFILL_CAP)


class MixedPrefillCapStaticContractTests(unittest.TestCase):
    def test_issue43_hotfix_declares_dynamic_issue80_cap_default_256(self):
        source = HOTFIX.read_text()

        self.assertTrue(
            "DSPARK_MIXED_PREFILL_TOKEN_CAP" in source,
            "issue43 hotfix must read DSPARK_MIXED_PREFILL_TOKEN_CAP",
        )
        self.assertTrue(
            "# [issue80-mixed-prefill-cap]" in source,
            "issue43 hotfix must carry the dynamic mixed-cap patch marker",
        )
        normalized_source = source.replace("\\", "")
        self.assertRegex(
            normalized_source,
            re.compile(
                r"os\.environ\.get\(\s*[\"']DSPARK_MIXED_PREFILL_TOKEN_CAP[\"']"
                r"\s*,\s*[\"']256[\"']"
            ),
        )

    def test_compose_exposes_mixed_prefill_cap_with_default_256(self):
        source = COMPOSE.read_text()

        expected = (
            'DSPARK_MIXED_PREFILL_TOKEN_CAP: '
            '"${DSPARK_MIXED_PREFILL_TOKEN_CAP:-256}"'
        )
        self.assertTrue(
            expected in source,
            "compose must expose DSPARK_MIXED_PREFILL_TOKEN_CAP with default 256",
        )

    def test_env_example_documents_cap_without_lowering_global_threshold(self):
        source = ENV_EXAMPLE.read_text()

        self.assertTrue(
            "LONG_PREFILL_TOKEN_THRESHOLD=1024" in source,
            "global prefill-only threshold must remain 1024",
        )
        self.assertTrue(
            "DSPARK_MIXED_PREFILL_TOKEN_CAP=256" in source,
            "env example must document the decode-active mixed cap as 256",
        )


if __name__ == "__main__":
    unittest.main()
