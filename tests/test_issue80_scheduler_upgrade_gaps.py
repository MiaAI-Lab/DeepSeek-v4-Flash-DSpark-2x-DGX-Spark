#!/usr/bin/env python3
"""Regression tests for the issue #80 scheduler review gaps."""
from __future__ import annotations

import contextlib
import io
import pathlib
import py_compile
import sys
import tempfile
import textwrap
import unittest

from tests.sim import test_issue43_scheduler_sim as sim


ROOT = pathlib.Path(__file__).resolve().parent.parent
HOTFIX = ROOT / "patches/hotfix-dsv4-issue43-decode-fairness-and-diag.py"
COMPOSE = ROOT / "docker-compose.dspark.yml"
TARGET_LITERAL = (
    'Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py")'
)
V2_MARK = "# [issue80-scheduler-current-v2]"
CURRENT_MARK = "# [issue80-scheduler-current-v3]"

# A compact but realistic scheduler fragment after the pre-issue80 issue #43
# patch. The legacy issue43 blocks and stock priority-preemption rollback are
# preserved verbatim so the upgrade path is tested against asserted anchors.
LEGACY_PATCHED_SCHEDULER = textwrap.dedent(
    '''\
    import itertools
    import time
    # [issue43-hotfix] os import for DSPARK_ISSUE43_SCHED_DIAG gate
    import os

    def init_logger(name):
        return None

    logger = init_logger(__name__)
    # [issue43-hotfix] per-step scheduler diagnostics gate (issue #43).
    # Set DSPARK_ISSUE43_SCHED_DIAG=1 in the container env to emit one
    # compact scheduled-tokens / decode-skip summary line per step.
    _ISSUE43_SCHED_DIAG = os.environ.get("DSPARK_ISSUE43_SCHED_DIAG", "0") not in ("0", "", "false", "False")

    class Scheduler:
        def _mamba_block_aligned_split(self, request, num_new_tokens):
            return num_new_tokens

        def schedule(self):
            scheduled_running_reqs = []
            req_to_new_blocks = {}
            num_scheduled_tokens = {}
            scheduled_spec_decode_tokens = {}
            scheduled_encoder_inputs = {}
            token_budget = 8192
            encoder_compute_budget = 0
            scheduled_timestamp = time.monotonic()
            preempted_reqs = []
            req_index = 0
            defer_prefills = False
            prefill_scheduled = False
            # [issue43-hotfix] per-step scheduler diagnostics (issue #43).
            # Tracks per-request scheduled prefill/decode token counts and
            # zero-token decode skips (by request_id and running-list pos).
            # Always built (cheap); only the step log line (below) is gated.
            issue43_step_diag = {"prefill": {}, "decode": {}, "skips": []}
            request = self.running[req_index]
            num_new_tokens = token_budget
            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )

            # [issue43-hotfix] bounded decode service during mixed prefill
            # steps (issue #43 ask #3). Generalizes the #27
            # max_num_partial_prefills cap: regardless of the configured
            # --long-prefill-token-threshold, never let a prefill chunk
            # consume so much remaining token budget that a decode-active
            # request later in self.running is forced to num_new_tokens==0
            # and skipped. Reserve >=1 decode step of tokens for every
            # not-yet-visited decode-active lane; if the reservation can't
            # be met alongside the prefill chunk, drop the chunk to 0 so the
            # zero-check below skips it (continue) and the decodes run.
            if getattr(request, "is_prefill_chunk", False):
                _dec_floor = 0
                for _ri in range(req_index + 1, len(self.running)):
                    _r = self.running[_ri]
                    if (_r.num_output_placeholders > 0 and
                            _r.num_computed_tokens + 2
                            - _r.num_output_placeholders
                            >= _r.num_prompt_tokens + _r.max_tokens):
                        continue
                    if self.current_step < _r.next_decode_eligible_step:
                        continue
                    if defer_prefills and getattr(_r, "is_prefill_chunk", False):
                        continue
                    if _r.num_computed_tokens >= _r.num_prompt_tokens:
                        _dec_floor += self.num_sampled_tokens_per_step
                if _dec_floor > 0:
                    num_new_tokens = min(
                        num_new_tokens,
                        max(0, token_budget - _dec_floor))

            if num_new_tokens == 0:
                # The request cannot be scheduled.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                # [issue43-hotfix] record zero-token decode skips (issue #43
                # ask #2): a request past its prompt with no pending async
                # max-tokens sentinel is decode-active and got skipped here.
                if (request.num_computed_tokens >= request.num_prompt_tokens
                        and request.num_output_placeholders == 0):
                    issue43_step_diag["skips"].append(
                        (request.request_id, req_index,
                         request.num_computed_tokens))
                req_index += 1

            if False:
                preempted_req = request
                self.running.remove(preempted_req)
                if preempted_req in scheduled_running_reqs:
                    preempted_req_id = preempted_req.request_id
                    scheduled_running_reqs.remove(preempted_req)
                    token_budget += num_scheduled_tokens.pop(preempted_req_id)
                    req_to_new_blocks.pop(preempted_req_id)
                    scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                    preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                        preempted_req_id, None
                    )
                    if preempted_encoder_inputs:
                        encoder_compute_budget += 1
                    req_index -= 1

            scheduled_running_reqs.append(request)
            prefill_scheduled |= request.is_prefill_chunk
            request_id = request.request_id
            req_to_new_blocks[request_id] = object()
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            # [issue43-hotfix] per-request scheduled-tokens record (issue
            # #43 ask #1). Decode-active => "decode", else prefill chunk.
            _is_dec = (request.num_computed_tokens >= request.num_prompt_tokens
                       and not request.is_prefill_chunk)
            issue43_step_diag[
                "decode" if _is_dec else "prefill"][request_id] = num_new_tokens
            req_index += 1

            # [issue43-hotfix] step summary (issue #43 asks #1/#2). Stash the
            # per-step diag for the live reproducer; emit a compact log line
            # only when DSPARK_ISSUE43_SCHED_DIAG=1 to keep default overhead 0.
            self.issue43_last_step_diag = issue43_step_diag
            if _ISSUE43_SCHED_DIAG and (issue43_step_diag["prefill"]
                                        or issue43_step_diag["decode"]
                                        or issue43_step_diag["skips"]):
                pass

            # Record the LoRAs in scheduled_running_reqs
            # Next, schedule the WAITING requests.
            if not preempted_reqs:
                load_kv_async = False
                num_computed_tokens = request.num_computed_tokens
                num_new_local_computed_tokens = 0
                num_external_computed_tokens = 0
                if load_kv_async:
                    num_new_tokens = 0
                else:
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if num_new_tokens > token_budget:
                        pass
                    num_new_tokens = min(num_new_tokens, token_budget)

                if self.need_mamba_block_aligned_split:
                    num_new_tokens = self._mamba_block_aligned_split(
                        request, num_new_tokens
                    )
            return issue43_step_diag
    '''
)

# The real anchors live inside the RUNNING-list loop. Keep the fixture readable
# above, then nest that fragment to the same indentation as the image target.
_LEGACY_LOOP_START = "        request = self.running[req_index]\n"
_LEGACY_LOOP_END = "\n        if False:\n"
_loop_start = LEGACY_PATCHED_SCHEDULER.index(_LEGACY_LOOP_START)
_loop_end = LEGACY_PATCHED_SCHEDULER.index(_LEGACY_LOOP_END, _loop_start)
_loop_body = LEGACY_PATCHED_SCHEDULER[_loop_start:_loop_end]
LEGACY_PATCHED_SCHEDULER = (
    LEGACY_PATCHED_SCHEDULER[:_loop_start]
    + "        if True:\n"
    + textwrap.indent(_loop_body, "    ")
    + LEGACY_PATCHED_SCHEDULER[_loop_end:]
)

# Likewise retain production's WAITING queue loop depth so the v3 anchor is the
# exact 20-space false-branch threshold block used by the preserved scheduler.
_WAIT_BODY_START = "            load_kv_async = False\n"
_WAIT_BODY_END = "        return issue43_step_diag\n"
_wait_start = LEGACY_PATCHED_SCHEDULER.index(_WAIT_BODY_START)
_wait_end = LEGACY_PATCHED_SCHEDULER.index(_WAIT_BODY_END, _wait_start)
_wait_body = LEGACY_PATCHED_SCHEDULER[_wait_start:_wait_end]
LEGACY_PATCHED_SCHEDULER = (
    LEGACY_PATCHED_SCHEDULER[:_wait_start]
    + "            while True:\n"
    + textwrap.indent(_wait_body, "    ")
    + LEGACY_PATCHED_SCHEDULER[_wait_end:]
)


def run_hotfix(target: pathlib.Path, *args: str) -> str:
    source = HOTFIX.read_text().replace(TARGET_LITERAL, f"Path({str(target)!r})")
    old_argv = sys.argv
    output = io.StringIO()
    try:
        sys.argv = [str(HOTFIX), *args]
        with contextlib.redirect_stdout(output):
            exec(compile(source, str(HOTFIX), "exec"), {})
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    finally:
        sys.argv = old_argv
    return output.getvalue()


class LegacyUpgradeTests(unittest.TestCase):
    def test_legacy_patch_upgrades_compiles_reports_current_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = pathlib.Path(tmp) / "scheduler.py"
            target.write_text(LEGACY_PATCHED_SCHEDULER)

            self.assertIn("LEGACY", run_hotfix(target, "--status"))
            run_hotfix(target)
            upgraded = target.read_text()

            self.assertIn(CURRENT_MARK, upgraded)
            self.assertIn(V2_MARK, upgraded)
            self.assertIn("DSPARK_MIXED_PREFILL_TOKEN_CAP", upgraded)
            self.assertIn("_issue80_has_eligible_decode", upgraded)
            self.assertEqual(
                upgraded.count("per-request scheduled-tokens record (issue"), 1,
                "legacy issue43 diagnostics must not be re-applied",
            )
            self.assertIn(
                'issue43_step_diag["prefill"].pop(preempted_req_id, None)',
                upgraded,
            )
            self.assertIn(
                'issue43_step_diag["decode"].pop(preempted_req_id, None)',
                upgraded,
            )
            cap_pos = upgraded.index("num_new_tokens, _ISSUE80_MIXED_PREFILL_TOKEN_CAP")
            mamba_pos = upgraded.index("if self.need_mamba_block_aligned_split", cap_pos)
            floor_pos = upgraded.index("_dec_floor = 0", mamba_pos)
            self.assertLess(cap_pos, mamba_pos)
            self.assertLess(mamba_pos, floor_pos)
            waiting_cap_pos = upgraded.index(CURRENT_MARK)
            waiting_async_zero_pos = upgraded.rfind(
                "                    num_new_tokens = 0", 0, waiting_cap_pos
            )
            waiting_threshold_pos = upgraded.rfind(
                "threshold = self.scheduler_config.long_prefill_token_threshold",
                0,
                waiting_cap_pos,
            )
            waiting_budget_pos = upgraded.index(
                "# chunked prefill has to be enabled explicitly", waiting_cap_pos
            )
            waiting_mamba_pos = upgraded.index(
                "if self.need_mamba_block_aligned_split", waiting_budget_pos
            )
            self.assertGreaterEqual(waiting_async_zero_pos, 0)
            self.assertGreaterEqual(waiting_threshold_pos, 0)
            self.assertLess(waiting_async_zero_pos, waiting_threshold_pos)
            self.assertLess(waiting_threshold_pos, waiting_cap_pos)
            self.assertLess(waiting_cap_pos, waiting_budget_pos)
            self.assertLess(waiting_budget_pos, waiting_mamba_pos)
            py_compile.compile(str(target), doraise=True)
            self.assertIn("CURRENT", run_hotfix(target, "--status"))

            before = target.read_bytes()
            output = run_hotfix(target)
            self.assertIn("already applied", output)
            self.assertEqual(target.read_bytes(), before)

    def test_status_distinguishes_stock_legacy_v2_and_current_v3(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = pathlib.Path(tmp) / "scheduler.py"
            target.write_text("# stock\n")
            self.assertIn("NOT APPLIED", run_hotfix(target, "--status"))

            target.write_text(LEGACY_PATCHED_SCHEDULER)
            self.assertRegex(run_hotfix(target, "--status"), r": LEGACY\s*$")
            run_hotfix(target)
            current = target.read_text()
            self.assertRegex(run_hotfix(target, "--status"), r": CURRENT\s*$")

            waiting_start = current.index(
                "                    # [issue80-scheduler-current-v3]"
            )
            waiting_end = current.index(
                "                    # chunked prefill has to be enabled", waiting_start
            )
            target.write_text(current[:waiting_start] + current[waiting_end:])
            self.assertIn(V2_MARK, target.read_text())
            self.assertNotIn(CURRENT_MARK, target.read_text())
            self.assertRegex(run_hotfix(target, "--status"), r": LEGACY_V2\s*$")

    def test_realistic_v2_to_v3_upgrade_adds_only_waiting_block(self):
        """Build current locally, regress it to v2, then exercise the upgrader."""
        with tempfile.TemporaryDirectory() as tmp:
            target = pathlib.Path(tmp) / "scheduler.py"
            target.write_text(LEGACY_PATCHED_SCHEDULER)
            run_hotfix(target)
            expected_current = target.read_text()

            waiting_start = expected_current.index(
                "                    # [issue80-scheduler-current-v3]"
            )
            waiting_end = expected_current.index(
                "                    # chunked prefill has to be enabled", waiting_start
            )
            v2 = expected_current[:waiting_start] + expected_current[waiting_end:]
            self.assertIn(V2_MARK, v2)
            self.assertNotIn(CURRENT_MARK, v2)
            target.write_text(v2)

            self.assertRegex(run_hotfix(target, "--status"), r": LEGACY_V2\s*$")
            run_hotfix(target)
            self.assertEqual(target.read_text(), expected_current)
            py_compile.compile(str(target), doraise=True)
            self.assertRegex(run_hotfix(target, "--status"), r": CURRENT\s*$")

            before = target.read_bytes()
            self.assertIn("already applied", run_hotfix(target))
            self.assertEqual(target.read_bytes(), before)

    def test_v2_upgrade_fails_closed_when_waiting_anchor_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = pathlib.Path(tmp) / "scheduler.py"
            target.write_text(LEGACY_PATCHED_SCHEDULER)
            run_hotfix(target)
            current = target.read_text()
            waiting_start = current.index(
                "                    # [issue80-scheduler-current-v3]"
            )
            waiting_end = current.index(
                "                    # chunked prefill has to be enabled", waiting_start
            )
            v2 = current[:waiting_start] + current[waiting_end:]
            v2 = v2.replace(
                "                    threshold = self.scheduler_config.long_prefill_token_threshold\n",
                "                    waiting_threshold = self.scheduler_config.long_prefill_token_threshold\n",
                1,
            )
            target.write_text(v2)
            before = target.read_bytes()

            with self.assertRaisesRegex(AssertionError, "WAITING"):
                run_hotfix(target)
            self.assertEqual(target.read_bytes(), before)

    def test_legacy_upgrade_fails_closed_when_an_asserted_anchor_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = pathlib.Path(tmp) / "scheduler.py"
            broken = LEGACY_PATCHED_SCHEDULER.replace(
                "scheduled_spec_decode_tokens.pop(preempted_req_id, None)\n",
                "",
            )
            target.write_text(broken)
            before = target.read_bytes()

            with self.assertRaisesRegex(AssertionError, "preemption"):
                run_hotfix(target)
            self.assertEqual(target.read_bytes(), before)


class MambaAndPreemptionBehaviorTests(unittest.TestCase):
    @staticmethod
    def _prefill(rid: str = "prefill") -> sim.Req:
        request = sim.Req(rid, prompt_tokens=4096, max_tokens=128)
        request.is_prefill_chunk = True
        return request

    @staticmethod
    def _decode(rid: str = "decode") -> sim.Req:
        request = sim.Req(rid, prompt_tokens=32, max_tokens=128,
                          sampled_tokens_per_step=1)
        request.num_computed_tokens = request.num_prompt_tokens
        return request

    def test_mixed_cap_is_aligned_before_floor_for_generic_mamba_scheduler(self):
        running = [self._prefill(), self._decode()]
        scheduled, _ = sim.step(
            running,
            token_budget=4096,
            current_step=1,
            long_threshold=1024,
            mixed_prefill_cap=255,
            mamba_block_size=128,
        )
        by_id = {request.request_id: tokens for request, tokens in scheduled}
        self.assertEqual(by_id, {"prefill": 128, "decode": 1})
        self.assertEqual(by_id["prefill"] % 128, 0)

    def test_preempted_scheduled_request_is_removed_from_both_diag_maps(self):
        prefill = self._prefill("victim")
        decode = self._decode("survivor")
        running = [prefill, decode]
        diag: dict = {}
        scheduled, leftover = sim.step(
            running,
            token_budget=1024,
            current_step=1,
            long_threshold=256,
            mixed_prefill_cap=255,
            diag=diag,
        )
        num_scheduled_tokens = {
            request.request_id: tokens for request, tokens in scheduled
        }

        restored = sim.rollback_scheduled_preemption(
            running, scheduled, num_scheduled_tokens, diag, "victim"
        )
        leftover += restored

        self.assertNotIn("victim", [request.request_id for request in running])
        self.assertNotIn("victim", [request.request_id for request, _ in scheduled])
        self.assertEqual(set(diag["prefill"]) | set(diag["decode"]),
                         set(num_scheduled_tokens))
        self.assertEqual(
            sum(diag["prefill"].values()) + sum(diag["decode"].values()),
            sum(num_scheduled_tokens.values()),
        )
        self.assertEqual(leftover + sum(num_scheduled_tokens.values()), 1024)


class ComposeFailClosedTests(unittest.TestCase):
    def test_each_critical_python_scheduler_patch_has_its_own_failure_guard(self):
        compose = COMPOSE.read_text()
        for name in (
            "hotfix-dsv4-issue27-partial-prefill-concurrency.py",
            "hotfix-dsv4-issue43-decode-fairness-and-diag.py",
        ):
            with self.subTest(name=name):
                self.assertIn(f"python3 /opt/{name} || exit 1;", compose)


if __name__ == "__main__":
    unittest.main()
