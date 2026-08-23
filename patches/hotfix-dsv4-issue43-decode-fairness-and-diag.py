#!/usr/bin/env python3
"""Hotfix (issue #43): bounded decode service during mixed prefill steps +
per-step scheduler diagnostics. Layers on top of the issue #27 hotfix
(``hotfix-dsv4-issue27-partial-prefill-concurrency.py``), which restored the
v1 scheduler's missing ``max_num_partial_prefills`` admission gate.

Why this exists (issue #43 follow-up)
-------------------------------------
The #27 fix caps concurrent in-flight prefills to ``max_num_partial_prefills``
(default 1) and ``--long-prefill-token-threshold`` caps each prefill chunk. It
cured the *per-step* decode starvation (worst ITL 2790 ms -> 67 ms) but the
reporter's six-cell cold retest still shows a wide *whole-request* decode-rate
spread (min/max 0.107-0.238). Issue #43 asks for:

  1. scheduled prefill/decode tokens per step and per request;
  2. zero-token decode skips by request and running-list position;
  3. bounded service for every decode-active request during mixed steps;
  4. per-lane p95 ITL and whole decode-window fairness (separate harness).

This hotfix delivers (1)-(3) inside the scheduler. (4) lives in
``scripts/reproduce-issue43-live.py``.)

What it changes (all in ``Scheduler.schedule``)
-----------------------------------------------
A. Bounded decode service (ask #3) -- the actual *fix*. While iterating the
   RUNNING list, when the current request is a prefill chunk, reserve one
   decode step's worth of token budget for every still-unvis ited,
   decode-active running request behind it, and cap the prefill chunk so that
   reservation is never violated. If the remaining budget can't satisfy both
   the prefill chunk and the decode reservation, the prefill chunk is dropped
   to 0 tokens and skipped (``continue``), letting the decode lanes run.
   This generalizes the #27 cap beyond ``max_num_partial_prefills=1``: it is
   a per-step, per-lane service floor that holds for any
   ``--long-prefill-token-threshold`` tuning, so mis-tuning the chunk cap can
   no longer resurrect #27-style decode skips. It is a no-op under the
   current default knob set (threshold=1024, budget=8192, cap=1), where the
   prefill chunk is already far below the decode reservation.

B. Zero-token decode-skip recording (ask #2). At the ``num_new_tokens == 0``
   skip branch, if the skipped request is decode-active (past its prompt and
   not at its async max-tokens sentinel), record
   ``(request_id, running_list_position, num_computed_tokens)``.

C. Per-request scheduled-token recording (ask #1). For every scheduled
   running request, record its scheduled token count keyed by request_id in
   ``self.issue43_last_step_diag["prefill"|"decode"]``. Decode-active requests
   land in ``"decode"``; prefill chunks land in ``"prefill"``.

D. Step summary log (ask #1). When ``DSPARK_ISSUE43_SCHED_DIAG=1`` is set in
   the container env, emit one compact line per scheduler step:
   ``[issue43-step N] running=K prefill_toks={..} decode_toks={..}
   decode_skips=[(rid,pos,computed)..]``. Off by default (zero overhead: the
   diag dict is cheap to build and only the log line is gated).

E. Issue #80 mixed-step chunk cap. Parse ``DSPARK_MIXED_PREFILL_TOKEN_CAP``
   once at module import (default 256; 0 disables; valid range 0..8192).
   Whenever any eligible decode lane exists anywhere in ``self.running``, cap
   each prefill chunk before generic Mamba block alignment and then apply the
   issue #43 floor. Prefill-only steps retain the global
   ``--long-prefill-token-threshold``.

   The v3 revision also applies that same schedule-start decision to the first
   chunk of a newly admitted or resumed WAITING request. The WAITING cap lives
   in the ``load_kv_async == False`` branch, after the global threshold and
   before budget checks and Mamba alignment; async remote-KV loads still
   schedule zero new tokens and are untouched.

F. Priority-preemption diagnostic rollback. When stock PRIORITY scheduling
   removes a request already scheduled in the current step and restores its
   token budget, remove that request from the prefill/decode diagnostic maps too
   so their totals continue to match ``num_scheduled_tokens``.

Versioned and idempotent. Current v2 sources receive only the WAITING block and
v3 marker. A source carrying only the old issue #43 marker receives all issue80
features, including the WAITING block. Fresh stock receives the complete current
patch. Every upgrade anchor is asserted and all edits remain in memory until the
single final write, so target drift fails closed. Safe to apply before or after
the issue #27 hotfix (independent regions).

Patches /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py
in-place inside the container (called from the compose entrypoint before
``exec vllm serve``).
"""
from pathlib import Path
import sys

P = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py")
MARK = "# [issue43-hotfix]"
V2_MARK = "# [issue80-scheduler-current-v2]"
CURRENT_MARK = "# [issue80-scheduler-current-v3]"


def patch_status(status_src):
    has_issue43 = MARK in status_src
    has_v2 = V2_MARK in status_src
    has_current = CURRENT_MARK in status_src
    if has_issue43 and has_v2 and has_current:
        return "CURRENT"
    if has_issue43 and has_v2 and not has_current:
        return "LEGACY_V2"
    if has_issue43 and not has_v2 and not has_current:
        return "LEGACY"
    if not has_issue43 and not has_v2 and not has_current:
        return "NOT APPLIED"
    return "INVALID/DRIFT"


if len(sys.argv) > 1 and sys.argv[1] == "--status":
    status_src = P.read_text() if P.is_file() else ""
    print("issue43 decode-fairness + diag     :", patch_status(status_src))
    raise SystemExit(0)
src = P.read_text()
status = patch_status(src)
if status == "CURRENT":
    print(f"[issue43-hotfix] already applied (current) to {P}")
    raise SystemExit(0)
assert status != "INVALID/DRIFT", (
    "issue43: inconsistent version markers; refusing to patch"
)
legacy_v2 = status == "LEGACY_V2"
legacy = status == "LEGACY"
fresh = status == "NOT APPLIED"

# --- 0. import os (module-level diag gate; fresh stock only) -----------------
A0_OLD = "import itertools\nimport time\n"
if fresh:
    assert A0_OLD in src, "issue43: import anchor not found; refusing to patch"
    src = src.replace(
        A0_OLD,
        A0_OLD
        + "# [issue43-hotfix] os import for DSPARK_ISSUE43_SCHED_DIAG gate\n"
        + "import os\n",
        1,
    )

# --- 1. module-level diag and mixed-cap constants ----------------------------
A1_OLD = "logger = init_logger(__name__)\n"
A1_LEGACY_OLD = (
    "_ISSUE43_SCHED_DIAG = os.environ.get(\"DSPARK_ISSUE43_SCHED_DIAG\", \"0\") "
    "not in (\"0\", \"\", \"false\", \"False\")\n"
)
A1_NEW = (
    A1_OLD
    + "# [issue43-hotfix] per-step scheduler diagnostics gate (issue #43).\n"
    + "# Set DSPARK_ISSUE43_SCHED_DIAG=1 in the container env to emit one\n"
    + "# compact scheduled-tokens / decode-skip summary line per step.\n"
    + "_ISSUE43_SCHED_DIAG = os.environ.get(\"DSPARK_ISSUE43_SCHED_DIAG\", \"0\") not in (\"0\", \"\", \"false\", \"False\")\n"
    + "# [issue80-scheduler-current-v2] Complete issue80 scheduler revision.\n"
    + "# [issue80-mixed-prefill-cap] Bound prefill chunks only while an eligible\n"
    + "# decode is active. Parse once at module import; 0 is the rollback knob.\n"
    + "_ISSUE80_MIXED_PREFILL_TOKEN_CAP_RAW = os.environ.get(\"DSPARK_MIXED_PREFILL_TOKEN_CAP\", \"256\")\n"
    + "try:\n"
    + "    _ISSUE80_MIXED_PREFILL_TOKEN_CAP = int(_ISSUE80_MIXED_PREFILL_TOKEN_CAP_RAW)\n"
    + "except ValueError as exc:\n"
    + "    raise ValueError(\"DSPARK_MIXED_PREFILL_TOKEN_CAP must be an integer in 0..8192\") from exc\n"
    + "if not 0 <= _ISSUE80_MIXED_PREFILL_TOKEN_CAP <= 8192:\n"
    + "    raise ValueError(\"DSPARK_MIXED_PREFILL_TOKEN_CAP must be in 0..8192\")\n"
)
if legacy:
    assert A1_LEGACY_OLD in src, (
        "issue43 legacy upgrade: diag constant anchor not found; refusing to patch"
    )
    issue80_constants = A1_NEW.split(A1_LEGACY_OLD, 1)[1]
    src = src.replace(A1_LEGACY_OLD, A1_LEGACY_OLD + issue80_constants, 1)
elif fresh:
    assert A1_OLD in src, "issue43: logger anchor not found; refusing to patch"
    src = src.replace(A1_OLD, A1_NEW, 1)

# --- 2. init diag dict and global decode eligibility scan --------------------
A2_OLD = "        prefill_scheduled = False\n"
A2_LEGACY_OLD = (
    "        issue43_step_diag = {\"prefill\": {}, \"decode\": {}, \"skips\": []}\n"
)
A2_NEW = (
    A2_OLD
    + "        # [issue43-hotfix] per-step scheduler diagnostics (issue #43).\n"
    + "        # Tracks per-request scheduled prefill/decode token counts and\n"
    + "        # zero-token decode skips (by request_id and running-list pos).\n"
    + "        # Always built (cheap); only the step log line (below) is gated.\n"
    + "        issue43_step_diag = {\"prefill\": {}, \"decode\": {}, \"skips\": []}\n"
    + "        # [issue80-mixed-prefill-cap] Determine whether any decode lane is\n"
    + "        # eligible once at schedule start. Scanning all of self.running\n"
    + "        # makes the mixed cap independent of running-list order. Mirror\n"
    + "        # issue #43 floor eligibility and explicitly exclude prefills.\n"
    + "        _issue80_has_eligible_decode = False\n"
    + "        if _ISSUE80_MIXED_PREFILL_TOKEN_CAP > 0:\n"
    + "            for _r in self.running:\n"
    + "                if (_r.num_output_placeholders > 0 and\n"
    + "                        _r.num_computed_tokens + 2\n"
    + "                        - _r.num_output_placeholders\n"
    + "                        >= _r.num_prompt_tokens + _r.max_tokens):\n"
    + "                    continue\n"
    + "                if self.current_step < _r.next_decode_eligible_step:\n"
    + "                    continue\n"
    + "                if getattr(_r, \"is_prefill_chunk\", False):\n"
    + "                    continue\n"
    + "                if _r.num_computed_tokens >= _r.num_prompt_tokens:\n"
    + "                    _issue80_has_eligible_decode = True\n"
    + "                    break\n"
)
if legacy:
    assert A2_LEGACY_OLD in src, (
        "issue43 legacy upgrade: step-diag anchor not found; refusing to patch"
    )
    issue80_eligibility = A2_NEW.split(A2_LEGACY_OLD, 1)[1]
    src = src.replace(A2_LEGACY_OLD, A2_LEGACY_OLD + issue80_eligibility, 1)
elif fresh:
    assert A2_OLD in src, (
        "issue43: prefill_scheduled anchor not found; refusing to patch"
    )
    src = src.replace(A2_OLD, A2_NEW, 1)

# --- 3. mixed cap -> Mamba alignment -> issue43 floor -> zero check ----------
A3_MAMBA = (
    "            if self.need_mamba_block_aligned_split:\n"
    "                num_new_tokens = self._mamba_block_aligned_split(\n"
    "                    request, num_new_tokens\n"
    "                )\n"
    "\n"
)
A3_CAP = (
    "            # [issue80-mixed-prefill-cap] Keep prefill-only scheduling at\n"
    "            # the global long-prefill threshold, but cap each mixed prefill\n"
    "            # before Mamba block alignment and issue #43's decode floor.\n"
    "            # 0 disables the cap.\n"
    "            if (_ISSUE80_MIXED_PREFILL_TOKEN_CAP > 0 and\n"
    "                    _issue80_has_eligible_decode and\n"
    "                    getattr(request, \"is_prefill_chunk\", False)):\n"
    "                num_new_tokens = min(\n"
    "                    num_new_tokens, _ISSUE80_MIXED_PREFILL_TOKEN_CAP)\n"
    "\n"
)
A3_FLOOR = (
    "            # [issue43-hotfix] bounded decode service during mixed prefill\n"
    "            # steps (issue #43 ask #3). Generalizes the #27\n"
    "            # max_num_partial_prefills cap: regardless of the configured\n"
    "            # --long-prefill-token-threshold, never let a prefill chunk\n"
    "            # consume so much remaining token budget that a decode-active\n"
    "            # request later in self.running is forced to num_new_tokens==0\n"
    "            # and skipped. Reserve >=1 decode step of tokens for every\n"
    "            # not-yet-visited decode-active lane; if the reservation can't\n"
    "            # be met alongside the prefill chunk, drop the chunk to 0 so the\n"
    "            # zero-check below skips it (continue) and the decodes run.\n"
    "            if getattr(request, \"is_prefill_chunk\", False):\n"
    "                _dec_floor = 0\n"
    "                for _ri in range(req_index + 1, len(self.running)):\n"
    "                    _r = self.running[_ri]\n"
    "                    if (_r.num_output_placeholders > 0 and\n"
    "                            _r.num_computed_tokens + 2\n"
    "                            - _r.num_output_placeholders\n"
    "                            >= _r.num_prompt_tokens + _r.max_tokens):\n"
    "                        continue\n"
    "                    if self.current_step < _r.next_decode_eligible_step:\n"
    "                        continue\n"
    "                    if defer_prefills and getattr(_r, \"is_prefill_chunk\", False):\n"
    "                        continue\n"
    "                    if _r.num_computed_tokens >= _r.num_prompt_tokens:\n"
    "                        _dec_floor += self.num_sampled_tokens_per_step\n"
    "                if _dec_floor > 0:\n"
    "                    num_new_tokens = min(\n"
    "                        num_new_tokens,\n"
    "                        max(0, token_budget - _dec_floor))\n"
    "\n"
)
A3_ZERO = "            if num_new_tokens == 0:\n"
A3_OLD = A3_MAMBA + A3_ZERO
A3_LEGACY_OLD = A3_MAMBA + A3_FLOOR + A3_ZERO
A3_NEW = A3_CAP + A3_MAMBA + A3_FLOOR + A3_ZERO
if legacy:
    assert A3_LEGACY_OLD in src, (
        "issue43 legacy upgrade: mamba/floor anchor not found; refusing to patch"
    )
    src = src.replace(A3_LEGACY_OLD, A3_NEW, 1)
elif fresh:
    assert A3_OLD in src, (
        "issue43: mamba-split/zero-check anchor not found; refusing to patch"
    )
    src = src.replace(A3_OLD, A3_NEW, 1)

# --- 4. record zero-token decode skips inside the skip branch ----------------
A4_OLD = (
    "                # NOTE(woosuk): Here, by doing `continue` instead of `break`,\n"
    "                # we do not strictly follow the FCFS scheduling policy and\n"
    "                # allow the lower-priority requests to be scheduled.\n"
    "                req_index += 1\n"
    "                continue\n"
)
A4_NEW = (
    "                # NOTE(woosuk): Here, by doing `continue` instead of `break`,\n"
    "                # we do not strictly follow the FCFS scheduling policy and\n"
    "                # allow the lower-priority requests to be scheduled.\n"
    "                # [issue43-hotfix] record zero-token decode skips (issue #43\n"
    "                # ask #2): a request past its prompt with no pending async\n"
    "                # max-tokens sentinel is decode-active and got skipped here.\n"
    "                if (request.num_computed_tokens >= request.num_prompt_tokens\n"
    "                        and request.num_output_placeholders == 0):\n"
    "                    issue43_step_diag[\"skips\"].append(\n"
    "                        (request.request_id, req_index,\n"
    "                         request.num_computed_tokens))\n"
    "                req_index += 1\n"
    "                continue\n"
)
if fresh:
    assert A4_OLD in src, "issue43: woosuk-skip anchor not found; refusing to patch"
    src = src.replace(A4_OLD, A4_NEW, 1)

# --- 5. record per-request scheduled tokens in the running branch -----------
A5_OLD = (
    "            scheduled_running_reqs.append(request)\n"
    "            prefill_scheduled |= request.is_prefill_chunk\n"
    "            request_id = request.request_id\n"
    "            req_to_new_blocks[request_id] = new_blocks\n"
    "            num_scheduled_tokens[request_id] = num_new_tokens\n"
    "            token_budget -= num_new_tokens\n"
    "            req_index += 1\n"
)
A5_NEW = (
    "            scheduled_running_reqs.append(request)\n"
    "            prefill_scheduled |= request.is_prefill_chunk\n"
    "            request_id = request.request_id\n"
    "            req_to_new_blocks[request_id] = new_blocks\n"
    "            num_scheduled_tokens[request_id] = num_new_tokens\n"
    "            token_budget -= num_new_tokens\n"
    "            # [issue43-hotfix] per-request scheduled-tokens record (issue\n"
    "            # #43 ask #1). Decode-active => \"decode\", else prefill chunk.\n"
    "            _is_dec = (request.num_computed_tokens >= request.num_prompt_tokens\n"
    "                       and not request.is_prefill_chunk)\n"
    "            issue43_step_diag[\n"
    "                \"decode\" if _is_dec else \"prefill\"][request_id] = num_new_tokens\n"
    "            req_index += 1\n"
)
if fresh:
    assert A5_OLD in src, (
        "issue43: running-schedule anchor not found; refusing to patch"
    )
    src = src.replace(A5_OLD, A5_NEW, 1)

# --- 6. step summary log + stash diag on self, at end of running loop -------
A6_OLD = "        # Record the LoRAs in scheduled_running_reqs\n"
A6_NEW = (
    "        # [issue43-hotfix] step summary (issue #43 asks #1/#2). Stash the\n"
    "        # per-step diag for the live reproducer; emit a compact log line\n"
    "        # only when DSPARK_ISSUE43_SCHED_DIAG=1 to keep default overhead 0.\n"
    "        self.issue43_last_step_diag = issue43_step_diag\n"
    "        if _ISSUE43_SCHED_DIAG and (issue43_step_diag[\"prefill\"]\n"
    "                                    or issue43_step_diag[\"decode\"]\n"
    "                                    or issue43_step_diag[\"skips\"]):\n"
    "            logger.info(\n"
    "                \"[issue43-step %d] run=%d prefill_toks=%s decode_toks=%s \"\n"
    "                \"decode_skips=%s\",\n"
    "                self.current_step, len(scheduled_running_reqs),\n"
    "                issue43_step_diag[\"prefill\"], issue43_step_diag[\"decode\"],\n"
    "                issue43_step_diag[\"skips\"])\n"
    "\n"
    "        # Record the LoRAs in scheduled_running_reqs\n"
)
if fresh:
    assert A6_OLD in src, (
        "issue43: end-of-running-loop anchor not found; refusing to patch"
    )
    src = src.replace(A6_OLD, A6_NEW, 1)

# --- 7. remove preempted scheduled requests from diagnostic totals -----------
# V2 already carries this block; old issue43 and fresh stock still need it.
if not legacy_v2:
    A7_OLD = "scheduled_spec_decode_tokens.pop(preempted_req_id, None)\n"
    assert src.count(A7_OLD) == 1, (
        "issue43: priority-preemption rollback anchor not found or ambiguous; "
        "refusing to patch"
    )
    a7_index = src.index(A7_OLD)
    a7_line_start = src.rfind("\n", 0, a7_index) + 1
    a7_indent = src[a7_line_start:a7_index]
    assert a7_indent and not a7_indent.strip(), (
        "issue43: priority-preemption indentation anchor invalid; refusing to patch"
    )
    A7_NEW = (
        A7_OLD
        + a7_indent
        + "# [issue43-hotfix] Roll back diagnostics with scheduler bookkeeping.\n"
        + a7_indent
        + "issue43_step_diag[\"prefill\"].pop(preempted_req_id, None)\n"
        + a7_indent
        + "issue43_step_diag[\"decode\"].pop(preempted_req_id, None)\n"
    )
    src = src.replace(A7_OLD, A7_NEW, 1)

# --- 8. cap the first chunk admitted from WAITING -----------------------------
# This anchor is deliberately wholly inside load_kv_async's false branch. It
# applies to new and resumed requests, but cannot touch async-load num_new_tokens
# == 0. It also preserves threshold -> cap -> budget/encoder -> Mamba ordering.
A8_OLD = (
    "                    threshold = self.scheduler_config.long_prefill_token_threshold\n"
    "                    if 0 < threshold < num_new_tokens:\n"
    "                        num_new_tokens = threshold\n"
    "\n"
    "                    # chunked prefill has to be enabled explicitly to allow\n"
)
A8_CAP = (
    "                    threshold = self.scheduler_config.long_prefill_token_threshold\n"
    "                    if 0 < threshold < num_new_tokens:\n"
    "                        num_new_tokens = threshold\n"
    "\n"
    "                    # [issue80-scheduler-current-v3] Complete issue80\n"
    "                    # scheduler revision: cap a new/resumed WAITING first\n"
    "                    # chunk when schedule-start found an eligible decode.\n"
    "                    # This false branch excludes async remote-KV load0.\n"
    "                    if (_ISSUE80_MIXED_PREFILL_TOKEN_CAP > 0 and\n"
    "                            _issue80_has_eligible_decode):\n"
    "                        num_new_tokens = min(\n"
    "                            num_new_tokens,\n"
    "                            _ISSUE80_MIXED_PREFILL_TOKEN_CAP)\n"
    "\n"
    "                    # chunked prefill has to be enabled explicitly to allow\n"
)
assert src.count(A8_OLD) == 1, (
    "issue43: WAITING false-branch threshold/budget anchor not found or "
    "ambiguous; refusing to patch"
)
src = src.replace(A8_OLD, A8_CAP, 1)

P.write_text(src)
if legacy_v2:
    action = "upgraded v2 patch on"
elif legacy:
    action = "upgraded legacy patch on"
else:
    action = "patched"
print(f"[issue43-hotfix] {action} {P}")
