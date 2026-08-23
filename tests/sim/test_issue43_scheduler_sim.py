#!/usr/bin/env python3
"""Pure-python simulator of the vLLM v1 ``Scheduler.schedule`` RUNNING-list
loop, with the issue #43 bounded-decode-service reservation and per-step
diagnostics. No torch / no GPU / no vLLM import needed.

Used by tests/test_issue43_deadlock.py and the live reproducer sanity to
prove that, across the six cold cells from issue #43 (32K/62K x2/x4/x8), under
the issue #27 cap (max_num_partial_prefills=1) + long_prefill_token_threshold
1024 + max_num_batched_tokens 8192, the reservation:
  - never lets a decode-active running lane get skipped (num_new_tokens==0)
    while a prefill chunk is also scheduled in the same step;
  - emits per-step diag whose scheduled-token sum equals max_num_batched_tokens
    whenever the step is budget-saturated;
and that the whole-request min/max asymmetry the reporter saw is structural
(TTFT stagger from serialized prefill), not per-step starvation.

This validates the scheduler *logic* the hotfix injects; the hotfix text-patch
itself is validated against the real container scheduler.py in
tests/test_issue43_patchapply.py.
"""
from __future__ import annotations
import sys
import statistics


class Req:
    """Faithful-enough stand-in for vllm.Request scheduling fields."""

    def __init__(self, rid: str, prompt_tokens: int, max_tokens: int,
                 sampled_tokens_per_step: int = 1):
        self.request_id = rid
        self.num_prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        # vllm: num_tokens_with_spec == prompt + output_so_far (+ spec, 0 here).
        self.num_output_token_ids = 0
        self.num_computed_tokens = 0
        self.num_output_placeholders = 0
        self.is_prefill_chunk = False
        self.next_decode_eligible_step = 0
        self.load_kv_async = False
        self._sps = sampled_tokens_per_step

    @property
    def num_tokens_with_spec(self):
        return self.num_prompt_tokens + self.num_output_token_ids

    def __repr__(self):
        return f"{self.request_id}(c={self.num_computed_tokens},p={self.num_prompt_tokens})"


def has_eligible_decode(self_running, current_step):
    """Mirror the production schedule-start decode eligibility scan."""
    for request in self_running:
        if (request.num_output_placeholders > 0 and
                request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens):
            continue
        if current_step < request.next_decode_eligible_step:
            continue
        if request.is_prefill_chunk:
            continue
        if request.num_computed_tokens >= request.num_prompt_tokens:
            return True
    return False


def step(self_running, token_budget, current_step, defer_prefills=False,
         long_threshold=1024, max_num_partial_prefills=1,
         num_sampled_tokens_per_step=1, in_flight_prefills=None,
         diag=None, floor=True, mixed_prefill_cap=None,
         mamba_block_size=None, eligible_decode_at_start=None):
    """One Scheduler.schedule RUNNING-list pass.

    Returns (scheduled:list[(req, ntok)], leftover_budget). Mutates each
    scheduled req's num_computed_tokens / is_prefill_chunk; updates
    in_flight_prefills (issue #27 admission gate is modeled by the caller
    admitting at most max_num_partial_prefills NEW prefills per step).
    `diag` (dict) is filled like self.issue43_last_step_diag.
    `mixed_prefill_cap` models issue #80's decode-active-only prefill cap. None
    or 0 disables it, matching the production rollback knob.
    `mamba_block_size` enables the generic hybrid-scheduler alignment step;
    the exact DeepSeek target leaves it as None because it has no Mamba layer.
    """
    if diag is not None:
        diag["prefill"] = {}
        diag["decode"] = {}
        diag["skips"] = []

    req_index = 0
    scheduled = []
    budget = token_budget
    # [issue80-mixed-prefill-cap] Determine mixed-mode once, before walking the
    # running list, so behavior does not depend on request order. Eligibility
    # mirrors issue #43's decode floor, while explicitly excluding prefills.
    eligible_decode = (
        has_eligible_decode(self_running, current_step)
        if eligible_decode_at_start is None
        else eligible_decode_at_start
    )
    while req_index < len(self_running) and budget > 0:
        request = self_running[req_index]

        # finished async sentinel / decode-cadence gates -- model the common
        # case (both pass) so the loop actually exercises scheduling.
        if (request.num_output_placeholders > 0 and
                request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens):
            req_index += 1
            continue
        if current_step < request.next_decode_eligible_step:
            req_index += 1
            continue
        if defer_prefills and request.is_prefill_chunk:
            req_index += 1
            continue

        # [issue43-sim] decode-active lanes request exactly one sampled
        # token step (sps); prefill chunks request remaining prompt tokens
        # capped by long_threshold. This mirrors vLLM: after prefill completes
        # the model samples one new output token per step, so a decoder's
        # num_new_tokens is sps (not the with_spec formula, which is 0 until an
        # output token actually exists).
        if (not request.is_prefill_chunk
                and request.num_computed_tokens >= request.num_prompt_tokens):
            num_new_tokens = num_sampled_tokens_per_step
        else:
            num_new_tokens = (request.num_tokens_with_spec
                              + request.num_output_placeholders
                              - request.num_computed_tokens)
            if 0 < long_threshold < num_new_tokens:
                num_new_tokens = long_threshold
        num_new_tokens = min(num_new_tokens, budget)

        # [issue80-mixed-prefill-cap] Keep the global threshold unchanged for
        # prefill-only steps, but bound mixed prefill chunks before the issue
        # #43 floor so decode token counts and floor accounting are preserved.
        if (mixed_prefill_cap and eligible_decode and
                request.is_prefill_chunk):
            num_new_tokens = min(num_new_tokens, mixed_prefill_cap)

        # Match the production ordering: the dynamic cap is an input to the
        # generic Mamba alignment helper, never a post-alignment override.
        if (mamba_block_size and request.is_prefill_chunk and
                num_new_tokens >= mamba_block_size):
            last_cache_position = (
                request.num_tokens_with_spec
                - request.num_tokens_with_spec % mamba_block_size
            )
            if request.num_computed_tokens + num_new_tokens < last_cache_position:
                num_new_tokens = (
                    num_new_tokens // mamba_block_size * mamba_block_size
                )

        # --- [issue43-hotfix] bounded decode service reservation ---
        # Gated by `floor`; disabled by the ablation (--no-floor) to prove
        # this reservation, not #27's serialized cap, is what keeps skips=0
        # once prefills are parallelized (cap>1).
        if floor and request.is_prefill_chunk:
            dec_floor = 0
            for ri in range(req_index + 1, len(self_running)):
                r = self_running[ri]
                if (r.num_output_placeholders > 0 and
                        r.num_computed_tokens + 2 - r.num_output_placeholders
                        >= r.num_prompt_tokens + r.max_tokens):
                    continue
                if current_step < r.next_decode_eligible_step:
                    continue
                if defer_prefills and r.is_prefill_chunk:
                    continue
                if r.num_computed_tokens >= r.num_prompt_tokens:
                    dec_floor += num_sampled_tokens_per_step
            if dec_floor > 0:
                num_new_tokens = min(num_new_tokens, max(0, budget - dec_floor))

        if num_new_tokens == 0:
            # decode-active skip record (issue #43 ask #2)
            if (request.num_computed_tokens >= request.num_prompt_tokens and
                    request.num_output_placeholders == 0 and diag is not None):
                diag["skips"].append((request.request_id, req_index,
                                      request.num_computed_tokens))
            req_index += 1
            continue

        # schedule it
        scheduled.append((request, num_new_tokens))
        budget -= num_new_tokens
        is_dec = (request.num_computed_tokens >= request.num_prompt_tokens
                  and not request.is_prefill_chunk)
        if diag is not None:
            diag["decode" if is_dec else "prefill"][request.request_id] = num_new_tokens

        # advance the request's computed tokens. A prefill chunk that finishes
        # its prompt this step becomes a decoder from the next step onward; a
        # decode-active lane consumes one sampled token step.
        was_prefill = request.is_prefill_chunk
        request.num_computed_tokens += num_new_tokens
        if was_prefill and request.num_computed_tokens >= request.num_prompt_tokens:
            request.is_prefill_chunk = False
            # prefill completion frees the issue #27 admission slot.
            if in_flight_prefills is not None and request in in_flight_prefills:
                in_flight_prefills.discard(request)
        elif not was_prefill:
            request.num_output_token_ids = min(
                request.max_tokens,
                request.num_output_token_ids + num_sampled_tokens_per_step)
        req_index += 1

    return scheduled, budget


def production_order_step(self_running, waiting, token_budget, current_step,
                          defer_prefills=False, long_threshold=1024,
                          max_num_partial_prefills=1,
                          num_sampled_tokens_per_step=1,
                          in_flight_prefills=None, floor=True,
                          mixed_prefill_cap=None, mamba_block_size=None):
    """Model one production-order RUNNING pass followed by WAITING admission.

    The mixed-mode decision is captured once at schedule start and reused by
    both phases. The returned diagnostic maps intentionally model issue43's
    RUNNING-pass diagnostics; ``scheduled`` models production's combined
    ``num_scheduled_tokens`` output for budget reconciliation.
    """
    eligible_decode = has_eligible_decode(self_running, current_step)
    diag = {}
    scheduled, budget = step(
        self_running,
        token_budget,
        current_step,
        defer_prefills=defer_prefills,
        long_threshold=long_threshold,
        max_num_partial_prefills=max_num_partial_prefills,
        num_sampled_tokens_per_step=num_sampled_tokens_per_step,
        in_flight_prefills=in_flight_prefills,
        diag=diag,
        floor=floor,
        mixed_prefill_cap=mixed_prefill_cap,
        mamba_block_size=mamba_block_size,
        eligible_decode_at_start=eligible_decode,
    )

    for request in waiting:
        if budget <= 0:
            break
        # The production async-load branch deliberately schedules zero new
        # work and bypasses the false-branch first-chunk cap.
        if request.load_kv_async:
            continue

        num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens
        if 0 < long_threshold < num_new_tokens:
            num_new_tokens = long_threshold
        if mixed_prefill_cap and eligible_decode:
            num_new_tokens = min(num_new_tokens, mixed_prefill_cap)
        num_new_tokens = min(num_new_tokens, budget)

        if mamba_block_size and num_new_tokens >= mamba_block_size:
            last_cache_position = (
                request.num_tokens_with_spec
                - request.num_tokens_with_spec % mamba_block_size
            )
            if request.num_computed_tokens + num_new_tokens < last_cache_position:
                num_new_tokens = (
                    num_new_tokens // mamba_block_size * mamba_block_size
                )
        if num_new_tokens == 0:
            break

        scheduled.append((request, num_new_tokens))
        budget -= num_new_tokens

    return scheduled, budget, diag


def rollback_scheduled_preemption(self_running, scheduled,
                                  num_scheduled_tokens, diag, request_id):
    """Model stock PRIORITY rollback plus issue43 diagnostic rollback."""
    victim = next(r for r in self_running if r.request_id == request_id)
    self_running.remove(victim)
    scheduled[:] = [item for item in scheduled if item[0] is not victim]
    restored = num_scheduled_tokens.pop(request_id)
    if diag is not None:
        diag["prefill"].pop(request_id, None)
        diag["decode"].pop(request_id, None)
    return restored


def simulate_cell(n_lanes, prompt_tokens, out_tokens=256,
                  budget=8192, long_threshold=1024,
                  max_num_partial_prefills=1, sps=1, floor=True):
    """Run a cold wave to completion, return per-lane
    (ttft_step, decode_steps, last_step, per_step_decode_skips)."""

    running = []
    inflight = set()           # issue #27 admission state
    t = 0
    # admission plan: all lanes arrive at step 0 (cold, simultaneous).
    pending = [Req(f"l{i}", prompt_tokens, out_tokens, sps) for i in range(n_lanes)]
    # per-lane telemetry
    ttft = {r.request_id: None for r in pending}
    done = {r.request_id: None for r in pending}
    decode_only_steps = {r.request_id: 0 for r in pending}
    total_decode_skips = 0
    trace = []  # diag per step

    while pending or running:
        t += 1
        diag = {}

        # admit at most max_num_partial_prefills NEW prefilling requests per
        # step. The cap is on currently-in-flight prefills (already admitted,
        # still chunking).
        while pending and len(inflight) < max_num_partial_prefills:
            # running has capacity (we model an unbounded running list but the
            # gate in vLLM is max_num_seqs; here N<=8 so fine).
            r = pending.pop(0)
            r.is_prefill_chunk = True
            inflight.add(r)
            running.append(r)

        sched, _budget = step(running, budget, t, False, long_threshold,
                              max_num_partial_prefills, sps, inflight, diag,
                              floor)
        trace.append(diag)
        total_decode_skips += len(diag["skips"])

        # post-step bookkeeping: TTFT, decode-step counts, completion.
        for r, n in sched:
            if (ttft[r.request_id] is None and not r.is_prefill_chunk
                    and r.num_output_token_ids > 0):
                ttft[r.request_id] = t
            if not r.is_prefill_chunk and r.num_output_token_ids > 0:
                decode_only_steps[r.request_id] += 1
            if r.num_output_token_ids >= r.max_tokens:
                done[r.request_id] = t
                if r in inflight:
                    inflight.discard(r)
        # remove finished
        running[:] = [r for r in running if done.get(r.request_id) is None]

    summary = []
    for r_id in ttft:
        # decode rate (whole-request, issue #43's metric) = out / (last-ttft)
        # but we measure BOTH whole-request and post-TTFT here.
        last = done[r_id]
        first = ttft[r_id]
        whole_window = last - 0           # includes TTFT stagger
        post_ttft_window = last - first    # issue #43 ask #4: decode window
        summary.append({
            "lane": r_id,
            "ttft": first,
            "last": last,
            "whole_tok_s": out_tokens / whole_window if whole_window > 0 else 0,
            "post_ttft_tok_s": out_tokens / post_ttft_window if post_ttft_window > 0 else 0,
        })
    return summary, total_decode_skips, t, trace


def minmax(xs):
    xs = [x for x in xs if x > 0]
    if not xs:
        return 0.0, 0.0, 0.0
    return min(xs), statistics.median(xs), max(xs)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=1,
                    help="max_num_partial_prefills (issue #27 knob). Default 1.")
    ap.add_argument("--no-floor", action="store_true",
                    help="disable the issue #43 decode floor (ablation: prove "
                         "the floor, not the #27 cap, keeps skips=0 at cap>1).")
    args = ap.parse_args()
    floor = not args.no_floor
    failures = []
    print(f"max_num_partial_prefills = {args.cap}  issue43 floor = {floor}")
    print(f"{'shape':>10} | {'whole min/med/max tok/s':>28} | "
          f"{'postTTFT min/med/max':>26} | {'whmin/max':>9} {'skips':>6} {'mspan':>6}")
    for in_tokens, n in [(32768, 2), (32768, 4), (32768, 8),
                         (63488, 2), (63488, 4), (63488, 8)]:
        res, skips, mspan, trace = simulate_cell(n, in_tokens, 256,
                                                 max_num_partial_prefills=args.cap,
                                                 floor=floor)
        whole = [r["whole_tok_s"] for r in res]
        post = [r["post_ttft_tok_s"] for r in res]
        wmn, wmed, wmx = minmax(whole)
        pmn, pmed, pmx = minmax(post)
        whole_ratio = wmn / wmx if wmx > 0 else 0
        print(f"{in_tokens//1024}K x{n:<2} | {wmn:6.2f}/{wmed:6.2f}/{wmx:6.2f}      | "
              f"{pmn:6.2f}/{pmed:6.2f}/{pmx:6.2f}     | {whole_ratio:9.3f} "
              f"{skips:6d} {mspan:6d}")
        # assert the actual scheduler guarantees:
        if skips != 0:
            failures.append((in_tokens, n, f"decode skips={skips} (bounded service violated)"))
        # diag invariant: every step that scheduled something has sum<=budget
        for d in trace:
            st = sum(d["prefill"].values()) + sum(d["decode"].values())
            if st > 8192:
                failures.append((in_tokens, n, f"diag sum {st} > budget 8192"))
    print()
    if failures:
        print("FAIL:")
        for f in failures:
            print("  ", f)
        sys.exit(1)
    print(f"PASS (cap={args.cap}, floor={floor}): zero decode skips (bounded service) and "
          "diag sums within budget for all six cold cells.")


if __name__ == "__main__":
    main()