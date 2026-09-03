#!/usr/bin/env python3
"""Hotfix: enforce SchedulerConfig.max_num_partial_prefills in the v1 scheduler.

Upstream vLLM 0.25.2.dev0 (ghcr.io/anemll/dspark-vllm-gx10:0.1.1) defines
``max_num_partial_prefills`` / ``max_long_partial_prefills`` on SchedulerConfig
but the v1 ``Scheduler.schedule`` admission loop never reads them — only
``max_num_seqs`` and ``token_budget`` gate new admissions. With chunked prefill
+ async scheduling + max_num_seqs>=8 and long_prefill_token_threshold=0
(default), multiple already-admitted-but-still-prefilling requests at the
front of ``self.running`` each consume up to ``max_num_batched_tokens`` per
step; decode-active requests later in the running list get
``num_new_tokens == 0`` and are skipped with ``continue`` (NOT preempted) —
producing severe, cold-only, zero-preemption decode lane starvation that
grows with prompt length. (Issue #27.)

Fix: at the top of the waiting-admission loop, break (don't admit a new
prefill request) once the number of in-flight partial prefills has reached
the cap. The cap is ``DSPARK_MAX_INFLIGHT_PREFILLS`` (1-3, default 1 via
compose) because this image rejects ``--max-num-partial-prefills``. It is
parsed once during ``Scheduler`` construction; unset, blank, nonpositive, or
malformed values fall back to ``SchedulerConfig.max_num_partial_prefills``
(stock 1), and malformed values emit one warning. ``self._inflight_prefills``
is maintained by ``_update_after_schedule`` (populated for requests still
needing more prefill chunks, discarded when they finish prefilling), so it
correctly reflects the currently-prefilling set. This restores the documented
concurrency cap of 1 by default, so at most one request prefill-chunks per
step and decode lanes behind it in ``self.running`` always receive budget
(chunk cap via ``--long-prefill-token-threshold`` keeps that one chunk below
``max_num_batched_tokens`` leaving room for decode tokens).

Idempotent: re-applying is a no-op once the marker is present.

Patches /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py
in-place inside the container (called from the compose entrypoint before
``exec vllm serve``).
"""
from pathlib import Path
import sys

P = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py")
MARK = "# [issue27-hotfix] enforce max_num_partial_prefills on admission"
if len(sys.argv) > 1 and sys.argv[1] == "--status":
    status_src = P.read_text() if P.is_file() else ""
    print("issue27 partial-prefill cap        :",
          "APPLIED" if MARK in status_src else "NOT APPLIED")
    raise SystemExit(0)
src = P.read_text()
if MARK in src:
    print(f"[issue27-hotfix] already applied to {P}")
    raise SystemExit(0)

INIT_ANCHOR = (
    "        # In-flight requests still prefilling (prefill chunks + in-progress\n"
    "        # async KV loads). Their remaining-block reservation gates async loads.\n"
    "        self._inflight_prefills: set[Request] = set()\n"
)
ADMISSION_ANCHOR = (
    "                num_running = len(self.running) + self.num_waiting_for_streaming_input\n"
    "                if num_running >= self.max_num_running_reqs:\n"
    "                    break\n"
)
assert INIT_ANCHOR in src, "scheduler init anchor not found; refusing to patch"
assert ADMISSION_ANCHOR in src, "admission guard anchor not found; refusing to patch"

INIT_INJECT = INIT_ANCHOR + (
    "\n"
    "        # [issue27-hotfix] parse the admission cap once, outside schedule().\n"
    "        _pp_cap_raw = __import__('os').environ.get(\n"
    "            'DSPARK_MAX_INFLIGHT_PREFILLS', ''\n"
    "        ).strip()\n"
    "        try:\n"
    "            _pp_cap = int(_pp_cap_raw, 10) if _pp_cap_raw else 0\n"
    "        except ValueError:\n"
    "            logger.warning(\n"
    "                'Invalid DSPARK_MAX_INFLIGHT_PREFILLS; using '\n"
    "                'SchedulerConfig.max_num_partial_prefills'\n"
    "            )\n"
    "            _pp_cap = 0\n"
    "        if _pp_cap <= 0:\n"
    "            _pp_cap = self.scheduler_config.max_num_partial_prefills\n"
    "        self._dspark_max_inflight_prefills = min(_pp_cap, 3)\n"
)

INJECT = ADMISSION_ANCHOR + (
    "\n"
    "                # [issue27-hotfix] enforce max_num_partial_prefills on admission.\n"
    "                # Upstream defines this field but the v1 scheduler never reads\n"
    "                # it, so without this gate N already-admitted-but-still-prefilling\n"
    "                # requests at the front of self.running consume the whole\n"
    "                # max_num_batched_tokens each step; decode-active requests behind\n"
    "                # them get num_new_tokens==0 and are skipped (continue, not preempt)\n"
    "                # -> zero-preemption decode starvation (issue #27). _inflight_prefills\n"
    "                # is the set of running requests still needing prefill chunks.\n"
    "                # DSPARK_MAX_INFLIGHT_PREFILLS is parsed and cached once\n"
    "                # during Scheduler construction, never in this hot loop.\n"
    "                if (\n"
    "                    self._dspark_max_inflight_prefills > 0\n"
    "                    and len(self._inflight_prefills)\n"
    "                    >= self._dspark_max_inflight_prefills\n"
    "                ):\n"
    "                    break\n"
)
src = src.replace(INIT_ANCHOR, INIT_INJECT, 1)
src = src.replace(ADMISSION_ANCHOR, INJECT, 1)
P.write_text(src)
print(f"[issue27-hotfix] patched {P}")