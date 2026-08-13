#!/usr/bin/env python3
"""Hotfix: enforce partial-prefill admission and protect live decode lanes.

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

Fix 1: at the top of the waiting-admission loop, break (don't admit a new
prefill request) once the number of in-flight partial prefills has reached
``max_num_partial_prefills``. ``self._inflight_prefills`` is maintained by
``_update_after_schedule`` (populated for requests still needing more prefill
chunks, discarded when they finish prefilling), so it correctly reflects the
currently-prefilling set. This restores the documented concurrency cap of 1
by default, so at most one request prefill-chunks per step and decode lanes
behind it in ``self.running`` always receive budget (chunk cap via
``--long-prefill-token-threshold`` keeps that one chunk below
``max_num_batched_tokens`` leaving room for decode tokens).

Fix 2: use a small prefill chunk while any decode request is live, but retain
the configured large chunk when the engine is prefill-only. A static large
chunk improves isolated cold prefill, but each mixed prefill/decode kernel then
creates multi-second inter-token gaps. ``DSPARK_DECODE_ACTIVE_PREFILL_THRESHOLD``
defaults to 1024 tokens and must be a multiple of the scheduler block size.

Idempotent: re-applying is a no-op once the marker is present.

Patches /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py
in-place inside the container (called from the compose entrypoint before
``exec vllm serve``).
"""
from pathlib import Path
import sys

P = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py"
)
src = P.read_text()
MARK = "# [issue27-hotfix] enforce max_num_partial_prefills on admission"
ADAPTIVE_MARK = "# [issue27-adaptive] shrink prefill chunks only while decode is live"

ANCHOR = (
    "                num_running = len(self.running) + self.num_waiting_for_streaming_input\n"
    "                if num_running >= self.max_num_running_reqs:\n"
    "                    break\n"
)
INJECT = ANCHOR + (
    "\n"
    "                # [issue27-hotfix] enforce max_num_partial_prefills on admission.\n"
    "                # Upstream defines this field but the v1 scheduler never reads\n"
    "                # it, so without this gate N already-admitted-but-still-prefilling\n"
    "                # requests at the front of self.running consume the whole\n"
    "                # max_num_batched_tokens each step; decode-active requests behind\n"
    "                # them get num_new_tokens==0 and are skipped (continue, not preempt)\n"
    "                # -> zero-preemption decode starvation (issue #27). _inflight_prefills\n"
    "                # is the set of running requests still needing prefill chunks.\n"
    "                if (\n"
    "                    self.scheduler_config.max_num_partial_prefills > 0\n"
    "                    and len(self._inflight_prefills)\n"
    "                    >= self.scheduler_config.max_num_partial_prefills\n"
    "                ):\n"
    "                    break\n"
)
statuses = []
if MARK not in src:
    assert ANCHOR in src, "admission guard anchor not found; refusing to patch"
    src = src.replace(ANCHOR, INJECT, 1)
    statuses.append("admission")

# Migrate the first adaptive revision. V2's async pipeline can have a live
# decode request outside ``self.running`` during the next scheduler call.
old_async_detector = (
    "        # Keep the larger configured threshold for an idle/cold-prefill lane.\n"
    "        active_prefill_threshold = (\n"
    "            self.scheduler_config.long_prefill_token_threshold\n"
    "        )\n"
    "        if any(not req.is_prefill_chunk for req in self.running):\n"
)
new_async_detector = (
    "        # Use the full live registry: async V2 scheduling can temporarily\n"
    "        # remove an in-flight decode from self.running. Keep the larger\n"
    "        # configured threshold only for a genuinely idle/prefill-only lane.\n"
    "        active_prefill_threshold = (\n"
    "            self.scheduler_config.long_prefill_token_threshold\n"
    "        )\n"
    "        if any(not req.is_prefill_chunk for req in self.requests.values()):\n"
)
if ADAPTIVE_MARK in src and old_async_detector in src:
    src = src.replace(old_async_detector, new_async_detector, 1)
    statuses.append("adaptive-async-registry")

# ``is_prefill_chunk`` becomes true again while speculative decode has
# uncomputed placeholders. Generated output is the durable signal that the
# request owns an interactive decode lane.
old_registry_detector = (
    "        if any(not req.is_prefill_chunk for req in self.requests.values()):\n"
)
output_detector = (
    "        decode_is_live = any(\n"
    "            req.num_output_tokens > 0\n"
    "            or (\n"
    "                req.num_computed_tokens >= req.num_prompt_tokens\n"
    "                and not req.is_prefill_chunk\n"
    "            )\n"
    "            for req in self.requests.values()\n"
    "        )\n"
    "        if decode_is_live:\n"
)
if ADAPTIVE_MARK in src and old_registry_detector in src:
    src = src.replace(old_registry_detector, output_detector, 1)
    statuses.append("adaptive-output-state")

if ADAPTIVE_MARK not in src:
    import_anchor = "import itertools\nimport time\n"
    assert import_anchor in src, "scheduler import anchor not found"
    src = src.replace(import_anchor, "import itertools\nimport os\nimport time\n", 1)

    init_anchor = (
        "        self.block_size = block_size\n"
        "        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size\n"
    )
    init_inject = (
        "        self.block_size = block_size\n"
        "        # [issue27-adaptive] mixed batches need a much smaller prefill\n"
        "        # chunk than prefill-only batches to bound inter-token gaps.\n"
        "        self._decode_active_prefill_threshold = int(\n"
        "            os.getenv(\"DSPARK_DECODE_ACTIVE_PREFILL_THRESHOLD\", \"1024\")\n"
        "        )\n"
        "        if (\n"
        "            self._decode_active_prefill_threshold < 0\n"
        "            or self._decode_active_prefill_threshold % block_size != 0\n"
        "        ):\n"
        "            raise ValueError(\n"
        "                \"DSPARK_DECODE_ACTIVE_PREFILL_THRESHOLD must be a \"\n"
        "                f\"non-negative multiple of {block_size}\"\n"
        "            )\n"
        "        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size\n"
    )
    assert init_anchor in src, "scheduler init anchor not found"
    src = src.replace(init_anchor, init_inject, 1)

    step_anchor = (
        "        # Whether the running batch contains any prefill requests.\n"
        "        prefill_scheduled = False\n"
    )
    step_inject = step_anchor + (
        "\n"
        "        # [issue27-adaptive] shrink prefill chunks only while decode is live.\n"
        "        # Use the full live registry: async V2 scheduling can temporarily\n"
        "        # remove an in-flight decode from self.running. Keep the larger\n"
        "        # configured threshold only for a genuinely idle/prefill-only lane.\n"
        "        active_prefill_threshold = (\n"
        "            self.scheduler_config.long_prefill_token_threshold\n"
        "        )\n"
        "        decode_is_live = any(\n"
        "            req.num_output_tokens > 0\n"
        "            or (\n"
        "                req.num_computed_tokens >= req.num_prompt_tokens\n"
        "                and not req.is_prefill_chunk\n"
        "            )\n"
        "            for req in self.requests.values()\n"
        "        )\n"
        "        if decode_is_live:\n"
        "            decode_threshold = self._decode_active_prefill_threshold\n"
        "            if decode_threshold > 0 and (\n"
        "                active_prefill_threshold <= 0\n"
        "                or decode_threshold < active_prefill_threshold\n"
        "            ):\n"
        "                active_prefill_threshold = decode_threshold\n"
    )
    assert step_anchor in src, "scheduler step anchor not found"
    src = src.replace(step_anchor, step_inject, 1)

    running_threshold = (
        "            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:\n"
        "                num_new_tokens = self.scheduler_config.long_prefill_token_threshold\n"
    )
    adaptive_running = (
        "            if 0 < active_prefill_threshold < num_new_tokens:\n"
        "                num_new_tokens = active_prefill_threshold\n"
    )
    assert running_threshold in src, "running prefill threshold anchor not found"
    src = src.replace(running_threshold, adaptive_running, 1)

    waiting_threshold = (
        "                    threshold = self.scheduler_config.long_prefill_token_threshold\n"
    )
    adaptive_waiting = "                    threshold = active_prefill_threshold\n"
    assert waiting_threshold in src, "waiting prefill threshold anchor not found"
    src = src.replace(waiting_threshold, adaptive_waiting, 1)
    statuses.append("adaptive")

if statuses:
    P.write_text(src)
    print(f"[issue27-hotfix] patched {','.join(statuses)}: {P}")
else:
    print(f"[issue27-hotfix] already applied to {P}")
