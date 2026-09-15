#!/usr/bin/env python3
"""Adaptive long-prefill chunk cap: 1024 only while decode lanes are active.

``--long-prefill-token-threshold 1024`` (issue #27/#43) keeps mixed-load
decode from starving, but a cold c=1 prefill still pays the per-chunk fixed
cost (collectives, indexer/compressor setup, launches) once per 1024 tokens.
When no running request is decode-active, raise the cap to
``max_num_batched_tokens`` so an 8K prompt is one chunk. Mixed load is
unchanged: any decode-active member of ``self.running`` restores the
configured 1024-token cap. Independent of the #27/#43 injection sites.

Idempotent. Patches ``vllm/v1/core/sched/scheduler.py``.
"""
from pathlib import Path
import sys

P = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py")
MARK = "# [dspark-adaptive-chunk]"
if len(sys.argv) > 1 and sys.argv[1] == "--status":
    status_src = P.read_text() if P.is_file() else ""
    print("dspark adaptive prefill chunk      :",
          "APPLIED" if MARK in status_src else "NOT APPLIED")
    raise SystemExit(0)
src = P.read_text()
if MARK in src:
    print(f"[dspark-adaptive-chunk] already applied to {P}")
    raise SystemExit(0)

RUNNING_OLD = (
    "            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:\n"
    "                num_new_tokens = self.scheduler_config.long_prefill_token_threshold\n"
)
WAITING_OLD = (
    "                    threshold = self.scheduler_config.long_prefill_token_threshold\n"
    "                    if 0 < threshold < num_new_tokens:\n"
    "                        num_new_tokens = threshold\n"
)
assert RUNNING_OLD in src, "dspark-adaptive-chunk: running-loop anchor not found; refusing to patch"
assert WAITING_OLD in src, "dspark-adaptive-chunk: waiting-loop anchor not found; refusing to patch"

RUNNING_NEW = (
    "            # [dspark-adaptive-chunk] keep the 1024 mixed-load floor only when\n"
    "            # a decode-active request is already running; otherwise one chunk\n"
    "            # can take the full max_num_batched_tokens (c=1 TTFT).\n"
    "            _chunk_cap = self.scheduler_config.long_prefill_token_threshold\n"
    "            if _chunk_cap > 0 and not any(\n"
    "                not r.is_prefill_chunk for r in self.running\n"
    "            ):\n"
    "                _chunk_cap = self.scheduler_config.max_num_batched_tokens\n"
    "            if 0 < _chunk_cap < num_new_tokens:\n"
    "                num_new_tokens = _chunk_cap\n"
)
WAITING_NEW = (
    "                    # [dspark-adaptive-chunk] same cap as the running loop.\n"
    "                    threshold = self.scheduler_config.long_prefill_token_threshold\n"
    "                    if threshold > 0 and not any(\n"
    "                        not r.is_prefill_chunk for r in self.running\n"
    "                    ):\n"
    "                        threshold = self.scheduler_config.max_num_batched_tokens\n"
    "                    if 0 < threshold < num_new_tokens:\n"
    "                        num_new_tokens = threshold\n"
)
src = src.replace(RUNNING_OLD, RUNNING_NEW, 1)
src = src.replace(WAITING_OLD, WAITING_NEW, 1)
compile(src, "scheduler.py", "exec")
P.write_text(src)
print(f"[dspark-adaptive-chunk] patched {P}")
