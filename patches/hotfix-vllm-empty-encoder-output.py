#!/usr/bin/env python3
"""Backport complete encoder-only EC producer output semantics.

The pinned Anemll runtime image fabricates token 0 when an encoder producer
sampled nothing, and its scheduler lacks the paired encoder-only completion
branch. Upstream vLLM commit 7ca49fbe4bab019e55d57cdc4b7fd3d55c67c1a6
returns one empty row per request and finishes an encoder-only request once its
prompt is consumed. This startup patch applies both changes to the pinned image.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

DEFAULT_OUTPUTS_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/outputs.py"
)
DEFAULT_SCHEDULER_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py"
)
OUTPUT_OLD_LINE = (
    "    sampled_token_ids: list[list[int]] = [[0] for _ in req_ids]\n"
)
OUTPUT_NEW_LINE = (
    "    sampled_token_ids: list[list[int]] = [[] for _ in req_ids]\n"
)
SCHEDULER_OLD_FLAGS = (
    "        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder\n"
    "\n"
)
SCHEDULER_NEW_FLAGS = (
    "        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder\n"
    "        mm_config = vllm_config.model_config.multimodal_config\n"
    "        self.is_mm_encoder_only = bool(mm_config and mm_config.mm_encoder_only)\n"
    "\n"
)
SCHEDULER_OLD_STOP = (
    "            elif request.pooling_params and pooler_output is not None:\n"
    "                # Pooling stops as soon as there is output.\n"
    "                request.status = RequestStatus.FINISHED_STOPPED\n"
    "                stopped = True\n"
    "\n"
)
SCHEDULER_NEW_STOP = (
    "            elif request.pooling_params and pooler_output is not None:\n"
    "                # Pooling stops as soon as there is output.\n"
    "                request.status = RequestStatus.FINISHED_STOPPED\n"
    "                stopped = True\n"
    "            elif (\n"
    "                self.is_mm_encoder_only\n"
    "                and request.num_computed_tokens >= request.num_prompt_tokens\n"
    "            ):\n"
    "                # An encoder instance publishes embeddings instead of sampling.\n"
    "                # Once its prompt is consumed, every encoder item has run.\n"
    "                request.status = RequestStatus.FINISHED_STOPPED\n"
    "                stopped = True\n"
    "\n"
)
PatchText = Callable[[str], tuple[str, str]]


def patch_outputs_text(source: str) -> tuple[str, str]:
    """Return updated outputs.py source and applied/skipped/drift status."""
    old_count = source.count(OUTPUT_OLD_LINE)
    new_count = source.count(OUTPUT_NEW_LINE)
    if old_count == 1 and new_count == 0:
        updated = source.replace(OUTPUT_OLD_LINE, OUTPUT_NEW_LINE, 1)
        compile(updated, "outputs.py", "exec")
        return updated, "applied"
    if old_count == 0 and new_count == 1:
        return source, "skipped"
    return source, f"drift:old={old_count},new={new_count}"


def patch_scheduler_text(source: str) -> tuple[str, str]:
    """Return updated scheduler.py source and applied/skipped/drift status."""
    old_counts = (
        source.count(SCHEDULER_OLD_FLAGS),
        source.count(SCHEDULER_OLD_STOP),
    )
    new_counts = (
        source.count(SCHEDULER_NEW_FLAGS),
        source.count(SCHEDULER_NEW_STOP),
    )
    if old_counts == (1, 1) and new_counts == (0, 0):
        updated = source.replace(SCHEDULER_OLD_FLAGS, SCHEDULER_NEW_FLAGS, 1)
        updated = updated.replace(SCHEDULER_OLD_STOP, SCHEDULER_NEW_STOP, 1)
        compile(updated, "scheduler.py", "exec")
        return updated, "applied"
    if old_counts == (0, 0) and new_counts == (1, 1):
        return source, "skipped"
    return source, f"drift:old={old_counts},new={new_counts}"


def _prepare(path: Path, patch_text: PatchText) -> tuple[str, str]:
    return patch_text(path.read_text(encoding="utf-8"))


def patch_files(outputs_path: Path, scheduler_path: Path) -> tuple[str, str]:
    """Preflight both files, then write only when neither source has drifted."""
    outputs_source = outputs_path.read_text(encoding="utf-8")
    scheduler_source = scheduler_path.read_text(encoding="utf-8")
    outputs_updated, outputs_status = patch_outputs_text(outputs_source)
    scheduler_updated, scheduler_status = patch_scheduler_text(scheduler_source)
    valid = {"applied", "skipped"}
    if outputs_status not in valid or scheduler_status not in valid:
        return outputs_status, scheduler_status
    if outputs_status == "applied":
        outputs_path.write_text(outputs_updated, encoding="utf-8")
    if scheduler_status == "applied":
        scheduler_path.write_text(scheduler_updated, encoding="utf-8")
    return outputs_status, scheduler_status


def _targets(argv: list[str], offset: int) -> tuple[Path, Path] | None:
    remaining = argv[offset:]
    if not remaining:
        return DEFAULT_OUTPUTS_TARGET, DEFAULT_SCHEDULER_TARGET
    if len(remaining) == 2:
        return Path(remaining[0]), Path(remaining[1])
    return None


def main(argv: list[str]) -> int:
    status_only = len(argv) > 1 and argv[1] == "--status"
    targets = _targets(argv, 2 if status_only else 1)
    if targets is None:
        print(f"usage: {argv[0]} [--status] [OUTPUTS.py SCHEDULER.py]", file=sys.stderr)
        return 2
    outputs_path, scheduler_path = targets
    missing = [str(path) for path in targets if not path.is_file()]
    if missing:
        print(
            f"[empty-encoder-output] missing target(s): {', '.join(missing)}",
            file=sys.stderr,
        )
        return 1

    if status_only:
        _, outputs_status = _prepare(outputs_path, patch_outputs_text)
        _, scheduler_status = _prepare(scheduler_path, patch_scheduler_text)
        applied = outputs_status == scheduler_status == "skipped"
        label = (
            "APPLIED"
            if applied
            else f"NOT APPLIED (outputs={outputs_status}, scheduler={scheduler_status})"
        )
        print("empty encoder output semantics    :", label)
        return 0 if applied else 1

    outputs_status, scheduler_status = patch_files(outputs_path, scheduler_path)
    if outputs_status in {"applied", "skipped"} and scheduler_status in {
        "applied",
        "skipped",
    }:
        print(
            "[empty-encoder-output] "
            f"outputs={outputs_status}, scheduler={scheduler_status}"
        )
        return 0
    print(
        "[empty-encoder-output] source drift; refusing to patch "
        f"(outputs={outputs_status}, scheduler={scheduler_status})",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
