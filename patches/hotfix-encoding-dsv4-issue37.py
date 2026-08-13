#!/usr/bin/env python3
"""Fix high/max reasoning-effort degeneration with DeepSeek V4 tools.

The stock encoder's high/max prefixes demand exhaustive, unbounded reasoning
and the complete deliberation before a tool call. That conflicts with the
DSML tools template and makes the model loop, leak DSML, or exhaust the output
budget without calling the requested tool.

This hotfix does two deliberately small things:

* replaces the unbounded no-tools high/max prompts with distinct, bounded,
  goal-directed prompts that reserve room for the answer; and
* suppresses the effort prefix on tool-bearing turns. Thinking mode remains
  enabled, but tool selection uses the model's known-good base prompt instead
  of the conflicting high/max injection (the fix validated in issue #37).

Usage (inside the container, after the encoder copy):
  python3 hotfix-encoding-dsv4-issue37.py
  python3 hotfix-encoding-dsv4-issue37.py /path/to/deepseek_v4_encoding.py
"""
from __future__ import annotations

import sys
from pathlib import Path

DEFAULT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/deepseek_v4_encoding.py"
)

OLD_PROMPTS = '''REASONING_EFFORT_PROMPTS: Dict[str, str] = {
    "low": "",
    "high": (
        "Reasoning Effort: Absolute maximum with no shortcuts permitted.\\n"
        "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\\n"
        "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\\n\\n"
    ),
    "max": (
        "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\\n"
        "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\\n"
        "Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\\n\\n"
    ),
}'''

NEW_PROMPTS = '''REASONING_EFFORT_PROMPTS: Dict[str, str] = {
    "low": "",
    "high": (
        "Use high reasoning effort adaptively, not as a minimum. For a "
        "straightforward task, perform one brief correctness check and answer "
        "directly; target at most 128 reasoning tokens. For a difficult task, "
        "analyze only alternatives that could change the answer and cap "
        "reasoning near 768 tokens. Never repeat completed checks. Stop when "
        "the answer is supported.\\n\\n"
    ),
    "max": (
        "Use maximum reasoning effort adaptively, not as a minimum. For a "
        "straightforward task, perform one brief correctness check and answer "
        "directly; target at most 128 reasoning tokens. For a genuinely "
        "difficult task, examine plausible alternatives and independently "
        "verify critical claims, but cap reasoning near 1536 tokens. Never "
        "repeat completed analysis. Stop when further thought is unlikely to "
        "change the result.\\n\\n"
    ),
}'''

OLD_GUARD = '''    if index == 0 and thinking_mode == "thinking":
        prompt += REASONING_EFFORT_PROMPTS[reasoning_effort]'''

NEW_GUARD = '''    # [issue37-hotfix] A high/max effort prefix conflicts with the DSML
    # instruction to finish reasoning before tools. Keep thinking enabled, but
    # use the model's known-good base tool prompt whenever schemas are present.
    if index == 0 and thinking_mode == "thinking" and not tools:
        prompt += REASONING_EFFORT_PROMPTS[reasoning_effort]'''


def patch_text(source: str) -> tuple[str, str]:
    """Return (updated_text, status), where status is applied|skipped|missing."""
    prompt_done = NEW_PROMPTS in source
    guard_done = NEW_GUARD in source
    if prompt_done and guard_done:
        return source, "skipped"
    if (not prompt_done and OLD_PROMPTS not in source) or (
        not guard_done and OLD_GUARD not in source
    ):
        return source, "missing"
    updated = source
    if not prompt_done:
        updated = updated.replace(OLD_PROMPTS, NEW_PROMPTS, 1)
    if not guard_done:
        updated = updated.replace(OLD_GUARD, NEW_GUARD, 1)
    return updated, "applied"


def patch_file(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    updated, status = patch_text(source)
    if status == "applied":
        path.write_text(updated, encoding="utf-8")
    return status


def main(argv: list[str]) -> int:
    target = Path(argv[1]) if len(argv) > 1 else DEFAULT_TARGET
    if not target.is_file():
        print(f"[FAIL] encoding file not found: {target}", file=sys.stderr)
        return 1
    status = patch_file(target)
    if status == "applied":
        print(f"[OK] Issue #37 high/max encoder patch applied: {target}")
        return 0
    if status == "skipped":
        print(f"[OK] Issue #37 high/max encoder patch already present: {target}")
        return 0
    print(
        f"[FAIL] Issue #37 encoder anchors not found; refusing partial patch: {target}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
