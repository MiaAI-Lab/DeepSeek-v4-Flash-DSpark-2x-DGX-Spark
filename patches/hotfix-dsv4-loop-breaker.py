#!/usr/bin/env python3
"""Bounded loop-breaker: finish a generation cleanly when it degenerates into
line-level repetition (the MiaAI #82 announce-loop family).

WHY THIS LAYER
--------------
The multi-turn tool/reasoning attractor (MiaAI #82, Anemll #3) has no upstream
root-cause fix. Our stack deliberately suppresses spurious stopping (draft-EOS
penalty + suppress-stops-in-reasoning), which converts the attractor's failure
mode from "occasional early stop" (recoverable: retry) into "unbounded babble
until max_tokens or the user aborts" (unrecoverable). This hotfix restores a
bound WITHOUT re-enabling spurious EOS: when the tail of a generation repeats
the same normalized line over and over, the request is finished cleanly with
finish_reason=stop and stop_reason="loop-breaker".

Context: issue #82 (multi-turn tool/reasoning attractor). See the 2026-08-27
issue comment for the replay methodology, ablations, and dose-response data
behind the thresholds below.

WHERE
-----
vllm/v1/engine/detokenizer.py (CPU side, per request, outside CUDA graphs).
``BaseIncrementalDetokenizer.update`` already returns a matched stop string to
the output processor, which finishes the request as FINISHED_STOPPED. We add a
repetition detector that returns a synthetic stop marker through the same path.
It runs AFTER the [suppress-stops-in-reasoning] guard and is NOT gated by it:
loops inside reasoning must also be broken.

DETECTION
---------
Completed lines only (up to the last newline), normalized by strip +
whitespace-collapse. Two tiers, counted per request:
  * long lines (>=8 chars): >= DSPARK_LOOP_BREAKER_REPEATS (default 6)
  * short sentence-like lines (3-7 chars, has a letter, ends . ! ? — "Now.",
    "OK."): >= DSPARK_LOOP_BREAKER_SHORT_REPEATS (default 15)
Tool-call/JSON safety: lines containing DSML markers, braces, or quotes are
never counted, and the breaker never fires while the tail of the stream sits
inside an apparent DSML block (no unmatched marker in the last 400 chars).
It also never fires before DSPARK_LOOP_BREAKER_MIN_TOKENS (default 64) output
tokens.

Runtime opt-out (process-wide): DSPARK_LOOP_BREAKER=0.
Skip applying this file: DSPARK_SKIP_LOOP_BREAKER_HOTFIX=1.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

P = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/detokenizer.py")
MARK = "# [loop-breaker]"

INIT_OLD = """        # Generation data
        self.output_text = ""
"""

INIT_NEW = """        # [loop-breaker] per-request repeated-line detector state.
        self._lb_counts: dict[str, int] = {}
        self._lb_scan: int = 0

        # Generation data
        self.output_text = ""
"""

RET_OLD = """                stop_string, truncate_to = stop
                if truncate_to != -1:
                    self.output_text = self.output_text[:truncate_to]

        return stop_string
"""

RET_NEW = """                stop_string, truncate_to = stop
                if truncate_to != -1:
                    self.output_text = self.output_text[:truncate_to]

        # [loop-breaker] bounded degenerate-repetition guard (MiaAI #82).
        if stop_string is None and _LB_ENABLED:
            hit = self._lb_check()
            if hit is not None:
                return hit

        return stop_string

    def _lb_check(self) -> str | None:
        # [loop-breaker] scan newly completed lines; fire on repetition.
        text = self.output_text
        end = text.rfind("\\n")
        if end <= self._lb_scan:
            return None
        chunk = text[self._lb_scan:end]
        self._lb_scan = end
        fired = False
        for raw in chunk.split("\\n"):
            line = " ".join(raw.split())
            if not line or len(line) > 160:
                continue
            # tool-call / JSON / code safety: never count structural lines
            if ("{" in line or "}" in line or '"' in line or "DSML" in line
                    or "\\uff5c" in line or "｜" in line):
                continue
            if len(line) >= 8:
                n = self._lb_counts.get(line, 0) + 1
                self._lb_counts[line] = n
                if n >= _LB_REPEATS:
                    fired = True
            elif len(line) >= 3 and line[-1] in ".!?" and any(c.isalpha() for c in line):
                n = self._lb_counts.get(line, 0) + 1
                self._lb_counts[line] = n
                if n >= _LB_SHORT_REPEATS:
                    fired = True
        if not fired:
            return None
        if self.num_output_tokens() < _LB_MIN_TOKENS:
            return None
        # do not cut an in-flight DSML tool call: if the stream tail still looks
        # like it is inside a tool block, hold fire (we re-check on later chunks).
        if "DSML" in text[-400:] or "｜" in text[-400:]:
            return None
        return "loop-breaker"
"""

# Two possible header states: suppress-stops-in-reasoning applied (adds
# ``import os``) or skipped (stock header). Anchor on whichever is present.
IMPORT_OLD_WITH_SUPPRESS = "import os\nfrom abc import ABC, abstractmethod\n"
IMPORT_OLD_STOCK = "from abc import ABC, abstractmethod\n"
IMPORT_NEW = """import os
from abc import ABC, abstractmethod

# [loop-breaker] process-wide config (read once at import).
_LB_ENABLED = os.environ.get("DSPARK_LOOP_BREAKER", "1") != "0"
_LB_REPEATS = int(os.environ.get("DSPARK_LOOP_BREAKER_REPEATS", "6"))
_LB_SHORT_REPEATS = int(os.environ.get("DSPARK_LOOP_BREAKER_SHORT_REPEATS", "15"))
_LB_MIN_TOKENS = int(os.environ.get("DSPARK_LOOP_BREAKER_MIN_TOKENS", "64"))
"""


def apply_text(src: str) -> tuple[str, str]:
    if MARK in src and "_lb_check" in src:
        return src, "skipped"
    missing = []
    if IMPORT_OLD_WITH_SUPPRESS in src:
        import_old = IMPORT_OLD_WITH_SUPPRESS
    elif IMPORT_OLD_STOCK in src:
        import_old = IMPORT_OLD_STOCK
    else:
        import_old = None
    if import_old is None:
        missing.append("import")
    if INIT_OLD not in src:
        missing.append("init")
    if RET_OLD not in src:
        missing.append("return")
    if missing:
        return src, "missing:" + ",".join(missing)
    out = src.replace(import_old, IMPORT_NEW, 1)
    out = out.replace(INIT_OLD, INIT_NEW, 1)
    out = out.replace(RET_OLD, RET_NEW, 1)
    return out, "applied"


def main(argv: list[str]) -> int:
    if os.environ.get("DSPARK_SKIP_LOOP_BREAKER_HOTFIX") == "1":
        print("[loop-breaker] skipped via DSPARK_SKIP_LOOP_BREAKER_HOTFIX=1")
        return 0
    if len(argv) > 1 and argv[1] == "--status":
        target = Path(argv[2]) if len(argv) > 2 else P
        applied = target.is_file() and MARK in target.read_text()
        print("loop-breaker                   :", "APPLIED" if applied else "NOT APPLIED")
        return 0
    target = Path(argv[1]) if len(argv) > 1 else P
    if not target.is_file():
        print(f"[loop-breaker] missing {target}", file=sys.stderr)
        return 1
    original = target.read_text(encoding="utf-8")
    new, status = apply_text(original)
    if status == "applied":
        target.write_text(new, encoding="utf-8")
        # fail closed: the patched file must still compile
        try:
            compile(new, str(target), "exec")
        except SyntaxError as e:
            target.write_text(original, encoding="utf-8")
            print(f"[loop-breaker] FAILED compile check, restored original: {e}",
                  file=sys.stderr)
            return 1
    print(f"[loop-breaker] {status}: {target}")
    return 0 if status.startswith("applied") or status == "skipped" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
