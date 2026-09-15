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
behind the thresholds below. This is a mitigation, not a #82 root-cause fix.

OPT-IN (default OFF)
--------------------
The detector is armed only by the exact value ``DSPARK_LOOP_BREAKER=1``, and the
compose chain applies this file only then: a default boot never touches
detokenizer.py and serves byte-identically to stock. While disabled the numeric
knobs are not read at all, so a malformed knob cannot affect a boot that does
not use it. The flag is re-read at vLLM import inside the patched module, so a
detokenizer patched by an earlier enabled boot stays inert whenever the flag is
not exactly "1".

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
whitespace-collapse. Two tiers, counted per request inside a bounded sliding
window:
  * long lines (>=8 chars): >= DSPARK_LOOP_BREAKER_REPEATS (default 6)
  * short sentence-like lines (3-7 chars, has a letter, ends . ! ? — "Now.",
    "OK."): >= DSPARK_LOOP_BREAKER_SHORT_REPEATS (default 15)
False-positive controls: fenced (``` / ~~~) and indented code blocks are never
counted, nor are lines carrying braces, quotes, backticks or DSML markers, and
nor are lines with no alphanumeric content (rules, tables, separators) or lines
longer than 160 chars. The breaker holds fire while the stream tail sits inside
an apparent DSML block (a DSML marker in the last 400 chars) and never fires
before DSPARK_LOOP_BREAKER_MIN_TOKENS (default 64) output tokens.

STATE BOUNDS
------------
Per request the detector keeps only the last ``max(64, 2 * max(repeats,
short_repeats))`` counted lines (a deque plus a same-keyed count dict), so a
long generation cannot grow the counters without bound; evicted lines stop
counting, so repeats must land inside the window to fire.

KNOBS (read and range-checked only while enabled)
-------------------------------------------------
  DSPARK_LOOP_BREAKER_REPEATS         2..1024     default 6
  DSPARK_LOOP_BREAKER_SHORT_REPEATS   2..1024     default 15
  DSPARK_LOOP_BREAKER_MIN_TOKENS      0..1000000  default 64
A lower bound of 2 is deliberate: 1 or 0 would fire on the first eligible line.
Malformed or out-of-range values fail the preflight/boot (exit 1) with the
offending variable named; they are never replaced by a silent default.

CLI
---
  python3 hotfix-dsv4-loop-breaker.py            apply (fail closed)
  python3 hotfix-dsv4-loop-breaker.py --check    preflight: knobs + target
  python3 hotfix-dsv4-loop-breaker.py --status   classify the target bytes
``--status`` inspects the file bytes even when the skip flag is set, and exits
nonzero unless every injected block is present exactly once and the module
still compiles; a partial patch (marker without the hooks) is never reported as
applied. Applies are staged in a same-directory temp file and ``os.replace()``d,
preserving the file mode, and are verified after the replace: a re-read or
decode failure, or bytes that no longer classify as applied, restore the
original and exit 1.
Skip applying this file: DSPARK_SKIP_LOOP_BREAKER_HOTFIX=1 (status queries are
never skipped).
"""
from __future__ import annotations

import os
import stat
import sys
import tempfile
from pathlib import Path

P = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/detokenizer.py")
MARK = "# [loop-breaker]"
USAGE = "usage: hotfix-dsv4-loop-breaker.py [--check|--status] [TARGET]"

# Single source of truth for the numeric knobs: (name, default, low, high).
# The CLI validates against this table and the same table is injected into the
# patched module, so both sides agree by construction.
KNOB_SPECS: tuple[tuple[str, int, int, int], ...] = (
    ("DSPARK_LOOP_BREAKER_REPEATS", 6, 2, 1024),
    ("DSPARK_LOOP_BREAKER_SHORT_REPEATS", 15, 2, 1024),
    ("DSPARK_LOOP_BREAKER_MIN_TOKENS", 64, 0, 1000000),
)

# Markers that mean "in a tool block" on the line and tail checks, and the
# characters that make a line structural (JSON, inline code, tables) rather
# than prose. Injected into the patched module.
MARKER_LITERAL = '("DSML", "\\uff5c")'
STRUCTURAL_LITERAL = '("{", "}", \'"\', "`", "|")'


class LoopBreakerConfigError(RuntimeError):
    """Enabled boot with a malformed or out-of-range knob."""


def resolve_knobs(environ=os.environ) -> tuple[int, int, int]:
    """Validated (repeats, short_repeats, min_tokens) for an enabled boot."""
    values = []
    for name, default, low, high in KNOB_SPECS:
        raw = environ.get(name, "")
        if raw == "":
            values.append(default)
            continue
        if not (raw.isascii() and raw.isdigit()):
            raise LoopBreakerConfigError(
                f"{name} must be a non-negative integer (got {raw!r})"
            )
        # Bound the decimal magnitude before int(): a normalized digit string
        # longer than the high bound cannot be in range, and converting it
        # first could raise an int() conversion error (Python's digit limit for
        # huge strings) instead of the named config error.
        digits = raw.lstrip("0")
        if len(digits) > len(str(high)):
            raise LoopBreakerConfigError(
                f"{name} must be between {low} and {high} (got {raw})"
            )
        value = int(digits) if digits else 0
        if not low <= value <= high:
            raise LoopBreakerConfigError(
                f"{name} must be between {low} and {high} (got {raw})"
            )
        values.append(value)
    return values[0], values[1], values[2]


INIT_OLD = """        # Generation data
        self.output_text = ""
"""

INIT_NEW = """        # [loop-breaker] bounded repeated-line detector state (issue #82).
        self._lb_counts: dict[str, int] = {}
        self._lb_hist: deque[str] = deque()
        self._lb_scan: int = 0
        self._lb_fence: str | None = None

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
        # [loop-breaker] scan newly completed lines; fire only past the bounds.
        text = self.output_text
        end = text.rfind("\\n")
        if end < 0:
            return None
        if end < self._lb_scan:
            # output_text was rewritten shorter (spec-decode rollback): drop the
            # window instead of rescanning offsets that no longer exist.
            self._lb_hist.clear()
            self._lb_counts.clear()
            self._lb_fence = None
            self._lb_scan = end
            return None
        if end == self._lb_scan:
            return None
        chunk = text[self._lb_scan:end]
        self._lb_scan = end
        fired = False
        for raw in chunk.split("\\n"):
            stripped = raw.strip()
            if stripped[:3] in ("```", "~~~"):
                # fence marker: fenced content is code, never counted
                if self._lb_fence is None:
                    self._lb_fence = stripped[:3]
                elif self._lb_fence == stripped[:3]:
                    self._lb_fence = None
                continue
            if self._lb_fence is not None:
                continue
            if raw.startswith("    ") or raw.startswith("\\t"):
                # indented code block: repeated code lines are legitimate
                continue
            line = " ".join(stripped.split())
            if not line or len(line) > 160:
                continue
            # tool-call / JSON / inline-code / table safety: never count these
            if any(marker in line for marker in _LB_MARKERS):
                continue
            if any(char in line for char in _LB_STRUCTURAL):
                continue
            if not any(char.isalnum() for char in line):
                # rules, table separators, bullet-only noise
                continue
            if len(line) >= 8:
                threshold = _LB_REPEATS
            elif 3 <= len(line) <= 7 and line[-1] in ".!?" and any(
                char.isalpha() for char in line
            ):
                threshold = _LB_SHORT_REPEATS
            else:
                continue
            if self._lb_count(line) >= threshold:
                fired = True
        if not fired:
            return None
        if self.num_output_tokens() < _LB_MIN_TOKENS:
            return None
        # do not cut an in-flight DSML tool call: if the stream tail still looks
        # like it is inside a tool block, hold fire (we re-check on later chunks).
        if any(marker in text[-400:] for marker in _LB_MARKERS):
            return None
        return "loop-breaker"

    def _lb_count(self, line: str) -> int:
        # [loop-breaker] bounded sliding window over counted lines: a long
        # generation cannot grow the counters without bound, and evicted lines
        # stop counting, so repeats must land inside the window to fire.
        hist = self._lb_hist
        if len(hist) >= _LB_WINDOW:
            old = hist.popleft()
            remaining = self._lb_counts[old] - 1
            if remaining > 0:
                self._lb_counts[old] = remaining
            else:
                del self._lb_counts[old]
        hist.append(line)
        count = self._lb_counts.get(line, 0) + 1
        self._lb_counts[line] = count
        return count
"""

# Two possible header states: suppress-stops-in-reasoning applied (adds
# ``import os``) or skipped (stock header). Anchor on whichever is present. The
# injected block goes above ``import os`` so both anchors stay contiguous for
# the sibling detokenizer patcher.
IMPORT_OLD_WITH_SUPPRESS = "import os\nfrom abc import ABC, abstractmethod\n"
IMPORT_OLD_STOCK = "from abc import ABC, abstractmethod\n"
_IMPORT_TEMPLATE = """import os
from collections import deque

# [loop-breaker] config, read once at import. Opt-in: only the exact value
# DSPARK_LOOP_BREAKER=1 arms the detector, so a detokenizer patched by an
# earlier enabled boot stays inert whenever the flag is off. The numeric knobs
# are not parsed at all while disabled. Ranges mirror the patch's preflight
# validation; a malformed value here means the preflight was bypassed, so the
# detector disarms (never abort a vLLM import, never truncate with an
# unintended threshold). The warning uses its own logger: this runs inside the
# import block, before the module's logger exists.
_LB_ENABLED = os.environ.get("DSPARK_LOOP_BREAKER", "0") == "1"
_LB_REPEATS, _LB_SHORT_REPEATS, _LB_MIN_TOKENS = 0, 0, 0
_LB_WINDOW = 1
_LB_SPECS = __LB_SPECS__
_LB_MARKERS = __LB_MARKERS__
_LB_STRUCTURAL = __LB_STRUCTURAL__
if _LB_ENABLED:
    import logging

    _lb_valid = True
    _lb_values = {}
    for _lb_name, _lb_default, _lb_low, _lb_high in _LB_SPECS:
        _lb_raw = os.environ.get(_lb_name, "")
        if _lb_raw == "":
            _lb_values[_lb_name] = _lb_default
            continue
        # Bound the decimal magnitude before int(): a normalized digit string
        # longer than the high bound cannot be in range, and converting it
        # first could raise an int() conversion error out of the import
        # instead of disarming the detector.
        _lb_digits = (
            _lb_raw.lstrip("0")
            if _lb_raw.isascii() and _lb_raw.isdigit()
            else None
        )
        if _lb_digits is not None and len(_lb_digits) <= len(str(_lb_high)):
            _lb_value = int(_lb_digits) if _lb_digits else 0
            if _lb_low <= _lb_value <= _lb_high:
                _lb_values[_lb_name] = _lb_value
                continue
        logging.getLogger("vllm.loop-breaker").warning(
            "[loop-breaker] %s=%r is outside %d..%d; detector disarmed",
            _lb_name, _lb_raw, _lb_low, _lb_high,
        )
        _lb_valid = False
        _LB_ENABLED = False
    if _lb_valid:
        _LB_REPEATS = _lb_values["DSPARK_LOOP_BREAKER_REPEATS"]
        _LB_SHORT_REPEATS = _lb_values["DSPARK_LOOP_BREAKER_SHORT_REPEATS"]
        _LB_MIN_TOKENS = _lb_values["DSPARK_LOOP_BREAKER_MIN_TOKENS"]
        _LB_WINDOW = max(64, 2 * max(_LB_REPEATS, _LB_SHORT_REPEATS))
    del _lb_valid, _lb_values
from abc import ABC, abstractmethod
"""
IMPORT_NEW = (
    _IMPORT_TEMPLATE.replace("__LB_SPECS__", repr(KNOB_SPECS))
    .replace("__LB_MARKERS__", MARKER_LITERAL)
    .replace("__LB_STRUCTURAL__", STRUCTURAL_LITERAL)
)

# The complete injected blocks, in apply order. ``classify`` reports "applied"
# only when every block is present exactly once and the module still compiles:
# a substring hit is not enough, because deleting executable injected code (a
# gutted method body, a dropped hook or state field) can leave the individual
# markers behind while the detector is silently disarmed. A partial patch
# (marker without the hooks) is a distinct, fail-closed state rather than an
# accepted idempotent hit.
PATCH_BLOCKS = (IMPORT_NEW, INIT_NEW, RET_NEW)


def classify(src: str) -> str:
    """Return applied|partial|stock for ``src`` (see PATCH_BLOCKS)."""
    counts = tuple(src.count(block) for block in PATCH_BLOCKS)
    if all(count == 1 for count in counts):
        try:
            compile(src, "<loop-breaker>", "exec")
        except (SyntaxError, ValueError):
            # Null bytes and other source-encoding garbage are not executable
            # injected code either.
            return "partial"
        return "applied"
    if any(counts) or MARK in src:
        return "partial"
    return "stock"


def apply_text(src: str) -> tuple[str, str]:
    """Return (new_source, status): applied|skipped|partial|missing:<parts>."""
    state = classify(src)
    if state == "applied":
        return src, "skipped"
    if state == "partial":
        return src, "partial"
    if IMPORT_OLD_WITH_SUPPRESS in src:
        import_old = IMPORT_OLD_WITH_SUPPRESS
    elif IMPORT_OLD_STOCK in src:
        import_old = IMPORT_OLD_STOCK
    else:
        import_old = None
    missing = []
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


def write_atomically(target: Path, payload: str, mode: int) -> None:
    # Same-directory staging + os.replace: a reader never sees a half-written
    # detokenizer, and the target's mode is carried over explicitly.
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.loop-breaker.", dir=str(target.parent)
    )
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp, mode)
        os.replace(temp, target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def main(argv: list[str]) -> int:
    mode = "apply"
    positional = []
    for arg in argv[1:]:
        if arg in ("--check", "--status"):
            mode = arg[2:]
        elif arg.startswith("-"):
            print(f"[loop-breaker] unknown option {arg!r}\n{USAGE}", file=sys.stderr)
            return 2
        else:
            positional.append(arg)
    target = Path(positional[0]) if positional else P
    enabled = os.environ.get("DSPARK_LOOP_BREAKER", "0") == "1"

    if mode == "status":
        # A query about bytes on disk: independent of the enable and skip
        # flags, and nonzero unless the complete patch is present.
        if not target.is_file():
            print(f"loop-breaker                   : NOT APPLIED (missing {target})")
            return 1
        state = classify(target.read_text(encoding="utf-8"))
        print(f"loop-breaker                   : {state.upper()} ({target})")
        return 0 if state == "applied" else 1

    if os.environ.get("DSPARK_SKIP_LOOP_BREAKER_HOTFIX") == "1":
        print("[loop-breaker] skipped via DSPARK_SKIP_LOOP_BREAKER_HOTFIX=1")
        return 0

    if not enabled:
        # Default-OFF boot: no knob is parsed and no byte is written.
        print(
            "[loop-breaker] disabled (DSPARK_LOOP_BREAKER is not 1); "
            f"nothing to do for {target}"
        )
        return 0

    try:
        repeats, short_repeats, min_tokens = resolve_knobs()
    except LoopBreakerConfigError as error:
        print(f"[loop-breaker] FAIL-CLOSED: {error}", file=sys.stderr)
        return 1
    if not target.is_file():
        print(f"[loop-breaker] missing {target}", file=sys.stderr)
        return 1
    original = target.read_text(encoding="utf-8")
    state = classify(original)
    if mode == "check":
        if state == "applied":
            report = "READY (already applied)"
        elif state == "partial":
            print(
                f"[loop-breaker] FAIL-CLOSED: {target} carries a partial patch; "
                "recreate the container from the image",
                file=sys.stderr,
            )
            return 1
        else:
            _, anchor_status = apply_text(original)
            if anchor_status != "applied":
                print(
                    f"[loop-breaker] FAIL-CLOSED: {target} does not carry the "
                    f"expected anchors ({anchor_status})",
                    file=sys.stderr,
                )
                return 1
            report = "READY"
        print(
            f"loop-breaker                   : {report} "
            f"(repeats={repeats} short_repeats={short_repeats} "
            f"min_tokens={min_tokens}) ({target})"
        )
        return 0

    new, status = apply_text(original)
    if status == "skipped":
        print(f"[loop-breaker] skipped: {target} (complete patch already present)")
        return 0
    if status != "applied":
        remedy = (
            "recreate the container from the image"
            if status == "partial"
            else "the pinned detokenizer source has drifted"
        )
        print(
            f"[loop-breaker] FAIL-CLOSED: {status} for {target}; {remedy}",
            file=sys.stderr,
        )
        return 1
    try:
        # Fail closed before writing: the patched module must still compile.
        compile(new, str(target), "exec")
    except SyntaxError as error:
        print(
            f"[loop-breaker] FAIL-CLOSED: patched source does not compile "
            f"({error}); {target} left untouched",
            file=sys.stderr,
        )
        return 1
    mode_bits = stat.S_IMODE(target.stat().st_mode)
    try:
        write_atomically(target, new, mode_bits)
    except OSError as error:
        print(
            f"[loop-breaker] FAIL-CLOSED: cannot write {target} ({error}); "
            "target left untouched",
            file=sys.stderr,
        )
        return 1
    failure = None
    try:
        written = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        failure = f"post-apply re-read failed ({error})"
    else:
        if written != new or classify(written) != "applied":
            failure = "post-apply verification failed"
    if failure is not None:
        # Never leave a written-but-unverified or unreadable module.
        try:
            write_atomically(target, original, mode_bits)
        except OSError as error:
            print(
                f"[loop-breaker] FAIL-CLOSED: {failure}; cannot restore "
                f"{target} ({error})",
                file=sys.stderr,
            )
            return 1
        print(
            f"[loop-breaker] FAIL-CLOSED: {failure}; original restored "
            f"({target})",
            file=sys.stderr,
        )
        return 1
    print(
        f"[loop-breaker] applied: {target} (repeats={repeats} "
        f"short_repeats={short_repeats} min_tokens={min_tokens})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
