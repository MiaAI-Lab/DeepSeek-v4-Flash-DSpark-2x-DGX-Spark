#!/usr/bin/env python3
"""Tell a reasoning LOOP apart from a heavy TAIL, from the text alone.

Both produce the same user-visible symptom -- `finish_reason=length`, empty
content -- but they are different failures and want different fixes:

  heavy tail : the model is still saying new things, it just says a lot of them.
               Fix by raising max_tokens.
  loop       : the model recycles material it has already emitted, forever.
               Raising max_tokens changes nothing except the bill.

WHAT IT MEASURES

  novelty(window) = fraction of word 8-grams in this window that have not
                    appeared anywhere earlier in the same trace.

  Windows are taken over the WORD SEQUENCE (700 words per window), not the raw
  character stream. Char-stream windows truncate tokens at every boundary; for
  short-period verbatim loops the 1-2 novel boundary fragments dominate the
  window's small distinct-shingle population and hold novelty above threshold
  -- a one-sentence verbatim loop (CJK or English) can plateau near 40% novelty
  and read as HEAVY TAIL forever. Tokenize once, window over words.

  A healthy trace keeps novelty high to the end. A looping trace collapses AND
  NEVER RECOVERS -- and the verdict enforces both halves: after onset, novelty
  must stay below threshold to the end of the trace. A collapse that recovers
  (a long verbatim re-quote of an earlier code block, say) is reported as
  transient repetition, not as a loop.

WHY NOT BLOCK UNIQUENESS

  We first measured "unique 120-character blocks" and it told us the runaways
  were NOT loops. That was wrong, and wrong in a way worth warning about: the
  loop is *templated*, not verbatim -- a small set of stock phrases recombined
  with one varying element each pass. On the same three traces:

      unique 120-char blocks : 22% / 92% / 66%
      unique word 8-grams    :  3.4% / 4.0% / 2.8%

  One of those traces looks almost perfectly novel at block granularity and is
  96% recycled at phrase level. Any fixed-window uniqueness check reports a
  templated loop as fresh text. Count recycled n-grams, or unique lines.

  (Note the contrast needs NON-overlapping blocks; with stride-1 blocks both
  metrics agree at ~3-5%, which is another way to avoid the trap.)

CALIBRATION

  Measured over three captured non-terminating traces of ~100k tokens each:
  novelty collapses at token ~7.9k / ~13.8k / ~29.7k and never recovers --
  afterwards it oscillates between 0 and 0.6%, never above. It does not sit at
  exactly zero, so do not test for `== 0`; the threshold below is 2%.
  Re-verified after the switch to word-sequence windows: onsets ~7.2k / ~13.7k
  / ~30.2k (each within one window of the char-stream calibration), and two
  genuine long heavy-tail controls -- real length-capped-but-healthy traces of
  ~31k and ~34k tokens -- never trip the threshold in any window.

KNOWN LIMITATIONS

  A FRAGMENTED loop -- recycled fragments interleaved with enough novel filler
  that no single window's novelty drops below threshold -- is not detected
  (reported by @brianmswheart on the PR thread; real traces exist). That shape
  needs a cumulative recycled-mass tier, not a window verdict; out of scope
  here, deliberately, so this tool stays a one-screen instrument.

USAGE

  python3 loop_detector.py trace.txt [trace2.txt ...]
  cat trace.txt | python3 loop_detector.py

  Feed it the REASONING stream (the `reasoning` field, not `content`).
"""
import re
import sys
import zlib

WINDOW_WORDS = 700  # words per window (~4k chars of English prose)
THRESHOLD = 0.02    # novelty below this counts as exhausted
CONSECUTIVE = 3     # windows in a row before calling it a loop
CHARS_PER_TOKEN = 4.0   # rough; only used to report a token index


def analyse(text: str):
    """One row per word-window: (char_pos, novelty, compression_ratio)."""
    words = [(m.start(), m.group().lower()) for m in re.finditer(r"\w+", text)]
    seen: set = set()
    rows = []
    for i in range(0, len(words), WINDOW_WORDS):
        chunk = words[i:i + WINDOW_WORDS]
        if len(chunk) < WINDOW_WORDS // 2:
            break
        toks = [w for _, w in chunk]
        shingles = {tuple(toks[j:j + 8]) for j in range(max(0, len(toks) - 7))}
        novelty = len(shingles - seen) / len(shingles) if shingles else 1.0
        seen |= shingles
        pos = chunk[0][0]
        end = chunk[-1][0] + len(chunk[-1][1])
        seg = text[pos:end].encode()
        compression = len(zlib.compress(seg, 6)) / max(1, len(seg))
        rows.append((pos, novelty, compression))
    return rows


def verdict(rows):
    """(loop_onset_char_pos or None, [transient (start,end) char spans]).

    A loop verdict requires the collapse to PERSIST to the end of the trace:
    CONSECUTIVE dry windows open a candidate onset, and any later window back
    above threshold cancels it and records the span as transient repetition.
    """
    dry = 0
    onset_idx = None
    transients = []
    for i, (pos, novelty, _) in enumerate(rows):
        if novelty < THRESHOLD:
            dry += 1
            if dry == CONSECUTIVE and onset_idx is None:
                onset_idx = i - (CONSECUTIVE - 1)  # first of the dry windows
        else:
            if onset_idx is not None:
                transients.append((rows[onset_idx][0], pos))
                onset_idx = None
            dry = 0
    return (rows[onset_idx][0] if onset_idx is not None else None), transients


def report(name: str, text: str) -> None:
    rows = analyse(text)
    if not rows:
        print(f"{name}: too short to judge ({len(text)} chars)")
        return
    onset, transients = verdict(rows)
    print(f"\n== {name}  ({len(text):,} chars, ~{int(len(text)/CHARS_PER_TOKEN):,} tokens)")
    print(f"   {'tok~':>9}  {'novelty':>7}  {'compress':>8}")
    step = max(1, len(rows) // 12)
    for pos, novelty, comp in rows[::step]:
        print(f"   {int(pos/CHARS_PER_TOKEN):>9,}  {novelty:>7.2%}  {comp:>8.3f}")
    for a, b in transients:
        print(f"   -- transient repetition: novelty collapsed at ~token "
              f"{int(a/CHARS_PER_TOKEN):,} and recovered by ~{int(b/CHARS_PER_TOKEN):,}; "
              f"not a loop.")
    if onset is None:
        print("   -> HEAVY TAIL: novelty never collapses for good. The model is "
              "still producing new material; raise max_tokens.")
    else:
        tail = [n for p, n, _ in rows if p >= onset]
        print(f"   -> LOOP from token ~{int(onset/CHARS_PER_TOKEN):,} "
              f"(novelty stays below {max(tail):.2%} to the end of the trace). "
              f"Raising max_tokens will not help.")


def main() -> int:
    args = sys.argv[1:]
    if not args:
        data = sys.stdin.read()
        if not data.strip():
            print(__doc__)
            return 1
        report("<stdin>", data)
        return 0
    for path in args:
        with open(path, encoding="utf-8", errors="replace") as fh:
            report(path, fh.read())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
