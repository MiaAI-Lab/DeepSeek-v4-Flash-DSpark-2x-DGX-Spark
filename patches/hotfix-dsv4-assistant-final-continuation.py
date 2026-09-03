#!/usr/bin/env python3
"""Hotfix: emit a generation header when a request effectively ends with a closed assistant turn, including when a trailing latest_reminder annotation follows it.

Symptom
-------
An agent harness gets stuck emitting empty turns: 1-2 generated tokens with
`finish_reason: "stop"`, no tool call, and fragments of hallucinated markup
(`<result observation="no-op"></content>`, stray `</parameter>` / `</name>` /
`<value>`), or a sentence fragment that continues the previous turn instead of
answering. Server metrics during a live incident: 6 of 37 requests generated
<= 10 tokens, all `stop`, zero `length` — the model chose to stop rather than
being truncated.

Cause
-----
`render_message()` appends the generation header (`ASSISTANT_SP_TOKEN` plus the
thinking token) only when the trailing message is `user` or `developer`:

    elif messages[index].get("role") in ["user", "developer"]:

When the request's `messages` array ends with an **assistant** message, that
branch is skipped, the turn is closed with EOS, and the prompt ends on a bare
EOS with no header. The model is asked to generate from a closed, turnless
state. Measured on this recipe:

    assistant-final: '...to the reviewer for a fresh verdict:<|end of sentence|>'
    user-final:      '...verdict:<EOS><|User|>continue<|Assistant|><think>'

It is self-sustaining: the harness records the resulting empty turn, so the next
request also ends with an assistant message. One bad turn locks the loop;
inserting any user-role message breaks it instantly. Requests reach this shape
whenever a harness retries after a mid-stream error and re-sends the partial
assistant turn, which is why the same prompt works for a long time and then
does not.

The same retry shape can carry a trailing `latest_reminder` annotation after
the re-sent partial assistant turn (the harness appends fresh context as a
reminder message). The reminder defeats the fix twice over: stock closes the
assistant turn with EOS, renders the bare reminder after it, and the prompt
still ends with no generation header — the model reads the reminder from the
same dead state.

Why a header and not `wo_eos`
-----------------------------
`encoding_dsv4.py` also has `assistant_msg_wo_eos_template`, and reopening the
trailing turn looks like a tempting one-line fix. It is wrong. Measured on one
prompt via /v1/completions, generated tokens:

    trailing turn   | stock (EOS) | wo_eos (reopen) | EOS + header
    partial         | 32          | 208             | 260
    complete        | 16 (fragment)| 1 (EMPTY, dead)| 140 (healthy)

Reopening a *complete* assistant turn just moves the dead state: the model has
nothing left to add and emits EOS immediately. Appending a fresh generation
header after the closed assistant turn is correct for both shapes and matches
the checkpoint encoder's existing generation transition.

Scope
-----
Only the final message of a request is affected, and only when it is an
assistant turn, or a `latest_reminder` whose immediate predecessor is an
assistant turn. Every other rendering path is untouched, including:
consecutive assistant messages mid-transcript; reminders mid-transcript; and
reminder tails directly after a `user`/`developer` message — those already
end inside the pending generation slot (the checkpoint emits
`ASSISTANT_SP_TOKEN` + thinking token *before* such a reminder) and must stay
byte-identical.

Gating and fail-closed operation
--------------------------------
The compose entrypoint invokes this script only when
`DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX` is exactly `1` (default `0` = stock
renderer, this script never runs), and chains it with `|| exit 1`. Because an
invocation means the operator asked for the fix, everything fails nonzero:

- encoder file missing (the prerequisite `encoding_dsv4.py` copy did not
  happen) — a gated-ON boot must not silently serve the buggy stock renderer;
- anchor text missing (upstream encoder drifted) — nothing is written;
- post-write self-check failure (patched module does not import, a fixed
  shape still renders without a generation header — assistant-final, or
  assistant-final plus trailing `latest_reminder` — or a
  user->latest_reminder tail gains a second header) — the original file
  bytes are restored first, then exit 1.

Idempotent: an already-patched encoder is not rewritten, but it must still pass
the self-check. Patches the encoding module the server actually loads, so it
must run AFTER the compose entrypoint copies `encoding_dsv4.py` into place.

Usage (inside container, after encoder copy):
  python3 hotfix-dsv4-assistant-final-continuation.py
  python3 hotfix-dsv4-assistant-final-continuation.py /path/to/deepseek_v4_encoding.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

DEFAULT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/deepseek_v4_encoding.py"
)
MARK = "[assistant-final-hotfix]"

OLD = (
    "    elif messages[index].get(\"role\") in [\"user\", \"developer\"]:\n"
    "        # Normal generation: append Assistant + thinking token\n"
)
NEW = (
    "    elif messages[index].get(\"role\") in [\"user\", \"developer\"] or (\n"
    f"        # {MARK} A request may legitimately end with an assistant turn\n"
    "        # (harness retry, continuation), optionally annotated by a\n"
    "        # trailing latest_reminder harness message. Without a generation\n"
    "        # header the prompt ends on a bare EOS (or a bare reminder after\n"
    "        # the closed turn) and the model generates from a dead state:\n"
    "        # immediate EOS, or raw DSML markup emitted as text. A reminder\n"
    "        # tail directly after user/developer already ends inside the\n"
    "        # pending generation slot and must stay byte-identical.\n"
    "        messages[index].get(\"role\") == \"assistant\"\n"
    "        and index == len(messages) - 1\n"
    "    ) or (\n"
    "        messages[index].get(\"role\") == \"latest_reminder\"\n"
    "        and index == len(messages) - 1\n"
    "        and index > 0\n"
    "        and messages[index - 1].get(\"role\") == \"assistant\"\n"
    "    ):\n"
    "        # Normal generation: append Assistant + thinking token\n"
)


def _self_check(target: Path) -> tuple[bool, str]:
    """Import the patched encoder and confirm the fix actually renders.

    A trailing-assistant transcript must end with the generation header
    (assistant speaker token + thinking token) instead of a bare EOS, the
    same shape annotated by a trailing latest_reminder must regain that
    fresh header after the reminder, and a user->latest_reminder tail must
    keep its stock single in-slot header (no second header appended).
    """
    spec = importlib.util.spec_from_file_location("enc_check", target)
    if spec is None or spec.loader is None:
        return False, f"cannot load module spec from {target}"
    enc = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(enc)
        base = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "A finished answer."},
        ]
        rendered = enc.encode_messages(base, "thinking", reasoning_effort="high")
        reminded = enc.encode_messages(
            base + [{"role": "latest_reminder", "content": "Fresh context."}],
            "thinking",
            reasoning_effort="high",
        )
        intact = enc.encode_messages(
            [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "latest_reminder", "content": "Fresh context."},
            ],
            "thinking",
            reasoning_effort="high",
        )
        speaker = getattr(
            enc, "ASSISTANT_SP_TOKEN", getattr(enc, "assistant_sp_token", None)
        )
        thinking = getattr(enc, "thinking_start_token", None)
    except Exception as err:  # broken/unimportable patch must fail closed
        return False, f"self-check raised {type(err).__name__}: {err}"
    if not speaker or not thinking:
        return False, "generation-header tokens are unavailable"

    def _ends_with_header(out):
        if isinstance(out, str):
            return out.endswith(speaker + thinking)
        if isinstance(out, (list, tuple)):
            return list(out[-2:]) == [speaker, thinking]
        raise TypeError(f"unexpected encoder output type: {type(out).__name__}")

    try:
        valid = _ends_with_header(rendered) and _ends_with_header(reminded)
        # The user->latest_reminder tail ends inside the pending generation
        # slot (exactly one header, before the reminder); an over-broad
        # transition would append a second header there.
        untouched = intact.count(speaker) == 1 and not _ends_with_header(intact)
    except TypeError as err:
        return False, str(err)
    if not valid:
        return False, f"generation header does not terminate render: {rendered[-80:]!r}"
    if not untouched:
        return False, "transition widened too far: user->latest_reminder tail changed"
    return True, "generation headers terminate assistant-final renders"


def main(argv: list[str]) -> int:
    target = Path(argv[1]) if len(argv) > 1 else DEFAULT_TARGET

    if not target.is_file():
        # Invoked == gated ON: a missing encoder is a prerequisite failure,
        # not a skip (compose chains this with `|| exit 1`).
        print(f"[FAIL] {MARK} encoder file not found: {target}", file=sys.stderr)
        return 1

    src = target.read_text(encoding="utf-8")

    if MARK in src:
        ok, why = _self_check(target)
        if ok:
            print(f"[OK] {MARK} already applied and verified: {target}")
            return 0
        print(
            f"[FAIL] {MARK} already applied but self-check failed: {why}",
            file=sys.stderr,
        )
        return 1

    if OLD not in src:
        print(f"[FAIL] {MARK} anchor not found in {target}", file=sys.stderr)
        return 1

    target.write_text(src.replace(OLD, NEW, 1), encoding="utf-8")
    ok, why = _self_check(target)
    if ok:
        print(f"[OK] {MARK} patched and verified: {target}")
        return 0

    # Fail closed: never leave a written-but-unverified encoder behind.
    target.write_text(src, encoding="utf-8")
    print(
        f"[FAIL] {MARK} self-check failed, original restored ({target}): {why}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
