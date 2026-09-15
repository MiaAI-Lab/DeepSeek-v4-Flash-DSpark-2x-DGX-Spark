#!/usr/bin/env python3
"""DSML recovery: recover DeepSeek V4 tool calls with malformed DSML wrappers
(opt-in port of open upstream vllm-project/vllm#52645, head 3df9776b0d).

DeepSeek V4 can intermittently emit an otherwise complete DSML
``<invoke name="...">`` block while the outer ``tool_calls`` opener is missing
or malformed (an observed DeepSeek-V4-Flash-0731 variant emits ``toolcalls``,
upstream #51914).  The pinned Anemll parser only enters the tool-call state
machine from CONTENT or REASONING on the exact outer opener, so the whole
invoke leaks verbatim into user-visible content (or reasoning) and the
structured tool call is lost -- the DSpark agent lane then sees prose instead
of a tool call.

The port teaches the streaming parser engine a *provisional* recovery path:

* ``deepseek_v4.py`` -- a bare ``<invoke name="..."`` seen from CONTENT or
  REASONING starts a provisional tool call; V3.2 ``function_calls`` wrappers
  become explicit passthrough states so foreign-wrapped invokes are kept
  verbatim and never recovered; ``</invoke>`` commits; the parser binds a
  validator that accepts only tool names declared by the *live request*
  (and rejects everything under ``tool_choice="none"``).
* ``streaming_parser_engine.py`` -- the engine buffers a provisional invoke's
  semantic events and raw text, validates the completed name, commits only on
  the configured ``INVOKE_END`` transition, then returns to CONTENT (one
  optional outer ``tool_calls`` closer is absorbed); every other exit --
  truncation, a stray outer closer, a rejected name, end of generation --
  restores the buffered text to its original content or reasoning state
  unchanged.  Parser-level drop tokens (EOS) never enter the buffers.
* ``parser_engine_config.py`` -- ``Transition`` gains the opt-in
  ``provisional_tool_call`` / ``commit_provisional_tool_call`` markers and
  ``ParserState`` the two foreign-wrapper states (no behavior change for
  configs that do not set them).
* ``adapters.py`` / ``abstract_parser.py`` / ``parser_engine.py`` -- the
  request's tools and ``tool_choice`` are mirrored into the reasoning-side
  engine before recovery validation (non-streaming and per-delta), and a
  rolled-back candidate parked in deferred reasoning is flushed on finish.

The normal correctly wrapped DSML path is unchanged --
``scripts/test-dsml-recovery.py`` proves stock/patched parity on it and
reproduces the upstream regression matrix against the pinned fixtures.

All six targets are touched by no other recipe hotfix, so each is pinned by
whole-file identity (stock and patched sha256+size).  The opt-in Compose gate
runs this before ``vllm serve``; it accepts only the pinned Anemll 0.1.1 vLLM
version.  Apply preflights every target before writing any, publishes one
same-directory atomic replace per file, rolls freshly written files back to
stock if a later file fails, and verifies an already-patched target without
rewriting it.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

PRODUCTION_ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")
EXPECTED_VLLM_VERSION = "0.25.2.dev0+g752a3a504.d20260714"
MARK = "# [dspark-dsml-recovery]"

REGIONS_CONFIG = (
    # ParserState: add the two foreign V3.2 wrapper states.
    (
rb'''    TOOL_NAME = auto()
    TOOL_ARGS = auto()
    TOOL_BETWEEN = auto()
''',
rb'''    TOOL_NAME = auto()
    TOOL_ARGS = auto()
    TOOL_BETWEEN = auto()
    # [dspark-dsml-recovery] upstream vllm#52645: V3.2 wrapper passthrough
    # states so foreign-wrapped invokes never enter the V4 recovery path.
    FOREIGN_BLOCK = auto()
    FOREIGN_REASONING_BLOCK = auto()
''',
    ),
    # Transition: add the provisional-start/commit recovery markers.
    (
rb'''    next_state: ParserState
    events: tuple[EventType, ...] = field(default_factory=tuple)
    skip_in_token_id_mode: bool = False
''',
rb'''    next_state: ParserState
    events: tuple[EventType, ...] = field(default_factory=tuple)
    skip_in_token_id_mode: bool = False
    # [dspark-dsml-recovery] upstream vllm#52645.
    # Treat this transition as a provisional tool-call recovery path. The
    # engine buffers its semantic events, validates the completed tool name
    # through a parser-supplied callback, and commits only after the call ends.
    provisional_tool_call: bool = False
    # Commit a buffered provisional call at this transition. Recovery paths
    # that leave tool arguments through any other transition are restored as
    # ordinary text.
    commit_provisional_tool_call: bool = False
''',
    ),
)

REGIONS_ENGINE = (
    # import Callable for the validator attribute.
    (
rb'''from collections.abc import Sequence
''',
rb'''from collections.abc import Callable, Sequence
''',
    ),
    # tool-exit terminal set + validator binding in __init__.
    (
rb'''        self.skip_tool_parsing = False
        self.reset(initial_state=initial_state)
''',
rb'''        # [dspark-dsml-recovery] upstream vllm#52645: terminals that leave a
        # tool state; one is absorbed right after a committed recovered invoke
        # (the optional outer tool_calls closer of an orphan invoke).
        self._tool_exit_terminals: frozenset[str] = frozenset(
            terminal
            for (state, terminal), tr in config.transitions.items()
            if state in self._TOOL_STATES and tr.next_state not in self._TOOL_STATES
        )

        self.skip_tool_parsing = False
        # Optional parser-owned validator used only by provisional tool-call
        # transitions. It intentionally survives reset() so the owning
        # ParserEngine can bind a stable callback once at construction.
        self.recovery_tool_name_validator: Callable[[str], bool] | None = None
        self.reset(initial_state=initial_state)
''',
    ),
    # reset: provisional recovery hold state.
    (
rb'''        self._scanner.reset()
        self._lexer.reset()
        self._reset_args_state()
''',
rb'''        self._scanner.reset()
        self._lexer.reset()
        self._reset_args_state()
        # [dspark-dsml-recovery] provisional orphan-invoke recovery hold.
        self._recovery_hold_active = False
        self._recovery_hold_events: list[SemanticEvent] = []
        self._recovery_hold_raw = ""
        self._recovery_hold_name = ""
        self._recovery_prior_state = self.state
        self._recovery_prior_tool_index = -1
        self._recovery_outer_closer_pending = False
''',
    ),
    # finish: abort an incomplete recovery hold.
    (
rb'''        events.extend(self._process_lex_tokens(self._lexer.flush()))

        if self._args_buffer:
''',
rb'''        events.extend(self._process_lex_tokens(self._lexer.flush()))

        # [dspark-dsml-recovery] restore an incomplete provisional invoke to
        # its original state; the abort also resets the args buffer.
        if self._recovery_hold_active:
            events.extend(self._abort_recovery_hold())

        if self._args_buffer:
''',
    ),
    # finish: close foreign blocks like their base states.
    (
rb'''        elif self.state == ParserState.REASONING:
            events.append(
                SemanticEvent(EventType.REASONING_END, tool_index=self.tool_index)
            )
            self.state = ParserState.CONTENT

        return events
''',
rb'''        elif self.state in (
            ParserState.REASONING,
            ParserState.FOREIGN_REASONING_BLOCK,
        ):
            events.append(
                SemanticEvent(EventType.REASONING_END, tool_index=self.tool_index)
            )
            self.state = ParserState.CONTENT
        elif self.state == ParserState.FOREIGN_BLOCK:
            self.state = ParserState.CONTENT

        return events
''',
    ),
    # _on_terminal: closer absorption, drop/abort rules, skip-mode priority.
    (
rb'''    def _on_terminal(self, terminal: str, value: str) -> list[SemanticEvent]:
        key = (self.state, terminal)
        transition = self.config.transitions.get(key)

        if transition is None:
            if (
                self._has_drops
                and terminal == DROP_TERMINAL
                # Preserve drop tokens when skip_tool_parsing is active so
                # the reasoning pass doesn't silently remove tokens that a
                # later tool-call pass might need to see.
                and not self.skip_tool_parsing
            ):
                return []
            return self._emit_for_state(value)

''',
rb'''    def _on_terminal(self, terminal: str, value: str) -> list[SemanticEvent]:
        # [dspark-dsml-recovery] absorb one optional outer closer right after
        # a committed recovered invoke.
        if (
            self._recovery_outer_closer_pending
            and self.state == ParserState.CONTENT
            and terminal in self._tool_exit_terminals
        ):
            self._recovery_outer_closer_pending = False
            return []

        key = (self.state, terminal)
        transition = self.config.transitions.get(key)

        if transition is None:
            if (
                self._has_drops
                and terminal == DROP_TERMINAL
                # Preserve drop tokens when skip_tool_parsing is active so
                # the reasoning pass doesn't silently remove tokens that a
                # later tool-call pass might need to see.
                and not self.skip_tool_parsing
            ):
                # [dspark-dsml-recovery] DROP_TERMINAL never enters the
                # provisional recovery buffers.
                return []
            # [dspark-dsml-recovery] a foreign terminal while the candidate
            # name is still open cannot belong to a recoverable invoke; roll
            # back, then reprocess the terminal from the restored state.
            # (In TOOL_ARGS, transition-free terminals -- DSML parameter
            # closers -- fall through and stay buffered as arguments.)
            if self._recovery_hold_active and self.state == ParserState.TOOL_NAME:
                events = self._abort_recovery_hold()
                events.extend(self._on_terminal(terminal, value))
                return events
            return self._emit_for_state(value)

        # [dspark-dsml-recovery] provisional transitions take the recovery
        # path even while the reasoning pass runs with skip_tool_parsing.
        if self.skip_tool_parsing and (
            transition.provisional_tool_call or self._recovery_hold_active
        ):
            return self._apply_transition(transition, value)

''',
    ),
    # _emit_for_state: buffer text while a provisional invoke is held.
    (
rb'''    def _emit_for_state(self, text: str) -> list[SemanticEvent]:
        if self.state == ParserState.TOOL_ARGS:
''',
rb'''    def _emit_for_state(self, text: str) -> list[SemanticEvent]:
        # [dspark-dsml-recovery] buffer text while a provisional invoke is
        # held for validation.
        if self._recovery_hold_active:
            return self._hold_recovery_text(text)
        return self._emit_for_state_now(text)

    def _hold_recovery_text(self, text: str) -> list[SemanticEvent]:
        self._recovery_hold_raw += text

        if self.state == ParserState.TOOL_NAME:
            self._recovery_hold_name += text
            # Tool names are expected to be compact. Avoid delaying an entire
            # response if ordinary prose happens to quote an unterminated
            # invoke marker.
            if len(self._recovery_hold_name) > 256 or "\n" in self._recovery_hold_name:
                return self._abort_recovery_hold()

        self._recovery_hold_events.extend(self._emit_for_state_now(text))
        return []

    def _emit_for_state_now(self, text: str) -> list[SemanticEvent]:
        if self.state == ParserState.TOOL_ARGS:
''',
    ),
    # _apply_transition: recovery hold machinery around _run_transition.
    (
rb'''    def _apply_transition(
        self,
        transition: Transition,
        value: str,
    ) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
''',
rb'''    def _apply_transition(
        self,
        transition: Transition,
        value: str,
    ) -> list[SemanticEvent]:
        # [dspark-dsml-recovery] upstream vllm#52645: orphan-invoke recovery.
        if self._recovery_hold_active:
            return self._advance_recovery_hold(transition, value)
        if transition.provisional_tool_call:
            if self.recovery_tool_name_validator is None:
                return self._emit_for_state(value)
            return self._begin_recovery_hold(transition, value)
        if (
            self._recovery_outer_closer_pending
            and self.state == ParserState.CONTENT
            and transition.next_state in self._TOOL_STATES
        ):
            self._recovery_outer_closer_pending = False
        return self._run_transition(transition, value)

    def _begin_recovery_hold(
        self,
        transition: Transition,
        value: str,
    ) -> list[SemanticEvent]:
        prior_state = self.state
        prior_tool_index = self.tool_index
        held_events = self._run_transition(transition, value)
        self._recovery_hold_active = True
        self._recovery_hold_events = held_events
        self._recovery_hold_raw = value
        self._recovery_hold_name = ""
        self._recovery_prior_state = prior_state
        self._recovery_prior_tool_index = prior_tool_index
        return []

    def _advance_recovery_hold(
        self,
        transition: Transition,
        value: str,
    ) -> list[SemanticEvent]:
        self._recovery_hold_raw += value

        if self.state == ParserState.TOOL_NAME:
            validator = self.recovery_tool_name_validator
            if validator is None or not validator(self._recovery_hold_name):
                return self._abort_recovery_hold()
            self._recovery_hold_events.extend(self._run_transition(transition, value))
            return []

        if self.state == ParserState.TOOL_ARGS:
            if not transition.commit_provisional_tool_call:
                return self._abort_recovery_hold()
            if self.skip_tool_parsing:
                # Reasoning-pass commit: hand the entire validated invoke to
                # the tool-call pass as content.
                raw = self._recovery_hold_raw
                prior_state = self._recovery_prior_state
                prior_tool_index = self._recovery_prior_tool_index
                self.state = ParserState.CONTENT
                self.tool_index = prior_tool_index
                self._reset_args_state()
                self._clear_recovery_hold()
                self._recovery_outer_closer_pending = True
                events: list[SemanticEvent] = []
                if prior_state == ParserState.REASONING:
                    events.append(
                        SemanticEvent(
                            EventType.REASONING_END,
                            tool_index=prior_tool_index,
                        )
                    )
                events.append(
                    SemanticEvent(
                        EventType.TEXT_CHUNK,
                        value=raw,
                        tool_index=prior_tool_index,
                    )
                )
                return events
            transition_events = self._run_transition(transition, value)
            self._recovery_hold_events.extend(transition_events)
            events = self._recovery_hold_events
            self._clear_recovery_hold()
            # A recovered invoke started outside a valid outer tool wrapper.
            # Do not leave it in TOOL_BETWEEN: subsequent text is content, and
            # a following bare invoke must enter the provisional path again.
            self.state = ParserState.CONTENT
            self._recovery_outer_closer_pending = True
            return events

        return self._abort_recovery_hold()

    def _abort_recovery_hold(self) -> list[SemanticEvent]:
        raw = self._recovery_hold_raw
        self.state = self._recovery_prior_state
        self.tool_index = self._recovery_prior_tool_index
        self._reset_args_state()
        self._clear_recovery_hold()
        return self._emit_for_state_now(raw)

    def _clear_recovery_hold(self) -> None:
        self._recovery_hold_active = False
        self._recovery_hold_events = []
        self._recovery_hold_raw = ""
        self._recovery_hold_name = ""

    def _run_transition(
        self,
        transition: Transition,
        value: str,
    ) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
''',
    ),
)

REGIONS_PARSER_ENGINE = (
    # finish_streaming: flush deferred reasoning through the strip path.
    (
rb'''    def finish_streaming(self) -> DeltaMessage | None:
        events = self._engine.finish()
        if events or self._deferred_content:
            return self._events_to_delta(events, finished=True)
        return None
''',
rb'''    def finish_streaming(self) -> DeltaMessage | None:
        events = self._engine.finish()
        # [dspark-dsml-recovery] upstream vllm#52645: a rolled-back
        # provisional invoke may sit entirely in deferred reasoning; flush it
        # through the same trailing-whitespace path as parse_delta.
        if events or self._deferred_content or self._deferred_reasoning:
            delta = self._events_to_delta(events, finished=True)
            return self._strip_trailing_reasoning(delta)
        return None
''',
    ),
)

REGIONS_ADAPTERS = (
    # extract_reasoning: sync request tools before the reasoning pass.
    (
rb'''    ) -> tuple[str | None, str | None]:
        with self._skip_tool_parsing():
            return self._parser_engine.extract_reasoning(model_output, request)
''',
rb'''    ) -> tuple[str | None, str | None]:
        # [dspark-dsml-recovery] upstream vllm#52645: sync request tools and
        # tool_choice into the engine before recovery validation.
        self.adjust_request(request)
        with self._skip_tool_parsing():
            return self._parser_engine.extract_reasoning(model_output, request)
''',
    ),
    # adjust_request: mirror tools/tool_choice into the reasoning engine.
    (
rb'''    ) -> ChatCompletionRequest | ResponsesRequest:
        return self._parser_engine.adjust_request(request)

    def has_engine_confirmed_reasoning_end(self) -> bool:
''',
rb'''    ) -> ChatCompletionRequest | ResponsesRequest:
        request = self._parser_engine.adjust_request(request)
        # [dspark-dsml-recovery] upstream vllm#52645: mirror the request's
        # tools and tool_choice into the reasoning-side engine so provisional
        # invoke recovery validates against the live request. Skip mode keeps
        # the reasoning pass from flipping tool-call suppression.
        with self._skip_tool_parsing():
            self._parser_engine._check_skip_tool_parsing(request)
        return request

    def has_engine_confirmed_reasoning_end(self) -> bool:
''',
    ),
)

REGIONS_ABSTRACT = (
    # parse_delta: per-delta request sync for engine-based reasoning.
    (
rb'''        # Reasoning extraction
        if self._in_reasoning_phase(state):
            delta_message = self.extract_reasoning_streaming(
                previous_text=state.previous_text,
                current_text=current_text,
                delta_text=delta_text,
                previous_token_ids=state.previous_token_ids,
                current_token_ids=current_token_ids,
                delta_token_ids=delta_token_ids,
            )
            reasoning_parser = self._reasoning_parser
            if reasoning_parser is not None and reasoning_parser.engine_based_streaming:
''',
rb'''        # Reasoning extraction
        if self._in_reasoning_phase(state):
            # [dspark-dsml-recovery] upstream vllm#52645: sync request tools
            # and tool_choice into the engine-based reasoning parser before
            # each delta so provisional invoke recovery validates against the
            # live request.
            reasoning_parser = self._reasoning_parser
            if reasoning_parser is not None and reasoning_parser.engine_based_streaming:
                reasoning_parser.adjust_request(request)
            delta_message = self.extract_reasoning_streaming(
                previous_text=state.previous_text,
                current_text=current_text,
                delta_text=delta_text,
                previous_token_ids=state.previous_token_ids,
                current_token_ids=current_token_ids,
                delta_token_ids=delta_token_ids,
            )
            if reasoning_parser is not None and reasoning_parser.engine_based_streaming:
''',
    ),
)

REGIONS_DEEPSEEK = (
    # import find_tool_name for the recovery validator.
    (
rb'''from vllm.tool_parsers.utils import find_tool_properties
''',
rb'''from vllm.tool_parsers.utils import find_tool_name, find_tool_properties
''',
    ),
    # foreign V3.2 wrapper token constants.
    (
rb'''DSML_INVOKE_END = f"</{_DSML}invoke>"
DSML_PARAM_CLOSE = f"</{_DSML}parameter>"
''',
rb'''DSML_INVOKE_END = f"</{_DSML}invoke>"
DSML_PARAM_CLOSE = f"</{_DSML}parameter>"
# [dspark-dsml-recovery] upstream vllm#52645: V3.2 wrapper tokens, kept
# verbatim so their inner invokes cannot enter the V4 orphan-recovery path.
DSML_FOREIGN_TOOL_START = f"<{_DSML}function_calls>"
DSML_FOREIGN_TOOL_END = f"</{_DSML}function_calls>"
''',
    ),
    # register the foreign terminals.
    (
rb'''            "INVOKE_END": DSML_INVOKE_END,
            "PARAM_CLOSE": DSML_PARAM_CLOSE,
        },
''',
rb'''            "INVOKE_END": DSML_INVOKE_END,
            "PARAM_CLOSE": DSML_PARAM_CLOSE,
            "FOREIGN_START": DSML_FOREIGN_TOOL_START,
            "FOREIGN_END": DSML_FOREIGN_TOOL_END,
        },
''',
    ),
    # foreign-block transitions + provisional CONTENT/REASONING invokes.
    (
rb'''            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
            (ParserState.TOOL_PREAMBLE, "INVOKE_PREFIX"): Transition(
''',
rb'''            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
            # [dspark-dsml-recovery] upstream vllm#52645.
            # Keep V3.2 wrappers verbatim so their inner invokes cannot enter
            # the V4 orphan-recovery path.
            (ParserState.CONTENT, "FOREIGN_START"): Transition(
                ParserState.FOREIGN_BLOCK,
                (EventType.TEXT_CHUNK,),
            ),
            (ParserState.FOREIGN_BLOCK, "FOREIGN_END"): Transition(
                ParserState.CONTENT,
                (EventType.TEXT_CHUNK,),
            ),
            (ParserState.FOREIGN_BLOCK, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
            (ParserState.REASONING, "FOREIGN_START"): Transition(
                ParserState.FOREIGN_REASONING_BLOCK,
                (EventType.REASONING_CHUNK,),
            ),
            (
                ParserState.FOREIGN_REASONING_BLOCK,
                "FOREIGN_END",
            ): Transition(
                ParserState.REASONING,
                (EventType.REASONING_CHUNK,),
            ),
            (
                ParserState.FOREIGN_REASONING_BLOCK,
                "TOOL_START",
            ): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.REASONING_END,),
            ),
            # DeepSeek V4 can intermittently omit or corrupt the outer
            # tool_calls wrapper while still emitting a complete
            # invoke. Hold this recovery path until the function name is
            # complete and verify that the request actually declared it.
            (ParserState.REASONING, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
                provisional_tool_call=True,
            ),
            (ParserState.CONTENT, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
                provisional_tool_call=True,
            ),
            (ParserState.TOOL_PREAMBLE, "INVOKE_PREFIX"): Transition(
''',
    ),
    # INVOKE_END commits a provisional recovery candidate.
    (
rb'''            (ParserState.TOOL_ARGS, "INVOKE_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
''',
rb'''            (ParserState.TOOL_ARGS, "INVOKE_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
                commit_provisional_tool_call=True,
            ),
''',
    ),
    # content events for the foreign states.
    (
rb'''        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.REASONING: EventType.REASONING_CHUNK,
            ParserState.TOOL_NAME: EventType.TOOL_NAME,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
''',
rb'''        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.REASONING: EventType.REASONING_CHUNK,
            ParserState.FOREIGN_BLOCK: EventType.TEXT_CHUNK,
            ParserState.FOREIGN_REASONING_BLOCK: EventType.REASONING_CHUNK,
            ParserState.TOOL_NAME: EventType.TOOL_NAME,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
''',
    ),
    # bind the request-scoped recovery tool-name validator.
    (
rb'''        self._arg_converter = self._convert_args

    def _convert_args(self, raw_args: str, partial: bool) -> str:
''',
rb'''        self._arg_converter = self._convert_args
        # [dspark-dsml-recovery] upstream vllm#52645: recovery validates the
        # completed tool name against the live request, not the constructor.
        self._recovery_request_tools: list[Tool] = list(tools or [])
        self._recovery_suppressed = False
        self._engine.recovery_tool_name_validator = self._can_recover_tool_name

    def _check_skip_tool_parsing(self, request) -> None:
        super()._check_skip_tool_parsing(request)
        self._recovery_request_tools = list(getattr(request, "tools", None) or [])
        self._recovery_suppressed = getattr(request, "tool_choice", None) == "none"

    def _can_recover_tool_name(self, name: str) -> bool:
        return bool(
            name
            and self._recovery_request_tools
            and not self._recovery_suppressed
            and find_tool_name(self._recovery_request_tools, name)
        )

    def _convert_args(self, raw_args: str, partial: bool) -> str:
''',
    ),
)


class Spec(NamedTuple):
    """One patch target: its regions and whole-file identity pins."""

    label: str
    rel_path: str
    regions: tuple[tuple[bytes, bytes], ...]
    stock_sha256: str
    stock_size: int
    patched_sha256: str
    patched_size: int


SPECS = (
    Spec(
        label="parser_engine_config.py",
        rel_path="parser/engine/parser_engine_config.py",
        regions=REGIONS_CONFIG,
        stock_sha256="0854bd50b239b3b5286f56e9254851f224822bf0e942c99f1fe3dabf3c2035a7",
        stock_size=3_557,
        patched_sha256="76ed8f12ddf889bc45316d7d57996522efb1f594129e8814b520191f46c08d06",
        patched_size=4_311,
    ),
    Spec(
        label="streaming_parser_engine.py",
        rel_path="parser/engine/streaming_parser_engine.py",
        regions=REGIONS_ENGINE,
        stock_sha256="4ac9135e12f286d32f5d8725630b49a54b68d902815ad598923f7130d0899e0b",
        stock_size=15_938,
        patched_sha256="cd7d8778f572c5ce883adb54db740615ef633c3b0612dde3af5659e9b3fe215b",
        patched_size=24_278,
    ),
    Spec(
        label="parser_engine.py",
        rel_path="parser/engine/parser_engine.py",
        regions=REGIONS_PARSER_ENGINE,
        stock_sha256="886bf6293b6b4cd082882e883c4c3d1ff16597500ed741f871a4ae67826db178",
        stock_size=39_873,
        patched_sha256="f8f403ad8e5908478dfe5b22adee83bb37fba75d59e5aab9fd815e00b05ff1e7",
        patched_size=40_173,
    ),
    Spec(
        label="adapters.py",
        rel_path="parser/engine/adapters.py",
        regions=REGIONS_ADAPTERS,
        stock_sha256="dc1c1317dbfb298e54b8d94ca0e66d2b0cb1e481c35cdcc60a815284bd8a6ef7",
        stock_size=7_551,
        patched_sha256="9d7437346d967aa0c095b9bbb80ec9374544fc1ff546760138b074ef82efa3ad",
        patched_size=8_160,
    ),
    Spec(
        label="abstract_parser.py",
        rel_path="parser/abstract_parser.py",
        regions=REGIONS_ABSTRACT,
        stock_sha256="fd4eb7a64ea97f59cfaec2b08c359fb460daa06e4ac5b42906b6877b34360859",
        stock_size=35_471,
        patched_sha256="e11c1b7834302639f7205ca7b1dc55e2dbe98d90e1b0fb7e97cc3cd4903ac365",
        patched_size=35_876,
    ),
    Spec(
        label="deepseek_v4.py",
        rel_path="parser/deepseek_v4.py",
        regions=REGIONS_DEEPSEEK,
        stock_sha256="97d7cd3c2affb37c29948aba321454090341294756d46fde2bf9f3623c30ee6f",
        stock_size=7_987,
        patched_sha256="2cc89a1b05e2f55a62ac7502e06d619f346fad4f15284c275d65d2fd4ff31d54",
        patched_size=11_435,
    ),
)


class HotfixError(RuntimeError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def transform(spec: Spec, stock: bytes) -> bytes:
    """Stock bytes -> patched bytes; refuses anything but exactly one site each."""
    if MARK.encode() in stock:
        raise HotfixError(f"{spec.label}: target already carries the dsml-recovery mark")
    patched = stock
    for old, new in spec.regions:
        if patched.count(old) != 1:
            raise HotfixError(f"{spec.label}: region not found exactly once")
        patched = patched.replace(old, new, 1)
    compile(patched, spec.label, "exec")
    if _sha256(patched) != spec.patched_sha256 or len(patched) != spec.patched_size:
        raise HotfixError(
            f"{spec.label}: transformed bytes do not match the pinned patched identity"
        )
    return patched


def _vllm_version(provider=importlib.metadata.version) -> str:
    try:
        version = provider("vllm")
    except importlib.metadata.PackageNotFoundError as error:
        raise HotfixError("vllm is not installed") from error
    if version != EXPECTED_VLLM_VERSION:
        raise HotfixError(
            f"unsupported vllm version {version!r}; expected {EXPECTED_VLLM_VERSION!r}"
        )
    return version


def inspect(
    spec: Spec, target: Path, *, provider=importlib.metadata.version
) -> tuple[str, bytes]:
    _vllm_version(provider)
    try:
        st = target.lstat()
    except FileNotFoundError:
        raise HotfixError(f"{spec.label}: target is missing")
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise HotfixError(f"{spec.label}: target is not a regular file")
    data = target.read_bytes()
    digest = _sha256(data)
    if digest == spec.patched_sha256 and len(data) == spec.patched_size:
        return "patched", data
    if digest == spec.stock_sha256 and len(data) == spec.stock_size:
        return "stock", data
    raise HotfixError(
        f"{spec.label}: unsupported target bytes sha256={digest} size={len(data)}; "
        "expected the pinned stock or patched identity"
    )


def _publish(target: Path, data: bytes) -> None:
    fd, tmp_name = tempfile.mkstemp(
        prefix=".dspark-dsml-recovery-", dir=str(target.parent)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def apply(root: Path, *, provider=importlib.metadata.version) -> dict[str, str]:
    """Preflight every target, then write only the ones still stock.

    If publishing or verifying any file fails after earlier files were
    written, the files written by this run are restored to their stock bytes
    before the error propagates, so a partially patched tree is never left
    behind.
    """
    states = {
        spec.label: inspect(spec, root / spec.rel_path, provider=provider)
        for spec in SPECS
    }
    outcomes: dict[str, str] = {}
    written: list[tuple[Path, bytes]] = []
    try:
        for spec in SPECS:
            state, data = states[spec.label]
            if state == "patched":
                outcomes[spec.label] = "already-patched"
                continue
            target = root / spec.rel_path
            _publish(target, transform(spec, data))
            written.append((target, data))
            verify_state, _ = inspect(spec, target, provider=provider)
            if verify_state != "patched":
                raise HotfixError(f"{spec.label}: post-apply verification failed")
            outcomes[spec.label] = "applied"
    except BaseException:
        for target, stock_data in reversed(written):
            _publish(target, stock_data)
        raise
    return outcomes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify compatibility only")
    parser.add_argument("--status", action="store_true", help="print the target states")
    parser.add_argument("--root", type=Path, default=PRODUCTION_ROOT)
    args = parser.parse_args(argv)
    try:
        if args.check or args.status:
            report = ", ".join(
                f"{spec.label}={inspect(spec, args.root / spec.rel_path)[0]}"
                for spec in SPECS
            )
        else:
            outcomes = apply(args.root)
            report = ", ".join(
                f"{spec.label}={outcomes[spec.label]}" for spec in SPECS
            )
        print(f"dspark-dsml-recovery: {report} (root={args.root})")
        return 0
    except HotfixError as error:
        print(f"dspark-dsml-recovery: FAIL-CLOSED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
