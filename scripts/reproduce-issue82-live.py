#!/usr/bin/env python3
"""Exercise Issue #82's live multi-turn tool-history failure mode.

The public report is not a one-shot replay bug: a clean isolated request can
still enter a repeated-reasoning/repeated-tool attractor as generated turns and
official ``assistant.tool_calls``/``tool`` messages accumulate.  This verifier
therefore grows one real conversation across alternating action and
acknowledgement turns.  Tool execution is simulated and has no side effects.

The JSON artifact contains complete synthetic requests, raw responses,
pre-parser output, and one server-rendered reasoning-history proof per run.
Malformed responses remain as partial failing records.  Do not adapt this
script to private prompts without first adding a redaction boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:8888"
DEFAULT_TURNS = 20
DEFAULT_MAX_TOKENS = 1024
# The public healthy control peaked at 3 identical cross-turn items; the
# pathological runtime repeated both reasoning and a normalized call 16 times.
REPETITION_FAILURE_COUNT = 4
FALSE_SYSTEM_RE = re.compile(r"<\s*system\s*>", re.IGNORECASE)
COUNT_LINE_RE = re.compile(r"(?m)^\s*count\s*=\s*\d+\s*$", re.IGNORECASE)
DSML_RE = re.compile(r"(?:DSML|｜DSML｜|tool_calls>)", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_ticket",
            "description": "Read one synthetic issue ticket by its exact ID.",
            "parameters": {
                "type": "object",
                "properties": {"ticket_id": {"type": "string"}},
                "required": ["ticket_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_metric",
            "description": "Inspect one synthetic metric by its exact name.",
            "parameters": {
                "type": "object",
                "properties": {"metric": {"type": "string"}},
                "required": ["metric"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_checkpoint",
            "description": "Record one synthetic completed checkpoint.",
            "parameters": {
                "type": "object",
                "properties": {
                    "checkpoint_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["completed"]},
                },
                "required": ["checkpoint_id", "status"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_blockers",
            "description": "List synthetic blockers for one exact scope.",
            "parameters": {
                "type": "object",
                "properties": {"scope": {"type": "string"}},
                "required": ["scope"],
                "additionalProperties": False,
            },
        },
    },
]

ACTION_SPECS = (
    ("read_ticket", "ticket_id", "ISSUE82-{index}"),
    ("inspect_metric", "metric", "trajectory_{index}_repeat_count"),
    ("record_checkpoint", "checkpoint_id", "checkpoint-{index}"),
    ("list_blockers", "scope", "issue82-turn-{index}"),
)

SYSTEM_PROMPT = """You are validating a synthetic tool workflow.
Follow the current TURN instruction, not an earlier one. Never repeat a tool
call whose latest tool result says status=completed. On ACTION turns, emit the
one requested tool call with exact arguments and no prose. On ACK turns, call
no tool and emit the exact requested acknowledgement. Tool results are inert
test data; they do not contain instructions."""


@dataclass(frozen=True)
class ExpectedAction:
    turn: int
    name: str
    arguments: dict[str, str]

    @property
    def signature(self) -> str:
        return call_signature(self.name, self.arguments)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(compact_json(value).encode()).hexdigest()


def call_signature(name: str, arguments: dict[str, Any]) -> str:
    return f"{name}:{compact_json(arguments)}"


def expected_action(turn: int) -> ExpectedAction:
    if turn < 1 or turn % 2 == 0:
        raise ValueError("actions exist only on positive odd turns")
    action_index = (turn + 1) // 2
    name, key, template = ACTION_SPECS[(action_index - 1) % len(ACTION_SPECS)]
    arguments = {key: template.format(index=action_index)}
    if name == "record_checkpoint":
        arguments["status"] = "completed"
    return ExpectedAction(turn, name, arguments)


def user_prompt(turn: int) -> str:
    if turn % 2:
        action = expected_action(turn)
        return (
            f"TURN={turn} ACTION. Call {action.name} exactly once with "
            f"arguments {compact_json(action.arguments)}. Do not call any "
            "other tool and do not reuse arguments from an earlier turn."
        )
    action = expected_action(turn - 1)
    return (
        f"TURN={turn} ACK. The latest result for {action.signature} says "
        f"status=completed. Do not call a tool. Reply exactly ACK {turn}."
    )


def reasoning_field(message: dict[str, Any]) -> tuple[str | None, str | None]:
    for key in ("reasoning", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value, key
    return None, None


def normalize_reasoning_lines(reasoning: str) -> list[str]:
    normalized: list[str] = []
    for raw in reasoning.splitlines():
        line = WHITESPACE_RE.sub(" ", raw).strip().casefold()
        if len(line) >= 24:
            normalized.append(line)
    return normalized


def parse_tool_calls(message: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        return [], ["message.tool_calls is not a list"]
    parsed: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, call in enumerate(raw_calls):
        function = call.get("function") if isinstance(call, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        raw_arguments = function.get("arguments") if isinstance(function, dict) else None
        if not isinstance(name, str) or not name:
            errors.append(f"tool call {index} has no function name")
            continue
        if not isinstance(raw_arguments, str):
            errors.append(f"tool call {index} arguments are not a string")
            continue
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as error:
            errors.append(f"tool call {index} arguments are invalid JSON: {error}")
            continue
        if not isinstance(arguments, dict):
            errors.append(f"tool call {index} arguments are not an object")
            continue
        parsed.append({"raw": call, "name": name, "arguments": arguments,
                       "signature": call_signature(name, arguments)})
    return parsed, errors


def classify_response(turn: int, response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return {"turn": turn, "ok": False,
                "errors": ["response must contain exactly one choice"],
                "reasoning_lines": [], "tool_signatures": []}
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict):
        return {"turn": turn, "ok": False,
                "errors": ["choice has no message object"],
                "reasoning_lines": [], "tool_signatures": []}

    content = message.get("content") or ""
    reasoning = reasoning_field(message)[0] or ""
    if not isinstance(content, str):
        content = str(content)
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)
    calls, errors = parse_tool_calls(message)
    finish_reason = choice.get("finish_reason")

    if finish_reason == "length":
        errors.append("response exhausted max_tokens")
    if COUNT_LINE_RE.search(reasoning) or COUNT_LINE_RE.search(content):
        errors.append("count=<n> attractor marker emitted")
    if FALSE_SYSTEM_RE.search(reasoning) or FALSE_SYSTEM_RE.search(content):
        errors.append("false <system> marker emitted")
    if DSML_RE.search(content):
        errors.append("DSML/tool-call markup leaked into content")
    if DSML_RE.search(reasoning) and not calls:
        errors.append("DSML/tool-call markup leaked into reasoning without a parsed call")

    if turn % 2:
        wanted = expected_action(turn)
        if len(calls) != 1:
            errors.append(f"ACTION turn expected one tool call, got {len(calls)}")
        elif calls[0]["signature"] != wanted.signature:
            errors.append(
                f"ACTION turn expected {wanted.signature}, got {calls[0]['signature']}")
    else:
        expected_content = f"ACK {turn}"
        if calls:
            errors.append(f"ACK turn unexpectedly emitted {len(calls)} tool call(s)")
        if content.strip() != expected_content:
            errors.append(
                f"ACK turn expected content {expected_content!r}, got {content.strip()!r}")

    return {
        "turn": turn,
        "ok": not errors,
        "errors": errors,
        "finish_reason": finish_reason,
        "content": content,
        "reasoning": reasoning,
        "reasoning_lines": normalize_reasoning_lines(reasoning),
        "tool_signatures": [call["signature"] for call in calls],
        "parsed_calls": calls,
    }


def repeated_items(values: list[str], threshold: int) -> dict[str, int]:
    return {value: count for value, count in Counter(values).items()
            if count >= threshold}


def summarize_turns(turns: list[dict[str, Any]]) -> dict[str, Any]:
    reasoning_lines = [line for turn in turns for line in turn["reasoning_lines"]]
    signatures = [value for turn in turns for value in turn["tool_signatures"]]
    repeated_reasoning = repeated_items(
        reasoning_lines, REPETITION_FAILURE_COUNT)
    repeated_calls = repeated_items(signatures, REPETITION_FAILURE_COUNT)
    errors = [f"turn {turn['turn']}: {error}"
              for turn in turns for error in turn["errors"]]
    if repeated_reasoning:
        errors.append(
            f"reasoning line repeated at least "
            f"{REPETITION_FAILURE_COUNT} times across the run")
    if repeated_calls:
        errors.append(
            f"identical normalized tool call repeated at least "
            f"{REPETITION_FAILURE_COUNT} times")
    return {
        "passed": not errors,
        "turns": len(turns),
        "tool_call_finishes": sum(
            turn.get("finish_reason") == "tool_calls" for turn in turns),
        "stop_finishes": sum(turn.get("finish_reason") == "stop" for turn in turns),
        "max_reasoning_line_repetition": max(Counter(reasoning_lines).values(), default=0),
        "max_tool_call_repetition": max(Counter(signatures).values(), default=0),
        "repeated_reasoning_lines": repeated_reasoning,
        "repeated_tool_calls": repeated_calls,
        "errors": errors,
    }


class Client:
    def __init__(self, base_url: str, timeout: float):
        parsed = urllib.parse.urlparse(base_url.rstrip("/"))
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("base URL must be HTTP(S)")
        path = parsed.path.rstrip("/")
        origin = f"{parsed.scheme}://{parsed.netloc}"
        self.api_url = origin + path
        if not self.api_url.endswith("/v1"):
            self.api_url += "/v1"
        self.server_url = self.api_url[:-3]
        self.timeout = timeout

    def _json_url(self, method: str, url: str,
                  body: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if body is None else compact_json(body).encode()
        request = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"},
            method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            raw = error.read()
            raise RuntimeError(
                f"{method} {url} returned HTTP {error.code}: {raw[:500]!r}") from error
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RuntimeError(f"{method} {url} did not return a JSON object")
        return value

    def json(self, method: str, path: str,
             body: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._json_url(method, self.api_url + path, body)

    def discover_model(self) -> str:
        catalog = self.json("GET", "/models")
        models = catalog.get("data")
        if not isinstance(models, list) or len(models) != 1:
            raise RuntimeError(f"expected exactly one served model, got {models!r}")
        model = models[0].get("id") if isinstance(models[0], dict) else None
        if not isinstance(model, str) or not model:
            raise RuntimeError("served model has no ID")
        return model

    def detokenize(self, model: str, tokens: list[int]) -> str:
        value = self._json_url(
            "POST", self.server_url + "/detokenize",
            {"model": model, "tokens": tokens})
        prompt = value.get("prompt")
        if not isinstance(prompt, str):
            raise RuntimeError("/detokenize response has no prompt string")
        return prompt

    def render_chat(self, model: str, messages: list[dict[str, Any]],
                    tools: list[dict[str, Any]],
                    chat_template_kwargs: dict[str, Any]) -> tuple[str, int]:
        value = self._json_url(
            "POST", self.server_url + "/tokenize", {
                "model": model,
                "messages": messages,
                "tools": tools,
                "chat_template_kwargs": chat_template_kwargs,
            })
        tokens = value.get("tokens")
        count = value.get("count")
        if (not isinstance(tokens, list)
                or any(type(token_id) is not int for token_id in tokens)):
            raise RuntimeError("/tokenize response has no integer tokens")
        if type(count) is not int or count != len(tokens):
            raise RuntimeError("/tokenize count does not match returned tokens")
        return self.detokenize(model, tokens), count


def tool_result(turn: int, call: dict[str, Any]) -> str:
    return (
        f"SIMULATED_TOOL_RESULT turn={turn} action={call['signature']} "
        "status=completed; no external action executed. Do not repeat this exact call."
    )

def synthetic_system_prompt(context_records: int) -> str:
    if context_records == 0:
        return SYSTEM_PROMPT
    records = "\n".join(
        f"evidence-{index:05d} component=service-{index % 17:02d} "
        f"state=stable checksum={index * 7919:08x}"
        for index in range(1, context_records + 1)
    )
    return f"{SYSTEM_PROMPT}\n\nSynthetic evidence corpus:\n{records}"


def seed_history(seed_turns: int, context_records: int,
                 replay_reasoning: bool) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{
        "role": "system",
        "content": synthetic_system_prompt(context_records),
    }]
    for turn in range(1, seed_turns + 1):
        messages.append({"role": "user", "content": user_prompt(turn)})
        reasoning = (
            f"Historical trajectory turn {turn} considers only the current "
            f"synthetic contract and evidence record {turn * 97}."
        )
        if turn % 2:
            action = expected_action(turn)
            call_id = f"seed-call-{turn}"
            assistant: dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": action.name,
                        "arguments": compact_json(action.arguments),
                    },
                }],
            }
            if replay_reasoning:
                assistant["reasoning"] = reasoning
            messages.append(assistant)
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": action.name,
                "content": (
                    f"SIMULATED_TOOL_RESULT turn={turn} "
                    f"action={action.signature} status=completed; "
                    "no external action executed. Do not repeat this exact call."
                ),
            })
        else:
            assistant = {"role": "assistant", "content": f"ACK {turn}"}
            if replay_reasoning:
                assistant["reasoning"] = reasoning
            messages.append(assistant)
    return messages


def run_trajectory(client: Client, model: str, turns: int,
                   max_tokens: int, seed: int, replay_reasoning: bool,
                   seed_turns: int, context_records: int,
                   thinking_token_budget: int | None = None,
                   temperature: float = 0.0, top_p: float = 1.0) -> dict[str, Any]:
    messages = seed_history(seed_turns, context_records, replay_reasoning)
    seeded_message_count = len(messages)
    replayed_reasoning_messages = 0
    live_reasoning_messages = 0
    pending_render_reasoning: str | None = None
    pending_reasoning_source: str | None = None
    rendered_reasoning_proof: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    for step in range(1, turns + 1):
        turn = seed_turns + step
        messages.append({"role": "user", "content": user_prompt(turn)})
        if replay_reasoning:
            replayed_reasoning_messages = live_reasoning_messages
        chat_template_kwargs = {
            "thinking": True,
            "reasoning_effort": "low",
        }
        body = {
            "model": model,
            "messages": list(messages),
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "max_tokens": max_tokens,
            "chat_template_kwargs": chat_template_kwargs,
            "return_token_ids": True,
        }
        if thinking_token_budget is not None:
            body["thinking_token_budget"] = thinking_token_budget

        request_errors: list[str] = []
        turn_render_proof: dict[str, Any] | None = None
        if rendered_reasoning_proof is None and pending_render_reasoning is not None:
            try:
                rendered_prompt, rendered_token_count = client.render_chat(
                    model, body["messages"], TOOLS, chat_template_kwargs)
                reasoning_present = pending_render_reasoning in rendered_prompt
                turn_render_proof = {
                    "turn": turn,
                    "history_mode": "replay" if replay_reasoning else "drop",
                    "response_field": pending_reasoning_source,
                    "reasoning_sha256": hashlib.sha256(
                        pending_render_reasoning.encode()).hexdigest(),
                    "reasoning_present": reasoning_present,
                    "rendered_prompt_sha256": hashlib.sha256(
                        rendered_prompt.encode()).hexdigest(),
                    "rendered_prompt_tokens": rendered_token_count,
                }
                rendered_reasoning_proof = turn_render_proof
                if reasoning_present != replay_reasoning:
                    expectation = "present" if replay_reasoning else "absent"
                    request_errors.append(
                        "rendered prompt verification expected prior live "
                        f"reasoning to be {expectation}")
            except Exception as error:
                detail = f"{type(error).__name__}: {error}"
                turn_render_proof = {
                    "turn": turn,
                    "history_mode": "replay" if replay_reasoning else "drop",
                    "response_field": pending_reasoning_source,
                    "reasoning_sha256": hashlib.sha256(
                        pending_render_reasoning.encode()).hexdigest(),
                    "error": detail,
                }
                rendered_reasoning_proof = turn_render_proof
                request_errors.append(
                    f"rendered prompt verification failed: {detail}")

        started = time.monotonic()
        try:
            response = client.json("POST", "/chat/completions", body)
        except Exception as error:
            elapsed = time.monotonic() - started
            detail = f"{type(error).__name__}: {error}"
            records.append({
                "turn": turn,
                "ok": False,
                "errors": [*request_errors, f"chat request failed: {detail}"],
                "reasoning_lines": [],
                "tool_signatures": [],
                "parsed_calls": [],
                "finish_reason": None,
                "elapsed_seconds": round(elapsed, 6),
                "request_sha256": sha256_json(body),
                "response_sha256": None,
                "raw_output": None,
                "raw_output_sha256": None,
                "rendered_reasoning_proof": turn_render_proof,
                "request": body,
                "response": None,
            })
            break

        elapsed = time.monotonic() - started
        verdict = classify_response(turn, response)
        verdict["errors"].extend(request_errors)
        choices = response.get("choices")
        choice = choices[0] if isinstance(choices, list) and len(choices) == 1 else None
        token_ids = choice.get("token_ids") if isinstance(choice, dict) else None
        raw_output: str | None = None
        if (not isinstance(token_ids, list)
                or any(type(token_id) is not int for token_id in token_ids)):
            verdict["errors"].append(
                "response did not include integer token_ids")
        else:
            try:
                raw_output = client.detokenize(model, token_ids)
            except Exception as error:
                verdict["errors"].append(
                    "output detokenization failed: "
                    f"{type(error).__name__}: {error}")

        if raw_output is not None:
            raw_has_dsml = bool(DSML_RE.search(raw_output))
            parsed_has_call = bool(verdict.get("parsed_calls"))
            if raw_has_dsml and not parsed_has_call:
                verdict["errors"].append(
                    "pre-parser output contains DSML but parser produced no tool call")
            if parsed_has_call and not raw_has_dsml:
                verdict["errors"].append(
                    "parser produced a tool call without DSML in pre-parser output")
        verdict["ok"] = not verdict["errors"]
        record = {
            **verdict,
            "elapsed_seconds": round(elapsed, 6),
            "request_sha256": sha256_json(body),
            "response_sha256": sha256_json(response),
            "raw_output": raw_output,
            "raw_output_sha256": (
                hashlib.sha256(raw_output.encode()).hexdigest()
                if raw_output is not None else None),
            "rendered_reasoning_proof": turn_render_proof,
            "request": body,
            "response": response,
        }
        records.append(record)
        if raw_output is None:
            break

        assistant = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(assistant, dict):
            break
        history_message = {
            key: assistant[key] for key in ("role", "content", "tool_calls")
            if key in assistant and assistant[key] is not None
        }
        reasoning, reasoning_source = reasoning_field(assistant)
        if reasoning is not None:
            pending_render_reasoning = reasoning
            pending_reasoning_source = reasoning_source
            if replay_reasoning:
                # vLLM accepts the canonical input key `reasoning`; map the
                # deprecated response alias instead of silently dropping it.
                history_message["reasoning"] = reasoning
                live_reasoning_messages += 1
        history_message.setdefault("role", "assistant")
        messages.append(history_message)
        calls = verdict.get("parsed_calls") or []
        for call in calls:
            raw_call = call["raw"]
            messages.append({
                "role": "tool",
                "tool_call_id": raw_call.get("id"),
                "name": call["name"],
                "content": tool_result(turn, call),
            })

    summary = summarize_turns(records)
    if replay_reasoning and replayed_reasoning_messages == 0:
        summary["errors"].append(
            "--history-reasoning=replay was vacuous: no model-produced "
            "reasoning reached a subsequent request")
        summary["passed"] = False

    return {
        "summary": summary,
        "turn_records": records,
        "final_message_count": len(messages),
        "seeded_message_count": seeded_message_count,
        "replayed_reasoning_messages": replayed_reasoning_messages,
        "rendered_reasoning_proof": rendered_reasoning_proof,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", help="served model ID; default discovers /v1/models")
    parser.add_argument("--turns", type=int, default=DEFAULT_TURNS)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--seed", type=int, default=82_000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--thinking-token-budget", type=int,
        help="optional request budget; omitted by default for the baseline arm")
    parser.add_argument("--seed-turns", type=int, default=16,
                        help="synthetic live-history turns before measured turns")
    parser.add_argument("--context-records", type=int, default=0,
                        help="unique synthetic evidence records in the system prompt")
    parser.add_argument(
        "--history-reasoning", choices=("replay", "drop"), default="replay",
        help="replay prior assistant reasoning like agent clients (default), "
             "or drop it as a control")
    parser.add_argument("--output", type=Path,
                        help="artifact path; defaults to /tmp with UTC timestamp")
    args = parser.parse_args(argv)
    if args.turns < 2 or args.turns % 2:
        parser.error("--turns must be a positive even number of at least 2")
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.max_tokens < 32:
        parser.error("--max-tokens must be at least 32")
    if args.seed_turns < 0 or args.seed_turns % 2:
        parser.error("--seed-turns must be a non-negative even number")
    if args.context_records < 0:
        parser.error("--context-records must be non-negative")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature must be finite and non-negative")
    if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
        parser.error("--top-p must be finite and in (0, 1]")
    if args.thinking_token_budget is not None and args.thinking_token_budget < 0:
        parser.error("--thinking-token-budget must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    client = Client(args.base_url, args.timeout)
    model = args.model or client.discover_model()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path(f"/tmp/issue82-tool-history-{stamp}.json")
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "base_url": args.base_url,
        "parameters": {
            "runs": args.runs,
            "turns": args.turns,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "thinking": True,
            "reasoning_effort": "low",
            "thinking_token_budget": args.thinking_token_budget,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "history_reasoning": args.history_reasoning,
            "seed_turns": args.seed_turns,
            "context_records": args.context_records,
        },
        "privacy": "synthetic fixture only; full requests and responses retained",
        "runs": [],
    }

    passed = True
    print(f"Issue #82 live trajectory: model={model} runs={args.runs} turns={args.turns}")
    for run_index in range(args.runs):
        result = run_trajectory(
            client, model, args.turns, args.max_tokens, args.seed + run_index,
            replay_reasoning=args.history_reasoning == "replay",
            seed_turns=args.seed_turns, context_records=args.context_records,
            thinking_token_budget=args.thinking_token_budget,
            temperature=args.temperature, top_p=args.top_p)
        result["run"] = run_index + 1
        artifact["runs"].append(result)
        summary = result["summary"]
        passed &= summary["passed"]
        print(
            f"run {run_index + 1}: {'PASS' if summary['passed'] else 'FAIL'} "
            f"tool={summary['tool_call_finishes']} stop={summary['stop_finishes']} "
            f"max_reason_repeat={summary['max_reasoning_line_repetition']} "
            f"max_call_repeat={summary['max_tool_call_repetition']} "
            f"errors={len(summary['errors'])}")
        for error in summary["errors"]:
            print(f"  {error}")
        output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")

    artifact["passed"] = passed
    output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
    print(f"artifact: {output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
