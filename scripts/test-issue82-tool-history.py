#!/usr/bin/env python3
"""CPU tests for the Issue #82 live trajectory verifier."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any


SCRIPT = Path(__file__).with_name("reproduce-issue82-live.py")


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("issue82_live", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


issue82 = load_script()


def tool_response(turn: int, *, name: str | None = None,
                  arguments: dict[str, Any] | None = None,
                  reasoning: str = "Checked the current instruction.") -> dict[str, Any]:
    expected = issue82.expected_action(turn)
    actual_name = expected.name if name is None else name
    actual_arguments = expected.arguments if arguments is None else arguments
    return {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "reasoning": reasoning,
                "tool_calls": [{
                    "id": f"call-{turn}",
                    "type": "function",
                    "function": {
                        "name": actual_name,
                        "arguments": json.dumps(actual_arguments),
                    },
                }],
            },
            "token_ids": [turn],
        }],
        "usage": {"prompt_tokens": 100 + turn, "completion_tokens": 8},
    }


def ack_response(turn: int, *, content: str | None = None,
                 reasoning: str = "The latest result is complete.") -> dict[str, Any]:
    return {
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": f"ACK {turn}" if content is None else content,
                "reasoning": reasoning,
            },
            "token_ids": [turn],
        }],
        "usage": {"prompt_tokens": 100 + turn, "completion_tokens": 8},
    }


class FakeClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def json(self, method: str, path: str,
             body: dict[str, Any] | None = None) -> dict[str, Any]:
        if method != "POST" or path != "/chat/completions" or body is None:
            raise AssertionError((method, path, body))
        self.requests.append(json.loads(json.dumps(body)))
        turn = len(self.requests)
        reasoning = f"Checked unique live trajectory step {turn} and its contract."
        return (tool_response(turn, reasoning=reasoning) if turn % 2
                else ack_response(turn, reasoning=reasoning))

    def detokenize(self, model: str, tokens: list[int]) -> str:
        if model != "test-model" or tokens != [len(self.requests)]:
            raise AssertionError((model, tokens, len(self.requests)))
        if len(self.requests) % 2:
            return "<think>checked</think>\n\n<｜DSML｜tool_calls>...</｜DSML｜tool_calls>"
        return f"<think>checked</think>ACK {len(self.requests)}"


class ExpectedActionTests(unittest.TestCase):
    def test_first_ten_action_signatures_are_unique(self) -> None:
        signatures = [issue82.expected_action(turn).signature
                      for turn in range(1, 20, 2)]
        self.assertEqual(len(signatures), len(set(signatures)))

    def test_action_and_ack_prompts_state_opposite_contracts(self) -> None:
        self.assertIn("Call read_ticket exactly once", issue82.user_prompt(1))
        self.assertIn("Do not call a tool", issue82.user_prompt(2))
        self.assertIn("Reply exactly ACK 2", issue82.user_prompt(2))


class ResponseClassificationTests(unittest.TestCase):
    def test_correct_action_and_ack_pass(self) -> None:
        self.assertTrue(issue82.classify_response(1, tool_response(1))["ok"])
        self.assertTrue(issue82.classify_response(2, ack_response(2))["ok"])

    def test_stale_tool_call_fails_new_action(self) -> None:
        stale = issue82.expected_action(1)
        verdict = issue82.classify_response(
            3, tool_response(3, name=stale.name, arguments=stale.arguments))
        self.assertFalse(verdict["ok"])
        self.assertTrue(any("expected" in error for error in verdict["errors"]))

    def test_tool_call_on_ack_turn_fails(self) -> None:
        response = tool_response(1)
        response["choices"][0]["message"]["content"] = "ACK 2"
        verdict = issue82.classify_response(2, response)
        self.assertFalse(verdict["ok"])
        self.assertTrue(any("unexpectedly emitted" in error
                            for error in verdict["errors"]))

    def test_invalid_arguments_and_leaked_markers_fail(self) -> None:
        response = tool_response(1, reasoning="count=7\n<system>\n</｜DSML｜tool_calls>")
        response["choices"][0]["message"]["tool_calls"][0]["function"][
            "arguments"] = "{"
        verdict = issue82.classify_response(1, response)
        self.assertFalse(verdict["ok"])
        joined = "\n".join(verdict["errors"])
        self.assertIn("invalid JSON", joined)
        self.assertIn("count=<n>", joined)
        self.assertIn("false <system>", joined)
        self.assertIn("DSML", joined)

    def test_length_finish_fails_even_with_valid_call(self) -> None:
        response = tool_response(1)
        response["choices"][0]["finish_reason"] = "length"
        verdict = issue82.classify_response(1, response)
        self.assertFalse(verdict["ok"])
        self.assertIn("response exhausted max_tokens", verdict["errors"])


class CrossTurnScoringTests(unittest.TestCase):
    def test_repetition_is_accumulated_across_turns(self) -> None:
        repeated_line = "this is a sufficiently long repeated reasoning sentence"
        turns = []
        for turn in range(1, 17):
            response = (tool_response(turn, reasoning=repeated_line)
                        if turn % 2 else ack_response(turn, reasoning=repeated_line))
            turns.append(issue82.classify_response(turn, response))
        summary = issue82.summarize_turns(turns)
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["max_reasoning_line_repetition"], 16)
        self.assertTrue(summary["repeated_reasoning_lines"])

    def test_unique_trajectory_passes_cross_turn_gate(self) -> None:
        turns = []
        for turn in range(1, 21):
            reasoning = f"Checked unique trajectory step {turn} and its current contract."
            response = (tool_response(turn, reasoning=reasoning)
                        if turn % 2 else ack_response(turn, reasoning=reasoning))
            turns.append(issue82.classify_response(turn, response))
        summary = issue82.summarize_turns(turns)
        self.assertTrue(summary["passed"], summary["errors"])
        self.assertEqual(summary["max_tool_call_repetition"], 1)


class TrajectoryTests(unittest.TestCase):
    def test_seed_history_matches_public_preloop_shape(self) -> None:
        replayed = issue82.seed_history(
            seed_turns=16, context_records=3, replay_reasoning=True)
        dropped = issue82.seed_history(
            seed_turns=16, context_records=0, replay_reasoning=False)
        self.assertEqual(len(replayed), 41)
        self.assertEqual(len(dropped), 41)
        self.assertEqual(
            sum(message["role"] == "assistant" for message in replayed), 16)
        self.assertEqual(
            sum(message["role"] == "tool" for message in replayed), 8)
        self.assertTrue(all("reasoning" in message for message in replayed
                            if message["role"] == "assistant"))
        self.assertTrue(all("reasoning" not in message for message in dropped
                            if message["role"] == "assistant"))
        self.assertEqual(replayed[0]["content"].count("evidence-"), 3)

    def test_live_shape_builds_official_tool_history(self) -> None:
        client = FakeClient()
        result = issue82.run_trajectory(
            client, "test-model", turns=20, max_tokens=1024, seed=82_000,
            replay_reasoning=False, seed_turns=0, context_records=0)
        self.assertTrue(result["summary"]["passed"], result["summary"]["errors"])
        self.assertEqual(result["final_message_count"], 51)
        self.assertEqual(len(client.requests), 20)

        last_messages = client.requests[-1]["messages"]
        assistants = [message for message in last_messages
                      if message["role"] == "assistant"]
        tools = [message for message in last_messages if message["role"] == "tool"]
        self.assertEqual(len(assistants), 19)
        self.assertEqual(len(tools), 10)
        self.assertTrue(all("reasoning" not in message
                            and "reasoning_content" not in message
                            for message in assistants))
        self.assertTrue(all(message.get("tool_call_id") for message in tools))
        self.assertEqual(
            client.requests[-1]["chat_template_kwargs"],
            {"thinking": True, "reasoning_effort": "low"})
        self.assertNotIn("thinking_token_budget", client.requests[-1])
        self.assertTrue(all(request["return_token_ids"]
                            for request in client.requests))
        self.assertTrue(all(record["raw_output"]
                            for record in result["turn_records"]))
        self.assertTrue(all(len(record["raw_output_sha256"]) == 64
                            for record in result["turn_records"]))

    def test_raw_output_and_parser_mismatch_fails(self) -> None:
        class MissingDSMLClient(FakeClient):
            def detokenize(self, model: str, tokens: list[int]) -> str:
                return "<think>checked</think>plain text"

        result = issue82.run_trajectory(
            MissingDSMLClient(), "test-model", turns=2, max_tokens=1024,
            seed=82_000, replay_reasoning=False, seed_turns=0,
            context_records=0)
        self.assertFalse(result["summary"]["passed"])
        self.assertTrue(any("without DSML" in error
                            for error in result["summary"]["errors"]))

    def test_agent_shape_replays_reasoning_without_deprecated_alias(self) -> None:
        client = FakeClient()
        issue82.run_trajectory(
            client, "test-model", turns=4, max_tokens=1024, seed=82_000,
            replay_reasoning=True, seed_turns=0, context_records=0)
        assistants = [message for message in client.requests[-1]["messages"]
                      if message["role"] == "assistant"]
        self.assertTrue(assistants)
        self.assertTrue(all(isinstance(message.get("reasoning"), str)
                            for message in assistants))
        self.assertTrue(all("reasoning_content" not in message
                            for message in assistants))


if __name__ == "__main__":
    unittest.main()
