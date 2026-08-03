#!/usr/bin/env python3
"""Semantic OpenAI-compatible smoke tests for the authenticated DSpark origin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.error
import urllib.request


def request_json(base_url: str, key: str, path: str, payload=None, timeout=300):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def assert_message(payload, *, require_reasoning=False):
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AssertionError("response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise AssertionError("choice has no message")
    if require_reasoning and not message.get("reasoning_content"):
        raise AssertionError("reasoning response has no reasoning_content")
    return message


def run_stream(base_url: str, key: str, model: str) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply exactly STREAM_OK."}],
        "temperature": 0,
        "stream": True,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    saw_choice = False
    saw_done = False
    with urllib.request.urlopen(request, timeout=300) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            event = line[5:].strip()
            if event == "[DONE]":
                saw_done = True
                break
            chunk = json.loads(event)
            saw_choice = saw_choice or bool(chunk.get("choices"))
    if not saw_choice or not saw_done:
        raise AssertionError("stream did not contain choices and [DONE]")


def run_once(base_url: str, key: str, model: str) -> None:
    models = request_json(base_url, key, "/models")
    ids = {item.get("id") for item in models.get("data", []) if isinstance(item, dict)}
    if model not in ids:
        raise AssertionError(f"{model} missing from /v1/models")

    chat = request_json(
        base_url,
        key,
        "/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": "Reply exactly OK."}], "temperature": 0},
    )
    if not assert_message(chat).get("content"):
        raise AssertionError("chat response content is empty")

    reasoning = request_json(
        base_url,
        key,
        "/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Think briefly, then answer: 2+2?"}],
            "reasoning_effort": "low",
            "temperature": 0,
        },
    )
    assert_message(reasoning, require_reasoning=True)

    tools = [{
        "type": "function",
        "function": {
            "name": "record_expense",
            "description": "Record a categorized expense",
            "parameters": {
                "type": "object",
                "properties": {"amount": {"type": "number"}, "scope": {"type": "string"}},
                "required": ["amount", "scope"],
            },
        },
    }]
    tool_response = request_json(
        base_url,
        key,
        "/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Use record_expense for a personal expense of 125."}],
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": "record_expense"}},
            "temperature": 0,
        },
    )
    calls = assert_message(tool_response).get("tool_calls")
    if not calls or calls[0].get("function", {}).get("name") != "record_expense":
        raise AssertionError("tool_calls did not select record_expense")
    json.loads(calls[0]["function"]["arguments"])
    run_stream(base_url, key, model)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("direct",), default="direct")
    parser.add_argument("--base-url", default="http://172.30.0.1:8888/v1")
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    key = args.key_file.read_text().strip()
    if not key:
        raise SystemExit("key file is empty")
    for _ in range(args.runs):
        run_once(args.base_url, key, args.model)
    print(f"semantic {args.profile} smoke passed ({args.runs} run(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
