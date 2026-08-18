#!/usr/bin/env python3
"""Strict live verification for OpenAI-compatible Chat and Responses APIs.

The verifier is intentionally dependency-free.  It checks semantic output,
SSE framing, stateful ``previous_response_id`` tool continuation, strict JSON
schema output, reasoning, invalid-field errors, appended multi-turn prefix
reuse, and client-disconnect cleanup.
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ALLOWED_RESPONSE_EVENTS = {
    "response.created", "response.in_progress", "response.output_item.added",
    "response.output_item.done", "response.content_part.added",
    "response.content_part.done", "response.output_text.delta",
    "response.output_text.done", "response.completed",
}


def has_usage(value: dict[str, Any]) -> bool:
    usage = value.get("usage")
    return (isinstance(usage, dict)
            and type(usage.get("input_tokens")) is int
            and type(usage.get("output_tokens")) is int)


def responses_output_text(value: dict[str, Any]) -> str:
    pieces: list[str] = []
    output = value.get("output")
    if not isinstance(output, list):
        raise ValueError("responses output missing")
    for item in output:
        content = item.get("content") if isinstance(item, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if (isinstance(part, dict) and part.get("type") == "output_text"
                    and isinstance(part.get("text"), str)):
                pieces.append(part["text"])
    if not pieces:
        raise ValueError("responses output text missing")
    return "".join(pieces)


def parse_responses_sse(raw: bytes) -> str:
    normalized = raw.replace(b"\r\n", b"\n")
    if not normalized.endswith(b"\n\n"):
        raise ValueError("responses SSE is not frame-terminated")
    events: list[str] = []
    deltas: list[str] = []
    completed: list[dict[str, Any]] = []
    for frame in normalized.split(b"\n\n")[:-1]:
        lines = frame.split(b"\n")
        if (not lines or any(not line or not (
                line.startswith(b"event: ") or line.startswith(b"data: "))
                for line in lines)):
            raise ValueError("invalid Responses SSE frame")
        event_lines = [line[7:].decode("ascii", "strict")
                       for line in lines if line.startswith(b"event: ")]
        data_lines = [line[6:] for line in lines if line.startswith(b"data: ")]
        if len(event_lines) > 1 or len(data_lines) != 1 or data_lines[0] == b"[DONE]":
            raise ValueError("invalid Responses SSE event/data cardinality")
        value = json.loads(data_lines[0])
        event_type = value.get("type") if isinstance(value, dict) else None
        if (not isinstance(event_type, str) or event_type not in ALLOWED_RESPONSE_EVENTS
                or (event_lines and event_lines[0] != event_type)):
            raise ValueError("unknown or mismatched Responses SSE event")
        events.append(event_type)
        if event_type == "response.output_text.delta":
            delta = value.get("delta")
            if not isinstance(delta, str):
                raise ValueError("Responses text delta is not a string")
            deltas.append(delta)
        if event_type == "response.completed":
            completed.append(value)
    if len(completed) != 1 or not events or events[-1] != "response.completed":
        raise ValueError("Responses SSE must end in exactly one completed event")
    response = completed[0].get("response")
    if not isinstance(response, dict) or not has_usage(response):
        raise ValueError("Responses completed event has no usage")
    return "".join(deltas)


def response_function_call(value: dict[str, Any], name: str) -> tuple[str, str]:
    calls = [item for item in value.get("output", [])
             if isinstance(item, dict) and item.get("type") == "function_call"]
    if len(calls) != 1 or calls[0].get("name") != name:
        raise ValueError("expected Responses function call missing")
    call_id, arguments, response_id = (
        calls[0].get("call_id"), calls[0].get("arguments"), value.get("id"))
    if not all(isinstance(item, str) and item
               for item in (call_id, arguments, response_id)):
        raise ValueError("Responses function call identity invalid")
    if json.loads(arguments) != {}:
        raise ValueError("Responses function arguments must be an empty object")
    return call_id, response_id


def safe_cached_floor(prompt_tokens: int, swa_window: int = 4096,
                      block_size: int = 256) -> int:
    return max(0, ((prompt_tokens - swa_window) // block_size) * block_size)


def metric_snapshot(text: str) -> dict[str, float]:
    names = {
        "vllm:generation_tokens_total": "generation_tokens",
        "vllm:request_success_total": "successful_requests",
        "vllm:prefix_cache_queries_total": "cache_queries",
        "vllm:prefix_cache_hits_total": "cache_hits",
    }
    result = {short: 0.0 for short in names.values()}
    result.update({"running": 0.0, "waiting": 0.0})
    for line in (row.strip() for row in text.splitlines()):
        for full, short in names.items():
            if line.startswith(full + " ") or line.startswith(full + "{"):
                result[short] += float(line.rsplit(" ", 1)[1])
        if (line.startswith("vllm:num_requests_running ")
                or line.startswith("vllm:num_requests_running{")):
            result["running"] += float(line.rsplit(" ", 1)[1])
        if (line.startswith("vllm:num_requests_waiting ")
                or line.startswith("vllm:num_requests_waiting{")):
            result["waiting"] += float(line.rsplit(" ", 1)[1])
    return result


def disconnect_counter_chain_valid(before: dict[str, float],
                                   opened: dict[str, float],
                                   samples: list[dict[str, float]]) -> bool:
    if not samples or opened["running"] < 1:
        return False
    previous = opened
    idle_index = None
    counters = ("generation_tokens", "successful_requests",
                "cache_queries", "cache_hits")
    for index, current in enumerate(samples):
        if any(current[name] < previous[name] for name in counters):
            return False
        if current["running"] == 0 and current["waiting"] == 0:
            idle_index = index
            break
        previous = current
    if idle_index is None:
        return False
    idle = samples[idle_index]
    return (idle["successful_requests"] == before["successful_requests"]
            and idle["generation_tokens"] - opened["generation_tokens"] <= 32
            and all(row["generation_tokens"] == idle["generation_tokens"]
                    and row["successful_requests"] == idle["successful_requests"]
                    for row in samples[idle_index:]))


class ResponseHTTPError(RuntimeError):
    def __init__(self, method: str, path: str, status: int, body: bytes):
        self.method = method
        self.path = path
        self.status = status
        self.body = body
        super().__init__(
            f"{method} {path} returned HTTP {status}: {body[:300]!r}")


class Client:
    def __init__(self, base_url: str, timeout: float):
        parsed = urllib.parse.urlparse(base_url.rstrip("/"))
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("base URL must be HTTP(S)")
        path = parsed.path.rstrip("/")
        self.api_path = path if path.endswith("/v1") else path + "/v1"
        self.server_path = self.api_path[:-3]
        self.origin = f"{parsed.scheme}://{parsed.netloc}"
        self.parsed = parsed
        self.timeout = timeout

    def request(self, method: str, path: str,
                body: dict[str, Any] | None = None) -> tuple[int, bytes]:
        raw = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.origin + path, data=raw,
            headers={"Content-Type": "application/json"}, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    def json(self, method: str, path: str,
             body: dict[str, Any] | None = None) -> dict[str, Any]:
        status, raw = self.request(method, path, body)
        if status != 200:
            raise ResponseHTTPError(method, path, status, raw)
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RuntimeError(f"{method} {path} did not return an object")
        return value

    def open_stream(self, path: str, body: dict[str, Any]):
        connection_type = (http.client.HTTPSConnection if self.parsed.scheme == "https"
                           else http.client.HTTPConnection)
        connection = connection_type(self.parsed.hostname, self.parsed.port,
                                     timeout=self.timeout)
        connection.request("POST", path, body=json.dumps(body),
                           headers={"Content-Type": "application/json"})
        return connection, connection.getresponse()


def chat_content(value: dict[str, Any]) -> str:
    choices = value.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Chat response choice cardinality invalid")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise ValueError("Chat content missing")
    return content


def cached_tokens(value: dict[str, Any]) -> int:
    usage = value.get("usage")
    details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    if type(cached) is not int:
        raise ValueError("usage.prompt_tokens_details.cached_tokens missing")
    return cached


def check_catalog(client: Client, model: str) -> dict[str, Any]:
    models = client.json("GET", client.api_path + "/models")
    ids = [item.get("id") for item in models.get("data", []) if isinstance(item, dict)]
    if ids.count(model) != 1:
        raise RuntimeError(f"served model alias {model!r} is not unique: {ids!r}")
    return {"model_present_once": True}


def check_responses(client: Client, model: str) -> dict[str, Any]:
    base = {"model": model, "input": "Reply with exactly RESPONSES.",
            "max_output_tokens": 64, "temperature": 0,
            "reasoning": {"effort": "none"}, "store": False}
    text = client.json("POST", client.api_path + "/responses", base)
    if not has_usage(text) or responses_output_text(text).strip() != "RESPONSES":
        raise RuntimeError("Responses text contract failed")

    status, stream_raw = client.request(
        "POST", client.api_path + "/responses", {**base, "stream": True})
    if status != 200 or parse_responses_sse(stream_raw).strip() != "RESPONSES":
        raise RuntimeError("Responses SSE contract failed")

    tool_request = {**base, "input": "Call ping.", "store": True,
        "tools": [{"type": "function", "name": "ping", "description": "ping",
                   "parameters": {"type": "object", "properties": {},
                                  "additionalProperties": False}}],
        "tool_choice": {"type": "function", "name": "ping"}}
    tool = client.json("POST", client.api_path + "/responses", tool_request)
    call_id, response_id = response_function_call(tool, "ping")
    try:
        continuation = client.json("POST", client.api_path + "/responses", {
            "model": model, "previous_response_id": response_id,
            "input": [{"type": "function_call_output", "call_id": call_id,
                       "output": '{"ok":true}'},
                      {"role": "user", "content": "Reply with exactly TOOL_OK."}],
            "max_output_tokens": 64, "temperature": 0,
            "reasoning": {"effort": "none"}, "store": False,
        })
    except ResponseHTTPError as error:
        if error.status == 404:
            raise RuntimeError(
                "Responses stateful continuation returned HTTP 404 for "
                "previous_response_id; vLLM disables the Responses API store "
                "by default. Restart vLLM with "
                "VLLM_ENABLE_RESPONSES_API_STORE=1.") from error
        raise
    if responses_output_text(continuation).strip() != "TOOL_OK":
        raise RuntimeError("Responses stateful tool continuation failed")

    structured = client.json("POST", client.api_path + "/responses", {
        **base, "input": "Return JSON integer value 7.",
        "text": {"format": {"type": "json_schema", "name": "value",
            "strict": True, "schema": {"type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"], "additionalProperties": False}}},
    })
    if json.loads(responses_output_text(structured)) != {"value": 7}:
        raise RuntimeError("Responses strict structured output failed")

    reasoning = client.json("POST", client.api_path + "/responses", {
        **base, "input": "Think briefly, then reply with exactly REASONED.",
        "reasoning": {"effort": "low"},
    })
    output = reasoning.get("output")
    usage = reasoning.get("usage")
    details = usage.get("output_tokens_details") if isinstance(usage, dict) else None
    reasoning_seen = (
        any(isinstance(item, dict) and item.get("type") == "reasoning"
            for item in (output if isinstance(output, list) else []))
        or (isinstance(details, dict)
            and type(details.get("reasoning_tokens")) is int
            and details["reasoning_tokens"] > 0))
    if responses_output_text(reasoning).strip() != "REASONED" or not reasoning_seen:
        raise RuntimeError("Responses reasoning contract failed")

    bad_status, bad_raw = client.request("POST", client.api_path + "/responses",
                                         {**base, "include": ["not_supported"]})
    try:
        bad_value = json.loads(bad_raw)
    except json.JSONDecodeError as error:
        raise RuntimeError("unsupported-field response is not JSON") from error
    if bad_status < 400 or not isinstance(bad_value.get("error"), dict):
        raise RuntimeError("unsupported-field error contract failed")
    return {"text": True, "stream": True, "stateful_tool": True,
            "structured": True, "reasoning": True, "invalid_field": True}


def build_prefix(client: Client, model: str, target: int, nonce: str) -> str:
    unit = "append-only cache datum alpha beta gamma delta epsilon "
    text = f"unique verifier nonce {nonce}\n" + unit * max(1, target // 3)
    while True:
        value = client.json("POST", client.server_path + "/tokenize", {
            "model": model, "messages": [{"role": "user", "content": text}],
            "chat_template_kwargs": {"thinking": False},
        })
        count = value.get("count")
        if type(count) is not int:
            raise RuntimeError("/tokenize count missing")
        if count >= target:
            return text
        text += unit * max(1, (target - count) // 3)


def check_appended_multiturn(client: Client, model: str, target: int,
                             turns: int) -> dict[str, Any]:
    nonce = secrets.token_hex(8)
    prefix = build_prefix(client, model, target, nonce)
    messages: list[dict[str, str]] = [{"role": "user", "content":
        prefix + "\nReply with exactly READY."}]
    ratios: list[float] = []
    for index in range(turns):
        expected = "READY" if index == 0 else f"ACK{index + 1}"
        value = client.json("POST", client.api_path + "/chat/completions", {
            "model": model, "messages": messages, "max_tokens": 128,
            "temperature": 0, "chat_template_kwargs": {"thinking": False},
        })
        if chat_content(value).strip() != expected:
            raise RuntimeError(f"appended turn {index + 1} semantic output failed")
        usage = value.get("usage")
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        if type(prompt_tokens) is not int:
            raise RuntimeError("Chat prompt token usage missing")
        cached = cached_tokens(value)
        floor = safe_cached_floor(prompt_tokens)
        if index and cached < floor:
            raise RuntimeError(
                f"appended turn {index + 1} cache hit {cached} below safe floor {floor}")
        if index:
            ratios.append(cached / prompt_tokens)
        if index + 1 < turns:
            messages.append({"role": "assistant", "content": expected})
            messages.append({"role": "user", "content":
                f"Appended turn {index + 2}; reply with exactly ACK{index + 2}."})
    return {"turns": turns, "cached_token_ratios": ratios}


def read_metrics(client: Client) -> dict[str, float]:
    status, raw = client.request("GET", client.server_path + "/metrics")
    if status != 200:
        raise RuntimeError("metrics endpoint unavailable")
    return metric_snapshot(raw.decode("utf-8", "replace"))


def check_disconnect(client: Client, model: str) -> dict[str, Any]:
    before = read_metrics(client)
    if before["running"] or before["waiting"]:
        raise RuntimeError("disconnect gate requires an idle server")
    payload = {"model": model, "messages": [{"role": "user", "content":
        "Produce a long numbered list until stopped. " + secrets.token_hex(8)}],
        "stream": True, "max_tokens": 4096, "ignore_eos": True,
        "temperature": 0, "chat_template_kwargs": {"thinking": False}}
    connection, response = client.open_stream(client.api_path + "/chat/completions", payload)
    if response.status != 200:
        connection.close()
        raise RuntimeError(f"disconnect stream returned HTTP {response.status}")
    deadline = time.monotonic() + 10
    opened = read_metrics(client)
    while opened["running"] < 1 and time.monotonic() < deadline:
        time.sleep(0.1)
        opened = read_metrics(client)
    if opened["running"] < 1:
        connection.close()
        raise RuntimeError("disconnect request was never observed running")
    prefix = b""
    while len(prefix) < 65_536:
        line = response.readline()
        if not line:
            break
        prefix += line
        if re.search(rb'"id"\s*:\s*"[^"]+"', prefix):
            break
    connection.close()
    if not re.search(rb'"id"\s*:\s*"[^"]+"', prefix):
        raise RuntimeError("disconnect stream request id missing")
    samples: list[dict[str, float]] = []
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        current = read_metrics(client)
        samples.append(current)
        if current["running"] == 0 and current["waiting"] == 0:
            time.sleep(2)
            samples.append(read_metrics(client))
            if not disconnect_counter_chain_valid(before, opened, samples):
                raise RuntimeError("disconnect counter chain invalid")
            return {"client_disconnected": True, "server_idle": True,
                    "counter_chain_valid": True}
        time.sleep(2)
    raise RuntimeError("server did not become idle after disconnect")


def run_gate(report: dict[str, Any], name: str, function, *args) -> None:
    try:
        report["gates"][name] = {"status": "PASS", "value": function(*args)}
    except Exception as error:  # continue collecting independent gates
        report["gates"][name] = {
            "status": "FAIL", "error": f"{type(error).__name__}: {error}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--multiturn-prefix-tokens", type=int, default=21_000)
    parser.add_argument("--multiturn-turns", type=int, default=6)
    parser.add_argument("--skip-multiturn", action="store_true")
    parser.add_argument("--skip-disconnect", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.multiturn_prefix_tokens < 4096 or args.multiturn_turns < 2:
        parser.error("multiturn prefix must be >=4096 and turns must be >=2")
    client = Client(args.base_url, args.timeout)
    report: dict[str, Any] = {
        "schema_version": 1, "base_url": args.base_url, "model": args.model,
        "started_at": datetime.now(timezone.utc).isoformat(), "gates": {},
    }
    run_gate(report, "catalog", check_catalog, client, args.model)
    run_gate(report, "responses", check_responses, client, args.model)
    if not args.skip_multiturn:
        run_gate(report, "appended_multiturn", check_appended_multiturn,
                 client, args.model, args.multiturn_prefix_tokens,
                 args.multiturn_turns)
    if not args.skip_disconnect:
        run_gate(report, "disconnect", check_disconnect, client, args.model)
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["status"] = ("PASS" if all(gate["status"] == "PASS"
                                      for gate in report["gates"].values())
                        else "FAIL")
    raw = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(raw)
    print(raw, end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
