#!/usr/bin/env python3
"""Bounded live canary for issue #136 on the async DSpark TP=2 lane.

Runs the exact strict-tool, named-tool, ignore-EOS structured-output, ordinary
tool, and plain-chat matrix required by the issue #136 rollout gate.  Request
bodies and credentials are never written to the redacted JSON report.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REQUEST_TIMEOUT_SECONDS = 120
SEQUENTIAL_CASES = 20
CONCURRENT_CASES = 100
CONCURRENCY = 4
IGNORE_EOS_CASES = 5
ORDINARY_TOOL_CASES = 10
PLAIN_CHAT_CASES = 10
TOOL_NAME = "record_event"


@dataclass(frozen=True)
class CaseResult:
    lane: str
    label: str
    ok: bool
    http_status: int | None
    elapsed_ms: int
    error: str | None


class Client:
    def __init__(self, base_url: str, api_key: str | None):
        parsed = urllib.parse.urlsplit(base_url.rstrip("/"))
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base URL must be a credential-free HTTP(S) URL")
        path = parsed.path.rstrip("/")
        self.api_path = path if path.endswith("/v1") else path + "/v1"
        self.server_path = self.api_path[: -len("/v1")]
        self.origin = f"{parsed.scheme}://{parsed.netloc}"
        self.api_key = api_key

    def request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> tuple[int | None, bytes | None, str | None, int]:
        headers = {"Accept": "application/json"}
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body, separators=(",", ":")).encode("utf-8")
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.origin + path, data=data, headers=headers, method=method
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(
                request, timeout=REQUEST_TIMEOUT_SECONDS
            ) as response:
                raw = response.read()
                return response.status, raw, None, round((time.monotonic() - started) * 1000)
        except urllib.error.HTTPError as error:
            # Deliberately discard the response body: a server or proxy must
            # not be able to reflect credential/request bytes into evidence.
            status = error.code
            error.close()
            return status, None, f"http-{status}", round((time.monotonic() - started) * 1000)
        except TimeoutError:
            return None, None, "timeout", round((time.monotonic() - started) * 1000)
        except urllib.error.URLError as error:
            reason = "timeout" if isinstance(error.reason, TimeoutError) else "transport-error"
            return None, None, reason, round((time.monotonic() - started) * 1000)
        except (OSError, ValueError):
            return None, None, "transport-error", round((time.monotonic() - started) * 1000)

    def health(self) -> tuple[bool, int | None, str | None, int]:
        status, _, error, elapsed = self.request("GET", self.server_path + "/health")
        return status == 200, status, error, elapsed


def _choice_message(value: dict[str, Any]) -> dict[str, Any]:
    choices = value.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("choice-cardinality")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("message-shape")
    return message


def validate_tool_call(value: dict[str, Any]) -> None:
    message = _choice_message(value)
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
        raise ValueError("tool-call-cardinality")
    function = calls[0].get("function")
    if not isinstance(function, dict) or function.get("name") != TOOL_NAME:
        raise ValueError("tool-call-name")
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        raise ValueError("tool-arguments-type")
    try:
        decoded = json.loads(arguments)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("tool-arguments-json") from error
    if (
        not isinstance(decoded, dict)
        or set(decoded) != {"event_id", "count"}
        or not isinstance(decoded["event_id"], str)
        or not decoded["event_id"]
        or type(decoded["count"]) is not int
    ):
        raise ValueError("tool-arguments-schema")


def validate_choice_shape(value: dict[str, Any]) -> None:
    _choice_message(value)


def validate_plain_chat(value: dict[str, Any]) -> None:
    message = _choice_message(value)
    if not isinstance(message.get("content"), str):
        raise ValueError("plain-content-shape")


def case_label(lane: str, index: int) -> str:
    # Vary termination placement relative to the five speculative slots without
    # putting any request content into the report beyond this synthetic label.
    return f"{lane}-{index:03d}-" + ("X" * (1 + (index * 11) % 73))


def strict_tool_payload(model: str, label: str, named: bool) -> dict[str, Any]:
    tool_choice: Any = (
        {"type": "function", "function": {"name": TOOL_NAME}}
        if named
        else "required"
    )
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"Record event {label} with count 7. Return the required tool call.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": TOOL_NAME,
                    "description": "Record one event",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "event_id": {"type": "string"},
                            "count": {"type": "integer"},
                        },
                        "required": ["event_id", "count"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": tool_choice,
        "parallel_tool_calls": False,
        "temperature": 0,
        "max_tokens": 512,
        "stream": False,
    }


def ignore_eos_payload(model: str, label: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"Return a JSON record for {label} with count 7.",
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "issue136_record",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "event_id": {"type": "string"},
                        "count": {"type": "integer"},
                    },
                    "required": ["event_id", "count"],
                    "additionalProperties": False,
                },
            },
        },
        "ignore_eos": True,
        "temperature": 0,
        "max_tokens": 32,
        "stream": False,
    }


def ordinary_tool_payload(model: str, label: str) -> dict[str, Any]:
    payload = strict_tool_payload(model, label, named=False)
    function = payload["tools"][0]["function"]
    function.pop("strict")
    return payload


def plain_chat_payload(model: str, label: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": f"Reply briefly with OK for {label}."}],
        "chat_template_kwargs": {"thinking": False},
        "temperature": 0,
        "max_tokens": 64,
        "stream": False,
    }


def run_case(
    client: Client,
    lane: str,
    label: str,
    payload: dict[str, Any],
    validator: Callable[[dict[str, Any]], None],
) -> CaseResult:
    try:
        status, raw, transport_error, elapsed = client.request(
            "POST", client.api_path + "/chat/completions", payload
        )
    except Exception:
        return CaseResult(lane, label, False, None, 0, "client-error")
    if transport_error is not None or status != 200 or raw is None:
        return CaseResult(lane, label, False, status, elapsed, transport_error or f"http-{status}")
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("response-not-object")
        validator(value)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return CaseResult(lane, label, False, status, elapsed, "response-json")
    except ValueError as error:
        # Validators use fixed tokens only; never serialize model output.
        return CaseResult(lane, label, False, status, elapsed, str(error))
    return CaseResult(lane, label, True, status, elapsed, None)


def resolve_api_key(cli_key: str | None) -> str | None:
    single = os.environ.get("VLLM_API_KEY", "")
    multiple = os.environ.get("DSPARK_API_KEYS", "")
    meaningful_multiple = bool(multiple.strip(" \t"))
    if cli_key is not None and (single or meaningful_multiple):
        raise ValueError("set API credentials through exactly one source")
    if single and meaningful_multiple:
        raise ValueError("VLLM_API_KEY and DSPARK_API_KEYS are mutually exclusive")
    if cli_key is not None:
        if not cli_key:
            raise ValueError("--api-key must not be empty")
        return cli_key
    if single:
        return single
    if meaningful_multiple:
        if any(character in multiple for character in "\r\n\v\f\\"):
            raise ValueError("DSPARK_API_KEYS has invalid separators")
        keys = multiple.split()
        if any(key.startswith("-") for key in keys):
            raise ValueError("DSPARK_API_KEYS contains an invalid token")
        return keys[0]
    return None


def write_report(output: Path, report: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--model", default="deepseek-v4-flash-dspark")
    parser.add_argument("--api-key", help="bearer key; prefer VLLM_API_KEY or DSPARK_API_KEYS")
    parser.add_argument("--output", type=Path, required=True, help="redacted JSON evidence path")
    args = parser.parse_args()
    try:
        api_key = resolve_api_key(args.api_key)
        client = Client(args.base_url, api_key)
    except ValueError as error:
        parser.error(str(error))

    started_at = datetime.now(timezone.utc).isoformat()
    results: list[CaseResult] = []
    initial_ok, initial_status, initial_error, initial_ms = client.health()
    if initial_ok:
        for index in range(SEQUENTIAL_CASES):
            label = case_label("sequential", index)
            results.append(
                run_case(
                    client,
                    "strict-sequential",
                    label,
                    strict_tool_payload(args.model, label, named=False),
                    validate_tool_call,
                )
            )

        concurrent_inputs = []
        for index in range(CONCURRENT_CASES):
            label = case_label("concurrent", index)
            concurrent_inputs.append(
                (
                    client,
                    "strict-concurrency4-named" if index % 2 else "strict-concurrency4-required",
                    label,
                    strict_tool_payload(args.model, label, named=bool(index % 2)),
                    validate_tool_call,
                )
            )
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = [executor.submit(run_case, *item) for item in concurrent_inputs]
            results.extend(future.result() for future in futures)

        for index in range(IGNORE_EOS_CASES):
            label = case_label("ignore-eos", index)
            results.append(
                run_case(
                    client,
                    "strict-json-ignore-eos",
                    label,
                    ignore_eos_payload(args.model, label),
                    validate_choice_shape,
                )
            )
        for index in range(ORDINARY_TOOL_CASES):
            label = case_label("ordinary-tool", index)
            results.append(
                run_case(
                    client,
                    "ordinary-tool-control",
                    label,
                    ordinary_tool_payload(args.model, label),
                    validate_tool_call,
                )
            )
        for index in range(PLAIN_CHAT_CASES):
            label = case_label("plain-chat", index)
            results.append(
                run_case(
                    client,
                    "plain-chat-control",
                    label,
                    plain_chat_payload(args.model, label),
                    validate_plain_chat,
                )
            )

    final_ok, final_status, final_error, final_ms = client.health()
    failures = sum(not result.ok for result in results)
    expected_cases = (
        SEQUENTIAL_CASES
        + CONCURRENT_CASES
        + IGNORE_EOS_CASES
        + ORDINARY_TOOL_CASES
        + PLAIN_CHAT_CASES
    )
    passed = initial_ok and final_ok and len(results) == expected_cases and failures == 0
    report = {
        "schema_version": 1,
        "issue": 136,
        "status": "PASS" if passed else "FAIL",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "authenticated": api_key is not None,
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "concurrency": CONCURRENCY,
        "expected_cases": expected_cases,
        "completed_cases": len(results),
        "failed_cases": failures,
        "initial_health": {
            "ok": initial_ok,
            "http_status": initial_status,
            "error": initial_error,
            "elapsed_ms": initial_ms,
        },
        "final_health": {
            "ok": final_ok,
            "http_status": final_status,
            "error": final_error,
            "elapsed_ms": final_ms,
        },
        "results": [asdict(result) for result in results],
        "redaction": "credentials, request bodies, response bodies, and headers omitted",
    }
    write_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
