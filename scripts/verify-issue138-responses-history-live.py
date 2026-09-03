#!/usr/bin/env python3
"""Verify issue #138 full-history Responses replay against a live server.

This dependency-free verifier is mode-strict: callers must assert that the
legacy type-less assistant item is either rejected (stock/default) or accepted
(explicit compatibility flag).  It uses ``store: false`` and full request
history; it does not exercise ``previous_response_id`` or the Responses store.
"""
from __future__ import annotations

import argparse
import json
import secrets
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SYSTEM_TEXT = "Issue 138 verifier. Follow exact-output instructions."
ISSUE_INPUT = [
    {
        "role": "system",
        "content": (
            "Current model: deepseek-v4-flash-0731\n"
            "Current date: 2026-08-25\n"
            " Additional info for this conversation:"
        ),
    },
    {
        "role": "user",
        "content": [{"type": "input_text", "text": "hello"}],
    },
    {
        "role": "assistant",
        "content": [{
            "type": "output_text",
            "text": (
                "Hello there! 👋 It's great to hear from you. \n\n"
                "How can I help you today? Whether you have a question, need "
                "assistance with a task, or just want to chat about something, "
                "I'm all ears. What's on your mind?"
            ),
        }],
    },
    {
        "role": "user",
        "content": [{"type": "input_text", "text": "your version"}],
    },
]

NEGATIVE_ITEMS = {
    "missing_text": {
        "role": "assistant",
        "content": [{"type": "output_text"}],
    },
    "non_string_text": {
        "role": "assistant",
        "content": [{"type": "output_text", "text": 7}],
    },
    "multipart_output": {
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": "one"},
            {"type": "output_text", "text": "two"},
        ],
    },
    "mixed_output_input": {
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": "one"},
            {"type": "input_text", "text": "two"},
        ],
    },
    "explicit_null_type": {
        "type": None,
        "role": "assistant",
        "content": [{"type": "output_text", "text": "one"}],
    },
    "non_assistant_role": {
        "role": "user",
        "content": [{"type": "output_text", "text": "one"}],
    },
}

POSITIVE_CONTROL_ITEMS = {
    "easy_input_text": {
        "role": "assistant",
        "content": [{"type": "input_text", "text": "valid easy input"}],
    },
    "canonical_missing_annotations": {
        "type": "message",
        "id": "msg_stock_fill",
        "status": "completed",
        "role": "assistant",
        "content": [{
            "type": "output_text",
            "text": "stock fills annotations",
        }],
    },
    "canonical_output_message": {
        "type": "message",
        "id": "msg_control",
        "status": "completed",
        "role": "assistant",
        "content": [{
            "type": "output_text",
            "text": "valid output",
            "annotations": [],
        }],
    },
}


class VerificationError(RuntimeError):
    pass


class Client:
    def __init__(self, base_url: str, timeout: float):
        parsed = urllib.parse.urlparse(base_url.rstrip("/"))
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("base URL must be HTTP(S)")
        path = parsed.path.rstrip("/")
        self.api_path = path if path.endswith("/v1") else path + "/v1"
        self.origin = f"{parsed.scheme}://{parsed.netloc}"
        self.timeout = timeout

    def request_json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        raw_body = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.origin + path,
            data=raw_body,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status = response.status
                raw = response.read()
        except urllib.error.HTTPError as error:
            status = error.code
            raw = error.read()
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VerificationError(
                f"{method} {path} returned non-JSON HTTP {status}: {raw[:300]!r}"
            ) from error
        if not isinstance(value, dict):
            raise VerificationError(
                f"{method} {path} returned non-object JSON with HTTP {status}"
            )
        return status, value


def request_body(model: str, input_items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": model,
        "input": input_items,
        "max_output_tokens": 256,
        "temperature": 0,
        "reasoning": {"effort": "none"},
        "store": False,
    }


def usage_is_valid(response: dict[str, Any]) -> bool:
    usage = response.get("usage")
    return (
        isinstance(usage, dict)
        and type(usage.get("input_tokens")) is int
        and type(usage.get("output_tokens")) is int
    )


def completed_output_text(response: dict[str, Any]) -> str:
    if response.get("status") != "completed":
        raise VerificationError("Responses result status is not completed")
    if not usage_is_valid(response):
        raise VerificationError("Responses result has no integer usage fields")
    output = response.get("output")
    if not isinstance(output, list):
        raise VerificationError("Responses result output is not a list")
    messages = [
        item
        for item in output
        if isinstance(item, dict)
        and item.get("type") == "message"
        and item.get("role") == "assistant"
    ]
    if len(messages) != 1:
        raise VerificationError("expected exactly one assistant output message")
    content = messages[0].get("content")
    if not isinstance(content, list) or len(content) != 1:
        raise VerificationError("assistant output must contain exactly one part")
    part = content[0]
    if (
        not isinstance(part, dict)
        or part.get("type") != "output_text"
        or not isinstance(part.get("text"), str)
    ):
        raise VerificationError("assistant output part is not string output_text")
    return part["text"]


def require_200(
    status: int,
    value: dict[str, Any],
    *,
    expected_text: str | None = None,
) -> str:
    if status != 200:
        raise VerificationError(f"expected HTTP 200, got {status}: {value!r}")
    text = completed_output_text(value)
    if expected_text is not None and text.strip() != expected_text:
        raise VerificationError(
            f"output mismatch: expected {expected_text!r}, got {text.strip()!r}"
        )
    if expected_text is None and not text.strip():
        raise VerificationError("assistant output text is empty")
    return text


def require_validation_400(status: int, value: dict[str, Any]) -> None:
    if status != 400:
        raise VerificationError(f"expected HTTP 400, got {status}: {value!r}")
    error = value.get("error")
    if not isinstance(error, dict):
        raise VerificationError("HTTP 400 response has no error object")
    if type(error.get("code")) is not int or error["code"] != 400:
        raise VerificationError("HTTP 400 error.code is not numeric 400")
    if not isinstance(error.get("message"), str) or not error["message"]:
        raise VerificationError("HTTP 400 error.message is missing")


def require_legacy_locus(value: dict[str, Any]) -> None:
    message = value["error"]["message"]
    if "output_text" not in message or "type" not in message:
        raise VerificationError(
            "legacy rejection does not identify the output-message/type locus"
        )


def post(
    client: Client,
    model: str,
    input_items: list[dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    return client.request_json(
        "POST", client.api_path + "/responses", request_body(model, input_items)
    )


def verify(
    client: Client,
    model: str,
    expected_mode: str,
    *,
    nonce: str | None = None,
) -> dict[str, Any]:
    if expected_mode not in {"rejected", "accepted"}:
        raise ValueError("expected_mode must be rejected or accepted")
    if nonce is None:
        nonce = secrets.token_urlsafe(9).replace("-", "_")
    turn1_sentinel = f"I138_T1_{nonce}"
    turn2_sentinel = f"I138_T2_{nonce}"
    initial_input = [
        {"role": "system", "content": SYSTEM_TEXT},
        {
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": f"Reply with exactly {turn1_sentinel}.",
            }],
        },
    ]

    status, turn1 = post(client, model, initial_input)
    turn1_text = require_200(status, turn1, expected_text=turn1_sentinel)
    output = turn1.get("output")
    if not isinstance(output, list):
        raise VerificationError("turn-1 output array is missing")

    final_user = {
        "role": "user",
        "content": [{
            "type": "input_text",
            "text": f"Reply with exactly {turn2_sentinel}.",
        }],
    }
    canonical_input = [*initial_input, *output, final_user]
    status, canonical = post(client, model, canonical_input)
    require_200(status, canonical, expected_text=turn2_sentinel)

    legacy_item = {
        "role": "assistant",
        "content": [{"type": "output_text", "text": turn1_text}],
    }
    legacy_input = [*initial_input, legacy_item, final_user]
    status, legacy = post(client, model, legacy_input)
    if expected_mode == "rejected":
        require_validation_400(status, legacy)
        require_legacy_locus(legacy)
    else:
        require_200(status, legacy, expected_text=turn2_sentinel)

    status, issue_fixture = post(client, model, ISSUE_INPUT)
    if expected_mode == "rejected":
        require_validation_400(status, issue_fixture)
        require_legacy_locus(issue_fixture)
    else:
        require_200(status, issue_fixture)

    gates: dict[str, Any] = {
        "turn1": "PASS",
        "canonical_replay": "PASS",
        "legacy_replay": "PASS",
        "reported_issue_fixture": "PASS",
    }

    if expected_mode == "accepted":
        semantic_legacy_item = {
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": f"The launch code is {nonce}.",
            }],
        }
        semantic_input = [
            {"role": "system", "content": SYSTEM_TEXT},
            semantic_legacy_item,
            {
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": (
                        "What launch code did the immediately preceding assistant "
                        "state? Reply with exactly the code and no other text."
                    ),
                }],
            },
        ]
        status, semantic = post(client, model, semantic_input)
        require_200(status, semantic, expected_text=nonce)
        gates["assistant_semantic_continuity"] = "PASS"

        for name, item in NEGATIVE_ITEMS.items():
            status, value = post(client, model, [
                {"role": "system", "content": SYSTEM_TEXT},
                item,
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Reply OK."}],
                },
            ])
            require_validation_400(status, value)
            gates[f"negative_{name}"] = "PASS"

        for name, item in POSITIVE_CONTROL_ITEMS.items():
            status, value = post(client, model, [
                {"role": "system", "content": SYSTEM_TEXT},
                item,
                {
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": f"Reply with exactly {turn2_sentinel}.",
                    }],
                },
            ])
            require_200(status, value, expected_text=turn2_sentinel)
            gates[f"positive_{name}"] = "PASS"

    return {
        "expected_legacy": expected_mode,
        "store": False,
        "gates": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument(
        "--expect-legacy",
        choices=("rejected", "accepted"),
        required=True,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "schema_version": 1,
        "base_url": args.base_url,
        "model": args.model,
        "expected_legacy": args.expect_legacy,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        result = verify(Client(args.base_url, args.timeout), args.model, args.expect_legacy)
        report.update(result)
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    raw = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(raw, encoding="utf-8")
    print(raw, end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
