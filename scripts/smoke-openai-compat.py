#!/usr/bin/env python3
"""Semantic OpenAI-compatible smoke tests for the authenticated DSpark origin."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import signal
import stat
import urllib.error
import urllib.request


THINKING_MODE_KWARGS = {
    "off": {"reasoning_effort": "none", "drop_thinking": False},
    "low": {"thinking": True, "reasoning_effort": "low", "drop_thinking": False},
    "high": {"thinking": True, "reasoning_effort": "high", "drop_thinking": False},
    "max": {"thinking": True, "reasoning_effort": "max", "drop_thinking": False},
}
HISTORY_REASONING_FIELDS = ("reasoning", "reasoning_content")
# The pinned OpenAI request model preserves `reasoning` at the live /tokenize
# boundary. `reasoning_content` remains an encoder-level compatibility spelling
# but is discarded by request-model validation before the custom encoder runs.
LIVE_HISTORY_REASONING_FIELDS = ("reasoning",)


def request_json_url(url: str, key: str, payload=None, timeout=300):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def request_json(base_url: str, key: str, path: str, payload=None, timeout=300):
    return request_json_url(
        f"{base_url.rstrip('/')}{path}", key, payload=payload, timeout=timeout
    )


def request_text(base_url: str, key: str, path: str, timeout=5) -> str:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        headers={"Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode()


def prometheus_metric_sum(text: str, name: str) -> float:
    """Sum one exact Prometheus metric across all label sets."""
    values = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        metric_name = line.split(None, 1)[0].split("{", 1)[0]
        if metric_name != name:
            continue
        try:
            values.append(float(line.rsplit(None, 1)[1]))
        except (IndexError, ValueError):
            continue
    return sum(values)


def read_key_file(path: Path) -> str:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"key file must be a regular file: {path}")
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError(f"key file must have mode 0600: {path}")
    key = path.read_text().strip()
    if not key:
        raise ValueError("key file is empty")
    return key


def readiness_result(
    profile: str, state: str, *, wall_timeout: int | None = None
) -> dict:
    result = {
        "profile": profile,
        "state": state,
        "ready": state in {"api-ready", "semantic-ready"},
    }
    if wall_timeout is not None:
        result["wall_timeout_seconds"] = wall_timeout
    return result


def _is_timeout(error: BaseException) -> bool:
    if isinstance(error, TimeoutError):
        return True
    return isinstance(error, urllib.error.URLError) and isinstance(
        error.reason, TimeoutError
    )


@contextmanager
def wall_clock_deadline(seconds: int):
    """Interrupt the current main-thread probe after a fixed wall deadline."""
    if seconds <= 0:
        raise ValueError("wall deadline must be positive")
    previous_handler = signal.getsignal(signal.SIGALRM)

    def deadline_reached(_signum, _frame):
        raise TimeoutError(f"semantic generation exceeded {seconds}s wall deadline")

    signal.signal(signal.SIGALRM, deadline_reached)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def probe_api_liveness(
    base_url: str, key: str, model: str, *, profile="direct-origin"
) -> dict:
    """Cheap auth-boundary and model-discovery probe; never generates tokens."""
    models_url = f"{base_url.rstrip('/')}/models"
    try:
        with urllib.request.urlopen(models_url, timeout=5) as response:
            response.read(1)
    except urllib.error.HTTPError as error:
        try:
            if error.code != 401:
                return readiness_result(profile, "unavailable/not-ready")
        finally:
            error.close()
    except (OSError, urllib.error.URLError):
        return readiness_result(profile, "host-down")
    else:
        return readiness_result(profile, "auth-boundary-missing")

    try:
        models = request_json(base_url, key, "/models", timeout=10)
    except urllib.error.HTTPError as error:
        try:
            state = "auth-required" if error.code == 401 else "unavailable/not-ready"
            return readiness_result(profile, state)
        finally:
            error.close()
    except (OSError, urllib.error.URLError):
        return readiness_result(profile, "host-down")
    ids = {item.get("id") for item in models.get("data", []) if isinstance(item, dict)}
    state = "api-ready" if model in ids else "unavailable/not-ready"
    return readiness_result(profile, state)


def probe_semantic_readiness(
    base_url: str,
    key: str,
    model: str,
    *,
    wall_timeout=120,
    profile="direct-origin",
) -> dict:
    """Run one bounded authenticated generation without lifecycle side effects."""
    phase = "models"
    try:
        with wall_clock_deadline(wall_timeout):
            models = request_json(
                base_url, key, "/models", timeout=min(10, wall_timeout)
            )
            ids = {
                item.get("id") for item in models.get("data", [])
                if isinstance(item, dict)
            }
            if model not in ids:
                return readiness_result(
                    profile, "unavailable/not-ready", wall_timeout=wall_timeout
                )
            phase = "generation"
            response = request_json(
                base_url,
                key,
                "/chat/completions",
                {
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": "Reply exactly READY.",
                    }],
                    "max_completion_tokens": 16,
                    "temperature": 0,
                    "chat_template_kwargs": {
                        "reasoning_effort": "none",
                        "drop_thinking": False,
                    },
                },
                timeout=wall_timeout,
            )
            message = assert_message(
                response, forbid_reasoning=True, require_content=True
            )
            normalized = message["content"].strip().rstrip(".! ").strip()
            if normalized != "READY":
                raise AssertionError("semantic canary did not return READY")
    except urllib.error.HTTPError as error:
        try:
            state = "auth-required" if error.code == 401 else "unavailable/not-ready"
            return readiness_result(profile, state, wall_timeout=wall_timeout)
        finally:
            error.close()
    except (AssertionError, OSError, urllib.error.URLError) as error:
        if phase == "models" and not _is_timeout(error):
            return readiness_result(profile, "host-down", wall_timeout=wall_timeout)
        if _is_timeout(error):
            try:
                metrics_base = base_url.rstrip("/").removesuffix("/v1")
                metrics = request_text(metrics_base, key, "/metrics", timeout=5)
            except Exception:
                metrics = ""
            active = (
                prometheus_metric_sum(metrics, "vllm:num_requests_running")
                + prometheus_metric_sum(metrics, "vllm:num_requests_waiting")
            )
            state = "busy/degraded" if active > 0 else "unavailable/not-ready"
            return readiness_result(profile, state, wall_timeout=wall_timeout)
        return readiness_result(profile, "unavailable/not-ready", wall_timeout=wall_timeout)
    return readiness_result(profile, "semantic-ready", wall_timeout=wall_timeout)


def write_probe_result(result: dict, output: Path | None) -> None:
    encoded = json.dumps(result, sort_keys=True)
    print(encoded)
    if output is not None:
        output.write_text(encoded + "\n")
        output.chmod(0o600)


def assert_message(
    payload, *, require_reasoning=False, forbid_reasoning=False, require_content=False
):
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AssertionError("response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise AssertionError("choice has no message")
    # The pinned DSpark runtime exposes generated thinking as ``reasoning``.
    # Some OpenAI-compatible clients and vLLM paths normalize the same value to
    # ``reasoning_content``.  Both are valid representations of the DeepSeek V4
    # chat-template contract, so the gate must accept either without weakening
    # the requirement that reasoning is actually present.
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    if require_reasoning and not reasoning:
        raise AssertionError("reasoning response has no reasoning or reasoning_content")
    if forbid_reasoning and reasoning:
        raise AssertionError("thinking-off response unexpectedly emitted reasoning")
    if require_content and not message.get("content"):
        raise AssertionError("response final content is empty")
    return message


def classify_completion(payload) -> dict:
    """Classify empty capped output without conflating it with model failure."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AssertionError("response has no choices")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise AssertionError("choice has no message")
    finish_reason = choice.get("finish_reason")
    content = message.get("content")
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    if content:
        return {
            "state": "complete",
            "finish_reason": finish_reason,
            "reasoning_present": bool(reasoning),
        }
    if finish_reason == "length":
        return {
            "state": "truncated",
            "finish_reason": finish_reason,
            "reasoning_present": bool(reasoning),
        }
    raise AssertionError(
        f"empty final content without budget truncation (finish_reason={finish_reason!r})"
    )


def tokenize_url(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1") + "/tokenize"


def rendered_prompt(payload) -> str:
    for key in ("prompt", "rendered_prompt", "text"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    token_strs = payload.get("token_strs")
    if isinstance(token_strs, list) and all(isinstance(token, str) for token in token_strs):
        return "".join(token_strs)
    raise AssertionError("tokenize response did not expose a rendered prompt or token_strs")


def assert_marker_inside_thinking(rendered: str, marker: str) -> None:
    marker_at = rendered.find(marker)
    if marker_at < 0:
        raise AssertionError(f"rendered prompt dropped history marker {marker}")
    thinking_start = rendered.rfind("<think>", 0, marker_at)
    thinking_end = rendered.find("</think>", marker_at + len(marker))
    if thinking_start < 0 or thinking_end < 0:
        raise AssertionError(f"history marker {marker} is not inside balanced thinking boundaries")


def history_render_payload(model: str, field: str, marker: str, *, preserve: bool):
    # Set both sides explicitly because the repaired server default preserves
    # history; ``True`` characterizes the Python encoder's stripping default.
    template_kwargs = {
        "thinking": True,
        "reasoning_effort": "low",
        "drop_thinking": not preserve,
    }
    assistant = {"role": "assistant", "content": "The prior final answer."}
    assistant[field] = marker
    return {
        "model": model,
        "messages": [
            {"role": "user", "content": "Remember this calculation."},
            assistant,
            {"role": "user", "content": "Continue from the prior calculation."},
        ],
        "chat_template_kwargs": template_kwargs,
        "add_generation_prompt": True,
        "return_token_strs": True,
    }


def run_history_preservation(
    base_url: str,
    key: str,
    model: str,
    *,
    fields: tuple[str, ...] = LIVE_HISTORY_REASONING_FIELDS,
) -> None:
    url = tokenize_url(base_url)
    for field in fields:
        marker = f"DS4_HISTORY_{field.upper()}_MARKER"
        stripped = rendered_prompt(
            request_json_url(
                url, key, history_render_payload(model, field, marker, preserve=False)
            )
        )
        try:
            assert_marker_inside_thinking(stripped, marker)
        except AssertionError:
            pass
        else:
            raise AssertionError(
                f"default drop_thinking control unexpectedly retained {field} history"
            )

        preserved = rendered_prompt(
            request_json_url(
                url, key, history_render_payload(model, field, marker, preserve=True)
            )
        )
        assert_marker_inside_thinking(preserved, marker)


def tool_history_payload(model: str, arguments) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "user", "content": "Record the first synthetic expense."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_synthetic_1",
                    "type": "function",
                    "function": {"name": "record_expense", "arguments": arguments},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_synthetic_1",
                "content": '{"status":"recorded","synthetic":true}',
            },
            {"role": "user", "content": "Continue with the second synthetic expense."},
        ],
        "chat_template_kwargs": {
            "reasoning_effort": "none",
            "drop_thinking": False,
        },
        "add_generation_prompt": True,
        "return_token_strs": True,
    }


def run_tool_history_render(base_url: str, key: str, model: str) -> None:
    url = tokenize_url(base_url)
    arguments = {"amount": 125, "scope": "synthetic"}
    rendered_string = rendered_prompt(
        request_json_url(url, key, tool_history_payload(model, json.dumps(arguments)))
    )
    rendered_dict = rendered_prompt(
        request_json_url(url, key, tool_history_payload(model, arguments))
    )
    if rendered_string != rendered_dict:
        raise AssertionError("string and dictionary tool history rendered differently")
    if 'name="arguments"' in rendered_dict:
        raise AssertionError("tool history wrapped dictionary under arguments parameter")


def run_multiturn_tool(
    base_url: str, key: str, model: str, tools: list[dict], first_message: dict
) -> None:
    calls = first_message.get("tool_calls")
    if not calls:
        raise AssertionError("first tool turn did not include tool_calls")
    first_call = calls[0]
    if first_call.get("function", {}).get("name") != "record_expense":
        raise AssertionError("first tool turn did not select record_expense")
    arguments = first_call.get("function", {}).get("arguments")
    if isinstance(arguments, str):
        json.loads(arguments)
    elif not isinstance(arguments, dict):
        raise AssertionError("first tool arguments were neither JSON text nor a dictionary")
    continuation = request_json(
        base_url,
        key,
        "/chat/completions",
        {
            "model": model,
            "messages": [
                {"role": "user", "content": "Record a synthetic personal expense of 125."},
                first_message,
                {
                    "role": "tool",
                    "tool_call_id": first_call.get("id", "call_synthetic_1"),
                    "content": '{"status":"recorded","synthetic":true}',
                },
                {
                    "role": "user",
                    "content": "Now use record_expense for a synthetic business expense of 2.",
                },
            ],
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": "record_expense"}},
            "temperature": 0,
        },
    )
    second_calls = assert_message(continuation).get("tool_calls")
    if not second_calls or second_calls[0].get("function", {}).get("name") != "record_expense":
        raise AssertionError("second tool turn did not select record_expense")
    second_arguments = second_calls[0]["function"].get("arguments")
    if isinstance(second_arguments, str):
        json.loads(second_arguments)
    elif not isinstance(second_arguments, dict):
        raise AssertionError("second tool arguments were not parseable")


def run_reasoning_modes(base_url: str, key: str, model: str) -> None:
    for mode, template_kwargs in THINKING_MODE_KWARGS.items():
        response = request_json(
            base_url,
            key,
            "/chat/completions",
            {
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": f"Think briefly if enabled, then include {mode.upper()}_OK in the final answer.",
                }],
                "chat_template_kwargs": template_kwargs,
                "temperature": 0,
            },
        )
        assert_message(
            response,
            require_reasoning=mode != "off",
            forbid_reasoning=mode == "off",
            require_content=True,
        )


def run_unknown_field_rejection(base_url: str, key: str, model: str) -> None:
    try:
        request_json(
            base_url,
            key,
            "/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "Reply exactly NEVER."}],
                "dspark_invented_top_level_field": True,
            },
            timeout=30,
        )
    except urllib.error.HTTPError as error:
        try:
            if error.code != 400:
                raise AssertionError(
                    f"unsupported chat field returned HTTP {error.code}, expected 400"
                )
        finally:
            error.close()
    else:
        raise AssertionError("unsupported chat field reached the model boundary")


def run_cap_probe(base_url: str, key: str, model: str) -> dict:
    response = request_json(
        base_url,
        key,
        "/chat/completions",
        {
            "model": model,
            "messages": [{
                "role": "user",
                "content": "Think through 17*19 before giving the final answer.",
            }],
            "max_completion_tokens": 1,
            "temperature": 0,
            "chat_template_kwargs": {
                "thinking": True,
                "reasoning_effort": "high",
                "drop_thinking": False,
            },
        },
        timeout=60,
    )
    state = classify_completion(response)
    if state["state"] != "truncated":
        raise AssertionError(f"cap probe did not exercise truncation: {state}")
    diagnostic = "yes" if state["reasoning_present"] else "no"
    print(
        "cap probe classified budget truncation "
        f"(finish_reason=length, reasoning_present={diagnostic})"
    )
    return state


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


def run_once(base_url: str, key: str, model: str, *, profile: str) -> None:
    models = request_json(base_url, key, "/models")
    ids = {item.get("id") for item in models.get("data", []) if isinstance(item, dict)}
    if model not in ids:
        raise AssertionError(f"{model} missing from /v1/models")

    run_reasoning_modes(base_url, key, model)
    if profile in {"direct", "direct-origin"}:
        # `/tokenize` and the origin-only strict-field canary intentionally are
        # not public LiteLLM routes; validate them at the authenticated origin.
        run_history_preservation(base_url, key, model)
        run_tool_history_render(base_url, key, model)
        run_unknown_field_rejection(base_url, key, model)
    run_cap_probe(base_url, key, model)

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
    first_message = assert_message(tool_response)
    run_multiturn_tool(base_url, key, model, tools, first_message)
    run_stream(base_url, key, model)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=("direct", "direct-origin", "private-litellm"),
        default="direct",
    )
    parser.add_argument("--base-url", default="http://172.30.0.1:8888/v1")
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--runs", type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--api-liveness", action="store_true")
    mode.add_argument("--semantic-canary", action="store_true")
    parser.add_argument("--wall-timeout", type=int, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.wall_timeout <= 120:
        parser.error("--wall-timeout must be between 1 and 120 seconds")
    try:
        key = read_key_file(args.key_file)
    except ValueError as error:
        parser.error(str(error))

    evidence_profile = "direct-origin" if args.profile == "direct" else args.profile
    if args.api_liveness:
        result = probe_api_liveness(
            args.base_url, key, args.model, profile=evidence_profile
        )
        write_probe_result(result, args.output)
        return 0 if result["ready"] else 1
    if args.semantic_canary:
        result = probe_semantic_readiness(
            args.base_url,
            key,
            args.model,
            wall_timeout=args.wall_timeout,
            profile=evidence_profile,
        )
        write_probe_result(result, args.output)
        return 0 if result["ready"] else 1

    for _ in range(args.runs):
        run_once(args.base_url, key, args.model, profile=args.profile)
    print(f"semantic {args.profile} smoke passed ({args.runs} run(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
