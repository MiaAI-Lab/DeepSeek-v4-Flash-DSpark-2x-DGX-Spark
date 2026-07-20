#!/usr/bin/env python3
"""DSpark-r0b0tlab NVFP4-KV benchmark harness.

The harness uses OpenAI-compatible SSE chat streaming, records first-token and
inter-token timing, snapshots vLLM Prometheus counters around every measured
repeat, and keeps client and server throughput separate.  Telemetry failures
are recorded as explicit errors; no probe is silently ignored.

Run separate decode (long generation) and prefill (long prompt, one output
token) campaigns.  The resulting JSON records the exact prompt and timing
definitions so the two rates cannot be conflated.

The synthetic tests for this module are scaffold tests.  They exercise parser
and transport contracts without proving a live server, GPU, or container run.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import math
import re
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


SANITY_PROMPTS = [
    ("math", "What is 17 * 23? Reply with only the number.", 32),
    ("code", "Write a Python one-liner to reverse a string. Code only.", 64),
    (
        "reasoning",
        "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies? Answer yes or no with one sentence of reasoning.",
        64,
    ),
    ("factual", "What is the capital of Australia? One word answer.", 32),
    ("instruction", "List exactly three colors of the rainbow, comma-separated, nothing else.", 32),
]
BENCH_PROMPT = (
    "Write a complete, idiomatic Python implementation of a binary search tree "
    "with insert, delete, search, traversal, height, docstrings, and tests. Code only."
)
CONCURRENCY_LEVELS = (1, 2, 4, 6, 8, 12, 16)

PROMETHEUS_COUNTERS = {
    "prompt_tokens": "vllm:prompt_tokens_total",
    "generation_tokens": "vllm:generation_tokens_total",
    "speculative_drafted_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "speculative_accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
}


class BenchmarkError(RuntimeError):
    """Raised for an expected benchmark transport, parsing, or probe error."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def error_text(context: str, exc: BaseException) -> str:
    """Return an explicit, stable error without dumping response bodies."""
    return f"{context}: {type(exc).__name__}: {exc}"


def _url(base_url: str, path: str) -> str:
    return urllib.parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def fetch_text(base_url: str, path: str, timeout: float = 10.0) -> str:
    url = _url(base_url, path)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise BenchmarkError(f"GET {path} returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise BenchmarkError(f"GET {path} failed: {exc.reason}") from exc
    except (TimeoutError, OSError, ValueError) as exc:
        raise BenchmarkError(error_text(f"GET {path} failed", exc)) from exc
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BenchmarkError(error_text(f"GET {path} returned non-UTF-8 data", exc)) from exc


def fetch_json(base_url: str, path: str, timeout: float = 10.0) -> Any:
    body = fetch_text(base_url, path, timeout=timeout)
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise BenchmarkError(error_text(f"GET {path} returned invalid JSON", exc)) from exc


def _decode_sse_line(raw: bytes | str) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace").rstrip("\r\n")
    return str(raw).rstrip("\r\n")


def _delta_text(delta: Mapping[str, Any]) -> str:
    value = delta.get("content")
    if value is None:
        # vLLM reasoning parsers may stream parsed thinking in ``reasoning``.
        # Keep the older OpenAI-compatible
        # ``reasoning_content`` spelling as a compatibility fallback.
        value = delta.get("reasoning")
    if value is None:
        value = delta.get("reasoning_content")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def _usage_int(usage: Mapping[str, Any], key: str) -> int | None:
    value = usage.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkError(f"usage.{key} must be an integer")
    converted = value
    if converted < 0:
        raise BenchmarkError(f"usage.{key} must not be negative")
    return converted


def _failed_request(started: float, clock: Callable[[], float], error: str) -> dict[str, Any]:
    try:
        elapsed = max(0.0, clock() - started)
    except (TypeError, ValueError) as exc:
        elapsed = 0.0
        error = f"{error}; clock failure: {error_text('clock', exc)}"
    return {
        "ok": False,
        "request_seconds": elapsed,
        "ttft_seconds": None,
        "itl_seconds": [],
        "tpot_seconds": None,
        "completion_tokens": 0,
        "prompt_tokens": None,
        "token_count_source": "unavailable",
        "nonempty_chunk_count": 0,
        "response_text": "",
        "usage": None,
        "errors": [error],
    }


def stream_chat(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 256,
    include_usage: bool = True,
    timeout: float = 300.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Send one OpenAI-compatible streaming chat request and measure arrivals.

    TTFT starts immediately before opening the request.  ITL is measured only
    between subsequent non-empty content/reasoning chunks.  Usage is taken
    from the final usage-bearing SSE event when available; a missing usage
    event is an explicit request error and uses a clearly-labelled fallback
    count for continued aggregate reporting.
    """
    started = clock()
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "chat_template_kwargs": {"thinking": False},
    }
    if include_usage:
        payload["stream_options"] = {"include_usage": True}
    request = urllib.request.Request(
        _url(base_url, "/v1/chat/completions"),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    chunks: list[str] = []
    arrivals: list[float] = []
    itl: list[float] = []
    errors: list[str] = []
    usage_obj: dict[str, Any] | None = None
    usage_prompt: int | None = None
    usage_completion: int | None = None
    saw_done = False
    try:
        response = opener(request, timeout=timeout)
        with response as stream:
            for raw in stream:
                line = _decode_sse_line(raw)
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    saw_done = True
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise BenchmarkError(error_text("invalid SSE JSON event", exc)) from exc
                if not isinstance(event, Mapping):
                    raise BenchmarkError("SSE event must be a JSON object")
                raw_usage = event.get("usage")
                if isinstance(raw_usage, Mapping):
                    usage_obj = dict(raw_usage)
                    usage_prompt = _usage_int(raw_usage, "prompt_tokens")
                    usage_completion = _usage_int(raw_usage, "completion_tokens")
                choices = event.get("choices")
                if not isinstance(choices, list):
                    choices = []
                for choice in choices:
                    if not isinstance(choice, Mapping):
                        raise BenchmarkError("SSE choices entries must be objects")
                    delta = choice.get("delta")
                    if not isinstance(delta, Mapping):
                        continue
                    text = _delta_text(delta)
                    if text:
                        now = clock()
                        if arrivals:
                            itl.append(max(0.0, now - arrivals[-1]))
                        arrivals.append(now)
                        chunks.append(text)
    except BenchmarkError as exc:
        errors.append(str(exc))
    except urllib.error.HTTPError as exc:
        errors.append(f"chat request returned HTTP {exc.code}")
    except urllib.error.URLError as exc:
        errors.append(f"chat request failed: {exc.reason}")
    except (TimeoutError, OSError, ValueError) as exc:
        errors.append(error_text("chat request failed", exc))
    except Exception as exc:
        errors.append(error_text("unexpected chat request failure", exc))

    try:
        finished = clock()
        request_seconds = max(0.0, finished - started)
    except (TypeError, ValueError) as exc:
        request_seconds = 0.0
        errors.append(error_text("request clock failed", exc))
    if not saw_done:
        errors.append("SSE stream ended without a [DONE] event")
    if not arrivals:
        errors.append("SSE stream contained no nonempty token or chunk")
    if include_usage and usage_obj is None:
        errors.append("include_usage was requested but no usage event was received")
    elif include_usage and (usage_prompt is None or usage_completion is None):
        errors.append("usage event did not include integer prompt_tokens and completion_tokens")
    token_source = "usage" if usage_completion is not None else "nonempty_chunks_fallback"
    completion_tokens = usage_completion if usage_completion is not None else len(arrivals)
    ttft = max(0.0, arrivals[0] - started) if arrivals else None
    return {
        "ok": not errors,
        "request_seconds": request_seconds,
        "ttft_seconds": ttft,
        "itl_seconds": itl,
        "tpot_seconds": statistics.mean(itl) if itl else None,
        "completion_tokens": completion_tokens,
        "prompt_tokens": usage_prompt,
        "token_count_source": token_source,
        "nonempty_chunk_count": len(arrivals),
        "response_text": "".join(chunks)[:200],
        "usage": usage_obj,
        "errors": errors,
    }


_SAMPLE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+([^\s]+)(?:\s+\d+)?$"
)
_LABEL_KEY_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


def _parse_prometheus_labels(block: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    pos = 0
    while pos < len(block):
        while pos < len(block) and block[pos].isspace():
            pos += 1
        key_match = _LABEL_KEY_RE.match(block, pos)
        if key_match is None:
            raise BenchmarkError(f"invalid Prometheus label near offset {pos}")
        key = key_match.group(0)
        pos = key_match.end()
        while pos < len(block) and block[pos].isspace():
            pos += 1
        if pos >= len(block) or block[pos] != "=":
            raise BenchmarkError(f"Prometheus label {key!r} is missing '='")
        pos += 1
        while pos < len(block) and block[pos].isspace():
            pos += 1
        if pos >= len(block) or block[pos] != '"':
            raise BenchmarkError(f"Prometheus label {key!r} is not quoted")
        pos += 1
        chars: list[str] = []
        while pos < len(block):
            char = block[pos]
            pos += 1
            if char == '"':
                break
            if char == "\\":
                if pos >= len(block):
                    raise BenchmarkError(f"Prometheus label {key!r} has a trailing escape")
                escaped = block[pos]
                pos += 1
                chars.append({"n": "\n", "r": "\r", "t": "\t"}.get(escaped, escaped))
            else:
                chars.append(char)
        else:
            raise BenchmarkError(f"Prometheus label {key!r} is unterminated")
        if key in labels:
            raise BenchmarkError(f"duplicate Prometheus label {key!r}")
        labels[key] = "".join(chars)
        while pos < len(block) and block[pos].isspace():
            pos += 1
        if pos < len(block):
            if block[pos] != ",":
                raise BenchmarkError(f"Prometheus labels are missing a comma near offset {pos}")
            pos += 1
    return labels


def parse_prometheus_samples(text: str) -> list[tuple[str, dict[str, str], float]]:
    """Parse Prometheus samples without relying on label order."""
    samples: list[tuple[str, dict[str, str], float]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE_RE.match(line)
        if match is None:
            continue
        name, label_block, raw_value = match.groups()
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise BenchmarkError(error_text(f"invalid Prometheus value on line {line_number}", exc)) from exc
        if not math.isfinite(value):
            raise BenchmarkError(f"non-finite Prometheus value on line {line_number}")
        labels = _parse_prometheus_labels(label_block or "")
        samples.append((name, labels, value))
    return samples


def prometheus_counter_snapshot(text: str, model: str) -> dict[str, float | None]:
    """Return model-scoped counters; label order and unrelated series do not matter."""
    values: dict[str, float] = {}
    for metric_name, labels, value in parse_prometheus_samples(text):
        field = next((key for key, name in PROMETHEUS_COUNTERS.items() if name == metric_name), None)
        if field is None or labels.get("model_name") != model:
            continue
        values[field] = values.get(field, 0.0) + value
    return {field: values.get(field) for field in PROMETHEUS_COUNTERS}


def counter_deltas(
    before: Mapping[str, float | None], after: Mapping[str, float | None]
) -> tuple[dict[str, float | None], list[str]]:
    deltas: dict[str, float | None] = {}
    errors: list[str] = []
    for field in PROMETHEUS_COUNTERS:
        old = before.get(field)
        new = after.get(field)
        if old is None or new is None:
            deltas[field] = None
            errors.append(f"missing Prometheus counter for {field} before or after repeat")
            continue
        if new < old:
            deltas[field] = new
            errors.append(f"Prometheus counter reset for {field}; used post-reset value")
            continue
        deltas[field] = new - old
    return deltas, errors


def _read_meminfo() -> dict[str, int]:
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise BenchmarkError(error_text("read /proc/meminfo failed", exc)) from exc
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, separator, rest = line.partition(":")
        if not separator:
            continue
        fields = rest.strip().split()
        if not fields:
            continue
        try:
            value = int(fields[0])
        except ValueError as exc:
            raise BenchmarkError(error_text(f"invalid /proc/meminfo value for {key}", exc)) from exc
        multiplier = 1024 if len(fields) > 1 and fields[1] == "kB" else 1
        values[key] = value * multiplier
    required = ("MemAvailable", "SwapTotal", "SwapFree")
    missing = [key for key in required if key not in values]
    if missing:
        raise BenchmarkError(f"/proc/meminfo missing required keys: {', '.join(missing)}")
    values["SwapUsed"] = max(0, values["SwapTotal"] - values["SwapFree"])
    return values


def _read_gpu_telemetry() -> dict[str, float]:
    command = [
        "nvidia-smi",
        "--query-gpu=power.draw,temperature.gpu,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError, TimeoutError) as exc:
        raise BenchmarkError(error_text("nvidia-smi telemetry failed", exc)) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "no stderr"
        raise BenchmarkError(f"nvidia-smi telemetry returned {completed.returncode}: {detail}")
    rows: list[tuple[float, float, float]] = []
    for line_number, line in enumerate(completed.stdout.splitlines(), start=1):
        parts = [part.strip() for part in line.split(",")]
        if not line.strip():
            continue
        if len(parts) != 3:
            raise BenchmarkError(f"nvidia-smi telemetry line {line_number} has {len(parts)} columns, expected 3")
        try:
            rows.append(tuple(float(part) for part in parts))
        except ValueError as exc:
            raise BenchmarkError(error_text(f"invalid nvidia-smi telemetry line {line_number}", exc)) from exc
    if not rows:
        raise BenchmarkError("nvidia-smi telemetry returned no GPU rows")
    return {
        "power_watts": statistics.mean(row[0] for row in rows),
        "temperature_c": statistics.mean(row[1] for row in rows),
        "utilization_gpu_percent": statistics.mean(row[2] for row in rows),
    }


def sample_telemetry(
    stop: threading.Event,
    samples: list[dict[str, float]],
    errors: list[str],
    interval: float = 1.0,
    gpu_reader: Callable[[], dict[str, float]] = _read_gpu_telemetry,
    memory_reader: Callable[[], dict[str, int]] = _read_meminfo,
) -> None:
    """Collect GPU and host telemetry until stop, retaining explicit failures."""
    while not stop.is_set():
        sample: dict[str, float] = {"timestamp": time.time()}
        try:
            sample.update(gpu_reader())
        except (BenchmarkError, OSError, subprocess.SubprocessError, TimeoutError, ValueError) as exc:
            errors.append(error_text("GPU telemetry sample failed", exc))
        except Exception as exc:
            errors.append(error_text("unexpected GPU telemetry sample failure", exc))
        try:
            memory = memory_reader()
            sample.update(
                {
                    "host_mem_available_bytes": float(memory["MemAvailable"]),
                    "host_swap_total_bytes": float(memory["SwapTotal"]),
                    "host_swap_free_bytes": float(memory["SwapFree"]),
                    "host_swap_used_bytes": float(memory["SwapUsed"]),
                }
            )
        except (BenchmarkError, OSError, KeyError, ValueError) as exc:
            errors.append(error_text("host memory telemetry sample failed", exc))
        except Exception as exc:
            errors.append(error_text("unexpected host memory telemetry sample failure", exc))
        if len(sample) > 1:
            samples.append(sample)
        stop.wait(interval)


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def numeric_summary(values: Iterable[float | int | None]) -> dict[str, float | int | None]:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p99": None, "min": None, "max": None}
    return {
        "count": len(numbers),
        "mean": statistics.mean(numbers),
        "p50": percentile(numbers, 0.50),
        "p90": percentile(numbers, 0.90),
        "p99": percentile(numbers, 0.99),
        "min": min(numbers),
        "max": max(numbers),
    }


def telemetry_summary(samples: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    mapping = {
        "power_watts": "power_watts",
        "temperature_c": "temperature_c",
        "utilization_gpu_percent": "utilization_gpu_percent",
        "host_mem_available_bytes": "host_mem_available_bytes",
        "host_swap_total_bytes": "host_swap_total_bytes",
        "host_swap_free_bytes": "host_swap_free_bytes",
        "host_swap_used_bytes": "host_swap_used_bytes",
    }
    result: dict[str, Any] = {}
    for output_key, sample_key in mapping.items():
        result[output_key] = numeric_summary(sample.get(sample_key) for sample in samples)
    return result


def _request_error(concurrency: int, error: str) -> dict[str, Any]:
    return {
        "ok": False,
        "request_seconds": None,
        "ttft_seconds": None,
        "itl_seconds": [],
        "tpot_seconds": None,
        "completion_tokens": 0,
        "prompt_tokens": None,
        "token_count_source": "unavailable",
        "nonempty_chunk_count": 0,
        "response_text": "",
        "usage": None,
        "concurrency": concurrency,
        "errors": [error],
    }


def run_batch(
    concurrency: int,
    prompt: str,
    max_tokens: int,
    base_url: str,
    model: str,
    include_usage: bool,
    timeout: float,
) -> list[dict[str, Any]]:
    with cf.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                stream_chat,
                base_url,
                model,
                prompt,
                max_tokens,
                include_usage,
                timeout,
            )
            for _ in range(concurrency)
        ]
        rows: list[dict[str, Any]] = []
        for future in cf.as_completed(futures):
            try:
                row = future.result()
            except Exception as exc:
                row = _request_error(concurrency, error_text("worker request failed", exc))
            row["concurrency"] = concurrency
            rows.append(row)
        return rows


def _request_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    request_seconds = [row.get("request_seconds") for row in rows]
    ttft_seconds = [row.get("ttft_seconds") for row in rows]
    itl_seconds = [interval for row in rows for interval in row.get("itl_seconds", [])]
    tpot_seconds = [row.get("tpot_seconds") for row in rows]
    return {
        "request_seconds": numeric_summary(request_seconds),
        "ttft_seconds": numeric_summary(ttft_seconds),
        "itl_seconds": numeric_summary(itl_seconds),
        "tpot_seconds": numeric_summary(tpot_seconds),
    }


def _repeat_row(
    repeat: int,
    concurrency: int,
    started: float,
    finished: float,
    requests: Sequence[Mapping[str, Any]],
    before: Mapping[str, float | None],
    after: Mapping[str, float | None],
    deltas: Mapping[str, float | None],
    metric_errors: Sequence[str],
    telemetry: Sequence[Mapping[str, float]],
    telemetry_errors: Sequence[str],
    extra_errors: Sequence[str],
) -> dict[str, Any]:
    elapsed = max(0.0, finished - started)
    client_tokens = sum(int(row.get("completion_tokens") or 0) for row in requests)
    server_prompt = deltas.get("prompt_tokens")
    server_decode = deltas.get("generation_tokens")
    all_errors = list(metric_errors) + list(telemetry_errors) + list(extra_errors)
    for request in requests:
        all_errors.extend(str(error) for error in request.get("errors", []))
    client_output_rate = client_tokens / elapsed if elapsed else None
    server_prompt_rate = server_prompt / elapsed if elapsed and server_prompt is not None else None
    server_decode_rate = server_decode / elapsed if elapsed and server_decode is not None else None
    return {
        "repeat": repeat,
        "concurrency": concurrency,
        "elapsed_seconds": elapsed,
        "requests_ok": sum(1 for row in requests if row.get("ok")),
        "requests_error": sum(1 for row in requests if not row.get("ok")),
        "client_output_tokens": client_tokens,
        "client_output_tokens_per_second": client_output_rate,
        "server_prompt_tokens_per_second": server_prompt_rate,
        "server_decode_tokens_per_second": server_decode_rate,
        "server_counter_before": dict(before),
        "server_counter_after": dict(after),
        "server_counter_deltas": dict(deltas),
        "server_token_deltas": {
            "prompt_tokens": server_prompt,
            "generation_tokens": server_decode,
            "speculative_drafted_tokens": deltas.get("speculative_drafted_tokens"),
            "speculative_accepted_tokens": deltas.get("speculative_accepted_tokens"),
        },
        "request_metrics": _request_metrics(requests),
        "telemetry": telemetry_summary(telemetry),
        "requests": [dict(row) for row in requests],
        "telemetry_sample_count": len(telemetry),
        "errors": all_errors,
        "ok": not all_errors and all(bool(row.get("ok")) for row in requests),
    }


def run_repeat(
    repeat: int,
    concurrency: int,
    prompt: str,
    max_tokens: int,
    base_url: str,
    model: str,
    include_usage: bool,
    timeout: float,
    telemetry_interval: float,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        before_text = fetch_text(base_url, "/metrics", timeout=timeout)
        before = prometheus_counter_snapshot(before_text, model)
    except BenchmarkError as exc:
        before = {field: None for field in PROMETHEUS_COUNTERS}
        errors.append(str(exc))
    samples: list[dict[str, float]] = []
    telemetry_errors: list[str] = []
    stop = threading.Event()
    worker = threading.Thread(
        target=sample_telemetry,
        args=(stop, samples, telemetry_errors, telemetry_interval),
        name=f"telemetry-c{concurrency}-r{repeat}",
        daemon=True,
    )
    worker.start()
    started = time.monotonic()
    try:
        requests = run_batch(concurrency, prompt, max_tokens, base_url, model, include_usage, timeout)
    except Exception as exc:
        requests = [_request_error(concurrency, error_text("repeat request batch failed", exc))]
        errors.append(error_text("repeat request batch failed", exc))
    finally:
        finished = time.monotonic()
        stop.set()
        worker.join(timeout=max(2.0, telemetry_interval + 1.0))
        if worker.is_alive():
            telemetry_errors.append("telemetry worker did not stop before repeat close")
    try:
        after_text = fetch_text(base_url, "/metrics", timeout=timeout)
        after = prometheus_counter_snapshot(after_text, model)
    except BenchmarkError as exc:
        after = {field: None for field in PROMETHEUS_COUNTERS}
        errors.append(str(exc))
    deltas, metric_errors = counter_deltas(before, after)
    if not samples:
        telemetry_errors.append("no telemetry samples were collected")
    return _repeat_row(
        repeat,
        concurrency,
        started,
        finished,
        requests,
        before,
        after,
        deltas,
        metric_errors,
        samples,
        telemetry_errors,
        errors,
    )


def extract_log_facts(logs: str) -> dict[str, str]:
    facts: dict[str, str] = {}
    patterns = {
        "kv_tokens": r"GPU KV cache size: ([0-9,]+) tokens",
        "max_conc": r"Maximum concurrency for ([0-9,]+) tokens per request: ([0-9.]+)x",
        "kv_mem": r"Available KV cache memory: ([0-9.]+) GiB",
        "kv_dtype": r"kv_cache_dtype=(\S+)",
        "cache_py_dtype": r"Using (\S+) data type to store kv cache",
        "attn_backend": r"attention_backend.*?(\S+)",
        "vllm_ver": r"vllm-([0-9.]+)",
        "model_class": r"(DeepSeekV4DSparkModel)",
        "nvfp4_ds_mla": r"(nvfp4_ds_mla)",
        "native_moe": r"(B12X|DEEPGEMM_MXFP4)",
        "deepgemm_e8m0": r"(DeepGEMM E8M0)",
        "nccl_ib": r"(Using network IB|NET/IB)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, logs)
        if match:
            facts[key] = match.group(1) if key != "max_conc" else f"{match.group(1)} @ {match.group(2)}x"
    return facts


def container_log_facts(container_name: str) -> dict[str, Any]:
    if not container_name or container_name.strip() != container_name:
        raise BenchmarkError("container name must be a non-empty exact name without surrounding whitespace")
    try:
        completed = subprocess.run(
            ["docker", "logs", "--tail", "2000", container_name],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, TimeoutError) as exc:
        raise BenchmarkError(error_text(f"docker logs for container {container_name!r} failed", exc)) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "no stderr"
        raise BenchmarkError(f"docker logs for exact container {container_name!r} returned {completed.returncode}: {detail}")
    return extract_log_facts(completed.stdout + completed.stderr)


def gpu_identity() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError, TimeoutError) as exc:
        raise BenchmarkError(error_text("GPU identity probe failed", exc)) from exc
    if completed.returncode != 0:
        raise BenchmarkError(f"GPU identity probe returned {completed.returncode}")
    identities: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if not line.strip():
            continue
        if len(parts) != 3:
            raise BenchmarkError("GPU identity probe returned an unexpected column count")
        identities.append({"name": parts[0], "memory_total": parts[1], "driver_version": parts[2]})
    if not identities:
        raise BenchmarkError("GPU identity probe returned no GPUs")
    return identities


def _model_catalog(value: Any, expected_model: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"count": 0, "ids": [], "error": "models response was not an object"}
    data = value.get("data")
    if not isinstance(data, list):
        return {"count": 0, "ids": [], "error": "models response did not contain a data list"}
    ids = [str(item.get("id")) for item in data if isinstance(item, Mapping) and item.get("id") is not None]
    result: dict[str, Any] = {"count": len(ids), "ids": ids}
    if expected_model not in ids:
        result["error"] = f"requested model {expected_model!r} was not present in /v1/models"
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8888")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="benchmark_dspark.json")
    parser.add_argument("--concurrency", nargs="+", type=int, choices=CONCURRENCY_LEVELS, default=list(CONCURRENCY_LEVELS))
    parser.add_argument("--repeats", type=int, default=3, help="measured repeats per concurrency after warmup")
    parser.add_argument("--warmup", type=int, default=1, help="warmup batches per concurrency")
    parser.add_argument("--include-usage", dest="include_usage", action="store_true", default=True)
    parser.add_argument("--no-include-usage", dest="include_usage", action="store_false")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--telemetry-interval", type=float, default=1.0)
    parser.add_argument("--container-name", required=True, help="exact run-owned container name used for log facts")
    parser.add_argument("--prompt", default=BENCH_PROMPT)
    parser.add_argument("--prompt-file", type=Path, help="UTF-8 prompt file; overrides --prompt")
    parser.add_argument(
        "--synthetic-prompt-words",
        type=int,
        help="build a deterministic long prompt with this many whitespace-delimited words",
    )
    parser.add_argument(
        "--unique-prefix-per-repeat",
        action="store_true",
        help="prepend a deterministic nonce to each warmup/repeat to prevent prefix-cache reuse",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.repeats < 1:
        raise BenchmarkError("--repeats must be at least 1")
    if args.warmup < 0:
        raise BenchmarkError("--warmup must not be negative")
    if args.timeout <= 0 or args.telemetry_interval <= 0:
        raise BenchmarkError("--timeout and --telemetry-interval must be positive")
    if args.max_tokens < 1:
        raise BenchmarkError("--max-tokens must be positive")
    if not args.concurrency:
        raise BenchmarkError("at least one concurrency level is required")
    if args.prompt_file is not None and args.synthetic_prompt_words is not None:
        raise BenchmarkError("--prompt-file and --synthetic-prompt-words are mutually exclusive")
    if args.synthetic_prompt_words is not None:
        if args.synthetic_prompt_words < 1:
            raise BenchmarkError("--synthetic-prompt-words must be positive")
        words = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta")
        args.prompt = " ".join(words[index % len(words)] for index in range(args.synthetic_prompt_words))
    elif args.prompt_file is not None:
        try:
            args.prompt = args.prompt_file.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise BenchmarkError(error_text(f"cannot read --prompt-file {args.prompt_file}", exc)) from exc
        if not args.prompt:
            raise BenchmarkError("--prompt-file must not be empty")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_args(args)
    except BenchmarkError as exc:
        print(f"ERROR: {exc}", flush=True)
        return 2
    started_utc = utc_now()
    result_errors: list[str] = []
    print("=" * 60)
    print("SANITY SUITE (streaming)")
    print("=" * 60)
    sanity: list[dict[str, Any]] = []
    for name, prompt, max_tokens in SANITY_PROMPTS:
        row = stream_chat(args.base_url, args.model, prompt, max_tokens, args.include_usage, args.timeout)
        row["name"] = name
        sanity.append(row)
        if row.get("errors"):
            result_errors.extend(f"sanity {name}: {error}" for error in row["errors"])
        print(
            f"  [{name}] {row['completion_tokens']} tok in {row['request_seconds']:.2f}s "
            f"→ {row['response_text'][:80]}",
            flush=True,
        )

    repeat_rows: list[dict[str, Any]] = []
    warmup_errors: list[str] = []
    print("\n" + "=" * 60)
    print("CONCURRENCY RAMP (streaming; measured repeats after warmup)")
    print("=" * 60)
    for concurrency in args.concurrency:
        for warmup_index in range(args.warmup):
            warmup_prompt = args.prompt
            if args.unique_prefix_per_repeat:
                warmup_prompt = f"[prefill-warmup-c{concurrency}-r{warmup_index + 1}] {args.prompt}"
            warmup_rows = run_batch(
                concurrency,
                warmup_prompt,
                args.max_tokens,
                args.base_url,
                args.model,
                args.include_usage,
                args.timeout,
            )
            for row in warmup_rows:
                if row.get("errors"):
                    warmup_errors.extend(
                        f"c{concurrency} warmup {warmup_index + 1}: {error}" for error in row["errors"]
                    )
        for repeat in range(1, args.repeats + 1):
            repeat_prompt = args.prompt
            if args.unique_prefix_per_repeat:
                repeat_prompt = f"[prefill-measured-c{concurrency}-r{repeat}] {args.prompt}"
            row = run_repeat(
                repeat,
                concurrency,
                repeat_prompt,
                args.max_tokens,
                args.base_url,
                args.model,
                args.include_usage,
                args.timeout,
                args.telemetry_interval,
            )
            repeat_rows.append(row)
            result_errors.extend(row["errors"])
            print(
                f"  c{concurrency:>2d} r{repeat}: client {row['client_output_tokens_per_second']!s:>8} tok/s | "
                f"server prompt {row['server_prompt_tokens_per_second']!s:>8} tok/s | "
                f"server decode {row['server_decode_tokens_per_second']!s:>8} tok/s | "
                f"{row['requests_ok']}/{concurrency} ok",
                flush=True,
            )

    try:
        models = fetch_json(args.base_url, "/v1/models", timeout=args.timeout)
        model_catalog = _model_catalog(models, args.model)
        if model_catalog.get("error"):
            result_errors.append(str(model_catalog["error"]))
    except BenchmarkError as exc:
        model_catalog = {"count": 0, "ids": [], "error": str(exc)}
        result_errors.append(str(exc))
    try:
        facts = container_log_facts(args.container_name)
    except BenchmarkError as exc:
        facts = {}
        result_errors.append(str(exc))
    try:
        gpu_info = gpu_identity()
    except BenchmarkError as exc:
        gpu_info = []
        result_errors.append(str(exc))

    finished_utc = utc_now()
    result = {
        "schema_version": 3,
        "run_metadata": {
            "started_utc": started_utc,
            "finished_utc": finished_utc,
            "base_url": args.base_url,
            "model": args.model,
            "concurrency_levels": list(args.concurrency),
            "warmup_batches_per_concurrency": args.warmup,
            "measured_repeats_per_concurrency": args.repeats,
            "include_usage": args.include_usage,
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "unique_prefix_per_repeat": args.unique_prefix_per_repeat,
            "timeout_seconds": args.timeout,
            "telemetry_interval_seconds": args.telemetry_interval,
            "container_name": args.container_name,
            "prometheus_counters": dict(PROMETHEUS_COUNTERS),
            "timing_definition": {
                "ttft": "request start to first nonempty SSE token/chunk",
                "itl": "intervals between subsequent nonempty SSE token/chunk arrivals",
                "tpot": "mean of subsequent nonempty SSE arrival intervals",
            },
        },
        "model_catalog": model_catalog,
        "gpu_identity": gpu_info,
        "sanity_suite": sanity,
        "repeat_rows": repeat_rows,
        "ramp": repeat_rows,
        "kv_facts": facts,
        "warmup_errors": warmup_errors,
        "errors": result_errors,
    }
    output = Path(args.output)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        print(error_text(f"failed to write benchmark output {output}", exc), flush=True)
        return 1
    print(f"\nResult saved to: {output}")
    print("\nKV Facts:")
    for key, value in facts.items():
        print(f"  {key}: {value}")
    if result_errors or warmup_errors:
        print(f"\nCompleted with {len(result_errors) + len(warmup_errors)} recorded error(s).", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
