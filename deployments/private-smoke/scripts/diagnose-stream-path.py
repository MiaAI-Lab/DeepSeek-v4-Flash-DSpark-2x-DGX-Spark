#!/usr/bin/env python3
"""Compare one streaming workload over vLLM loopback and the auth proxy."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time
import urllib.request
import uuid


def synthetic_prompt(nonce: str) -> str:
    block = (
        "Analyze a fictional warehouse routing policy using only synthetic data. "
        "Compare latency, safety, observability, and rollback constraints. "
    )
    return (
        f"Nonce {nonce}. " + block * 10
        + "Use only fictional warehouse vocabulary. Return exactly 128 numbered "
        "lowercase English words, then stop. Do not quote the prompt."
    )


def make_payload(model: str, nonce: str) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": synthetic_prompt(nonce)}],
        "max_tokens": 512,
        "temperature": 0.6,
        "top_p": 0.95,
        "chat_template_kwargs": {"thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def stream_sample(base_url: str, key: str, payload: dict, request_id: str) -> dict:
    item = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "X-Request-Id": request_id,
        },
    )
    started = time.monotonic()
    first_token = None
    last_token = None
    finish_reason = None
    usage = None
    saw_done = False
    with urllib.request.urlopen(item, timeout=900) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                saw_done = True
                break
            event = json.loads(body)
            if event.get("usage"):
                if usage is not None:
                    raise AssertionError("multiple usage blocks")
                usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                generated = (
                    delta.get("reasoning")
                    or delta.get("reasoning_content")
                    or delta.get("content")
                )
                if generated:
                    now = time.monotonic()
                    first_token = now if first_token is None else first_token
                    last_token = now
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
    ended = time.monotonic()
    if not saw_done or first_token is None or last_token is None or usage is None:
        raise AssertionError("incomplete stream")
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(completion_tokens, int) or completion_tokens < 2:
        raise AssertionError("invalid completion token count")
    decode_seconds = last_token - first_token
    if decode_seconds <= 0:
        raise AssertionError("invalid decode interval")
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "ttft_seconds": first_token - started,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": (completion_tokens - 1) / decode_seconds,
        "elapsed_seconds": ended - started,
    }


def summarize(samples: list[dict]) -> dict:
    return {
        "median_decode_tokens_per_second": statistics.median(
            item["decode_tokens_per_second"] for item in samples
        ),
        "median_ttft_seconds": statistics.median(item["ttft_seconds"] for item in samples),
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-base-url", default="http://127.0.0.1:8889/v1")
    parser.add_argument("--proxy-base-url", default="http://172.30.0.1:8888/v1")
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmups < 1 or args.samples < 3:
        raise SystemExit("diagnosis requires at least one warmup and three paired samples")

    key = args.key_file.read_text().strip()
    paths = {
        "loopback": args.direct_base_url,
        "authenticated_proxy": args.proxy_base_url,
    }

    # Warm each path symmetrically. The same immutable payload is serialized for
    # both paths; only the URL and transport-level request ID differ.
    for index in range(args.warmups):
        payload = make_payload(args.model, f"warmup-{index}-{uuid.uuid4()}")
        for name in paths:
            stream_sample(
                paths[name], key, deepcopy(payload), f"diagnostic-{name}-{uuid.uuid4()}"
            )

    samples = {name: [] for name in paths}
    for index in range(args.samples):
        payload = make_payload(args.model, f"pair-{index}-{uuid.uuid4()}")
        # Alternate order so thermal drift and cache warmth do not favor a path.
        order = list(paths) if index % 2 == 0 else list(reversed(paths))
        for name in order:
            samples[name].append(
                stream_sample(
                    paths[name], key, deepcopy(payload), f"diagnostic-{name}-{uuid.uuid4()}"
                )
            )

    summaries = {name: summarize(items) for name, items in samples.items()}
    direct = summaries["loopback"]["median_decode_tokens_per_second"]
    proxied = summaries["authenticated_proxy"]["median_decode_tokens_per_second"]
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "workload": {
            "max_tokens": 512,
            "temperature": 0.6,
            "top_p": 0.95,
            "thinking": "off",
            "paired_payloads": True,
            "alternating_path_order": True,
        },
        "paths": summaries,
        "comparison": {
            "loopback_to_proxy_decode_ratio": direct / proxied,
            "proxy_is_primary_bottleneck": direct >= 50.0 and direct / proxied >= 1.15,
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        args.output.chmod(0o600)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
