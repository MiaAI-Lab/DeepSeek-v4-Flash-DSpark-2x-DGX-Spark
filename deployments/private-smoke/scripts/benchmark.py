#!/usr/bin/env python3
"""No-retry streaming performance and request-correlation gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time
import urllib.request
import uuid


PROM_COUNTER = "vllm:request_success_total"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        raise RuntimeError("redirects are forbidden in the no-retry benchmark")


OPENER = urllib.request.build_opener(NoRedirect)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def request(base_url: str, key: str, path: str, *, data=None, request_id=None, timeout=900):
    headers = {"Authorization": f"Bearer {key}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if request_id:
        headers["X-Request-Id"] = request_id
    item = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=None if data is None else json.dumps(data).encode(),
        headers=headers,
    )
    return OPENER.open(item, timeout=timeout)


def success_counter(base_url: str, key: str) -> float:
    with request(base_url.removesuffix("/v1"), key, "/metrics", timeout=30) as response:
        text = response.read().decode()
    total = 0.0
    found = False
    for line in text.splitlines():
        if line.startswith(PROM_COUNTER) and not line.startswith("#"):
            total += float(line.rsplit(" ", 1)[1])
            found = True
    if not found:
        raise AssertionError(f"missing Prometheus counter {PROM_COUNTER}")
    return total


def synthetic_prompt(nonce: str) -> str:
    block = (
        "Analyze a fictional warehouse routing policy using only synthetic data. "
        "Compare latency, safety, observability, and rollback constraints. "
    )
    return (
        f"Nonce {nonce}. " + block * 10
        + "Produce a detailed numbered technical assessment. Do not quote the prompt."
    )


def stream_sample(base_url: str, key: str, model: str, request_id: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": synthetic_prompt(request_id)}],
        "max_tokens": 512,
        "temperature": 0.6,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.monotonic()
    first_token = None
    last_token = None
    finish_reason = None
    usage_blocks = []
    stream_ids = set()
    saw_done = False
    with request(base_url, key, "/chat/completions", data=payload, request_id=request_id) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                saw_done = True
                break
            chunk = json.loads(body)
            if chunk.get("id"):
                stream_ids.add(chunk["id"])
            if chunk.get("usage"):
                usage_blocks.append(chunk["usage"])
            for choice in chunk.get("choices") or []:
                content = (choice.get("delta") or {}).get("content")
                if content:
                    now = time.monotonic()
                    first_token = now if first_token is None else first_token
                    last_token = now
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
    ended = time.monotonic()
    if not saw_done or first_token is None or last_token is None:
        raise AssertionError("incomplete stream")
    if len(usage_blocks) != 1:
        raise AssertionError(f"expected one usage block, got {len(usage_blocks)}")
    usage = usage_blocks[0]
    completion_tokens = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    if not isinstance(completion_tokens, int) or completion_tokens < 2:
        raise AssertionError("invalid completion_tokens")
    if not isinstance(prompt_tokens, int) or not (180 <= prompt_tokens <= 360):
        raise AssertionError(f"synthetic prompt is not about 256 tokens: {prompt_tokens}")
    if finish_reason not in {"stop", "length"}:
        raise AssertionError(f"wrong finish_reason: {finish_reason}")
    if len(stream_ids) != 1:
        raise AssertionError("stream chunks have missing or inconsistent IDs")
    decode_seconds = last_token - first_token
    if decode_seconds <= 0:
        raise AssertionError("invalid first_token/last_token interval")
    return {
        "client_request_id": request_id,
        "origin_request_id": next(iter(stream_ids)),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "ttft_seconds": first_token - started,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": (completion_tokens - 1) / decode_seconds,
        "elapsed_seconds": ended - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", choices=("direct", "litellm", "hermes"), required=True)
    parser.add_argument("--base-url", default="http://172.30.0.1:8888/v1")
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.concurrency != 1 or args.samples < 20 or args.warmups < 3:
        raise SystemExit("acceptance requires concurrency=1, at least 3 warmups, and 20 samples")
    key = args.key_file.read_text().strip()
    before = success_counter(args.base_url, key)
    cold = stream_sample(args.base_url, key, args.model, f"dspark-cold-{uuid.uuid4()}")
    warmups = [
        stream_sample(args.base_url, key, args.model, f"dspark-warmup-{uuid.uuid4()}")
        for _ in range(args.warmups)
    ]
    samples = [
        stream_sample(args.base_url, key, args.model, f"dspark-measured-{uuid.uuid4()}")
        for _ in range(args.samples)
    ]
    after = success_counter(args.base_url, key)
    expected_completions = 1 + args.warmups + args.samples
    metric_delta = after - before
    if metric_delta != expected_completions:
        raise AssertionError(
            f"request correlation failed: metric_delta={metric_delta}, expected={expected_completions}"
        )
    ids = {sample["client_request_id"] for sample in [cold, *warmups, *samples]}
    if len(ids) != expected_completions:
        raise AssertionError("client request IDs are not unique")
    median_decode = statistics.median(item["decode_tokens_per_second"] for item in samples)
    p95_ttft = percentile([item["ttft_seconds"] for item in samples], 0.95)
    accepted = median_decode >= 50.0 and p95_ttft <= 5.0
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "layer": args.layer,
        "workload": {"max_tokens": 512, "temperature": 0.6, "top_p": 0.95, "concurrency": 1},
        "cold": cold,
        "discarded_warmups": warmups,
        "samples": samples,
        "summary": {
            "median_decode_tokens_per_second": median_decode,
            "p95_ttft_seconds": p95_ttft,
            "metric_delta": metric_delta,
            "expected_completions": expected_completions,
            "litellm_attempts_expected": 0 if args.layer == "direct" else expected_completions,
            "accepted": accepted,
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        args.output.chmod(0o600)
    else:
        print(rendered, end="")
    if not accepted:
        raise SystemExit("direct performance threshold failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
