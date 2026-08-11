#!/usr/bin/env python3
"""Streaming long-context decode gate for the Issue #22 regression."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import stat
import time
import urllib.request


MIN_POST_FIRST_TOKENS = 64
MIN_DECODE_TPS = 10.0
PROMPT_SUFFIX = (
    "\n\nDesign a failure-isolating validation plan for a two-node distributed "
    "model-serving rollout. Reason through the tradeoffs and provide at least "
    "12 numbered steps, each as a complete sentence, followed by a conclusion. "
    "Do not stop early."
)
STREAM_CHAT_TEMPLATE_KWARGS = {
    "thinking": True,
    "reasoning_effort": "max",
    "drop_thinking": False,
}


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = os.path.expandvars(value)
    return values


def read_key_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("key file must be a regular file, not a symlink")
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError("key file must have mode 0600")
    value = path.read_text().strip()
    if not value:
        raise ValueError("key file is empty")
    return value


def request_json(url: str, key: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def tokenize(base_url: str, key: str, model: str, content: str, timeout: float) -> int:
    response = request_json(
        base_url.rstrip("/").removesuffix("/v1") + "/tokenize",
        key,
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "chat_template_kwargs": {"reasoning_effort": "none", "drop_thinking": False},
            "add_generation_prompt": True,
        },
        timeout,
    )
    count = response.get("count")
    if not isinstance(count, int) or count <= 0:
        raise RuntimeError("tokenize response did not contain a positive count")
    return count


def build_prompt(
    base_url: str,
    key: str,
    model: str,
    target: int,
    timeout: float,
    prompt_nonce: str,
) -> tuple[str, int]:
    prefix = f" dspark-probe-nonce-{prompt_nonce}"
    unit = " dspark-decode-datum"
    repeats = max(1, target // 4)
    content = prefix + unit * repeats + PROMPT_SUFFIX
    observed = 0
    for _ in range(8):
        observed = tokenize(base_url, key, model, content, timeout)
        if target - 64 <= observed <= target + 64:
            return content, observed
        repeats = max(1, int(repeats * target / observed))
        content = prefix + unit * repeats + PROMPT_SUFFIX
    raise RuntimeError(f"could not calibrate long-context prompt: observed={observed}")


def stream_trial(base_url: str, key: str, model: str, prompt: str, timeout: float) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": MIN_POST_FIRST_TOKENS + 1,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": STREAM_CHAT_TEMPLATE_KWARGS,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_at = None
    last_at = None
    usage: dict = {}
    saw_done = False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            value = line[5:].strip()
            if value == "[DONE]":
                saw_done = True
                break
            event = json.loads(value)
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            choices = event.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                    now = time.perf_counter()
                    first_at = first_at or now
                    last_at = now
    if not saw_done or first_at is None or last_at is None:
        raise RuntimeError("stream did not produce timed tokens and [DONE]")
    completion_tokens = usage.get("completion_tokens", 0)
    if not isinstance(completion_tokens, int):
        completion_tokens = 0
    decode_tokens = max(0, completion_tokens - 1)
    decode_seconds = max(0.0, last_at - first_at)
    return {
        "ttft_s": first_at - started,
        "decode_seconds": decode_seconds,
        "completion_tokens": completion_tokens,
        "post_first_tokens": decode_tokens,
        "decode_tps": decode_tokens / decode_seconds if decode_seconds > 0 else 0.0,
    }


def evaluate_decode_gate(trials: list[dict], baseline_tps: float | None = None) -> dict:
    if baseline_tps is not None and (not math.isfinite(baseline_tps) or baseline_tps <= 0):
        raise ValueError("baseline_tps must be finite and greater than zero")
    valid = [trial for trial in trials if trial.get("post_first_tokens", 0) >= MIN_POST_FIRST_TOKENS]
    minimum_tps = min((trial.get("decode_tps", 0.0) for trial in valid), default=0.0)
    checks = {
        "repeated_trials": len(valid) >= 2,
        "at_least_64_post_first_tokens": len(valid) == len(trials) and bool(valid),
        "absolute_decode_tps": minimum_tps >= MIN_DECODE_TPS,
    }
    if baseline_tps is not None:
        checks["five_x_vulnerable_baseline"] = minimum_tps >= baseline_tps * 5
    return {"passed": all(checks.values()), "checks": checks, "minimum_decode_tps": minimum_tps}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path(".env.dspark"))
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--target-prompt-tokens", type=int, default=600_000)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--baseline-tps", type=float)
    parser.add_argument("--evidence-identity")
    parser.add_argument("--prompt-nonce")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.target_prompt_tokens < 600_000:
        parser.error("--target-prompt-tokens must be at least 600000")
    if args.trials < 2:
        parser.error("--trials must be at least 2")
    if args.baseline_tps is not None and (
        not math.isfinite(args.baseline_tps) or args.baseline_tps <= 0
    ):
        parser.error("--baseline-tps must be finite and greater than zero")
    values = parse_env(args.env_file)
    prompt_nonce = args.prompt_nonce or values.get("LONG_CONTEXT_DECODE_PROMPT_NONCE", "")
    if not re.fullmatch(r"[A-Za-z0-9._-]{8,128}", prompt_nonce):
        parser.error(
            "--prompt-nonce or LONG_CONTEXT_DECODE_PROMPT_NONCE must be 8-128 "
            "characters from A-Z, a-z, 0-9, dot, underscore, or hyphen"
        )
    key = read_key_file(args.key_file or Path(values.get("VLLM_ORIGIN_KEY_FILE", "")))
    base_url = args.base_url or f"http://{values.get('VLLM_PROXY_HOST', '172.30.0.1')}:{values.get('VLLM_PROXY_PORT', '8888')}/v1"
    model = args.model or values.get("SERVED_MODEL_NAME", "deepseek-v4-flash-0731")
    prompt, observed = build_prompt(
        base_url, key, model, args.target_prompt_tokens, args.timeout, prompt_nonce
    )
    trials = [stream_trial(base_url, key, model, prompt, args.timeout) for _ in range(args.trials)]
    gate = evaluate_decode_gate(trials, args.baseline_tps)
    report = {
        "schema": "dspark-long-context-decode/v1",
        "model": model,
        "target_prompt_tokens": args.target_prompt_tokens,
        "observed_prompt_tokens": observed,
        "baseline_tps": args.baseline_tps,
        "evidence_identity": args.evidence_identity,
        "prompt_nonce": prompt_nonce,
        "chat_template_kwargs": STREAM_CHAT_TEMPLATE_KWARGS,
        "identical_prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
        "trials": trials,
        "gate": gate,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        args.output.chmod(0o600)
    print(encoded, end="")
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
