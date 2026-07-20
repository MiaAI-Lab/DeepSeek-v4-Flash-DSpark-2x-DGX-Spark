#!/usr/bin/env python3
"""Deterministic long-context retrieval gate for an OpenAI-compatible endpoint."""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888")
    parser.add_argument("--model", default="deepseek-v4-flash-dspark")
    parser.add_argument("--words", type=int, default=90000)
    parser.add_argument("--minimum-prompt-tokens", type=int, default=80000)
    parser.add_argument("--nonce", default="default", help="prefix nonce used to defeat prefix-cache reuse")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def build_prompt(words: int, nonce: str, needle: str) -> str:
    vocabulary = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta")
    filler = " ".join(vocabulary[index % len(vocabulary)] for index in range(words))
    return (
        f"Retrieval run nonce: {nonce}. The secret retrieval code is {needle}. Remember it exactly.\n"
        f"Unrelated filler begins: {filler}\n"
        "Return only the secret retrieval code stated at the beginning."
    )


def response_text_and_prompt_tokens(response: dict) -> tuple[str, int]:
    if not response:
        return "", 0
    message = response.get("choices", [{}])[0].get("message", {})
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    text = str(content or reasoning)
    prompt_tokens = int(response.get("usage", {}).get("prompt_tokens", 0))
    return text, prompt_tokens


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.words < 1 or args.minimum_prompt_tokens < 1:
        build_parser().error("--words and --minimum-prompt-tokens must be positive")

    needle = "ORBIT-NVFP4-74291"
    prompt = build_prompt(args.words, args.nonce, needle)
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 512,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    error = None
    response: dict = {}
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as handle:
            response = json.load(handle)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - started
    text, prompt_tokens = response_text_and_prompt_tokens(response)
    errors: list[str] = []
    if error:
        errors.append(error)
    if needle not in text:
        errors.append("needle missing from response")
    if prompt_tokens < args.minimum_prompt_tokens:
        errors.append(
            f"prompt token count {prompt_tokens} below required minimum {args.minimum_prompt_tokens}"
        )
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not errors else "FAIL",
        "model": args.model,
        "synthetic_words": args.words,
        "nonce": args.nonce,
        "prompt_tokens": prompt_tokens,
        "minimum_prompt_tokens": args.minimum_prompt_tokens,
        "elapsed_seconds": elapsed,
        "needle": needle,
        "response_text": text,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
