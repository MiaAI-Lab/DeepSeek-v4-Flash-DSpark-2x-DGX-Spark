#!/usr/bin/env python3
"""Two-pass vision reasoning for the DSpark stack (0731 + Qwen3-VL sidecar).

Architecture (2026-08-11): DeepSeek-V4-Flash-0731 serves text-only on :8888;
a local Qwen3-VL-4B sidecar on :8889 does the seeing. This script chains:

- Pass 1 (extraction): image -> sidecar VL -> detailed text description
- Pass 2 (reasoning): description -> 0731 with reasoning_effort=max

Usage:
    python3 scripts/vision-reason.py --image photo.jpg \
        --question "Is this home pet-friendly? Reason step by step."
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

EXTRACT_PROMPT = (
    "Describe this image in precise factual detail: people, animals, objects, "
    "colors, positions, actions, visible text, background. Focus on anything "
    "relevant to this question: {question}"
)


def chat(base_url: str, model: str, messages: list, *, thinking, max_tokens: int,
         temperature: float) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": thinking,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
    choice = resp["choices"][0]
    content = (choice["message"].get("content") or "").strip()
    if not content:
        raise RuntimeError(
            f"empty content (finish_reason={choice.get('finish_reason')}, "
            f"completion_tokens={resp['usage']['completion_tokens']})"
        )
    return content


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True, help="image file path (jpeg/png)")
    ap.add_argument("--question", required=True, help="question to reason about")
    ap.add_argument("--base-url", default="http://127.0.0.1:8888",
                    help="reasoning model endpoint (DeepSeek 0731)")
    ap.add_argument("--model", default="deepseek-v4-flash-0731")
    ap.add_argument("--extract-base-url", default="http://127.0.0.1:8889",
                    help="vision extractor endpoint (VL sidecar)")
    ap.add_argument("--extract-model", default="qwen3-vl-4b")
    ap.add_argument("--extract-tokens", type=int, default=800)
    ap.add_argument("--reason-tokens", type=int, default=8000)
    ap.add_argument("--temperature", type=float, default=0.6, help="pass 2 only")
    ap.add_argument("--show-description", action="store_true",
                    help="print the pass-1 description to stdout before the answer")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.is_file():
        ap.error(f"image not found: {img_path}")
    mime = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(img_path.read_bytes()).decode()

    # Pass 1 — extraction on the VL sidecar (Qwen3-VL; enable_thinking off)
    desc = chat(
        args.extract_base_url, args.extract_model,
        [{"role": "user", "content": [
            {"type": "text", "text": EXTRACT_PROMPT.format(question=args.question)},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]}],
        thinking={"enable_thinking": False},
        max_tokens=args.extract_tokens,
        temperature=0.0,
    )
    logger.info("pass 1 (extract): %d chars", len(desc))
    if args.show_description:
        print(f"--- description ---\n{desc}\n--- answer ---")

    # Pass 2 — max reasoning over the description (text-only: stable)
    answer = chat(
        args.base_url, args.model,
        [{"role": "user", "content":
          f"Here is a description of an image:\n\"{desc}\"\n\n"
          f"Based on this description, answer the following. {args.question}"}],
        thinking={"thinking": True, "reasoning_effort": "max"},
        max_tokens=args.reason_tokens,
        temperature=args.temperature,
    )
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
