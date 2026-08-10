#!/usr/bin/env python3
"""Native multimodal OpenAI chat smoke against DSpark vLLM /v1/chat/completions."""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def make_fixture_png(path: Path | None = None) -> bytes:
    """Solid red 64x64 PNG — answers should mention red if vision works."""
    try:
        from PIL import Image
    except ImportError:
        # Minimal 1x1 red PNG
        return base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        )
    # Pure red solid — color QA is more reliable at ≥256 and with pure (255,0,0).
    img = Image.new("RGB", (256, 256), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    if path:
        path.write_bytes(data)
    return data


def post_json(url: str, payload: dict, timeout: float = 300.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_content(resp: dict) -> str:
    try:
        msg = resp["choices"][0]["message"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"unexpected response: {resp}") from exc
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return (content or reasoning or "").strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=os_env("VLLM_BASE_URL", "http://127.0.0.1:8888"))
    ap.add_argument("--model", default=os_env("SERVED_MODEL_NAME", "deepseek-v4-flash-0731-vision"))
    ap.add_argument("--image", default=None, help="Path to image (default: synthetic red)")
    ap.add_argument("--out", default=None, help="Write full JSON response")
    ap.add_argument("--text-only", action="store_true")
    ap.add_argument("--multiturn", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--thinking", default="false", choices=["false", "true", "low", "high"])
    args = ap.parse_args()

    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    chat_kwargs: dict = {"thinking": False}
    if args.thinking == "true":
        chat_kwargs = {"thinking": True, "reasoning_effort": "low"}
    elif args.thinking in ("low", "high"):
        chat_kwargs = {"thinking": True, "reasoning_effort": args.thinking}

    if args.text_only:
        payload = {
            "model": args.model,
            "messages": [
                {"role": "user", "content": "Reply with exactly: VISION_TEXT_OK"}
            ],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "chat_template_kwargs": chat_kwargs,
        }
        resp = post_json(url, payload)
        text = extract_content(resp)
        print(text)
        if args.out:
            Path(args.out).write_text(json.dumps(resp, indent=2), encoding="utf-8")
        return 0 if text else 1

    if args.image:
        raw = Path(args.image).read_bytes()
    else:
        raw = make_fixture_png()
    b64 = base64.b64encode(raw).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    if args.multiturn:
        # Image first (history-before-image confuses solid-color QA on this stack),
        # then text follow-up that still depends on the prior image turn.
        color_q = (
            "What color is this solid image? "
            "One word: red/green/blue/black/white."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": color_q},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
        r1 = post_json(
            url,
            {
                "model": args.model,
                "messages": messages,
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "chat_template_kwargs": chat_kwargs,
            },
        )
        a1 = extract_content(r1)
        messages.append({"role": "assistant", "content": a1})
        messages.append(
            {
                "role": "user",
                "content": "Confirm the color you just named in one word only.",
            }
        )
        r2 = post_json(
            url,
            {
                "model": args.model,
                "messages": messages,
                "max_tokens": 32,
                "temperature": 0,
                "chat_template_kwargs": chat_kwargs,
            },
        )
        a2 = extract_content(r2)
        out = {"turn1": r1, "turn2": r2, "answers": [a1, a2]}
        print(json.dumps({"answers": [a1, a2]}, indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        ok = bool(a1 and a2)
        if ok and "red" not in a1.lower():
            print(
                f"WARN: multiturn color answer {a1!r} may miss red fixture",
                file=sys.stderr,
            )
        return 0 if ok else 1

    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "What color is this solid image? "
                            "One word: red/green/blue/black/white."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "chat_template_kwargs": chat_kwargs,
    }
    t0 = time.time()
    try:
        resp = post_json(url, payload)
    except urllib.error.URLError as exc:
        print(f"request failed: {exc}", file=sys.stderr)
        return 2
    dt = time.time() - t0
    text = extract_content(resp)
    print(f"latency_s={dt:.2f}")
    print(f"content={text!r}")
    if args.out:
        Path(args.out).write_text(json.dumps(resp, indent=2), encoding="utf-8")
    # Soft image-dependence check for red fixture
    lower = text.lower()
    color_hit = any(w in lower for w in ("red", "crimson", "scarlet", "maroon", "pink"))
    if not text:
        print("FAIL: empty content", file=sys.stderr)
        return 1
    if not color_hit:
        print(
            "WARN: answer may not reflect red fixture pixels; still non-empty",
            file=sys.stderr,
        )
    return 0


def os_env(key: str, default: str) -> str:
    import os

    return os.environ.get(key, default)


if __name__ == "__main__":
    raise SystemExit(main())
