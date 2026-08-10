#!/usr/bin/env python3
"""N-trial solid-color QA gate for native MoonViT vision under DSpark.

Sends N identical forced-choice requests per solid-color fixture at
temperature=0 and requires per-color pass rates (pixel dependence):
  red   >= --red-threshold   (default 0.90; synonyms red/crimson/scarlet)
  black >= --threshold       (default 0.80)
  white >= --threshold
  hue (green|blue) >= --hue-threshold (default 0.50; reported honestly)

Also asserts text-only VISION_TEXT_OK on the same endpoint and captures
DSpark spec-decode counters (non-zero acceptance required when --check-dspark).

Exit 0 = all gates pass; 1 = gate failure; 2 = transport/usage error.
Writes a machine-readable status JSON (default results/smoke-mm-status.json).
"""

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

COLOR_PROMPT = (
    "What color is this solid image? One word: red/green/blue/black/white."
)

FIXTURES: dict[str, tuple[int, int, int]] = {
    "red": (255, 0, 0),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
}

# Accepted answer tokens per fixture (goal: red/crimson/scarlet for red).
SYNONYMS: dict[str, tuple[str, ...]] = {
    "red": ("red", "crimson", "scarlet"),
    "black": ("black",),
    "white": ("white",),
    "green": ("green",),
    "blue": ("blue",),
}


def fixture_png_b64(rgb: tuple[int, int, int], size: int) -> str:
    from PIL import Image

    img = Image.new("RGB", (size, size), color=rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def post_json(url: str, payload: dict, timeout: float = 300.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_content(resp: dict) -> str:
    msg = resp["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return (content or reasoning or "").strip()


def answer_matches(color: str, text: str) -> bool:
    lower = text.lower()
    return any(syn in lower for syn in SYNONYMS[color])


def spec_decode_snapshot(base_url: str) -> dict:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/metrics", timeout=15) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception as exc:  # metrics unreachable — report, don't gate here
        return {"error": str(exc)}
    accepted = drafted = None
    per_pos: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            accepted = float(line.rsplit(" ", 1)[-1])
        elif line.startswith("vllm:spec_decode_num_draft_tokens_total"):
            drafted = float(line.rsplit(" ", 1)[-1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total"):
            pos = line.split('position="', 1)[-1].split('"', 1)[0]
            per_pos[pos] = float(line.rsplit(" ", 1)[-1])
    out = {"accepted_tokens": accepted, "draft_tokens": drafted, "per_pos": per_pos}
    if accepted and drafted:
        out["accept_rate"] = round(accepted / drafted, 4) if drafted else None
    return out


def run_color_trials(
    url: str,
    model: str,
    color: str,
    trials: int,
    size: int,
    max_tokens: int,
    delay_s: float,
) -> dict:
    b64 = fixture_png_b64(FIXTURES[color], size)
    parts = [
        {"type": "text", "text": COLOR_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]
    answers: list[str] = []
    hits = 0
    errors = 0
    for i in range(trials):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": parts}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"thinking": False},
        }
        try:
            resp = post_json(url, payload)
            text = extract_content(resp)
        except Exception as exc:
            errors += 1
            answers.append(f"ERROR: {exc}")
            continue
        answers.append(text)
        hits += int(answer_matches(color, text))
        if delay_s and i + 1 < trials:
            time.sleep(delay_s)
    rate = hits / max(1, trials)
    return {
        "trials": trials,
        "hits": hits,
        "errors": errors,
        "pass_rate": round(rate, 3),
        "answers": answers,
    }


def run_text_check(url: str, model: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: VISION_TEXT_OK"}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"thinking": False},
    }
    resp = post_json(url, payload)
    text = extract_content(resp)
    return {"ok": "VISION_TEXT_OK" in text, "content": text}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8888")
    ap.add_argument("--model", default="deepseek-v4-flash-0731-vision")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--colors", default="red,black,white,green,blue")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--red-threshold", type=float, default=0.90)
    ap.add_argument("--hue-threshold", type=float, default=0.50)
    ap.add_argument("--delay-s", type=float, default=0.0)
    ap.add_argument("--check-dspark", action="store_true", default=True)
    ap.add_argument("--no-check-dspark", dest="check_dspark", action="store_false")
    ap.add_argument("--skip-text-check", action="store_true")
    ap.add_argument("--out", default="results/smoke-mm-status.json")
    args = ap.parse_args()

    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    colors = [c.strip() for c in args.colors.split(",") if c.strip()]
    for c in colors:
        if c not in FIXTURES:
            print(f"unknown color {c!r}; known: {sorted(FIXTURES)}", file=sys.stderr)
            return 2

    spec_before = spec_decode_snapshot(args.base_url) if args.check_dspark else None

    results: dict[str, dict] = {}
    failures: list[str] = []

    for color in colors:
        r = run_color_trials(
            url, args.model, color, args.trials, args.size, args.max_tokens, args.delay_s
        )
        results[color] = r
        threshold = (
            args.red_threshold
            if color == "red"
            else (args.hue_threshold if color in ("green", "blue") else args.threshold)
        )
        r["threshold"] = threshold
        r["ok"] = r["pass_rate"] >= threshold and r["errors"] == 0
        print(
            f"{color:6s} {r['hits']}/{r['trials']} "
            f"(rate={r['pass_rate']:.2f} threshold={threshold:.2f}) "
            f"answers={r['answers']}",
            flush=True,
        )
        if not r["ok"]:
            failures.append(
                f"{color}: pass_rate {r['pass_rate']:.2f} < {threshold:.2f}"
                + (f" ({r['errors']} transport errors)" if r["errors"] else "")
            )

    text_check = None
    if not args.skip_text_check:
        try:
            text_check = run_text_check(url, args.model, max(args.max_tokens, 32))
        except Exception as exc:
            text_check = {"ok": False, "error": str(exc)}
        print(f"text-only VISION_TEXT_OK: {text_check}", flush=True)
        if not text_check.get("ok"):
            failures.append("text-only check failed (VISION_TEXT_OK missing)")

    spec_after = spec_decode_snapshot(args.base_url) if args.check_dspark else None
    dspark_ok = None
    if args.check_dspark and spec_after:
        acc = (spec_after or {}).get("accepted_tokens")
        dspark_ok = acc is not None and acc > 0
        if not dspark_ok:
            failures.append(f"DSpark acceptance collapsed: {spec_after}")

    status = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "base_url": args.base_url,
        "protocol": {
            "prompt": COLOR_PROMPT,
            "temperature": 0,
            "image_size": args.size,
            "content_order": "text_then_image",
            "thinking": False,
        },
        "colors": results,
        "text_only": text_check,
        "dspark": {"before": spec_before, "after": spec_after, "ok": dspark_ok},
        "ok": not failures,
        "failures": failures,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")

    if failures:
        print("FAIL: " + "; ".join(failures), file=sys.stderr)
        return 1
    print("PASS: all color/text/DSpark gates green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
