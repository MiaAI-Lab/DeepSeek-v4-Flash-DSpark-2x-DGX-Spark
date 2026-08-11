"""Prime Agent skill: call the local Qwen3-VL sidecar (no MCP HTTP required)."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_BASE_URL = "http://127.0.0.1:8889"
DEFAULT_MODEL = "qwen3-vl-4b"
MAX_EDGE = 2048
MAX_COMPARE = 4

EXTRACT_PROMPT = (
    "Describe this image in precise factual detail: people, animals, objects, "
    "colors, positions, actions, visible text, background. Focus on anything "
    "relevant to this question: {question}"
)
OCR_PROMPT = (
    "Extract all visible text from this image. Preserve reading order and line "
    "breaks when possible. If there is no text, say so briefly. Do not describe "
    "the scene beyond what is needed to locate the text."
)
COMPARE_PROMPT = (
    "Compare these images carefully. Note similarities and differences in "
    "subjects, colors, layout, text, and setting. Focus on anything relevant "
    "to this question: {question}"
)


def _cfg() -> tuple[str, str, int]:
    # Optional sidecar.env written by install_harnesses.py next to this package.
    env_file = Path(__file__).resolve().parents[2] / "sidecar.env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())
    base = os.environ.get("DSPARK_VL_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    model = os.environ.get("DSPARK_VL_MODEL", DEFAULT_MODEL)
    tokens = int(os.environ.get("DSPARK_VL_MAX_TOKENS", "1024"))
    return base, model, tokens


def _downscale(raw: bytes, mime: str) -> tuple[bytes, str]:
    try:
        from PIL import Image
    except ImportError:
        return raw, mime
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:  # noqa: BLE001
        return raw, mime
    w, h = img.size
    if max(w, h) <= MAX_EDGE:
        return raw, mime
    scale = MAX_EDGE / float(max(w, h))
    img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), "image/jpeg"


def _load_data_uri(path_or_url: str) -> str:
    parsed = urlparse(path_or_url)
    if parsed.scheme in ("http", "https"):
        with urllib.request.urlopen(path_or_url, timeout=60) as resp:
            raw = resp.read()
            mime = resp.headers.get_content_type() or "image/jpeg"
    else:
        path = Path(path_or_url).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"image file not found: {path_or_url}. Pass an absolute path or http(s) URL."
            )
        raw = path.read_bytes()
        guessed, _ = mimetypes.guess_type(str(path))
        mime = guessed if guessed and guessed.startswith("image/") else "image/jpeg"
    raw, mime = _downscale(raw, mime)
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _sidecar_chat_sync(content_parts: list[dict[str, Any]]) -> str:
    base, model, tokens = _cfg()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content_parts}],
        "max_tokens": tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = f"{base}/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(
            f"vision sidecar HTTP {exc.code} at {url}. "
            f"Is Qwen3-VL up on :8889? Detail: {detail or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"vision sidecar unreachable at {url} ({exc.reason}). "
            "Start ./start-deepseek-v4-flash-dspark.sh with ENABLE_VL_SIDECAR=1."
        ) from exc
    try:
        text = (body["choices"][0]["message"].get("content") or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected sidecar response: {body!r}") from exc
    if not text:
        raise RuntimeError("vision sidecar returned empty content")
    return text


async def describe_image(path_or_url: str, question: str = "What is in this image?") -> str:
    """Detailed factual description; optional question focuses the extract."""
    uri = await asyncio.to_thread(_load_data_uri, path_or_url)
    parts = [
        {"type": "image_url", "image_url": {"url": uri}},
        {"type": "text", "text": EXTRACT_PROMPT.format(question=question)},
    ]
    return await asyncio.to_thread(_sidecar_chat_sync, parts)


async def ocr_image(path_or_url: str) -> str:
    """Extract visible text from an image."""
    uri = await asyncio.to_thread(_load_data_uri, path_or_url)
    parts = [
        {"type": "image_url", "image_url": {"url": uri}},
        {"type": "text", "text": OCR_PROMPT},
    ]
    return await asyncio.to_thread(_sidecar_chat_sync, parts)


async def compare_images(paths: list[str], question: str = "What differs?") -> str:
    """Compare up to 4 images."""
    if not paths:
        raise ValueError("paths must be a non-empty list")
    if len(paths) > MAX_COMPARE:
        raise ValueError(f"too many images ({len(paths)}); limit is {MAX_COMPARE}")
    parts: list[dict[str, Any]] = []
    for p in paths:
        uri = await asyncio.to_thread(_load_data_uri, p)
        parts.append({"type": "image_url", "image_url": {"url": uri}})
    parts.append({"type": "text", "text": COMPARE_PROMPT.format(question=question)})
    return await asyncio.to_thread(_sidecar_chat_sync, parts)


async def run(path_or_url: str, question: str = "What is in this image?") -> str:
    """Default skill entry — same as describe_image."""
    return await describe_image(path_or_url, question=question)
