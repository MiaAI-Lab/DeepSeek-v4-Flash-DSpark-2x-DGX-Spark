"""MCP server: describe / OCR / compare via local Qwen3-VL sidecar (:8889)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .images import VisionToolError, load_image_as_data_uri, load_many

DEFAULT_BASE_URL = "http://127.0.0.1:8889"
DEFAULT_MODEL = "qwen3-vl-4b"

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

mcp = FastMCP(
    "ds4f-vision",
    instructions=(
        "Local vision tools for DeepSeek-V4-Flash-0731. Call describe_image "
        "(or ocr_image / compare_images) with a filesystem path or URL; the "
        "Qwen3-VL sidecar returns text the reasoning model can use. Do not "
        "switch models for vision."
    ),
)


def _cfg() -> tuple[str, str, int]:
    base = os.environ.get("DSPARK_VL_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    model = os.environ.get("DSPARK_VL_MODEL", DEFAULT_MODEL)
    tokens = int(os.environ.get("DSPARK_VL_MAX_TOKENS", "1024"))
    return base, model, tokens


def _sidecar_chat(content_parts: list[dict], *, max_tokens: int | None = None) -> str:
    base, model, default_tokens = _cfg()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content_parts}],
        "max_tokens": max_tokens or default_tokens,
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
        # Retry once with aggressive downscale is handled at load time; here
        # surface actionable sidecar errors.
        raise VisionToolError(
            f"vision sidecar HTTP {exc.code} at {url}. "
            f"Is Qwen3-VL up on :8889? Detail: {detail or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise VisionToolError(
            f"vision sidecar unreachable at {url} ({exc.reason}). "
            "Start the stack with ./start-deepseek-v4-flash-dspark.sh "
            "(ENABLE_VL_SIDECAR=1) and wait until :8889 /v1/models lists qwen3-vl-4b."
        ) from exc
    except TimeoutError as exc:
        raise VisionToolError(
            f"vision sidecar timed out at {url}. Try a smaller image or retry."
        ) from exc

    try:
        choice = body["choices"][0]
        text = (choice["message"].get("content") or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise VisionToolError(f"unexpected sidecar response: {body!r}") from exc
    if not text:
        fr = choice.get("finish_reason")
        raise VisionToolError(f"sidecar returned empty content (finish_reason={fr})")
    return text


def _run(fn):
    try:
        return fn()
    except VisionToolError as exc:
        return f"Error: {exc}"


@mcp.tool()
def describe_image(path_or_url: str, question: Optional[str] = None) -> str:
    """Describe an image via the local Qwen3-VL sidecar (port 8889).

    Use this whenever the user references a local image path or URL and you
    need visual facts. Pass the user's question so the description focuses on
    what matters; then reason over the returned text yourself (0731 stays on
    text-only input — do not switch models).

    Args:
        path_or_url: Absolute filesystem path or http(s) URL to an image.
        question: Optional focus question (e.g. sweater color / setting).
    """

    def _inner() -> str:
        uri = load_image_as_data_uri(path_or_url)
        q = (question or "Describe everything important in the image.").strip()
        return _sidecar_chat(
            [
                {"type": "text", "text": EXTRACT_PROMPT.format(question=q)},
                {"type": "image_url", "image_url": {"url": uri}},
            ]
        )

    return _run(_inner)


@mcp.tool()
def ocr_image(path_or_url: str) -> str:
    """Extract visible text from an image via the local Qwen3-VL sidecar.

    Args:
        path_or_url: Absolute filesystem path or http(s) URL to an image.
    """

    def _inner() -> str:
        uri = load_image_as_data_uri(path_or_url)
        return _sidecar_chat(
            [
                {"type": "text", "text": OCR_PROMPT},
                {"type": "image_url", "image_url": {"url": uri}},
            ]
        )

    return _run(_inner)


@mcp.tool()
def compare_images(paths: list[str], question: str) -> str:
    """Compare up to 4 images via the local Qwen3-VL sidecar.

    Args:
        paths: List of absolute paths or http(s) URLs (max 4).
        question: What to compare / decide.
    """

    def _inner() -> str:
        if not question or not str(question).strip():
            raise VisionToolError("question is required for compare_images")
        uris = load_many(list(paths))
        parts: list[dict] = [
            {
                "type": "text",
                "text": COMPARE_PROMPT.format(question=question.strip()),
            }
        ]
        for i, uri in enumerate(uris, 1):
            parts.append({"type": "text", "text": f"Image {i}:"})
            parts.append({"type": "image_url", "image_url": {"url": uri}})
        return _sidecar_chat(parts, max_tokens=max(_cfg()[2], 1200))

    return _run(_inner)


def main() -> None:
    # stdio transport for pi-mcp-adapter / uvx
    mcp.run()


if __name__ == "__main__":
    main()
