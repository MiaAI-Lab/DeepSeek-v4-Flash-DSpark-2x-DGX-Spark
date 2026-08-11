"""Image loading / encoding helpers for the VL sidecar."""

from __future__ import annotations

import base64
import io
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

# Qwen3-VL sidecar limit in compose; keep MCP aligned.
MAX_IMAGES = 4
# Downscale when either side exceeds this (or encoded payload is huge).
MAX_SIDE = 2048
# Soft ceiling for base64 payload before we force another downscale pass (~6 MiB raw).
MAX_RAW_BYTES = 6 * 1024 * 1024

MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


class VisionToolError(Exception):
    """User-facing tool error (returned as text, not a crash)."""


def _guess_mime(path: Path, data: bytes) -> str:
    mime = MIME_BY_SUFFIX.get(path.suffix.lower())
    if mime:
        return mime
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed and guessed.startswith("image/"):
        return guessed
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _downscale(data: bytes, mime: str) -> tuple[bytes, str]:
    """Resize / re-encode oversized images so the sidecar accepts them."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # noqa: BLE001 — surface as tool error
        raise VisionToolError(f"could not decode image: {exc}") from exc

    w, h = img.size
    needs = max(w, h) > MAX_SIDE or len(data) > MAX_RAW_BYTES
    if not needs:
        return data, mime

    scale = min(MAX_SIDE / max(w, h), 1.0)
    # Extra shrink when file is still huge after dimension clamp.
    if len(data) > MAX_RAW_BYTES:
        scale = min(scale, (MAX_RAW_BYTES / max(len(data), 1)) ** 0.5)

    if scale < 1.0:
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    out = io.BytesIO()
    # Prefer JPEG for bulk; keep PNG for graphics with alpha when small enough.
    if mime == "image/png" and len(data) <= MAX_RAW_BYTES and max(img.size) <= MAX_SIDE:
        img.save(out, format="PNG", optimize=True)
        return out.getvalue(), "image/png"

    if img.mode == "L":
        img = img.convert("RGB")
    quality = 85
    while True:
        out.seek(0)
        out.truncate(0)
        img.save(out, format="JPEG", quality=quality, optimize=True)
        encoded = out.getvalue()
        if len(encoded) <= MAX_RAW_BYTES or quality <= 50:
            return encoded, "image/jpeg"
        quality -= 10


def load_image_as_data_uri(path_or_url: str) -> str:
    """Load a local path or http(s) URL into a data: URI (possibly downscaled)."""
    raw = (path_or_url or "").strip()
    if not raw:
        raise VisionToolError("path_or_url is empty")

    if raw.startswith("data:image/"):
        return raw

    parsed = urlparse(raw)
    if parsed.scheme in ("http", "https"):
        try:
            req = urllib.request.Request(
                raw,
                headers={"User-Agent": "ds4f-vision-mcp/0.1"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
                ctype = (resp.headers.get_content_type() or "").strip()
        except urllib.error.HTTPError as exc:
            raise VisionToolError(
                f"failed to fetch image URL ({exc.code}): {raw}"
            ) from exc
        except urllib.error.URLError as exc:
            raise VisionToolError(
                f"failed to fetch image URL: {raw} ({exc.reason})"
            ) from exc
        mime = ctype if ctype.startswith("image/") else "image/jpeg"
        name = Path(parsed.path).name or "remote.jpg"
        data, mime = _downscale(data, mime if mime.startswith("image/") else _guess_mime(Path(name), data))
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"

    # Treat as local filesystem path (expand ~).
    path = Path(raw).expanduser()
    if not path.exists():
        raise VisionToolError(
            f"image file not found: {path}. Pass an absolute path or http(s) URL."
        )
    if not path.is_file():
        raise VisionToolError(f"not a file: {path}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise VisionToolError(f"cannot read image file {path}: {exc}") from exc
    if not data:
        raise VisionToolError(f"image file is empty: {path}")

    mime = _guess_mime(path, data)
    data, mime = _downscale(data, mime)
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def load_many(paths: list[str]) -> list[str]:
    if not paths:
        raise VisionToolError("at least one image path/URL is required")
    if len(paths) > MAX_IMAGES:
        raise VisionToolError(
            f"too many images ({len(paths)}); sidecar limit is {MAX_IMAGES} per request"
        )
    return [load_image_as_data_uri(p) for p in paths]
