"""Official DeepSeek-V4-Flash-Vision-Exp image preprocessor.

Adapted from deepseek-ai/DeepSeek-V4-Flash-Vision-Exp ``inference/image_processor.py``
(MIT). Adds a PIL entry point so vLLM can feed decoded images without going
back through URLs. Video is not part of this checkpoint: GIF is decoded as a
still RGB frame (PIL's first frame).
"""
from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Sequence, TYPE_CHECKING
from urllib.request import urlopen

if TYPE_CHECKING:
    import torch
    from PIL import Image as PILImage


def _pil():
    """Load Pillow on first image decode. CPU CI does not install it."""
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise ModuleNotFoundError(
            "Pillow is required to decode Vision-Exp images"
        ) from exc
    return Image, ImageOps


IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
COMPRESS_PAD_TO = 4
IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"
# Vision-Exp added-token id (vocab_size 129280). In-vocab, so hash MoE
# layers 0–2 look up tid2eid[129264] unless image rows skip the table (#175).
IMAGE_TOKEN_ID = 129264
# Encoder-cache identity suffix. Layout length depends on start_pos % 4;
# vLLM's mm hash is image bytes only (issue #172).
LAYOUT_HASH_TAG = "vlexp-ntok"


def compress_pad_tokens(start_pos: int) -> int:
    """IMAGE_PAD tokens prepended so the image block starts on a 4-token boundary."""
    return COMPRESS_PAD_TO - 1 - int(start_pos) % COMPRESS_PAD_TO


def image_block_num_tokens(n_llm_h: int, n_llm_w: int, start_pos: int) -> int:
    """LLM tokens in one N-layout block (start/end, row newlines, compress pad).

    A 40×19 ViT grid (patch 14, downsample 3) is a 7×14 LLM grid: 122 plus
    compress_pad 0–3 → 122–125. Issue #172 was 125 placeholders vs 124 embeds.
    """
    n_llm_h = int(n_llm_h)
    n_llm_w = int(n_llm_w)
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = (rows // 2 * row_len % 2) * 2
    core = n_llm_h * row_len + row_len * pad_h
    return compress_pad_tokens(start_pos) + 1 + core + pad_last + 1


def salt_mm_image_hash(digest: str, num_tokens: int) -> str:
    """Fold block length into vLLM's content-only mm hash (issue #172)."""
    return f"{digest}:{LAYOUT_HASH_TAG}{int(num_tokens)}"


def token_routing_kind(input_tokens: Any, image_token_id: int = IMAGE_TOKEN_ID) -> str:
    """Classify a token row as text-only, image-only, or mixed (issue #175)."""
    if input_tokens is None:
        return "text"
    if hasattr(input_tokens, "reshape") and hasattr(input_tokens, "numel"):
        flat = input_tokens.reshape(-1)
        n = int(flat.numel())
        if n == 0:
            return "text"
        n_img = int((flat == int(image_token_id)).sum().item())
    else:
        seq = list(input_tokens)
        n = len(seq)
        if n == 0:
            return "text"
        n_img = sum(1 for tok in seq if int(tok) == int(image_token_id))
    if n_img == 0:
        return "text"
    if n_img == n:
        return "image"
    return "mixed"


# ``token_routing_kind`` ends in ``.sum().item()``. The runner classifies the
# batch in ``embed_input_ids`` before the step's ``ForwardContext`` exists, then
# the first eager MoE gate moves that model's one-slot value onto the fresh
# context. Later gates read it without another host sync.
_ROUTING_KIND_CTX_ATTR = "_dspark_vision_exp_routing_kind"
_FORWARD_CONTEXT_HOOKS = None


def routing_kind_from_mm(n_placeholders: int, n_tokens: int) -> str:
    """Routing kind from the multimodal placeholder count. No host sync.

    ``n_placeholders`` is ``is_multimodal.sum()``, which ``embed_input_ids``
    already reads for the issue #172 length check, so this costs nothing extra.
    ``is_multimodal`` is the runner's authoritative "these rows get image
    embeddings" mask, i.e. exactly the rows that must route with ``bias_vl``.
    """
    if n_tokens <= 0:
        raise ValueError(
            f"Vision-Exp routing kind needs a token count, got n_tokens={n_tokens!r} "
            f"with n_placeholders={n_placeholders!r}"
        )
    if n_placeholders <= 0:
        return "text"
    if n_placeholders >= n_tokens:
        return "image"
    return "mixed"




def _forward_context_or_none():
    """The current ``ForwardContext``, or None when no forward is running.

    Resolved once and cached. A missing symbol is a hard failure, exactly like
    this patch's dependency on ``_require_is_multimodal``: degrading quietly
    would put 43 host syncs per forward back with nothing to show for it. The
    import stays lazy so the CPU test suite can import this module without vLLM.
    """
    global _FORWARD_CONTEXT_HOOKS
    if _FORWARD_CONTEXT_HOOKS is None:
        from vllm import forward_context as _fc

        _FORWARD_CONTEXT_HOOKS = (
            _fc.is_forward_context_available,
            _fc.get_forward_context,
        )
    available, get_ctx = _FORWARD_CONTEXT_HOOKS
    if not available():
        return None
    return get_ctx()


def current_routing_kind(
    input_tokens: Any,
    image_token_id: int = IMAGE_TOKEN_ID,
    kind_cell: Any = None,
) -> str:
    """Return one routing kind per eager forward.

    ``kind_cell`` belongs to one language-model instance and is shared only
    with that model's gates. A missing publication falls back to one token scan
    memoized on the current ``ForwardContext``.
    """
    ctx = _forward_context_or_none()
    if ctx is None:
        return token_routing_kind(input_tokens, image_token_id)
    kind = getattr(ctx, _ROUTING_KIND_CTX_ATTR, None)
    if kind is None:
        if kind_cell is not None:
            kind = kind_cell[0]
            kind_cell[0] = None
        if kind is None:
            kind = token_routing_kind(input_tokens, image_token_id)
        # Fail during warmup if ForwardContext stops accepting attributes;
        # silently restoring 43 scans per forward is not a safe fallback.
        setattr(ctx, _ROUTING_KIND_CTX_ATTR, kind)
    return kind


def is_vision_exp_weight_name(name: str) -> bool:
    """Checkpoint keys for the ViT/Aligner/image tokens.

    Anemll ``load_weights`` treats any name containing ``w1`` as a stacked
    ``gate_up_proj`` shard. Aligner's ``w1`` and ViT MLP ``w1`` are full
    Linear layers, so those keys must bypass the stacked mapping.
    """
    return (
        name.startswith("aligner.")
        or name.startswith("vision.")
        or name.startswith("image_")
        or name.startswith("model.aligner.")
        or name.startswith("model.vision.")
        or name.startswith("model.image_")
    )


def is_unregistered_router_bias(name: str, param_names: Any) -> bool:
    """True when a gate bias has no Parameter.

    Hash MoE (layers 0–2) does not register ``e_score_correction_bias``.
    0731 omitted those tensors; Vision-Exp dumps ``ffn.gate.bias`` /
    ``bias_vl`` there anyway. Anemll's mapper rewrites them to
    ``e_score_correction_bias`` / ``_vl``, then ``load_weights`` KeyErrors.
    The DSpark draft loader uses ``endswith(".ffn.gate.bias")``, which
    misses ``bias_vl``, so unmapped ``.ffn.gate.bias_vl`` is also unused
    until that remap is patched.
    """
    if name in param_names:
        return False
    return name.endswith(
        (
            ".ffn.gate.e_score_correction_bias",
            ".ffn.gate.e_score_correction_bias_vl",
            ".ffn.gate.bias_vl",
            ".ffn.gate.bias",
        )
    )


def looks_like_chw(shape: Sequence[int]) -> bool:
    """True when a 3-D array should be treated as C×H×W rather than H×W×C.

    Last axis in ``{1, 3, 4}`` also looks like HWC. The old check therefore
    skipped transpose whenever width was 1, 3, or 4, so RGB CHW from
    ``np.transpose(pil, (2, 0, 1))`` of a 4-wide image became a black
    3-pixel-tall RGBA. Prefer CHW when the leading axis is RGB (C=3), which
    matches vLLM's layout and the ``(3, 6, 4)`` unit test. Gray/RGBA leading
    1/4 still use the unambiguous rule (last axis not a channel count).
    """
    if len(shape) != 3:
        return False
    channels = (1, 3, 4)
    c, _h, w = int(shape[0]), int(shape[1]), int(shape[2])
    if c not in channels:
        return False
    if w not in channels:
        return True
    return c == 3


def as_pil(item: Any) -> PILImage.Image:
    """Normalize vLLM image items (PIL, HWC/CHW array, tensor, dict) to RGB PIL."""
    Image, _ImageOps = _pil()
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    if isinstance(item, dict):
        for key in ("image_pil", "pil", "image"):
            value = item.get(key)
            if value is None or isinstance(value, dict):
                continue
            return as_pil(value)
        raise TypeError(f"Unsupported image dict keys: {sorted(item)!r}")
    array = item
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]
    if torch is not None and isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    try:
        import numpy as np
    except ImportError as exc:
        raise TypeError(f"Unsupported image item type: {type(item)!r}") from exc
    if not hasattr(array, "ndim"):
        raise TypeError(f"Unsupported image item type: {type(item)!r}")
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif looks_like_chw(arr.shape):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3:
        raise TypeError(f"Unsupported image array shape: {arr.shape!r}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if np.issubdtype(arr.dtype, np.floating):
        peak = float(arr.max()) if arr.size else 0.0
        if peak <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype("uint8")
    else:
        arr = np.clip(arr, 0, 255).astype("uint8")
    if arr.shape[-1] == 4:
        return Image.fromarray(arr, mode="RGBA").convert("RGB")
    return Image.fromarray(arr[..., :3], mode="RGB")


@dataclass
class ImageInput:
    start: int
    patches: Any
    n_vit_h: int
    n_vit_w: int
    types: Any
    perm: Any


def vision_args_from_config(config: Any) -> SimpleNamespace:
    return SimpleNamespace(
        vision_patch_size=int(getattr(config, "vision_patch_size", 14)),
        vision_downsample_ratio=int(getattr(config, "vision_downsample_ratio", 3)),
        vision_max_n_token=int(getattr(config, "vision_max_n_token", 384)),
        vision_min_pixels=int(getattr(config, "vision_min_pixels", 147456)),
        vision_max_wh_ratio=getattr(config, "vision_max_wh_ratio", 8),
        vision_n_layers=int(getattr(config, "vision_n_layers", 0)),
        vision_dim=int(getattr(config, "vision_dim", 1024)),
        vision_n_heads=int(getattr(config, "vision_n_heads", 16)),
        vision_inter_dim=int(getattr(config, "vision_inter_dim", 2816)),
        vision_rope_theta=float(getattr(config, "vision_rope_theta", 10000.0)),
        vocab_size=int(getattr(config, "vocab_size", 129280)),
        dim=int(getattr(config, "hidden_size", getattr(config, "dim", 4096))),
        hidden_size=int(getattr(config, "hidden_size", getattr(config, "dim", 4096))),
    )


def grid_tokens(best_height, best_width, patch_size, downsample_ratio):
    """Number of LLM tokens the aligner grid occupies (N-layout, incl. row/align padding)."""
    n_llm_h = math.ceil((best_height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((best_width // patch_size) / downsample_ratio)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    elif max_h_float < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        assert max_w > 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(
            max_w * patch_size * downsample_ratio / width,
            max_h * patch_size * downsample_ratio / height,
        )
        best_width = math.floor(width * beta / patch_size) * patch_size
        best_height = math.floor(height * beta / patch_size) * patch_size
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def safe_resize(height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token):
    max_n_token -= COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    budget = max_n_token
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, budget
        )
        budget -= 1
    return n_llm_h, n_llm_w, best_height, best_width


def load_image_bytes(record) -> bytes:
    """Load image bytes from raw/base64 data, an Anthropic source, URL, or path."""
    data = record.get("data")
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return base64.b64decode(data)

    source = record.get("source")
    if isinstance(source, dict):
        if source.get("data") is not None:
            return base64.b64decode(source["data"])
        if source.get("url"):
            return load_image_bytes({"url": source["url"]})

    url = record.get("url")
    if isinstance(url, str) and url:
        if url.startswith("data:"):
            header, _, payload = url.partition(",")
            if ";base64" not in header:
                raise ValueError(f"Unsupported data URL encoding: {header}")
            return base64.b64decode(payload)
        if url.startswith(("http://", "https://")):
            with urlopen(url, timeout=30) as response:
                return response.read()
        with open(url, "rb") as file:
            return file.read()

    raise ValueError(f"Cannot load image from record: {list(record.keys())}")


def pil_to_patches(image: PILImage.Image, args) -> tuple[Any, int, int, int, int]:
    """Resize/pad one RGB image and return ViT patches plus LLM grid sizes."""
    import torch

    _Image, ImageOps = _pil()

    p = args.vision_patch_size
    image = image.convert("RGB")
    width, height = image.size
    if args.vision_max_wh_ratio is not None and width > height * args.vision_max_wh_ratio:
        width = height * args.vision_max_wh_ratio
    if 0 < width * height < args.vision_min_pixels:
        ratio = (args.vision_min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    n_llm_h, n_llm_w, best_height, best_width = safe_resize(
        height,
        width,
        best_height,
        best_width,
        p,
        args.vision_downsample_ratio,
        args.vision_max_n_token,
    )
    n_vit_h, n_vit_w = best_height // p, best_width // p
    src_w, src_h = image.size
    if args.vision_max_wh_ratio is not None and src_w >= args.vision_max_wh_ratio * src_h:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    try:
        import numpy as np

        x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    except ImportError:
        pixels = (
            image.get_flattened_data()
            if hasattr(image, "get_flattened_data")
            else image.getdata()
        )
        x = torch.tensor(pixels, dtype=torch.float32)
        x = x.view(image.height, image.width, 3).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
    patches = (
        x.reshape(3, n_vit_h, p, n_vit_w, p)
        .permute(1, 3, 0, 2, 4)
        .reshape(n_vit_h * n_vit_w, 3, p, p)
    )
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def load_image(record, args):
    """Load and transform one image record into ViT patches."""
    Image, _ImageOps = _pil()
    with Image.open(io.BytesIO(load_image_bytes(record))) as source:
        image = source.convert("RGB")
        return pil_to_patches(image, args)


def build_image_block(n_llm_h: int, n_llm_w: int, start_pos: int):
    """Builds the N-layout token types (final order) and the aligner-row order for IMAGE slots."""
    import torch

    compress_pad = compress_pad_tokens(start_pos)
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    types = torch.tensor(
        ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_PAD] * (row_len * pad_h),
        dtype=torch.int64,
    )
    order = torch.arange(rows * row_len).view(rows // 2, 2, row_len).transpose(1, 2).reshape(-1)
    image_idx = torch.full((rows * row_len,), -1, dtype=torch.int64)
    image_idx.view(rows, row_len)[:n_llm_h, :n_llm_w] = torch.arange(n_llm_h * n_llm_w).view(
        n_llm_h, n_llm_w
    )
    perm = image_idx[order]
    perm = perm[perm >= 0]
    types = torch.cat(
        [
            torch.full((compress_pad,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_START]),
            types[order],
            torch.full((pad_last,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_END]),
        ]
    )
    return types, perm
