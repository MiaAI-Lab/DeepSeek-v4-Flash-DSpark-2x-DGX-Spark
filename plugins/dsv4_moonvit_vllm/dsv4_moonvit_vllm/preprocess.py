"""NaViT-style image resize / patch grid math for MoonViT (Kimi-K2.6).

Ported from vLLM ``kimi_k25_vision_fused.navit_resize_image`` and WebBrain limits.
Does not require SGLang or the full HF Kimi processor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# WebBrain deepseek_vision.max_image_tokens = 512 → in_patch_limit = 512*2*2
DEFAULT_MAX_IMAGE_TOKENS = 512
DEFAULT_PATCH_SIZE = 14
DEFAULT_MERGE = 2
DEFAULT_PATCH_LIMIT_ON_ONE_SIDE = 512
DEFAULT_IMAGE_MEAN = (0.5, 0.5, 0.5)
DEFAULT_IMAGE_STD = (0.5, 0.5, 0.5)


@dataclass(frozen=True)
class ResizeConfig:
    num_tokens: int
    new_width: int
    new_height: int
    pad_width: int
    pad_height: int
    grid_t: int
    grid_h: int
    grid_w: int

    @property
    def padded_width(self) -> int:
        return self.new_width + self.pad_width

    @property
    def padded_height(self) -> int:
        return self.new_height + self.pad_height

    @property
    def num_patches(self) -> int:
        return self.grid_t * self.grid_h * self.grid_w


def navit_resize_image(
    width: int,
    height: int,
    *,
    patch_size: int = DEFAULT_PATCH_SIZE,
    merge_kernel_size: int = DEFAULT_MERGE,
    in_patch_limit: int | None = None,
    patch_limit_on_one_side: int = DEFAULT_PATCH_LIMIT_ON_ONE_SIDE,
    fixed_output_tokens: int | None = None,
    max_image_tokens: int = DEFAULT_MAX_IMAGE_TOKENS,
) -> ResizeConfig:
    """Compute resize/pad/grid for a single image → merged LLM image tokens."""
    if in_patch_limit is None:
        in_patch_limit = int(max_image_tokens) * int(merge_kernel_size) * int(
            merge_kernel_size
        )

    s1 = math.sqrt(
        in_patch_limit
        / (max(1.0, width // patch_size) * max(1.0, height // patch_size))
    )
    s2 = patch_limit_on_one_side * patch_size / width
    s3 = patch_limit_on_one_side * patch_size / height
    scale = min(1.0, s1, s2, s3)
    new_w = min(max(1, int(width * scale)), patch_limit_on_one_side * patch_size)
    new_h = min(max(1, int(height * scale)), patch_limit_on_one_side * patch_size)

    factor = merge_kernel_size * patch_size
    pad_height = (factor - new_h % factor) % factor
    pad_width = (factor - new_w % factor) % factor

    if fixed_output_tokens is not None:
        num_tokens = int(fixed_output_tokens)
    else:
        token_height = (new_h + pad_height) // factor
        token_width = (new_w + pad_width) // factor
        num_tokens = token_height * token_width

    # Enforce training envelope: shrink until merged tokens ≤ max_image_tokens.
    if fixed_output_tokens is None and num_tokens > max_image_tokens and num_tokens > 0:
        shrink = math.sqrt(max_image_tokens / num_tokens)
        new_w = max(factor, int(new_w * shrink))
        new_h = max(factor, int(new_h * shrink))
        # Snap down to merge factor
        new_w = max(factor, (new_w // factor) * factor)
        new_h = max(factor, (new_h // factor) * factor)
        pad_height = 0
        pad_width = 0
        token_height = new_h // factor
        token_width = new_w // factor
        num_tokens = token_height * token_width
        # Final clamp if still slightly over due to integer math
        while num_tokens > max_image_tokens and (token_height > 1 or token_width > 1):
            if token_width >= token_height and token_width > 1:
                token_width -= 1
            elif token_height > 1:
                token_height -= 1
            else:
                break
            new_w = token_width * factor
            new_h = token_height * factor
            num_tokens = token_height * token_width

    padded_h = new_h + pad_height
    padded_w = new_w + pad_width
    grid_h = padded_h // patch_size
    grid_w = padded_w // patch_size
    return ResizeConfig(
        num_tokens=int(num_tokens),
        new_width=int(new_w),
        new_height=int(new_h),
        pad_width=int(pad_width),
        pad_height=int(pad_height),
        grid_t=1,
        grid_h=int(grid_h),
        grid_w=int(grid_w),
    )


def default_media_proc_cfg(
    max_image_tokens: int = DEFAULT_MAX_IMAGE_TOKENS,
) -> dict[str, Any]:
    """Kimi-K2.6 media_proc_cfg with WebBrain image-token cap."""
    merge = DEFAULT_MERGE
    return {
        "in_patch_limit": int(max_image_tokens) * merge * merge,
        "patch_size": DEFAULT_PATCH_SIZE,
        "image_mean": list(DEFAULT_IMAGE_MEAN),
        "image_std": list(DEFAULT_IMAGE_STD),
        "merge_kernel_size": merge,
        "fixed_output_tokens": None,
        "patch_limit_on_one_side": DEFAULT_PATCH_LIMIT_ON_ONE_SIDE,
        "in_patch_limit_each_frame": int(max_image_tokens) * merge * merge,
        "in_patch_limit_video": None,
        "sample_fps": 2.0,
        "max_num_frames_each_video": None,
        "temporal_merge_kernel_size": 4,
        "timestamp_mode": "hh:mm:ss.fff",
    }


def pil_to_pixel_values_and_grid(
    image: Any,
    *,
    max_image_tokens: int = DEFAULT_MAX_IMAGE_TOKENS,
    media_proc_cfg: dict[str, Any] | None = None,
) -> tuple[Any, list[int], int]:
    """Resize/normalize a PIL image into MoonViT patch tensors.

    Returns:
        pixel_values: ``torch.FloatTensor`` of shape ``(num_patches, 3, P, P)``
        grid_thw: ``[t, h, w]``
        num_tokens: merged LLM image token count (≤ max_image_tokens typically)
    """
    import numpy as np
    import torch
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError(f"expected PIL.Image, got {type(image)}")

    cfg = media_proc_cfg or default_media_proc_cfg(max_image_tokens)

    # Prefer vLLM's fused Kimi preprocessor when available (numba path) so
    # patch layout / normalize_lut match production Kimi-K2.5/2.6 exactly.
    try:
        from vllm.transformers_utils.processors.kimi_k25_vision_fused import (
            KimiK25FusedVisionProcessor,
        )

        fused = KimiK25FusedVisionProcessor(cfg)
        out = fused.preprocess(
            [{"type": "image", "image": image}], return_tensors="pt"
        )
        pv = out["pixel_values"]
        if isinstance(pv, list):
            pv = pv[0]
        grid = out["grid_thws"]
        if hasattr(grid, "tolist"):
            g = grid.tolist()
            grid_thw = g[0] if g and isinstance(g[0], (list, tuple)) else g
        else:
            grid_thw = list(grid[0])
        grid_thw = [int(x) for x in grid_thw]
        t, h, w = grid_thw
        n_patches = t * h * w
        merge = cfg["merge_kernel_size"]
        mk = int(merge[0] if isinstance(merge, (list, tuple)) else merge)
        num_tokens = (h // mk) * (w // mk)
        num_tokens = min(int(num_tokens), int(max_image_tokens))
        if pv.shape[0] != n_patches:
            # Some fused builds pack as (1, N, 3, P, P)
            pv = pv.reshape(-1, *pv.shape[-3:])
        return pv.float(), grid_thw, num_tokens
    except Exception:
        pass  # fall through to pure-numpy path
    patch_size = int(cfg["patch_size"])
    merge = cfg["merge_kernel_size"]
    if isinstance(merge, (list, tuple)):
        merge_k = int(merge[0])
    else:
        merge_k = int(merge)

    width, height = image.size
    resize = navit_resize_image(
        width,
        height,
        patch_size=patch_size,
        merge_kernel_size=merge_k,
        in_patch_limit=int(cfg["in_patch_limit"]),
        patch_limit_on_one_side=int(cfg["patch_limit_on_one_side"]),
        fixed_output_tokens=cfg.get("fixed_output_tokens"),
        max_image_tokens=max_image_tokens,
    )

    rgb = image.convert("RGB")
    resized = rgb.resize(
        (resize.new_width, resize.new_height), resample=Image.Resampling.BICUBIC
    )
    arr = np.asarray(resized, dtype=np.float32)  # H, W, 3  (RGB)
    # Some MoonViT/OpenCV stacks expect BGR channel order. Controlled by
    # DSV4_MOONVIT_CHANNEL_ORDER=rgb|bgr (default rgb; set bgr if red↔blue).
    import os

    # Optional channel order override (rgb default — Kimi fused uses RGB).
    channel_order = os.environ.get("DSV4_MOONVIT_CHANNEL_ORDER", "rgb").lower()
    if channel_order == "bgr":
        arr = arr[..., ::-1].copy()
    if resize.pad_height or resize.pad_width:
        # Pad with neutral grey in uint8 space (matches Kimi fused pad=0 on
        # unnormalized frames → 0 → normalized (0-mean)/std). Keep pad=0.
        arr = np.pad(
            arr,
            ((0, resize.pad_height), (0, resize.pad_width), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    mean = np.asarray(cfg["image_mean"], dtype=np.float32)
    std = np.asarray(cfg["image_std"], dtype=np.float32)
    # Match vLLM kimi_k25_vision_fused normalize_lut: (v/255 - mean) * (1/std)
    arr = (arr / 255.0 - mean) * (1.0 / std)  # H, W, 3

    # Patchify to (t*h*w, 3, P, P) — same index order as fused _write_fused_patches
    # (patch major: row-major over patches, then C, y%P, x%P).
    h, w, _ = arr.shape
    assert h % patch_size == 0 and w % patch_size == 0
    gh, gw = h // patch_size, w // patch_size
    patches = (
        arr.reshape(gh, patch_size, gw, patch_size, 3)
        .transpose(0, 2, 4, 1, 3)
        .reshape(gh * gw, 3, patch_size, patch_size)
    )
    # Contiguous float32 for torch; avoid shared memory views.
    patches = np.ascontiguousarray(patches)
    pixel_values = torch.from_numpy(patches.copy())
    grid_thw = [1, gh, gw]
    return pixel_values, grid_thw, resize.num_tokens
