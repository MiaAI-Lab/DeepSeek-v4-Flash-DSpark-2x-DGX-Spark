"""Palette-cycle routing for MoonViT image token positions on DeepSeek V4.

Ported from WebBrain ``sglang_ext/deepseek_vision_sglang/routing.py`` semantics:
image positions get deterministic palette IDs; text token IDs stay unchanged.
Hash MoE layers require valid in-vocab route IDs (placeholder 129280 is OOV).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

# Canonical 64-ID palette from WebBrain DeepSeek-V4-Flash-0731-Vision-BF16.
DEFAULT_ROUTING_PALETTE: tuple[int, ...] = (
    0,
    1,
    2,
    8,
    9,
    10,
    12,
    74,
    81,
    110,
    114,
    240,
    17081,
    25312,
    30711,
    58279,
    7637,
    8936,
    45556,
    52073,
    7743,
    8347,
    13203,
    19795,
    44418,
    62970,
    79038,
    6381,
    48025,
    109859,
    29629,
    91213,
    90662,
    121562,
    8570,
    25568,
    3685,
    81916,
    14638,
    50590,
    101211,
    24832,
    75337,
    131,
    15170,
    79723,
    84052,
    20866,
    48327,
    72234,
    15507,
    128,
    16760,
    34135,
    36264,
    59037,
    3839,
    29854,
    109646,
    64,
    23442,
    6584,
    10255,
    17173,
)

IMAGE_TOKEN_ID_DEFAULT = 129280


def _as_int_list(values: Any) -> list[int]:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(value) for value in values]


def palette_cycle_replacements(
    input_ids: Sequence[int] | torch.Tensor,
    *,
    image_token_id: int = IMAGE_TOKEN_ID_DEFAULT,
    palette: Sequence[int] = DEFAULT_ROUTING_PALETTE,
) -> list[tuple[int, int]]:
    """Return ``(flat_index, route_id)`` for every image-placeholder position.

    Contiguous runs of ``image_token_id`` form one image span. Within a span the
    palette phase is ``offset_in_span % len(palette)`` so chunked-prefill
    slices that keep absolute offsets align with WebBrain training.
    """
    if not palette:
        raise ValueError("routing palette cannot be empty")
    ids = _as_int_list(input_ids)
    replacements: list[tuple[int, int]] = []
    i = 0
    n = len(ids)
    while i < n:
        if ids[i] != image_token_id:
            i += 1
            continue
        start = i
        while i < n and ids[i] == image_token_id:
            offset = i - start
            replacements.append((i, int(palette[offset % len(palette)])))
            i += 1
    return replacements


def apply_palette_cycle(
    input_ids: torch.Tensor | list[int],
    *,
    image_token_id: int = IMAGE_TOKEN_ID_DEFAULT,
    palette: Sequence[int] = DEFAULT_ROUTING_PALETTE,
    clone: bool = True,
) -> torch.Tensor | list[int]:
    """Replace only image-placeholder IDs with palette route IDs.

    Text token IDs are never rewritten. Decode tokens (no placeholders) pass
    through unchanged.

    Tensor path is **device-safe** (no ``.tolist()``) so it works under CUDA
    graph capture.
    """
    if isinstance(palette, torch.Tensor):
        if palette.numel() == 0:
            raise ValueError("routing palette cannot be empty")
    elif not palette:
        raise ValueError("routing palette cannot be empty")

    if isinstance(input_ids, torch.Tensor):
        result = input_ids.clone() if clone else input_ids
        flat = result.view(-1)
        if flat.numel() == 0:
            return result
        is_img = flat == int(image_token_id)
        ones = is_img.to(dtype=torch.int64)
        # Rising edges of is_img start a new image span.
        shifted = torch.cat(
            [torch.zeros(1, dtype=torch.bool, device=flat.device), is_img[:-1]]
        )
        new_run = is_img & ~shifted
        img_cum = torch.cumsum(ones, dim=0)
        # base = number of image tokens before this span (propagated via cummax)
        bases = torch.where(new_run, img_cum - 1, torch.zeros_like(img_cum))
        bases = torch.cummax(bases, dim=0).values
        bases = torch.where(is_img, bases, torch.zeros_like(bases))
        phase = (img_cum - bases - 1).clamp(min=0)
        # Prefer a pre-placed CUDA buffer (module.register_buffer) so graph
        # capture never host→device copies the palette.
        if isinstance(palette, torch.Tensor):
            if palette.device != flat.device or palette.dtype != flat.dtype:
                # Same-device view/cast only when already on GPU; otherwise
                # require caller to pass a device-resident buffer.
                if palette.is_cuda and flat.is_cuda:
                    pal = palette.to(dtype=flat.dtype)
                elif not flat.is_cuda:
                    pal = palette.to(device=flat.device, dtype=flat.dtype)
                else:
                    raise RuntimeError(
                        "palette tensor must already live on the input device "
                        "for CUDA-graph-safe palette_cycle"
                    )
            else:
                pal = palette
        else:
            # CPU / unit-test path only
            pal = torch.tensor(list(palette), device=flat.device, dtype=flat.dtype)
        route = pal[phase % pal.numel()]
        flat = torch.where(is_img, route, flat)
        return flat.view_as(result)

    # List path (unit tests / CPU scripting)
    result = list(input_ids) if clone else input_ids
    ids_list = result
    if not any(int(x) == image_token_id for x in ids_list):
        return result
    replacements = palette_cycle_replacements(
        ids_list, image_token_id=image_token_id, palette=palette
    )
    for pos, route_id in replacements:
        result[pos] = route_id
    return result


def routing_replacements_for_spans(
    *,
    extend_prefix_lens: Sequence[int],
    extend_seq_lens: Sequence[int],
    image_spans: Sequence[Sequence[tuple[int, int]]],
    palette: Sequence[int],
) -> list[tuple[int, int]]:
    """SGLang-compatible span routing for chunked prefill unit tests.

    ``image_spans[req]`` is a list of ``(image_start, image_end_inclusive)``
    absolute prompt offsets. Returns flattened ``(position, route_id)``.
    """
    if not palette:
        raise ValueError("routing palette cannot be empty")
    prefixes = _as_int_list(extend_prefix_lens)
    lengths = _as_int_list(extend_seq_lens)
    if not (len(prefixes) == len(lengths) == len(image_spans)):
        raise ValueError("request metadata lengths differ")

    replacements: list[tuple[int, int]] = []
    flat_request_start = 0
    for prefix_len, seq_len, spans in zip(prefixes, lengths, image_spans, strict=True):
        chunk_start = prefix_len
        chunk_end = prefix_len + seq_len
        for image_start, image_end_inclusive in spans:
            overlap_start = max(chunk_start, int(image_start))
            overlap_end = min(chunk_end, int(image_end_inclusive) + 1)
            for absolute_position in range(overlap_start, overlap_end):
                flat_position = flat_request_start + absolute_position - chunk_start
                palette_index = (absolute_position - int(image_start)) % len(palette)
                replacements.append((flat_position, int(palette[palette_index])))
        flat_request_start += seq_len
    return replacements
