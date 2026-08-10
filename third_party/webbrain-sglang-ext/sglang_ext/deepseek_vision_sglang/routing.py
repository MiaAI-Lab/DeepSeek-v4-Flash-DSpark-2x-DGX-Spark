from __future__ import annotations

from collections.abc import Sequence
from typing import Any


ROUTING_IDS_ATTR = "_deepseek_vision_routing_ids"


def _as_int_list(values: Any) -> list[int]:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(value) for value in values]


def _item_is_image(item: Any) -> bool:
    modality = getattr(item, "modality", None)
    name = getattr(modality, "name", modality)
    return str(name).upper() == "IMAGE"


def _item_offsets(item: Any) -> list[tuple[int, int]]:
    offsets = getattr(item, "offsets", None) or []
    return [(int(start), int(end)) for start, end in offsets]


def routing_replacements(
    *,
    extend_prefix_lens: Sequence[int],
    extend_seq_lens: Sequence[int],
    mm_inputs: Sequence[Any],
    palette: Sequence[int],
) -> list[tuple[int, int]]:
    """Return flattened ``(position, route_id)`` replacements for image slots.

    SGLang flattens each request's current extend chunk into one token vector.  An
    item's offsets remain absolute, inclusive prompt offsets.  Basing the palette
    phase on ``absolute_position - image_start`` keeps routing identical when a long
    image prefix is split across chunked-prefill batches.
    """
    if not palette:
        raise ValueError("routing palette cannot be empty")
    prefixes = _as_int_list(extend_prefix_lens)
    lengths = _as_int_list(extend_seq_lens)
    inputs = list(mm_inputs or [])
    if not (len(prefixes) == len(lengths) == len(inputs)):
        raise ValueError("SGLang request metadata lengths differ")

    replacements: list[tuple[int, int]] = []
    flat_request_start = 0
    for prefix_len, seq_len, request_mm in zip(prefixes, lengths, inputs, strict=True):
        chunk_start = prefix_len
        chunk_end = prefix_len + seq_len
        items = getattr(request_mm, "mm_items", None) if request_mm is not None else None
        for item in items or []:
            if not _item_is_image(item):
                continue
            for image_start, image_end_inclusive in _item_offsets(item):
                overlap_start = max(chunk_start, image_start)
                overlap_end = min(chunk_end, image_end_inclusive + 1)
                for absolute_position in range(overlap_start, overlap_end):
                    flat_position = flat_request_start + absolute_position - chunk_start
                    palette_index = (absolute_position - image_start) % len(palette)
                    replacements.append((flat_position, int(palette[palette_index])))
        flat_request_start += seq_len
    return replacements


def build_sglang_routing_ids(input_ids: Any, forward_batch: Any, palette: Sequence[int]):
    """Clone SGLang token IDs and replace only image slots with palette IDs."""
    if input_ids is None:
        raise ValueError("input_ids are required to build DeepSeek routing IDs")
    result = input_ids.clone() if hasattr(input_ids, "clone") else list(input_ids)
    forward_mode = getattr(forward_batch, "forward_mode", None)
    is_extend = getattr(forward_mode, "is_extend", None)
    if callable(is_extend) and not is_extend():
        # Image placeholders are consumed during prefill. SGLang deliberately
        # keeps ``mm_inputs`` attached to later decode batches, while the
        # extend-only prefix/length metadata is absent. Generated decode tokens
        # must therefore retain their ordinary token IDs.
        return result
    mm_inputs = getattr(forward_batch, "mm_inputs", None)
    if not mm_inputs:
        return result
    replacements = routing_replacements(
        extend_prefix_lens=getattr(forward_batch, "extend_prefix_lens_cpu", None),
        extend_seq_lens=getattr(forward_batch, "extend_seq_lens_cpu", None),
        mm_inputs=mm_inputs,
        palette=palette,
    )
    for position, route_id in replacements:
        result[position] = route_id
    return result
