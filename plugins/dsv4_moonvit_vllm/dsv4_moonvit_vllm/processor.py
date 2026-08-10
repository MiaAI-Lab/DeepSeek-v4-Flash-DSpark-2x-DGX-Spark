"""vLLM multimodal processor: OpenAI image parts → MoonViT pixel_values + <image> expand."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
from PIL import Image

from .config import (
    deepseek_vision_dict,
    image_token_id,
    max_image_tokens,
    routing_palette,
)
from .preprocess import (
    default_media_proc_cfg,
    pil_to_pixel_values_and_grid,
)

logger = logging.getLogger(__name__)

IMAGE_PLACEHOLDER = "<image>"


def _find_nth_media_token_span(
    prompt_ids: Sequence[int],
    tokenizer: Any,
    media_tok: str,
    n: int = 0,
) -> tuple[int, int] | None:
    """Locate the *n*-th token span covering ``media_tok`` under BPE merges.

    Uses offset mapping to find all occurrences and returns the *n*-th one
    (0-indexed).  Each image placeholder expands independently.
    """
    ids = [int(x) for x in prompt_ids]
    if not ids or not media_tok:
        return None
    full = tokenizer.decode(ids)

    # Try offset mapping first (most reliable for BPE merges).
    try:
        enc = tokenizer(
            full, add_special_tokens=False, return_offsets_mapping=True
        )
        enc_ids = [int(x) for x in enc["input_ids"]]
        if enc_ids == ids and "offset_mapping" in enc:
            occurrence = 0
            char_start = 0
            while True:
                idx = full.find(media_tok, char_start)
                if idx < 0:
                    return None
                char_end = idx + len(media_tok)
                start_t: int | None = None
                end_t: int | None = None
                for i, (a, b) in enumerate(enc["offset_mapping"]):
                    a_i, b_i = int(a), int(b)
                    if b_i <= idx:
                        continue
                    if start_t is None and a_i < char_end and b_i > idx:
                        start_t = i
                    if a_i < char_end:
                        end_t = i + 1
                if start_t is not None and end_t is not None and end_t > start_t:
                    if occurrence == n:
                        return start_t, end_t
                    occurrence += 1
                char_start = char_end
    except Exception:
        pass

    # Fallback: progressive decode (slower but works without offset_mapping).
    occurrence = 0
    char_start = 0
    while True:
        idx = full.find(media_tok, char_start)
        if idx < 0:
            return None
        char_end = idx + len(media_tok)
        start_t = None
        for i in range(len(ids)):
            prefix = tokenizer.decode(ids[: i + 1])
            if start_t is None and len(prefix) > idx:
                start_t = i
            if start_t is not None and len(prefix) >= char_end:
                if occurrence == n:
                    return start_t, i + 1
                occurrence += 1
                char_start = char_end
                break
        else:
            return None


def _import_vllm_mm():
    from transformers import BatchFeature

    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
    from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems
    from vllm.multimodal.processing import (
        BaseDummyInputsBuilder,
        BaseMultiModalProcessor,
        BaseProcessingInfo,
        PromptReplacement,
        PromptUpdate,
    )

    return {
        "BatchFeature": BatchFeature,
        "MULTIMODAL_REGISTRY": MULTIMODAL_REGISTRY,
        "MultiModalFieldConfig": MultiModalFieldConfig,
        "MultiModalKwargsItems": MultiModalKwargsItems,
        "ImageProcessorItems": ImageProcessorItems,
        "MultiModalDataItems": MultiModalDataItems,
        "BaseDummyInputsBuilder": BaseDummyInputsBuilder,
        "BaseMultiModalProcessor": BaseMultiModalProcessor,
        "BaseProcessingInfo": BaseProcessingInfo,
        "PromptReplacement": PromptReplacement,
        "PromptUpdate": PromptUpdate,
    }


def build_processor_classes():
    """Lazily build processor classes bound to vLLM multimodal APIs."""
    mm = _import_vllm_mm()
    BaseProcessingInfo = mm["BaseProcessingInfo"]
    BaseMultiModalProcessor = mm["BaseMultiModalProcessor"]
    BaseDummyInputsBuilder = mm["BaseDummyInputsBuilder"]
    BatchFeature = mm["BatchFeature"]
    MultiModalFieldConfig = mm["MultiModalFieldConfig"]
    PromptReplacement = mm["PromptReplacement"]
    ImageProcessorItems = mm["ImageProcessorItems"]

    class Dsv4MoonVitProcessingInfo(BaseProcessingInfo):
        def __init__(self, ctx) -> None:
            super().__init__(ctx)
            hf_config = self.get_hf_config()
            self.hf_config = hf_config
            adapter = deepseek_vision_dict(hf_config)
            self.media_token = adapter.get("image_placeholder", IMAGE_PLACEHOLDER)
            self.media_token_id = image_token_id(hf_config)
            # In-vocab palette used as expanded placeholder ids (129280 is OOV
            # and fails vLLM input_processor max-token checks). Model routing
            # already expects these palette route IDs on image spans.
            self.route_palette = routing_palette(hf_config)
            self.max_image_tokens = int(
                adapter.get("max_image_tokens", max_image_tokens())
            )
            self.media_proc_cfg = default_media_proc_cfg(self.max_image_tokens)

        def get_supported_mm_limits(self) -> Mapping[str, int | None]:
            return {"image": 4}

        def get_mm_max_tokens_per_item(
            self, seq_len: int, mm_counts: Mapping[str, int]
        ) -> Mapping[str, int]:
            return {"image": self.max_image_tokens}

        def image_token_count(self, image: Image.Image) -> int:
            # Must match encode path exactly (same resize + grid → token count).
            _, _, ntok = pil_to_pixel_values_and_grid(
                image,
                max_image_tokens=self.max_image_tokens,
                media_proc_cfg=self.media_proc_cfg,
            )
            return int(ntok)

    class Dsv4MoonVitMultiModalProcessor(BaseMultiModalProcessor[Dsv4MoonVitProcessingInfo]):
        def _get_mm_fields_config(
            self,
            hf_inputs: Any,
            hf_processor_mm_kwargs: Mapping[str, object],
        ) -> Mapping[str, Any]:
            grid_thws = hf_inputs.get("grid_thws", torch.empty((0, 3)))
            if not isinstance(grid_thws, torch.Tensor):
                grid_thws = torch.as_tensor(grid_thws)
            if grid_thws.numel() == 0:
                grid_sizes = torch.empty((0,), dtype=torch.long)
            else:
                grid_sizes = grid_thws.prod(-1)
            return dict(
                pixel_values=MultiModalFieldConfig.flat_from_sizes("image", grid_sizes),
                # Keep grids on GPU with pixels — MoonViT FLASH_ATTN is CUDA-only.
                grid_thws=MultiModalFieldConfig.batched("image"),
            )

        def _hf_processor_applies_updates(
            self,
            prompt_text: str,
            mm_items: Any,
            hf_processor_mm_kwargs: Mapping[str, object],
            tokenization_kwargs: Mapping[str, object],
        ) -> bool:
            # Always False: deepseek_v4 encodes literal "<image>" as ordinary
            # text tokens ([30, 10253, 32]), and the processor cache path forces
            # enable_hf_prompt_update=False so prompt_ids come from that encode.
            # vLLM PromptReplacement must expand those text tokens → N media ids.
            return False

        def _call_hf_processor(
            self,
            prompt: str,
            mm_data: Mapping[str, object],
            mm_kwargs: Mapping[str, object],
            tok_kwargs: Mapping[str, object],
        ) -> Any:
            # ProcessorBatchItems.get_processor_data() uses plural keys ("images").
            images = []
            for key in ("images", "image", "vision_chunk", "vision"):
                if key in mm_data and mm_data[key] is not None:
                    val = mm_data[key]
                    if isinstance(val, list):
                        images.extend(val)
                    else:
                        images.append(val)
            images = [im for im in images if im is not None]

            tokenizer = self.info.get_tokenizer()
            media_tok = self.info.media_token
            if not isinstance(prompt, str):
                prompt = ""

            # Ensure the literal placeholder string is present for text matching.
            if len(images) == 1 and media_tok not in prompt:
                prompt = f"{prompt}{media_tok}" if prompt else media_tok

            n_ph = prompt.count(media_tok) if images else 0
            # For multi-image: ensure at least one placeholder per image.
            # Clients may re-attach images in multiturn (N >= placeholders is OK
            # — vLLM counts unique images, prompt may have fewer <image> markers).
            if images and n_ph == 0:
                raise ValueError(
                    f"prompt has 0 {media_tok!r} placeholder(s) but "
                    f"request has {len(images)} image(s); mm_keys="
                    f"{list(mm_data.keys())}"
                )

            # Encode prompt as normal text — "<image>" becomes tokenizer text
            # tokens (not OOV 129280). PromptReplacement replaces that span.
            input_ids = list(
                map(int, tokenizer.encode(prompt, add_special_tokens=False))
            )

            pixel_chunks: list[torch.Tensor] = []
            grids: list[list[int]] = []
            for img in images:
                if not isinstance(img, Image.Image):
                    if hasattr(img, "shape"):
                        import numpy as np

                        arr = img if isinstance(img, np.ndarray) else img.numpy()
                        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                            arr = arr.transpose(1, 2, 0)
                        img = Image.fromarray(arr.astype("uint8"))
                    else:
                        raise TypeError(f"unsupported image type {type(img)}")
                pv, grid, ntok = pil_to_pixel_values_and_grid(
                    img,
                    max_image_tokens=self.info.max_image_tokens,
                    media_proc_cfg=self.info.media_proc_cfg,
                )
                ntok = min(int(ntok), self.info.max_image_tokens)
                if ntok < 1:
                    raise ValueError("image produced zero tokens")
                pixel_chunks.append(pv)
                grids.append(grid)

            data: dict[str, Any] = {"input_ids": [input_ids]}
            if pixel_chunks:
                data["pixel_values"] = torch.cat(pixel_chunks, dim=0)
                data["grid_thws"] = torch.tensor(grids, dtype=torch.long)
            print(
                f"[dsv4_moonvit] HF processor n_ids={len(input_ids)} "
                f"n_images={len(images)} grids={grids} "
                f"prompt={prompt[:80]!r}",
                flush=True,
            )
            return BatchFeature(data=data)

        def _get_prompt_updates(
            self,
            mm_items: Any,
            hf_processor_mm_kwargs: Mapping[str, Any],
            out_mm_kwargs: Any,
        ) -> Sequence[Any]:
            media_token_id = int(self.info.media_token_id)
            media_tok = self.info.media_token
            tokenizer = self.info.get_tokenizer()
            image_text_token_ids = list(
                map(int, tokenizer.encode(media_tok, add_special_tokens=False))
            )
            print(
                f"[dsv4_moonvit] prompt_updates target_str={media_tok!r} "
                f"target_ids={image_text_token_ids} media_id={media_token_id}",
                flush=True,
            )

            palette = list(self.info.route_palette)
            if not palette:
                raise RuntimeError("routing palette is empty")

            def get_replacement(item_idx: int):
                images = mm_items.get_items("image", (ImageProcessorItems,))
                n = self.info.image_token_count(images[item_idx])
                # In-vocab palette cycle — never emit OOV media_token_id (129280).
                ids = [int(palette[i % len(palette)]) for i in range(n)]
                print(
                    f"[dsv4_moonvit] replacement item={item_idx} n={n} "
                    f"size={getattr(images[item_idx], 'size', None)} "
                    f"head_ids={ids[:4]}",
                    flush=True,
                )
                return ids

            # Target token ids are best-effort; real matching is in
            # _apply_prompt_updates (BPE-merge-safe). Keep a string target so
            # vLLM's text fallback can also see the literal placeholder.
            return [
                PromptReplacement(
                    modality="image",
                    target=media_tok,
                    replacement=get_replacement,
                ),
            ]

        def _apply_prompt_updates(
            self,
            token_ids: list[int],
            mm_prompt_updates: Any,
        ) -> tuple[list[int], Mapping[str, list[Any]]]:
            """Replace ``<image>`` under BPE merges; avoid text re-encode of OOV.

            vLLM's default path falls back to string replace + re-tokenize when
            the exact target ids are missing. Re-encoding destroys OOV media
            ids (129280). We locate the real token span and expand in-place.
            """
            from collections import defaultdict

            from vllm.multimodal.processing.processor import (
                PlaceholderFeaturesInfo,
                _seq2tokens,
            )

            tokenizer = self.info.get_tokenizer()
            media_tok = self.info.media_token
            new_ids = [int(x) for x in token_ids]
            matched_updates: dict[str, list] = defaultdict(list)

            # Process modalities in registry order (one span per image).
            for modality, items in mm_prompt_updates.items():
                for item_idx, alts in enumerate(items):
                    if not alts:
                        raise RuntimeError(
                            f"no prompt updates for mm_items[{modality!r}][{item_idx}]"
                        )
                    update = alts[0]
                    content_tokens = list(
                        map(int, _seq2tokens(tokenizer, update.content.full))
                    )
                    span = _find_nth_media_token_span(
                        new_ids, tokenizer, media_tok, n=0
                    )
                    if span is None:
                        head = new_ids[:24]
                        raise RuntimeError(
                            "Failed to locate media placeholder "
                            f"{media_tok!r} in prompt_ids (len={len(new_ids)} "
                            f"head={head}). Ensure the chat prompt contains "
                            f"the literal {media_tok!r} string."
                        )
                    start, end = span
                    print(
                        f"[dsv4_moonvit] apply_updates modality={modality} "
                        f"item={item_idx} span=[{start}:{end}] "
                        f"replaced={new_ids[start:end]} -> n={len(content_tokens)}",
                        flush=True,
                    )
                    new_ids = new_ids[:start] + content_tokens + new_ids[end:]
                    matched_updates[modality].append([update])

            placeholders = self._find_mm_placeholders(
                new_ids,
                dict(matched_updates),
            )
            # Sanity: one PlaceholderFeaturesInfo per item.
            for modality, items in mm_prompt_updates.items():
                got = placeholders.get(modality, [])
                if len(got) != len(items):
                    raise RuntimeError(
                        f"Expected {len(items)} {modality} placeholders after "
                        f"expand, found {len(got)}; new_ids head={new_ids[:16]}"
                    )
            return new_ids, placeholders

        def _maybe_apply_prompt_updates(
            self,
            mm_items: Any,
            prompt_ids: list[int],
            mm_kwargs: Any,
            mm_prompt_updates: Any,
            is_update_applied: bool,
        ):
            print(
                f"[dsv4_moonvit] maybe_apply is_update_applied={is_update_applied} "
                f"prompt_ids={prompt_ids[:20]} len={len(prompt_ids)} "
                f"updates={list(mm_prompt_updates.keys()) if mm_prompt_updates else None}",
                flush=True,
            )
            try:
                return super()._maybe_apply_prompt_updates(
                    mm_items=mm_items,
                    prompt_ids=prompt_ids,
                    mm_kwargs=mm_kwargs,
                    mm_prompt_updates=mm_prompt_updates,
                    is_update_applied=is_update_applied,
                )
            except Exception as exc:
                tok = self.info.get_tokenizer()
                media_tok = self.info.media_token
                span = _find_nth_media_token_span(prompt_ids, tok, media_tok, n=0)
                print(
                    f"[dsv4_moonvit] diagnostic span={span} "
                    f"decode={tok.decode(prompt_ids)!r}",
                    flush=True,
                )
                raise exc

    class Dsv4MoonVitDummyInputsBuilder(BaseDummyInputsBuilder[Dsv4MoonVitProcessingInfo]):
        def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
            n = mm_counts.get("image", 0)
            return self.info.media_token * n

        def get_dummy_mm_data(
            self,
            seq_len: int,
            mm_counts: Mapping[str, int],
            mm_options: Mapping[str, Any] | None = None,
        ) -> Mapping[str, Any]:
            n = mm_counts.get("image", 0)
            # Prefer sizes that land near the training envelope without OOM.
            images = [
                Image.new("RGB", (448, 448), color=(128, 64, 32)) for _ in range(n)
            ]
            return {"image": images}

    return {
        "info": Dsv4MoonVitProcessingInfo,
        "processor": Dsv4MoonVitMultiModalProcessor,
        "dummy": Dsv4MoonVitDummyInputsBuilder,
        "MULTIMODAL_REGISTRY": mm["MULTIMODAL_REGISTRY"],
    }
