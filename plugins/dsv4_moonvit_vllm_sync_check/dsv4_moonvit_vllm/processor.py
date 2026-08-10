"""vLLM multimodal processor: OpenAI image parts → MoonViT pixel_values + <image> expand."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
from PIL import Image

from .config import deepseek_vision_dict, image_token_id, max_image_tokens
from .preprocess import (
    default_media_proc_cfg,
    navit_resize_image,
    pil_to_pixel_values_and_grid,
)

logger = logging.getLogger(__name__)

IMAGE_PLACEHOLDER = "<image>"


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
            self.max_image_tokens = int(
                adapter.get("max_image_tokens", max_image_tokens())
            )
            self.media_proc_cfg = default_media_proc_cfg(self.max_image_tokens)

        def get_supported_mm_limits(self) -> Mapping[str, int | None]:
            return {"image": 1}

        def get_mm_max_tokens_per_item(
            self, seq_len: int, mm_counts: Mapping[str, int]
        ) -> Mapping[str, int]:
            return {"image": self.max_image_tokens}

        def image_token_count(self, image: Image.Image) -> int:
            w, h = image.size
            cfg = self.media_proc_cfg
            merge = cfg["merge_kernel_size"]
            merge_k = int(merge[0] if isinstance(merge, (list, tuple)) else merge)
            r = navit_resize_image(
                w,
                h,
                patch_size=int(cfg["patch_size"]),
                merge_kernel_size=merge_k,
                in_patch_limit=int(cfg["in_patch_limit"]),
                patch_limit_on_one_side=int(cfg["patch_limit_on_one_side"]),
                fixed_output_tokens=cfg.get("fixed_output_tokens"),
                max_image_tokens=self.max_image_tokens,
            )
            return min(r.num_tokens, self.max_image_tokens)

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
                grid_thws=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
            )

        def _call_hf_processor(
            self,
            prompt: str,
            mm_data: Mapping[str, object],
            mm_kwargs: Mapping[str, object],
            tok_kwargs: Mapping[str, object],
        ) -> Any:
            images = mm_data.get("image") or []
            if not isinstance(images, list):
                images = [images]
            if len(images) > 1:
                raise ValueError(
                    "DeepSeek V4 MoonViT v1 accepts at most one image per request"
                )

            tokenizer = self.info.get_tokenizer()
            # Encode text around <image> placeholders without requiring the
            # sentinel to exist in the vocab.
            parts = prompt.split(self.info.media_token)
            # If the chat path did not insert <image>, append one when images exist.
            if len(images) == 1 and len(parts) == 1:
                # Prefer replacing nothing — vLLM PromptReplacement expands target.
                # Put a single placeholder at the end of the user content.
                prompt = prompt + self.info.media_token
                parts = prompt.split(self.info.media_token)

            if len(parts) - 1 != len(images):
                raise ValueError(
                    f"prompt has {len(parts) - 1} {self.info.media_token!r} "
                    f"placeholder(s) but request has {len(images)} image(s)"
                )

            input_ids: list[int] = []
            pixel_chunks: list[torch.Tensor] = []
            grids: list[list[int]] = []
            for idx, part in enumerate(parts):
                if part:
                    input_ids.extend(
                        tokenizer.encode(part, add_special_tokens=False)
                    )
                if idx < len(images):
                    img = images[idx]
                    if not isinstance(img, Image.Image):
                        # may already be array-like
                        img = Image.fromarray(img) if hasattr(img, "shape") else img
                    pv, grid, ntok = pil_to_pixel_values_and_grid(
                        img,
                        max_image_tokens=self.info.max_image_tokens,
                        media_proc_cfg=self.info.media_proc_cfg,
                    )
                    if ntok > self.info.max_image_tokens:
                        raise ValueError(
                            f"image expands to {ntok} tokens > "
                            f"max_image_tokens={self.info.max_image_tokens}"
                        )
                    pixel_chunks.append(pv)
                    grids.append(grid)
                    # Single sentinel; PromptReplacement expands to ntok ids.
                    input_ids.append(self.info.media_token_id)

            data: dict[str, Any] = {"input_ids": torch.tensor([input_ids], dtype=torch.long)}
            if pixel_chunks:
                data["pixel_values"] = torch.cat(pixel_chunks, dim=0)
                data["grid_thws"] = torch.tensor(grids, dtype=torch.long)
            return BatchFeature(data=data)

        def _get_prompt_updates(
            self,
            mm_items: Any,
            hf_processor_mm_kwargs: Mapping[str, Any],
            out_mm_kwargs: Any,
        ) -> Sequence[Any]:
            media_token_id = self.info.media_token_id

            def get_replacement(item_idx: int):
                images = mm_items.get_items("image", (ImageProcessorItems,))
                n = self.info.image_token_count(images[item_idx])
                return [media_token_id] * n

            return [
                PromptReplacement(
                    modality="image",
                    target=[media_token_id],
                    replacement=get_replacement,
                ),
            ]

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
