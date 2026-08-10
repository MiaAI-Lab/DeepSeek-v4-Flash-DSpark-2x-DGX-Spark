"""DeepseekV4MoonVitForCausalLM — native MoonViT multimodal wrapper for vLLM + DSpark."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .config import (
    ENV_PROJECTOR,
    ENV_TOWER,
    deepseek_vision_dict,
    image_token_id,
    resolve_projector_path,
    resolve_tower_path,
    routing_palette,
    vision_config_dict,
)
from .moonvit import encode_image_tokens, load_tower_and_projector
from .projector import PatchMerger
from .routing import apply_palette_cycle
from .wrapper import TransparentLanguageModelProxy, assert_dspark_transparency

logger = logging.getLogger(__name__)


def _find_weights_in_model_dir(model_path: str | None) -> tuple[Path | None, Path | None]:
    if not model_path:
        return None, None
    root = Path(model_path)
    if not root.exists():
        # HF id — skip
        return None, None
    tower = root / "vision_tower.safetensors"
    proj = root / "mm_projector.safetensors"
    return (tower if tower.is_file() else None, proj if proj.is_file() else None)


try:
    from vllm.model_executor.models.interfaces import (
        SupportsEagle3 as _SupportsEagle3,
        SupportsMultiModal as _SupportsMultiModal,
    )
except Exception:  # pragma: no cover - unit tests without vLLM
    class _SupportsMultiModal:  # type: ignore[no-redef]
        pass

    class _SupportsEagle3:  # type: ignore[no-redef]
        pass


class DeepseekV4MoonVitForCausalLM(
    TransparentLanguageModelProxy, nn.Module, _SupportsMultiModal, _SupportsEagle3
):
    """Wraps DeepseekV4ForCausalLM with in-process MoonViT + PatchMerger.

    DSpark transparency:
    - ``forward(..., **kwargs)`` delegates to the language model
    - ``lm_head`` / unknown attrs proxy to ``language_model``
    - palette-cycle rewrites image placeholder IDs so hash MoE keeps valid routes
    """

    supports_multimodal: bool = True
    supports_multimodal_raw_input_only: bool = False
    # Replicate full MoonViT on each rank when --mm-encoder-tp-mode data
    # (required for correct BF16 WebBrain tower load under TP=2).
    supports_encoder_tp_data: bool = True
    requires_raw_input_tokens: bool = False
    supports_eagle3: bool = True
    _has_oov_mm_tokens: bool = True
    _language_model_names: list[str] = ["language_model"]
    _tower_model_names: list[str] = ["vision_tower"]

    def __init__(self, vllm_config: Any, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.vllm_config = vllm_config
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        self.config = hf_config
        self.image_token_id = image_token_id(hf_config)
        self.route_palette = routing_palette(hf_config)
        # Device-resident palette for CUDA-graph-safe routing rewrites.
        self.register_buffer(
            "route_palette_tensor",
            torch.tensor(self.route_palette, dtype=torch.long),
            persistent=False,
        )
        adapter = deepseek_vision_dict(hf_config)
        self.max_image_tokens = int(adapter.get("max_image_tokens", 512))

        # Language model first (weights loaded via load_weights later).
        from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=hf_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["DeepseekV4ForCausalLM"],
            )

        vision_dict = vision_config_dict(hf_config)
        model_path = getattr(model_config, "model", None) or getattr(
            model_config, "model_path", None
        )
        dir_tower, dir_proj = _find_weights_in_model_dir(model_path)
        tower_path = resolve_tower_path() or dir_tower
        proj_path = resolve_projector_path() or dir_proj

        # Prefer the worker GPU. Loading on CPU leaves MoonViT FLASH_ATTN on the
        # CPU backend and crashes at encode time (flash_attn_maxseqlen_wrapper).
        try:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        except Exception:
            device = torch.device("cpu")
        dtype = getattr(model_config, "dtype", torch.bfloat16)
        if dtype is None:
            dtype = torch.bfloat16

        with self._mark_tower_model(vllm_config, "image"):
            tower_mod, projector_mod, vision_meta = load_tower_and_projector(
                vision_dict=vision_dict,
                tower_path=tower_path,
                projector_path=proj_path,
                device=device,
                dtype=dtype if isinstance(dtype, torch.dtype) else torch.bfloat16,
            )
            # Assign via object.__setattr__ path Module understands; avoid
            # reading self.vision_tower before it exists (proxy __getattr__).
            if tower_mod is None:
                tower_mod = nn.Identity()
            # Ensure CUDA even if loader left modules on CPU (e.g. partial load).
            if device.type == "cuda":
                tower_mod = tower_mod.to(device=device)
                projector_mod = projector_mod.to(device=device)
            self.vision_tower = tower_mod
            self.mm_projector = projector_mod
            self._vision_meta = vision_meta
            vision_meta["device"] = str(device)

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        logger.info(
            "DeepseekV4MoonVitForCausalLM ready: tower=%s projector=%s image_token_id=%s",
            self._vision_meta.get("tower"),
            self._vision_meta.get("projector_loaded"),
            self.image_token_id,
        )

    # --- SupportsMultiModal hooks -------------------------------------------------

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return "<image>"
        raise ValueError(f"Unsupported modality: {modality}")

    def embed_multimodal(self, **kwargs: object) -> list[torch.Tensor] | None:
        pixel_values = kwargs.get("pixel_values")
        grid_thws = kwargs.get("grid_thws")
        if pixel_values is None:
            return None
        if isinstance(pixel_values, list):
            pixel_values = torch.cat(pixel_values, dim=0)
        if not isinstance(grid_thws, torch.Tensor):
            grid_thws = torch.as_tensor(grid_thws)

        if isinstance(self.vision_tower, nn.Identity):
            raise RuntimeError(
                "MoonViT tower failed to load; set DSV4_MOONVIT_TOWER or place "
                "vision_tower.safetensors in the model directory"
            )

        # Flatten possible batch layouts
        if pixel_values.ndim == 5:
            # (B, N, C, H, W)
            b, n = pixel_values.shape[:2]
            pixel_values = pixel_values.reshape(b * n, *pixel_values.shape[2:])
        grid_thws = grid_thws.reshape(-1, grid_thws.shape[-1])

        device = next(self.vision_tower.parameters()).device
        dtype = next(self.vision_tower.parameters()).dtype
        pixel_values = pixel_values.to(device=device, dtype=dtype)
        grid_thws = grid_thws.to(device=device)

        # Encode each image separately to preserve token boundaries.
        embeddings: list[torch.Tensor] = []
        patch_offset = 0
        for i in range(grid_thws.shape[0]):
            t, h, w = [int(x) for x in grid_thws[i].tolist()]
            n_patches = t * h * w
            pv = pixel_values[patch_offset : patch_offset + n_patches]
            patch_offset += n_patches
            emb = encode_image_tokens(
                self.vision_tower,
                self.mm_projector if isinstance(self.mm_projector, PatchMerger) else self.mm_projector,
                pv,
                [t, h, w],
            )
            if emb.shape[0] > self.max_image_tokens:
                emb = emb[: self.max_image_tokens]
            print(
                f"[dsv4_moonvit] encode_image grid={[t,h,w]} "
                f"pv={tuple(pv.shape)} emb={tuple(emb.shape)} "
                f"mean={float(emb.float().mean()):.4f} "
                f"std={float(emb.float().std()):.4f} "
                f"norm={float(emb.float().norm()):.2f}",
                flush=True,
            )
            embeddings.append(emb)
        return embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.models.utils import _merge_multimodal_embeddings
        from vllm.model_executor.models.interfaces import _require_is_multimodal

        # Mask OOV image tokens before text embedding lookup.
        routed_ids = apply_palette_cycle(
            input_ids,
            image_token_id=self.image_token_id,
            palette=self.route_palette_tensor,
            clone=True,
        )
        # For embedding lookup, use zeros (or first palette id) at image slots so
        # embed_tokens never sees OOV 129280; multimodal embeds overwrite them.
        # After palette_cycle, image slots are valid in-vocab route IDs, so
        # embed_tokens is safe without masked_fill (avoids CPU→CUDA copies that
        # break CUDA graph capture during warmup when is_multimodal is on CPU).
        text_ids = routed_ids
        if is_multimodal is not None and is_multimodal.device == text_ids.device:
            text_ids = text_ids.masked_fill(is_multimodal, 0)

        inputs_embeds = self.language_model.embed_input_ids(text_ids)

        if multimodal_embeddings is None or (
            hasattr(multimodal_embeddings, "__len__") and len(multimodal_embeddings) == 0
        ):
            return inputs_embeds

        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=_require_is_multimodal(is_multimodal),
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> Any:
        # Hash MoE requires valid in-vocab route IDs. Memory profiling and some
        # multimodal paths pass embeds without token ids — synthesize zeros.
        if input_ids is None and inputs_embeds is not None:
            n = inputs_embeds.shape[0]
            # Hash MoE kernels expect int32 route ids (not int64).
            input_ids = torch.zeros(
                n, dtype=torch.int32, device=inputs_embeds.device
            )
        # Hash MoE requires valid in-vocab route IDs on image positions.
        if input_ids is not None:
            want_dtype = input_ids.dtype
            input_ids = apply_palette_cycle(
                input_ids,
                image_token_id=self.image_token_id,
                palette=self.route_palette_tensor,
                clone=True,
            )
            if input_ids.dtype != want_dtype:
                input_ids = input_ids.to(dtype=want_dtype)
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return self.language_model.compute_logits(hidden_states, **kwargs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load language weights; vision is loaded from explicit safetensors paths."""

        def language_weights():
            for name, tensor in weights:
                if name.startswith("vision_tower.") or name.startswith("mm_projector."):
                    continue
                yield name.removeprefix("language_model."), tensor

        return self.language_model.load_weights(language_weights())

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        # Prefer protocol default (walk into language_model.model Eagle mixin).
        try:
            from vllm.model_executor.models.interfaces import SupportsEagle3

            return SupportsEagle3.set_aux_hidden_state_layers(self, layers)
        except Exception:
            lm = self.language_model
            if hasattr(lm, "set_aux_hidden_state_layers"):
                return lm.set_aux_hidden_state_layers(layers)
            if hasattr(lm, "model") and hasattr(lm.model, "_set_aux_hidden_state_layers"):
                return lm.model._set_aux_hidden_state_layers(layers)
            raise

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        try:
            from vllm.model_executor.models.interfaces import SupportsEagle3

            return SupportsEagle3.get_eagle3_default_aux_hidden_state_layers(self)
        except Exception:
            lm = self.language_model
            if hasattr(lm, "get_eagle3_default_aux_hidden_state_layers"):
                return lm.get_eagle3_default_aux_hidden_state_layers()
            if hasattr(lm, "get_eagle3_aux_hidden_state_layers"):
                return lm.get_eagle3_aux_hidden_state_layers()
            return ()

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return self.get_eagle3_default_aux_hidden_state_layers()

    # SupportsMultiModal mark helpers (vLLM 0.25+)
    def _mark_language_model(self, vllm_config: Any):
        from vllm.model_executor.models.interfaces import SupportsMultiModal

        return SupportsMultiModal._mark_language_model(self, vllm_config)

    def _mark_tower_model(self, vllm_config: Any, modality: str):
        from vllm.model_executor.models.interfaces import SupportsMultiModal

        return SupportsMultiModal._mark_tower_model(self, vllm_config, modality)

    def transparency_report(self) -> dict[str, bool]:
        return assert_dspark_transparency(self)


# Register multimodal processor when module is imported under vLLM.
def _try_register_multimodal() -> None:
    try:
        from vllm.model_executor.models.registry import ModelRegistry

        from .processor import build_processor_classes

        classes = build_processor_classes()
        MULTIMODAL_REGISTRY = classes["MULTIMODAL_REGISTRY"]

        MULTIMODAL_REGISTRY.register_processor(
            classes["processor"],
            info=classes["info"],
            dummy_inputs=classes["dummy"],
        )(DeepseekV4MoonVitForCausalLM)

        ModelRegistry.register_model(
            "DeepseekV4MoonVitForCausalLM",
            "dsv4_moonvit_vllm.model:DeepseekV4MoonVitForCausalLM",
        )
        logger.info("Registered DeepseekV4MoonVitForCausalLM with multimodal processor")
    except Exception as exc:
        logger.debug("Multimodal registration deferred/failed: %s", exc)


_try_register_multimodal()
