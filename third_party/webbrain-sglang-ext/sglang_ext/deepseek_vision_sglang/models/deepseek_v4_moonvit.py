from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.configs.kimi_k25 import KimiK25VisionConfig
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v4 import (
    DeepseekV4ForCausalLM as SGLangDeepseekV4ForCausalLM,
)
from sglang.srt.models.kimi_k25 import (
    K2VLMultiModalProjector,
    MoonViT3dPretrainedModel,
    mm_projection_auto,
)
from sglang.srt.multimodal.mm_utils import materialize_multimodal_features
from sglang.srt.runtime_context import get_parallel, get_server_args

from deepseek_vision_sglang.routing import ROUTING_IDS_ATTR, build_sglang_routing_ids


def _config_dict(config, name: str) -> dict:
    value = getattr(config, name, None)
    if value is None:
        raise ValueError(f"config.json is missing {name}")
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"config.json field {name} must be an object")


class RoutingAwareDeepseekV4ForCausalLM(SGLangDeepseekV4ForCausalLM):
    """Recover route IDs when SGLang's multimodal routine supplies embeddings."""

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if input_ids is None:
            input_ids = getattr(forward_batch, ROUTING_IDS_ATTR, None)
        if input_ids is None:
            raise RuntimeError("DeepSeek V4 routing IDs were not attached to the forward batch")
        return super().forward(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )


class DeepseekV4ForCausalLM(nn.Module):
    """MoonViT wrapper that deliberately overwrites SGLang's text-only V4 entry.

    Keeping the architecture name unchanged is required: SGLang's DeepSeek V4
    attention, FP4 expert-layout, and memory-pool selection all key off this exact
    architecture string. ``SGLANG_EXTERNAL_MODEL_PACKAGE`` performs the overwrite.
    """

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        adapter = _config_dict(config, "deepseek_vision")
        vision_config = KimiK25VisionConfig(**_config_dict(config, "vision_config"))
        self.route_palette = tuple(int(value) for value in adapter["routing_palette"])
        if not self.route_palette:
            raise ValueError("deepseek_vision.routing_palette cannot be empty")
        if min(self.route_palette) < 0 or max(self.route_palette) >= int(config.vocab_size):
            raise ValueError("deepseek_vision.routing_palette contains an invalid token ID")
        if int(vision_config.text_hidden_size) != int(config.hidden_size):
            raise ValueError("vision projector output size differs from DeepSeek hidden size")
        # SGLang v0.5.16 exposes the resolved multimodal flags through
        # ServerArgs.  Newer development builds also provide get_mm(), but the
        # released container used by HF Inference Endpoints does not.  Reading
        # this stable ServerArgs field keeps the external model compatible with
        # both layouts.
        if get_server_args().mm_enable_dp_encoder:
            raise NotImplementedError(
                "DeepSeek V4 + MoonViT has not validated SGLang encoder data parallelism"
            )
        if get_parallel().pp_size > 1:
            raise NotImplementedError(
                "DeepSeek V4 + MoonViT has not validated pipeline parallelism"
            )

        self.vision_tower = MoonViT3dPretrainedModel(
            vision_config,
            use_data_parallel=False,
            prefix="vision_tower",
        )
        self.mm_projector = K2VLMultiModalProjector(vision_config)
        self.language_model = RoutingAwareDeepseekV4ForCausalLM(
            config,
            quant_config,
            prefix=prefix,
        )
        if hasattr(self.language_model, "dtype"):
            target_dtype = self.language_model.dtype
            self.vision_tower = self.vision_tower.to(dtype=target_dtype)
            self.mm_projector = self.mm_projector.to(dtype=target_dtype)
        self._loaded_vision_params: set[str] = set()

    @property
    def model(self):
        return self.language_model

    def __setattr__(self, name, value):
        if name == "model":
            return
        super().__setattr__(name, value)

    @property
    def pp_group(self):
        return self.language_model.pp_group

    @property
    def start_layer(self) -> int:
        return self.language_model.start_layer

    @property
    def end_layer(self) -> int:
        return self.language_model.end_layer

    @property
    def routed_experts_weights_of_layer(self):
        return self.language_model.routed_experts_weights_of_layer

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def _materialize_image_features(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        device = self.vision_tower.device
        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        parallel = get_parallel()
        server_args = get_server_args()
        consumer_count = max(getattr(server_args, "tp_size", parallel.attn_tp_size), 1)
        device_index = device.index
        if device.type == "cuda" and device_index is None:
            device_index = torch.cuda.current_device()
        features = []
        for item in items:
            if device.type == "cuda":
                item.reconstruct(device_index, ipc_consumer_count=consumer_count)
            features.append(item.feature)
        return materialize_multimodal_features(features, device=device, dtype=dtype)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        grid_thws = []
        for item in items:
            grid = item.model_specific_data.get("image_grid_thw")
            if grid is None:
                grid = item.model_specific_data["grid_thws"]
            grid_thws.append(grid)
        grid_thws = torch.concat(grid_thws, dim=0)
        pixel_values = self._materialize_image_features(items)
        image_features = self.vision_tower(pixel_values, grid_thws)
        # ``mm_projection_auto`` preserves per-image boundaries and therefore
        # returns a tuple of tensors.  SGLang's embedding cache expects the
        # data-embedding callback to return one packed tensor and performs its
        # own split using each item's placeholder span.
        return torch.cat(mm_projection_auto(self.mm_projector, image_features), dim=0)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(
            input_ids, mm_inputs
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        if get_embedding:
            raise NotImplementedError("embedding mode is not supported by this generative adapter")
        routing_ids = build_sglang_routing_ids(input_ids, forward_batch, self.route_palette)
        setattr(forward_batch, ROUTING_IDS_ATTR, routing_ids)
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            multimodal_model=self,
            data_embedding_funcs={Modality.IMAGE: self.get_image_feature},
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters(remove_duplicate=False))

        def load_component(name: str, loaded_weight: torch.Tensor) -> bool:
            tower_name = name.removeprefix("vision_tower.")
            tower_name = tower_name.replace("wqkv.", "attn.qkv_proj.").replace(
                "wo.", "attn.proj."
            )
            tower_param = f"vision_tower.{tower_name}"
            projector_name = name.removeprefix("mm_projector.")
            projector_name = projector_name.replace("proj.0", "linear_1").replace(
                "proj.2", "linear_2"
            )
            projector_param = f"mm_projector.{projector_name}"
            target = None
            if tower_param in params:
                target = tower_param
            elif projector_param in params:
                target = projector_param
            if target is None:
                return False
            param = params[target]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, loaded_weight)
            self._loaded_vision_params.add(target)
            return True

        def language_weights():
            for name, loaded_weight in weights:
                if load_component(name, loaded_weight):
                    continue
                yield name.removeprefix("language_model."), loaded_weight

        self.language_model.load_weights(language_weights())

        required = {
            name
            for name in params
            if name.startswith("vision_tower.") or name.startswith("mm_projector.")
        }
        missing = sorted(required - self._loaded_vision_params)
        if missing:
            sample = ", ".join(missing[:8])
            raise RuntimeError(
                f"vision_tower.safetensors/mm_projector.safetensors did not initialize "
                f"{len(missing)} parameter(s): {sample}"
            )

    def post_load_weights(self):
        self.language_model.post_load_weights()

    @property
    def stacked_params_mapping(self):
        return getattr(self.language_model, "stacked_params_mapping", [])

    @property
    def expert_params_mapping(self):
        return getattr(self.language_model, "expert_params_mapping", [])

    def mutate_weight_preload(self, name):
        return self.language_model.mutate_weight_preload(name)

    def custom_scale_remap(self, name):
        return self.language_model.custom_scale_remap(name)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=None,
        )


EntryClass = [DeepseekV4ForCausalLM]
