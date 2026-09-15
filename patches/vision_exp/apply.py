"""Monkey-patch Anemll DeepseekV4ForCausalLM for Vision-Exp images.

Called at the end of ``vllm/models/deepseek_v4/nvidia/model.py`` after the
stock classes are defined. Video is not registered: the checkpoint has no
video tower.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .image_processor import (
    IMAGE,
    IMAGE_TOKEN_ID,
    current_routing_kind,
    is_unregistered_router_bias,
    is_vision_exp_weight_name,
    routing_kind_from_mm,
    vision_args_from_config,
)
from .processor import IMAGE_PLACEHOLDER, register_vision_exp_processor
from .vision import Aligner, ViT

VISION_MAPPER_PREFIXES = {
    "vision.": "model.vision.",
    "aligner.": "model.aligner.",
    "image_start": "model.image_start",
    "image_end": "model.image_end",
    "image_newline": "model.image_newline",
    "image_pad": "model.image_pad",
}
VISION_MAPPER_SUFFIXES = {
    ".ffn.gate.bias_vl": ".ffn.gate.e_score_correction_bias_vl",
}


def _as_rows(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return [value]
        return [value[i] for i in range(value.shape[0])]
    return [value]


def _scalar(value: Any) -> int:
    if hasattr(value, "reshape"):
        return int(value.reshape(-1)[0].item())
    return int(value)


def _extend_weights_mapper(mapper):
    from vllm.model_executor.models.utils import WeightsMapper

    extra = WeightsMapper(
        orig_to_new_prefix=dict(VISION_MAPPER_PREFIXES),
        orig_to_new_suffix=dict(VISION_MAPPER_SUFFIXES),
    )
    return mapper | extra


def _install_vision_tower(model: nn.Module, config) -> None:
    if getattr(model, "vision", None) is not None:
        return
    args = vision_args_from_config(config)
    if args.vision_n_layers <= 0:
        model.vision = None
        model.aligner = None
        return
    model.vision = ViT(args)
    model.aligner = Aligner(args)
    hidden = args.dim
    model.image_start = nn.Parameter(torch.empty(hidden))
    model.image_end = nn.Parameter(torch.empty(hidden))
    model.image_newline = nn.Parameter(torch.empty(hidden))
    model.image_pad = nn.Parameter(torch.empty(hidden))


@torch.inference_mode()
def encode_image(model: nn.Module, patches: torch.Tensor, n_vit_h: int, n_vit_w: int):
    return model.aligner(model.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)


def merge_one_image(
    model: nn.Module,
    patches: torch.Tensor,
    n_vit_h: int,
    n_vit_w: int,
    types: torch.Tensor,
    perm: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    types = types[:num_tokens].to(dtype=torch.long)
    perm = perm.to(dtype=torch.long)
    perm = perm[perm >= 0]
    n_patches = int(n_vit_h) * int(n_vit_w)
    patches = patches[:n_patches]
    device = next(model.vision.parameters()).device
    dtype = next(model.aligner.parameters()).dtype
    patches = patches.to(device=device, dtype=dtype)
    embeds = encode_image(model, patches, int(n_vit_h), int(n_vit_w))[perm.to(device)]
    params = torch.stack(
        [
            model.image_start,
            model.image_pad,
            model.image_pad,
            model.image_newline,
            model.image_end,
        ]
    ).to(device=device, dtype=embeds.dtype)
    block = params[types.to(device)]
    image_mask = types.to(device) == IMAGE
    if int(image_mask.sum()) != int(embeds.shape[0]):
        raise RuntimeError(
            f"Vision-Exp layout mismatch: {int(image_mask.sum())} IMAGE slots, "
            f"{int(embeds.shape[0])} aligner tokens"
        )
    block = block.clone()
    block[image_mask] = embeds.to(block.dtype)
    return block


def _mm_embed_rows(multimodal_embeddings: Any) -> int:
    if hasattr(multimodal_embeddings, "shape"):
        return int(multimodal_embeddings.shape[0])
    total = 0
    for part in multimodal_embeddings:
        total += _mm_embed_rows(part)
    return total


def _bias_data(param: Any) -> Any:
    if param is None:
        return None
    return param.data if hasattr(param, "data") else param


def fused_topk_bias_split_vl(
    *,
    hidden_states: Any,
    gating_output: Any,
    scoring_func: str,
    e_score_correction_bias: Any,
    e_score_correction_bias_vl: Any,
    topk: int,
    renormalize: bool,
    indices_type: Any,
    input_tokens: Any,
    hash_indices_table: Any,
    routed_scaling_factor: float,
    kind: Any = None,
    kind_cell: Any = None,
) -> tuple[Any, Any]:
    """Route image placeholder rows with bias_vl and no hash table (issue #175)."""
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        fused_topk_bias,
    )

    def _call(hs, go, bias, tokens, hash_tab):
        return fused_topk_bias(
            hidden_states=hs,
            gating_output=go,
            scoring_func=scoring_func,
            e_score_correction_bias=bias,
            topk=topk,
            renormalize=renormalize,
            indices_type=indices_type,
            input_tokens=tokens,
            hash_indices_table=hash_tab,
            routed_scaling_factor=routed_scaling_factor,
        )

    vl = _bias_data(e_score_correction_bias_vl)
    if kind is None:
        kind = current_routing_kind(input_tokens, kind_cell=kind_cell)
    if vl is None or kind == "text":
        return _call(
            hidden_states,
            gating_output,
            e_score_correction_bias,
            input_tokens,
            hash_indices_table,
        )
    if kind == "image":
        return _call(hidden_states, gating_output, vl, None, None)

    tokens = input_tokens.reshape(-1)
    n_rows = int(hidden_states.size(0))
    if int(tokens.numel()) != n_rows:
        return _call(
            hidden_states,
            gating_output,
            e_score_correction_bias,
            input_tokens,
            hash_indices_table,
        )
    image_mask = tokens == IMAGE_TOKEN_ID
    text_mask = ~image_mask
    w_text, id_text = _call(
        hidden_states[text_mask],
        gating_output[text_mask],
        e_score_correction_bias,
        tokens[text_mask],
        hash_indices_table,
    )
    w_img, id_img = _call(
        hidden_states[image_mask],
        gating_output[image_mask],
        vl,
        None,
        None,
    )
    topk_w = w_text.new_empty((n_rows, w_text.shape[1]))
    topk_id = id_text.new_empty((n_rows, id_text.shape[1]))
    topk_w[text_mask] = w_text
    topk_w[image_mask] = w_img
    topk_id[text_mask] = id_text
    topk_id[image_mask] = id_img
    return topk_w, topk_id


def _append_fused_shared_experts(router: Any, topk_weights: Any, topk_ids: Any):
    n = int(getattr(router, "num_fused_shared_experts", 0) or 0)
    if n <= 0:
        return topk_weights, topk_ids
    m = topk_ids.shape[0]
    base = router.global_num_experts
    shared_ids = torch.arange(
        base, base + n, dtype=topk_ids.dtype, device=topk_ids.device
    ).expand(m, n)
    shared_w = torch.full(
        (m, n),
        router.shared_expert_weight,
        dtype=topk_weights.dtype,
        device=topk_weights.device,
    )
    return (
        torch.cat([topk_weights, shared_w], dim=-1),
        torch.cat([topk_ids, shared_ids], dim=-1),
    )


def _wrap_router_compute_routing(router: Any, gate: Any) -> None:
    if getattr(router, "_dspark_bias_vl_wrapped", False):
        return
    if not hasattr(router, "_compute_routing"):
        return
    orig = router._compute_routing

    def _compute_routing(
        hidden_states,
        router_logits,
        indices_type,
        *,
        input_ids=None,
    ):
        vl = getattr(gate, "e_score_correction_bias_vl", None)
        # Graph capture cannot .item() / host-branch on token ids (issue #175).
        # Decode graphs only replay text tokens; image prefill stays eager.
        capturing = False
        try:
            capturing = bool(torch.cuda.is_current_stream_capturing())
        except Exception:
            capturing = False
        # Order matters: capture must never resolve a kind (no host sync,
        # no .item()) -- decode graphs only ever replay text tokens.
        if vl is None or capturing:
            return orig(
                hidden_states, router_logits, indices_type, input_ids=input_ids
            )
        kind = current_routing_kind(
            input_ids, kind_cell=getattr(router, "_dspark_routing_kind_cell", None)
        )
        if kind == "text":
            return orig(
                hidden_states, router_logits, indices_type, input_ids=input_ids
            )
        topk_weights, topk_ids = fused_topk_bias_split_vl(
            hidden_states=hidden_states,
            gating_output=router_logits,
            scoring_func=router.scoring_func,
            e_score_correction_bias=_bias_data(getattr(router, "e_score_correction_bias", None)),
            e_score_correction_bias_vl=vl,
            topk=router.top_k,
            renormalize=router.renormalize,
            indices_type=indices_type,
            input_tokens=input_ids,
            hash_indices_table=getattr(router, "_hash_indices_table", None),
            routed_scaling_factor=getattr(router, "routed_scaling_factor", 1.0),
            kind=kind,
        )
        return _append_fused_shared_experts(router, topk_weights, topk_ids)

    router._compute_routing = _compute_routing
    router._dspark_bias_vl_wrapped = True


def embed_multimodal(self, **kwargs: object):
    pixel_values = kwargs.get("pixel_values")
    if pixel_values is None:
        return []
    pixels = _as_rows(pixel_values)
    n_vit_h = _as_rows(kwargs.get("n_vit_h"))
    n_vit_w = _as_rows(kwargs.get("n_vit_w"))
    types = _as_rows(kwargs.get("types"))
    perm = _as_rows(kwargs.get("perm"))
    num_tokens = _as_rows(kwargs.get("num_tokens"))
    inner = self.model
    if getattr(inner, "vision", None) is None:
        raise RuntimeError("Vision-Exp tower was not constructed on DeepseekV4Model")
    out = []
    for i, patches in enumerate(pixels):
        out.append(
            merge_one_image(
                inner,
                patches,
                _scalar(n_vit_h[i]),
                _scalar(n_vit_w[i]),
                types[i],
                perm[i],
                _scalar(num_tokens[i]),
            )
        )
    return out


def attach_routing_kind_cell(root: nn.Module, kind_cell: list[Any]) -> int:
    """Share one model-local publication cell with its Vision-Exp MoE gates."""
    attached = 0
    for mod in root.modules():
        gate = getattr(mod, "gate", None)
        if gate is None or getattr(gate, "e_score_correction_bias_vl", None) is None:
            continue
        mod._dspark_routing_kind_cell = kind_cell
        router = getattr(getattr(mod, "experts", None), "router", None)
        if router is not None:
            router._dspark_routing_kind_cell = kind_cell
        attached += 1
    return attached


def _num_tokens(input_ids: Any) -> int:
    """Row count of a token tensor. ``numel`` is metadata: no host sync.

    Never guesses: a 0 here would make any non-empty placeholder count look
    like a full-image batch.
    """
    if hasattr(input_ids, "numel"):
        return int(input_ids.numel())
    try:
        return len(input_ids)
    except TypeError as exc:
        raise TypeError(
            f"Vision-Exp cannot size input_ids of type {type(input_ids).__name__}"
        ) from exc


def make_embed_input_ids(orig_lm_embed):
    """Wrap the stock ``embed_input_ids`` and classify the batch once (#175).

    This runs once per forward, ahead of every MoE layer, and already knows
    whether the step carries multimodal embeddings -- so a text-only step is
    classified with zero device->host syncs.
    """

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        *,
        is_multimodal: Any = None,
    ):
        # Default for every early return below: no mm embeddings == text.
        self._dspark_routing_kind_cell[0] = "text"
        text_embeds = orig_lm_embed(self, input_ids)
        if multimodal_embeddings is None:
            return text_embeds
        try:
            empty = len(multimodal_embeddings) == 0
        except TypeError:
            empty = False
        if empty:
            return text_embeds
        from vllm.model_executor.models.interfaces import _require_is_multimodal
        from vllm.model_executor.models.utils import _merge_multimodal_embeddings

        is_mm = _require_is_multimodal(is_multimodal)
        n_placeholders = (
            int(is_mm.sum().item()) if hasattr(is_mm, "sum") else int(is_mm)
        )
        n_embeds = _mm_embed_rows(multimodal_embeddings)
        if n_placeholders != n_embeds:
            raise ValueError(
                "Vision-Exp placeholder/embedding mismatch: "
                f"{n_embeds} multimodal tokens vs {n_placeholders} placeholders. "
                "Image block length depends on start_pos%4; a content-only encoder "
                "cache hit can reuse the wrong compress_pad (issue #172)."
            )
        # Reuses the sum above: the image path stays at one sync per forward.
        self._dspark_routing_kind_cell[0] = routing_kind_from_mm(
            n_placeholders, _num_tokens(input_ids)
        )
        return _merge_multimodal_embeddings(
            inputs_embeds=text_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_mm,
        )

    return embed_input_ids


@classmethod
def get_placeholder_str(cls, modality: str, i: int) -> str | None:
    if modality.startswith("image"):
        return IMAGE_PLACEHOLDER
    raise ValueError(
        f"DeepSeek-V4-Flash-Vision-Exp supports images only, got {modality!r}"
    )


def apply_vision_exp(
    *,
    DeepseekV4Model,
    DeepseekV4ForCausalLM,
    DeepseekV4MoE,
) -> None:
    from vllm.model_executor.models.interfaces import SupportsMultiModal

    orig_model_init = DeepseekV4Model.__init__

    def model_init(self, *, vllm_config, prefix: str = ""):
        orig_model_init(self, vllm_config=vllm_config, prefix=prefix)
        _install_vision_tower(self, vllm_config.model_config.hf_config)

    DeepseekV4Model.__init__ = model_init
    DeepseekV4Model.encode_image = lambda self, patches, n_h, n_w: encode_image(
        self, patches, n_h, n_w
    )

    orig_model_load = DeepseekV4Model.load_weights

    def model_load_weights(self, weights):
        try:
            from vllm.model_executor.model_loader.weight_utils import (
                default_weight_loader,
            )
        except ImportError:
            default_weight_loader = None
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()

        def _filtered():
            for name, weight in weights:
                if is_unregistered_router_bias(name, params_dict):
                    continue
                if not is_vision_exp_weight_name(name):
                    yield name, weight
                    continue
                if name not in params_dict:
                    raise KeyError(
                        f"Vision-Exp weight {name!r} has no module parameter; "
                        "the ViT/Aligner tower was not constructed"
                    )
                param = params_dict[name]
                loader = getattr(param, "weight_loader", None) or default_weight_loader
                if loader is not None:
                    loader(param, weight)
                else:
                    param.data.copy_(weight)
                loaded.add(name)

        loaded |= orig_model_load(self, _filtered())
        return loaded

    DeepseekV4Model.load_weights = model_load_weights

    orig_moe_init = DeepseekV4MoE.__init__

    def moe_init(self, vllm_config, prefix: str = ""):
        orig_moe_init(self, vllm_config, prefix)
        config = vllm_config.model_config.hf_config
        if getattr(config, "vision_n_layers", 0) > 0:
            self.gate.e_score_correction_bias_vl = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
            experts = getattr(self, "experts", None)
            router = getattr(experts, "router", None)
            if router is not None:
                _wrap_router_compute_routing(router, self.gate)

    DeepseekV4MoE.__init__ = moe_init

    orig_moe_forward = DeepseekV4MoE.forward

    def moe_forward(self, hidden_states, input_ids=None):
        vl = getattr(getattr(self, "gate", None), "e_score_correction_bias_vl", None)
        if not getattr(self, "use_mega_moe", False) or vl is None:
            return orig_moe_forward(self, hidden_states, input_ids)
        import vllm.models.deepseek_v4.nvidia.model as nvidia_mod

        prev = nvidia_mod.fused_topk_bias

        def _split_ftb(*args, **kwargs):
            if args:
                raise TypeError(
                    "issue #175 mega-MoE wrap expects fused_topk_bias keyword args"
                )
            return fused_topk_bias_split_vl(
                e_score_correction_bias_vl=vl,
                kind_cell=getattr(self, "_dspark_routing_kind_cell", None),
                **kwargs,
            )

        nvidia_mod.fused_topk_bias = _split_ftb
        try:
            return orig_moe_forward(self, hidden_states, input_ids)
        finally:
            nvidia_mod.fused_topk_bias = prev

    DeepseekV4MoE.forward = moe_forward

    orig_lm_init = DeepseekV4ForCausalLM.__init__

    def lm_init(self, *, vllm_config, prefix: str = ""):
        orig_lm_init(self, vllm_config=vllm_config, prefix=prefix)
        self.hf_to_vllm_mapper = _extend_weights_mapper(self.hf_to_vllm_mapper)
        self._dspark_routing_kind_cell = [None]
        attached = attach_routing_kind_cell(self, self._dspark_routing_kind_cell)
        if getattr(vllm_config.model_config.hf_config, "vision_n_layers", 0) > 0:
            if attached == 0:
                raise RuntimeError(
                    "Vision-Exp found no bias_vl MoE gate to bind to this model; "
                    "the per-forward routing kind would never be consumed and "
                    "every gate would host-sync again (issue #175)"
                )

    DeepseekV4ForCausalLM.__init__ = lm_init
    DeepseekV4ForCausalLM.hf_to_vllm_mapper = _extend_weights_mapper(
        DeepseekV4ForCausalLM.hf_to_vllm_mapper
    )

    orig_lm_embed = DeepseekV4ForCausalLM.embed_input_ids

    DeepseekV4ForCausalLM.embed_input_ids = make_embed_input_ids(orig_lm_embed)

    if SupportsMultiModal not in DeepseekV4ForCausalLM.__mro__:
        DeepseekV4ForCausalLM.__bases__ = (
            DeepseekV4ForCausalLM.__bases__[0],
            SupportsMultiModal,
            *DeepseekV4ForCausalLM.__bases__[1:],
        )
    DeepseekV4ForCausalLM.supports_multimodal = True
    DeepseekV4ForCausalLM.supports_multimodal_raw_input_only = False
    # Hash MoE (layers 0–2) looks up tid2eid with input_ids. The MM runner
    # otherwise sets input_ids=None whenever inputs_embeds is present.
    # Issue #175 also needs raw ids so image placeholders can take bias_vl.
    DeepseekV4ForCausalLM.requires_raw_input_tokens = True
    DeepseekV4ForCausalLM.get_placeholder_str = get_placeholder_str
    DeepseekV4ForCausalLM.embed_multimodal = embed_multimodal

    # SupportsMultiModal.get_language_model() returns the first child with
    # embed_input_ids (DeepseekV4Model). DSpark/Eagle3 then read `.model` on
    # that child. This class *is* the LM wrapper, same as 0731.
    def get_language_model(self):
        return self

    def set_aux_hidden_state_layers(self, layers):
        self.model._set_aux_hidden_state_layers(layers)

    def get_eagle3_default_aux_hidden_state_layers(self):
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    DeepseekV4ForCausalLM.get_language_model = get_language_model
    DeepseekV4ForCausalLM.set_aux_hidden_state_layers = set_aux_hidden_state_layers
    DeepseekV4ForCausalLM.get_eagle3_default_aux_hidden_state_layers = (
        get_eagle3_default_aux_hidden_state_layers
    )
    register_vision_exp_processor(DeepseekV4ForCausalLM)
