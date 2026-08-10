"""MoonViT tower load helpers (vLLM Kimi-K2.5 3D tower when available)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .projector import PatchMerger

logger = logging.getLogger(__name__)


def build_vision_config_object(vision_dict: dict[str, Any]) -> Any:
    """Build KimiK25VisionConfig if vLLM is importable; else a simple namespace."""
    try:
        from vllm.transformers_utils.configs.kimi_k25 import KimiK25VisionConfig

        # mm_hidden_size for Kimi projector linear_2 out dim (= LLM hidden)
        kwargs = dict(vision_dict)
        # Drop keys the config may not accept
        known = {
            "patch_size",
            "init_pos_emb_height",
            "init_pos_emb_width",
            "init_pos_emb_time",
            "pos_emb_type",
            "num_attention_heads",
            "num_hidden_layers",
            "hidden_size",
            "intermediate_size",
            "merge_kernel_size",
            "video_attn_type",
            "merge_type",
            "mm_projector_type",
            "mm_hidden_size",
            "projector_hidden_act",
            "projector_ln_eps",
        }
        filtered = {k: v for k, v in kwargs.items() if k in known}
        mk = filtered.get("merge_kernel_size", [2, 2])
        if isinstance(mk, list):
            filtered["merge_kernel_size"] = tuple(mk)
        return KimiK25VisionConfig(**filtered)
    except Exception:
        from types import SimpleNamespace

        ns = SimpleNamespace(**vision_dict)
        mk = getattr(ns, "merge_kernel_size", [2, 2])
        if isinstance(mk, list):
            ns.merge_kernel_size = tuple(mk)
        return ns


def load_tower_and_projector(
    *,
    vision_dict: dict[str, Any],
    tower_path: str | Path | None,
    projector_path: str | Path | None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[nn.Module | None, PatchMerger, dict[str, Any]]:
    """Construct tower (optional if path missing) + pure PatchMerger.

    The pure PatchMerger is always constructed so unit tests and the plugin
    share one weight path. The vLLM ``MoonViT3dPretrainedModel`` is used for
    the tower when importable; otherwise tower is None (math-only mode).
    """
    projector = PatchMerger(
        mm_hidden_size=int(vision_dict.get("hidden_size", 1152)),
        text_hidden_size=int(
            vision_dict.get("text_hidden_size")
            or vision_dict.get("mm_hidden_size")
            or 4096
        ),
        merge_kernel_size=tuple(vision_dict.get("merge_kernel_size") or (2, 2)),
        eps=float(vision_dict.get("projector_ln_eps", 1e-5)),
    )
    meta: dict[str, Any] = {"tower": None, "projector_loaded": False, "tower_loaded": False}

    if projector_path is not None and Path(projector_path).is_file():
        projector.load_webbrain_safetensors(projector_path, device=device, dtype=dtype)
        meta["projector_loaded"] = True
        meta["projector_path"] = str(projector_path)
    else:
        projector.to(device=device, dtype=dtype)

    tower: nn.Module | None = None
    if tower_path is not None and Path(tower_path).is_file():
        try:
            from vllm.model_executor.models.kimi_k25_vit import MoonViT3dPretrainedModel

            vcfg = build_vision_config_object(vision_dict)
            # Tower expects hidden_size 1152; mm_hidden_size is projector out.
            if hasattr(vcfg, "hidden_size"):
                pass
            tower = MoonViT3dPretrainedModel(vcfg, quant_config=None, prefix="vision_tower")
            missing, unexpected = _load_tower_weights(tower, tower_path)
            tower = tower.to(device=device, dtype=dtype)
            tower.eval()
            meta["tower"] = "MoonViT3dPretrainedModel"
            meta["tower_loaded"] = True
            meta["tower_path"] = str(tower_path)
            meta["tower_missing"] = missing[:8]
            meta["tower_unexpected"] = unexpected[:8]
            logger.info(
                "Loaded MoonViT tower from %s (missing=%d unexpected=%d)",
                tower_path,
                len(missing),
                len(unexpected),
            )
        except Exception as exc:
            logger.warning("Could not load vLLM MoonViT tower: %s", exc)
            meta["tower_error"] = str(exc)

    projector.eval()
    return tower, projector, meta


def _tp_copy_(param: torch.nn.Parameter, tensor: torch.Tensor) -> None:
    """Copy full checkpoint tensor into possibly TP-sharded param."""
    if param.data.shape == tensor.shape:
        param.data.copy_(tensor)
        return
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        tp = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
    except Exception:
        tp, rank = 1, 0

    if tp > 1 and param.data.ndim == tensor.ndim:
        # Column-parallel shard (output features on dim 0)
        if (
            param.data.shape[0] * tp == tensor.shape[0]
            and param.data.shape[1:] == tensor.shape[1:]
        ):
            param.data.copy_(tensor.chunk(tp, dim=0)[rank])
            return
        # Row-parallel shard (input features on dim 1)
        if (
            len(param.data.shape) > 1
            and param.data.shape[1] * tp == tensor.shape[1]
            and param.data.shape[0] == tensor.shape[0]
        ):
            param.data.copy_(tensor.chunk(tp, dim=1)[rank])
            return
        # Bias column-parallel
        if param.data.ndim == 1 and param.data.shape[0] * tp == tensor.shape[0]:
            param.data.copy_(tensor.chunk(tp, dim=0)[rank])
            return

    loader = getattr(param, "weight_loader", None)
    if loader is not None:
        loader(param, tensor)
        return
    raise ValueError(
        f"cannot load weight: param {tuple(param.data.shape)} vs file {tuple(tensor.shape)}"
    )


def _load_tower_weights(tower: nn.Module, path: str | Path) -> tuple[list[str], list[str]]:
    """Map WebBrain ``vision_tower.safetensors`` keys onto vLLM MoonViT3d.

    Uses each parameter's ``weight_loader`` when present so Column/RowParallel
    layers correctly shard full BF16 checkpoints under TP>1.
    """
    from safetensors.torch import load_file

    raw = load_file(str(path), device="cpu")
    params = dict(tower.named_parameters(remove_duplicate=False))
    buffers = dict(tower.named_buffers())
    loaded: set[str] = set()
    unmatched_file_keys: list[str] = []

    def _candidates(name: str) -> list[str]:
        key = name.replace("wqkv.", "attn.qkv_proj.").replace("wo.", "attn.proj.")
        outs = [
            key,
            name,
            key.removeprefix("vision_tower."),
            name.removeprefix("vision_tower."),
        ]
        seen: set[str] = set()
        ordered: list[str] = []
        for k in outs:
            if k not in seen:
                seen.add(k)
                ordered.append(k)
        return ordered

    for name, tensor in raw.items():
        target = None
        for cand in _candidates(name):
            if cand in params:
                target = cand
                break
            if cand in buffers:
                if buffers[cand].shape == tensor.shape:
                    buffers[cand].copy_(tensor)
                else:
                    _tp_copy_(buffers[cand], tensor)  # type: ignore[arg-type]
                loaded.add(cand)
                target = cand
                break
        if target is None:
            unmatched_file_keys.append(name)
            continue
        if target not in params:
            continue
        param = params[target]
        try:
            loader = getattr(param, "weight_loader", None)
            if loader is not None:
                loader(param, tensor)
            else:
                _tp_copy_(param, tensor)
        except Exception:
            _tp_copy_(param, tensor)
        loaded.add(target)

    required = set(params) | set(buffers)
    missing = sorted(required - loaded)
    return missing, unmatched_file_keys[:16]


@torch.inference_mode()
def encode_image_tokens(
    tower: nn.Module | None,
    projector: PatchMerger,
    pixel_values: torch.Tensor,
    grid_thw: list[int] | torch.Tensor,
) -> torch.Tensor:
    """pixels → tower → projector → ``(T, 4096)``."""
    if tower is None:
        raise RuntimeError("vision tower is not loaded")
    if isinstance(grid_thw, list):
        grid = torch.tensor([grid_thw], dtype=torch.long, device=pixel_values.device)
    else:
        grid = grid_thw
        if grid.ndim == 1:
            grid = grid.unsqueeze(0)
    device = next(tower.parameters()).device
    dtype = next(tower.parameters()).dtype
    pixel_values = pixel_values.to(device=device, dtype=dtype)
    grid = grid.to(device=device)

    # MoonViT3d forward already applies tpool_patch_merger → list[(N,4,C)] or tensor
    vt_out = tower(pixel_values, grid)
    if isinstance(vt_out, (list, tuple)):
        feats = []
        for chunk in vt_out:
            feats.append(projector(chunk.to(dtype=projector.pre_norm.weight.dtype)))
        return torch.cat(feats, dim=0)
    return projector(vt_out.to(dtype=projector.pre_norm.weight.dtype))
