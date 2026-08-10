"""Vision config helpers for DeepSeek-V4 + MoonViT overlay."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Inline constants so host scripts (no torch) can import this module.
IMAGE_TOKEN_ID_DEFAULT = 129280
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

TOWER_SHA256 = "1382c41f1a4afc91791ade630e2b1e1cef68cc5a1e09668a45970a5d5e1b8f15"
PROJECTOR_SHA256 = "7024d9d5c9714c7abbc09abda015f083b7d7b107745eb78879f019bf4721577a"

ENV_TOWER = "DSV4_MOONVIT_TOWER"
ENV_PROJECTOR = "DSV4_MOONVIT_PROJECTOR"
ENV_MAX_IMAGE_TOKENS = "DSV4_MOONVIT_MAX_IMAGE_TOKENS"
ENV_ENABLED = "DSV4_MOONVIT_ENABLED"


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


def max_image_tokens() -> int:
    return int(os.environ.get(ENV_MAX_IMAGE_TOKENS, "512"))


def resolve_tower_path(explicit: str | None = None) -> Path | None:
    path = explicit or os.environ.get(ENV_TOWER)
    if path:
        return Path(path)
    return None


def resolve_projector_path(explicit: str | None = None) -> Path | None:
    path = explicit or os.environ.get(ENV_PROJECTOR)
    if path:
        return Path(path)
    return None


def deepseek_vision_dict(cfg: Any | None = None) -> dict[str, Any]:
    if cfg is None:
        return {
            "schema_version": 1,
            "tower_model_id": "moonshotai/Kimi-K2.6",
            "tower_revision": "7eb5002f6aadc958aed6a9177b7ed26bb94011bb",
            "image_placeholder": "<image>",
            "image_placeholder_token_id": IMAGE_TOKEN_ID_DEFAULT,
            "max_image_tokens": max_image_tokens(),
            "routing_policy": "palette_cycle",
            "routing_palette": list(DEFAULT_ROUTING_PALETTE),
        }
    value = getattr(cfg, "deepseek_vision", None)
    if value is None and isinstance(cfg, dict):
        value = cfg.get("deepseek_vision")
    if value is None:
        return deepseek_vision_dict(None)
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return dict(value)


def vision_config_dict(cfg: Any | None = None) -> dict[str, Any]:
    default = {
        "model_type": "kimi_k25",
        "patch_size": 14,
        "init_pos_emb_height": 64,
        "init_pos_emb_width": 64,
        "init_pos_emb_time": 4,
        "pos_emb_type": "divided_fixed",
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "vt_num_attention_heads": 16,
        "vt_num_hidden_layers": 27,
        "vt_hidden_size": 1152,
        "vt_intermediate_size": 4304,
        "merge_kernel_size": [2, 2],
        "video_attn_type": "spatial_temporal",
        "merge_type": "sd2_tpool",
        "mm_projector_type": "patchmerger",
        "mm_hidden_size": 4096,  # LLM inject dim (WebBrain projector out)
        "projector_hidden_act": "gelu",
        "projector_ln_eps": 1e-05,
        "text_hidden_size": 4096,
    }
    if cfg is None:
        return default
    value = getattr(cfg, "vision_config", None)
    if value is None and isinstance(cfg, dict):
        value = cfg.get("vision_config")
    if value is None:
        return default
    if isinstance(value, dict):
        out = dict(default)
        out.update(value)
        # Ensure projector targets LLM hidden size, not tower width.
        text_hs = out.get("text_hidden_size", 4096)
        # WebBrain ships mm_hidden_size=1152 for tower; projector Linear out is 4096.
        # Our PatchMerger / Kimi projector linear_2 uses mm_hidden_size as out dim.
        out["mm_hidden_size"] = int(text_hs)
        return out
    if hasattr(value, "to_dict"):
        out = dict(default)
        out.update(value.to_dict())
        out["mm_hidden_size"] = int(out.get("text_hidden_size", 4096))
        return out
    return default


def image_token_id(cfg: Any | None = None) -> int:
    if cfg is None:
        return IMAGE_TOKEN_ID_DEFAULT
    for key in ("image_token_id", "media_placeholder_token_id"):
        val = getattr(cfg, key, None)
        if val is None and isinstance(cfg, dict):
            val = cfg.get(key)
        if val is not None:
            return int(val)
    adapter = deepseek_vision_dict(cfg)
    return int(adapter.get("image_placeholder_token_id", IMAGE_TOKEN_ID_DEFAULT))


def routing_palette(cfg: Any | None = None) -> tuple[int, ...]:
    adapter = deepseek_vision_dict(cfg)
    pal = adapter.get("routing_palette") or list(DEFAULT_ROUTING_PALETTE)
    return tuple(int(x) for x in pal)


def merge_vision_into_config_json(
    text_config: dict[str, Any],
    vision_source: dict[str, Any] | None = None,
    *,
    architecture: str = "DeepseekV4MoonVitForCausalLM",
) -> dict[str, Any]:
    """Build overlay config.json from official 0731 + WebBrain vision fields."""
    out = dict(text_config)
    if vision_source is None:
        vision_source = {
            "vision_config": vision_config_dict(None),
            "deepseek_vision": deepseek_vision_dict(None),
            "image_token_id": IMAGE_TOKEN_ID_DEFAULT,
            "media_placeholder_token_id": IMAGE_TOKEN_ID_DEFAULT,
        }
    for key in (
        "vision_config",
        "deepseek_vision",
        "image_token_id",
        "media_placeholder_token_id",
    ):
        if key in vision_source:
            out[key] = vision_source[key]
    # Force projector out dim in nested vision_config
    vc = dict(out.get("vision_config") or {})
    vc["mm_hidden_size"] = int(vc.get("text_hidden_size") or out.get("hidden_size") or 4096)
    out["vision_config"] = vc
    out["architectures"] = [architecture]
    return out


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_artifact_shas(
    tower_path: str | Path,
    projector_path: str | Path,
) -> dict[str, Any]:
    tower_sha = sha256_file(tower_path)
    proj_sha = sha256_file(projector_path)
    return {
        "tower_path": str(tower_path),
        "projector_path": str(projector_path),
        "tower_sha256": tower_sha,
        "projector_sha256": proj_sha,
        "tower_ok": tower_sha == TOWER_SHA256,
        "projector_ok": proj_sha == PROJECTOR_SHA256,
        "expected_tower": TOWER_SHA256,
        "expected_projector": PROJECTOR_SHA256,
    }
