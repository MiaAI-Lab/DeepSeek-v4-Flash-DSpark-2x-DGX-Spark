"""PatchMerger projector: LN → 2×2 merge → Linear(4608→4608) → GELU → Linear(4608→4096).

Matches WebBrain ``mm_projector.safetensors`` keys:
  pre_norm.{weight,bias}, proj.0.{weight,bias}, proj.2.{weight,bias}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchMerger(nn.Module):
    """Pure-torch PatchMerger used for unit tests and weight verification."""

    def __init__(
        self,
        mm_hidden_size: int = 1152,
        text_hidden_size: int = 4096,
        merge_kernel_size: tuple[int, int] = (2, 2),
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        mh, mw = merge_kernel_size
        self.merge_kernel_size = (int(mh), int(mw))
        self.mm_hidden_size = int(mm_hidden_size)
        self.text_hidden_size = int(text_hidden_size)
        self.merged_dim = self.mm_hidden_size * self.merge_kernel_size[0] * self.merge_kernel_size[1]
        self.pre_norm = nn.LayerNorm(self.mm_hidden_size, eps=eps)
        self.linear_1 = nn.Linear(self.merged_dim, self.merged_dim, bias=True)
        self.linear_2 = nn.Linear(self.merged_dim, self.text_hidden_size, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: either
              - ``(N, merge_h*merge_w, C)`` after spatial tpool merge, or
              - ``(N * merge_h * merge_w, C)`` packed pre-merge patches (less common).
        Returns:
            ``(N, text_hidden_size)`` projected tokens.
        """
        x = self.pre_norm(image_features)
        if x.ndim == 3:
            # (N, K, C) → (N, K*C)
            n, k, c = x.shape
            expected_k = self.merge_kernel_size[0] * self.merge_kernel_size[1]
            if k != expected_k:
                raise ValueError(
                    f"expected merge group size {expected_k}, got {k}"
                )
            if c != self.mm_hidden_size:
                raise ValueError(f"expected channel {self.mm_hidden_size}, got {c}")
            x = x.reshape(n, self.merged_dim)
        elif x.ndim == 2:
            if x.shape[-1] == self.mm_hidden_size:
                if x.shape[0] % (self.merge_kernel_size[0] * self.merge_kernel_size[1]) != 0:
                    raise ValueError(
                        "packed patch count not divisible by merge kernel area"
                    )
                x = x.reshape(-1, self.merged_dim)
            elif x.shape[-1] != self.merged_dim:
                raise ValueError(
                    f"unexpected feature dim {x.shape[-1]} "
                    f"(want {self.mm_hidden_size} or {self.merged_dim})"
                )
        else:
            raise ValueError(f"unexpected image_features rank {x.ndim}")

        x = self.linear_1(x)
        x = F.gelu(x)
        x = self.linear_2(x)
        return x

    def load_webbrain_safetensors(
        self,
        path: str | Path,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = torch.bfloat16,
    ) -> list[str]:
        """Load ``mm_projector.safetensors`` (proj.0/proj.2 naming)."""
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover
            raise ImportError("safetensors is required to load the projector") from exc

        path = Path(path)
        state = load_file(str(path), device="cpu")
        mapping = {
            "pre_norm.weight": "pre_norm.weight",
            "pre_norm.bias": "pre_norm.bias",
            "proj.0.weight": "linear_1.weight",
            "proj.0.bias": "linear_1.bias",
            "proj.2.weight": "linear_2.weight",
            "proj.2.bias": "linear_2.bias",
        }
        own = self.state_dict()
        loaded: list[str] = []
        for src, dst in mapping.items():
            if src not in state:
                raise KeyError(f"missing projector tensor {src} in {path}")
            tensor = state[src]
            if own[dst].shape != tensor.shape:
                raise ValueError(
                    f"shape mismatch for {src}: file {tuple(tensor.shape)} "
                    f"vs module {tuple(own[dst].shape)}"
                )
            own[dst] = tensor
            loaded.append(src)
        self.load_state_dict(own)
        if dtype is not None:
            self.to(dtype=dtype)
        if device is not None:
            self.to(device=device)
        return loaded


def project_tower_merged_features(
    projector: PatchMerger,
    tower_outputs: list[torch.Tensor] | torch.Tensor,
) -> torch.Tensor:
    """Apply projector to one or more per-image merged feature tensors.

    Concatenates along token dim after projection so callers get a single
    ``(T_total, 4096)`` packed tensor (T ≤ 512 per image in v1).
    """
    if isinstance(tower_outputs, torch.Tensor):
        outs = [tower_outputs]
    else:
        outs = list(tower_outputs)
    projected = [projector(o) for o in outs]
    return torch.cat(projected, dim=0)


def expected_projector_param_count() -> int:
    """WebBrain projector has 40,119,040 parameters."""
    return 40_119_040


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def describe_projector_contract() -> dict[str, Any]:
    return {
        "merge_kernel_size": [2, 2],
        "tower_hidden": 1152,
        "merged_dim": 4608,
        "llm_hidden": 4096,
        "max_merged_tokens": 512,
        "param_count": expected_projector_param_count(),
    }
