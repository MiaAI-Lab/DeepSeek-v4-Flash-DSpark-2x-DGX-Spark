#!/usr/bin/env python3
"""Opt-in runtime refusal-direction ablation for DeepSeek V4 Flash.

The pinned Anemll image and the optional Stage-C lane carry different versions
of nvidia/model.py, so this applies a narrow, fail-closed source transformation
instead of replacing that whole file.  ABLATE=0 verifies that the ephemeral
container is stock and only maintains the AOT-cache compatibility stamp.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "vllm/models/deepseek_v4/nvidia/model.py"
)
MARK = "# [dsv4-runtime-ablation-v1]"
CACHE_STAMP_VERSION = "dsv4-runtime-ablation-v1"

GLOBALS = r'''# [dsv4-runtime-ablation-v1]
# Runtime refusal-direction ablation.  An empty file path disables the hook.
_DSV4_ABLATE_FILE = os.environ.get("DSV4_ABLATE_FILE", "").strip() or None
_DSV4_ABLATE_LAMBDA = float(os.environ.get("DSV4_ABLATE_LAMBDA", "3.5"))
_DSV4_ABLATE_LAYERS = os.environ.get("DSV4_ABLATE_LAYERS", "10-42").strip()
_DSV4_ABLATE_MATCH = re.fullmatch(r"(\d+)\s*-\s*(\d+)", _DSV4_ABLATE_LAYERS)
if _DSV4_ABLATE_FILE is not None:
    if not math.isfinite(_DSV4_ABLATE_LAMBDA):
        raise ValueError("DSV4_ABLATE_LAMBDA must be finite")
    if _DSV4_ABLATE_MATCH is None:
        raise ValueError(
            "DSV4_ABLATE_LAYERS must look like '10-42' "
            f"(got {_DSV4_ABLATE_LAYERS!r})"
        )
_DSV4_ABLATE_LAYER_LO = (
    int(_DSV4_ABLATE_MATCH.group(1)) if _DSV4_ABLATE_MATCH else -1
)
_DSV4_ABLATE_LAYER_HI = (
    int(_DSV4_ABLATE_MATCH.group(2)) if _DSV4_ABLATE_MATCH else -2
)
if (
    _DSV4_ABLATE_FILE is not None
    and _DSV4_ABLATE_LAYER_LO > _DSV4_ABLATE_LAYER_HI
):
    raise ValueError("DSV4_ABLATE_LAYERS lower bound exceeds upper bound")
del _DSV4_ABLATE_MATCH
'''

INIT = '''        # FILE set: every layer keeps a real tensor so all decoder
        # layers share one torch.compile/AOT signature.  Unset is stock/inert.
        self._ablate_lambda = 0.0
        if _DSV4_ABLATE_FILE is not None:
            self.register_buffer(
                "_refusal_dir",
                torch.zeros(self.hidden_size, dtype=torch.float32),
                persistent=False,
            )
            self._init_refusal_ablation(prefix)
        else:
            self._refusal_dir = None

'''

METHODS = r'''    def _init_refusal_ablation(self, prefix: str) -> None:
        """Load the configured direction for selected target decoder layers."""
        match = re.search(r"layers\.(\d+)$", prefix or "")
        if match is None:
            return
        layer_idx = int(match.group(1))
        if not (_DSV4_ABLATE_LAYER_LO <= layer_idx <= _DSV4_ABLATE_LAYER_HI):
            return

        raw = torch.load(
            _DSV4_ABLATE_FILE,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
            weights_only=True,
        )
        direction = raw.get("broad") if isinstance(raw, dict) else None
        if direction is None and isinstance(raw, dict):
            direction = raw.get("directions")
            if isinstance(direction, torch.Tensor) and direction.dim() == 2:
                direction = direction[0]
        if not isinstance(direction, torch.Tensor) or direction.numel() != self.hidden_size:
            raise ValueError(
                f"DSV4_ABLATE_FILE: expected a {self.hidden_size}-dim direction "
                f"tensor, got {type(direction).__name__}"
            )

        direction = direction.reshape(-1).to(torch.float32)
        if not bool(torch.isfinite(direction).all()):
            raise ValueError("DSV4_ABLATE_FILE direction contains non-finite values")
        norm = direction.norm()
        if not bool(torch.isfinite(norm)) or norm.item() <= 0.0:
            raise ValueError("DSV4_ABLATE_FILE direction has invalid zero/non-finite norm")
        self._refusal_dir = (direction / norm).contiguous()
        self._ablate_lambda = _DSV4_ABLATE_LAMBDA

    def _ablate_refusal_direction(self, x: torch.Tensor) -> torch.Tensor:
        """Apply y <- y - lambda * (y dot v) * v in fp32."""
        direction = self._refusal_dir
        projection = (x.to(torch.float32) @ direction).unsqueeze(-1)
        return x - (self._ablate_lambda * projection * direction).to(x.dtype)

'''


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected one anchor, found {count}")
    return source.replace(old, new, 1)


def patch_text(source: str) -> tuple[str, int]:
    """Return patched source and the number of post-attention hook sites."""
    if MARK in source:
        required = (
            "def _init_refusal_ablation",
            "def _ablate_refusal_direction",
            "self._ablate_refusal_direction(x)",
        )
        if not all(part in source for part in required):
            raise ValueError("partial/corrupt runtime-ablation patch marker")
        return source, source.count("x = self._ablate_refusal_direction(x)")

    class_start = source.find("class DeepseekV4DecoderLayer(nn.Module):")
    class_end = source.find("\nclass DeepseekV4Model(", class_start)
    if class_start < 0 or class_end < 0:
        raise ValueError("DeepseekV4DecoderLayer/DeepseekV4Model anchors not found")

    source = _replace_once(
        source,
        "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n",
        "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
        "import math\nimport os\n",
        "imports",
    )

    # Re-locate after the import insertion.
    class_start = source.find("class DeepseekV4DecoderLayer(nn.Module):")
    class_end = source.find("\nclass DeepseekV4Model(", class_start)
    source = source[:class_start] + GLOBALS + "\n\n" + source[class_start:]

    class_start = source.find("class DeepseekV4DecoderLayer(nn.Module):")
    class_end = source.find("\nclass DeepseekV4Model(", class_start)
    decoder = source[class_start:class_end]
    init_anchor = "        self.hidden_size = config.hidden_size\n\n"
    decoder = _replace_once(decoder, init_anchor, init_anchor + INIT, "decoder init")

    method_matches = list(
        re.finditer(r"^    def (?:_forward_cuda|forward)\(", decoder, re.MULTILINE)
    )
    if not method_matches:
        raise ValueError("decoder forward anchor not found")
    method_pos = method_matches[0].start()
    decoder = decoder[:method_pos] + METHODS + decoder[method_pos:]

    attention_pattern = re.compile(
        r"^(?P<indent>\s*)x = self\.attn\(positions, x, None\)$", re.MULTILINE
    )
    hook_sites = 0

    def add_hook(match: re.Match[str]) -> str:
        nonlocal hook_sites
        hook_sites += 1
        indent = match.group("indent")
        return (
            match.group(0)
            + "\n"
            + indent
            + "if _DSV4_ABLATE_FILE is not None:\n"
            + indent
            + "    x = self._ablate_refusal_direction(x)"
        )

    decoder = attention_pattern.sub(add_hook, decoder)
    # Known lanes: pinned Anemll has one call; the current Stage-C source has
    # three (two CUDA branches plus its alternate path).  Refuse unknown shape.
    if hook_sites not in (1, 2, 3):
        raise ValueError(f"decoder attention hook count drifted: {hook_sites}")

    updated = source[:class_start] + decoder + source[class_end:]
    compile(updated, str(TARGET), "exec")
    return updated, hook_sites


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_direction(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"ablation direction is missing: {path}")
    import torch

    raw = torch.load(path, map_location="cpu", weights_only=True)
    direction = raw.get("broad") if isinstance(raw, dict) else None
    if direction is None and isinstance(raw, dict):
        direction = raw.get("directions")
        if isinstance(direction, torch.Tensor) and direction.dim() == 2:
            direction = direction[0]
    if not isinstance(direction, torch.Tensor) or direction.numel() != 4096:
        raise ValueError("ablation direction must contain a 4096-d 'broad' tensor")
    direction = direction.reshape(-1).float()
    if not bool(torch.isfinite(direction).all()) or direction.norm().item() <= 0.0:
        raise ValueError("ablation direction must be finite and non-zero")
    return _sha256(path)


def apply_patch(path: Path = TARGET) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"DeepSeek V4 model source is missing: {path}")
    source = path.read_text(encoding="utf-8")
    updated, hook_sites = patch_text(source)
    if updated == source:
        print(f"[dsv4-ablation] already applied ({hook_sites} attention hook site(s))")
        return hook_sites

    mode = path.stat().st_mode
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.ablate.", dir=path.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp, mode)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    print(f"[dsv4-ablation] applied to {path} ({hook_sites} attention hook site(s))")
    return hook_sites


def sync_compile_cache_stamp(enabled: bool, direction_hash: str = "") -> None:
    root = Path(os.environ.get("VLLM_CACHE_ROOT", "~/.cache/vllm")).expanduser()
    stamp_path = root / ".dsv4_ablate_stamp"
    compile_cache = root / "torch_compile_cache"
    patch_hash = _sha256(Path(__file__))
    stamp = "|".join(
        (
            CACHE_STAMP_VERSION,
            f"enabled={int(enabled)}",
            f"lambda={os.environ.get('DSV4_ABLATE_LAMBDA', '3.5') if enabled else '-'}",
            f"layers={os.environ.get('DSV4_ABLATE_LAYERS', '10-42') if enabled else '-'}",
            f"direction={direction_hash if enabled else '-'}",
            f"patch={patch_hash}",
        )
    )
    old = stamp_path.read_text(encoding="utf-8").strip() if stamp_path.is_file() else ""
    if old == stamp:
        print("[dsv4-ablation] AOT cache stamp unchanged")
        return

    root.mkdir(parents=True, exist_ok=True)
    # The first installation in stock/off mode inherits a known-stock cache.
    # Every actual mode/config transition must discard the incompatible graph.
    if old or enabled:
        print(
            "[dsv4-ablation] AOT cache stamp changed; removing "
            f"{compile_cache}"
        )
        if compile_cache.exists():
            shutil.rmtree(compile_cache)

    temp = stamp_path.with_name(f"{stamp_path.name}.tmp.{os.getpid()}")
    temp.write_text(stamp + "\n", encoding="utf-8")
    os.replace(temp, stamp_path)


def run() -> int:
    enabled_raw = os.environ.get("ABLATE", "0")
    if enabled_raw not in ("0", "1"):
        raise ValueError(f"ABLATE must be 0 or 1 (got {enabled_raw!r})")
    enabled = enabled_raw == "1"

    if not TARGET.is_file():
        raise FileNotFoundError(f"DeepSeek V4 model source is missing: {TARGET}")

    direction_hash = ""
    if enabled:
        direction_file = os.environ.get("DSV4_ABLATE_FILE", "").strip()
        if not direction_file:
            raise ValueError("ABLATE=1 requires DSV4_ABLATE_FILE")
        direction_hash = validate_direction(Path(direction_file))
        apply_patch()
    else:
        source = TARGET.read_text(encoding="utf-8")
        if MARK in source:
            raise RuntimeError(
                "ABLATE=0 found a patched model.py; recreate the container from the "
                "stock image rather than restarting it with a changed environment"
            )
        print("[dsv4-ablation] disabled; stock model.py unchanged")

    sync_compile_cache_stamp(enabled, direction_hash)
    if enabled:
        print(
            "[dsv4-ablation] enabled: "
            f"lambda={os.environ.get('DSV4_ABLATE_LAMBDA', '3.5')} "
            f"layers={os.environ.get('DSV4_ABLATE_LAYERS', '10-42')} "
            f"direction_sha256={direction_hash}"
        )
    return 0


def main() -> int:
    try:
        if len(sys.argv) == 2 and sys.argv[1] == "--status":
            source = TARGET.read_text(encoding="utf-8") if TARGET.is_file() else ""
            print("APPLIED" if MARK in source else "NOT APPLIED")
            return 0
        if len(sys.argv) != 1:
            print(f"usage: {Path(sys.argv[0]).name} [--status]", file=sys.stderr)
            return 2
        return run()
    except Exception as exc:
        print(f"[dsv4-ablation] FATAL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
