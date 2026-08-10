#!/usr/bin/env python3
"""Stage overlay model dir: symlink official 0731 + MoonViT weights + vision config."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugins" / "dsv4_moonvit_vllm"))

from dsv4_moonvit_vllm.config import (  # noqa: E402
    load_json,
    merge_vision_into_config_json,
    verify_artifact_shas,
)


def resolve_0731_snapshot(explicit: str | None, hf_home: Path) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_dir():
            raise SystemExit(f"0731 path not a directory: {p}")
        return p
    hub = hf_home / "hub" / "models--deepseek-ai--DeepSeek-V4-Flash-0731" / "snapshots"
    preferred = hub / "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
    if preferred.is_dir():
        return preferred
    snaps = sorted(hub.glob("*")) if hub.is_dir() else []
    if not snaps:
        raise SystemExit(f"no 0731 snapshot under {hub}")
    return snaps[-1]


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    """Symlink with a *relative* target when possible.

    Absolute host paths like ``/home/mia/.cache/...`` break inside the
    container where the same tree is mounted at ``/cache/huggingface``.
    Relative links stay valid on both host and container layouts.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    src = src.resolve()
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return
    try:
        target = os.path.relpath(src, start=dst.parent.resolve())
    except ValueError:
        # Different drives / roots — fall back to absolute.
        target = str(src)
    os.symlink(target, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output",
        default=os.environ.get(
            "DSPARK_MOONVIT_MODEL_DIR",
            str(Path.home() / ".cache/huggingface/dsv4-0731-moonvit"),
        ),
    )
    ap.add_argument("--text-snapshot", default=os.environ.get("DSV4_TEXT_SNAPSHOT"))
    ap.add_argument(
        "--vision-src",
        default=os.environ.get(
            "DSV4_MOONVIT_SRC",
            str(Path.home() / ".cache/huggingface/webbrain-0731-moonvit-src"),
        ),
    )
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
    ap.add_argument("--copy", action="store_true", help="Copy instead of symlink")
    ap.add_argument(
        "--architecture",
        default="DeepseekV4MoonVitForCausalLM",
    )
    args = ap.parse_args()

    out = Path(args.output)
    text = resolve_0731_snapshot(args.text_snapshot, Path(args.hf_home))
    vision = Path(args.vision_src)
    tower = vision / "vision_tower.safetensors"
    proj = vision / "mm_projector.safetensors"
    if not tower.is_file() or not proj.is_file():
        raise SystemExit(f"missing vision weights under {vision}")

    rep = verify_artifact_shas(tower, proj)
    if not (rep["tower_ok"] and rep["projector_ok"]):
        print(json.dumps(rep, indent=2))
        raise SystemExit("SHA-256 mismatch; refusing to stage")

    out.mkdir(parents=True, exist_ok=True)
    # Symlink every file/dir from text snapshot except config.json (rewritten).
    for entry in text.iterdir():
        if entry.name == "config.json":
            continue
        link_or_copy(entry.resolve(), out / entry.name, copy=args.copy)

    # Vision weights
    link_or_copy(tower.resolve(), out / "vision_tower.safetensors", copy=args.copy)
    link_or_copy(proj.resolve(), out / "mm_projector.safetensors", copy=args.copy)

    text_cfg = load_json(text / "config.json")
    vision_cfg = {}
    vcfg_path = vision / "config.json"
    if vcfg_path.is_file():
        raw = load_json(vcfg_path)
        for k in (
            "vision_config",
            "deepseek_vision",
            "image_token_id",
            "media_placeholder_token_id",
        ):
            if k in raw:
                vision_cfg[k] = raw[k]

    merged = merge_vision_into_config_json(
        text_cfg, vision_cfg or None, architecture=args.architecture
    )
    (out / "config.json").write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")

    # Small marker for ops
    (out / "MOONVIT_OVERLAY.json").write_text(
        json.dumps(
            {
                "text_snapshot": str(text),
                "vision_src": str(vision),
                "tower_sha256": rep["tower_sha256"],
                "projector_sha256": rep["projector_sha256"],
                "architecture": args.architecture,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"staged moonvit model dir: {out}")
    print(f"  text: {text}")
    print(f"  architecture: {args.architecture}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
