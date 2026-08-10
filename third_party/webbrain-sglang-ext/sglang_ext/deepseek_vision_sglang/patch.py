from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from . import SGLANG_SOURCE_COMMIT


OLD = "hidden_states = self.embed_tokens(input_ids)"
NEW = "hidden_states = input_embeds if input_embeds is not None else self.embed_tokens(input_ids)"
FORWARD_ANCHOR = "class DeepseekV4Model(nn.Module):"


def resolve_sglang_deepseek_v4_source() -> Path:
    spec = importlib.util.find_spec("sglang.srt.models.deepseek_v4")
    if spec is None or spec.origin is None:
        raise RuntimeError("could not locate sglang.srt.models.deepseek_v4")
    return Path(spec.origin).resolve()


def patch_deepseek_v4_source(path: str | Path, *, check_only: bool = False) -> bool:
    """Apply the one-line routing-aware embedding patch.

    Returns ``True`` when the source needed a patch and ``False`` when it was already
    patched.  ``check_only`` validates the exact source anchor without writing.
    """
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if FORWARD_ANCHOR not in text:
        raise RuntimeError("not a recognized SGLang DeepSeek V4 model source")
    if text.count(NEW) == 1:
        return False
    count = text.count(OLD)
    if count != 1:
        raise RuntimeError(
            "expected exactly one SGLang DeepSeek V4 embedding site; "
            f"found {count}. The extension is pinned to {SGLANG_SOURCE_COMMIT}."
        )
    if not check_only:
        source.write_text(text.replace(OLD, NEW), encoding="utf-8")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Patch the pinned SGLang DeepSeek V4 loader")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("source", nargs="?", help="deepseek_v4.py; auto-detected when omitted")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.source).resolve() if args.source else resolve_sglang_deepseek_v4_source()
    needed = patch_deepseek_v4_source(source, check_only=args.check)
    state = "patchable" if needed and args.check else "patched" if needed else "already-patched"
    print(f"{source}: {state}; pinned SGLang commit {SGLANG_SOURCE_COMMIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
