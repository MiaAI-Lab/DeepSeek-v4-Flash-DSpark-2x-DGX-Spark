#!/usr/bin/env python3
"""Fail-closed Issue #22 patch for nvfp4_ds_mla kernel dispatch."""
from __future__ import annotations

import sys
from pathlib import Path


DEFAULT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashmla_sparse.py"
)
OLD = '        use_fp8_cache = self.kv_cache_dtype == "fp8_ds_mla"'
NEW = (
    '        use_fp8_cache = self.kv_cache_dtype in '
    '("fp8_ds_mla", "nvfp4_ds_mla")'
)


def patch_text(source: str) -> tuple[str, str]:
    if NEW in source:
        return source, "skipped"
    if OLD not in source:
        return source, "missing"
    return source.replace(OLD, NEW, 1), "applied"


def patch_file(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    updated, status = patch_text(source)
    if status == "applied":
        path.write_text(updated, encoding="utf-8")
    return status


def main(argv: list[str]) -> int:
    target = Path(argv[1]) if len(argv) > 1 else DEFAULT_TARGET
    if not target.is_file():
        print(f"[FAIL] vLLM MLA backend not found: {target}", file=sys.stderr)
        return 1
    status = patch_file(target)
    if status in {"applied", "skipped"}:
        verb = "applied" if status == "applied" else "already present"
        print(f"[OK] Issue #22 patch {verb}: {target}")
        return 0
    print(
        f"[FAIL] nvfp4_ds_mla dispatch pattern not found; refusing unknown runtime: {target}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
