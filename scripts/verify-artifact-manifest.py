#!/usr/bin/env python3
"""Create and compare deterministic manifests for pinned model snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create(snapshot: Path, revision: str, image: str) -> dict[str, object]:
    if not snapshot.is_dir():
        raise ValueError(f"snapshot is not a directory: {snapshot}")
    files: list[dict[str, object]] = []
    for path in sorted(snapshot.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"snapshot symlink is not allowed: {path.relative_to(snapshot)}")
        if not path.is_file():
            continue
        files.append({
            "path": path.relative_to(snapshot).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256(path),
            "executable": bool(path.stat().st_mode & 0o111),
        })
    return {
        "schema_version": 1,
        "model_revision": revision,
        "image": image,
        "files": files,
    }


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create_parser = sub.add_parser("create")
    create_parser.add_argument("--snapshot", type=Path, required=True)
    create_parser.add_argument("--revision", required=True)
    create_parser.add_argument("--image", required=True)
    create_parser.add_argument("--output", type=Path, required=True)
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--left", type=Path, required=True)
    compare_parser.add_argument("--right", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "create":
            write_manifest(args.output, create(args.snapshot, args.revision, args.image))
            return 0
        left = json.loads(args.left.read_text())
        right = json.loads(args.right.read_text())
        if left != right:
            print("artifact manifest mismatch", file=sys.stderr)
            return 1
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
