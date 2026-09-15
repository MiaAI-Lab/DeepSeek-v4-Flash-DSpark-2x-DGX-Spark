#!/usr/bin/env python3
"""Materialize the Keys Vision-Exp ablit Hub cache without re-fetching stock tensors.

Official Vision-Exp is already on disk. Ablit only edits layers.10–35 attn.wo_b
(26 tensors, packed into shards 00012–00037). Shared files are hardlinked from
the official snapshot; only unique blobs are copied from a completed ablit
cache (head) or listed for Hub fetch.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

OFFICIAL_REPO = "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
ABLITERATED_REPO = "drowzeys/keys-DeepSeekV4Flash-Vision-EXP-ablit"
DEFAULT_OFFICIAL_REV = "86f746b36186f0e567729a5c06a8c918caba82a9"
DEFAULT_ABLITERATED_REV = "48095b3452a17f3e3ae8f77892399389c45de9e1"


def hub_dir(hf_home: Path, repo_id: str) -> Path:
    return hf_home / "hub" / f"models--{repo_id.replace('/', '--')}"


def snapshot_files(snap: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in snap.rglob("*"):
        if p.is_dir():
            continue
        out[str(p.relative_to(snap))] = p.resolve()
    return out


def classify(official_snap: Path, ablit_snap: Path) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
    official = snapshot_files(official_snap)
    ablit = snapshot_files(ablit_snap)
    shared: list[tuple[str, Path]] = []
    unique: list[tuple[str, Path]] = []
    for rel, ablit_blob in ablit.items():
        official_blob = official.get(rel)
        if official_blob is not None and official_blob.name == ablit_blob.name:
            shared.append((rel, official_blob))
        else:
            unique.append((rel, ablit_blob))
    return shared, unique


def ensure_link(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        if dest.exists() and src.exists() and dest.stat().st_ino == src.stat().st_ino:
            return
        dest.unlink()
    os.link(src, dest)


def write_snapshot_link(snap_dir: Path, rel: str, blob: Path) -> None:
    link = snap_dir / rel
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    rel_blob = os.path.relpath(blob, start=link.parent)
    link.symlink_to(rel_blob)


def write_manifest(official_snap: Path, ablit_snap: Path, dest: Path) -> None:
    shared, unique = classify(official_snap, ablit_snap)
    payload = {
        "official_revision": official_snap.name,
        "abliterated_revision": ablit_snap.name,
        "shared": [{"rel": rel, "blob": blob.name} for rel, blob in shared],
        "unique": [{"rel": rel, "blob": blob.name, "size": blob.stat().st_size} for rel, blob in unique],
    }
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote manifest {dest} unique={len(unique)} shared={len(shared)}", file=sys.stderr)


def materialize_from_manifest(hf_home: Path, manifest_path: Path) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    official_rev = payload["official_revision"]
    ablit_rev = payload["abliterated_revision"]
    official_root = hub_dir(hf_home, OFFICIAL_REPO)
    ablit_root = hub_dir(hf_home, ABLITERATED_REPO)
    official_snap = official_root / "snapshots" / official_rev
    official_blobs = official_root / "blobs"
    dest_blobs = ablit_root / "blobs"
    dest_snap = ablit_root / "snapshots" / ablit_rev
    if not official_snap.is_dir():
        raise SystemExit(f"missing official snapshot: {official_snap}")
    dest_blobs.mkdir(parents=True, exist_ok=True)
    dest_snap.mkdir(parents=True, exist_ok=True)

    official_by_name = {p.name: p for p in official_blobs.iterdir() if p.is_file()}
    for item in payload["shared"]:
        src = official_by_name.get(item["blob"])
        if src is None:
            raise SystemExit(f"official blob missing: {item['blob']} ({item['rel']})")
        dest_blob = dest_blobs / item["blob"]
        ensure_link(src, dest_blob)
        write_snapshot_link(dest_snap, item["rel"], dest_blob)

    missing = []
    unique_bytes = 0
    for item in payload["unique"]:
        dest_blob = dest_blobs / item["blob"]
        if not dest_blob.exists():
            missing.append(item["rel"])
            continue
        unique_bytes += dest_blob.stat().st_size
        write_snapshot_link(dest_snap, item["rel"], dest_blob)
    if missing:
        raise SystemExit(f"unique ablit blobs missing ({len(missing)}): {missing[:8]}")

    refs = ablit_root / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_bytes(ablit_rev.encode("ascii"))
    print(
        f"overlay ready: {dest_snap} unique={len(payload['unique'])} "
        f"({unique_bytes / 1024**3:.2f} GiB) shared={len(payload['shared'])}",
        file=sys.stderr,
    )


def materialize(
    hf_home: Path,
    official_rev: str,
    ablit_rev: str,
    unique_src_home: Path | None,
) -> None:
    official_root = hub_dir(hf_home, OFFICIAL_REPO)
    ablit_root = hub_dir(hf_home, ABLITERATED_REPO)
    official_snap = official_root / "snapshots" / official_rev
    src_ablit_root = hub_dir(unique_src_home, ABLITERATED_REPO) if unique_src_home else ablit_root
    src_ablit_snap = src_ablit_root / "snapshots" / ablit_rev
    if not official_snap.is_dir():
        raise SystemExit(f"missing official snapshot: {official_snap}")
    if not src_ablit_snap.is_dir():
        raise SystemExit(f"missing ablit snapshot (unique shards): {src_ablit_snap}")

    shared, unique = classify(official_snap, src_ablit_snap)
    unique_bytes = sum(p.stat().st_size for _, p in unique)
    shared_bytes = sum(p.stat().st_size for _, p in shared)
    print(
        f"overlay: shared={len(shared)} ({shared_bytes / 1024**3:.2f} GiB hardlink) "
        f"unique={len(unique)} ({unique_bytes / 1024**3:.2f} GiB copy)",
        file=sys.stderr,
    )

    dest_snap = ablit_root / "snapshots" / ablit_rev
    dest_blobs = ablit_root / "blobs"
    dest_blobs.mkdir(parents=True, exist_ok=True)
    dest_snap.mkdir(parents=True, exist_ok=True)

    for rel, official_blob in shared:
        dest_blob = dest_blobs / official_blob.name
        ensure_link(official_blob, dest_blob)
        write_snapshot_link(dest_snap, rel, dest_blob)

    for rel, src_blob in unique:
        dest_blob = dest_blobs / src_blob.name
        dest_blob.parent.mkdir(parents=True, exist_ok=True)
        if not dest_blob.exists():
            print(f"copy unique {rel} ({src_blob.stat().st_size / 1024**3:.2f} GiB)", file=sys.stderr)
            shutil.copy2(src_blob, dest_blob)
        write_snapshot_link(dest_snap, rel, dest_blob)

    refs = ablit_root / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_bytes(ablit_rev.encode("ascii"))
    print(f"overlay ready: {dest_snap}", file=sys.stderr)


def rsync_unique(src_home: Path, dest_host: str, dest_home: Path, official_rev: str, ablit_rev: str) -> None:
    src_official = hub_dir(src_home, OFFICIAL_REPO) / "snapshots" / official_rev
    src_ablit = hub_dir(src_home, ABLITERATED_REPO) / "snapshots" / ablit_rev
    shared, unique = classify(src_official, src_ablit)
    dest_blobs = hub_dir(dest_home, ABLITERATED_REPO) / "blobs"
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", dest_host, f"mkdir -p {dest_blobs}"],
        check=True,
    )
    files = [p.name for _, p in unique]
    list_file = src_home / ".ablit-unique-blobs.txt"
    list_file.write_text("\n".join(files) + "\n", encoding="utf-8")
    src_blob_dir = hub_dir(src_home, ABLITERATED_REPO) / "blobs"
    cmd = [
        "rsync",
        "-a",
        "--info=progress2",
        f"--files-from={list_file}",
        f"{src_blob_dir}/",
        f"{dest_host}:{dest_blobs}/",
    ]
    print("rsync unique blobs →", dest_host, file=sys.stderr)
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-home", type=Path, default=Path(os.environ.get("HF_CACHE", os.path.expanduser("~/.cache/huggingface"))))
    p.add_argument("--official-revision", default=DEFAULT_OFFICIAL_REV)
    p.add_argument("--abliterated-revision", default=DEFAULT_ABLITERATED_REV)
    p.add_argument("--unique-src-home", type=Path, default=None, help="HF home that already has the unique ablit blobs")
    p.add_argument("--write-manifest", type=Path, default=None)
    p.add_argument("--from-manifest", type=Path, default=None)
    p.add_argument("--rsync-unique-to", default=None, help="user@host to push unique blobs to before overlay")
    p.add_argument("--remote-hf-home", type=Path, default=None)
    args = p.parse_args()
    if args.write_manifest:
        official_snap = hub_dir(args.hf_home, OFFICIAL_REPO) / "snapshots" / args.official_revision
        ablit_snap = hub_dir(args.unique_src_home or args.hf_home, ABLITERATED_REPO) / "snapshots" / args.abliterated_revision
        write_manifest(official_snap, ablit_snap, args.write_manifest)
        return
    if args.rsync_unique_to:
        if not args.remote_hf_home:
            raise SystemExit("--remote-hf-home is required with --rsync-unique-to")
        rsync_unique(
            args.unique_src_home or args.hf_home,
            args.rsync_unique_to,
            args.remote_hf_home,
            args.official_revision,
            args.abliterated_revision,
        )
        return
    if args.from_manifest:
        materialize_from_manifest(args.hf_home, args.from_manifest)
        return
    materialize(args.hf_home, args.official_revision, args.abliterated_revision, args.unique_src_home)


if __name__ == "__main__":
    main()
