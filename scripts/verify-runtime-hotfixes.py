#!/usr/bin/env python3
"""Verify reviewed hotfix inputs and their effective runtime state."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


MANIFEST_PATH = Path("recipe/runtime-hotfixes.manifest.json")
ISSUE21_MARKER = "if isinstance(raw, dict):"
ISSUE22_MARKER = 'self.kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")'
ISSUE21_PATCHER = "/opt/dspark-hotfixes/hotfix-encoding-dsv4-issue21.py"
RUNTIME_ISSUE22 = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashmla_sparse.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(root: Path) -> dict:
    manifest_file = root / MANIFEST_PATH
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("schema") != "dspark-runtime-hotfixes/v1":
        raise ValueError("unsupported runtime hotfix manifest schema")
    base = manifest.get("base_image", "")
    if "@sha256:" not in base or len(base.rsplit("@sha256:", 1)[1]) != 64:
        raise ValueError("base image is not pinned by sha256 digest")
    observed: dict[str, str] = {}
    for relative, expected in manifest.get("inputs", {}).items():
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"manifest input must be a regular file: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"manifest checksum mismatch: {relative}")
        observed[relative] = actual
    if len(observed) != 3:
        raise ValueError("runtime manifest must pin exactly three build inputs")
    return {"base_image": base, "inputs": observed}


def docker_markers(image: str) -> dict:
    code = (
        "from pathlib import Path; import json; "
        f"print(json.dumps({{'issue21_patcher_present': {ISSUE21_MARKER!r} in Path({ISSUE21_PATCHER!r}).read_text(), "
        f"'issue22_runtime_present': {ISSUE22_MARKER!r} in Path({RUNTIME_ISSUE22!r}).read_text()}}))"
    )
    result = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "python3", image, "-c", code],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("could not inspect derived runtime image")
    return json.loads(result.stdout)


def verify_image(image: str) -> dict:
    markers = docker_markers(image)
    if markers.get("issue21_patcher_present") is not True:
        raise ValueError("Issue #21 patcher is absent from derived image")
    if markers.get("issue22_runtime_present") is not True:
        raise ValueError("Issue #22 is absent from derived runtime")
    image_id = subprocess.run(
        ["docker", "image", "inspect", image, "-f", "{{.Id}}"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    return {
        "image": image,
        "image_id": image_id,
        "issue21_patcher_present": True,
        "issue22_runtime_present": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--image")
    args = parser.parse_args()
    try:
        report = {"manifest": verify_manifest(args.repo_root.resolve())}
        if args.image:
            report["runtime"] = verify_image(args.image)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"runtime-hotfix-verification=failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
