#!/usr/bin/env python3
"""Fail-closed verifier for the DeepSeek-V4-Flash DSpark release candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "model_id": "deepseek-ai/DeepSeek-V4-Flash-DSpark",
    "model_revision": "913f0657a874f76844e2e91cbe706dbcaceeb6d7",
    "base_digest": "sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8",
    "vllm": "0.25.2.dev0+g752a3a504.d20260714",
    "torch": "2.11.0+cu130",
    "cuda": "13.0",
    "flashinfer": "0.6.15",
}
FORBIDDEN_PUBLIC_TEXT = (
    "/home/",
    "192.168.",
    "r0b0tdgx",
    "r0b0t-dgx",
    "gn100-2eea",
    "GITHUB_TOKEN",
    "HF_" + "TOKEN=hf_",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksums(manifest: Path, errors: list[str]) -> None:
    if not manifest.is_file():
        errors.append(f"missing checksum manifest: {manifest}")
        return
    seen: set[Path] = set()
    for number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", raw)
        if not match:
            errors.append(f"checksum line {number} is malformed")
            continue
        expected, relative = match.groups()
        candidate = (ROOT / relative).resolve()
        try:
            candidate.relative_to(ROOT.resolve())
        except ValueError:
            errors.append(f"checksum path escapes repository: {relative}")
            continue
        if candidate in seen:
            errors.append(f"duplicate checksum path: {relative}")
            continue
        seen.add(candidate)
        if not candidate.is_file():
            errors.append(f"checksummed file is missing: {relative}")
        elif sha256(candidate) != expected:
            errors.append(f"checksum mismatch: {relative}")
    if not seen:
        errors.append("checksum manifest is empty")


def verify_image(image: str, expected_revision: str, errors: list[str]) -> None:
    result = subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        errors.append(f"cannot inspect image {image!r}: {result.stderr.strip()}")
        return
    try:
        inspected = json.loads(result.stdout)[0]
    except (json.JSONDecodeError, IndexError, TypeError) as exc:
        errors.append(f"invalid docker inspect output: {exc}")
        return
    image_id = str(inspected.get("Id", ""))
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        errors.append("image ID is not immutable sha256")
    labels = inspected.get("Config", {}).get("Labels", {}) or {}
    required = {
        "org.opencontainers.image.revision": expected_revision,
        "io.r0b0tlab.model.id": EXPECTED["model_id"],
        "io.r0b0tlab.model.revision": EXPECTED["model_revision"],
        "io.r0b0tlab.kv-cache.dtype": "nvfp4_ds_mla",
        "io.r0b0tlab.speculative.method": "dspark",
    }
    for key, expected in required.items():
        if labels.get(key) != expected:
            errors.append(f"image label {key} must equal {expected!r}")
    entrypoint = inspected.get("Config", {}).get("Entrypoint")
    if entrypoint != ["/usr/local/bin/dspark-entrypoint.sh"]:
        errors.append("image entrypoint bypasses the audited entrypoint")


def verify(
    summary_path: Path, image: str | None, allow_dirty: bool, source_only: bool
) -> list[str]:
    errors: list[str] = []
    manifest = load_json(ROOT / "docker/runtime-manifest.production.json")
    summary = load_json(summary_path)

    entrypoint_text = (ROOT / "scripts/entrypoint.sh").read_text(encoding="utf-8")
    launcher_text = (ROOT / "run-dspark-dual-gb10.sh").read_text(encoding="utf-8")
    for marker in (
        "requires an explicit --kv-cache-dtype nvfp4_ds_mla argument",
        "requires exactly one explicit --kv-cache-dtype argument",
        "accepts only an explicit vLLM serve command",
    ):
        if marker not in entrypoint_text:
            errors.append(f"entrypoint lost fail-closed marker: {marker}")
    for marker in (
        "verify_model_checkpoint.py",
        "Head/worker image IDs differ",
        "native-v025 cannot use the legacy Stage-C image",
        "legacy-stage-c requires DSPARK_VLLM_IMAGE",
        "DSPARK_DUAL_NODE_PREFLIGHT_PASS",
    ):
        if marker not in launcher_text:
            errors.append(f"launcher lost preflight/lane marker: {marker}")

    base_image = str(manifest.get("base_image", ""))
    if manifest.get("model_id") != EXPECTED["model_id"] or manifest.get("model_revision") != EXPECTED["model_revision"]:
        errors.append("runtime manifest model identity is unexpected")
    if not base_image.endswith("@" + EXPECTED["base_digest"]):
        errors.append("runtime manifest base image is not pinned to the expected digest")
    version_fields = {
        "vllm_version": "vllm",
        "torch_version": "torch",
        "cuda_version": "cuda",
        "flashinfer_version": "flashinfer",
    }
    for field, expected_key in version_fields.items():
        if manifest.get(field) != EXPECTED[expected_key]:
            errors.append(f"{field} must equal {EXPECTED[expected_key]!r}")
    if manifest.get("kv_cache_dtype") != "nvfp4_ds_mla":
        errors.append("runtime manifest lost nvfp4_ds_mla")
    if manifest.get("production_profile", {}) != {
        "max_model_len": 200000,
        "max_num_seqs": 16,
        "max_num_batched_tokens": 16384,
        "gpu_memory_utilization": 0.84,
        "mtp_tokens": 5,
    }:
        errors.append("production profile differs from the qualified profile")
    capacity_384k = manifest.get("capacity_profile_384k", {})
    if capacity_384k.get("runtime_lane") != "legacy-stage-c" or capacity_384k.get("max_model_len") != 384000:
        errors.append("384K profile must remain a separate legacy-stage-c lane")
    long_profile = manifest.get("long_context_profile", {})
    if long_profile.get("runtime_lane") != "legacy-stage-c":
        errors.append("1M profile must remain a separate legacy-stage-c lane")

    startup_text = (ROOT / "results/production-candidate/startup-evidence.txt").read_text(
        encoding="utf-8"
    )
    grounded_kv = {
        "kv_tokens": "462,426",
        "kv_mem": "16.25",
        "max_conc": "200,000 @ 2.31x",
        "kv_dtype": "nvfp4_ds_mla",
    }
    startup_fragments = (
        "GPU KV cache size: 462,426 tokens",
        "Available KV cache memory: 16.25 GiB",
        "Maximum concurrency for 200,000 tokens per request: 2.31x",
    )
    if not all(fragment in startup_text for fragment in startup_fragments):
        errors.append("committed startup evidence does not contain the promoted KV facts")
    if summary.get("kv_facts_final") is None:
        errors.append("summary is missing final KV facts")
    else:
        for key, expected in grounded_kv.items():
            if summary["kv_facts_final"].get(key) != expected:
                errors.append(f"kv_facts_final.{key} is not grounded in startup evidence")

    candidate = summary.get("candidate", {})
    required_candidate = {
        "mtp_tokens_selected": 5,
        "max_model_len": 200000,
        "max_num_seqs": 16,
        "max_num_batched_tokens": 16384,
        "gpu_memory_utilization": 0.84,
        "kv_cache_dtype": "nvfp4_ds_mla",
        "moe_backend": "flashinfer_b12x",
    }
    for key, expected in required_candidate.items():
        if candidate.get(key) != expected:
            errors.append(f"candidate.{key} must equal {expected!r}")
    if summary.get("runtime_gate", {}).get("status") != "PASS":
        errors.append("runtime gate did not pass")
    long_gate = summary.get("long_context_uncached", {})
    if long_gate.get("status") != "PASS" or long_gate.get("prompt_tokens", 0) < 100000:
        errors.append("uncached 100K long-context gate did not pass")
    staggered = summary.get("staggered_c16", {})
    if staggered.get("status") != "PASS" or staggered.get("requests_ok") != 16:
        errors.append("staggered c16 gate did not pass 16/16")
    prefill = summary.get("prefill_uncached_approximately_18k_tokens", {})
    if not prefill.get("all_requests_ok") or prefill.get("prompt_tps_median", 0) <= 0:
        errors.append("uncached prefill evidence is missing or failed")

    sweep = summary.get("decode_k_sweep", {})
    confirm = summary.get("decode_k5_confirmation", {})
    for level in ("1", "2", "4", "8"):
        k5 = sweep.get("5", {}).get(level, {}).get("decode_tps_median", 0)
        if not all(k5 > sweep.get(k, {}).get(level, {}).get("decode_tps_median", 0) for k in ("3", "4")):
            errors.append(f"K=5 did not win the matched decode sweep at concurrency {level}")
    if confirm.get("1", {}).get("requests_ok") != 5:
        errors.append("K=5 c1 confirmation is incomplete")
    if confirm.get("16", {}).get("requests_ok") != 80:
        errors.append("K=5 c16 confirmation is incomplete")
    if confirm.get("16", {}).get("decode_tps_median", 0) <= sweep.get("3", {}).get("16", {}).get("decode_tps_median", 0):
        errors.append("K=5 did not beat K=3 in the c16 confirmation median")

    verify_checksums(ROOT / "results/production-candidate/SHA256SUMS", errors)

    safety = subprocess.run(
        [sys.executable, str(ROOT / "scripts/public_safety_scan.py"), str(ROOT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if safety.returncode != 0:
        detail = (safety.stdout + safety.stderr).strip()
        errors.append(f"public safety scan failed: {detail}")

    public_paths = (
        ROOT / "README.md",
        ROOT / "RESULTS.md",
        ROOT / "PRIVACY.md",
        ROOT / "docker/runtime-manifest.production.json",
        summary_path,
    )
    for path in public_paths:
        if not path.is_file():
            errors.append(f"missing public artifact: {path}")
            continue
        text = path.read_text(errors="replace")
        for forbidden in FORBIDDEN_PUBLIC_TEXT:
            if forbidden.lower() in text.lower():
                errors.append(f"forbidden public text {forbidden!r} in {path.relative_to(ROOT)}")

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    if not allow_dirty:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        if dirty:
            errors.append("repository is dirty; exact-candidate verification requires a clean tree")
    if image:
        verify_image(image, revision, errors)
    elif not source_only:
        errors.append("exact release verification requires --image; use --source-only only when Docker is intentionally unavailable")
    return errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary", type=Path, default=ROOT / "results/production-candidate/summary.json"
    )
    parser.add_argument("--image", help="exact local image to verify against git HEAD")
    parser.add_argument("--source-only", action="store_true", help="verify tracked source/evidence without a local Docker image")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.image and args.source_only:
            raise ValueError("--image and --source-only are mutually exclusive")
        errors = verify(args.summary, args.image, args.allow_dirty, args.source_only)
    except Exception as exc:
        print(f"RELEASE_VERIFY_FAIL: {type(exc).__name__}: {exc}")
        return 1
    if errors:
        for error in errors:
            print(f"RELEASE_VERIFY_FAIL: {error}")
        return 1
    print("DSPARK_RELEASE_VERIFY_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
