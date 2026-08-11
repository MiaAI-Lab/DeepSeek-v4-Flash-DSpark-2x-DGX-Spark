#!/usr/bin/env python3
"""Create and verify a sanitized, identity-bound Qwen retirement manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import shlex
import stat
import subprocess
import sys


EXPECTED_CONTAINER = "urbanplan-qwen"
EXPECTED_PROJECT = "urbanplan-spark-api"
EXPECTED_SERVICE = "qwen"
EXPECTED_MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
EXPECTED_REVISION = "491c2f1ea524c639598bf8fa787a93fed5a6fbce"
SERVICE_ROOT = Path("/home/plexiz/services/urbanplan-spark-api")
MODEL_ROOT = Path("/home/plexiz/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4")
FORBIDDEN_ROOTS = (
    Path("/home/plexiz/services/litellm"),
    Path("/home/plexiz/services/plexiz-litellm"),
)


def run_json(*command: str):
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def file_type(mode: int) -> str:
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISLNK(mode):
        return "symlink"
    return "other"


def target_metadata(kind: str, path: Path) -> dict:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return {"kind": kind, "path": str(path), "realpath": str(path), "exists": False}
    return {
        "kind": kind,
        "path": str(path),
        "realpath": str(path.resolve(strict=True)),
        "exists": True,
        "file_type": file_type(info.st_mode),
        "st_dev": info.st_dev,
        "st_ino": info.st_ino,
        "st_uid": info.st_uid,
    }


def inspect_container(name: str) -> dict:
    items = run_json("docker", "inspect", name)
    if len(items) != 1:
        raise RuntimeError(f"expected exactly one container named {name}")
    return items[0]


def command_value(arguments: list[str], flag: str) -> str | None:
    try:
        return arguments[arguments.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def running_vllm_arguments(name: str) -> list[str]:
    result = subprocess.run(
        ["docker", "top", name, "-eo", "pid,args"], check=True, text=True, capture_output=True
    )
    for line in result.stdout.splitlines()[1:]:
        if "vllm serve" in line:
            process_args = line.strip().split(maxsplit=1)[1]
            tokens = shlex.split(process_args)
            try:
                start = tokens.index("vllm")
            except ValueError:
                for index, token in enumerate(tokens):
                    if token.endswith("/vllm"):
                        start = index
                        break
                else:
                    continue
            return tokens[start:]
    raise RuntimeError("could not find the live vllm serve process")


def supervisor_units() -> list[dict]:
    result = subprocess.run(
        ["systemctl", "list-units", "--all", "--plain", "--no-legend"],
        check=False, text=True, capture_output=True,
    )
    units = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if fields and any(word in line.lower() for word in ("qwen", "urbanplan")):
            units.append({"unit": fields[0], "load": fields[1], "active": fields[2]})
    return units


def litellm_success_count(model_names: tuple[str, ...]) -> dict:
    process = subprocess.Popen(
        ["docker", "logs", "--since", "720h", "--tail", "200000", "plexiz-litellm"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    count = 0
    assert process.stdout is not None
    for line in process.stdout:
        lower = line.lower()
        if any(name.lower() in lower for name in model_names) and any(
            marker in lower for marker in ('"status_code": 200', "status_code=200", "success")
        ):
            count += 1
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError("could not read the existing LiteLLM logs for Qwen usage evidence")
    return {
        "source": "plexiz-litellm container logs",
        "window_hours": 720,
        "successful_line_count": count,
        "semantics": "lines containing a Qwen model name and an explicit 200/success marker",
        "source_available": True,
    }


def create_manifest(name: str) -> dict:
    item = inspect_container(name)
    config = item.get("Config", {})
    labels = config.get("Labels") or {}
    host = item.get("HostConfig", {})
    arguments = running_vllm_arguments(name)
    project = labels.get("com.docker.compose.project")
    service = labels.get("com.docker.compose.service")
    working_dir = labels.get("com.docker.compose.project.working_dir")
    config_file = labels.get("com.docker.compose.project.config_files")
    model = command_value(arguments, "serve")
    revision = command_value(arguments, "--revision")
    if name != EXPECTED_CONTAINER or project != EXPECTED_PROJECT or service != EXPECTED_SERVICE:
        raise RuntimeError("container Compose identity does not match the recorded Qwen service")
    if model != EXPECTED_MODEL or revision != EXPECTED_REVISION:
        raise RuntimeError("Qwen model or revision does not match the expected retirement target")
    if Path(working_dir or "") != SERVICE_ROOT:
        raise RuntimeError("unexpected Qwen service root")

    mounts = []
    secret_paths = []
    for mount in item.get("Mounts", []):
        selected = {
            "type": mount.get("Type"),
            "source": mount.get("Source"),
            "destination": mount.get("Destination"),
            "read_write": mount.get("RW"),
        }
        mounts.append(selected)
        if str(mount.get("Destination", "")).startswith("/run/secrets/"):
            secret_paths.append(str(mount.get("Source")))

    bindings = host.get("PortBindings") or {}
    manifest = {
        "schema_version": 1,
        "run_id": f"qwen-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(4)}",
        "created_at": utc_now(),
        "container": {
            "name": name,
            "container_id": item.get("Id"),
            "image_id": item.get("Image"),
            "image_reference": config.get("Image"),
            "project": project,
            "service": service,
            "config_hash": labels.get("com.docker.compose.config-hash"),
            "config_file": config_file,
            "working_dir": working_dir,
            "restart_policy": (host.get("RestartPolicy") or {}).get("Name"),
            "network_mode": host.get("NetworkMode"),
            "running": bool((item.get("State") or {}).get("Running")),
            "health": ((item.get("State") or {}).get("Health") or {}).get("Status"),
            "port_bindings": bindings,
        },
        "model": {"id": EXPECTED_MODEL, "revision": revision, "served_names": ["urbanplan-qwen", "plexiz-qwen3.6-35b"]},
        "mounts": mounts,
        "secret_paths": sorted(secret_paths),
        "supervisors": supervisor_units(),
        "litellm_success_30d": litellm_success_count(("urbanplan-qwen", "plexiz-qwen3.6-35b")),
        "filesystem_targets": [
            target_metadata("service_root", SERVICE_ROOT),
            target_metadata("model_root", MODEL_ROOT),
        ],
    }
    return manifest


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text())
    required = {"schema_version", "run_id", "created_at", "filesystem_targets"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"manifest missing fields: {sorted(missing)}")
    return data


def verify_targets(manifest: dict) -> None:
    for expected in manifest["filesystem_targets"]:
        if not expected.get("exists"):
            continue
        path = Path(expected["path"])
        current = target_metadata(str(expected["kind"]), path)
        if not current.get("exists"):
            raise RuntimeError(f"target disappeared: {path}")
        if current.get("file_type") == "symlink":
            raise RuntimeError(f"target became a symlink: {path}")
        for field in ("realpath", "st_dev", "st_ino", "st_uid", "file_type"):
            if current.get(field) != expected.get(field):
                raise RuntimeError(f"target metadata changed for {path}: {field}")


def verify_no_supervisors(manifest: dict) -> None:
    recorded = manifest.get("supervisors") or []
    if recorded:
        names = ", ".join(str(item.get("unit", "unknown")) for item in recorded)
        raise RuntimeError(f"recorded Qwen supervisor entries block retirement: {names}")
    current = supervisor_units()
    if current:
        names = ", ".join(str(item.get("unit", "unknown")) for item in current)
        raise RuntimeError(f"current Qwen supervisor entries block retirement: {names}")


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_current_identity(manifest: dict) -> dict:
    expected = manifest["container"]
    current = inspect_container(expected["name"])
    labels = current.get("Config", {}).get("Labels") or {}
    checks = {
        "container_id": current.get("Id"),
        "image_id": current.get("Image"),
        "config_hash": labels.get("com.docker.compose.config-hash"),
        "project": labels.get("com.docker.compose.project"),
        "service": labels.get("com.docker.compose.service"),
        "restart_policy": (current.get("HostConfig", {}).get("RestartPolicy") or {}).get("Name"),
    }
    for field, value in checks.items():
        if expected.get(field) != value:
            raise RuntimeError(f"live Qwen identity changed: {field}")
    verify_targets(manifest)
    return current


def write_new_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"refusing to overwrite evidence output: {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def refresh_stopped_manifest(manifest_path: Path, output: Path) -> None:
    manifest = load_manifest(manifest_path)
    current = verify_current_identity(manifest)
    if current.get("State", {}).get("Running") is not False:
        raise RuntimeError("stopped manifest refresh requires the Qwen container to be stopped")
    refreshed = dict(manifest)
    refreshed["original_created_at"] = manifest["created_at"]
    refreshed["original_manifest_sha256"] = manifest_sha256(manifest_path)
    refreshed["inventory_method"] = "stopped-identity-refresh"
    refreshed["created_at"] = utc_now()
    refreshed["container"] = dict(manifest["container"])
    refreshed["container"]["running"] = False
    refreshed["container"]["health"] = (
        (current.get("State", {}).get("Health") or {}).get("Status")
    )
    write_new_json(output, refreshed)


def verify_live(manifest_path: Path, max_age_hours: float) -> None:
    manifest = load_manifest(manifest_path)
    age = datetime.now(timezone.utc) - parse_time(manifest["created_at"])
    if age.total_seconds() < 0 or age.total_seconds() > max_age_hours * 3600:
        raise RuntimeError("manifest is stale")
    verify_current_identity(manifest)


def verify_report(manifest_path: Path, report_path: Path, gates: list[str], max_age_hours: float) -> None:
    manifest = load_manifest(manifest_path)
    report = json.loads(report_path.read_text())
    if report.get("accepted") is not True:
        raise RuntimeError("gate report is not accepted: true")
    if report.get("run_id") != manifest.get("run_id"):
        raise RuntimeError("gate report run_id mismatch")
    if report.get("manifest_sha256") != manifest_sha256(manifest_path):
        raise RuntimeError("gate report manifest_sha256 mismatch")
    age = datetime.now(timezone.utc) - parse_time(report["created_at"])
    if age.total_seconds() < 0 or age.total_seconds() > max_age_hours * 3600:
        raise RuntimeError("gate report is stale")
    for gate in gates:
        if report.get("gates", {}).get(gate) is not True:
            raise RuntimeError(f"required gate did not pass: {gate}")
    if report.get("purge_eligible") is True:
        pins = report.get("pins") or {}
        pin_hash = hashlib.sha256(
            json.dumps(pins, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if report.get("pin_set_sha256") != pin_hash:
            raise RuntimeError("gate report pin_set_sha256 mismatch")
        report_root = report_path.parent.resolve(strict=True)
        previous = "0" * 64
        for item in report.get("evidence_chain") or []:
            artifact = (report_root / str(item.get("artifact_path", ""))).resolve(strict=True)
            try:
                artifact.relative_to(report_root)
            except ValueError as exc:
                raise RuntimeError("evidence artifact escapes the acceptance run") from exc
            artifact_hash = manifest_sha256(artifact)
            if artifact_hash != item.get("artifact_sha256"):
                raise RuntimeError(f"evidence artifact hash mismatch: {item.get('name')}")
            expected = hashlib.sha256(
                f"{previous}:{item.get('name')}:{artifact_hash}:{pin_hash}".encode()
            ).hexdigest()
            if item.get("previous_sha256") != previous or item.get("entry_sha256") != expected:
                raise RuntimeError(f"evidence chain mismatch: {item.get('name')}")
            previous = expected
        if not report.get("evidence_chain") or report.get("chain_head_sha256") != previous:
            raise RuntimeError("evidence chain head mismatch")


def quarantine(manifest_path: Path, quarantine_root: Path) -> Path:
    manifest = load_manifest(manifest_path)
    verify_targets(manifest)
    allowed = {"service_root": SERVICE_ROOT, "model_root": MODEL_ROOT}
    for forbidden in FORBIDDEN_ROOTS:
        if forbidden == SERVICE_ROOT or forbidden == MODEL_ROOT:
            raise RuntimeError("forbidden active gateway path overlaps Qwen target")
    run_root = quarantine_root / manifest["run_id"]
    run_root.mkdir(parents=True, mode=0o700, exist_ok=False)
    moved = []
    try:
        for target in manifest["filesystem_targets"]:
            if not target.get("exists"):
                continue
            source = Path(target["path"])
            if source != allowed.get(target["kind"]) or source.resolve() != source:
                raise RuntimeError(f"target is outside its canonical allowlist: {source}")
            if source.stat().st_dev != run_root.stat().st_dev:
                raise RuntimeError(f"quarantine is not on the same device as {source}")
            destination = run_root / target["kind"]
            os.rename(source, destination)
            moved.append((source, destination))
    except Exception:
        for source, destination in reversed(moved):
            if destination.exists() and not source.exists():
                os.rename(destination, source)
        raise
    return run_root


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory")
    inventory.add_argument("--container", default=EXPECTED_CONTAINER)
    inventory.add_argument("--output", type=Path)
    inventory.add_argument("--check-only", action="store_true")
    targets = sub.add_parser("verify-targets")
    targets.add_argument("--manifest", type=Path, required=True)
    supervisors = sub.add_parser("verify-no-supervisors")
    supervisors.add_argument("--manifest", type=Path, required=True)
    refresh = sub.add_parser("refresh-stopped")
    refresh.add_argument("--manifest", type=Path, required=True)
    refresh.add_argument("--output", type=Path, required=True)
    live = sub.add_parser("verify-live")
    live.add_argument("--manifest", type=Path, required=True)
    live.add_argument("--max-age-hours", type=float, default=24)
    report = sub.add_parser("verify-report")
    report.add_argument("--manifest", type=Path, required=True)
    report.add_argument("--report", type=Path, required=True)
    report.add_argument("--required-gate", action="append", default=[])
    report.add_argument("--max-age-hours", type=float, default=24)
    move = sub.add_parser("quarantine")
    move.add_argument("--manifest", type=Path, required=True)
    move.add_argument("--quarantine-root", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "inventory":
        payload = create_manifest(args.container)
        rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.check_only:
            sys.stdout.write(rendered)
        else:
            if not args.output:
                parser.error("inventory requires --output unless --check-only is used")
            write_new_json(args.output, payload)
        return 0
    if args.command == "refresh-stopped":
        refresh_stopped_manifest(args.manifest, args.output)
        return 0
    if args.command == "verify-targets":
        verify_targets(load_manifest(args.manifest))
    elif args.command == "verify-no-supervisors":
        verify_no_supervisors(load_manifest(args.manifest))
    elif args.command == "verify-live":
        verify_live(args.manifest, args.max_age_hours)
    elif args.command == "verify-report":
        verify_report(args.manifest, args.report, args.required_gate, args.max_age_hours)
    elif args.command == "quarantine":
        print(quarantine(args.manifest, args.quarantine_root))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"qwen manifest error: {error}", file=sys.stderr)
        raise SystemExit(1)
