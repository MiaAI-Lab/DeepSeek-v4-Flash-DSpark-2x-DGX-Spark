#!/usr/bin/env python3
"""Manifest-bound, worker-first DeepSeek TP=2 recovery transaction."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


MAX_JSON_BYTES = 65_536
TARGET_ID = "deepseek-v4-flash-dspark"
EXPECTED_FIELDS = (
    "repositoryRevision",
    "imageDigest",
    "configurationHash",
    "artifactRevision",
    "observerRevision",
    "adapterRevision",
)


class RecoveryError(RuntimeError):
    pass


class DeploymentExecutor:
    """Allowlisted adapter over the checked-in recovery-specific script modes."""

    def __init__(self, script_dir, inspections, command_timeout_seconds=180):
        self.script_dir = Path(script_dir).resolve()
        self.inspections = inspections
        self.command_timeout_seconds = command_timeout_seconds

    def _run(self, arguments, *, ambiguous_on_failure=False):
        try:
            result = subprocess.run(
                [str(self.script_dir / arguments[0]), *arguments[1:]],
                cwd=self.script_dir,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.command_timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("ambiguous_transport_result") from error
        if result.returncode != 0:
            reason = (
                "ambiguous_transport_result"
                if ambiguous_on_failure
                else "command_failed"
            )
            raise RuntimeError(reason)
        return result.stdout.strip()

    def __call__(self, phase, host, _context):
        rank = "head" if host == "john" else "worker"
        if phase == "inspect":
            return dict(self.inspections[host])
        if phase == "stop":
            self._run(
                ["stop-deepseek-v4-flash-dspark.sh", "--recovery-rank", rank],
                ambiguous_on_failure=True,
            )
            return {"stopped": True}
        if phase == "confirm_stopped":
            self._run(
                ["status-deepseek-v4-flash-dspark.sh", "--recovery-stopped", rank]
            )
            return {"stopped": True}
        if phase == "start":
            self._run(
                ["start-deepseek-v4-flash-dspark.sh", "--recovery-rank", rank],
                ambiguous_on_failure=True,
            )
            return {"started": True}
        if phase == "wait_fresh_generation":
            old = self.inspections[host]["processGeneration"]
            for _attempt in range(30):
                generation = self._run(
                    [
                        "status-deepseek-v4-flash-dspark.sh",
                        "--recovery-generation",
                        rank,
                    ]
                )
                if generation and generation != old:
                    return {"processGeneration": generation}
                time.sleep(5)
            raise RuntimeError("fresh_generation_timeout")
        if phase == "stream_verify":
            self._run(
                ["smoke-deepseek-v4-flash-dspark.sh", "--single-stream-recovery"],
                ambiguous_on_failure=True,
            )
            return {"firstEvent": True, "completed": True, "requestCount": 1}
        raise RuntimeError("operation_not_allowed")


def hash_manifest(value):
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def load_json(path):
    try:
        raw = Path(path).read_bytes()
        if len(raw) > MAX_JSON_BYTES:
            raise RecoveryError("input_too_large")
        value = json.loads(raw)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise RecoveryError("invalid_json_input") from error
    if not isinstance(value, dict):
        raise RecoveryError("invalid_json_input")
    return value


def parse_time(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise RecoveryError("invalid_expiry") from error
    if parsed.tzinfo is None:
        raise RecoveryError("invalid_expiry")
    return parsed


def atomic_write_json(path, value, *, exclusive=False):
    raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode() + b"\n"
    if len(raw) > MAX_JSON_BYTES:
        raise RecoveryError("receipt_too_large")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise RecoveryError("generation_already_claimed") from error
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_inputs(manifest, authorization, now):
    issues = []
    if manifest.get("contractVersion") != 1 or authorization.get("contractVersion") != 1:
        issues.append("contract_version_mismatch")
    if manifest.get("targetId") != TARGET_ID or authorization.get("targetId") != TARGET_ID:
        issues.append("target_mismatch")
    if manifest.get("topology") != "deepseek_tp2":
        issues.append("topology_mismatch")
    if authorization.get("manifestHash") != hash_manifest(manifest):
        issues.append("manifest_hash_mismatch")
    ranks = manifest.get("ranks")
    if not isinstance(ranks, dict) or set(ranks) != {"john", "ofus"}:
        issues.append("rank_manifest_mismatch")
    elif ranks["john"].get("sourceRank") != 0 or ranks["ofus"].get("sourceRank") != 1:
        issues.append("rank_manifest_mismatch")
    generation = authorization.get("incidentGeneration")
    if not isinstance(generation, int) or generation < 1:
        issues.append("incident_generation_invalid")
    nonce = authorization.get("commandNonce")
    if not isinstance(nonce, str) or len(nonce) < 12:
        issues.append("command_nonce_invalid")
    if parse_time(authorization.get("expiresAt")) <= now:
        issues.append("authorization_expired")
    for field in EXPECTED_FIELDS:
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            issues.append(f"{field}_missing")
    if issues:
        raise RecoveryError(",".join(issues))


def _preflight(manifest, authorization, now, executor):
    inspections = {}
    for host in ("john", "ofus"):
        try:
            inspection = executor("inspect", host, manifest)
        except RuntimeError as error:
            raise RecoveryError(f"inspect_failed:{host}") from error
        if not isinstance(inspection, dict):
            raise RecoveryError(f"inspect_invalid:{host}")
        if inspection.get("hostId") != host:
            raise RecoveryError(f"hostId_mismatch:{host}")
        if inspection.get("sourceRank") != manifest["ranks"][host]["sourceRank"]:
            raise RecoveryError(f"sourceRank_mismatch:{host}")
        if inspection.get("targetId") != TARGET_ID:
            raise RecoveryError(f"targetId_mismatch:{host}")
        if inspection.get("manifestHash") != authorization["manifestHash"]:
            raise RecoveryError(f"manifestHash_mismatch:{host}")
        if inspection.get("incidentGeneration") != authorization["incidentGeneration"]:
            raise RecoveryError(f"incidentGeneration_mismatch:{host}")
        if inspection.get("commandNonce") != authorization["commandNonce"]:
            raise RecoveryError(f"commandNonce_mismatch:{host}")
        observed_at = parse_time(inspection.get("observedAt"))
        age = (now - observed_at).total_seconds()
        if age < 0 or age > 45:
            raise RecoveryError(f"inspection_stale:{host}")
        for field in EXPECTED_FIELDS:
            if inspection.get(field) != manifest.get(field):
                raise RecoveryError(f"{field}_mismatch:{host}")
        generation = inspection.get("processGeneration")
        if not isinstance(generation, str) or not generation:
            raise RecoveryError(f"processGeneration_missing:{host}")
        inspections[host] = inspection
    return inspections


def recover(
    *,
    manifest_path,
    authorization_path,
    receipt_dir,
    now,
    execute=False,
    executor=None,
    clock=None,
):
    manifest = load_json(manifest_path)
    authorization = load_json(authorization_path)
    _validate_inputs(manifest, authorization, now)
    if executor is None:
        raise RecoveryError("executor_required")
    inspections = _preflight(manifest, authorization, now, executor)
    if not execute:
        return {
            "status": "validated_dry_run",
            "incidentGeneration": authorization["incidentGeneration"],
            "preflightHosts": ["john", "ofus"],
        }

    receipt_path = Path(receipt_dir) / (
        f"incident-{authorization['incidentGeneration']}.json"
    )
    receipt = {
        "contractVersion": 1,
        "incidentGeneration": authorization["incidentGeneration"],
        "targetId": TARGET_ID,
        "manifestHash": authorization["manifestHash"],
        "commandNonce": authorization["commandNonce"],
        "status": "authorized",
        "oldRankGenerations": {
            host: inspections[host]["processGeneration"] for host in ("john", "ofus")
        },
    }
    atomic_write_json(receipt_path, receipt, exclusive=True)

    expiry = parse_time(authorization["expiresAt"])
    clock = clock or (lambda: now)

    def action(phase, host):
        if clock() >= expiry:
            receipt.update(status="circuit_open", failedPhase="deadline", failedHost=host)
            atomic_write_json(receipt_path, receipt)
            raise RecoveryError("authorization_expired")
        try:
            return executor(phase, host, manifest)
        except RuntimeError as error:
            if "ambiguous" in str(error):
                receipt.update(status="ambiguous", failedPhase=phase, failedHost=host)
                atomic_write_json(receipt_path, receipt)
                raise RecoveryError("ambiguous_outcome") from error
            receipt.update(status="circuit_open", failedPhase=phase, failedHost=host)
            atomic_write_json(receipt_path, receipt)
            raise RecoveryError(f"{phase}_failed:{host}") from error

    try:
        action("stop", "john")
        action("stop", "ofus")
        for host in ("john", "ofus"):
            stopped = action("confirm_stopped", host)
            if stopped.get("stopped") is not True:
                raise RuntimeError("stop_not_confirmed")

        rank_generations = {}
        for host in ("ofus", "john"):
            action("start", host)
            fresh = action("wait_fresh_generation", host)
            generation = fresh.get("processGeneration")
            if not generation or generation == inspections[host]["processGeneration"]:
                raise RuntimeError("fresh_generation_not_observed")
            rank_generations[host] = generation

        verification = action("stream_verify", "john")
        if verification != {
            "firstEvent": True,
            "completed": True,
            "requestCount": 1,
        }:
            raise RuntimeError("stream_verification_invalid")
    except RecoveryError:
        raise
    except RuntimeError as error:
        receipt.update(status="circuit_open", failedPhase="validation")
        atomic_write_json(receipt_path, receipt)
        raise RecoveryError(str(error)) from error

    receipt.update(
        status="completed",
        rankGenerations={
            "john": rank_generations["john"],
            "ofus": rank_generations["ofus"],
        },
        streamVerification=verification,
    )
    atomic_write_json(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--receipt-dir", required=True)
    parser.add_argument("--john-inspection", required=True)
    parser.add_argument("--ofus-inspection", required=True)
    parser.add_argument("--script-dir", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    inspections = {
        "john": load_json(args.john_inspection),
        "ofus": load_json(args.ofus_inspection),
    }
    executor = DeploymentExecutor(args.script_dir, inspections)
    result = recover(
        manifest_path=args.manifest,
        authorization_path=args.authorization,
        receipt_dir=args.receipt_dir,
        now=datetime.now(timezone.utc),
        execute=args.execute,
        executor=executor,
        clock=lambda: datetime.now(timezone.utc),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
