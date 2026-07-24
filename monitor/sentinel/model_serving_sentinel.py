#!/usr/bin/env python3
"""Emit one bounded, independently sequenced U4 evidence envelope per rank."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


MAX_JSON_BYTES = 16_384
FRESH_SECONDS = 45
METRIC_NAMES = (
    "runningRequests",
    "waitingRequests",
    "iterationTokens",
    "generatedTokens",
    "completedRequests",
    "requestAttributedKvActivity",
)


def isoformat(value):
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def parse_time(value):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None


def atomic_write_json(path, value):
    raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode() + b"\n"
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("evidence payload too large")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _valid_progress(
    progress,
    observer_revision,
    source_rank,
    process_generation,
    now,
):
    if not isinstance(progress, dict):
        return False
    observed_at = parse_time(progress.get("observedAt"))
    metrics = progress.get("metrics")
    return bool(
        progress.get("contractVersion") == 1
        and progress.get("scope") == "rank_worker"
        and progress.get("observerRevision") == observer_revision
        and progress.get("sourceRank") == source_rank
        and progress.get("processGeneration") == process_generation
        and observed_at is not None
        and 0 <= (now - observed_at).total_seconds() <= FRESH_SECONDS
        and progress.get("lifecycle") in {"serving", "loading", "stopping", "failed"}
        and isinstance(metrics, dict)
        and all(
            isinstance(metrics.get(name), (int, float))
            and not isinstance(metrics.get(name), bool)
            and metrics[name] >= 0
            for name in METRIC_NAMES
        )
    )


def build_evidence(
    *,
    progress,
    target_id,
    observer_id,
    observer_revision,
    source_host,
    source_rank,
    source_boot_id,
    process_generation,
    configuration_revision,
    manifest_hash,
    prior_state,
    now,
):
    valid = _valid_progress(
        progress, observer_revision, source_rank, process_generation, now
    )
    metrics = (
        {name: progress["metrics"][name] for name in METRIC_NAMES}
        if valid
        else {name: 0 for name in METRIC_NAMES}
    )
    sequence = int(prior_state.get("sourceSequence", 0)) + 1
    observed_at = isoformat(now)
    evidence = {
        "contractVersion": 1,
        "targetId": target_id,
        "topology": "deepseek_tp2",
        "observerId": observer_id,
        "sourceHost": source_host,
        "sourceRank": source_rank,
        "sourceBootId": source_boot_id,
        "processGeneration": process_generation,
        "configurationRevision": configuration_revision,
        "manifestHash": manifest_hash,
        "sourceSequence": sequence,
        "observedAt": observed_at,
        "receivedAt": observed_at,
        "lifecycle": progress.get("lifecycle") if valid else "unknown",
        "metrics": metrics,
    }
    return evidence, {"sourceSequence": sequence}


def load_json(path):
    try:
        raw = Path(path).read_bytes()
        if len(raw) > MAX_JSON_BYTES:
            return {}
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def collect(args):
    now = datetime.now(timezone.utc)
    state = load_json(args.state)
    evidence, next_state = build_evidence(
        progress=load_json(args.progress),
        target_id=args.target_id,
        observer_id=args.observer_id,
        observer_revision=args.observer_revision,
        source_host=args.source_host,
        source_rank=args.source_rank,
        source_boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        process_generation=args.process_generation,
        configuration_revision=args.configuration_revision,
        manifest_hash=args.manifest_hash,
        prior_state=state,
        now=now,
    )
    atomic_write_json(args.state, next_state)
    atomic_write_json(args.output, evidence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-id", default="deepseek-v4-flash-dspark")
    parser.add_argument("--observer-id", required=True)
    parser.add_argument("--observer-revision", required=True)
    parser.add_argument("--source-host", required=True)
    parser.add_argument("--source-rank", required=True, type=int, choices=(0, 1))
    parser.add_argument("--process-generation", required=True)
    parser.add_argument("--configuration-revision", required=True)
    parser.add_argument("--manifest-hash", required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=15)
    args = parser.parse_args()
    while True:
        collect(args)
        if not args.watch:
            return
        time.sleep(max(args.interval, 1))


if __name__ == "__main__":
    main()
