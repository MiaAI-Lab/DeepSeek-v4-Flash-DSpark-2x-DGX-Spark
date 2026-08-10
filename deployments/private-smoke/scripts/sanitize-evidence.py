#!/usr/bin/env python3
"""Reject secrets, private addresses, host paths, and non-allowlisted JSON."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


CREDENTIAL_KEYS = re.compile(
    r"(^|_)(api_?key|authorization|bearer|cookie|password|private_?key|secret|token)($|_)",
    re.IGNORECASE,
)
NON_CREDENTIAL_METRIC_KEYS = {
    "reported_token_capacity",
}
CREDENTIAL_VALUES = (
    re.compile(r"\bBearer\s+\S+", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?:https?|postgres(?:ql)?):\/\/[^\s/:]+:[^\s/@]+@", re.IGNORECASE),
)
IP_CANDIDATE = re.compile(r"(?<![A-Za-z0-9])(?:\d{1,3}\.){3}\d{1,3}(?![A-Za-z0-9])")
HOST_PATH = re.compile(r"(?:^|[\s='\"])(/(?:Users|home|var|tmp|private|opt)/[^\s'\"]*)")


class EvidenceError(ValueError):
    pass


def scan(value: Any, location: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise EvidenceError(f"non-string object key at {location}")
            if (
                key not in NON_CREDENTIAL_METRIC_KEYS
                and CREDENTIAL_KEYS.search(key)
                and child not in (None, False, True, 0, "")
            ):
                raise EvidenceError(f"credential-shaped field is populated at {location}.{key}")
            scan(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            scan(child, f"{location}[{index}]")
    elif isinstance(value, str):
        for pattern in CREDENTIAL_VALUES:
            if pattern.search(value):
                raise EvidenceError(f"credential-like value at {location}")
        if HOST_PATH.search(value):
            raise EvidenceError(f"host path at {location}")
        for candidate in IP_CANDIDATE.findall(value):
            try:
                address = ipaddress.ip_address(candidate)
            except ValueError:
                continue
            if address.is_private or address.is_loopback or address.is_link_local or address in ipaddress.ip_network("100.64.0.0/10"):
                raise EvidenceError(f"private address at {location}")


def resolve_pointer(root: dict, pointer: str) -> dict:
    if not pointer.startswith("#/"):
        raise EvidenceError(f"unsupported schema reference: {pointer}")
    current: Any = root
    for part in pointer[2:].split("/"):
        current = current[part.replace("~1", "/").replace("~0", "~")]
    if not isinstance(current, dict):
        raise EvidenceError(f"schema reference is not an object: {pointer}")
    return current


def type_matches(value: Any, expected: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "null": value is None,
    }.get(expected, False)


def validate(value: Any, schema: dict, root: dict, location: str = "$") -> None:
    if "$ref" in schema:
        validate(value, resolve_pointer(root, schema["$ref"]), root, location)
        return
    if "const" in schema and value != schema["const"]:
        raise EvidenceError(f"wrong constant at {location}")
    if "enum" in schema and value not in schema["enum"]:
        raise EvidenceError(f"value outside enum at {location}")
    expected = schema.get("type")
    matches = any(type_matches(value, item) for item in expected) if isinstance(expected, list) else type_matches(value, expected)
    if expected and not matches:
        raise EvidenceError(f"wrong type at {location}: expected {expected}")
    if isinstance(value, dict):
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise EvidenceError(f"missing fields at {location}: {', '.join(missing)}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise EvidenceError(f"extra fields at {location}: {', '.join(extra)}")
        additional = schema.get("additionalProperties")
        for key, child in value.items():
            child_schema = properties.get(key)
            if child_schema is None and isinstance(additional, dict):
                child_schema = additional
            if child_schema is not None:
                validate(child, child_schema, root, f"{location}.{key}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise EvidenceError(f"too few items at {location}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise EvidenceError(f"too many items at {location}")
        if schema.get("uniqueItems"):
            encoded = [json.dumps(item, sort_keys=True) for item in value]
            if len(encoded) != len(set(encoded)):
                raise EvidenceError(f"duplicate items at {location}")
        if "items" in schema:
            for index, child in enumerate(value):
                validate(child, schema["items"], root, f"{location}[{index}]")
    if isinstance(value, str) and "pattern" in schema and not re.search(schema["pattern"], value):
        raise EvidenceError(f"string pattern mismatch at {location}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise EvidenceError(f"number below minimum at {location}")
        if "maximum" in schema and value > schema["maximum"]:
            raise EvidenceError(f"number above maximum at {location}")


def validate_acceptance_semantics(payload: dict, schema: dict) -> None:
    if schema.get("$id") != "https://plexiz.invalid/schemas/dual-dspark-acceptance.json":
        return
    pins = payload["pins"]
    pin_hash = hashlib.sha256(
        json.dumps(pins, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if payload["pin_set_sha256"] != pin_hash:
        raise EvidenceError("pin_set_sha256 does not match the allowlisted pins")
    previous = "0" * 64
    for index, item in enumerate(payload["evidence_chain"]):
        if item["previous_sha256"] != previous:
            raise EvidenceError(f"broken evidence chain at index {index}")
        expected = hashlib.sha256(
            f"{previous}:{item['name']}:{item['artifact_sha256']}:{pin_hash}".encode()
        ).hexdigest()
        if item["entry_sha256"] != expected:
            raise EvidenceError(f"wrong evidence entry hash at index {index}")
        previous = expected
    if payload["chain_head_sha256"] != previous:
        raise EvidenceError("chain_head_sha256 does not match the evidence chain")
    all_gates = all(payload["gates"].values())
    all_functional = all(item["accepted"] for item in payload["functional_runs"])
    all_performance = payload["performance"]["direct"]["accepted"] and payload["performance"]["litellm"]["accepted"]
    expected_accepted = all_gates and all_functional and all_performance and payload["soak"]["accepted"]
    if payload["accepted"] != expected_accepted:
        raise EvidenceError("accepted does not match its component gates")
    if payload["purge_eligible"] != payload["accepted"]:
        raise EvidenceError("purge_eligible must equal accepted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--schema", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.scan_only and args.schema is None:
        parser.error("--schema is required unless --scan-only is used")
    raw = args.input.read_text(encoding="utf-8") if args.input else sys.stdin.read()
    payload = json.loads(raw)
    scan(payload)
    if args.schema:
        schema = json.loads(args.schema.read_text(encoding="utf-8"))
        validate(payload, schema, schema)
        if isinstance(payload, dict):
            validate_acceptance_semantics(payload, schema)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            descriptor = os.open(
                args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
            )
        except FileExistsError as error:
            raise EvidenceError("refusing to overwrite evidence output") from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(rendered)
    elif not args.scan_only:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EvidenceError, json.JSONDecodeError, OSError, KeyError) as error:
        print(f"evidence rejected: {error}", file=sys.stderr)
        raise SystemExit(1)
