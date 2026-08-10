#!/usr/bin/env python3
"""Verify or regenerate the pinned chat-completion request-field contract."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "scripts/chat-completion-request-fields.json"
PROXY_PATH = ROOT / "scripts/origin-auth-proxy.py"
COMPOSE_PATH = ROOT / "docker-compose.dspark.yml"
START_MARKER = "# BEGIN GENERATED CHAT COMPLETION REQUEST FIELDS"
END_MARKER = "# END GENERATED CHAT COMPLETION REQUEST FIELDS"


def fail(message: str) -> None:
    raise ValueError(message)


def checked_string_list(contract: dict, name: str) -> list[str]:
    value = contract.get(name)
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        fail(f"{name} must be a list of non-empty strings")
    if value != sorted(set(value)):
        fail(f"{name} must be sorted and duplicate-free")
    return value


def load_contract() -> dict:
    contract = json.loads(CONTRACT_PATH.read_text())
    if contract.get("contract_version") != 1:
        fail("unsupported contract_version")
    image = contract.get("pinned_runtime_image")
    if not isinstance(image, str) or not re.fullmatch(r"[^@]+@sha256:[0-9a-f]{64}", image):
        fail("pinned_runtime_image must use an immutable sha256 digest")
    commit = contract.get("vllm_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        fail("vllm_commit must be a full Git commit")
    schema = checked_string_list(contract, "schema_fields")
    extensions = checked_string_list(contract, "repository_extensions")
    allowed = checked_string_list(contract, "allowed_fields")
    payload = checked_string_list(contract, "repository_payload_fields")
    if allowed != sorted(set(schema) | set(extensions)):
        fail("allowed_fields must be the schema/extension union")
    missing = sorted(set(payload) - set(allowed))
    if missing:
        fail(f"repository payload fields are not allowed: {missing}")
    return contract


def proxy_fields() -> list[str]:
    tree = ast.parse(PROXY_PATH.read_text(), filename=str(PROXY_PATH))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "CHAT_COMPLETION_REQUEST_FIELDS" for target in node.targets):
            continue
        value = node.value
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "frozenset"
            and len(value.args) == 1
        ):
            fail("CHAT_COMPLETION_REQUEST_FIELDS must be a literal frozenset")
        fields = ast.literal_eval(value.args[0])
        if not isinstance(fields, (set, tuple, list)) or not all(isinstance(item, str) for item in fields):
            fail("proxy allowlist is not a string collection")
        return sorted(fields)
    fail("proxy allowlist assignment is missing")


def schema_fields_from_container(contract: dict) -> list[str]:
    model = contract["schema_model"]
    module_name, class_name = model.rsplit(".", 1)
    snippet = (
        "import json; "
        f"from {module_name} import {class_name}; "
        f"print(json.dumps(sorted({class_name}.model_fields)))"
    )
    result = subprocess.run(
        [
            "docker", "run", "--rm", "--entrypoint", "python3",
            contract["pinned_runtime_image"], "-c", snippet,
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode:
        fail(f"pinned-container schema extraction failed: {result.stderr.strip()}")
    return normalize_schema_json(result.stdout)


def normalize_schema_json(raw: str) -> list[str]:
    fields = json.loads(raw)
    if not isinstance(fields, list) or not all(isinstance(item, str) and item for item in fields):
        fail("schema field extraction must produce a JSON string array")
    return sorted(set(fields))


def render_proxy_block(fields: list[str]) -> str:
    body = "\n".join(f'        "{field}",' for field in fields)
    return (
        f"{START_MARKER}\n"
        "CHAT_COMPLETION_REQUEST_FIELDS = frozenset(\n"
        "    {\n"
        f"{body}\n"
        "    }\n"
        ")\n"
        f"{END_MARKER}"
    )


def regenerate(contract: dict, schema_fields: list[str]) -> None:
    contract["schema_fields"] = schema_fields
    contract["allowed_fields"] = sorted(
        set(schema_fields) | set(contract["repository_extensions"])
    )
    missing = sorted(set(contract["repository_payload_fields"]) - set(contract["allowed_fields"]))
    if missing:
        fail(f"new pinned schema dropped repository payload fields: {missing}")
    source = PROXY_PATH.read_text()
    start = source.find(START_MARKER)
    end = source.find(END_MARKER)
    if start < 0 or end < start:
        fail("generated proxy allowlist markers are missing")
    end += len(END_MARKER)
    PROXY_PATH.write_text(source[:start] + render_proxy_block(contract["allowed_fields"]) + source[end:])
    CONTRACT_PATH.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")


def verify_local(contract: dict) -> None:
    if proxy_fields() != contract["allowed_fields"]:
        fail("proxy allowlist differs from checked-in allowed_fields; regenerate it")
    compose = COMPOSE_PATH.read_text()
    if contract["pinned_runtime_image"] not in compose:
        fail("contract image does not match the pinned compose image")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="check the checked-in contract (default)")
    parser.add_argument(
        "--check-container", action="store_true",
        help="extract ChatCompletionRequest.model_fields inside the pinned image and compare",
    )
    parser.add_argument(
        "--schema-fields", type=Path,
        help="read a JSON field array instead of starting the pinned container",
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="regenerate the manifest and proxy block from the pinned container/schema JSON",
    )
    args = parser.parse_args()
    if args.generate and args.check:
        parser.error("--generate and --check are mutually exclusive")
    if args.schema_fields and args.check_container:
        parser.error("--schema-fields and --check-container are mutually exclusive")

    try:
        contract = load_contract()
        extracted = None
        if args.schema_fields:
            extracted = normalize_schema_json(args.schema_fields.read_text())
        elif args.check_container or args.generate:
            extracted = schema_fields_from_container(contract)
        if args.generate:
            regenerate(contract, extracted)
            contract = load_contract()
        verify_local(contract)
        if extracted is not None and extracted != contract["schema_fields"]:
            fail("pinned ChatCompletionRequest.model_fields differ from the checked-in schema")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"chat-completion field contract failed: {exc}", file=sys.stderr)
        return 1
    print("chat-completion field contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
