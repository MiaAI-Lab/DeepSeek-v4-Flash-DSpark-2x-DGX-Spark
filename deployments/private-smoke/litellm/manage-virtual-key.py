#!/usr/bin/env python3
"""Rotate or revoke the long-lived model-scoped LiteLLM virtual key safely."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
import tempfile
import urllib.error
import urllib.request

MODEL = "deepseek-v4-flash-0731-smoke"


def read_key_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"key file must be a regular file: {path}")
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError(f"key file must have mode 0600: {path}")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"key file is empty: {path}")
    return value


def call(base: str, path: str, token: str | None, body: dict | None, expected: set[int]) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    root = base.rstrip("/").removesuffix("/v1")
    request = urllib.request.Request(root + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status = response.status
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        status = error.code
        raw = error.read()
        error.close()
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            payload = {}
    if status not in expected:
        raise RuntimeError(f"unexpected status {status} for LiteLLM administrative operation")
    return status, payload


def authenticate(base: str, key: str) -> bool:
    status, payload = call(base, "/v1/models", key, None, {200, 401, 403})
    if status != 200:
        return False
    return {item.get("id") for item in payload.get("data", [])} == {MODEL}


def delete_key(base: str, master: str, key: str) -> None:
    call(base, "/key/delete", master, {"keys": [key]}, {200})


def write_receipt(path: Path | None, payload: dict) -> None:
    if path is None:
        print(json.dumps(payload, sort_keys=True))
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("refusing to overwrite key lifecycle receipt") from error
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def rotate(base: str, master_file: Path, key_file: Path, receipt: Path | None) -> None:
    master = read_key_file(master_file)
    old = read_key_file(key_file)
    if not authenticate(base, old):
        raise RuntimeError("existing virtual key is not authenticated; refusing rotation")
    _, response = call(
        base,
        "/key/generate",
        master,
        {
            "models": [MODEL],
            "allowed_routes": ["/v1/models", "/v1/models/*", "/v1/chat/completions"],
            # LiteLLM requires aliases to be unique while the old and replacement
            # keys overlap during proof. Reusing the stable alias returns 400.
            "key_alias": "hermes-deepseek-smoke-rotate-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
            "metadata": {"scope": "synthetic-hermes-smoke"},
        },
        {200},
    )
    new = response.get("key")
    if not isinstance(new, str) or not new.startswith("sk-") or new == old:
        raise RuntimeError("LiteLLM did not return a distinct replacement key")
    if not authenticate(base, new):
        delete_key(base, master, new)
        raise RuntimeError("replacement key failed authentication; old key retained")

    fd, temporary_name = tempfile.mkstemp(prefix=key_file.name + ".rotate-", dir=key_file.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(new + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Install the already-proven replacement before revoking the old key.
        # A crash in the overlap window leaves two valid keys, not an outage or
        # an invalid on-disk credential. The old value remains in memory only.
        os.replace(temporary, key_file)
        key_file.chmod(0o600)
        delete_key(base, master, old)
        old_key_rejected = not authenticate(base, old)
        new_key_authenticated = authenticate(base, new)
        if not old_key_rejected and new_key_authenticated:
            # Revocation failed: restore the still-valid old file and remove the
            # replacement so the operation is rolled back without ambiguity.
            replacement_fd, replacement_name = tempfile.mkstemp(
                prefix=key_file.name + ".rollback-", dir=key_file.parent
            )
            with os.fdopen(replacement_fd, "w", encoding="utf-8") as handle:
                os.fchmod(handle.fileno(), 0o600)
                handle.write(old + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(replacement_name, key_file)
            delete_key(base, master, new)
            raise RuntimeError("old-key revocation failed; rotation rolled back")
        if not old_key_rejected or not new_key_authenticated:
            raise RuntimeError(
                "rotation proof failed; replacement file retained for service continuity"
            )
    finally:
        if temporary.exists():
            temporary.unlink()
    write_receipt(receipt, {
        "schema_version": 1,
        "operation": "rotate",
        "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_scope": MODEL,
        "old_key_rejected": True,
        "new_key_authenticated": True,
        "key_file_mode_0600": True,
    })


def revoke(base: str, master_file: Path, key_file: Path, receipt: Path | None) -> None:
    master = read_key_file(master_file)
    old = read_key_file(key_file)
    delete_key(base, master, old)
    old_key_rejected = not authenticate(base, old)
    if not old_key_rejected:
        raise RuntimeError("revoked key still authenticates")
    key_file.unlink()
    write_receipt(receipt, {
        "schema_version": 1,
        "operation": "revoke",
        "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_scope": MODEL,
        "old_key_rejected": True,
        "key_file_removed": True,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("rotate", "revoke"))
    parser.add_argument("--base-url", default="http://127.0.0.1:4001")
    parser.add_argument("--master-key-file", required=True, type=Path)
    parser.add_argument("--key-file", required=True, type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.operation == "rotate":
        rotate(args.base_url, args.master_key_file, args.key_file, args.receipt)
    else:
        revoke(args.base_url, args.master_key_file, args.key_file, args.receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
