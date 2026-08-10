#!/usr/bin/env python3
"""Run the pinned Model Serving Minefield doctor without secret argv/logging."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

PINNED_COMMIT = "2b453b8a69dbaf8dc9d521dc2d6212cdaceb8169"
REPOSITORY = "https://github.com/Blackwellboy/model-serving-minefield"
HELPER = r"""
import contextlib
import io
import sys
from minefield.data import minefield_doctor
key = sys.stdin.read().strip()
if not key:
    raise SystemExit("empty key on private stdin")
sys.argv = [
    "minefield-doctor", "--base-url", sys.argv[1], "--model", sys.argv[2],
    "--api-key", key, "--json", sys.argv[3],
]
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    minefield_doctor.main()
"""


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = os.path.expandvars(value)
    return values


def read_mode_0600(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("key file must be a regular file")
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError("key file must have mode 0600")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError("key file is empty")
    return value


def summarize(raw: dict) -> dict:
    coverage = raw.get("coverage")
    if not isinstance(coverage, dict):
        raise ValueError("Minefield report has no structured coverage")
    executed_ids = coverage.get("executed")
    problem_ids = coverage.get("problems")
    inconclusive_ids = coverage.get("inconclusive")
    unimplemented = coverage.get("not_implemented_count")
    registry_total = coverage.get("registry_total")
    for name, value in (
        ("executed", executed_ids), ("problems", problem_ids),
        ("inconclusive", inconclusive_ids),
    ):
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"Minefield {name} coverage is malformed")
    if not isinstance(unimplemented, int) or unimplemented < 0:
        raise ValueError("Minefield unimplemented count is malformed")
    if not isinstance(registry_total, int) or registry_total <= 0:
        raise ValueError("Minefield registry total is malformed")
    if unimplemented > registry_total or any(
        len(items) > registry_total for items in (executed_ids, problem_ids, inconclusive_ids)
    ):
        raise ValueError("Minefield coverage counts exceed the registry")
    return {
        "schema": "dspark-minefield-summary/v1",
        "commit": PINNED_COMMIT,
        "requests_made": int(raw.get("requests_made", 0)),
        "registry_total": registry_total,
        "executed": len(executed_ids),
        "problem": len(problem_ids),
        "inconclusive": len(inconclusive_ids),
        "unimplemented": unimplemented,
        "executed_trap_ids": sorted(executed_ids),
        "problem_trap_ids": sorted(problem_ids),
        "inconclusive_trap_ids": sorted(inconclusive_ids),
    }


def run_checked(
    argv: list[str], *, timeout: float, **kwargs
) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv, check=True, text=True, capture_output=True, timeout=timeout, **kwargs
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", default=PINNED_COMMIT)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env.dspark"))
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--setup-timeout", type=float, default=300.0)
    parser.add_argument("--doctor-timeout", type=float, default=600.0)
    args = parser.parse_args()
    if args.setup_timeout <= 0 or args.doctor_timeout <= 0:
        parser.error("timeouts must be positive")
    if args.commit != PINNED_COMMIT:
        parser.error(f"only pinned commit {PINNED_COMMIT} is permitted")
    env = parse_env(args.env_file)
    key_path = args.key_file or Path(env.get("VLLM_ORIGIN_KEY_FILE", ""))
    key = read_mode_0600(key_path)
    base_url = args.base_url or (
        f"http://{env.get('VLLM_PROXY_HOST', '172.30.0.1')}:"
        f"{env.get('VLLM_PROXY_PORT', '8888')}/v1"
    )
    model = args.model or env.get("SERVED_MODEL_NAME", "deepseek-v4-flash-0731")

    with tempfile.TemporaryDirectory(prefix="dspark-minefield-") as temporary:
        root = Path(temporary)
        checkout = root / "checkout"
        venv = root / "venv"
        run_checked(["git", "init", "--quiet", str(checkout)], timeout=args.setup_timeout)
        run_checked(["git", "-C", str(checkout), "remote", "add", "origin", REPOSITORY], timeout=args.setup_timeout)
        run_checked(["git", "-C", str(checkout), "fetch", "--quiet", "--depth", "1", "origin", PINNED_COMMIT], timeout=args.setup_timeout)
        run_checked(["git", "-C", str(checkout), "checkout", "--quiet", "--detach", "FETCH_HEAD"], timeout=args.setup_timeout)
        observed = run_checked(["git", "-C", str(checkout), "rev-parse", "HEAD"], timeout=args.setup_timeout).stdout.strip()
        if observed != PINNED_COMMIT:
            raise RuntimeError("isolated checkout did not resolve to the pinned commit")
        run_checked([sys.executable, "-m", "venv", str(venv)], timeout=args.setup_timeout)
        run_checked([str(venv / "bin/python"), "-m", "pip", "install", "--quiet", "--no-deps", str(checkout)], timeout=args.setup_timeout)
        raw_path = root / "doctor-raw.json"
        # The only secret transfer is this private stdin pipe. The child imports
        # the installed pinned doctor and adds --api-key only to in-memory
        # sys.argv; the OS process argv and shell history never contain the key.
        completed = subprocess.run(
            [str(venv / "bin/python"), "-c", HELPER, base_url, model, str(raw_path)],
            input=key, text=True, capture_output=True, check=False,
            timeout=args.doctor_timeout,
        )
        if completed.returncode != 0:
            raise RuntimeError("pinned Minefield quick doctor failed")
        summary = summarize(json.loads(raw_path.read_text(encoding="utf-8")))

    args.json.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        fd = os.open(args.json, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise SystemExit("refusing to overwrite Minefield evidence") from error
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: summary[key] for key in ("executed", "problem", "inconclusive", "unimplemented")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.TimeoutExpired as error:
        phase = Path(error.cmd[0]).name if error.cmd else "subprocess"
        raise SystemExit(f"pinned Minefield {phase} phase timed out") from error
