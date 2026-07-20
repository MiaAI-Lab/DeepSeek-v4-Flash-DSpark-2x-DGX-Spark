#!/usr/bin/env python3
"""Live fail-closed semantic, tool, NVFP4-KV, and native-backend gate."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 300) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"request failed for {url}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"non-object response from {url}")
    return value


def docker_logs(container: str, host: str | None) -> str:
    command = ["docker", "logs", "--tail", "10000", container]
    if host:
        command = ["ssh", host, *command]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    if completed.returncode:
        raise RuntimeError(f"docker logs failed on {host or 'local'}: {completed.stderr.strip()}")
    return completed.stdout + completed.stderr


def chat_payload(model: str, prompt: str, max_tokens: int = 64) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"thinking": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888")
    parser.add_argument("--model", default="deepseek-v4-flash-dspark")
    parser.add_argument("--container", default="dspark_vllm")
    parser.add_argument("--worker-host")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    errors: list[str] = []
    evidence: dict[str, Any] = {}

    try:
        models = request_json(f"{base}/v1/models")
        entries = models.get("data") if isinstance(models.get("data"), list) else []
        ids = [entry.get("id") for entry in entries if isinstance(entry, dict)]
        evidence["models"] = {"ids": ids, "entries": entries}
        if args.model not in ids:
            errors.append(f"served model {args.model!r} missing from /v1/models")
    except Exception as exc:
        evidence["models"] = None
        errors.append(str(exc))

    try:
        semantic = request_json(
            f"{base}/v1/chat/completions",
            chat_payload(args.model, "Compute 7 multiplied by 19. Reply with only the number."),
        )
        content = str(semantic["choices"][0]["message"].get("content") or "").strip()
        evidence["semantic"] = {"content": content, "pass": content == "133"}
        if content != "133":
            errors.append(f"semantic canary expected 133, got {content!r}")
    except Exception as exc:
        evidence["semantic"] = None
        errors.append(f"semantic canary failed: {exc}")

    try:
        tool_payload = chat_payload(args.model, "Use get_weather for Paris.")
        tool_payload["tools"] = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Return weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]
        tool_payload["tool_choice"] = {"type": "function", "function": {"name": "get_weather"}}
        tool = request_json(f"{base}/v1/chat/completions", tool_payload)
        calls = tool["choices"][0]["message"].get("tool_calls") or []
        names = [call.get("function", {}).get("name") for call in calls if isinstance(call, dict)]
        evidence["tool_call"] = {"names": names, "pass": "get_weather" in names}
        if "get_weather" not in names:
            errors.append(f"tool canary did not emit get_weather: {names}")
    except Exception as exc:
        evidence["tool_call"] = None
        errors.append(f"tool canary failed: {exc}")

    needle = "R0B0TLAB-DSVF-NVFP4-7319"
    filler = "A deterministic context sentence for retrieval validation. " * 900
    prompt = f"{filler}\nSecret code: {needle}\n{filler}\nReply with the secret code only."
    try:
        retrieval = request_json(
            f"{base}/v1/chat/completions", chat_payload(args.model, prompt, max_tokens=48), timeout=600
        )
        content = str(retrieval["choices"][0]["message"].get("content") or "").strip()
        usage = retrieval.get("usage")
        evidence["retrieval"] = {"content": content, "usage": usage, "pass": needle in content}
        if needle not in content:
            errors.append("retrieval canary did not return the exact code")
    except Exception as exc:
        evidence["retrieval"] = None
        errors.append(f"retrieval canary failed: {exc}")

    try:
        head_logs = docker_logs(args.container, None)
        worker_logs = docker_logs(args.container, args.worker_host) if args.worker_host else ""
        logs = head_logs + "\n" + worker_logs
        required = {
            "dspark": r"DSpark",
            "nvfp4_kv": r"nvfp4_ds_mla",
            "native_moe": r"flashinfer_b12x|B12X",
            "rdma": r"NET/IB|Using network IB",
        }
        forbidden = {
            "marlin": r"Using[^\n]*Marlin|moe_backend[^\n]*marlin|MARLIN",
            "emulation": r"moe_backend[^\n]*emulation|Using[^\n]*emulation",
            "fallback": r"fall(?:ing)? back to",
        }
        required_hits = {name: bool(re.search(pattern, logs, re.I)) for name, pattern in required.items()}
        forbidden_hits = {name: bool(re.search(pattern, logs, re.I)) for name, pattern in forbidden.items()}
        evidence["runtime_logs"] = {
            "required": required_hits,
            "forbidden": forbidden_hits,
            "head_log_bytes": len(head_logs),
            "worker_log_bytes": len(worker_logs),
        }
        for name, present in required_hits.items():
            if not present:
                errors.append(f"required runtime marker missing: {name}")
        for name, present in forbidden_hits.items():
            if present:
                errors.append(f"forbidden active runtime marker present: {name}")
    except Exception as exc:
        evidence["runtime_logs"] = None
        errors.append(f"runtime log gate failed: {exc}")

    result = {"schema_version": 1, "status": "PASS" if not errors else "FAIL", "evidence": evidence, "errors": errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "errors": errors}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
