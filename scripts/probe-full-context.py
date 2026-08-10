#!/usr/bin/env python3
"""Bounded near-maximum-context gate for the authenticated DSpark origin.

The credential is read into process memory from a mode-0600 regular file. The
report contains only numeric/operator evidence; prompt and response bodies are
never written or printed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import threading
import time
import urllib.request

GIB = 1024 ** 3
MIN_MEM_AVAILABLE_BYTES = 8 * GIB
MAX_MEMORY_PSI_FULL_AVG10 = 5.0
MAX_PRESSURED_SAMPLE_RATIO = 0.10


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = os.path.expandvars(value)
    return values


def read_key_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("key file must be a regular file, not a symlink")
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError("key file must have mode 0600")
    key = path.read_text().strip()
    if not key:
        raise ValueError("key file is empty")
    return key


def request_json(url: str, key: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def tokenize(base_url: str, key: str, model: str, content: str, timeout: float) -> int:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "chat_template_kwargs": {"reasoning_effort": "none", "drop_thinking": False},
        "add_generation_prompt": True,
    }
    result = request_json(base_url.removesuffix("/v1") + "/tokenize", key, payload, timeout)
    count = result.get("count")
    if not isinstance(count, int) or count <= 0:
        raise RuntimeError("tokenize response did not contain a positive count")
    return count


def build_near_max_prompt(base_url: str, key: str, model: str, target: int, timeout: float) -> tuple[str, int]:
    unit = " dspark-capacity-datum"
    repeats = max(1, target // 4)
    content = unit * repeats
    observed = 0
    for _ in range(8):
        observed = tokenize(base_url, key, model, content, timeout)
        if target - 64 <= observed <= target + 64:
            return content, observed
        repeats = max(1, int(repeats * target / observed))
        content = unit * repeats
    if observed < target:
        raise RuntimeError(f"could not construct near-max prompt: {observed} < {target}")
    return content, observed


def _run_node(worker: str | None, command: list[str], timeout: float = 20) -> str:
    argv = command if worker is None else [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", worker,
        " ".join(shlex.quote(part) for part in command),
    ]
    result = subprocess.run(argv, text=True, capture_output=True, timeout=timeout, check=False)
    if result.returncode:
        node = "head" if worker is None else "worker"
        raise RuntimeError(f"{node} observation command failed ({command[0]})")
    return result.stdout


def memory_sample(node: str, worker: str | None) -> dict:
    output = _run_node(worker, ["sh", "-c", "cat /proc/meminfo; printf '\n---PSI---\n'; cat /proc/pressure/memory"])
    available = re.search(r"^MemAvailable:\s+(\d+)\s+kB$", output, re.MULTILINE)
    pressure = re.search(r"^full\s+avg10=([0-9.]+)", output, re.MULTILINE)
    if not available or not pressure:
        raise RuntimeError(f"could not parse {node} memory observation")
    return {
        "node": node,
        "observed_unix_s": time.time(),
        "mem_available_bytes": int(available.group(1)) * 1024,
        "pressure_full_avg10": float(pressure.group(1)),
    }


def memory_pair(worker: str) -> list[dict]:
    with ThreadPoolExecutor(max_workers=2) as executor:
        head = executor.submit(memory_sample, "head", None)
        remote = executor.submit(memory_sample, "worker", worker)
        return [head.result(), remote.result()]


def restart_count(worker: str | None) -> int:
    output = _run_node(
        worker,
        ["sh", "-c", "ids=$(docker ps -q --filter name=vllm-dspark); [ -n \"$ids\" ] && docker inspect -f '{{.RestartCount}}' $ids || printf '0\n'"],
    )
    values = [int(value) for value in re.findall(r"(?m)^\d+$", output)]
    return sum(values)


def recent_runtime_has(worker: str | None, since: int, pattern: str) -> bool:
    output = _run_node(
        worker,
        ["sh", "-c", f"ids=$(docker ps -q --filter name=vllm-dspark); [ -z \"$ids\" ] || docker logs --since {since} $ids 2>&1"],
    )
    return re.search(pattern, output, re.IGNORECASE) is not None


def positive_usage_token_count(usage: dict, field: str) -> int:
    value = usage.get(field)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def evaluate_full_context_gate(*, memory_samples: list[dict], target_prompt_tokens: int,
                               observed_prompt_tokens: int, completion_tokens: int,
                               restart_delta: int, oom_detected: bool) -> dict:
    nodes = {sample["node"] for sample in memory_samples}
    psi_values = [sample["pressure_full_avg10"] for sample in memory_samples]
    max_psi = max(psi_values, default=float("inf"))
    pressured_ratio = (
        sum(value > 0.0 for value in psi_values) / len(psi_values)
        if psi_values else 1.0
    )
    checks = {
        "near_max_prefill": observed_prompt_tokens >= target_prompt_tokens - 64,
        "one_token_decode": completion_tokens >= 1,
        "both_nodes_observed": {"head", "worker"}.issubset(nodes),
        "memory_headroom": bool(memory_samples) and all(
            sample["mem_available_bytes"] >= MIN_MEM_AVAILABLE_BYTES for sample in memory_samples
        ),
        "memory_pressure": (
            bool(memory_samples)
            and max_psi <= MAX_MEMORY_PSI_FULL_AVG10
            and pressured_ratio <= MAX_PRESSURED_SAMPLE_RATIO
        ),
        "no_restart_or_oom": restart_delta == 0 and not oom_detected,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "max_memory_psi_full_avg10": round(max_psi, 6),
            "pressured_sample_ratio": round(pressured_ratio, 6),
        },
    }


def write_report(path: Path | None, report: dict) -> None:
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded)
    print(encoded, end="")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("near-max",), default="near-max")
    parser.add_argument("--max-output-tokens", type=int, default=1)
    parser.add_argument("--target-prompt-tokens", type=int)
    parser.add_argument("--env-file", type=Path, default=Path(".env.dspark"))
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--sample-interval", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/health-rollout/full-context.json"))
    args = parser.parse_args()
    if args.max_output_tokens != 1:
        parser.error("near-max gate requires --max-output-tokens 1")
    values = parse_env(args.env_file)
    key_path = args.key_file or Path(values.get("VLLM_ORIGIN_KEY_FILE", ""))
    key = read_key_file(key_path)
    base_url = args.base_url or f"http://{values.get('VLLM_PROXY_HOST', '172.30.0.1')}:{values.get('VLLM_PROXY_PORT', '8888')}/v1"
    model = args.model or values.get("SERVED_MODEL_NAME", "deepseek-v4-flash-0731")
    max_model_len = int(values.get("MAX_MODEL_LEN", "1048576"))
    target = args.target_prompt_tokens or (max_model_len - 256)
    worker = values.get("WORKER_HOST") or None
    if not worker:
        raise SystemExit("WORKER_HOST must be set in the env file")

    started_unix = int(time.time())
    before_restarts = restart_count(None) + restart_count(worker)
    samples: list[dict] = memory_pair(worker)
    prompt, constructed_tokens = build_near_max_prompt(base_url, key, model, target, args.timeout)
    stop = threading.Event()
    observation_error: list[str] = []

    def sample_loop():
        while not stop.wait(args.sample_interval):
            try:
                samples.extend(memory_pair(worker))
            except Exception as error:  # evidence must fail closed, without request bodies
                observation_error.append(str(error))
                return

    sampler = threading.Thread(target=sample_loop, daemon=True)
    sampler.start()
    request_started = time.perf_counter()
    response = request_json(
        base_url.rstrip("/") + "/chat/completions",
        key,
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 1,
            "temperature": 0,
            "chat_template_kwargs": {"reasoning_effort": "none", "drop_thinking": False},
        },
        args.timeout,
    )
    elapsed = time.perf_counter() - request_started
    stop.set()
    sampler.join(timeout=args.sample_interval + 20)
    samples.extend(memory_pair(worker))
    usage = response.get("usage") or {}
    completion_tokens = positive_usage_token_count(usage, "completion_tokens")
    observed_prompt_tokens = positive_usage_token_count(usage, "prompt_tokens")
    after_restarts = restart_count(None) + restart_count(worker)
    oom_detected = recent_runtime_has(None, started_unix, r"out of memory|\boom\b") or recent_runtime_has(worker, started_unix, r"out of memory|\boom\b")
    gate = evaluate_full_context_gate(
        memory_samples=samples,
        target_prompt_tokens=target,
        observed_prompt_tokens=observed_prompt_tokens,
        completion_tokens=completion_tokens,
        restart_delta=after_restarts - before_restarts,
        oom_detected=oom_detected,
    )
    if observation_error:
        gate["checks"]["observation_complete"] = False
        gate["passed"] = False
    report = {
        "schema": "dspark-full-context-gate/v1",
        "profile": args.profile,
        "model": model,
        "target_prompt_tokens": target,
        "constructed_prompt_tokens": constructed_tokens,
        "observed_prompt_tokens": observed_prompt_tokens,
        "completion_tokens": completion_tokens,
        "elapsed_s": elapsed,
        "restart_delta": after_restarts - before_restarts,
        "oom_detected": oom_detected,
        "memory_samples": samples,
        "gate": gate,
    }
    write_report(args.output, report)
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
