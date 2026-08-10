#!/usr/bin/env python3
"""Six-sequence MTP scheduler characterization with structured gate evidence."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import threading
import time
import urllib.request



def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
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
        url, data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def tokenize(base_url: str, key: str, model: str, content: str, timeout: float) -> int:
    result = request_json(
        base_url.removesuffix("/v1") + "/tokenize", key,
        {"model": model, "messages": [{"role": "user", "content": content}],
         "chat_template_kwargs": {"reasoning_effort": "none", "drop_thinking": False},
         "add_generation_prompt": True}, timeout,
    )
    count = result.get("count")
    if not isinstance(count, int) or count <= 0:
        raise RuntimeError("tokenize response did not contain a positive count")
    return count


def build_prompt(base_url: str, key: str, model: str, target: int, marker: str, timeout: float) -> tuple[str, int]:
    unit = " scheduler-context-datum"
    repeats = max(1, target // 4)
    content = unit * repeats + f"\nReply with exactly {marker}."
    observed = 0
    for _ in range(6):
        observed = tokenize(base_url, key, model, content, timeout)
        if target - 8 <= observed <= target + 32:
            return content, observed
        repeats = max(1, int(repeats * target / observed))
        content = unit * repeats + f"\nReply with exactly {marker}."
    if observed < target:
        raise RuntimeError(f"could not construct scheduler prompt: {observed} < {target}")
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
    output = _run_node(worker, ["cat", "/proc/meminfo"])
    total = re.search(r"^MemTotal:\s+(\d+)\s+kB$", output, re.MULTILINE)
    available = re.search(r"^MemAvailable:\s+(\d+)\s+kB$", output, re.MULTILINE)
    if not total or not available:
        raise RuntimeError(f"could not parse {node} memory observation")
    total_bytes = int(total.group(1)) * 1024
    available_bytes = int(available.group(1)) * 1024
    return {"node": node, "observed_unix_s": time.time(),
            "mem_available_bytes": available_bytes,
            "memory_used_bytes": total_bytes - available_bytes}


def memory_pair(worker: str) -> list[dict]:
    with ThreadPoolExecutor(max_workers=2) as executor:
        head = executor.submit(memory_sample, "head", None)
        remote = executor.submit(memory_sample, "worker", worker)
        return [head.result(), remote.result()]


def restart_count(worker: str | None) -> int:
    output = _run_node(worker, ["sh", "-c", "ids=$(docker ps -q --filter name=vllm-dspark); [ -n \"$ids\" ] && docker inspect -f '{{.RestartCount}}' $ids || printf '0\n'"])
    return sum(int(value) for value in re.findall(r"(?m)^\d+$", output))


def recent_runtime_has(worker: str | None, since: int, pattern: str) -> bool:
    output = _run_node(worker, ["sh", "-c", f"ids=$(docker ps -q --filter name=vllm-dspark); [ -z \"$ids\" ] || docker logs --since {since} $ids 2>&1"])
    return re.search(pattern, output, re.IGNORECASE) is not None


def percentile95(values: list[float]) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def evaluate_scheduler_gate(*, requests: list[dict], concurrency: int, mtp: int,
                            max_num_batched_tokens: int, baseline: dict | None,
                            restart_delta: int, oom_detected: bool,
                            eager_fallback_detected: bool) -> dict:
    p95_ttft = percentile95([float(item["ttft_s"]) for item in requests])
    p95_decode = percentile95([float(item["decode_latency_s"]) for item in requests])
    baseline = baseline or {}
    base_ttft = baseline.get("p95_ttft_s")
    base_decode = baseline.get("p95_decode_latency_s")
    baseline_available = isinstance(base_ttft, (int, float)) and isinstance(base_decode, (int, float)) and base_ttft > 0 and base_decode > 0
    ttft_ok = bool(baseline_available) and p95_ttft <= base_ttft * 1.40
    decode_ok = bool(baseline_available) and p95_decode <= base_decode * 1.25
    checks = {
        "six_sequence_shape": concurrency == 6 and len(requests) == 6,
        "mtp_5": mtp == 5,
        "target_scheduler_tokens": max_num_batched_tokens == 8216,
        "correctness": len(requests) == concurrency and all(item.get("correct") for item in requests),
        "no_restart_or_oom": restart_delta == 0 and not oom_detected,
        "no_eager_fallback": not eager_fallback_detected,
        "baseline_available": bool(baseline_available),
        "ttft_regression_within_40_percent": ttft_ok,
        "decode_regression_within_25_percent": decode_ok,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "p95_ttft_s": p95_ttft,
            "p95_decode_latency_s": p95_decode,
            "baseline_p95_ttft_s": base_ttft,
            "baseline_p95_decode_latency_s": base_decode,
            "ttft_regression_ratio": p95_ttft / base_ttft if baseline_available else None,
            "decode_latency_regression_ratio": p95_decode / base_decode if baseline_available else None,
        },
    }


def stream_one(base_url: str, key: str, model: str, index: int, prompt: str,
               prompt_tokens: int, timeout: float, start: threading.Event) -> dict:
    marker = f"SCHEDULER_OK_{index}"
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 64, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"reasoning_effort": "none", "drop_thinking": False}}
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body, separators=(",", ":")).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    start.wait(timeout=30)
    began = time.perf_counter()
    first = None
    output_parts: list[str] = []
    usage: dict = {}
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            piece = (delta.get("reasoning") or "") + (delta.get("reasoning_content") or "") + (delta.get("content") or "")
            if piece:
                if first is None:
                    first = time.perf_counter()
                output_parts.append(piece)
    finished = time.perf_counter()
    first = first or finished
    output_tokens = int(usage.get("completion_tokens") or max(1, len("".join(output_parts).split())))
    return {
        "index": index,
        "correct": marker in "".join(output_parts),
        "prompt_tokens": int(usage.get("prompt_tokens") or prompt_tokens),
        "output_tokens": output_tokens,
        "ttft_s": first - began,
        "elapsed_s": finished - began,
        "decode_latency_s": (finished - first) / max(1, output_tokens - 1),
    }


def load_baseline(
    path: Path, *, model: str, concurrency: int, mtp: int, target_prompt_tokens: int
) -> tuple[dict, str]:
    encoded = path.read_bytes()
    payload = json.loads(encoded)
    expected_configuration = {
        "concurrency": concurrency,
        "mtp": mtp,
        "target_prompt_tokens": target_prompt_tokens,
        "max_num_batched_tokens": 8192,
    }
    requests = payload.get("requests")
    gate = payload.get("gate")
    if (
        payload.get("schema") != "dspark-scheduler-baseline/v1"
        or payload.get("baseline_capture") is not True
        or payload.get("model") != model
        or payload.get("configuration") != expected_configuration
        or not isinstance(requests, list)
        or len(requests) != concurrency
        or not all(isinstance(item, dict) and item.get("correct") is True for item in requests)
        or not isinstance(gate, dict)
        or gate.get("passed") is not True
    ):
        raise ValueError("scheduler baseline provenance/configuration is invalid")
    metrics = gate.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("scheduler baseline metrics are missing")
    for field in ("p95_ttft_s", "p95_decode_latency_s"):
        value = metrics.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"scheduler baseline metric is invalid: {field}")
    return metrics, hashlib.sha256(encoded).hexdigest()


def write_report(path: Path, report: dict) -> None:
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)
    print(encoded, end="")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--mtp", type=int, default=5)
    parser.add_argument("--target-prompt-tokens", type=int, default=8192)
    parser.add_argument("--env-file", type=Path, default=Path(".env.dspark"))
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--baseline", type=Path, default=Path("artifacts/health-rollout/scheduler-baseline.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/health-rollout/scheduler-current.json"))
    parser.add_argument(
        "--capture-baseline", action="store_true",
        help="capture authoritative pre-change latency evidence at --baseline",
    )
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()
    if args.capture_baseline and (args.baseline.exists() or args.baseline.is_symlink()):
        raise SystemExit(f"refusing to overwrite baseline evidence: {args.baseline}")
    if args.concurrency != 6 or args.mtp != 5:
        parser.error("acceptance gate requires --concurrency 6 --mtp 5")
    values = parse_env(args.env_file)
    configured_mtp = int(values.get("MTP_NUM_TOKENS", "5"))
    max_batched = int(values.get("MAX_NUM_BATCHED_TOKENS", "8216"))
    if configured_mtp != args.mtp:
        raise SystemExit("configured MTP_NUM_TOKENS does not match --mtp")
    key = read_key_file(args.key_file or Path(values.get("VLLM_ORIGIN_KEY_FILE", "")))
    base_url = args.base_url or f"http://{values.get('VLLM_PROXY_HOST', '172.30.0.1')}:{values.get('VLLM_PROXY_PORT', '8888')}/v1"
    model = args.model or values.get("SERVED_MODEL_NAME", "deepseek-v4-flash-0731")
    worker = values.get("WORKER_HOST") or None
    if not worker:
        raise SystemExit("WORKER_HOST must be set in the env file")
    if not args.capture_baseline and not args.baseline.is_file():
        raise SystemExit(f"baseline evidence is required: {args.baseline}")
    if args.capture_baseline:
        baseline = None
        baseline_sha256 = None
    else:
        baseline, baseline_sha256 = load_baseline(
            args.baseline,
            model=model,
            concurrency=args.concurrency,
            mtp=args.mtp,
            target_prompt_tokens=args.target_prompt_tokens,
        )

    prompts = []
    for index in range(args.concurrency):
        marker = f"SCHEDULER_OK_{index}"
        prompts.append(build_prompt(base_url, key, model, args.target_prompt_tokens, marker, args.timeout))
    started_unix = int(time.time())
    before_restarts = restart_count(None) + restart_count(worker)
    samples: list[dict] = memory_pair(worker)
    stop = threading.Event()
    observation_error: list[str] = []

    def sample_loop():
        while not stop.wait(args.sample_interval):
            try:
                samples.extend(memory_pair(worker))
            except Exception as error:
                observation_error.append(str(error))
                return

    sampler = threading.Thread(target=sample_loop, daemon=True)
    sampler.start()

    def run_batch() -> list[dict]:
        batch_start = threading.Event()
        batch: list[dict] = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(
                    stream_one, base_url, key, model, index, prompt, count,
                    args.timeout, batch_start,
                )
                for index, (prompt, count) in enumerate(prompts)
            ]
            batch_start.set()
            for future in as_completed(futures):
                batch.append(future.result())
        batch.sort(key=lambda item: item["index"])
        return batch

    # Characterize steady-state scheduling rather than one-time JIT/prefix-cache
    # population. The warmup is still inside restart/OOM/memory observation.
    warmup_results = run_batch()
    results = run_batch()
    stop.set()
    sampler.join(timeout=args.sample_interval + 20)
    samples.extend(memory_pair(worker))
    after_restarts = restart_count(None) + restart_count(worker)
    oom = recent_runtime_has(None, started_unix, r"out of memory|\boom\b") or recent_runtime_has(worker, started_unix, r"out of memory|\boom\b")
    eager = recent_runtime_has(None, started_unix, r"fall(?:ing)? back to eager|eager fallback|running in eager mode") or recent_runtime_has(worker, started_unix, r"fall(?:ing)? back to eager|eager fallback|running in eager mode")
    gate = evaluate_scheduler_gate(
        requests=results, concurrency=args.concurrency, mtp=args.mtp,
        max_num_batched_tokens=max_batched,
        baseline={
            "p95_ttft_s": percentile95([float(item["ttft_s"]) for item in results]),
            "p95_decode_latency_s": percentile95(
                [float(item["decode_latency_s"]) for item in results]
            ),
        } if args.capture_baseline else baseline,
        restart_delta=after_restarts - before_restarts, oom_detected=oom,
        eager_fallback_detected=eager,
    )
    gate["checks"]["warmup_correctness"] = (
        len(warmup_results) == args.concurrency
        and all(item.get("correct") for item in warmup_results)
    )
    gate["passed"] = all(gate["checks"].values())
    if args.capture_baseline:
        # A baseline is accepted on correctness/observability, not by comparing
        # the lane to itself. Current-lane runs consume these measured p95s.
        gate["checks"].pop("ttft_regression_within_40_percent", None)
        gate["checks"].pop("decode_regression_within_25_percent", None)
        gate["checks"].pop("target_scheduler_tokens", None)
        gate["checks"]["baseline_capture"] = True
        gate["passed"] = all(gate["checks"].values())
    if observation_error:
        gate["checks"]["observation_complete"] = False
        gate["passed"] = False
    peak = {node: max(sample["memory_used_bytes"] for sample in samples if sample["node"] == node)
            for node in ("head", "worker")}
    report = {
        "schema": (
            "dspark-scheduler-baseline/v1" if args.capture_baseline
            else "dspark-scheduler-gate/v1"
        ),
        "model": model,
        "baseline_capture": args.capture_baseline,
        "baseline_sha256": baseline_sha256,
        "warmup_requests": len(warmup_results),
        "configuration": {"concurrency": args.concurrency, "mtp": args.mtp,
                          "target_prompt_tokens": args.target_prompt_tokens,
                          "max_num_batched_tokens": max_batched},
        "requests": results, "peak_memory_used_bytes": peak,
        "restart_delta": after_restarts - before_restarts,
        "oom_detected": oom, "eager_fallback_detected": eager, "gate": gate,
    }
    destination = args.baseline if args.capture_baseline else args.output
    write_report(destination, report)
    destination.chmod(0o600)
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
