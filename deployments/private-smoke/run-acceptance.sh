#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
FIXTURE="$SCRIPT_DIR/fixtures/suite.json"
SCHEMA="$SCRIPT_DIR/schemas/acceptance.schema.json"
SANITIZER="$SCRIPT_DIR/scripts/sanitize-evidence.py"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env.dspark}"
LITELLM_ENV_FILE="${LITELLM_ENV_FILE:-$SCRIPT_DIR/litellm/.env}"
ACTIVE_GATEWAY_SNAPSHOT="${ACTIVE_GATEWAY_SNAPSHOT:-$ROOT_DIR/artifacts/active-gateway-snapshot.json}"
MODE=""
RUN_DIR=""
HERMES_RESULTS=""
FAILURE_STAGE="startup"
LIVE_STARTED=0

usage() {
  echo "Usage: $0 --validate-fixtures" >&2
  echo "       $0 --live [--run-dir PATH] [--hermes-results DIR]" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --validate-fixtures) MODE="fixtures"; shift ;;
    --live) MODE="live"; shift ;;
    --run-dir) RUN_DIR="${2:?missing --run-dir value}"; shift 2 ;;
    --hermes-results) HERMES_RESULTS="${2:?missing --hermes-results value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
[ -n "$MODE" ] || { usage; exit 2; }

validate_fixtures() {
  python3 - "$FIXTURE" <<'PY'
import json
from pathlib import Path
import sys
fixture = json.loads(Path(sys.argv[1]).read_text())
assert fixture["schema_version"] == 1
assert fixture["performance"] == {
    "cold_runs": 1, "warmups": 3, "samples": 20, "concurrency": 1,
    "prompt_tokens_min": 180, "prompt_tokens_max": 360, "max_tokens": 512,
    "temperature": 0.6, "top_p": 0.95,
    "median_decode_tokens_per_second_min": 50.0,
    "p95_ttft_seconds_max": 5.0,
}
assert fixture["soak"]["duration_seconds"] == 1800
assert fixture["soak"]["sample_interval_seconds"] == 5
assert fixture["soak"]["concurrency"] == 1
assert fixture["synthetic_expenses"]["expected_grand_total_centavos"] == 374025
PY
  python3 - <<'PY' | "$SANITIZER" --schema "$SCHEMA" >/dev/null
import hashlib, json
h = "a" * 64
performance = {"median_decode_tokens_per_second": 55.0, "p95_ttft_seconds": 2.0, "request_count": 24, "origin_completion_delta": 24, "accepted": True}
gates = {name: True for name in ("fabric", "artifacts", "qwen_stopped", "direct", "gateway", "hermes", "performance", "soak", "isolation", "sanitization", "public_gateway_unchanged")}
pins = {"model_revision": "b" * 40, "runtime_image_digest": "sha256:" + h, "repo_commit": "c" * 40, "config_sha256": h}
pin_hash = hashlib.sha256(json.dumps(pins, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
chain = []; previous = "0" * 64
for i in range(7):
  entry = hashlib.sha256(f"{previous}:gate-{i}:{h}:{pin_hash}".encode()).hexdigest()
  chain.append({"name": f"gate-{i}", "artifact_path": f"gate-{i}.json", "artifact_sha256": h, "previous_sha256": previous, "entry_sha256": entry})
  previous = entry
print(json.dumps({
  "schema_version": 1, "run_id": "fixture-run", "created_at": "2026-08-02T12:00:00Z",
  "accepted": True, "manifest_sha256": h, "fixture_sha256": h, "pin_set_sha256": pin_hash,
  "chain_head_sha256": previous, "pins": pins,
  "gates": gates,
  "functional_runs": [{"artifact_sha256": h, "accepted": True, "api_calls": 3, "gateway_attested": True}, {"artifact_sha256": h, "accepted": True, "api_calls": 3, "gateway_attested": True}],
  "performance": {"direct": performance, "litellm": performance, "median_decode_overhead_ratio": 1.0, "p95_ttft_overhead_seconds": 0.1},
  "soak": {"accepted": True, "duration_seconds": 1800, "sample_interval_seconds": 5, "request_count": 10, "origin_completion_delta": 10, "gateway_attempt_delta": 10, "failed_requests": 0, "sample_error_count": 0, "max_idle_gap_seconds": 0.1, "node_samples": 360, "min_head_mem_available_gib": 5.0, "min_worker_mem_available_gib": 5.0, "max_requests_running": 1.0, "max_requests_waiting": 0.0, "preemption_delta": 0.0, "max_rank_restarts": 0, "max_node_sample_gap_seconds": 5.1, "kv_cache_usage_peak": 0.5, "speculative_acceptance_mean": None},
  "evidence_chain": chain, "purge_eligible": True,
}))
PY
  for planted in \
    '{"value":"sk-abcdefghijklmnopqrstuvwxyz"}' \
    '{"value":"100.64.10.20"}' \
    '{"value":"/home/plexiz/private"}'; do
    if printf '%s' "$planted" | "$SANITIZER" --scan-only >/dev/null 2>&1; then
      echo "Sanitizer accepted a planted canary." >&2
      return 1
    fi
  done
  echo "Acceptance fixture, schema, and sanitizer canaries passed."
}

if [ "$MODE" = "fixtures" ]; then
  validate_fixtures
  exit 0
fi

[ -f "$ENV_FILE" ] || { echo "Missing $ENV_FILE" >&2; exit 1; }
[ -f "$LITELLM_ENV_FILE" ] || { echo "Missing $LITELLM_ENV_FILE" >&2; exit 1; }
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
# shellcheck disable=SC1090
source "$LITELLM_ENV_FILE"
set +a

if [ -z "$RUN_DIR" ]; then
  RUN_DIR="$(find "$ROOT_DIR/artifacts/acceptance" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort | tail -n 1)"
fi
[ -n "$RUN_DIR" ] && [ -d "$RUN_DIR" ] || { echo "Acceptance run directory is missing." >&2; exit 1; }
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
HERMES_RESULTS="${HERMES_RESULTS:-$RUN_DIR/hermes}"
QWEN_MANIFEST="$RUN_DIR/qwen-manifest.json"
DIRECT_INTERIM="$RUN_DIR/acceptance.json"
DIRECT_BENCHMARK="$RUN_DIR/benchmark-direct.json"
GATEWAY_BENCHMARK="$RUN_DIR/benchmark-litellm.json"
NODE_EVIDENCE="$RUN_DIR/node-evidence.json"
NODE_EVIDENCE_AFTER="$RUN_DIR/node-evidence-after-soak.json"
SOAK_EVIDENCE="$RUN_DIR/soak.json"
FINAL_REPORT="$RUN_DIR/accepted.json"
REJECTED_REPORT="$RUN_DIR/rejected.json"

write_rejection() {
  [ ! -e "$REJECTED_REPORT" ] || return 0
  python3 - "$FAILURE_STAGE" <<'PY' | "$SANITIZER" --scan-only --output "$REJECTED_REPORT" || true
from datetime import datetime, timezone
import json, sys
print(json.dumps({"schema_version": 1, "accepted": False, "failure_stage": sys.argv[1], "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}))
PY
  chmod 0600 "$REJECTED_REPORT" 2>/dev/null || true
}

cleanup_failed_acceptance() {
  local status=$?
  local cleaned=0
  trap - ERR
  write_rejection
  if [ "$LIVE_STARTED" -eq 1 ]; then
    if ENV_FILE="$ENV_FILE" LITELLM_ENV_FILE="$LITELLM_ENV_FILE" \
      "$SCRIPT_DIR/scripts/cleanup-acceptance.sh"; then
      cleaned=1
    fi
  fi
  if [ "$LIVE_STARTED" -eq 1 ] && [ "$cleaned" -ne 1 ]; then
    echo "Acceptance failed at $FAILURE_STAGE; WARNING: at least one DeepSeek rank may still be running. Qwen remains stopped." >&2
  else
    echo "Acceptance failed at $FAILURE_STAGE; DeepSeek and the private gateway were disabled. Qwen remains stopped." >&2
  fi
  exit "$status"
}
trap cleanup_failed_acceptance ERR
LIVE_STARTED=1

for required in "$QWEN_MANIFEST" "$DIRECT_INTERIM" "$DIRECT_BENCHMARK" "$NODE_EVIDENCE" "$FIXTURE"; do
  [ -f "$required" ] || { echo "Missing prerequisite evidence." >&2; false; }
done
[ ! -e "$FINAL_REPORT" ] || { echo "Refusing to overwrite accepted report." >&2; false; }

FAILURE_STAGE="preconditions"
git -C "$ROOT_DIR" diff --quiet
git -C "$ROOT_DIR" diff --cached --quiet
ENV_FILE="$ENV_FILE" "$ROOT_DIR/status-deepseek-v4-flash-dspark.sh" --expect running
[ "$(docker inspect -f '{{.State.Running}}' urbanplan-qwen 2>/dev/null)" = "false" ]
python3 "$ROOT_DIR/scripts/qwen_manifest.py" verify-live --manifest "$QWEN_MANIFEST" --max-age-hours 24
python3 - "$DIRECT_INTERIM" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report["accepted"] is False
assert report["gates"]["fabric"] and report["gates"]["artifacts"] and report["gates"]["direct"]
PY

spend_count() {
  docker exec dspark-private-litellm-postgres-1 psql -X -A -t \
    -U litellm_smoke -d litellm_smoke -c 'SELECT count(*) FROM "LiteLLM_SpendLogs";' | tr -d '[:space:]'
}

wait_for_spend_delta() {
  local before="$1" expected="$2" current=""
  for _ in $(seq 1 60); do
    current="$(spend_count)"
    [ "$((current - before))" -eq "$expected" ] && { printf '%s\n' "$current"; return 0; }
    sleep 1
  done
  echo "LiteLLM spend-log delta did not reach $expected." >&2
  return 1
}

FAILURE_STAGE="gateway"
"$SCRIPT_DIR/litellm/smoke.sh" --all-interfaces
gateway_count_before="$(spend_count)"
python3 "$SCRIPT_DIR/scripts/benchmark.py" --layer litellm --warmups 3 --samples 20 --concurrency 1 \
  --base-url "http://${HEAD_TAILSCALE_IP}:4001/v1" --key-file "$LITELLM_VIRTUAL_KEY_FILE" \
  --model deepseek-v4-flash-0731-smoke \
  --origin-metrics-base-url "http://${VLLM_PROXY_HOST:-172.30.0.1}:${VLLM_PROXY_PORT:-8888}/v1" \
  --origin-key-file "$VLLM_ORIGIN_KEY_FILE" --output "$GATEWAY_BENCHMARK"
wait_for_spend_delta "$gateway_count_before" 24 >/dev/null

FAILURE_STAGE="hermes"
mapfile -t hermes_files < <(find "$HERMES_RESULTS" -maxdepth 1 -type f -name 'hermes-smoke-*.json' -print | sort)
[ "${#hermes_files[@]}" -eq 2 ] || { echo "Hermes acceptance requires exactly two result files." >&2; false; }
python3 - "${hermes_files[@]}" <<'PY'
import json
import re
import subprocess
import sys

def spend_rows(request_id):
    if not re.fullmatch(r"hermes-smoke-[a-f0-9-]{8,64}", request_id):
        raise AssertionError("invalid Hermes request id")
    sql = (
        'SELECT row_to_json(t)::text FROM "LiteLLM_SpendLogs" AS t '
        f"WHERE to_jsonb(t)::text LIKE '%{request_id}%';"
    )
    output = subprocess.check_output([
        "docker", "exec", "dspark-private-litellm-postgres-1",
        "psql", "-X", "-A", "-t", "-U", "litellm_smoke", "-d", "litellm_smoke",
        "-c", sql,
    ], text=True)
    return [json.loads(line) for line in output.splitlines() if line.strip()]

for path in sys.argv[1:]:
    item = json.load(open(path))
    assert item["accepted"] is True
    assert item["model"] == "deepseek-v4-flash-0731-smoke"
    assert item["provider"] == "custom:deepseek-smoke"
    assert item["shared_state"]["unchanged"] is True
    assert all(item["negative_checks"].values())
    assert item["request_ids"] == [item["run_id"]]
    rows = spend_rows(item["run_id"])
    assert len(rows) == item["usage"]["api_calls"]
    for row in rows:
        encoded = json.dumps(row, sort_keys=True)
        assert item["run_id"] in encoded
        assert "deepseek-v4-flash-0731-smoke" in encoded
        assert "hermes-deepseek-smoke" in encoded
assert len({item["suite_pin_sha256"] for item in map(lambda p: json.load(open(p)), sys.argv[1:])}) == 1
PY

FAILURE_STAGE="soak"
SOAK_DURATION_SECONDS="${SOAK_DURATION_SECONDS:-1800}"
SOAK_SAMPLE_INTERVAL_SECONDS="${SOAK_SAMPLE_INTERVAL_SECONDS:-5}"
[ "$SOAK_DURATION_SECONDS" -eq 1800 ] && [ "$SOAK_SAMPLE_INTERVAL_SECONDS" -eq 5 ] || {
  echo "Live acceptance requires the full 1800-second soak and 5-second samples." >&2
  false
}

python3 - "$SCRIPT_DIR/scripts/benchmark.py" "$WORKER_HOST" \
  "http://${HEAD_TAILSCALE_IP}:4001/v1" "$LITELLM_VIRTUAL_KEY_FILE" \
  "http://${VLLM_PROXY_HOST:-172.30.0.1}:${VLLM_PROXY_PORT:-8888}/v1" "$VLLM_ORIGIN_KEY_FILE" \
  "$SOAK_DURATION_SECONDS" "$SOAK_SAMPLE_INTERVAL_SECONDS" "$SOAK_EVIDENCE" <<'PY'
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

benchmark_path, worker, gateway_url, gateway_key_path, origin_url, origin_key_path, duration_raw, interval_raw, output = sys.argv[1:]
duration, interval = int(duration_raw), int(interval_raw)
spec = importlib.util.spec_from_file_location("dspark_benchmark", benchmark_path)
bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
gateway_key = Path(gateway_key_path).read_text().strip()
origin_key = Path(origin_key_path).read_text().strip()
stop = threading.Event()
node_rows, metric_rows, sample_errors = [], [], []

def prometheus():
    with bench.request(origin_url.removesuffix("/v1"), origin_key, "/metrics", timeout=30) as response:
        return response.read().decode()

def metric(text, name, default=0.0):
    values = []
    for line in text.splitlines():
        if line.startswith(name) and not line.startswith("#"):
            try: values.append(float(line.rsplit(" ", 1)[1]))
            except ValueError: pass
    return sum(values) if values else default

def spend_count():
    command = ["docker", "exec", "dspark-private-litellm-postgres-1", "psql", "-X", "-A", "-t", "-U", "litellm_smoke", "-d", "litellm_smoke", "-c", 'SELECT count(*) FROM "LiteLLM_SpendLogs";']
    return int(subprocess.check_output(command, text=True).strip())

def node_value(command, remote=False):
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", "-o", "ConnectionAttempts=1", worker, command] if remote else ["bash", "-lc", command]
    return subprocess.check_output(argv, text=True, timeout=4).strip()

def node_sample(remote=False):
    output = node_value(
        "awk '/MemAvailable/ {print $2/1024/1024}' /proc/meminfo; "
        "docker inspect -f '{{.RestartCount}}' deepseek-v4-flash-vllm-dspark-1",
        remote,
    ).splitlines()
    if len(output) != 2:
        raise RuntimeError("node sample did not return memory and restart count")
    return float(output[0]), int(output[1])

def sampler():
    next_at = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as node_pool:
        while not stop.is_set():
            started = time.monotonic()
            try:
                head_future = node_pool.submit(node_sample)
                worker_future = node_pool.submit(node_sample, True)
                head_mem, head_restart = head_future.result()
                worker_mem, worker_restart = worker_future.result()
                metrics = prometheus()
                node_rows.append((started, head_mem, worker_mem, head_restart, worker_restart))
                specs = [float(line.rsplit(" ", 1)[1]) for line in metrics.splitlines() if "spec" in line.lower() and "accept" in line.lower() and not line.startswith("#")]
                metric_rows.append((
                    metric(metrics, "vllm:num_requests_running"),
                    metric(metrics, "vllm:num_requests_waiting"),
                    metric(metrics, "vllm:kv_cache_usage_perc"),
                    metric(metrics, "vllm:num_preemptions_total"),
                    statistics.mean(specs) if specs else None,
                ))
            except Exception as error:
                sample_errors.append(type(error).__name__)
            next_at += interval
            stop.wait(max(0.0, next_at - time.monotonic()))

def one_request(request_id):
    payload = json.dumps({
        "model": "deepseek-v4-flash-0731-smoke",
        "messages": [{"role": "user", "content": f"Synthetic soak nonce {request_id}. Return a concise numbered reliability checklist."}],
        "max_tokens": 128, "temperature": 0.6, "top_p": 0.95, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    request = urllib.request.Request(gateway_url.rstrip("/") + "/chat/completions", data=payload, headers={"Authorization": f"Bearer {gateway_key}", "Content-Type": "application/json", "X-Request-ID": request_id})
    done, usage, finish = False, [], None
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"): continue
            body = line[5:].strip()
            if body == "[DONE]": done = True; break
            chunk = json.loads(body)
            if chunk.get("usage"): usage.append(chunk["usage"])
            for choice in chunk.get("choices") or []:
                if choice.get("finish_reason") is not None: finish = choice["finish_reason"]
    if not done or len(usage) != 1 or finish not in {"stop", "length"}:
        raise RuntimeError("incomplete soak stream")

before_origin = bench.success_counter(origin_url, origin_key)
before_gateway = spend_count()
before_metrics = prometheus()
preempt_before = metric(before_metrics, "vllm:num_preemptions_total")
thread = threading.Thread(target=sampler, daemon=True); thread.start()
started = time.monotonic(); deadline = started + duration
request_count, failed, idle_gaps, prior_end = 0, 0, [], None
try:
    while time.monotonic() < deadline:
        request_started = time.monotonic()
        if prior_end is not None: idle_gaps.append(request_started - prior_end)
        try:
            one_request(f"dspark-soak-{uuid.uuid4()}")
            request_count += 1
        except Exception:
            failed += 1
            raise
        prior_end = time.monotonic()
finally:
    stop.set(); thread.join(timeout=interval + 5)
elapsed = int(time.monotonic() - started)
after_origin = bench.success_counter(origin_url, origin_key)
deadline_flush = time.monotonic() + 60
after_gateway = spend_count()
while after_gateway - before_gateway < request_count and time.monotonic() < deadline_flush:
    time.sleep(1); after_gateway = spend_count()
origin_delta = int(after_origin - before_origin)
gateway_delta = after_gateway - before_gateway
preempt_after = metric(prometheus(), "vllm:num_preemptions_total")
gaps = [node_rows[i][0] - node_rows[i-1][0] for i in range(1, len(node_rows))]
spec_values = [row[4] for row in metric_rows if row[4] is not None]
minimum_samples = int(duration / interval * 0.9)
sample_error_limit = max(3, int(duration / interval * 0.01))
accepted = all((
    elapsed >= duration, request_count > 0, failed == 0,
    origin_delta == request_count, gateway_delta == request_count,
    len(sample_errors) <= sample_error_limit, len(node_rows) >= minimum_samples,
    max(idle_gaps or [0.0]) <= 1.0, max(gaps or [0.0]) <= interval * 2.5,
    min(row[1] for row in node_rows) >= 4.0, min(row[2] for row in node_rows) >= 4.0,
    max(row[3] for row in node_rows) == 0, max(row[4] for row in node_rows) == 0,
    max(row[0] for row in metric_rows) <= 1, max(row[1] for row in metric_rows) == 0,
    preempt_after - preempt_before == 0,
))
result = {
    "accepted": accepted, "duration_seconds": elapsed, "sample_interval_seconds": interval,
    "request_count": request_count, "origin_completion_delta": origin_delta,
    "gateway_attempt_delta": gateway_delta, "failed_requests": failed,
    "sample_error_count": len(sample_errors),
    "max_idle_gap_seconds": round(max(idle_gaps or [0.0]), 6), "node_samples": len(node_rows),
    "min_head_mem_available_gib": round(min(row[1] for row in node_rows), 3),
    "min_worker_mem_available_gib": round(min(row[2] for row in node_rows), 3),
    "max_requests_running": max(row[0] for row in metric_rows),
    "max_requests_waiting": max(row[1] for row in metric_rows),
    "preemption_delta": preempt_after - preempt_before,
    "max_rank_restarts": max(max(row[3], row[4]) for row in node_rows),
    "max_node_sample_gap_seconds": round(max(gaps or [0.0]), 6),
    "kv_cache_usage_peak": max(row[2] for row in metric_rows),
    "speculative_acceptance_mean": round(statistics.mean(spec_values), 6) if spec_values else None,
}
target = Path(output); target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); target.chmod(0o600)
if not accepted: raise SystemExit("soak acceptance thresholds failed")
PY
"$SANITIZER" --scan-only <"$SOAK_EVIDENCE" >/dev/null
"$SCRIPT_DIR/litellm/smoke.sh" --all-interfaces
ENV_FILE="$ENV_FILE" "$SCRIPT_DIR/scripts/collect-node-evidence.sh" "$NODE_EVIDENCE_AFTER"

FAILURE_STAGE="report"
[ -f "$ACTIVE_GATEWAY_SNAPSHOT" ] || { echo "Missing active gateway snapshot." >&2; false; }
GATEWAY_SNAPSHOT_EVIDENCE="$RUN_DIR/active-gateway-snapshot.json"
cp "$ACTIVE_GATEWAY_SNAPSHOT" "$GATEWAY_SNAPSHOT_EVIDENCE"
chmod 0600 "$GATEWAY_SNAPSHOT_EVIDENCE"
python3 - "$QWEN_MANIFEST" "$DIRECT_BENCHMARK" "$GATEWAY_BENCHMARK" "$NODE_EVIDENCE_AFTER" \
  "$SOAK_EVIDENCE" "$FIXTURE" "$DIRECT_INTERIM" "${hermes_files[0]}" "${hermes_files[1]}" \
  "$GATEWAY_SNAPSHOT_EVIDENCE" "$ENV_FILE" "$ROOT_DIR" <<'PY' \
  | "$SANITIZER" --schema "$SCHEMA" --output "$FINAL_REPORT"
from datetime import datetime, timezone
import hashlib, json
from pathlib import Path
import subprocess, sys

manifest_path, direct_path, gateway_path, node_path, soak_path, fixture_path, interim_path, hermes_a, hermes_b, gateway_snapshot, env_path, root = map(Path, sys.argv[1:])
def load(path): return json.loads(path.read_text())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def env_file(path):
    values = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1); values[key] = value
    return values
env = env_file(env_path)
manifest, direct, gateway, nodes, soak, interim = map(load, (manifest_path, direct_path, gateway_path, node_path, soak_path, interim_path))
hermes = [load(hermes_a), load(hermes_b)]
if not direct["summary"]["accepted"] or not gateway["summary"]["accepted"] or not soak["accepted"]:
    raise SystemExit("performance or soak evidence is not accepted")
if not all(item["rank_running"] and item["rank_restart_count"] == 0 for item in nodes["nodes"]):
    raise SystemExit("rank state failed after soak")
if not all(item["accepted"] and item["shared_state"]["unchanged"] for item in hermes):
    raise SystemExit("Hermes evidence is not accepted")
runtime = env["DSPARK_VLLM_IMAGE"]
if "@sha256:" not in runtime: raise SystemExit("runtime image is not pinned")
hermes_pin_inputs = [root / "deployments/private-smoke/hermes/config.yaml", root / "deployments/private-smoke/hermes/fixtures/transform-input.json", root / "deployments/private-smoke/hermes/fixtures/tool-contract.json"]
expected_hermes_pin = hashlib.sha256(b"".join(path.read_bytes() for path in hermes_pin_inputs)).hexdigest()
if {item["suite_pin_sha256"] for item in hermes} != {expected_hermes_pin}:
    raise SystemExit("Hermes pin changed between functional runs")
config_inputs = [root / "docker-compose.dspark.yml", root / "deployments/private-smoke/litellm/config.yaml", *hermes_pin_inputs, fixture_path]
config_hash = hashlib.sha256(b"".join(path.read_bytes() for path in config_inputs)).hexdigest()
repo_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
pins = {"model_revision": env["DSPARK_MODEL_REVISION"], "runtime_image_digest": "sha256:" + runtime.rsplit("@sha256:", 1)[1], "repo_commit": repo_commit, "config_sha256": config_hash}
pin_hash = hashlib.sha256(json.dumps(pins, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
evidence = [("qwen-manifest", manifest_path), ("direct", direct_path), ("gateway", gateway_path), ("nodes", node_path), ("hermes-a", hermes_a), ("hermes-b", hermes_b), ("soak", soak_path), ("public-gateway", gateway_snapshot)]
previous = "0" * 64; chain = []
for name, path in evidence:
    try:
        artifact_path = path.resolve().relative_to(manifest_path.parent.resolve()).as_posix()
    except ValueError as exc:
        raise SystemExit(f"evidence artifact is outside the acceptance run: {name}") from exc
    artifact = digest(path)
    entry = hashlib.sha256(f"{previous}:{name}:{artifact}:{pin_hash}".encode()).hexdigest()
    chain.append({"name": name, "artifact_path": artifact_path, "artifact_sha256": artifact, "previous_sha256": previous, "entry_sha256": entry})
    previous = entry
def summary(item):
    value = item["summary"]
    return {"median_decode_tokens_per_second": value["median_decode_tokens_per_second"], "p95_ttft_seconds": value["p95_ttft_seconds"], "request_count": value["expected_completions"], "origin_completion_delta": value["metric_delta"], "accepted": value["accepted"]}
d, g = summary(direct), summary(gateway)
gates = {name: True for name in ("fabric", "artifacts", "qwen_stopped", "direct", "gateway", "hermes", "performance", "soak", "isolation", "sanitization", "public_gateway_unchanged")}
report = {
  "schema_version": 1, "run_id": manifest["run_id"], "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "accepted": True,
  "manifest_sha256": digest(manifest_path), "fixture_sha256": digest(fixture_path), "pin_set_sha256": pin_hash, "chain_head_sha256": previous,
  "pins": pins, "gates": gates,
  "functional_runs": [{"artifact_sha256": digest(path), "accepted": item["accepted"], "api_calls": item["usage"]["api_calls"], "gateway_attested": True} for path, item in ((hermes_a, hermes[0]), (hermes_b, hermes[1]))],
  "performance": {"direct": d, "litellm": g, "median_decode_overhead_ratio": g["median_decode_tokens_per_second"] / d["median_decode_tokens_per_second"], "p95_ttft_overhead_seconds": g["p95_ttft_seconds"] - d["p95_ttft_seconds"]},
  "soak": soak, "evidence_chain": chain, "purge_eligible": True,
}
print(json.dumps(report))
PY

# Final canary scan: accepted evidence itself must contain no secret, private
# address, absolute host path, unknown field, or broken evidence_chain.
"$SANITIZER" --schema "$SCHEMA" --input "$FINAL_REPORT" >/dev/null
python3 "$ROOT_DIR/scripts/qwen_manifest.py" verify-report --manifest "$QWEN_MANIFEST" \
  --report "$FINAL_REPORT" --max-age-hours 24 --required-gate fabric --required-gate artifacts \
  --required-gate direct --required-gate hermes

trap - ERR
echo "Acceptance passed and is purge_eligible: $FINAL_REPORT"
echo "DeepSeek and the private LiteLLM remain manually started for inspection; no supervisor was installed."
