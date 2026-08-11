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
MAX_MEMORY_PSI_FULL_AVG10="$(python3 - "$ROOT_DIR/scripts/probe-full-context.py" <<'PY'
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("dspark_full_context", sys.argv[1])
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
print(probe.MAX_MEMORY_PSI_FULL_AVG10)
PY
)"
MODE=""
RUN_DIR=""
HERMES_RESULTS=""
FAILURE_STAGE="startup"
LIVE_STARTED=0

usage() {
  echo "Usage: $0 --validate-fixtures" >&2
  echo "       $0 --live [--run-dir PATH] [--hermes-results DIR]" >&2
  echo "       $0 --resume-capacity [--run-dir PATH] [--hermes-results DIR]" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --validate-fixtures) MODE="fixtures"; shift ;;
    --live) MODE="live"; shift ;;
    --resume-capacity) MODE="resume-capacity"; shift ;;
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
assert fixture["soak"]["speculative_metrics"] == {
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
    "draft": "vllm:spec_decode_num_draft_tokens_total",
}
assert fixture["synthetic_expenses"]["expected_grand_total_centavos"] == 374025
PY
  python3 - <<'PY' | "$SANITIZER" --schema "$SCHEMA" >/dev/null
import hashlib, json
h = "a" * 64
performance = {"median_decode_tokens_per_second": 55.0, "p95_ttft_seconds": 2.0, "request_count": 24, "origin_completion_delta": 24, "accepted": True}
gates = {name: True for name in ("fabric", "artifacts", "qwen_stopped", "direct", "gateway", "hermes", "performance", "soak", "isolation", "sanitization", "public_gateway_unchanged", "minefield", "external_gateway", "prompt_reasoning_canary", "full_context", "long_context_decode", "scheduler")}
pins = {"model_revision": "b" * 40, "runtime_image_digest": "sha256:" + h, "repo_commit": "c" * 40, "config_sha256": h}
pin_hash = hashlib.sha256(json.dumps(pins, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
chain = []; previous = "0" * 64
for i in range(7):
  entry = hashlib.sha256(f"{previous}:gate-{i}:{h}:{pin_hash}".encode()).hexdigest()
  chain.append({"name": f"gate-{i}", "artifact_path": f"gate-{i}.json", "artifact_sha256": h, "previous_sha256": previous, "entry_sha256": entry})
  previous = entry
print(json.dumps({
  "schema_version": 2, "run_id": "fixture-run", "created_at": "2026-08-02T12:00:00Z",
  "accepted": True, "manifest_sha256": h, "fixture_sha256": h, "pin_set_sha256": pin_hash,
  "chain_head_sha256": previous, "pins": pins,
  "gates": gates,
  "functional_runs": [{"artifact_sha256": h, "accepted": True, "api_calls": 3, "gateway_attested": True}, {"artifact_sha256": h, "accepted": True, "api_calls": 3, "gateway_attested": True}],
  "performance": {"direct": performance, "litellm": performance, "median_decode_overhead_ratio": 1.0, "p95_ttft_overhead_seconds": 0.1},
  "semantic_readiness": {"direct_origin": {"state": "semantic-ready", "artifact_sha256": h}, "private_litellm": {"state": "semantic-ready", "artifact_sha256": h}},
  "soak": {"accepted": True, "duration_seconds": 1800, "sample_interval_seconds": 5, "request_count": 10, "origin_completion_delta": 10, "gateway_attempt_delta": 10, "failed_requests": 0, "sample_error_count": 0, "max_idle_gap_seconds": 0.1, "node_samples": 360, "min_head_mem_available_gib": 9.0, "min_worker_mem_available_gib": 9.0, "max_memory_psi_full_avg10": 0.67, "max_requests_running": 1.0, "max_requests_waiting": 0.0, "preemption_delta": 0.0, "max_rank_restarts": 0, "max_node_sample_gap_seconds": 5.1, "kv_cache_usage_peak": 0.5, "prefix_cache_queries_delta": 10.0, "prefix_cache_hits_delta": 6.0, "prefix_cache_reuse_ratio": 0.6, "speculative_accepted_tokens_delta": 0.0, "speculative_draft_tokens_delta": 0.0, "speculative_acceptance_ratio": None, "speculative_acceptance_observation": "not-observed"},
  "rollout_evidence": {"process_readiness": {"head_running": True, "worker_running": True, "restart_count": 0}, "api_readiness": {"authenticated": True, "model_discovery": True}, "semantic_readiness": {"direct_origin": True, "private_litellm": True}, "kv_cache": {"configured_bytes": 12884901888, "reported_token_capacity": 1048576}, "rank_participation": {"world_size": 2, "both_ranks_participated": True}, "memory": {"min_head_mem_available_gib": 9.0, "min_worker_mem_available_gib": 9.0, "max_memory_psi_full_avg10": 0.0}, "prefix_cache": {"queries_delta": 10.0, "hits_delta": 6.0, "reuse_ratio": 0.6}, "speculative_decode": {"accepted_tokens_delta": 0.0, "draft_tokens_delta": 0.0, "acceptance_ratio": None}, "minefield": {"commit": "2b453b8a69dbaf8dc9d521dc2d6212cdaceb8169", "executed": 20, "problem": 0, "inconclusive": 2, "unimplemented": 80}, "external_gateway": {"unauthenticated_status": 401, "authenticated_generation": True}, "prompt_reasoning_canaries_absent": True, "message_logging_disabled": True},
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

CAPACITY_EVIDENCE_IDENTITY="$(python3 - "$ROOT_DIR" "$FIXTURE" <<'PY'
import hashlib, json, os, subprocess, sys
from pathlib import Path
root, fixture = Path(sys.argv[1]), Path(sys.argv[2])
hermes = [
    root / "deployments/private-smoke/hermes/config.yaml",
    root / "deployments/private-smoke/hermes/fixtures/transform-input.json",
    root / "deployments/private-smoke/hermes/fixtures/tool-contract.json",
]
inputs = [
    root / "docker-compose.dspark.yml",
    root / "recipe/runtime-hotfixes.manifest.json",
    root / "patches/hotfix-encoding-dsv4-issue21.py",
    root / "patches/hotfix-nvfp4-ds-mla-issue22.py",
    root / "deployments/private-smoke/litellm/config.yaml",
    *hermes,
    fixture,
]
runtime = os.environ["DSPARK_VLLM_IMAGE"]
runtime_digest = runtime if runtime.startswith("sha256:") else "sha256:" + runtime.rsplit("@sha256:", 1)[1]
pins = {
    "model_revision": os.environ["DSPARK_MODEL_REVISION"],
    "runtime_image_digest": runtime_digest,
    "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    "config_sha256": hashlib.sha256(b"".join(path.read_bytes() for path in inputs)).hexdigest(),
}
print(hashlib.sha256(json.dumps(pins, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
PY
)"

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
DIRECT_SEMANTIC_EVIDENCE="$RUN_DIR/semantic-direct-origin.json"
LITELLM_SEMANTIC_EVIDENCE="$RUN_DIR/semantic-private-litellm.json"
MINEFIELD_EVIDENCE="$RUN_DIR/minefield.json"
EXTERNAL_GATEWAY_EVIDENCE="$RUN_DIR/external-gateway-auth.json"
CANARY_EVIDENCE="$RUN_DIR/prompt-reasoning-canary-absence.json"
FULL_CONTEXT_EVIDENCE="$RUN_DIR/full-context.json"
LONG_CONTEXT_DECODE_EVIDENCE="$RUN_DIR/long-context-decode.json"
SCHEDULER_EVIDENCE="$RUN_DIR/scheduler-current.json"
SCHEDULER_BASELINE="${SCHEDULER_BASELINE:-$ROOT_DIR/artifacts/health-rollout/scheduler-baseline.json}"
PROMPT_LOG_CANARY="DSPARK_PROMPT_LOG_CANARY_U5_20260809"
REASONING_LOG_CANARY="DSPARK_REASONING_LOG_CANARY_U5_20260809"
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

if [ "$MODE" = "live" ]; then
for required in "$QWEN_MANIFEST" "$DIRECT_INTERIM" "$DIRECT_BENCHMARK" "$NODE_EVIDENCE" "$FIXTURE"; do
  [ -f "$required" ] || { echo "Missing prerequisite evidence." >&2; false; }
done
[ ! -e "$FINAL_REPORT" ] || { echo "Refusing to overwrite accepted report." >&2; false; }

FAILURE_STAGE="preconditions"
git -C "$ROOT_DIR" diff --quiet
git -C "$ROOT_DIR" diff --cached --quiet
ENV_FILE="$ENV_FILE" "$ROOT_DIR/status-deepseek-v4-flash-dspark.sh" --expect running
python3 "$ROOT_DIR/scripts/smoke-openai-compat.py" \
  --semantic-canary --profile direct-origin --wall-timeout 120 \
  --base-url "http://${VLLM_PROXY_HOST:-172.30.0.1}:${VLLM_PROXY_PORT:-8888}/v1" \
  --key-file "$VLLM_ORIGIN_KEY_FILE" --model "${SERVED_MODEL_NAME:-deepseek-v4-flash-dspark}" \
  --output "$DIRECT_SEMANTIC_EVIDENCE"
[ "$(docker inspect -f '{{.State.Running}}' urbanplan-qwen 2>/dev/null)" = "false" ]
python3 "$ROOT_DIR/scripts/qwen_manifest.py" verify-live --manifest "$QWEN_MANIFEST" --max-age-hours 24
python3 - "$DIRECT_INTERIM" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report["accepted"] is False
assert report["gates"]["fabric"] and report["gates"]["artifacts"] and report["gates"]["direct"]
PY

benchmark_spend_count() {
  local benchmark_file="$1"
  python3 - "$benchmark_file" <<'PY' | docker exec -i dspark-private-litellm-postgres-1 \
    psql -X -A -t -U litellm_smoke -d litellm_smoke | tr -d '[:space:]'
import json
import re
from pathlib import Path
import sys

benchmark = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rows = [benchmark["cold"], *benchmark["discarded_warmups"], *benchmark["samples"]]
# LiteLLM 1.92.0 keys LiteLLM_SpendLogs.request_id from the response object's
# id (get_spend_logs_id), not from the caller's X-Request-ID header.  The
# benchmark records that streamed response id as origin_request_id.  Match the
# database primary key exactly so message logging can remain disabled.
request_ids = [row["origin_request_id"] for row in rows]
if len(request_ids) != 24 or len(set(request_ids)) != 24:
    raise SystemExit("benchmark origin response IDs are missing or duplicated")
if not all(re.fullmatch(r"chatcmpl-[a-f0-9]{16}", item) for item in request_ids):
    raise SystemExit("benchmark origin response ID format is invalid")
ids = ",".join("'" + item + "'" for item in request_ids)
print(
    'SELECT count(*) FROM "LiteLLM_SpendLogs" AS t '
    f'WHERE t.request_id IN ({ids});'
)
PY
}

wait_for_benchmark_spend() {
  local benchmark_file="$1" expected="$2" current=""
  for _ in $(seq 1 120); do
    current="$(benchmark_spend_count "$benchmark_file")"
    [ "$current" -eq "$expected" ] && { printf '%s\n' "$current"; return 0; }
    [ "$current" -lt "$expected" ] || {
      echo "Benchmark request IDs produced duplicate spend rows: $current > $expected." >&2
      return 1
    }
    sleep 1
  done
  echo "LiteLLM spend rows for benchmark request IDs did not reach $expected (last=$current)." >&2
  return 1
}

check_canary_absence() {
  local canary_since
  canary_since="$(date --iso-8601=seconds)"
  python3 - "$VLLM_ORIGIN_KEY_FILE" "$LITELLM_VIRTUAL_KEY_FILE" \
    "http://${VLLM_PROXY_HOST:-172.30.0.1}:${VLLM_PROXY_PORT:-8888}/v1" \
    "http://${HEAD_TAILSCALE_IP}:4001/v1" "$PROMPT_LOG_CANARY" "$REASONING_LOG_CANARY" <<'PY'
import json
from pathlib import Path
import sys
import urllib.request
origin_key_path, gateway_key_path, origin, gateway, prompt_marker, reasoning_marker = sys.argv[1:]
def send(base, key, model):
    payload = {"model": model, "messages": [
        {"role": "user", "content": "Return a short acknowledgement."},
        {"role": "assistant", "content": "Acknowledged.", "reasoning_content": reasoning_marker},
        {"role": "user", "content": prompt_marker},
    ], "max_completion_tokens": 16, "temperature": 0,
       "chat_template_kwargs": {"reasoning_effort": "none", "drop_thinking": False}}
    request = urllib.request.Request(base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Authorization": "Bearer " + Path(key).read_text().strip(), "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 200 or not json.load(response).get("choices"):
            raise SystemExit("canary generation failed")
send(origin, origin_key_path, "deepseek-v4-flash-0731")
send(gateway, gateway_key_path, "deepseek-v4-flash-0731-smoke")
PY
  grep -F 'turn_off_message_logging: true' "$SCRIPT_DIR/litellm/config.yaml" >/dev/null
  local canary_logs
  canary_logs="$(mktemp)"; chmod 0600 "$canary_logs"
  if ! docker logs --since "$canary_since" deepseek-v4-flash-origin-auth-proxy-1 >>"$canary_logs" 2>&1 ||
     ! docker logs --since "$canary_since" deepseek-v4-flash-vllm-dspark-1 >>"$canary_logs" 2>&1 ||
     ! docker logs --since "$canary_since" dspark-private-litellm-litellm-1 >>"$canary_logs" 2>&1 ||
     ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" \
       "docker logs --since $(printf '%q' "$canary_since") deepseek-v4-flash-vllm-dspark-1 2>&1" >>"$canary_logs"; then
    rm -f "$canary_logs"
    echo "Could not read every proxy/vLLM/LiteLLM canary log source" >&2
    return 1
  fi
  if ! python3 - "$canary_logs" "$RUN_DIR" "$PROMPT_LOG_CANARY" "$REASONING_LOG_CANARY" <<'PY'
from pathlib import Path
import sys
log_path, run_dir = map(Path, sys.argv[1:3])
markers = [item.encode() for item in sys.argv[3:]]
paths = [log_path, *sorted(path for path in run_dir.rglob("*") if path.is_file())]
for path in paths:
    content = path.read_bytes()
    if any(marker in content for marker in markers):
        raise SystemExit("prompt/reasoning canary leaked into logs or evidence")
PY
  then
    rm -f "$canary_logs"
    return 1
  fi
  rm -f "$canary_logs"
  printf '%s\n' '{"schema_version":1,"prompt_reasoning_canaries_absent":true,"message_logging_disabled":true}' \
    | "$SANITIZER" --scan-only --output "$CANARY_EVIDENCE"
}

FAILURE_STAGE="gateway"
python3 "$ROOT_DIR/scripts/smoke-openai-compat.py" \
  --semantic-canary --profile private-litellm --wall-timeout 120 \
  --base-url "http://${HEAD_TAILSCALE_IP}:4001/v1" \
  --key-file "$LITELLM_VIRTUAL_KEY_FILE" --model deepseek-v4-flash-0731-smoke \
  --output "$LITELLM_SEMANTIC_EVIDENCE"
"$SCRIPT_DIR/litellm/smoke.sh" --all-interfaces
EXTERNAL_LITELLM_BASE_URL="${EXTERNAL_LITELLM_BASE_URL:-http://${HEAD_TAILSCALE_IP}:4001/v1}"
python3 - "$EXTERNAL_LITELLM_BASE_URL" "$LITELLM_VIRTUAL_KEY_FILE" <<'PY' \
  | "$SANITIZER" --scan-only --output "$EXTERNAL_GATEWAY_EVIDENCE"
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request
base, key_path = sys.argv[1:]
def call(token, body=None):
    request = urllib.request.Request(base.rstrip("/") + ("/models" if body is None else "/chat/completions"),
        data=None if body is None else json.dumps(body).encode(),
        headers={} if token is None else {"Authorization": "Bearer " + token, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        status = error.code; error.close(); return status, {}
unauthenticated_status, _ = call(None)
if unauthenticated_status not in (401, 403):
    raise SystemExit("external gateway did not enforce authentication")
key = Path(key_path).read_text().strip()
status, payload = call(key, {"model": "deepseek-v4-flash-0731-smoke", "messages": [{"role": "user", "content": "Reply exactly EXTERNAL_GATEWAY_OK."}], "max_completion_tokens": 32, "temperature": 0})
if status != 200 or not payload.get("choices"):
    raise SystemExit("external authenticated generation failed")
print(json.dumps({"schema_version": 1, "unauthenticated_status": unauthenticated_status, "authenticated_generation": True}))
PY
python3 "$ROOT_DIR/scripts/run-minefield-pinned.py" \
  --commit 2b453b8a69dbaf8dc9d521dc2d6212cdaceb8169 \
  --env-file "$ENV_FILE" --json "$MINEFIELD_EVIDENCE"
"$SANITIZER" --scan-only --input "$MINEFIELD_EVIDENCE" >/dev/null
python3 "$SCRIPT_DIR/scripts/benchmark.py" --layer litellm --warmups 3 --samples 20 --concurrency 1 \
  --base-url "http://${HEAD_TAILSCALE_IP}:4001/v1" --key-file "$LITELLM_VIRTUAL_KEY_FILE" \
  --model deepseek-v4-flash-0731-smoke \
  --origin-metrics-base-url "http://${VLLM_PROXY_HOST:-172.30.0.1}:${VLLM_PROXY_PORT:-8888}/v1" \
  --origin-key-file "$VLLM_ORIGIN_KEY_FILE" --output "$GATEWAY_BENCHMARK"
wait_for_benchmark_spend "$GATEWAY_BENCHMARK" 24 >/dev/null
check_canary_absence

FAILURE_STAGE="hermes"
mapfile -t hermes_files < <(find "$HERMES_RESULTS" -maxdepth 1 -type f -name 'hermes-smoke-*.json' -print | sort)
[ "${#hermes_files[@]}" -eq 2 ] || { echo "Hermes acceptance requires exactly two result files." >&2; false; }
python3 - "${hermes_files[@]}" <<'PY'
import json
import re
import subprocess
import sys
import time

def spend_rows(response_ids):
    if not response_ids or not all(re.fullmatch(r"chatcmpl-[a-f0-9]{16}", item) for item in response_ids):
        raise AssertionError("invalid Hermes origin response ids")
    ids = ",".join("'" + item + "'" for item in response_ids)
    sql = (
        'SELECT row_to_json(t)::text FROM "LiteLLM_SpendLogs" AS t '
        f"WHERE t.request_id IN ({ids});"
    )
    command = [
        "docker", "exec", "dspark-private-litellm-postgres-1",
        "psql", "-X", "-A", "-t", "-U", "litellm_smoke", "-d", "litellm_smoke",
        "-c", sql,
    ]
    for _ in range(120):
        output = subprocess.check_output(command, text=True)
        rows = [json.loads(line) for line in output.splitlines() if line.strip()]
        if len(rows) == len(response_ids):
            return rows
        if len(rows) > len(response_ids):
            raise AssertionError("duplicate Hermes spend rows")
        time.sleep(1)
    raise AssertionError("Hermes spend rows did not become visible")

all_origin_response_ids = []
for path in sys.argv[1:]:
    item = json.load(open(path))
    assert item["accepted"] is True
    assert item["model"] == "deepseek-v4-flash-0731-smoke"
    assert item["provider"] == "custom:deepseek-smoke"
    assert item["shared_state"]["unchanged"] is True
    assert all(item["negative_checks"].values())
    assert item["request_ids"] == [item["run_id"]]
    assert len(item["origin_response_ids"]) == item["usage"]["api_calls"]
    assert len(set(item["origin_response_ids"])) == item["usage"]["api_calls"]
    rows = spend_rows(item["origin_response_ids"])
    assert len(rows) == item["usage"]["api_calls"]
    for row in rows:
        encoded = json.dumps(row, sort_keys=True)
        assert "deepseek-v4-flash-0731-smoke" in encoded
        assert "hermes-deepseek-smoke" in encoded
    all_origin_response_ids.extend(item["origin_response_ids"])
assert len(set(all_origin_response_ids)) == len(all_origin_response_ids)
assert len({item["suite_pin_sha256"] for item in map(lambda p: json.load(open(p)), sys.argv[1:])}) == 1
PY

FAILURE_STAGE="soak"
SOAK_DURATION_SECONDS="${SOAK_DURATION_SECONDS:-1800}"
SOAK_SAMPLE_INTERVAL_SECONDS="${SOAK_SAMPLE_INTERVAL_SECONDS:-5}"
[ "$SOAK_DURATION_SECONDS" -eq 1800 ] && [ "$SOAK_SAMPLE_INTERVAL_SECONDS" -eq 5 ] || {
  echo "Live acceptance requires the full 1800-second soak and 5-second samples." >&2
  false
}

python3 - "$SCRIPT_DIR/scripts/benchmark.py" "$SCRIPT_DIR/scripts/soak_spend.py" "$WORKER_HOST" \
  "http://${HEAD_TAILSCALE_IP}:4001/v1" "$LITELLM_VIRTUAL_KEY_FILE" \
  "http://${VLLM_PROXY_HOST:-172.30.0.1}:${VLLM_PROXY_PORT:-8888}/v1" "$VLLM_ORIGIN_KEY_FILE" \
  "$SOAK_DURATION_SECONDS" "$SOAK_SAMPLE_INTERVAL_SECONDS" "$MAX_MEMORY_PSI_FULL_AVG10" \
  "$SOAK_EVIDENCE" <<'PY'
from __future__ import annotations
import importlib.util
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

benchmark_path, spend_helper_path, worker, gateway_url, gateway_key_path, origin_url, origin_key_path, duration_raw, interval_raw, max_memory_psi_raw, output = sys.argv[1:]
duration, interval = int(duration_raw), int(interval_raw)
max_memory_psi_full_avg10 = float(max_memory_psi_raw)
spec = importlib.util.spec_from_file_location("dspark_benchmark", benchmark_path)
bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
spend_spec = importlib.util.spec_from_file_location("dspark_soak_spend", spend_helper_path)
soak_spend = importlib.util.module_from_spec(spend_spec); spend_spec.loader.exec_module(soak_spend)
gateway_key = Path(gateway_key_path).read_text().strip()
origin_key = Path(origin_key_path).read_text().strip()
stop = threading.Event()
node_rows, metric_rows, sample_errors = [], [], []

def prometheus():
    with bench.request(origin_url.removesuffix("/v1"), origin_key, "/metrics", timeout=30) as response:
        return response.read().decode()

def metric(text, name):
    values = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric_name = line.split(None, 1)[0].split("{", 1)[0]
        if metric_name == name:
            try:
                value = float(line.rsplit(None, 1)[1])
            except (IndexError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
    if not values:
        raise RuntimeError(f"required Prometheus metric is missing: {name}")
    return sum(values)

def node_value(command, remote=False):
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", "-o", "ConnectionAttempts=1", worker, command] if remote else ["bash", "-lc", command]
    return subprocess.check_output(argv, text=True, timeout=4).strip()

def node_sample(remote=False):
    output = node_value(
        "awk '/MemAvailable/ {print $2/1024/1024}' /proc/meminfo; "
        "docker inspect -f '{{.RestartCount}}' deepseek-v4-flash-vllm-dspark-1; "
        "awk '/^full / {for (i=1;i<=NF;i++) if ($i ~ /^avg10=/) {sub(/^avg10=/, \"\", $i); print $i}}' /proc/pressure/memory",
        remote,
    ).splitlines()
    if len(output) != 3:
        raise RuntimeError("node sample did not return memory, restart count, and memory PSI")
    return float(output[0]), int(output[1]), float(output[2])

def sampler():
    next_at = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as node_pool:
        while not stop.is_set():
            started = time.monotonic()
            try:
                head_future = node_pool.submit(node_sample)
                worker_future = node_pool.submit(node_sample, True)
                head_mem, head_restart, head_psi = head_future.result()
                worker_mem, worker_restart, worker_psi = worker_future.result()
                metrics = prometheus()
                node_rows.append((started, head_mem, worker_mem, head_restart, worker_restart, head_psi, worker_psi))
                metric_rows.append((
                    metric(metrics, "vllm:num_requests_running"),
                    metric(metrics, "vllm:num_requests_waiting"),
                    metric(metrics, "vllm:kv_cache_usage_perc"),
                    metric(metrics, "vllm:num_preemptions_total"),
                ))
            except Exception as error:
                sample_errors.append(type(error).__name__)
            next_at += interval
            stop.wait(max(0.0, next_at - time.monotonic()))

SOAK_SHARED_PREFIX = (
    "Synthetic reliability context: validate deterministic service health, bounded "
    "latency, memory headroom, queue stability, cache reuse, and error isolation. "
) * 160

def one_request(request_id):
    payload = json.dumps({
        "model": "deepseek-v4-flash-0731-smoke",
        "messages": [{"role": "user", "content": (
            SOAK_SHARED_PREFIX
            + f"\nRequest nonce {request_id}. Return a concise numbered reliability checklist."
        )}],
        "max_tokens": 128, "temperature": 0.6, "top_p": 0.95, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    request = urllib.request.Request(gateway_url.rstrip("/") + "/chat/completions", data=payload, headers={"Authorization": f"Bearer {gateway_key}", "Content-Type": "application/json", "X-Request-ID": request_id})
    done, usage, finish, response_ids = False, [], None, set()
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"): continue
            body = line[5:].strip()
            if body == "[DONE]": done = True; break
            chunk = json.loads(body)
            if chunk.get("id"): response_ids.add(chunk["id"])
            if chunk.get("usage"): usage.append(chunk["usage"])
            for choice in chunk.get("choices") or []:
                if choice.get("finish_reason") is not None: finish = choice["finish_reason"]
    if not done or len(usage) != 1 or finish not in {"stop", "length"} or len(response_ids) != 1:
        raise RuntimeError("incomplete soak stream")
    return next(iter(response_ids))

before_origin = bench.success_counter(origin_url, origin_key)
before_metrics = prometheus()
preempt_before = metric(before_metrics, "vllm:num_preemptions_total")
accepted_before = metric(before_metrics, "vllm:spec_decode_num_accepted_tokens_total")
draft_before = metric(before_metrics, "vllm:spec_decode_num_draft_tokens_total")
prefix_queries_before = metric(before_metrics, "vllm:prefix_cache_queries_total")
prefix_hits_before = metric(before_metrics, "vllm:prefix_cache_hits_total")
thread = threading.Thread(target=sampler, daemon=True); thread.start()
started = time.monotonic(); deadline = started + duration
request_count, failed, idle_gaps, prior_end, origin_response_ids = 0, 0, [], None, []
try:
    while time.monotonic() < deadline:
        request_started = time.monotonic()
        if prior_end is not None: idle_gaps.append(request_started - prior_end)
        try:
            origin_response_ids.append(one_request(f"dspark-soak-{uuid.uuid4()}"))
            request_count += 1
        except Exception:
            failed += 1
            raise
        prior_end = time.monotonic()
finally:
    stop.set(); thread.join(timeout=interval + 5)
elapsed = int(time.monotonic() - started)
after_origin = bench.success_counter(origin_url, origin_key)
origin_delta = int(after_origin - before_origin)
gateway_delta = soak_spend.wait_for_spend(origin_response_ids)
after_metrics = prometheus()
preempt_after = metric(after_metrics, "vllm:num_preemptions_total")
accepted_after = metric(after_metrics, "vllm:spec_decode_num_accepted_tokens_total")
draft_after = metric(after_metrics, "vllm:spec_decode_num_draft_tokens_total")
prefix_queries_after = metric(after_metrics, "vllm:prefix_cache_queries_total")
prefix_hits_after = metric(after_metrics, "vllm:prefix_cache_hits_total")
prefix_queries_delta = prefix_queries_after - prefix_queries_before
prefix_hits_delta = prefix_hits_after - prefix_hits_before
prefix_cache_reuse_ratio = prefix_hits_delta / prefix_queries_delta if prefix_queries_delta > 0 else None
accepted_delta = accepted_after - accepted_before
draft_delta = draft_after - draft_before
speculative_acceptance_ratio = (
    accepted_delta / draft_delta if draft_delta > 0 else None
)
speculative_acceptance_observation = (
    "observed" if draft_delta > 0 else "not-observed"
)
gaps = [node_rows[i][0] - node_rows[i-1][0] for i in range(1, len(node_rows))]
minimum_samples = int(duration / interval * 0.9)
sample_error_limit = max(3, int(duration / interval * 0.01))
accepted = all((
    elapsed >= duration, request_count > 0, failed == 0,
    origin_delta == request_count, gateway_delta == request_count,
    len(sample_errors) <= sample_error_limit, len(node_rows) >= minimum_samples,
    max(idle_gaps or [0.0]) <= 1.0, max(gaps or [0.0]) <= interval * 2.5,
    min(row[1] for row in node_rows) >= 8.0, min(row[2] for row in node_rows) >= 8.0,
    max(row[3] for row in node_rows) == 0, max(row[4] for row in node_rows) == 0,
    max(max(row[5], row[6]) for row in node_rows) <= max_memory_psi_full_avg10,
    prefix_queries_delta > 0, prefix_hits_delta >= 0,
    prefix_cache_reuse_ratio is not None and prefix_cache_reuse_ratio > 0.5,
    max(row[0] for row in metric_rows) <= 1, max(row[1] for row in metric_rows) == 0,
    preempt_after - preempt_before == 0,
    accepted_delta >= 0, draft_delta >= 0,
    accepted_delta <= draft_delta if draft_delta > 0 else accepted_delta == 0,
))
result = {
    "accepted": accepted, "duration_seconds": elapsed, "sample_interval_seconds": interval,
    "request_count": request_count, "origin_completion_delta": origin_delta,
    "gateway_attempt_delta": gateway_delta, "failed_requests": failed,
    "sample_error_count": len(sample_errors),
    "max_idle_gap_seconds": round(max(idle_gaps or [0.0]), 6), "node_samples": len(node_rows),
    "min_head_mem_available_gib": round(min(row[1] for row in node_rows), 3),
    "min_worker_mem_available_gib": round(min(row[2] for row in node_rows), 3),
    "max_memory_psi_full_avg10": round(max(max(row[5], row[6]) for row in node_rows), 6),
    "max_requests_running": max(row[0] for row in metric_rows),
    "max_requests_waiting": max(row[1] for row in metric_rows),
    "preemption_delta": preempt_after - preempt_before,
    "max_rank_restarts": max(max(row[3], row[4]) for row in node_rows),
    "max_node_sample_gap_seconds": round(max(gaps or [0.0]), 6),
    "kv_cache_usage_peak": max(row[2] for row in metric_rows),
    "prefix_cache_queries_delta": prefix_queries_delta,
    "prefix_cache_hits_delta": prefix_hits_delta,
    "prefix_cache_reuse_ratio": round(prefix_cache_reuse_ratio, 6),
    "speculative_accepted_tokens_delta": accepted_delta,
    "speculative_draft_tokens_delta": draft_delta,
    "speculative_acceptance_ratio": (
        round(speculative_acceptance_ratio, 6)
        if speculative_acceptance_ratio is not None else None
    ),
    "speculative_acceptance_observation": speculative_acceptance_observation,
}
target = Path(output); target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); target.chmod(0o600)
if not accepted: raise SystemExit("soak acceptance thresholds failed")
PY
"$SANITIZER" --scan-only <"$SOAK_EVIDENCE" >/dev/null
"$SCRIPT_DIR/litellm/smoke.sh" --all-interfaces
ENV_FILE="$ENV_FILE" "$SCRIPT_DIR/scripts/collect-node-evidence.sh" "$NODE_EVIDENCE_AFTER"
else
  FAILURE_STAGE="resume-capacity"
  for required in     "$QWEN_MANIFEST" "$DIRECT_INTERIM" "$DIRECT_BENCHMARK" "$GATEWAY_BENCHMARK"     "$NODE_EVIDENCE_AFTER" "$SOAK_EVIDENCE" "$DIRECT_SEMANTIC_EVIDENCE"     "$LITELLM_SEMANTIC_EVIDENCE" "$MINEFIELD_EVIDENCE"     "$EXTERNAL_GATEWAY_EVIDENCE" "$CANARY_EVIDENCE" "$FIXTURE"; do
    [ -f "$required" ] || { echo "Missing pre-capacity acceptance evidence: $required" >&2; false; }
  done
  [ ! -e "$FINAL_REPORT" ] || { echo "Refusing to overwrite accepted report." >&2; false; }
  git -C "$ROOT_DIR" diff --quiet
  git -C "$ROOT_DIR" diff --cached --quiet
  ENV_FILE="$ENV_FILE" "$ROOT_DIR/status-deepseek-v4-flash-dspark.sh" --expect running
  "$SCRIPT_DIR/litellm/smoke.sh" --all-interfaces
  [ "$(docker inspect -f '{{.State.Running}}' urbanplan-qwen 2>/dev/null)" = "false" ]
  python3 "$ROOT_DIR/scripts/qwen_manifest.py" verify-live     --manifest "$QWEN_MANIFEST" --max-age-hours 24
  python3 - "$SOAK_EVIDENCE" <<'PY'
import json, sys
soak = json.load(open(sys.argv[1]))
if soak.get("accepted") is not True or soak.get("duration_seconds", 0) < 1800:
    raise SystemExit("resume requires an accepted full-duration soak")
PY
  mapfile -t hermes_files < <(find "$HERMES_RESULTS" -maxdepth 1 -type f -name 'hermes-smoke-*.json' -print | sort)
  [ "${#hermes_files[@]}" -eq 2 ] || {
    echo "Capacity resume requires exactly two Hermes result files." >&2
    false
  }
fi

FAILURE_STAGE="full-context"
reuse_full_context=0
if [ "$MODE" = "resume-capacity" ] && [ -e "$FULL_CONTEXT_EVIDENCE" ]; then
  if python3 - "$FULL_CONTEXT_EVIDENCE" "$CAPACITY_EVIDENCE_IDENTITY" <<'PY'
import json, os, sys, time
path, expected_identity = sys.argv[1:]
report = json.load(open(path))
age = time.time() - os.stat(path).st_mtime
if (age < 0 or age > 24 * 3600 or report.get("gate", {}).get("passed") is not True
        or report.get("evidence_identity") != expected_identity):
    raise SystemExit(1)
PY
  then
    reuse_full_context=1
  else
    attempt=1
    previous_attempt="$FULL_CONTEXT_EVIDENCE.attempt-$attempt"
    while [ -e "$previous_attempt" ]; do
      attempt=$((attempt + 1))
      previous_attempt="$FULL_CONTEXT_EVIDENCE.attempt-$attempt"
    done
    mv "$FULL_CONTEXT_EVIDENCE" "$previous_attempt"
  fi
fi
if [ "$reuse_full_context" -eq 0 ]; then
  python3 "$ROOT_DIR/scripts/probe-full-context.py" \
    --env-file "$ENV_FILE" --key-file "$VLLM_ORIGIN_KEY_FILE" \
    --evidence-identity "$CAPACITY_EVIDENCE_IDENTITY" \
    --output "$FULL_CONTEXT_EVIDENCE"
fi
"$SANITIZER" --scan-only <"$FULL_CONTEXT_EVIDENCE" >/dev/null

FAILURE_STAGE="long-context-decode"
: "${LONG_CONTEXT_DECODE_BASELINE_TPS:?LONG_CONTEXT_DECODE_BASELINE_TPS is required for the 5x regression gate}"
: "${LONG_CONTEXT_DECODE_PROMPT_NONCE:?LONG_CONTEXT_DECODE_PROMPT_NONCE is required to defeat stale prefix-cache reuse}"
reuse_long_context_decode=0
if [ "$MODE" = "resume-capacity" ] && [ -e "$LONG_CONTEXT_DECODE_EVIDENCE" ]; then
  if python3 - "$LONG_CONTEXT_DECODE_EVIDENCE" "$LONG_CONTEXT_DECODE_BASELINE_TPS" "$CAPACITY_EVIDENCE_IDENTITY" "$LONG_CONTEXT_DECODE_PROMPT_NONCE" <<'PY'
import json, math, os, sys, time
path, baseline, expected_identity, expected_nonce = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
report = json.load(open(path))
age = time.time() - os.stat(path).st_mtime
if (age < 0 or age > 24 * 3600 or report.get("gate", {}).get("passed") is not True
        or not math.isclose(report.get("baseline_tps", -1), baseline)
        or report.get("evidence_identity") != expected_identity
        or report.get("prompt_nonce") != expected_nonce):
    raise SystemExit(1)
PY
  then
    reuse_long_context_decode=1
  else
    attempt=1
    previous_attempt="$LONG_CONTEXT_DECODE_EVIDENCE.attempt-$attempt"
    while [ -e "$previous_attempt" ]; do
      attempt=$((attempt + 1))
      previous_attempt="$LONG_CONTEXT_DECODE_EVIDENCE.attempt-$attempt"
    done
    mv "$LONG_CONTEXT_DECODE_EVIDENCE" "$previous_attempt"
  fi
fi
if [ "$reuse_long_context_decode" -eq 0 ]; then
  python3 "$ROOT_DIR/scripts/probe-long-context-decode.py" \
    --env-file "$ENV_FILE" --key-file "$VLLM_ORIGIN_KEY_FILE" \
    --baseline-tps "$LONG_CONTEXT_DECODE_BASELINE_TPS" \
    --evidence-identity "$CAPACITY_EVIDENCE_IDENTITY" \
    --prompt-nonce "$LONG_CONTEXT_DECODE_PROMPT_NONCE" \
    --output "$LONG_CONTEXT_DECODE_EVIDENCE"
fi
"$SANITIZER" --scan-only <"$LONG_CONTEXT_DECODE_EVIDENCE" >/dev/null

FAILURE_STAGE="scheduler"
[ -f "$SCHEDULER_BASELINE" ] || { echo "Missing scheduler baseline: $SCHEDULER_BASELINE" >&2; false; }
python3 "$ROOT_DIR/scripts/benchmark-scheduler.py" \
  --env-file "$ENV_FILE" --key-file "$VLLM_ORIGIN_KEY_FILE" \
  --baseline "$SCHEDULER_BASELINE" --output "$SCHEDULER_EVIDENCE"
"$SANITIZER" --scan-only <"$SCHEDULER_EVIDENCE" >/dev/null

FAILURE_STAGE="report"
[ -f "$ACTIVE_GATEWAY_SNAPSHOT" ] || { echo "Missing active gateway snapshot." >&2; false; }
GATEWAY_SNAPSHOT_EVIDENCE="$RUN_DIR/active-gateway-snapshot.json"
cp "$ACTIVE_GATEWAY_SNAPSHOT" "$GATEWAY_SNAPSHOT_EVIDENCE"
chmod 0600 "$GATEWAY_SNAPSHOT_EVIDENCE"
python3 - "$RUN_DIR" "$FIXTURE" "${hermes_files[0]}" "${hermes_files[1]}" \
  "$ENV_FILE" "$ROOT_DIR" <<'PY' \
  | "$SANITIZER" --schema "$SCHEMA" --output "$FINAL_REPORT"
from datetime import datetime, timezone
import hashlib, json
from pathlib import Path
import subprocess, sys

run_dir, fixture_path, hermes_a, hermes_b, env_path, root = map(Path, sys.argv[1:])
manifest_path = run_dir / "qwen-manifest.json"
direct_path = run_dir / "benchmark-direct.json"
gateway_path = run_dir / "benchmark-litellm.json"
node_path = run_dir / "node-evidence-after-soak.json"
soak_path = run_dir / "soak.json"
interim_path = run_dir / "acceptance.json"
gateway_snapshot = run_dir / "active-gateway-snapshot.json"
direct_semantic_path = run_dir / "semantic-direct-origin.json"
litellm_semantic_path = run_dir / "semantic-private-litellm.json"
minefield_path = run_dir / "minefield.json"
external_gateway_path = run_dir / "external-gateway-auth.json"
canary_path = run_dir / "prompt-reasoning-canary-absence.json"
full_context_path = run_dir / "full-context.json"
long_context_decode_path = run_dir / "long-context-decode.json"
scheduler_path = run_dir / "scheduler-current.json"
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
direct_semantic, litellm_semantic = map(load, (direct_semantic_path, litellm_semantic_path))
minefield, external_gateway, canary = map(load, (minefield_path, external_gateway_path, canary_path))
full_context, long_context_decode, scheduler = map(load, (full_context_path, long_context_decode_path, scheduler_path))
if full_context.get("gate", {}).get("passed") is not True:
    raise SystemExit("full-context evidence is not accepted")
if long_context_decode.get("gate", {}).get("passed") is not True:
    raise SystemExit("long-context decode evidence is not accepted")
if scheduler.get("gate", {}).get("passed") is not True:
    raise SystemExit("scheduler evidence is not accepted")
if direct_semantic != {"profile": "direct-origin", "ready": True, "state": "semantic-ready", "wall_timeout_seconds": 120}:
    raise SystemExit("direct-origin semantic readiness evidence is invalid")
if litellm_semantic != {"profile": "private-litellm", "ready": True, "state": "semantic-ready", "wall_timeout_seconds": 120}:
    raise SystemExit("private LiteLLM semantic readiness evidence is invalid")
hermes = [load(hermes_a), load(hermes_b)]
if not direct["summary"]["accepted"] or not gateway["summary"]["accepted"] or not soak["accepted"]:
    raise SystemExit("performance or soak evidence is not accepted")
if not all(item["rank_running"] and item["rank_restart_count"] == 0 for item in nodes["nodes"]):
    raise SystemExit("rank state failed after soak")
if {item["role"] for item in nodes["nodes"]} != {"head", "worker"}:
    raise SystemExit("both rank roles were not observed")
if max(item["memory_psi_full_avg10"] for item in nodes["nodes"]) != 0.0:
    raise SystemExit("post-soak point-in-time memory PSI must be zero")
if minefield.get("commit") != "2b453b8a69dbaf8dc9d521dc2d6212cdaceb8169":
    raise SystemExit("Minefield evidence is not pinned")
for field in ("executed", "problem", "inconclusive", "unimplemented"):
    if not isinstance(minefield.get(field), int) or minefield[field] < 0:
        raise SystemExit("Minefield coverage counts are invalid")
if external_gateway.get("unauthenticated_status") not in (401, 403) or external_gateway.get("authenticated_generation") is not True:
    raise SystemExit("external gateway authentication boundary failed")
if canary != {"schema_version": 1, "prompt_reasoning_canaries_absent": True, "message_logging_disabled": True}:
    raise SystemExit("prompt/reasoning canary evidence is invalid")
if not all(item["accepted"] and item["shared_state"]["unchanged"] for item in hermes):
    raise SystemExit("Hermes evidence is not accepted")
runtime = env["DSPARK_VLLM_IMAGE"]
if not (runtime.startswith("sha256:") or "@sha256:" in runtime): raise SystemExit("runtime image is not pinned")
import re
capacity_matches = []
started_at = subprocess.check_output(
    ["docker", "inspect", "-f", "{{.State.StartedAt}}", "deepseek-v4-flash-vllm-dspark-1"],
    text=True,
).strip()
process = subprocess.Popen(
    ["docker", "logs", "--since", started_at, "deepseek-v4-flash-vllm-dspark-1"],
    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
assert process.stdout is not None
for line in process.stdout:
    capacity_matches.extend(
        re.findall(r"GPU KV cache size:\s*([0-9,]+)\s*tokens", line)
    )
if process.wait() != 0:
    raise SystemExit("could not read runtime logs for KV token capacity")
if not capacity_matches:
    raise SystemExit("runtime did not report KV token capacity")
reported_capacities = [int(value.replace(",", "")) for value in capacity_matches]
if any(value < 1048576 for value in reported_capacities):
    raise SystemExit("current runtime reported KV token capacity below advertised context")
reported_token_capacity = min(reported_capacities)
hermes_pin_inputs = [root / "deployments/private-smoke/hermes/config.yaml", root / "deployments/private-smoke/hermes/fixtures/transform-input.json", root / "deployments/private-smoke/hermes/fixtures/tool-contract.json"]
expected_hermes_pin = hashlib.sha256(b"".join(path.read_bytes() for path in hermes_pin_inputs)).hexdigest()
if {item["suite_pin_sha256"] for item in hermes} != {expected_hermes_pin}:
    raise SystemExit("Hermes pin changed between functional runs")
config_inputs = [root / "docker-compose.dspark.yml", root / "recipe/runtime-hotfixes.manifest.json", root / "patches/hotfix-encoding-dsv4-issue21.py", root / "patches/hotfix-nvfp4-ds-mla-issue22.py", root / "deployments/private-smoke/litellm/config.yaml", *hermes_pin_inputs, fixture_path]
config_hash = hashlib.sha256(b"".join(path.read_bytes() for path in config_inputs)).hexdigest()
repo_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
runtime_digest = runtime if runtime.startswith("sha256:") else "sha256:" + runtime.rsplit("@sha256:", 1)[1]
pins = {"model_revision": env["DSPARK_MODEL_REVISION"], "runtime_image_digest": runtime_digest, "repo_commit": repo_commit, "config_sha256": config_hash}
pin_hash = hashlib.sha256(json.dumps(pins, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
evidence = [("qwen-manifest", manifest_path), ("semantic-direct-origin", direct_semantic_path), ("semantic-private-litellm", litellm_semantic_path), ("direct", direct_path), ("gateway", gateway_path), ("nodes", node_path), ("hermes-a", hermes_a), ("hermes-b", hermes_b), ("soak", soak_path), ("full-context", full_context_path), ("long-context-decode", long_context_decode_path), ("scheduler", scheduler_path), ("minefield", minefield_path), ("external-gateway-auth", external_gateway_path), ("prompt-reasoning-canary", canary_path), ("public-gateway", gateway_snapshot)]
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
gates = {name: True for name in ("fabric", "artifacts", "qwen_stopped", "direct", "gateway", "hermes", "performance", "soak", "isolation", "sanitization", "public_gateway_unchanged", "minefield", "external_gateway", "prompt_reasoning_canary", "full_context", "long_context_decode", "scheduler")}
report = {
  "schema_version": 2, "run_id": manifest["run_id"], "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "accepted": True,
  "manifest_sha256": digest(manifest_path), "fixture_sha256": digest(fixture_path), "pin_set_sha256": pin_hash, "chain_head_sha256": previous,
  "pins": pins, "gates": gates,
  "functional_runs": [{"artifact_sha256": digest(path), "accepted": item["accepted"], "api_calls": item["usage"]["api_calls"], "gateway_attested": True} for path, item in ((hermes_a, hermes[0]), (hermes_b, hermes[1]))],
  "performance": {"direct": d, "litellm": g, "median_decode_overhead_ratio": g["median_decode_tokens_per_second"] / d["median_decode_tokens_per_second"], "p95_ttft_overhead_seconds": g["p95_ttft_seconds"] - d["p95_ttft_seconds"]},
  "semantic_readiness": {"direct_origin": {"state": direct_semantic["state"], "artifact_sha256": digest(direct_semantic_path)}, "private_litellm": {"state": litellm_semantic["state"], "artifact_sha256": digest(litellm_semantic_path)}},
  "rollout_evidence": {
    "process_readiness": {"head_running": True, "worker_running": True, "restart_count": 0},
    "api_readiness": {"authenticated": True, "model_discovery": True},
    "semantic_readiness": {"direct_origin": True, "private_litellm": True},
    "kv_cache": {"configured_bytes": int(env["KV_CACHE_MEMORY_BYTES"]), "reported_token_capacity": reported_token_capacity},
    "rank_participation": {"world_size": 2, "both_ranks_participated": True},
    "memory": {"min_head_mem_available_gib": soak["min_head_mem_available_gib"], "min_worker_mem_available_gib": soak["min_worker_mem_available_gib"], "max_memory_psi_full_avg10": soak["max_memory_psi_full_avg10"]},
    "prefix_cache": {"queries_delta": soak["prefix_cache_queries_delta"], "hits_delta": soak["prefix_cache_hits_delta"], "reuse_ratio": soak["prefix_cache_reuse_ratio"]},
    "speculative_decode": {"accepted_tokens_delta": soak["speculative_accepted_tokens_delta"], "draft_tokens_delta": soak["speculative_draft_tokens_delta"], "acceptance_ratio": soak["speculative_acceptance_ratio"]},
    "minefield": {key: minefield[key] for key in ("commit", "executed", "problem", "inconclusive", "unimplemented")},
    "external_gateway": {"unauthenticated_status": external_gateway["unauthenticated_status"], "authenticated_generation": external_gateway["authenticated_generation"]},
    "prompt_reasoning_canaries_absent": True, "message_logging_disabled": True,
  },
  "soak": soak, "evidence_chain": chain, "purge_eligible": True,
}
print(json.dumps(report))
PY

# Final canary scan: accepted evidence itself must contain no secret, private
# address, absolute host path, unknown field, or broken evidence_chain.
"$SANITIZER" --schema "$SCHEMA" --input "$FINAL_REPORT" >/dev/null
python3 "$ROOT_DIR/scripts/qwen_manifest.py" verify-report --manifest "$QWEN_MANIFEST" \
  --report "$FINAL_REPORT" --max-age-hours 24 --required-gate fabric --required-gate artifacts \
  --required-gate direct --required-gate hermes \
  --required-gate full_context --required-gate long_context_decode --required-gate scheduler

trap - ERR
echo "Acceptance passed and is purge_eligible: $FINAL_REPORT"
echo "DeepSeek and LiteLLM remain operator-controlled; PostgreSQL alone retains its intentional unless-stopped policy."
