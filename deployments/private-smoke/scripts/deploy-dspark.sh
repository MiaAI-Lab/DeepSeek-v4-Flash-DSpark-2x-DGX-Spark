#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env.dspark}"
MODE=""
RESUME_RUN_DIR=""
QWEN_STOPPED=0

usage() { echo "Usage: deploy-dspark.sh --prepare-only|--direct-gate|--resume-direct RUN_DIR"; }
while [ "$#" -gt 0 ]; do
  case "$1" in
    --prepare-only) MODE="prepare" ;;
    --direct-gate) MODE="direct" ;;
    --resume-direct) shift; MODE="resume"; RESUME_RUN_DIR="${1:-}" ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
[ -n "$MODE" ] || { usage >&2; exit 2; }
[ -f "$ENV_FILE" ] || { echo "Missing $ENV_FILE" >&2; exit 1; }
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

if [ "$MODE" = "prepare" ]; then
  nccl_commit="da0b547b1b9c6e3b1d4c15578087874522ae3761"
  nccl_image="dspark-nccl-tests:$nccl_commit"
  nccl_context="$(mktemp -d /tmp/dspark-nccl-build.XXXXXX)"
  runtime_export_dir="$(mktemp -d /tmp/dspark-runtime-export.XXXXXX)"
  runtime_archive="$runtime_export_dir/runtime-hotfixes.tar"
  trap 'find "$nccl_context" -depth -delete; find "$runtime_export_dir" -depth -delete' EXIT
  docker build -t "$nccl_image" -f - "$nccl_context" \
    <"$ROOT_DIR/deployments/private-smoke/network/Dockerfile.nccl-tests"
  head_nccl="$(docker image inspect "$nccl_image" -f '{{.Id}}')"
  worker_nccl="$(ssh "$WORKER_HOST" "docker image inspect '$nccl_image' -f '{{.Id}}'" 2>/dev/null || true)"
  # Build once and transfer the exact image. Independent builds can differ in
  # image metadata even when their source and filesystem contents are equal.
  if [ "$head_nccl" != "$worker_nccl" ]; then
    docker save "$nccl_image" | \
      ssh -o BatchMode=yes "$WORKER_HOST" docker load >/dev/null
  fi
  worker_nccl="$(ssh "$WORKER_HOST" "docker image inspect '$nccl_image' -f '{{.Id}}'")"
  [ "$head_nccl" = "$worker_nccl" ] || { echo "NCCL test image IDs differ between ranks." >&2; exit 1; }

  runtime_tag="deepseek-v4-flash-dspark:runtime-hotfixes"
  "$ROOT_DIR/build-anemll-runtime-hotfixes.sh" --tag "$runtime_tag"
  head_runtime="$(docker image inspect "$runtime_tag" -f '{{.Id}}')"
  worker_runtime="$(ssh "$WORKER_HOST" "docker image inspect '$runtime_tag' -f '{{.Id}}'" 2>/dev/null || true)"
  if [ "$head_runtime" != "$worker_runtime" ]; then
    docker save "$runtime_tag" -o "$runtime_archive"
    ssh -o BatchMode=yes "$WORKER_HOST" docker load <"$runtime_archive" >/dev/null
  fi
  worker_runtime="$(ssh "$WORKER_HOST" "docker image inspect '$runtime_tag' -f '{{.Id}}'")"
  [ "$head_runtime" = "$worker_runtime" ] || {
    echo "Runtime hotfix image IDs differ between ranks." >&2
    exit 1
  }
  python3 "$ROOT_DIR/scripts/verify-runtime-hotfixes.py" \
    --repo-root "$ROOT_DIR" --image "$runtime_tag"
  printf 'Set DSPARK_VLLM_IMAGE=%s in the deployment env before preflight.\n' "$head_runtime"
  ENV_FILE="$ENV_FILE" "$ROOT_DIR/prepare-dspark-model-cache.sh"
  exit 0
fi

ACCEPTANCE_ROOT="$ROOT_DIR/artifacts/acceptance"
if [ "$MODE" = "resume" ]; then
  [ -n "$RESUME_RUN_DIR" ] && [ -d "$RESUME_RUN_DIR" ] || {
    echo "--resume-direct requires an existing acceptance run directory." >&2
    exit 1
  }
  acceptance_root="$(realpath "$ACCEPTANCE_ROOT")"
  RUN_DIR="$(realpath "$RESUME_RUN_DIR")"
  case "$RUN_DIR/" in
    "$acceptance_root"/*/) ;;
    *) echo "Resume run must be inside $acceptance_root." >&2; exit 1 ;;
  esac
else
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  RUN_DIR="$ACCEPTANCE_ROOT/$timestamp"
  mkdir -p "$RUN_DIR"
  chmod 0700 "$RUN_DIR"
fi
QWEN_MANIFEST="$RUN_DIR/qwen-manifest.json"
PRESTOP_REPORT="$RUN_DIR/prestop-gates.json"
FINAL_REPORT="$RUN_DIR/acceptance.json"

cleanup_failed_deploy() {
  local status=$?
  trap - ERR
  if [ "$QWEN_STOPPED" -eq 1 ]; then
    echo "Direct deployment failed; cleaning both DSpark ranks. Qwen remains stopped." >&2
    ENV_FILE="$ENV_FILE" "$ROOT_DIR/stop-deepseek-v4-flash-dspark.sh" || true
    ENV_FILE="$ENV_FILE" "$ROOT_DIR/status-deepseek-v4-flash-dspark.sh" --expect stopped || true
  fi
  exit "$status"
}
trap cleanup_failed_deploy ERR

if [ "$MODE" = "resume" ]; then
  [ -f "$QWEN_MANIFEST" ] && [ -f "$PRESTOP_REPORT" ] || {
    echo "Resume run is missing its Qwen manifest or pre-stop gate report." >&2
    exit 1
  }
  [ ! -e "$FINAL_REPORT" ] || { echo "Resume run already has a final report." >&2; exit 1; }
  "$SCRIPT_DIR/stop-qwen.sh" --verify-only
  python3 "$ROOT_DIR/scripts/qwen_manifest.py" verify-live \
    --manifest "$QWEN_MANIFEST" --max-age-hours 24
  python3 "$ROOT_DIR/scripts/qwen_manifest.py" verify-report \
    --manifest "$QWEN_MANIFEST" --report "$PRESTOP_REPORT" --max-age-hours 24 \
    --required-gate fabric --required-gate artifacts
  QWEN_STOPPED=1
  echo "Resuming direct gate from verified stopped-Qwen evidence: $RUN_DIR"
else
  "$SCRIPT_DIR/inventory-qwen.sh" --output "$QWEN_MANIFEST"
  ENV_FILE="$ENV_FILE" "$SCRIPT_DIR/preflight.sh" --all \
    --manifest "$QWEN_MANIFEST" --report "$PRESTOP_REPORT"
  "$SCRIPT_DIR/stop-qwen.sh" --manifest "$QWEN_MANIFEST" --gate-report "$PRESTOP_REPORT"
  QWEN_STOPPED=1
fi

ENV_FILE="$ENV_FILE" "$ROOT_DIR/start-deepseek-v4-flash-dspark.sh"
# The deploy gate is a bounded, non-destructive generation. The comprehensive
# direct suite below remains lifecycle smoke and is never called by status.
ENV_FILE="$ENV_FILE" "$ROOT_DIR/status-deepseek-v4-flash-dspark.sh" --semantic
python3 "$ROOT_DIR/scripts/smoke-openai-compat.py" --profile direct --runs 2 \
  --base-url "http://${VLLM_PROXY_HOST:-172.30.0.1}:${VLLM_PROXY_PORT:-8888}/v1" \
  --key-file "$VLLM_ORIGIN_KEY_FILE" \
  --model "${SERVED_MODEL_NAME:-deepseek-v4-flash-0731}"
python3 "$SCRIPT_DIR/benchmark.py" --layer direct --warmups 3 --samples 20 --concurrency 1 \
  --base-url "http://${VLLM_PROXY_HOST:-172.30.0.1}:${VLLM_PROXY_PORT:-8888}/v1" \
  --key-file "$VLLM_ORIGIN_KEY_FILE" --model "$SERVED_MODEL_NAME" \
  --output "$RUN_DIR/benchmark-direct.json"
ENV_FILE="$ENV_FILE" "$SCRIPT_DIR/collect-node-evidence.sh" "$RUN_DIR/node-evidence.json"

python3 - "$QWEN_MANIFEST" "$PRESTOP_REPORT" "$FINAL_REPORT" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
_, prior_path, output_path = map(Path, sys.argv[1:])
report = json.loads(prior_path.read_text())
report["created_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
report["gates"]["direct"] = True
report["accepted"] = False
report["state"] = "direct-passed-awaiting-gateway-and-hermes"
output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
output_path.chmod(0o600)
PY
trap - ERR
echo "Direct DSpark gate passed. Evidence: $RUN_DIR"
