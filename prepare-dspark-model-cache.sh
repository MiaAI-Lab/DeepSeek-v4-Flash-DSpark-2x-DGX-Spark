#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"

usage() {
  cat <<'EOF'
Usage: ./prepare-dspark-model-cache.sh [--official | --abliterated] [--yes]

Downloads DeepSeek-V4-Flash-0731 weights (official or abliterated) into HF_CACHE
on this node, then optionally on the worker.

  --official      Download deepseek-ai/DeepSeek-V4-Flash-0731 (sets ABLITERATED=0)
  --abliterated   Download Keys abliterated weights (sets ABLITERATED=1)
  --yes           Non-interactive: use ABLITERATED from .env.dspark (or 0)

With no flags and a TTY, you are asked which checkpoint to download.
Worker recurse (PREPARE_WORKER=0) never re-asks — it uses the already-chosen model.
EOF
}

CLI_CHOICE=""
ASSUME_YES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --official) CLI_CHOICE=0 ;;
    --abliterated) CLI_CHOICE=1 ;;
    --yes|-y) ASSUME_YES=1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

if [ -n "${THIS_NODE_HF_CACHE:-}" ]; then
  HF_CACHE="$THIS_NODE_HF_CACHE"
fi

DSPARK_MODEL_OFFICIAL="${DSPARK_MODEL_OFFICIAL:-deepseek-ai/DeepSeek-V4-Flash-0731}"
DSPARK_MODEL_ABLITERATED="${DSPARK_MODEL_ABLITERATED:-drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32}"
: "${HF_CACHE:=$HOME/.cache/huggingface}"
: "${HF_DOWNLOAD_WORKERS:=1}"
: "${DSPARK_VLLM_IMAGE:=vllm-dspark-runtime:dspark-nvfp4-stage-c}"
# Anemll image ships python at /usr/bin/python3 (Stage-C used /opt/env/bin/python).
: "${IMAGE_PYTHON:=/usr/bin/python3}"
# VL sidecar is TP=2 across head+worker; cache on both nodes for offline serve.
: "${VL_SIDECAR_MODEL:=cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit}"
: "${PREPARE_VL_SIDECAR_MODEL:=1}"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

upsert_env_kv() {
  local file="$1" key="$2" val="$3"
  if [ ! -f "$file" ]; then
    return 0
  fi
  if grep -qE "^${key}=" "$file"; then
    sed -i -E "s|^${key}=.*|${key}=${val}|" "$file"
  else
    printf '\n%s=%s\n' "$key" "$val" >> "$file"
  fi
}

resolve_checkpoint() {
  # Worker recurse: trust env already set by head (no prompt).
  if [ "${PREPARE_WORKER:-1}" = "0" ]; then
    if [ "${ABLITERATED:-0}" = "1" ]; then
      DSPARK_MODEL="$DSPARK_MODEL_ABLITERATED"
    else
      DSPARK_MODEL="$DSPARK_MODEL_OFFICIAL"
    fi
    return 0
  fi

  local choice=""
  if [ -n "$CLI_CHOICE" ]; then
    choice="$CLI_CHOICE"
  elif [ "$ASSUME_YES" = "1" ]; then
    choice="${ABLITERATED:-0}"
  elif [ -t 0 ] && [ -t 1 ]; then
    local default="${ABLITERATED:-0}"
    echo "" >&2
    echo "Which DeepSeek-V4-Flash-0731 checkpoint should be downloaded?" >&2
    echo "  [0] Official  — $DSPARK_MODEL_OFFICIAL" >&2
    echo "  [1] Abliterated — $DSPARK_MODEL_ABLITERATED" >&2
    echo "Current .env.dspark ABLITERATED=${ABLITERATED:-unset} (default answer: $default)" >&2
    while true; do
      printf "Enter 0 or 1 [%s]: " "$default" >&2
      read -r answer || true
      answer="${answer:-$default}"
      case "$answer" in
        0|1) choice="$answer"; break ;;
        *) echo "Please enter 0 (official) or 1 (abliterated)." >&2 ;;
      esac
    done
  else
    choice="${ABLITERATED:-0}"
    echo "prepare: non-interactive; using ABLITERATED=$choice from env" >&2
  fi

  ABLITERATED="$choice"
  if [ "$ABLITERATED" = "1" ]; then
    DSPARK_MODEL="$DSPARK_MODEL_ABLITERATED"
  else
    DSPARK_MODEL="$DSPARK_MODEL_OFFICIAL"
  fi
  export ABLITERATED DSPARK_MODEL

  if [ -f "$ENV_FILE" ]; then
    upsert_env_kv "$ENV_FILE" "ABLITERATED" "$ABLITERATED"
    echo "prepare: wrote ABLITERATED=$ABLITERATED into $ENV_FILE" >&2
  fi
}

verify_local_image() {
  docker image inspect "$DSPARK_VLLM_IMAGE" >/dev/null || {
    echo "Missing local Docker image $DSPARK_VLLM_IMAGE. Run ./build-dspark-vllm-runtime.sh first." >&2
    exit 1
  }
}

verify_worker_image() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_HOST" "docker image inspect '$DSPARK_VLLM_IMAGE' >/dev/null" || {
    echo "Missing worker Docker image $DSPARK_VLLM_IMAGE. Run ./build-dspark-vllm-runtime.sh first." >&2
    exit 1
  }
}

need_cmd docker
mkdir -p "$HF_CACHE"
verify_local_image
resolve_checkpoint

# Serve profiles keep HF_HUB_OFFLINE=1; download must ignore that for this run only.
echo "prepare: forcing HF online for download (serve can keep HF_HUB_OFFLINE=1)" >&2
echo "prepare: ABLITERATED=${ABLITERATED:-0} → $DSPARK_MODEL" >&2

run_download() {
  local model="${1:?model id required}"
  echo "prepare: downloading $model → $HF_CACHE" >&2
  docker run --rm -i \
    -v "${HF_CACHE}:/cache/huggingface" \
    -e HF_HOME=/cache/huggingface \
    -e HF_HUB_OFFLINE=0 \
    -e TRANSFORMERS_OFFLINE=0 \
    -e HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
    -e DSPARK_MODEL="$model" \
    -e HF_DOWNLOAD_WORKERS="$HF_DOWNLOAD_WORKERS" \
    --entrypoint "$IMAGE_PYTHON" \
    "$DSPARK_VLLM_IMAGE" \
    -c 'from huggingface_hub import snapshot_download; import os; print(snapshot_download(os.environ["DSPARK_MODEL"], max_workers=int(os.environ.get("HF_DOWNLOAD_WORKERS", "1"))))'
}

verify_cache() {
  local model="${1:?model id required}"
  # local_files_only=True; force offline so verify never re-hits the hub.
  docker run --rm -i \
    -v "${HF_CACHE}:/cache/huggingface" \
    -e HF_HOME=/cache/huggingface \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
    -e DSPARK_MODEL="$model" \
    --entrypoint "$IMAGE_PYTHON" \
    "$DSPARK_VLLM_IMAGE" \
    - <<'PY'
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

path = Path(snapshot_download(os.environ["DSPARK_MODEL"], local_files_only=True))
index_path = path / "model.safetensors.index.json"
if index_path.is_file():
    index = json.loads(index_path.read_text())
    needed = sorted(set(index["weight_map"].values()))
else:
    needed = sorted(p.name for p in path.glob("*.safetensors"))
    if not needed:
        raise SystemExit(f"no safetensors found under {path}")
missing = [name for name in needed if not (path / name).exists()]
print(f"snapshot={path}")
print(f"safetensor_shards={len(needed)}")
print(f"missing_shards={len(missing)}")
if missing:
    for name in missing[:20]:
        print(f"missing {name}")
    raise SystemExit(1)
PY
}

run_download "$DSPARK_MODEL"
verify_cache "$DSPARK_MODEL"

# VL sidecar TP=2 needs weights on every rank (head prepare + worker recurse).
if [ "${PREPARE_VL_SIDECAR_MODEL:-1}" = "1" ]; then
  echo "prepare: VL sidecar model ($VL_SIDECAR_MODEL)" >&2
  run_download "$VL_SIDECAR_MODEL"
  verify_cache "$VL_SIDECAR_MODEL"
fi

if [ "${PREPARE_WORKER:-1}" = "1" ]; then
  : "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE or environment}"
  WORKER_DIR="${WORKER_SCRIPT_DIR:-${WORKER_DIR:-$SCRIPT_DIR}}"
  WORKER_HF_CACHE="${WORKER_HF_CACHE:-$HF_CACHE}"
  need_cmd ssh
  need_cmd scp
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_HOST" "mkdir -p '$WORKER_DIR' '$WORKER_HF_CACHE'"
  verify_worker_image
  scp "$SCRIPT_DIR/prepare-dspark-model-cache.sh" "${WORKER_HOST}:${WORKER_DIR}/prepare-dspark-model-cache.sh"
  scp "$ENV_FILE" "${WORKER_HOST}:${WORKER_DIR}/.env.dspark"
  ssh "$WORKER_HOST" "cd '$WORKER_DIR' && chmod +x ./prepare-dspark-model-cache.sh && env -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK -u HEADLESS ENV_FILE='.env.dspark' THIS_NODE_HF_CACHE='$WORKER_HF_CACHE' PREPARE_WORKER=0 ABLITERATED='$ABLITERATED' ./prepare-dspark-model-cache.sh --yes"
fi
