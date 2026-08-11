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

Official downloads default to DSPARK_REVISION=9e165c30… (tested pin). Override
via DSPARK_REVISION in .env.dspark, or clear it to follow tip of main.
Abliterated uses DSPARK_REVISION_ABLITERATED (default: unpinned).

With no flags and a TTY, you are asked which checkpoint to download.
Worker recurse (PREPARE_WORKER=0) never re-asks — it uses the already-chosen model.

Optional / experimental: set PREPARE_VL_SIDECAR_MODEL=1 to also download
VL_SIDECAR_MODEL (Qwen3-VL) on head + worker. Default is 0 (text-only ship).
See README §Experimental: Vision.
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

: "${DSPARK_MODEL:=deepseek-ai/DeepSeek-V4-Flash-DSpark}"
: "${DSPARK_MODEL_REVISION:?DSPARK_MODEL_REVISION must pin the Hugging Face snapshot commit}"
: "${HF_CACHE:=$HOME/.cache/huggingface}"
: "${HF_DOWNLOAD_WORKERS:=4}"
: "${PREPARE_DOWNLOAD:=1}"
: "${DSPARK_VLLM_IMAGE:=vllm-dspark-runtime:dspark-nvfp4-stage-c}"
# Anemll image ships python at /usr/bin/python3 (Stage-C used /opt/env/bin/python).
: "${IMAGE_PYTHON:=/usr/bin/python3}"
# VL sidecar is optional/experimental (TP=2). Default off for the text-only ship;
# set PREPARE_VL_SIDECAR_MODEL=1 (and ENABLE_VL_SIDECAR=1) when experimenting.
: "${VL_SIDECAR_MODEL:=cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit}"
: "${PREPARE_VL_SIDECAR_MODEL:=0}"

resolve_revision() {
  # Export a single DSPARK_REVISION for prepare + compose (may be empty = unpin).
  if [ "${ABLITERATED:-0}" = "1" ]; then
    DSPARK_REVISION="${DSPARK_REVISION_ABLITERATED:-}"
  elif [ -n "${DSPARK_REVISION+x}" ]; then
    # Explicitly set in env (including empty → follow main tip).
    :
  else
    DSPARK_REVISION="$DEFAULT_OFFICIAL_REVISION"
  fi
  export DSPARK_REVISION
}

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
need_cmd python3
mkdir -p "$HF_CACHE"
verify_local_image
resolve_checkpoint
resolve_revision

run_download() {
  local model="${1:?model id required}"
  local revision="${2:-}"
  if [ -n "$revision" ]; then
    echo "prepare: downloading $model @ $revision → $HF_CACHE" >&2
  else
    echo "prepare: downloading $model → $HF_CACHE" >&2
  fi
  docker run --rm -i \
    -v "${HF_CACHE}:/cache/huggingface" \
    -e HF_HOME=/cache/huggingface \
    -e HF_HUB_OFFLINE=0 \
    -e TRANSFORMERS_OFFLINE=0 \
    -e HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
    -e HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-0}" \
    -e DSPARK_MODEL="$DSPARK_MODEL" \
    -e DSPARK_MODEL_REVISION="$DSPARK_MODEL_REVISION" \
    -e HF_DOWNLOAD_WORKERS="$HF_DOWNLOAD_WORKERS" \
    --entrypoint "$IMAGE_PYTHON" \
    "$DSPARK_VLLM_IMAGE" \
    -c 'from huggingface_hub import snapshot_download; import os; print(snapshot_download(os.environ["DSPARK_MODEL"], revision=os.environ["DSPARK_MODEL_REVISION"], max_workers=int(os.environ.get("HF_DOWNLOAD_WORKERS", "1"))))'
}

verify_cache() {
  local model="${1:?model id required}"
  local revision="${2:-}"
  # local_files_only=True; force offline so verify never re-hits the hub.
  docker run --rm -i \
    -v "${HF_CACHE}:/cache/huggingface" \
    -e HF_HOME=/cache/huggingface \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
    -e DSPARK_MODEL="$DSPARK_MODEL" \
    -e DSPARK_MODEL_REVISION="$DSPARK_MODEL_REVISION" \
    --entrypoint "$IMAGE_PYTHON" \
    "$DSPARK_VLLM_IMAGE" \
    - <<'PY'
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

path = Path(snapshot_download(os.environ["DSPARK_MODEL"], revision=os.environ["DSPARK_MODEL_REVISION"], local_files_only=True))
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
print(f"revision={revision or 'default-branch'}")
print(f"safetensor_shards={len(needed)}")
print(f"missing_shards={len(missing)}")
if missing:
    for name in missing[:20]:
        print(f"missing {name}")
    raise SystemExit(1)
PY
}

case "$PREPARE_DOWNLOAD" in
  1)
    echo "prepare: forcing HF online for download (serve can keep HF_HUB_OFFLINE=1)" >&2
    run_download
    ;;
  0) echo "prepare: using transferred cache; skipping online download" >&2 ;;
  *) echo "PREPARE_DOWNLOAD must be 0 or 1" >&2; exit 2 ;;
esac
verify_cache

model_cache_name="models--${DSPARK_MODEL//\//--}"
snapshot_dir="$HF_CACHE/hub/$model_cache_name/snapshots/$DSPARK_MODEL_REVISION"
manifest_file="${ARTIFACT_MANIFEST_FILE:-$SCRIPT_DIR/artifacts/model-manifest.json}"
python3 "$SCRIPT_DIR/scripts/verify-artifact-manifest.py" create \
  --snapshot "$snapshot_dir" \
  --revision "$DSPARK_MODEL_REVISION" \
  --image "$DSPARK_VLLM_IMAGE" \
  --output "$manifest_file"
echo "artifact_manifest=$manifest_file"

if [ "${PREPARE_WORKER:-1}" = "1" ]; then
  : "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE or environment}"
  WORKER_DIR="${WORKER_SCRIPT_DIR:-${WORKER_DIR:-$SCRIPT_DIR}}"
  WORKER_HF_CACHE="${WORKER_HF_CACHE:-$HF_CACHE}"
  need_cmd ssh
  need_cmd scp
  need_cmd rsync
  WORKER_ENV_FILE="${WORKER_ENV_FILE:-$SCRIPT_DIR/.env.dspark.worker}"
  python3 "$SCRIPT_DIR/scripts/generate-node-env.py" \
    --source "$ENV_FILE" \
    --head-output "${HEAD_ENV_FILE:-$SCRIPT_DIR/.env.dspark.head}" \
    --worker-output "$WORKER_ENV_FILE"
  worker_model_cache_dir="$WORKER_HF_CACHE/hub/$model_cache_name"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_HOST" \
    "command -v rsync >/dev/null; mkdir -p '$WORKER_DIR/scripts' '$worker_model_cache_dir'"
  verify_worker_image
  echo "Transferring verified model cache to $WORKER_HOST over the private fabric." >&2
  rsync -a --partial --safe-links --exclude='*.incomplete' --exclude='trees/' --info=progress2 \
    "$HF_CACHE/hub/$model_cache_name/" "$WORKER_HOST:$worker_model_cache_dir/"
  scp "$SCRIPT_DIR/prepare-dspark-model-cache.sh" "${WORKER_HOST}:${WORKER_DIR}/prepare-dspark-model-cache.sh"
  scp "$SCRIPT_DIR/scripts/verify-artifact-manifest.py" "${WORKER_HOST}:${WORKER_DIR}/scripts/verify-artifact-manifest.py"
  scp "$WORKER_ENV_FILE" "${WORKER_HOST}:${WORKER_DIR}/.env.dspark"
  ssh "$WORKER_HOST" "cd '$WORKER_DIR' && chmod +x ./prepare-dspark-model-cache.sh && env -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK -u HEADLESS ENV_FILE='.env.dspark' THIS_NODE_HF_CACHE='$WORKER_HF_CACHE' PREPARE_WORKER=0 PREPARE_DOWNLOAD=0 ./prepare-dspark-model-cache.sh"
  worker_manifest_remote="$WORKER_DIR/artifacts/model-manifest.json"
  worker_manifest_local="${WORKER_ARTIFACT_MANIFEST_FILE:-$SCRIPT_DIR/artifacts/model-manifest.worker.json}"
  scp "${WORKER_HOST}:${worker_manifest_remote}" "$worker_manifest_local"
  python3 "$SCRIPT_DIR/scripts/verify-artifact-manifest.py" compare \
    --left "$manifest_file" --right "$worker_manifest_local"
  echo "worker_artifact_manifest=$worker_manifest_local"
fi
