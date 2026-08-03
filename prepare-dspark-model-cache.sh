#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"

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

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
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

run_download() {
  docker run --rm -i \
    -v "${HF_CACHE}:/cache/huggingface" \
    -e HF_HOME=/cache/huggingface \
    -e HF_HUB_OFFLINE=0 \
    -e TRANSFORMERS_OFFLINE=0 \
    -e HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
    -e DSPARK_MODEL="$DSPARK_MODEL" \
    -e DSPARK_MODEL_REVISION="$DSPARK_MODEL_REVISION" \
    -e HF_DOWNLOAD_WORKERS="$HF_DOWNLOAD_WORKERS" \
    --entrypoint "$IMAGE_PYTHON" \
    "$DSPARK_VLLM_IMAGE" \
    -c 'from huggingface_hub import snapshot_download; import os; print(snapshot_download(os.environ["DSPARK_MODEL"], revision=os.environ["DSPARK_MODEL_REVISION"], max_workers=int(os.environ.get("HF_DOWNLOAD_WORKERS", "1"))))'
}

verify_cache() {
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
index = json.loads(index_path.read_text())
needed = sorted(set(index["weight_map"].values()))
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
  rsync -a --partial --safe-links --info=progress2 \
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
