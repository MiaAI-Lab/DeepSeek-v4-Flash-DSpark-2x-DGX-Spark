#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"

usage() {
  cat <<'EOF'
Usage: ./prepare-dspark-model-cache.sh [--official | --abliterated] [--yes]

Downloads DeepSeek-V4-Flash-Vision-Exp weights (official or abliterated) into HF_CACHE
on this node, then onto the worker (default DSPARK_WORKER_HF_NFS=0). Set
DSPARK_WORKER_HF_NFS=1 to skip the worker copy; start then exports this cache over NFS.

  --official      Download deepseek-ai/DeepSeek-V4-Flash-Vision-Exp (sets ABLITERATED=0)
  --abliterated   Download Keys abliterated weights (sets ABLITERATED=1)
  --yes           Non-interactive: use ABLITERATED from .env.dspark (or 0)

Official downloads default to DSPARK_REVISION=86f746b3… (Vision-Exp pin). Override
via DSPARK_REVISION in .env.dspark, or clear it to follow tip of main.
Abliterated uses DSPARK_REVISION_ABLITERATED (default: unpinned). The Hub id is
gated (auto-approve after RESPONSIBLE_USE.md). Prefer overlaying the 26 edited
shards onto the already-cached official Vision-Exp dump
(scripts/overlay-vision-exp-ablit-cache.py) instead of re-fetching 157 GiB.

Hub auth is automatic when HF_TOKEN or HUGGING_FACE_HUB_TOKEN is exported in
the calling shell (it wins over .env.dspark). Otherwise prepare uses the env
file, then ~/.cache/huggingface/token from huggingface-cli login.

With no flags and a TTY, you are asked which checkpoint to download.
Worker recurse (PREPARE_WORKER=0) never re-asks — it uses the already-chosen model.
EOF
}

CLI_CHOICE=""
ASSUME_YES=0
# Snapshot Hub credentials from the calling shell before .env.dspark is
# sourced — an empty HF_TOKEN= in the env file must not wipe an exported token.
_SHELL_HF_TOKEN="${HF_TOKEN-}"
_SHELL_HF_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN-}"
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

DSPARK_MODEL_OFFICIAL="${DSPARK_MODEL_OFFICIAL:-deepseek-ai/DeepSeek-V4-Flash-Vision-Exp}"
DSPARK_MODEL_ABLITERATED="${DSPARK_MODEL_ABLITERATED:-drowzeys/keys-DeepSeekV4Flash-Vision-EXP-ablit}"
# Official tested pin. Override with DSPARK_REVISION=<sha> or clear with
# DSPARK_REVISION= to follow tip of main. Abliterated uses DSPARK_REVISION_ABLITERATED
# (default empty = tip of that repo).
DEFAULT_OFFICIAL_REVISION="86f746b36186f0e567729a5c06a8c918caba82a9"
: "${HF_CACHE:=$HOME/.cache/huggingface}"
: "${HF_DOWNLOAD_WORKERS:=1}"
: "${DSPARK_VLLM_IMAGE:=vllm-dspark-runtime:dspark-nvfp4-stage-c}"
# Anemll image ships python at /usr/bin/python3 (Stage-C used /opt/env/bin/python).
: "${IMAGE_PYTHON:=/usr/bin/python3}"

# Hugging Face token. Never printed. docker inspect on the short-lived
# download container can still show it, same as any -e secret.
# Precedence: exported shell env → .env.dspark → huggingface-cli token file.
_HF_TOKEN_SRC=""
if [ -n "${_SHELL_HF_TOKEN}" ]; then
  HF_TOKEN="${_SHELL_HF_TOKEN}"
  _HF_TOKEN_SRC="shell HF_TOKEN"
elif [ -n "${_SHELL_HF_HUB_TOKEN}" ]; then
  HF_TOKEN="${_SHELL_HF_HUB_TOKEN}"
  _HF_TOKEN_SRC="shell HUGGING_FACE_HUB_TOKEN"
else
  HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
  if [ -n "$HF_TOKEN" ]; then
    _HF_TOKEN_SRC=".env.dspark"
  fi
fi
if [ -z "$HF_TOKEN" ]; then
  for _hf_token_file in \
    "${HF_CACHE}/token" \
    "${HOME}/.cache/huggingface/token"
  do
    if [ -f "$_hf_token_file" ]; then
      HF_TOKEN="$(tr -d '[:space:]' < "$_hf_token_file")"
      if [ -n "$HF_TOKEN" ]; then
        _HF_TOKEN_SRC="$_hf_token_file"
        break
      fi
    fi
  done
  unset _hf_token_file
fi
unset _SHELL_HF_TOKEN _SHELL_HF_HUB_TOKEN
case "${HF_TOKEN}" in
  *$'\n'*|*$'\r'*)
    echo "error: HF_TOKEN must be a single line" >&2
    exit 2
    ;;
esac
DOCKER_HF_TOKEN_ARGS=()
WORKER_HF_TOKEN_ENV=""
if [ -n "$HF_TOKEN" ]; then
  DOCKER_HF_TOKEN_ARGS=(-e "HF_TOKEN=${HF_TOKEN}" -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}")
  WORKER_HF_TOKEN_ENV="HF_TOKEN=$(printf '%q' "$HF_TOKEN") HUGGING_FACE_HUB_TOKEN=$(printf '%q' "$HF_TOKEN")"
  echo "prepare: Hugging Face token: set (from ${_HF_TOKEN_SRC}; redacted)" >&2
else
  echo "prepare: Hugging Face token: unset (anonymous Hub; slower / rate-limited)" >&2
fi
unset _HF_TOKEN_SRC

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
    echo "Which DeepSeek-V4-Flash checkpoint should be downloaded?" >&2
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
resolve_revision

# Serve profiles keep HF_HUB_OFFLINE=1; download must ignore that for this run only.
echo "prepare: forcing HF online for download (serve can keep HF_HUB_OFFLINE=1)" >&2
if [ -n "${DSPARK_REVISION:-}" ]; then
  echo "prepare: ABLITERATED=${ABLITERATED:-0} → $DSPARK_MODEL @ $DSPARK_REVISION" >&2
else
  echo "prepare: ABLITERATED=${ABLITERATED:-0} → $DSPARK_MODEL @ (tip of default branch)" >&2
fi

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
    -e HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
    -e HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}" \
    -e DSPARK_MODEL="$model" \
    -e DSPARK_REVISION="$revision" \
    -e HF_DOWNLOAD_WORKERS="$HF_DOWNLOAD_WORKERS" \
    "${DOCKER_HF_TOKEN_ARGS[@]}" \
    --entrypoint "$IMAGE_PYTHON" \
    "$DSPARK_VLLM_IMAGE" \
    - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = os.environ["DSPARK_MODEL"]
revision = (os.environ.get("DSPARK_REVISION") or "").strip() or None
kwargs = {
    "repo_id": repo_id,
    "max_workers": int(os.environ.get("HF_DOWNLOAD_WORKERS", "1")),
}
token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
if token:
    kwargs["token"] = token
if revision:
    kwargs["revision"] = revision
path = Path(snapshot_download(**kwargs))
print(path)
# Keep offline serve of the bare repo id working: pin refs/main to this commit
# when we downloaded a specific revision (issue #19).
if revision:
    refs_dir = path.parent.parent / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "main").write_text(f"{revision}\n", encoding="utf-8")
    print(f"refs/main -> {revision}")
PY
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
    -e HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
    -e HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}" \
    -e DSPARK_MODEL="$model" \
    -e DSPARK_REVISION="$revision" \
    --entrypoint "$IMAGE_PYTHON" \
    "$DSPARK_VLLM_IMAGE" \
    - <<'PY'
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = os.environ["DSPARK_MODEL"]
revision = (os.environ.get("DSPARK_REVISION") or "").strip() or None
kwargs = {"repo_id": repo_id, "local_files_only": True}
if revision:
    kwargs["revision"] = revision
path = Path(snapshot_download(**kwargs))
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

run_download "$DSPARK_MODEL" "${DSPARK_REVISION:-}"
verify_cache "$DSPARK_MODEL" "${DSPARK_REVISION:-}"

if [ "${PREPARE_WORKER:-1}" = "1" ]; then
  : "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE or environment}"
  if [ "${DSPARK_WORKER_HF_NFS:-0}" = "1" ]; then
    echo "prepare: DSPARK_WORKER_HF_NFS=1 — checkpoint stays on this node ($HF_CACHE)."
    echo "prepare: worker will mount it over NFS at start (no second Hub download)."
    echo "prepare: set DSPARK_WORKER_HF_NFS=0 to copy weights onto the worker instead."
  else
    WORKER_DIR="${WORKER_SCRIPT_DIR:-${WORKER_DIR:-$SCRIPT_DIR}}"
    WORKER_HF_CACHE="${WORKER_HF_CACHE:-$HF_CACHE}"
    need_cmd ssh
    need_cmd scp
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_HOST" "mkdir -p '$WORKER_DIR' '$WORKER_HF_CACHE'"
    verify_worker_image
    scp "$SCRIPT_DIR/prepare-dspark-model-cache.sh" "${WORKER_HOST}:${WORKER_DIR}/prepare-dspark-model-cache.sh"
    scp "$ENV_FILE" "${WORKER_HOST}:${WORKER_DIR}/.env.dspark"
    ssh "$WORKER_HOST" "cd '$WORKER_DIR' && chmod +x ./prepare-dspark-model-cache.sh && env -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK -u HEADLESS ENV_FILE='.env.dspark' THIS_NODE_HF_CACHE='$WORKER_HF_CACHE' PREPARE_WORKER=0 ABLITERATED='$ABLITERATED' DSPARK_REVISION='${DSPARK_REVISION:-}' DSPARK_REVISION_ABLITERATED='${DSPARK_REVISION_ABLITERATED:-}' ${WORKER_HF_TOKEN_ENV} ./prepare-dspark-model-cache.sh --yes"
    if [ -n "${WORKER2_HOST:-}" ]; then
      WORKER2_DIR="${WORKER2_SCRIPT_DIR:-${WORKER2_DIR:-$WORKER_DIR}}"
      WORKER2_HF_CACHE="${WORKER2_HF_CACHE:-$WORKER_HF_CACHE}"
      echo "prepare: also downloading onto worker2 $WORKER2_HOST → $WORKER2_HF_CACHE"
      ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER2_HOST" "docker image inspect '$DSPARK_VLLM_IMAGE' >/dev/null" || {
        echo "Missing worker2 Docker image $DSPARK_VLLM_IMAGE." >&2
        exit 1
      }
      ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER2_HOST" "mkdir -p '$WORKER2_DIR' '$WORKER2_HF_CACHE'"
      scp "$SCRIPT_DIR/prepare-dspark-model-cache.sh" "${WORKER2_HOST}:${WORKER2_DIR}/prepare-dspark-model-cache.sh"
      scp "$ENV_FILE" "${WORKER2_HOST}:${WORKER2_DIR}/.env.dspark"
      ssh "$WORKER2_HOST" "cd '$WORKER2_DIR' && chmod +x ./prepare-dspark-model-cache.sh && env -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK -u HEADLESS ENV_FILE='.env.dspark' THIS_NODE_HF_CACHE='$WORKER2_HF_CACHE' PREPARE_WORKER=0 ABLITERATED='$ABLITERATED' DSPARK_REVISION='${DSPARK_REVISION:-}' DSPARK_REVISION_ABLITERATED='${DSPARK_REVISION_ABLITERATED:-}' ${WORKER_HF_TOKEN_ENV} ./prepare-dspark-model-cache.sh --yes"
    fi
  fi
fi
