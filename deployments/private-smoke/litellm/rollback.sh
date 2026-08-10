#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="${LITELLM_ENV_FILE:-$SCRIPT_DIR/.env}"
COMPOSE_FILE="${LITELLM_COMPOSE_FILE:-$SCRIPT_DIR/docker-compose.yml}"
PROJECT="${LITELLM_PROJECT:-dspark-private-litellm}"
[ -f "$ENV_FILE" ] || { echo "Missing LiteLLM env file: $ENV_FILE" >&2; exit 1; }
reload_litellm_env() {
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
}
reload_litellm_env
COMPOSE=(docker compose -p "$PROJECT" --env-file "$ENV_FILE" -f "$COMPOSE_FILE")
DB_USER="${POSTGRES_USER:-litellm_smoke}"
DB_NAME="${POSTGRES_DB:-litellm_smoke}"
POSTGRES_IMAGE="postgres@sha256:b797483593b82cbea9a7ee41c88f324a90d10d9c2504d40e755d91c75456366d"

usage() {
  cat >&2 <<EOF
Usage:
  $0 snapshot --bundle OUTSIDE_REPO [--receipt FILE] [--critical FILE ...]
  $0 verify --bundle OUTSIDE_REPO [--receipt SANITIZED_FILE]
  $0 migrate --bundle OUTSIDE_REPO --key-file MODE_0600_KEY
  $0 restore --bundle OUTSIDE_REPO --key-file MODE_0600_KEY
  $0 verify-existing-key --key-file MODE_0600_KEY
The snapshot captures exact critical inputs plus a validated PostgreSQL custom
logical dump. restore is the reverse path and recovers DeepSeek worker-first.
EOF
}

mode_of() { stat -c '%a' "$1" 2>/dev/null || stat -f '%Lp' "$1"; }
require_0600() {
  [ ! -L "$1" ] && [ -f "$1" ] && [ "$(mode_of "$1")" = "600" ] || {
    echo "Required regular mode-0600 file: $1" >&2; return 1;
  }
}
require_bundle() {
  [ ! -L "$1" ] && [ -d "$1" ] && [ "$(mode_of "$1")" = "700" ] || {
    echo "Rollback bundle must be a real mode-0700 directory: $1" >&2; return 1;
  }
}
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

compose_exec() { "${COMPOSE[@]}" exec -T postgres "$@"; }
validate_dump() {
  require_0600 "$1"
  docker run --rm -i --network none --read-only --cap-drop ALL \
    --security-opt no-new-privileges:true --entrypoint pg_restore \
    "$POSTGRES_IMAGE" --list <"$1" >/dev/null
}
wait_postgres() {
  for _ in $(seq 1 60); do
    compose_exec pg_isready -U "$DB_USER" -d "$DB_NAME" >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "PostgreSQL did not become ready" >&2
  return 1
}
wait_litellm() {
  local container health
  for _ in $(seq 1 90); do
    container="$("${COMPOSE[@]}" ps -q litellm 2>/dev/null || true)"
    if [ -n "$container" ]; then
      health="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$container" 2>/dev/null || true)"
      [ "$health" = "healthy" ] && return 0
    fi
    sleep 2
  done
  echo "LiteLLM did not become healthy" >&2
  return 1
}

restore_database() {
  local dump="$1"
  validate_dump "$dump"
  compose_exec dropdb --if-exists --force -U "$DB_USER" "$DB_NAME"
  compose_exec createdb -U "$DB_USER" "$DB_NAME"
  compose_exec pg_restore --no-owner --no-acl -U "$DB_USER" -d "$DB_NAME" <"$dump"
}

verify_bundle() {
  local bundle="$1" line expected relative actual
  require_bundle "$bundle"
  for required in SHA256SUMS RESTORE.tsv postgres.dump; do require_0600 "$bundle/$required"; done
  while IFS='  ' read -r expected relative; do
    [ -n "$expected" ] && [ -n "$relative" ] || { echo "Malformed SHA256SUMS" >&2; return 1; }
    case "$relative" in /*|*..*) echo "Unsafe manifest path: $relative" >&2; return 1;; esac
    require_0600 "$bundle/$relative"
    actual="$(sha256_file "$bundle/$relative")"
    [ "$actual" = "$expected" ] || { echo "Manifest mismatch: $relative" >&2; return 1; }
  done <"$bundle/SHA256SUMS"
  # On rollout hosts sha256sum -c is an independent manifest parser/check.
  if command -v sha256sum >/dev/null 2>&1; then (cd "$bundle" && sha256sum -c SHA256SUMS >/dev/null); fi
  validate_dump "$bundle/postgres.dump"
  while IFS=$'\t' read -r relative destination original_mode; do
    [ -n "$relative" ] && [ -n "$destination" ] || { echo "Malformed RESTORE.tsv" >&2; return 1; }
    case "$relative" in /*|*..*) echo "Unsafe restore member" >&2; return 1;; esac
    case "$destination" in /*) :;; *) echo "Restore destination is not absolute" >&2; return 1;; esac
    [[ "$original_mode" =~ ^[0-7]{3,4}$ ]] || { echo "Invalid original mode" >&2; return 1; }
    require_0600 "$bundle/$relative"
  done <"$bundle/RESTORE.tsv"
  echo "rollback-bundle=verified"
}

snapshot() {
  local bundle="$1" receipt="$2"; shift 2
  case "$(cd "$(dirname "$bundle")" && pwd)/$(basename "$bundle")" in "$ROOT_DIR"/*)
    echo "Rollback bundle must be outside the repository" >&2; return 1;; esac
  mkdir -m 0700 "$bundle" 2>/dev/null || {
    echo "Refusing to overwrite rollback bundle" >&2
    return 1
  }
  mkdir -m 0700 "$bundle/inputs"
  local mapping="$bundle/RESTORE.tsv" index=0 source relative mode
  : >"$mapping"; chmod 0600 "$mapping"
  if [ "$#" -eq 0 ]; then
    set -- "$ROOT_DIR/.env.dspark" "$ROOT_DIR/.env.dspark.head" "$ROOT_DIR/.env.dspark.worker" \
      "$ROOT_DIR/docker-compose.dspark.yml" "$ENV_FILE" "$COMPOSE_FILE" "$SCRIPT_DIR/config.yaml" \
      "$LITELLM_VIRTUAL_KEY_FILE"
  fi
  for source in "$@"; do
    [ ! -L "$source" ] && [ -f "$source" ] || { echo "Missing/unsafe critical input: $source" >&2; return 1; }
    case "$source" in *$'\n'*|*$'\t'*) echo "Unsafe critical input name" >&2; return 1;; esac
    index=$((index + 1)); relative="inputs/$(printf '%03d' "$index")-$(basename "$source")"
    mode="$(mode_of "$source")"
    cp -p "$source" "$bundle/$relative"; chmod 0600 "$bundle/$relative"
    printf '%s\t%s\t%s\n' "$relative" "$(cd "$(dirname "$source")" && pwd)/$(basename "$source")" "$mode" >>"$mapping"
  done
  compose_exec pg_dump -U "$DB_USER" -d "$DB_NAME" -Fc --no-owner --no-acl >"$bundle/postgres.dump"
  chmod 0600 "$bundle/postgres.dump"
  validate_dump "$bundle/postgres.dump"
  : >"$bundle/SHA256SUMS"
  while IFS= read -r relative; do
    printf '%s  %s\n' "$(sha256_file "$bundle/$relative")" "$relative" >>"$bundle/SHA256SUMS"
  done < <(cd "$bundle" && find inputs -type f -print | LC_ALL=C sort; printf '%s\n' RESTORE.tsv postgres.dump)
  chmod 0600 "$bundle/SHA256SUMS"
  verify_bundle "$bundle" >/dev/null
  if [ -n "$receipt" ]; then
    python3 - "$bundle" <<'PY' | "$SCRIPT_DIR/../scripts/sanitize-evidence.py" --scan-only --output "$receipt"
import hashlib, json
from pathlib import Path
import sys
bundle = Path(sys.argv[1])
files = [line.split("  ", 1)[1].strip() for line in (bundle / "SHA256SUMS").read_text().splitlines()]
print(json.dumps({"schema_version": 1, "bundle_manifest_sha256": hashlib.sha256((bundle / "SHA256SUMS").read_bytes()).hexdigest(), "critical_file_count": sum(name.startswith("inputs/") for name in files), "database_dump_validated": True, "all_bundle_files_mode_0600": True}))
PY
    chmod 0600 "$receipt"
  fi
  echo "rollback-bundle=created"
}

verify_existing_key() {
  local key_file="$1"
  require_0600 "$key_file"
  python3 - "$key_file" "${LITELLM_BASE_URL:-http://${HEAD_TAILSCALE_IP:?}:4001/v1}" <<'PY'
import json
from pathlib import Path
import sys
import urllib.request
key = Path(sys.argv[1]).read_text().strip()
request = urllib.request.Request(sys.argv[2].rstrip("/") + "/models", headers={"Authorization": "Bearer " + key})
with urllib.request.urlopen(request, timeout=30) as response:
    payload = json.load(response)
if response.status != 200 or not payload.get("data"):
    raise SystemExit("existing key authentication failed")
print("existing-key=authenticated")
PY
}

restore_files() {
  local bundle="$1" relative destination original_mode
  while IFS=$'\t' read -r relative destination original_mode; do
    [ ! -L "$destination" ] || { echo "Refusing symlink restore destination" >&2; return 1; }
    [ -d "$(dirname "$destination")" ] || { echo "Missing restore parent" >&2; return 1; }
    install -m "$original_mode" "$bundle/$relative" "$destination"
    [ "$(sha256_file "$destination")" = "$(sha256_file "$bundle/$relative")" ] || {
      echo "Restored input hash mismatch" >&2; return 1;
    }
  done <"$bundle/RESTORE.tsv"
}

migrate() {
  local bundle="$1" key_file="$2"
  verify_bundle "$bundle" >/dev/null
  require_0600 "$key_file"
  "${COMPOSE[@]}" down --remove-orphans
  # Compose creates/attaches dspark-private-litellm-pgdata; this is the
  # fail-closed tmpfs->volume path preserving the existing key database.
  "${COMPOSE[@]}" up -d postgres
  wait_postgres
  restore_database "$bundle/postgres.dump"
  "${COMPOSE[@]}" up -d litellm
  wait_litellm
  verify_existing_key "$key_file"
  echo "migration=complete"
}

restore_all() {
  local bundle="$1" key_file="$2"
  verify_bundle "$bundle" >/dev/null
  require_0600 "$key_file"
  # Stop head then worker, restore exact inputs/database, then use the existing
  # worker-first start lifecycle. Never attempt a head-first recovery.
  ENV_FILE="${DSPARK_ENV_FILE:-$ROOT_DIR/.env.dspark}" "$ROOT_DIR/stop-deepseek-v4-flash-dspark.sh"
  "${COMPOSE[@]}" down --remove-orphans
  restore_files "$bundle"
  reload_litellm_env
  DB_USER="${POSTGRES_USER:-litellm_smoke}"
  DB_NAME="${POSTGRES_DB:-litellm_smoke}"
  "${COMPOSE[@]}" up -d postgres
  wait_postgres
  restore_database "$bundle/postgres.dump"
  ENV_FILE="${DSPARK_ENV_FILE:-$ROOT_DIR/.env.dspark}" "$ROOT_DIR/start-deepseek-v4-flash-dspark.sh"
  "${COMPOSE[@]}" up -d litellm
  wait_litellm
  verify_existing_key "$key_file"
  echo "rollback=restored-worker-first"
}

command="${1:-}"; shift || true
bundle=""; receipt=""; key_file=""; critical=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --bundle) bundle="${2:?missing bundle}"; shift 2;;
    --receipt) receipt="${2:?missing receipt}"; shift 2;;
    --key-file) key_file="${2:?missing key file}"; shift 2;;
    --critical) critical+=("${2:?missing critical file}"); shift 2;;
    *) usage; exit 2;;
  esac
done
case "$command" in
  snapshot) [ -n "$bundle" ] || { usage; exit 2; }; snapshot "$bundle" "$receipt" "${critical[@]}";;
  verify)
    [ -n "$bundle" ] || { usage; exit 2; }
    verify_bundle "$bundle"
    if [ -n "$receipt" ]; then
      require_0600 "$receipt"
      "$SCRIPT_DIR/../scripts/sanitize-evidence.py" --scan-only --input "$receipt" >/dev/null
      echo "rollback-receipt=sanitized"
    fi
    ;;
  migrate) [ -n "$bundle" ] && [ -n "$key_file" ] || { usage; exit 2; }; migrate "$bundle" "$key_file";;
  restore) [ -n "$bundle" ] && [ -n "$key_file" ] || { usage; exit 2; }; restore_all "$bundle" "$key_file";;
  verify-existing-key) [ -n "$key_file" ] || { usage; exit 2; }; verify_existing_key "$key_file";;
  *) usage; exit 2;;
esac
