#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TAG="${DSPARK_HOTFIX_IMAGE_TAG:-deepseek-v4-flash-dspark:runtime-hotfixes}"
EXPORT_PATH=""

usage() {
  echo "Usage: $0 [--tag IMAGE_TAG] [--export /absolute/path/image.tar]" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tag) [ "$#" -ge 2 ] || { usage; exit 2; }; TAG="$2"; shift 2 ;;
    --export) [ "$#" -ge 2 ] || { usage; exit 2; }; EXPORT_PATH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

if [ -n "$EXPORT_PATH" ] && [[ "$EXPORT_PATH" != /* ]]; then
  echo "--export must be an absolute path" >&2
  exit 2
fi

for command in docker git python3; do
  command -v "$command" >/dev/null 2>&1 || { echo "Missing $command" >&2; exit 1; }
done

python3 "$SCRIPT_DIR/scripts/verify-runtime-hotfixes.py" --repo-root "$SCRIPT_DIR"

manifest_path="$SCRIPT_DIR/recipe/runtime-hotfixes.manifest.json"
inputs=(recipe/runtime-hotfixes.manifest.json)
while IFS= read -r relative; do
  inputs+=("$relative")
done < <(python3 -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["inputs"]))' "$manifest_path")
for relative in "${inputs[@]}"; do
  git -C "$SCRIPT_DIR" ls-files --error-unmatch "$relative" >/dev/null
  case "$relative" in
    *.env|*.env.*|*.key|*.pem|*.secrets*|artifacts/*|results/*)
      echo "Refusing sensitive build input: $relative" >&2
      exit 1
      ;;
  esac
done

context="$(mktemp -d "${TMPDIR:-/tmp}/dspark-runtime-build-XXXXXX")"
cleanup() { rm -rf -- "$context"; }
trap cleanup EXIT
for relative in "${inputs[@]}"; do
  mkdir -p "$context/$(dirname "$relative")"
  install -m 0644 "$SCRIPT_DIR/$relative" "$context/$relative"
done

base_image="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_image"])' "$manifest_path")"
issue21_commit="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["issue21_upstream_commit"])' "$manifest_path")"
issue22_commit="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["issue22_upstream_commit"])' "$manifest_path")"
docker build --pull=false \
  --build-arg "BASE_IMAGE=$base_image" \
  --build-arg "ISSUE21_UPSTREAM_COMMIT=$issue21_commit" \
  --build-arg "ISSUE22_UPSTREAM_COMMIT=$issue22_commit" \
  -f "$context/recipe/Dockerfile.anemll-runtime-hotfixes" -t "$TAG" "$context"
python3 "$SCRIPT_DIR/scripts/verify-runtime-hotfixes.py" --repo-root "$SCRIPT_DIR" --image "$TAG"

image_id="$(docker image inspect "$TAG" -f '{{.Id}}')"
if [ -n "$EXPORT_PATH" ]; then
  if [ -e "$EXPORT_PATH" ]; then
    echo "Refusing to overwrite export: $EXPORT_PATH" >&2
    exit 1
  fi
  umask 077
  docker save "$TAG" -o "$EXPORT_PATH"
fi
printf 'DSPARK_VLLM_IMAGE=%s\n' "$image_id"
printf 'DSPARK_HOTFIX_IMAGE_TAG=%s\n' "$TAG"
if [ -n "$EXPORT_PATH" ]; then printf 'DSPARK_HOTFIX_IMAGE_EXPORT=%s\n' "$EXPORT_PATH"; fi
