#!/bin/sh
set -eu

require_secret() {
  variable="$1"
  eval "path=\${$variable:-}"
  [ -n "$path" ] && [ -f "$path" ] && [ ! -L "$path" ] || {
    echo "missing secret file for $variable" >&2
    exit 1
  }
  mode="$(stat -c '%a' "$path")"
  [ "$mode" = "600" ] || { echo "$variable must reference a mode-0600 file" >&2; exit 1; }
}

require_secret MASTER_KEY_FILE
require_secret ORIGIN_KEY_FILE
require_secret DATABASE_PASSWORD_FILE
IFS= read -r master_key <"$MASTER_KEY_FILE"
IFS= read -r origin_key <"$ORIGIN_KEY_FILE"
IFS= read -r database_password <"$DATABASE_PASSWORD_FILE"
[ -n "$master_key" ] && [ -n "$origin_key" ] && [ -n "$database_password" ] || exit 1
case "$master_key" in sk-*) ;; *) echo "LiteLLM master key must start with sk-" >&2; exit 1 ;; esac
encoded_password="$(printf '%s' "$database_password" | python3 -c 'import sys,urllib.parse; print(urllib.parse.quote(sys.stdin.read(), safe=""))')"
export LITELLM_MASTER_KEY="$master_key"
export VLLM_ORIGIN_KEY="$origin_key"
export DATABASE_URL="postgresql://litellm_smoke:${encoded_password}@postgres:5432/litellm_smoke"
unset master_key origin_key database_password encoded_password
exec litellm "$@"
