#!/usr/bin/env bash
# Optional 3-node TP=3 launcher. The default
# ./start-deepseek-v4-flash-dspark.sh path stays TP=2 (one worker).
#
# Pads attention groups 8→9 so 3 divides the shard. Patch:
#   patches/tp3/apply_tp3_patch.py
# from https://github.com/localaiguyy/DeepSeek-V4-Flash-DSpark-3x-DGX-Spark
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--max-num-seqs N] [--host HOST] [--port PORT]

Start DeepSeek-V4-Flash DSpark on three DGX Sparks (tensor parallel 3).
Requires WORKER_HOST and WORKER2_HOST in .env.dspark.

Options:
  --max-num-seqs N   Concurrent slots (vLLM --max-num-seqs). CUDA-graph
                     capture size is N * (MTP_NUM_TOKENS + 1) rounded up to a
                     multiple of 8 (48 at N=6).
                     Default: TP3_MAX_NUM_SEQS in .env.dspark, else MAX_NUM_SEQS
                     (the 2-node value, usually 6). The 2-node start ignores
                     TP3_MAX_NUM_SEQS.
  --host HOST        Passed through to start-deepseek-v4-flash-dspark.sh
  --port PORT        Passed through to start-deepseek-v4-flash-dspark.sh

This does not replace ./start-deepseek-v4-flash-dspark.sh (TP=2).

After boot, prove the shard (not just that HTTP is up):
  scripts/validate_tp3.sh 127.0.0.1:8888

Patch or seqs changes need a recreate, not a container restart:
  ./stop-deepseek-v4-flash-dspark.sh && ./start-tp3.sh
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --max-num-seqs)
      [ "$#" -ge 2 ] && [ -n "$2" ] || { echo "--max-num-seqs requires a value." >&2; exit 2; }
      export _START_TP3_MAX_NUM_SEQS="$2"
      shift 2
      ;;
    --max-num-seqs=*)
      export _START_TP3_MAX_NUM_SEQS="${1#*=}"
      [ -n "${_START_TP3_MAX_NUM_SEQS}" ] || { echo "--max-num-seqs requires a value." >&2; exit 2; }
      shift
      ;;
    *)
      break
      ;;
  esac
done

if [ ! -f "$SCRIPT_DIR/patches/tp3/apply_tp3_patch.py" ]; then
  echo "Missing $SCRIPT_DIR/patches/tp3/apply_tp3_patch.py" >&2
  exit 1
fi

export DSPARK_TP3=1
export TP_SIZE=3
export NNODES=3
exec "$SCRIPT_DIR/start-deepseek-v4-flash-dspark.sh" "$@"
