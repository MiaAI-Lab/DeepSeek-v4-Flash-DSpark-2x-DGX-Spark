#!/usr/bin/env bash
# One-boot A/B measurement set. Writes results/ab-<label>-<UTC>.md and prints it.
# Usage: scripts/ab-measure.sh <label> [--long]   (--long adds a 128K c=1 TTFT trial, ~3 min)
set -uo pipefail
cd "$(dirname "$0")/.."
LABEL="${1:?label}"; LONG="${2:-}"
BASE=http://127.0.0.1:8888; MODEL=deepseek-v4-flash-vision-exp
OUT="results/ab-${LABEL}-$(date -u +%Y%m%dT%H%M%SZ).md"; mkdir -p results
WORKER="$(grep -E '^WORKER_HOST=' .env.dspark | cut -d= -f2)"
{
echo "# A/B measurement: $LABEL ($(date -u +%FT%TZ))"
echo; echo "## Effective knobs (.env.dspark)"; echo '```'
grep -E '^(GPU_MEMORY_UTILIZATION_TEXT|B12X_W4A16_TC_DECODE|DSPARK_ENABLE_(REPLICATE_MARKOV|ADAPTIVE_CHUNK|SP_INDEXER|DEEPGEMM_SM121_ALIAS)|DSPARK_SP_INDEXER_MIN_KEYS|DSPARK_MAX_INFLIGHT_PREFILLS|MTP_NUM_TOKENS|MAX_NUM_SEQS|NCCL_IB_HCA|WORKER_NCCL_IB_HCA)=' .env.dspark; echo '```'
echo; echo "## Boot gates"; echo '```'
docker logs deepseek-v4-flash-vllm-dspark-1 2>&1 | grep -E "dspark-(sp-indexer|adaptive-chunk|replicate-markov)\] patched|sm121 alias .*APPLIED|Truncating|Available KV cache memory|GPU KV cache size|Maximum concurrency|jit_monitor.py:129|EngineDead|NCCL WARN" | sed -E 's/^\([^)]*\) //' | cut -c1-160 | sort -u
docker logs deepseek-v4-flash-vllm-dspark-1 2>&1 | grep -oE "Capturing CUDA graphs \(FULL\).*100%\|[^|]*\| [0-9]+/[0-9]+" | sed -E 's/.*\| ([0-9]+\/[0-9]+)/FULL graphs captured \1/' | tail -1
echo "-- worker:"; ssh -o BatchMode=yes "$WORKER" 'docker logs deepseek-v4-flash-vllm-dspark-1 2>&1 | grep -E "Available KV cache memory|jit_monitor.py:129|EngineDead|NCCL WARN" | sed -E "s/^\([^)]*\) //" | cut -c1-160 | sort -u'; echo '```'
echo; echo "## Host memory (must stay >= 3 GB available, swap I/O ~0 during decode)"; echo '```'
echo "head:   $(grep -E 'MemAvailable' /proc/meminfo) swap-used=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print (t-f)/1048576" GB"}' /proc/meminfo)"
ssh -o BatchMode=yes "$WORKER" 'echo "worker: $(grep MemAvailable /proc/meminfo) swap-used=$(awk "/SwapTotal/{t=\$2}/SwapFree/{f=\$2}END{print (t-f)/1048576\" GB\"}" /proc/meminfo)"'; echo '```'
echo; echo "## Quality probe (temp 0, thinking off) — expect 391 / Tokyo"; echo '```'
for q in "What is 17 multiplied by 23? Reply with only the number." "What is the capital of Japan? One word."; do
  curl -s -m 120 $BASE/v1/chat/completions -H 'Content-Type: application/json' -d "$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"messages":[{"role":"user","content":sys.argv[2]}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"thinking":False}}))' "$MODEL" "$q")" | python3 -c 'import json,sys; print(repr((json.load(sys.stdin)["choices"][0]["message"].get("content") or "")[:60]))'
done; echo '```'
echo; echo "## Natural-prose acceptance (healthy: pos0 >= 0.85 at temp 0)"; echo '```'
python3 scripts/natural-acceptance-window.py --base-url $BASE/v1 --model $MODEL --temperature 0 2>&1 | tail -1
python3 scripts/natural-acceptance-window.py --base-url $BASE/v1 --model $MODEL --temperature 0.6 --top-p 0.95 2>&1 | tail -1; echo '```'
echo; echo "## Decode (bench-miaai, p=256; reference pre-change TP=2: c1 62-83, c6 agg 156-162, c2 ~47-57)"; echo '```'
for c in 1 6 2; do python3 scripts/bench-miaai.py --base-url $BASE/v1 --model $MODEL --prompt 256 --concurrency $c --repeat 3 2>&1 | grep -E "^trial|FINAL"; done; echo '```'
echo; echo "## Spec acceptance on the numbered-word bench (reference: 45-51 %)"; echo '```'
python3 scripts/spec-acceptance.py --base-url $BASE/v1 --model $MODEL --trials 3 --prompt 256 2>&1 | grep -E "OVERALL|pos[0-5]"; echo '```'
echo; echo "## Prefill TTFT c=1 (reference: 8K ~4.8 s / 32K ~18.5-23 s / 128K ~79-80 s)"; echo '```'
for p in 8192 32768; do python3 scripts/bench-miaai.py --base-url $BASE/v1 --model $MODEL --prompt $p --concurrency 1 --repeat 2 2>&1 | grep -E "^trial"; done
if [ "$LONG" = "--long" ]; then python3 scripts/bench-miaai.py --base-url $BASE/v1 --model $MODEL --prompt 131072 --concurrency 1 --repeat 1 2>&1 | grep -E "^trial"; fi; echo '```'
echo; echo "## Swap activity during a c=1 run (si/so KB/s should be ~0)"; echo '```'
( vmstat 1 25 | awk 'NR>2{si+=$7; so+=$8; n++} END{printf "head:   samples=%d swap-in=%d KB swap-out=%d KB\n", n, si, so}' & )
python3 scripts/bench-miaai.py --base-url $BASE/v1 --model $MODEL --prompt 256 --concurrency 1 --repeat 1 >/dev/null 2>&1; sleep 26; echo '```'
echo; echo "## Post-run log check"; echo '```'
docker logs --since 30m deepseek-v4-flash-vllm-dspark-1 2>&1 | grep -cE "jit_monitor.py:129" | sed 's/^/mid-serve JIT warnings (last 30 min): /'
docker logs --since 30m deepseek-v4-flash-vllm-dspark-1 2>&1 | grep -ciE "EngineDead|Failed to advance|sample_tokens.*timeout|NCCL WARN" | sed 's/^/errors (last 30 min): /'; echo '```'
} 2>&1 | tee "$OUT"
echo; echo "written: $OUT"
