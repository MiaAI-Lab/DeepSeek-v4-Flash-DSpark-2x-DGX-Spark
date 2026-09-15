#!/usr/bin/env bash
# One-knob boot + measurement driver (used by the 2026-09-02 A/B session).
# Usage: scripts/ab-boot.sh LABEL KEY VAL [--long]
#   - sets KEY=VAL in .env.dspark (sed on the existing line; the line must already exist)
#   - stop → start (2-node lane), health, boot gates on both ranks, host-memory gate (>= 3 GB both nodes)
#   - scripts/ab-measure.sh LABEL [--long], then 5 extra c=1 trials (8 total; c=1 noise on this lane is ~±10 tok/s)
#   - everything goes to results/ab-logs/boot-LABEL.log; last line is BOOT_DONE (grep for it)
# Exit 1 = boot failed (docker logs tails saved to results/bootfail-LABEL-*.log), 2 = memory gate failed.
# AB_MEM_GATE_KB overrides the 3 GB host-memory gate (kB, default 3000000).
set -uo pipefail
cd "$(dirname "$0")/.."
L=$1; K=$2; V=$3; LONG=${4:-}
D=results/ab-logs; mkdir -p $D
# K becomes a grep/sed -E pattern and V a sed replacement: validate K as an
# env-var name (alnum + underscore is literal in ERE, so no regex injection),
# reject multiline V (the env file is line-based), and escape & | \ in V for
# the replacement side — an unescaped & expands to the whole match and an
# unescaped | ends the s||| expression early, corrupting .env.dspark.
case "$K" in ''|*[!A-Za-z0-9_]*) echo "KEY must be an env-var name, got: $K" >&2; echo BOOT_DONE; exit 2 ;; esac
case "$V" in *$'\n'*|*$'\r'*) echo "VAL must be single-line" >&2; echo BOOT_DONE; exit 2 ;; esac
grep -qE "^${K}=" .env.dspark || { echo "no ${K}= line in .env.dspark"; echo BOOT_DONE; exit 3; }
V_SED=$(printf '%s' "$V" | sed 's/[&|\\]/\\&/g')
sed -i -E "s|^${K}=.*|${K}=${V_SED}|" .env.dspark
echo "== env: $(grep -E "^${K}=" .env.dspark)"
./stop-deepseek-v4-flash-dspark.sh > $D/stop-$L.log 2>&1; echo "stop_rc=$?"
./start-deepseek-v4-flash-dspark.sh > $D/start-$L.log 2>&1; echo "start_rc=$?"; tail -2 $D/start-$L.log
WORKER="$(grep -E '^WORKER_HOST=' .env.dspark | head -n1 | cut -d= -f2)"
curl -fsS -m 5 http://127.0.0.1:8888/health >/dev/null && echo HEALTH_OK || { echo HEALTH_FAIL; docker logs --tail 200 deepseek-v4-flash-vllm-dspark-1 > results/bootfail-$L-head.log 2>&1; ssh -o BatchMode=yes "$WORKER" 'docker logs --tail 200 deepseek-v4-flash-vllm-dspark-1 2>&1' > results/bootfail-$L-worker.log; echo BOOT_DONE; exit 1; }
G='dspark-(sp-indexer|adaptive-chunk|replicate-markov)\] patched|sm121 alias .*APPLIED|Truncating|Available KV cache memory|EngineDead|NCCL WARN'
echo "== gates head"; docker logs deepseek-v4-flash-vllm-dspark-1 2>&1 | grep -E "$G" | sed -E 's/^\([^)]*\) //' | cut -c1-140 | sort -u
docker logs deepseek-v4-flash-vllm-dspark-1 2>&1 | grep -oE "Capturing CUDA graphs \(FULL\).*100%\|[^|]*\| [0-9]+/[0-9]+" | tail -1 | grep -oE '[0-9]+/[0-9]+$' | sed 's/^/FULL graphs /'
echo "== gates worker"; ssh -o BatchMode=yes "$WORKER" "docker logs deepseek-v4-flash-vllm-dspark-1 2>&1 | grep -E '$G' | sed -E 's/^\([^)]*\) //' | cut -c1-140 | sort -u"
echo "== mem"; grep MemAvailable /proc/meminfo; ssh -o BatchMode=yes "$WORKER" 'grep MemAvailable /proc/meminfo'
MA=$(awk '/MemAvailable/{print $2}' /proc/meminfo); MW=$(ssh -o BatchMode=yes "$WORKER" "awk '/MemAvailable/{print \$2}' /proc/meminfo")
GATE=${AB_MEM_GATE_KB:-3000000}; echo "mem gate: ${GATE} kB"
# Fail closed: this script runs `set -u` WITHOUT -e, so an empty MW (worker
# ssh failed) made `[ "" -lt "$GATE" ]` error out inside the || — the gate
# evaluated false and the bench ran against a possibly-broken boot.
case "$MA" in ''|*[!0-9]*) echo "MEM_GATE: head MemAvailable unreadable" >&2; echo BOOT_DONE; exit 2 ;; esac
case "$MW" in ''|*[!0-9]*) echo "MEM_GATE: worker MemAvailable unreadable (ssh failed?)" >&2; echo BOOT_DONE; exit 2 ;; esac
if [ "$MA" -lt "$GATE" ] || [ "$MW" -lt "$GATE" ]; then echo "MEM_BELOW_GATE — not measuring"; echo BOOT_DONE; exit 2; fi
docker logs deepseek-v4-flash-vllm-dspark-1 > $D/logs-$L-head.txt 2>&1; ssh -o BatchMode=yes "$WORKER" 'docker logs deepseek-v4-flash-vllm-dspark-1 2>&1' > $D/logs-$L-worker.txt
curl -s http://127.0.0.1:8888/metrics | grep -E '^vllm:num_requests_running'
echo "== ab-measure"; bash scripts/ab-measure.sh "$L" $LONG 2>&1 | sed -n '/## Quality/,$p'
echo "== extra c=1 x5"; python3 scripts/bench-miaai.py --base-url http://127.0.0.1:8888/v1 --model deepseek-v4-flash-vision-exp --prompt 256 --concurrency 1 --repeat 5 2>&1 | grep -E "^trial|FINAL"
echo "== swap after: head $(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print (t-f)/1048576" GB"}' /proc/meminfo) worker $(ssh -o BatchMode=yes "$WORKER" 'awk "/SwapTotal/{t=\$2}/SwapFree/{f=\$2}END{print (t-f)/1048576\" GB\"}" /proc/meminfo')"
echo BOOT_DONE
