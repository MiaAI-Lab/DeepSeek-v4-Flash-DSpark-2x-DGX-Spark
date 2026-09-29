#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ssh -o BatchMode=yes -o ConnectTimeout=10 dgx3@10.0.3.1 \
  'NODE_RANK=1 HEADLESS=1 VLLM_HOST_IP=10.0.3.1 docker compose -p dgx34-v028 -f ~/vllm-upgrade-028/docker-compose.yml up -d'
NODE_RANK=0 HEADLESS= VLLM_HOST_IP=10.0.3.2 docker compose -p dgx34-v028 -f docker-compose.yml up -d
for attempt in $(seq 1 180); do
  if curl -fsS --max-time 3 http://127.0.0.1:8888/health >/dev/null; then
    curl -fsS --max-time 3 http://127.0.0.1:8888/version
    exit 0
  fi
  if [ "$(docker inspect --format '{{.State.Status}}' dgx34-v028-vllm-1)" = exited ]; then
    docker logs --tail 60 dgx34-v028-vllm-1
    exit 1
  fi
  sleep 10
done
echo 'Timed out waiting for vLLM 0.28 health' >&2
exit 1
