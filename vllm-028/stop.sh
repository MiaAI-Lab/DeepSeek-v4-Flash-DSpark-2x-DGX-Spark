#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
NODE_RANK=0 VLLM_HOST_IP=10.0.3.2 docker compose -p dgx34-v028 -f docker-compose.yml stop -t 30
ssh -o BatchMode=yes -o ConnectTimeout=10 dgx3@10.0.3.1 \
  'NODE_RANK=1 VLLM_HOST_IP=10.0.3.1 docker compose -p dgx34-v028 -f ~/vllm-upgrade-028/docker-compose.yml stop -t 30'
