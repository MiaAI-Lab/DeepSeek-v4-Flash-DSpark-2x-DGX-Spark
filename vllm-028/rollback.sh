#!/usr/bin/env bash
# Run as root on spark-4502; restores the untouched legacy recipe/service.
set -euo pipefail
systemctl disable --now deepseek-vllm-028.service
runuser -u dgx4 -- docker update --restart=unless-stopped deepseek-v4-flash-vllm-dspark-1
runuser -u dgx4 -- ssh dgx3@10.0.3.1 'docker update --restart=unless-stopped deepseek-v4-flash-vllm-dspark-1'
systemctl enable --now deepseek-v4-flash-dspark.service
