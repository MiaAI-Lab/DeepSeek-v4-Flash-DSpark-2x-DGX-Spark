#!/usr/bin/env bash
# Run as root on spark-4502, after both nodes have the candidate image.
set -euo pipefail
DEPLOY=/home/dgx4/vllm-upgrade-028
runuser -u dgx4 -- docker image inspect dgx34-vllm:0.28.0 >/dev/null
runuser -u dgx4 -- ssh -o BatchMode=yes dgx3@10.0.3.1 'docker image inspect dgx34-vllm:0.28.0 >/dev/null'
# Avoid the legacy ExecStop script deleting the rollback containers.
install -d /run/systemd/system/deepseek-v4-flash-dspark.service.d
printf '[Service]\nExecStop=\nExecStop=/usr/bin/true\n' > /run/systemd/system/deepseek-v4-flash-dspark.service.d/upgrade-preserve.conf
systemctl daemon-reload
systemctl disable --now deepseek-v4-flash-dspark.service
rm /run/systemd/system/deepseek-v4-flash-dspark.service.d/upgrade-preserve.conf
runuser -u dgx4 -- docker update --restart=no deepseek-v4-flash-vllm-dspark-1
runuser -u dgx4 -- ssh dgx3@10.0.3.1 'docker update --restart=no deepseek-v4-flash-vllm-dspark-1'
runuser -u dgx4 -- docker stop -t 30 deepseek-v4-flash-vllm-dspark-1
runuser -u dgx4 -- ssh dgx3@10.0.3.1 'docker stop -t 30 deepseek-v4-flash-vllm-dspark-1'
runuser -u dgx4 -- docker run --rm --gpus all --entrypoint python3 dgx34-vllm:0.28.0 -c 'import torch; x=torch.ones(16,device="cuda"); assert (x+x).sum().item()==32; print("rank0 CUDA_OK")'
runuser -u dgx4 -- ssh dgx3@10.0.3.1 "docker run --rm --gpus all --entrypoint python3 dgx34-vllm:0.28.0 -c 'import torch; x=torch.ones(16,device=\"cuda\"); assert (x+x).sum().item()==32; print(\"rank1 CUDA_OK\")'"
install -m 644 "$DEPLOY/deepseek-vllm-028.service" /etc/systemd/system/deepseek-vllm-028.service
systemctl daemon-reload
systemctl enable deepseek-vllm-028.service
systemctl start --no-block deepseek-vllm-028.service
