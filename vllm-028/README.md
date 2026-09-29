# DGX3/4：DeepSeek V4 Flash 升级 vLLM 0.28.0

目标机器是合作方 `spark-4502`（dgx4 / rank0 / 10.0.3.2）和
`spark-1ea7`（dgx3 / rank1 / 10.0.3.1），不是 Royal Spark01/02。
实施日期：2026-09-29。升级完成，07:02 UTC 最后核验通过。

## 验收结果

- 两节点镜像 ID 一致；主节点 `/version` 返回 `0.28.0`，`/health` 返回 200。
- 主节点新服务 active/enabled，旧服务 inactive/disabled。
- 两容器运行中，restart count 为 0，`OOMKilled=false`，启动及验收日志无
  ERROR/Traceback。DSpark 接受 token 计数已增长，推测解码实际生效。
- 算术、中文、low thinking、函数调用及参数 JSON、SSE 流式终止、双请求并发全部通过。
- 24,025-token 输入正确返回预设数字，耗时 36.42 秒。
- 108,025-token 输入正确返回预设数字，耗时 62.34 秒，其中 23,808 tokens
  命中前缀缓存。这是功能验收，不是隔离负载下的性能基准。
- 1M 为配置和启动 KV 容量验证通过的上限；本次未发送完整 1M-token 请求。
- 未进行主机重启测试；已检查 systemd 开机启用状态及容器 restart policy。

原始结果见 `validation-20260929.json` 和 `runtime-final-20260929.json`。

## 版本和配置

- vLLM：从 `0.25.2.dev0+g752a3a504.d20260714` 升级到 `0.28.0`。
- 基础镜像：`docker.m.daocloud.io/vllm/vllm-openai:v0.28.0`，拉取的 manifest
  digest 为 `sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`。
- 部署镜像：`dgx34-vllm:0.28.0`；两节点 image ID 均为
  `sha256:23ec5c1704ff06c43af1a897ce1a900e39166890b5acddc0a39712d87b162b32`。
- PyTorch `2.13.0+cu130`，Triton `3.7.1`，FlashInfer `0.6.18`。
- FlashInfer Python/cubin/JIT cache 成套升级，覆盖 vLLM 的
  `flashinfer-python==0.6.16.post3` 元数据固定依赖，以使用 GB10 sparse MLA
  DSv4 kernels。`pip check` 会报告该固定版本差异；不将其误称为依赖完全一致。
- 标准模型 `deepseek-ai/DeepSeek-V4-Flash-0731`，revision
  `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`，保留原有缓存和权重。
- 保留 API `:8888`、模型 ID `deepseek-v4-flash-0731`、原有绑定/鉴权方式、
  TP2/RoCE、DSpark K=5 probabilistic、默认 low thinking。
- 保留 `max_model_len=1048576`、`max_num_seqs=8`、batch token budget 8192、
  GPU utilization 0.835。
- 旧定制 `nvfp4_ds_mla` 不在官方 0.28 参数集合内，改用 `fp8_ds_mla`。
  启动报告 KV cache 1,757,420 tokens，1M 单请求容量检查通过。
- 不套用其他机器的 Vision-Exp 权重补丁，也不套用旧 0.25 热补丁。
  vLLM 源代码未修改。

## 运行与构建

两节点部署目录均为 `~/vllm-upgrade-028`。主节点服务
`deepseek-vllm-028.service` 在开机时先拉起 rank1，再拉起 rank0；两节点容器
均为 `dgx34-v028-vllm-1`，restart policy 为 `unless-stopped`。

```bash
# 主节点构建；若直连 GitHub 不通，按实际网络提供临时 HTTP CONNECT 代理。
docker build --network host -t dgx34-vllm:0.28.0 .
docker save dgx34-vllm:0.28.0 | zstd -1 -T4 |
  ssh dgx3@10.0.3.1 'zstd -d | docker load'

sudo systemctl status deepseek-vllm-028
curl -fsS http://127.0.0.1:8888/version
curl -fsS http://127.0.0.1:8888/health
cd ~/vllm-upgrade-028 && python3 validate.py
```

构建依赖下载曾通过临时 SSH 转发使用代理；运行环境不依赖该代理。
首次启动需读取约 155 GiB checkpoint 并进行内核编译/预热。

## 回退

两节点 `~/vllm-upgrade-028/recipe-backup` 保存原 recipe；旧镜像另外标记为
`dspark-vllm-gx10:rollback-20260929`。原容器
`deepseek-v4-flash-vllm-dspark-1` 已保留，停用自动重启。
主节点旧服务 `deepseek-v4-flash-dspark.service` 停用，原文件未覆盖。

从主节点执行：

```bash
sudo bash /home/dgx4/vllm-upgrade-028/rollback.sh
```

回退会停掉新服务、恢复旧容器 restart policy，并重新启用原服务及启动脚本。
备份目录含原始环境配置和容器信息，权限 0700；不要复制进仓库。

参考：[vLLM 0.28.0 官方发布说明](https://github.com/vllm-project/vllm/releases/tag/v0.28.0)。
