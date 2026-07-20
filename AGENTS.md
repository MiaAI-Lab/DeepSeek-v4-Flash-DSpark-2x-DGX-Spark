# AGENTS.md

## Purpose

This repository packages the optimized **DSpark-r0b0tlab** implementation of
`deepseek-ai/DeepSeek-V4-Flash-DSpark` for two NVIDIA GB10 / DGX Spark nodes.
The production lane uses the native vLLM 0.25 DSpark speculator, FlashInfer
B12X MoE, RoCE, and `nvfp4_ds_mla` KV cache. The older Stage-C overlay remains
for historical reproducibility; do not confuse it with the production image.

## Release identity

- Platform: two Linux aarch64 GB10 / SM121 nodes, TP=2
- Base image: `ghcr.io/anemll/dspark-vllm-gx10@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8`
- Base source revision: `47503f8e38dadd4dededca798150db2619594fce`
- vLLM: `0.25.2.dev0+g752a3a504.d20260714`
- PyTorch: `2.11.0+cu130`; CUDA: `13.0`; FlashInfer: `0.6.15`
- Model revision: `913f0657a874f76844e2e91cbe706dbcaceeb6d7`
- Production KV cache: `nvfp4_ds_mla` (mandatory, not an experiment/fallback)
- Production MoE backend: `flashinfer_b12x`
- Speculator: DSpark; depth selected only from measured matched-profile evidence

`docker/runtime-manifest.production.json` is the machine-readable contract.

## Profiles

- `profiles/dspark-r0b0tlab-production.env`: throughput lane, 200K context ceiling,
  16 sequences, 16K batched tokens.
- `profiles/dspark-r0b0tlab-1m.env`: conservative one-million-token lane, two
  sequences, 8K batched tokens.
- `profiles/dspark-r0b0tlab-384k.env`: retained historical evidence profile.

Do not promote a profile until semantic, tool, retrieval, static/staggered
concurrency, MTP-acceptance, server-counter throughput, and native-log gates pass.

## Build and verification

```bash
./scripts/ci-verify.sh
./scripts/build-production-image.sh
docker run --rm --gpus all dspark-r0b0tlab:production-candidate audit
```

The canonical repository gate includes shell syntax, Python compilation,
contract tests, benchmark scaffold tests, profile rendering, publication
checksums, source presence, and public-safety scanning. Synthetic tests are
scaffold evidence only.

Live gate:

```bash
python3 scripts/runtime_gate.py \
  --base-url http://127.0.0.1:8888 \
  --worker-host worker-host \
  --output results/<run>/runtime-gate.json
```

Benchmark decode and prefill separately with `scripts/benchmark_dspark.py`.
Use long generation for decode; use a long fixed prompt and one output token for
prefill. Report server prompt/decode counters separately from client SSE rate.

## Native and no-regression gates

Required live markers on both ranks:

- DSpark speculator/model loaded
- `nvfp4_ds_mla`
- FlashInfer B12X
- NCCL `NET/IB`
- SM121 / CUDA 13.0 runtime audit

Blocking markers:

- active Marlin
- emulation
- backend fallback
- empty, repetitive, template-leaking, or semantically wrong output
- request errors or collapsed DSpark acceptance
- static-only concurrency without staggered/ragged success

Do not publish a decode gain that regresses prefill, quality, tool calls,
retrieval, NVFP4-KV capacity, request success, or the 1M profile.

## Build constraints

- Keep `MAX_JOBS=6`, `NVCC_THREADS=2`, and `FLASHINFER_NVCC_THREADS=2`.
- Keep CUDA 13.0 aligned with PyTorch cu130.
- Use `--shm-size=64g`, unlimited memlock, `IPC_LOCK`, and
  `/dev/infiniband` passthrough on both nodes.
- Resolve `NCCL_IB_GID_INDEX` and interface names from the live hosts; do not
  assume they are stable across driver/network changes.
- Use worker-first startup and verify both image identities before model load.
- Do not redistribute model weights or commit cache paths, LAN addresses,
  credentials, raw private logs, or unreviewed evidence.

## Attribution and publication

Preserve upstream authorship and Apache/MIT notices. This is an r0b0tlab
repository; do not open upstream issues or PRs unless explicitly requested.
Run `python3 scripts/public_safety_scan.py .` before every public push.
Published claims must point to machine-readable evidence and an immutable image
digest, not merely a successful build or healthy HTTP endpoint.
