# AGENTS.md — DSpark-r0b0tlab reproducibility guide

This repository packages the DSpark-r0b0tlab DeepSeek V4 Flash DSpark runtime for two DGX Spark / GB10-class nodes.

## Identity and claims

- Refer to this implementation as **DSpark-r0b0tlab**.
- Do not add legacy third-party branding to docs, reports, badges, or artifact text.
- Keep claims evidence-scoped:
  - 1M profile: `max_model_len=1048576`, `MAX_NUM_SEQS=2`, strict-code-only needle retrieval passes at ~903K actual prompt for 10/50/90% depths.
  - 384K profile: 300K target produces ~271K actual prompt and passes 10/50/90% depths; c4 publication ladder repeats at 140.7–158.5 tok/s.
  - Do not claim unconstrained 900K long-form quality from strict exact-code retrieval alone.

## Reproducible environment

Use the checked-in profile and scripts instead of ad-hoc shell exports:

```bash
cp .env.dspark.example .env.dspark
# edit WORKER_HOST, MASTER_ADDR, NCCL_* interface values, and HF_CACHE
./build-dspark-vllm-runtime.sh
./validate-dspark-config.sh
./prepare-dspark-model-cache.sh
./start-deepseek-v4-flash-dspark.sh
```

For the validated 1M profile, copy values from:

```text
profiles/dspark-r0b0tlab-1m.env
```

Expected core values:

```env
DSPARK_VLLM_IMAGE=vllm-dspark-runtime:dspark-nvfp4-stage-c
MAX_MODEL_LEN=1048576
MAX_NUM_SEQS=2
MAX_NUM_BATCHED_TOKENS=8192
GPU_MEMORY_UTILIZATION=0.88
MTP_NUM_TOKENS=5
VLLM_DSPARK_CONFIDENCE_SCHEDULER=off
VLLM_DSPARK_FUSED_MARKOV_ARGMAX=0
VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1
```

## Host prep for 1M runs

Before launching the 1M profile on GB10 nodes:

```bash
sudo systemctl stop gdm3 || true
sudo systemctl disable gdm3 || true
systemctl is-active gdm3 || true
```

Verify no display manager or desktop session is consuming GPU memory before starting DSpark.

## Container reproducibility

The container build is staged:

1. `recipe/Dockerfile.dspark-runtime-overlay`
2. `recipe/nvfp4/Dockerfile.stage-a`
3. `recipe/nvfp4/Dockerfile.stage-b`
4. `recipe/nvfp4/Dockerfile.stage-c`

The final runtime tag must remain:

```text
vllm-dspark-runtime:dspark-nvfp4-stage-c
```

Canonical repository green is:

```bash
./scripts/ci-verify.sh
```

When changing overlay/runtime code, run that canonical gate before committing. If the change affects the container image itself, also rebuild:

```bash
./build-dspark-vllm-runtime.sh
./validate-dspark-config.sh
```

If hardware is available, also run a live smoke:

```bash
./smoke-deepseek-v4-flash-dspark.sh
python3 benchmarks/needle_acceptance_sweep.py --tokens 1000000 --depths 0.1 0.5 0.9 --chars-per-token 4.35 --max-tokens 32 --strict-code-only --outdir benchmark-results/one_million_strict_sweep_<timestamp>
```

## Artifact hygiene

- Do not commit `.env.dspark`, credentials, private keys, or local cache paths.
- Prefer sanitized docs/profiles over raw container inspect dumps.
- Benchmark artifacts committed to GitHub must not include local home paths, private IPs, hostnames, tokens, or secrets unless explicitly documented as placeholders.
- Temporary verification scripts belong under `/tmp/hermes-verify-*` and should be removed after use.
