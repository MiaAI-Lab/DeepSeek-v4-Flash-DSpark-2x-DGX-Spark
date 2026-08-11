# VL 4-bit KV coexist chase — 2026-08-11

Goal: Qwen3-VL-4B TP=2 with 4-bit KV @ 32k + 0731 `nvfp4_ds_mla` GPU KV ≥ 2_000_000 with VL up.

Image: `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (`sha256:3430d661…`). GB10 = **SM12.1**.

## Verdict

| Target | Result |
|--------|--------|
| VL non-fp8 4-bit KV @ 32k TP=2 | **PASS** — `int4_per_token_head` + `TRITON_ATTN` |
| True `--kv-cache-dtype nvfp4` | **BLOCKED** on GB10 (see Phase 1) |
| Main GPU KV ≥ 2_000_000 with VL up | **MISS** — best coexist **1,598,877** (peak alone **1,886,417**) |
| Text + vision smoke | **PASS** |
| Stack left running | **YES** (green coexist profile below) |

## Best running coexist profile

| Knob | Value |
|------|-------|
| `GPU_MEMORY_UTILIZATION` | **0.82** |
| `MAX_NUM_SEQS` | **4** (capture 4×7=28) |
| Main KV dtype | `nvfp4_ds_mla` |
| Main **GPU KV cache size** | **1,598,877** tokens (Available KV ≈ 15.55 GiB) |
| `VL_SIDECAR_KV_CACHE_DTYPE` | **`int4_per_token_head`** |
| `VL_SIDECAR_ATTENTION_BACKEND` | `TRITON_ATTN` |
| `VL_SIDECAR_GPU_UTIL` | **0.03** |
| `VL_SIDECAR_MAX_MODEL_LEN` | 32768 |
| VL **GPU KV cache size** | **55,232** tokens (≈1.68× of 32k; Available KV ≈ 1.01 GiB) |
| Smoke text `:8888` | `OK` |
| Smoke vision `:8889` | solid-red JPEG → `Red` |

Also proven once: main **0.82** + VL util **0.022** → VL KV **34,048** (≥1.03× of 32k) while main stayed at **1,630,111**.

## Phase 0 — baseline (fp8 VL)

Pre-start available RAM: head ~117 Gi, worker ~110 Gi (GB10 unified; nvidia-smi FB N/A).

| Service | Config | Result |
|---------|--------|--------|
| Main | util 0.80, `nvfp4_ds_mla` | **UP** — Available KV 14.48 GiB; **GPU KV 1,488,433** |
| VL | util 0.03, `fp8`, TRITON, 32k | **FAIL** — need 1.12 GiB KV, got 0.78 GiB (est. max_len 22704) |

## Phase 1 — nvfp4 mitigations (4 attempts)

### 1a. `nvfp4` + `TRITON_ATTN`

```
ValueError: Selected backend AttentionBackendEnum.TRITON_ATTN is not valid
for this configuration. Reason: ['kv_cache_dtype not supported']
```

Triton `supported_kv_cache_dtypes` includes `fp8` / `int4_per_token_head`, **not** `nvfp4`.

### 1b. `nvfp4` + `FLASHINFER`

```
ValueError: Selected backend AttentionBackendEnum.FLASHINFER is not valid
for this configuration. Reason: ['kv_cache_dtype not supported']
```

Probe inside image: `DeviceCapability(12,1)`, `is_device_capability_family(100)=False`,
`supports_trtllm_attention(prefill/decode)=False`, `FlashInferBackend.supports_kv_cache_dtype("nvfp4")=False`.
FlashInfer path requires SM100 trtllm-gen (`--kv-cache-dtype nvfp4 requires the SM100 trtllm-gen FlashInfer path`).

### 1c. Runtime patch / SM100 force

Not applied: lying about SM100 cannot create trtllm-gen kernels for SM12.1. `nvfp4_ds_mla` on DeepSeek is a different MLA layout — does not enable Qwen attention nvfp4.

### 1d. Proven 4-bit equivalent: `int4_per_token_head` + `TRITON_ATTN`

**SUCCESS.** Page math: INT4 uses `head_dim = head_size//2` (uint8) vs FP8 full head_dim → ~½ KV bytes/token vs fp8. Boot lists `qwen3-vl-4b`, `Using int4_per_token_head data type to store kv cache`, KV sized for 32k.

## Phase 2 — shrink VL util

| Util | vs main | Result |
|------|---------|--------|
| 0.03 | 0.80 settled | OK — 45,376 tok / 3.34 GiB |
| 0.025 | 0.80 settled | FAIL — Available 1.08 GiB then no cache blocks |
| 0.022 | 0.82 settled | OK — 34,048 tok / 0.62 GiB |
| 0.03 | 0.82 settled | OK — 55,232 tok / 1.01 GiB (**left running**) |

Free-memory gate: VL refuses start when `free < util * 121.63 GiB` (e.g. 3.17 free < 0.03→3.65 needed). `VLLM_SKIP_INIT_MEMORY_CHECK=1` wired into `docker-compose.vl-sidecar.yml` but the free-memory util check still fires on this image.

## Phase 3 — maximize main KV

| Main util | MAX_NUM_SEQS | Main GPU KV | VL |
|-----------|--------------|-------------|-----|
| 0.80 | 6 | 1,488,433 | fp8 fail / later int4 OK after settle |
| 0.82 | 4 | **1,630,111** then **1,598,877** | int4 OK at 0.022 / 0.03 |
| 0.835 | 6 | 1,811,802 | NCCL CUDA OOM on VL init |
| 0.84 | 4 | **1,886,417** | FAIL — free 1.19 GiB < 0.018× total |

### Why ≥2M + VL is blocked

Extrapolating 0.84 → 1.886M @ ~18.7 GiB Available KV: **2.0M ≈ 19.9 GiB** → main util ~**0.85**. At 0.84 worker free after main was **1.19 GiB**; VL int4 32k needs ~**0.60 GiB KV** plus AWQ shard weights/NCCL — the util free-memory check alone needs ≥~2.2 GiB at util 0.018. Raising main to 2M starves VL TP=2 on worker.

Peak N with VL up: **1,630,111**. Peak N without VL: **1,886,417**. Neither reaches 2_000_000 on this boot shape (weights ~79 GiB/rank + graphs).

## Phase 4 — verify commands

```bash
curl -sS http://127.0.0.1:8888/v1/models
curl -sS http://127.0.0.1:8889/v1/models
# text
curl -sS http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash-0731","messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"thinking":false}}'
# vision (results/smoke-solid-red.jpg)
# → content "Red"
```

## Config files touched

- `.env.dspark` — live coexist profile
- `.env.dspark.example` — VL dtype / coexist notes
- `docker-compose.vl-sidecar.yml` — pass `VLLM_SKIP_INIT_MEMORY_CHECK`
- `CHANGELOG.md`, `plugins/dspark_vision_mcp/README.md`

## Next fix (if chasing 2M later)

1. FlashInfer / vLLM build with SM120 nvfp4 attention (true nvfp4) to cut VL further, **or**
2. Head-only VL TP=1 (breaks TP=2 hard target) to free worker for main util ≥0.85, **or**
3. Smaller VL weights / enforce_eager / lower VL max_num_seqs to shrink activation so util 0.015 still clears 32k under main 0.85.
