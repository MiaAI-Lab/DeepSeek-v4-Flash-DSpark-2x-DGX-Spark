# Setup — A/B replicas, models, and step-by-step reproduction

## Hardware / fabric

- 4× NVIDIA DGX Spark (GB10, SM121, 128 GB unified each), 200 Gb RoCE fabric.
- Two independent **TP=2** replicas (one GPU per node, 2 nodes per replica).
- Fabric interface `enp1s0f1np1`, HCA `rocep1s0f1`, `NCCL_IB_GID_INDEX=3`.

## Model (same weights for both replicas)

| | |
|---|---|
| Model | `deepseek-ai/DeepSeek-V4-Flash-DSpark` (MLA + sparse indexer, FP8 weights, DSpark γ=5 / rank-256 Markov head) |
| KV cache | `fp8` |
| Speculative | `{"method":"dspark","num_speculative_tokens":5}` |
| Context | `max-model-len 262144` (recipe also supports up to ~900K single-stream) |

> Both replicas serve the **same** DeepSeek-V4-Flash-DSpark weights. They differ
> only in **packaging/launch recipe** (and, for B, the concurrency patch). There is
> no separate "rafael model" vs "tonyd2wild model" — same checkpoint, two recipes.

## Runtime image (both)

`vllm-dspark-runtime:clean` — a **thin overlay** (`COPY` the DSpark vLLM source files
+ `py_compile`) on the prebuilt base `ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready`
(mirror of `aidendle94/sparkrun-vllm-ds4-gb10:production-ready`, the "unholy-fusion"
build carrying the B12X MoE kernels for GB10). The DSpark overlay source is Rafael
Caricio's integration; the TonyD2Wild repo vendors those same files.

## Replica A — Rafael Caricio stack (control, unpatched)

| | |
|---|---|
| Repo | `rafaelcaricio/spark_vllm_docker` + `rafaelcaricio/vllm` (fork `codex/dspark-harness-integration`) |
| Nodes | head API endpoint and worker RoCE address from your `.env.dspark` |
| Launch | compose + `unholy` entrypoint + a wrapper that rewrites `method:mtp → dspark` at start |
| `max-num-seqs` | 1 (control) |
| Single-stream | ~52–54 tok/s |

## Replica B — TonyD2Wild packaged recipe (frontier sandbox, **patched**)

| | |
|---|---|
| Repo | DSpark-r0b0tlab implementation (vendors Rafael's overlay; worker-first DSpark-r0b0tlab launch) |
| Nodes | head API endpoint and worker RoCE address from your `.env.dspark` |
| Launch | self-contained compose, direct `vllm serve ... method:dspark`, worker-first start script |
| `max-num-seqs` | swept 1 → 16 (with this patch) |
| Single-stream | ~52 tok/s |

---

## Step-by-step (applies to anyone's DSpark + DeepSeek-V4 setup)

1. **Stand up a working DSpark replica** using either recipe above (build
   `vllm-dspark-runtime:clean`, copy the model, configure `.env`, launch
   worker-first). Confirm single-stream works at `max-num-seqs=1`.

2. **Apply the patch** to your overlay checkout (the dir containing `vllm/`):
   ```bash
   git apply -p1 patches/keys-concurrency.patch
   ```

3. **Rebuild the runtime image on every node** (the overlay is baked at build time):
   ```bash
   ./build-dspark-vllm-runtime.sh      # builds head + worker
   ```

4. **Configure for concurrency** in your `.env`:
   ```
   MAX_NUM_SEQS=16
   VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1   # required for the ragged path
   ```

5. **Restart worker-first**, wait for `/v1/models` → 200 (~5 min: load + cudagraph).

6. **Verify** (scripts in `benchmarks/`):
   ```bash
   python3 benchmarks/correctness_test.py http://<head>:<port>        # must print OUTPUT IDENTICAL: True
   python3 benchmarks/staggered_bench.py  http://<head>:<port> 16 0.4 # 16/16 ok, acceptance ~0.55
   python3 benchmarks/bench_concurrent.py http://<head>:<port> 1,2,4,8,16
   ```

7. **Scale out (optional):** run a second patched TP=2 replica and put a
   least-connections router in front for ~2× aggregate / concurrency.
