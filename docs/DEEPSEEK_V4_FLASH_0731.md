# DeepSeek V4 Flash 0731

`deepseek-ai/DeepSeek-V4-Flash-0731` supersedes the preview checkpoint while retaining the same `DeepseekV4ForCausalLM` and DSpark speculative-decoding structure.

## Checkpoint

- Repository: `deepseek-ai/DeepSeek-V4-Flash-0731`
- Tested revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` (default `DSPARK_REVISION` in `.env.dspark.example`; prepare + `vllm serve --revision` both honor it)
- Context: `1048576`
- DSpark block size: `5`
- Quantization metadata: FP8 weights
- Architecture: text-only causal language model

The published checkpoint has no vision processor, projector, or vision tower.
This recipe serves **text-only** 0731 on `:8888`.

## Serving Profile

The default two-Spark profile uses MTP-5 probabilistic speculation, NVFP4 MLA KV cache, prefix caching, chunked prefill, asynchronous scheduling, CUDA graphs, and the `deepseek_v4` tokenizer, reasoning parser, and tool-call parser.

The model card does not ship a Jinja chat template. It includes an `encoding` package that defines message encoding and output parsing, including `low`, `high`, and `max` reasoning effort. Validate multi-turn role boundaries, reasoning separation, and tool calls after runtime upgrades because successful weight loading alone does not prove encoding compatibility.

Set `DSPARK_ENCODING_FILE` to the checkpoint's `encoding/encoding_dsv4.py` path inside the container when the runtime image predates the checkpoint. The launcher installs that encoder into vLLM before import, on both ranks. It also corrects pre-0731 tokenizer wrappers that mapped `low` reasoning effort to `high`, and applies the Issue #21 hotfix so `encode_arguments_to_dsml` accepts dict tool `arguments` (not only JSON strings) — otherwise multi-turn tool history can be poisoned. These changes are required for the 0731 `reasoning_content`, reasoning-effort, and tool-argument semantics.

The equivalent Unsloth GGUF Jinja template implements the same central DS4
behavior—DSML tools, `<think>` boundaries, `high`/`max` instruction prefixes,
and retention of tool-turn `reasoning_content`—but it is not loaded by this
vLLM profile. Here, request controls are consumed by vLLM's custom tokenizer
wrapper and passed to the Python encoder. The underlying implementations fall
back to non-thinking when no kwarg exists, but this recipe defaults
`DEFAULT_THINKING=low` to match DeepSeek V4's intended base reasoning mode.
The setting accepts `off`, `low`, `high`, or `max`; `low` opens
`<think>` but adds no effort instruction. For pi, use
`pi-models.dspark.example.json`; it maps pi's off/low/high/max selector
to request-level `chat_template_kwargs.thinking` and
`chat_template_kwargs.reasoning_effort`.

## Benchmark Method

Run `scripts/benchmark-0731.py` against a warmed endpoint. The default sweep covers 256, 2K, 8K, 32K, and 128K prompt tokens at concurrency 1, 2, 4, and 6. Each request has a distinct first cache block so prefix caching cannot make later cases reuse earlier prefill work. It streams each response, records time to first token, prefill throughput, per-request decode throughput, and aggregate decode throughput using API-reported token counts from naturally completed responses. It does not impose a server-side output limit.

```bash
python3 scripts/benchmark-0731.py \
  --base-url http://127.0.0.1:8888/v1 \
  --model deepseek-v4-flash-0731 \
  --output results/deepseek-v4-flash-0731.json
```

## Two-Spark Results

Measured on two DGX Sparks connected over ConnectX-7 with tensor parallelism 2. The endpoint used MTP-5 probabilistic speculation, NVFP4 MLA KV cache, CUDA graphs, prefix caching, chunked prefill, and a 1,048,576-token context. Values are medians across requests except aggregate throughput.

| Prompt | Concurrency | TTFT (s) | Prefill tok/s | Decode tok/s | Aggregate tok/s |
|---:|---:|---:|---:|---:|---:|
| 256 | 1 | 0.63 | 447 | 75.4 | 69.1 |
| 256 | 2 | 0.81 | 357 | 58.3 | 104.9 |
| 256 | 4 | 1.26 | 222 | 46.8 | 164.5 |
| 256 | 6 | 1.42 | 197 | 36.9 | 191.2 |
| 2,048 | 1 | 0.81 | 2,563 | 68.8 | 62.0 |
| 2,048 | 2 | 1.11 | 1,911 | 57.0 | 97.6 |
| 2,048 | 4 | 1.38 | 1,505 | 44.0 | 154.7 |
| 2,048 | 6 | 6.06 | 342 | 34.7 | 143.7 |
| 8,192 | 1 | 4.80 | 1,713 | 73.9 | 43.7 |
| 8,192 | 2 | 7.51 | 1,176 | 49.8 | 56.2 |
| 8,192 | 4 | 14.50 | 578 | 37.4 | 72.3 |
| 8,192 | 6 | 18.38 | 454 | 23.6 | 73.1 |
| 32,768 | 1 | 22.96 | 1,428 | 64.0 | 16.6 |
| 32,768 | 2 | 26.82 | 1,287 | 41.5 | 24.8 |
| 32,768 | 4 | 44.85 | 756 | 17.4 | 26.7 |
| 32,768 | 6 | 60.75 | 550 | 10.8 | 27.9 |
| 131,072 | 1 | 78.75 | 1,665 | 65.2 | 5.9 |
| 131,072 | 2 | 111.17 | 1,306 | 30.9 | 6.6 |

The 131,072-token concurrency-4 probe did not complete within the 180-second measurement window, while the server remained healthy. Its partial values are retained in the raw JSON as capacity-bound evidence and are intentionally excluded from the throughput table.

A separate 900,000-token acceptance request completed with 899,994 API-reported prompt tokens, 900,000 total tokens, 1,028.85-second TTFT, and approximately 874.8 prefill tok/s. The response returned the requested sentinel and confirms the full 1,048,576-token serving profile beyond configuration metadata alone.

Raw measurements are in `results/deepseek-v4-flash-0731-2x-dgx-spark.json`.

## Regular Graph Opt-Out

Anemll `0.1.1` automatically enables breakable CUDA graphs for DeepSeek V4 when `VLLM_USE_BREAKABLE_CUDAGRAPH` is absent. The default recipe now sets it to `0`, which preserves the regular CUDA graph path without disabling CUDA graphs or enabling eager execution.

A matched 520-token natural-completion probe used temperature `0.2`, top-p `0.95`, MTP-5 probabilistic speculation, `MAX_NUM_SEQS=6`, `MAX_NUM_BATCHED_TOKENS=8192`, and the full 1,048,576-token context. Every measured response completed at its requested stop marker without chat-template leakage.

| Mode | Breakable graphs | Regular graphs | Change |
|---|---:|---:|---:|
| C1 decode, warm median | 74.55 tok/s | 95.9 tok/s | +28.6% |
| C2 aggregate decode, median | 134.2 tok/s | 151.8 tok/s | +13.1% |
| C4 aggregate decode | not measured | 263.7 tok/s | - |
| C6 aggregate decode | not measured | 340.5 tok/s | - |

The matched 14K-token prefill probes remained within normal run variance: warm C1 moved from 1,770-1,781 to 1,857 tok/s, while C2 moved from 1,920-1,954 to 1,943-1,987 tok/s. This setting is a decode improvement, not a claim that prefill is 28.6% faster.
