# DSML syntax sampling temperature — vLLM vs ds4 (DwarfStar)

Why the same DeepSeek-V4-Flash weights score lower on
[tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) under this
repo's vLLM profile than under the ds4 engine
([MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark),
wrapping [Entrpi/ds4](https://github.com/Entrpi/ds4) `batched-serving`) when both
runs use temperature 1.0 with thinking on.

## The asymmetry

Tool calls in DeepSeek V4 are emitted as DSML markup: `<｜DSML｜tool_calls>` /
`<｜DSML｜invoke name=…>` / `<｜DSML｜parameter …>` blocks with JSON payloads.

**ds4:** while the model emits DSML *syntax* tokens (tags, invoke headers, JSON
punctuation), sampling is forced to `temperature=0`. Only argument *payloads*
(string bodies, file contents) use the request's sampling parameters. Structural
tokens can never be derailed by temperature.
(upstream README, "Tool call handling and canonicalization";
[fork ds4_server.c](https://raw.githubusercontent.com/Entrpi/ds4/batched-serving/ds4_server.c),
plus a hand-written `try_repair_dsml` fallback for malformed output.)

**vLLM (this recipe):** every token is sampled at the request temperature —
DSML headers, parameter tags, braces, and payload alike. There is no per-region
temperature mechanism. At `temperature=1.0` each structural token is a
full-temperature sample that can come out malformed.

## Why it moves the benchmark score

tool-eval-bench reads tool calls **only** from the structured
`message.tool_calls` field of `/v1/chat/completions` responses — content text is
never scanned for tool-call markup
([adapters/openai_compat.py](https://raw.githubusercontent.com/SeraphimSerapis/tool-eval-bench/main/src/tool_eval_bench/adapters/openai_compat.py)).
A call whose DSML is malformed enough that `--tool-call-parser deepseek_v4`
cannot parse it into structured `tool_calls` is scored as "no tool call"
(`missing_step` failure). So at temp 1.0:

- ds4 loses points only when the *payload* is wrong (a genuine model decision).
- vLLM additionally loses points whenever *syntax* sampling goes wrong — a
  mechanical serving artifact, not a model-quality signal.

This effect is context-length independent and is not diluted at small contexts:
a single corrupted structural token anywhere in the call breaks it.

## What this is NOT

- Not a KV-cache precision effect. At tool-eval-bench context sizes (a few
  thousand tokens), the difference between this recipe's `nvfp4_ds_mla` KV and
  ds4's E4M3/FP8 MLA KV is second-order; KV quantization error is per-token and
  does not compound over turns.
- Not an MTP/spec-decode effect. DSpark drafts are verified against the target;
  `MTP_NUM_TOKENS` changes throughput, not output (the small batched-mode
  residual in `docs/PATCHES.md` is orthogonal).
- Not a weights effect. This recipe serves FP8 weights; ds4 serves ~2-bit GGUF
  experts with Q8 attention/output. If weights dominated, vLLM would score
  higher.

## Mitigations on the vLLM side

No server-side switch exists today. Options, cheapest first:

1. **Run the bench at temp 0.** vLLM is greedy everywhere, so syntax cannot be
   derailed — ds4's syntax robustness for free, at the cost of also making
   payloads greedy. As a diagnostic this cleanly separates "syntax corruption"
   from "genuine capability gap".
2. **Structured outputs with xgrammar structural tags.** Grammar masking makes
   structural positions have exactly one valid token, which is deterministic in
   practice at any temperature. No DSML grammar ships with vLLM; structural-tag
   function calling is still an open RFC
   ([vllm-project/vllm#32142](https://github.com/vllm-project/vllm/issues/32142)).
   Requires authoring a DSML structural-tag grammar and verifying it composes
   with the `deepseek_v4` tool parser and streaming.
3. **Custom logits processor.** vLLM v1 supports pluggable logits processors;
   one could track DSML parse state per request and force argmax inside syntax
   regions. Effectively a port of ds4's hand-written C logic.

## Suggested experiment

Rerun tool-eval-bench against this stack at `temperature=0` (thinking on,
everything else unchanged) and compare per-scenario failure kinds with the
temp-1.0 run:

- Gap mostly closes → the score difference was sampling asymmetry (this
  document), not model or serving quality.
- Gap remains → look at the multi-turn encoding layer next: the bench strips
  `reasoning_content` from history, and this profile wires the checkpoint's
  `encoding/encoding_dsv4.py` in via `--tokenizer-mode deepseek_v4` (see
  `docs/DEEPSEEK_V4_FLASH_0731.md`, "Serving Profile") — a mismatch there
  degrades multi-turn tool scenarios specifically.
