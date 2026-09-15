---
name: Bug report
about: The recipe is not behaving as expected, produces an error, or numbers do not match the README.
title: ""
labels: "bug"
assignees: ""
---

<!-- Thank you for using this recipe!

     If you are looking for support, please check the README and docs/ENVS.md first,
     or reach out on X:
      * https://x.com/MiaAI_lab

     If you have found a bug, then fill out the template below.
-->

---

## Environment

<!-- Fill in what applies to your setup. docs/ENVS.md lists every knob and its default. -->

- Hardware / nodes: <!-- e.g. 2x DGX Spark (GB10, 128 GB unified memory), 3x via start-tp3.sh, ... -->
- Interconnect: <!-- e.g. ConnectX-7 RoCE (NCCL_IB_HCA=...), 10GbE, ... -->
- Image / vLLM version: <!-- `docker images | grep dspark` — the pinned image is ghcr.io/anemll/dspark-vllm-gx10:0.1.1, vLLM 0.25.2.dev0 -->
- Checkpoint (`DSPARK_MODEL` + `DSPARK_REVISION`): <!-- e.g. DeepSeek-V4-Flash-0731 @ 9e165c30, or DeepSeek-V4-Flash-Vision-Exp @ 86f746b3 -->
- Served name (`SERVED_MODEL_NAME`): <!-- e.g. deepseek-v4-flash-0731 -->
- How you started the serve: <!-- e.g. ./start-deepseek-v4-flash-dspark.sh, ./switch-model.sh abliterated, custom compose -->
- Relevant `.env` values: <!-- e.g. MAX_MODEL_LEN, MAX_NUM_SEQS, GPU_MEMORY_UTILIZATION_TEXT, MTP_NUM_TOKENS, DSPARK_MAX_INFLIGHT_PREFILLS, LONG_PREFILL_TOKEN_THRESHOLD, ABLITERATED, DEFAULT_THINKING, any DSPARK_ENABLE_* opt-ins you turned on -->

---

## Steps to Reproduce

<!-- Full steps so that we can reproduce the problem. -->

1. <!-- e.g. `./start-deepseek-v4-flash-dspark.sh` — what it printed up to the failure -->
2. ... <!-- the request or action that shows the bug -->
3. ... <!-- e.g. "curl /v1/models returns a different served name than configured" -->

**Expected results:** <!-- what did you expect to happen? -->

**Actual results:** <!-- what did you actually see happen? -->

---

### Additional context

Add anything else: a minimal failing request, JSON responses, `docker inspect` output, and so on.

<details>
<summary>Minimal reproduction sample</summary>

<!--
      If the bug is about model output or API behavior, attach a minimal reproducible
      request below between the lines with the backticks.
-->

```bash
curl -s http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash-0731",
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 256
  }'
```

</details>

<details>
  <summary>Logs</summary>

<!--
      Paste the log output below between the lines with the backticks, and mention
      whether it came from the start script, `docker logs vllm-dspark-1` (head),
      the worker node, or a client.

      Common culprits worth checking before filing:
        * Boot fails downloading weights -> HF_HUB_OFFLINE=1 with a cold cache; warm it
          with prepare-dspark-model-cache.sh first.
        * Vision-Exp rejects MTP_NUM_TOKENS -> it must be >= 5 and divisible by 3,
          or set DSPARK_ENABLE_DSPARK_BLOCK_K=1 for k=dspark_block_size.
        * Decode collapses under concurrent long prompts -> check
          DSPARK_MAX_INFLIGHT_PREFILLS and LONG_PREFILL_TOKEN_THRESHOLD (docs/ENVS.md).
        * `Unknown vLLM environment variable` warnings -> the knob is not registered
          on the Anemll 0.1.1 image; check docs/ENVS.md for the supported surface.
-->

```

```

</details>

<!--
      Consider also attaching screenshots and/or videos to better illustrate the issue.

      You can upload them directly on GitHub.
      Beware that video file size is limited to 10MB.
-->
