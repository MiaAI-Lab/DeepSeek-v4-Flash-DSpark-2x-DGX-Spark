# Model-serving recovery evidence

This directory contains the passive evidence and transactional recovery
components for the John/Ofus DeepSeek TP=2 deployment.

`observer/vllm_request_observer.py` is a content-free FastAPI middleware on
John. It records request lifecycle only. The custom Anemll vLLM topology does
not expose a supported HTTP middleware hook for independent rank iteration
progress, and Ofus runs headless. Therefore HTTP lifecycle, HTTP health, and GPU
utilization are never promoted into rank-progress evidence.

`sitecustomize.py` installs the genuine worker hook on both ranks. It is pinned
to vLLM `0.25.2.dev0+g752a3a504.d20260714`, the exact
`vllm.v1.worker.gpu_worker.Worker.execute_model(self, scheduler_output)`
signature, observer revision, and an owner-only host capability. It publishes
scheduled-token and completion deltas only after `execute_model` succeeds.

`sentinel/model_serving_sentinel.py` accepts that atomic `rank_worker` snapshot
only when it is fresh and matches the local rank, process generation, and
observer revision. If the hook cannot install or the artifact is absent, stale,
or mismatched, the sentinel emits `lifecycle: unknown` with zero progress
counters. This intentionally prevents automatic deadlock confirmation.

`recovery/recover_deepseek_serving.py` preflights independently collected John
and Ofus inspection snapshots before claiming one incident-generation receipt.
Its execution adapter uses only recovery-specific modes in the checked-in
start, stop, status, and smoke scripts. A timeout or uncertain post-dispatch
result is receipted as `ambiguous` and is never retried under the same or a new
invocation.

Run:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

Recovery remains coordinator-authorized. Sentinels never stop or start serving
processes.
