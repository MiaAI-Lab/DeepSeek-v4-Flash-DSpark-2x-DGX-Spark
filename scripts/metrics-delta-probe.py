#!/usr/bin/env python3
"""Per-request drafter acceptance by content type and temperature, from /metrics deltas.

vLLM does not report acceptance per request. On an otherwise idle engine, the difference of the
spec-decode counters before and after one request is that request's exact draft/accept statistics,
including per-position acceptance. This probe runs a small matrix (prose, code, verbatim copy,
thinking on) at several temperatures, single stream, waits for an idle gap before each case and flags
any case that overlapped with other traffic (`contaminated`).

Usage:
    python3 scripts/metrics-delta-probe.py [--base-url http://127.0.0.1:8888] [--model NAME] [--api-key KEY]
                                           [--wait-max 1500] [--only prose_t0.7 code_t0.3 ...]
Writes results/metrics-delta-probe-<UTC>.json. `--base-url` accepts the `/v1` form too; `--api-key` (or the
DSPARK_API_KEY env var) is sent as a bearer token when DSPARK_API_KEYS guards the chat endpoint.
"""
import argparse
import json
import os
import re
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://127.0.0.1:8888")
ap.add_argument("--model", default="deepseek-v4-flash-vision-exp")
ap.add_argument("--api-key", default=os.environ.get("DSPARK_API_KEY", ""))
ap.add_argument("--wait-max", type=int, default=1500, help="seconds to wait for an idle engine per case")
ap.add_argument("--only", nargs="*", default=[])
args = ap.parse_args()
BASE = args.base_url.rstrip("/")
if BASE.endswith("/v1"):          # accept the /v1 form the other scripts in this repo use
    BASE = BASE[:-3]
HEADERS = {"Content-Type": "application/json"}
if args.api_key:
    HEADERS["Authorization"] = "Bearer " + args.api_key
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, time.strftime("metrics-delta-probe-%Y%m%dT%H%M%SZ.json", time.gmtime()))
KEYS = ["vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total", "vllm:generation_tokens_total"]
POS = ("vllm:spec_decode_num_accepted_tokens_per_pos", "vllm:spec_decode_num_accepted_tokens_per_pos_total")


def snap():
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    d = {}
    for line in txt.splitlines():
        if not line.startswith("vllm:"):
            continue
        name = line.split("{")[0].split(" ")[0]
        val = float(line.rsplit(" ", 1)[1])
        if name in KEYS or name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
            d[name] = val
        elif name in POS:
            m = re.search(r'position="(\d+)"', line)
            if m:
                d["pos" + m.group(1)] = val
    return d


def wait_idle():
    waited, idle = 0, 0
    while idle < 2:                       # two idle polls 10 s apart
        s = snap()
        idle = 0 if (s.get("vllm:num_requests_running", 0) or s.get("vllm:num_requests_waiting", 0)) else idle + 1
        if idle < 2:
            if waited >= args.wait_max:
                return False
            time.sleep(10)
            waited += 10
    return True


def run(case):
    body = {"model": args.model, "messages": case["messages"], "stream": True,
            "stream_options": {"include_usage": True}, "max_tokens": case["max_tokens"],
            "temperature": case["temperature"]}
    body["chat_template_kwargs"] = ({"enable_thinking": True, "thinking": True, "reasoning_effort": "high"}
                                    if case.get("thinking") else {"enable_thinking": False, "thinking": False})
    if not wait_idle():
        return {"name": case["name"], "skipped": "engine busy"}
    before = snap()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers=HEADERS)
    t0, t_first, usage, finish = time.time(), None, None, None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                delta = ch.get("delta", {})
                if (delta.get("content") or delta.get("reasoning_content")) and t_first is None:
                    t_first = time.time()
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    t_end = time.time()
    after = snap()
    drafts = after[KEYS[0]] - before[KEYS[0]]
    dtok = after[KEYS[1]] - before[KEYS[1]]
    acc = after[KEYS[2]] - before[KEYS[2]]
    gen = after[KEYS[3]] - before[KEYS[3]]
    pos = [after.get(f"pos{i}", 0) - before.get(f"pos{i}", 0) for i in range(8) if f"pos{i}" in after]
    decode_s = t_end - (t_first or t0)
    ct = (usage or {}).get("completion_tokens") or 0
    return {"name": case["name"], "temperature": case["temperature"], "thinking": bool(case.get("thinking")),
            "finish": finish, "completion_tokens": ct, "gen_tokens_metric": gen,
            "drafts": drafts, "draft_tokens": dtok, "accepted": acc,
            "acceptance_pct": round(100 * acc / max(dtok, 1), 1),
            "tokens_per_step": round(1 + acc / max(drafts, 1), 3),
            "per_pos_pct": [round(100 * p / max(drafts, 1), 1) for p in pos],
            "ttft_s": round((t_first or t_end) - t0, 3), "decode_s": round(decode_s, 2),
            "tok_per_s": round(gen / max(decode_s, 1e-9), 1),
            "step_ms": round(1000 * decode_s / max(drafts, 1), 2),
            # another request overlapped if the engine counted more tokens than this request produced
            "contaminated": bool(after.get("vllm:num_requests_running", 0) or gen > ct * 1.02 + 2)}


PROSE = ("Write a 400-word essay for game developers on why a multiplayer game needs server-authoritative "
         "physics. Plain prose, no lists, no headings.")
CODE = ("Write a complete Three.js ES-module scene: an InstancedMesh grid of 8x8x8 cubes with per-instance "
        "colors, OrbitControls, a directional light that cycles day and night over 60 seconds, and a resize "
        "handler. Reply with ONE fenced javascript code block and nothing else.")
SNIPPET = "\n".join(f"const v{i} = state.grid[{i}] * SCALE + offset{i % 4}; // cell {i}" for i in range(120))
COPY = ("Here is a JavaScript snippet:\n```javascript\n" + SNIPPET + "\n```\n"
        "Return the SAME snippet in full inside one fenced code block, changing only the identifier SCALE "
        "to CELL_SCALE everywhere. No commentary.")

cases = []
for t in (0.0, 0.3, 0.7, 1.0):
    cases.append({"name": f"prose_t{t}", "messages": [{"role": "user", "content": PROSE}], "temperature": t, "max_tokens": 700})
for t in (0.0, 0.3, 0.7, 1.0):
    cases.append({"name": f"code_t{t}", "messages": [{"role": "user", "content": CODE}], "temperature": t, "max_tokens": 1100})
for t in (0.7, 1.0):
    cases.append({"name": f"copy_edit_t{t}", "messages": [{"role": "user", "content": COPY}], "temperature": t, "max_tokens": 2500})
cases.append({"name": "think_prose_t0.7", "messages": [{"role": "user", "content": PROSE}], "temperature": 0.7, "max_tokens": 4000, "thinking": True})
cases.append({"name": "think_code_t0.7", "messages": [{"role": "user", "content": CODE}], "temperature": 0.7, "max_tokens": 5000, "thinking": True})

results = []
for case in cases:
    if args.only and case["name"] not in args.only:
        continue
    r = run(case)
    results.append(r)
    if "skipped" in r:
        print(f"{r['name']:>18}: SKIPPED ({r['skipped']})")
        continue
    print(f"{r['name']:>18}: gen={r['gen_tokens_metric']:>5.0f} fin={str(r['finish']):<6} acc={r['acceptance_pct']:5.1f}%  "
          f"tok/step={r['tokens_per_step']:.2f}  per-pos={r['per_pos_pct']}  step={r['step_ms']:.1f} ms  "
          f"{r['tok_per_s']:.1f} tok/s  ttft={r['ttft_s']:.2f}s" + ("  CONTAMINATED" if r["contaminated"] else ""))
    time.sleep(1.0)
json.dump(results, open(OUT, "w"), indent=1)
print("saved", OUT)
