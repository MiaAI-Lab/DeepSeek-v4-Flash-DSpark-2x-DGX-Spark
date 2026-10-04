#!/usr/bin/env python3
"""Decode step time and prefill throughput versus context length, single stream, idle engine.

Builds nested filler prefixes (so the prefix cache covers the shorter ones) at the requested token
counts, asks for 300 tokens of prose, and reports TTFT, computed-vs-cached prompt tokens, step time
(from the spec-decode draft counter delta), acceptance and tokens/s at each context size.

Usage:
    python3 scripts/context-step-probe.py [--base-url http://127.0.0.1:8888] [--model NAME] [--api-key KEY]
                                          [2000 25000 75000 150000]
Writes results/context-step-probe-<UTC>.json. `--base-url` accepts the `/v1` form too; `--api-key` (or the
DSPARK_API_KEY env var) is sent as a bearer token when DSPARK_API_KEYS guards the chat endpoint.
"""
import argparse
import json
import os
import random
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://127.0.0.1:8888")
ap.add_argument("--model", default="deepseek-v4-flash-vision-exp")
ap.add_argument("--api-key", default=os.environ.get("DSPARK_API_KEY", ""))
ap.add_argument("--wait-max", type=int, default=1500)
ap.add_argument("targets", nargs="*", type=int, default=[2000, 25000, 75000, 150000])
args = ap.parse_args()
BASE = args.base_url.rstrip("/")
if BASE.endswith("/v1"):          # accept the /v1 form the other scripts in this repo use
    BASE = BASE[:-3]
HEADERS = {"Content-Type": "application/json"}
if args.api_key:
    HEADERS["Authorization"] = "Bearer " + args.api_key
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, time.strftime("context-step-probe-%Y%m%dT%H%M%SZ.json", time.gmtime()))
KEYS = ["vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total", "vllm:generation_tokens_total"]


def snap():
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    d = {}
    for line in txt.splitlines():
        if not line.startswith("vllm:"):
            continue
        name = line.split("{")[0].split(" ")[0]
        if name in KEYS or name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
            d[name] = float(line.rsplit(" ", 1)[1])
    return d


def wait_idle():
    waited, idle = 0, 0
    while idle < 2:
        s = snap()
        idle = 0 if (s.get("vllm:num_requests_running", 0) or s.get("vllm:num_requests_waiting", 0)) else idle + 1
        if idle < 2:
            if waited >= args.wait_max:
                return False
            time.sleep(10)
            waited += 10
    return True


def count_tokens(text):
    body = {"model": args.model, "messages": [{"role": "user", "content": text}], "add_generation_prompt": True}
    req = urllib.request.Request(BASE + "/tokenize", data=json.dumps(body).encode(), headers=HEADERS)
    return json.load(urllib.request.urlopen(req, timeout=120))["count"]


random.seed(7)
WORDS = ("chunk mesh shader light camera physics server client packet tick entity quest guild dungeon portal "
         "texture normal buffer index vertex frame budget latency memory cache thread worker queue").split()
words = []
for i in range(int(max(args.targets) / 1.3) + 2000):
    w = WORDS[random.randrange(len(WORDS))]
    words.append(w + ("," if i % 9 == 8 else "") + ("." if i % 23 == 22 else ""))
QUESTION = ("\n\nIgnore the log above except as context. In about 300 words of plain prose, explain to a game "
            "developer why server-authoritative physics matters in a multiplayer game. No lists, no headings.")

results = []
for target in args.targets:
    n = int(target / 1.3)
    text = "Session log:\n" + " ".join(words[:n]) + QUESTION
    tok = count_tokens(text)
    n = int(n * target / max(tok, 1))                      # one calibration pass toward the target
    text = "Session log:\n" + " ".join(words[:n]) + QUESTION
    if not wait_idle():
        results.append({"target": target, "skipped": "engine busy"})
        print(target, "SKIPPED busy")
        continue
    body = {"model": args.model, "messages": [{"role": "user", "content": text}], "stream": True,
            "stream_options": {"include_usage": True}, "max_tokens": 300, "temperature": 0.7,
            "chat_template_kwargs": {"enable_thinking": False, "thinking": False}}
    before = snap()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers=HEADERS)
    t0, t_first, usage = time.time(), None, None
    with urllib.request.urlopen(req, timeout=3600) as r:
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
                if ch.get("delta", {}).get("content") and t_first is None:
                    t_first = time.time()
    t_end = time.time()
    after = snap()
    drafts = after[KEYS[0]] - before[KEYS[0]]
    acc = after[KEYS[2]] - before[KEYS[2]]
    dtok = after[KEYS[1]] - before[KEYS[1]]
    gen = after[KEYS[3]] - before[KEYS[3]]
    decode_s = t_end - (t_first or t0)
    u = usage or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    ptok = u.get("prompt_tokens", tok)
    res = {"target": target, "prompt_tokens": ptok, "cached_tokens": cached,
           "computed_tokens": (ptok - cached) if cached is not None else None,
           "ttft_s": round((t_first or t_end) - t0, 2),
           "prefill_tok_per_s": round((ptok - (cached or 0)) / max((t_first or t_end) - t0, 1e-9), 0),
           "gen_tokens": gen, "drafts": drafts,
           "acceptance_pct": round(100 * acc / max(dtok, 1), 1), "tokens_per_step": round(1 + acc / max(drafts, 1), 2),
           "step_ms": round(1000 * decode_s / max(drafts, 1), 1), "tok_per_s": round(gen / max(decode_s, 1e-9), 1),
           "contaminated": bool(after.get("vllm:num_requests_running", 0))}
    results.append(res)
    print(f"ctx {ptok:>7,} tok (cached {cached}): ttft {res['ttft_s']:6.2f}s ({res['prefill_tok_per_s']:.0f} new tok/s)  "
          f"step {res['step_ms']:6.1f} ms  {res['tok_per_s']:5.1f} tok/s  acc {res['acceptance_pct']:5.1f}%  "
          f"tok/step {res['tokens_per_step']:.2f}" + ("  CONTAMINATED" if res["contaminated"] else ""))
    time.sleep(2)
json.dump(results, open(OUT, "w"), indent=1)
print("saved", OUT)
