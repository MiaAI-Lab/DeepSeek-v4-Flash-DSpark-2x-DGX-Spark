#!/usr/bin/env python3
"""Reproduce issue #26 (prefix-cache retention across x8 cold->warm->warm)
on the LIVE dspark server, with the issue #27 fix applied.

Protocol from issue #26:
  - 8 unique-prefix prompts at target tokens
  - cold wave: 8 concurrent; verify 0 cache hits
  - warm1: replay the EXACT same 8 wire payloads concurrently; count hits
  - warm2: replay again
  - hit = per-lane usage cached_tokens == input - 256 AND server
    prefix_cache_hits_total / prompt_tokens_cached_total deltas agree

Reports per-wave aggregate cache-hit ratio and per-lane cached_tokens.
"""
import json, os, time, urllib.request, sys, threading, statistics

# Endpoint/model are overridable the same way reproduce-issue43-live.py already
# does it; the hardcoded defaults only match a deployment served on 8888 under
# the recipe's default model name, so a rig serving on another port or with a
# custom SERVED_MODEL_NAME could not run this repro at all.
API = os.environ.get("DSPARK_API", "http://127.0.0.1:8888")
MODEL = os.environ.get("DSPARK_MODEL", "deepseek-v4-flash-0731")
N_LANES = int(sys.argv[1]) if len(sys.argv) > 1 else 8
TARGET_IN = int(sys.argv[2]) if len(sys.argv) > 2 else 32768
OUT = 256
SEED_BASE = 26_000


def post(url, body, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def tokenize(text):
    with post(f"{API}/tokenize", {"model": MODEL, "prompt": text}) as r:
        return json.load(r)["count"]


def build_unique_prompt(target, nonce):
    front = (f"issue-26 lane-nonce-{nonce} " +
             "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789" * 64)
    unit = "benchmark context datum alpha beta gamma delta epsilon "
    text = front + "\n\n" + unit * (target // 3)
    while True:
        c = tokenize(text)
        if c >= target:
            return text, c
        text += unit * max(1, (target - c) // 3)


def run_lane(idx, prompt, seed):
    res = {"idx": idx, "cached_tokens": 0, "prompt_tokens": 0,
           "error": None, "usage": None}
    body = {
        "model": MODEL, "prompt": prompt,
        "max_tokens": OUT, "min_tokens": OUT,
        "ignore_eos": True, "temperature": 0.0, "seed": seed,
        "stream": True, "stream_options": {"include_usage": True},
    }
    try:
        with post(f"{API}/v1/completions", body, timeout=900) as resp:
            buf = b""
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        continue
                    try:
                        ev = json.loads(data)
                    except Exception:
                        continue
                    if ev.get("usage"):
                        u = ev["usage"]
                        res["usage"] = u
                        res["prompt_tokens"] = u.get("prompt_tokens", 0)
                        # OpenAI completions usage: prompt_tokens_details.cached_tokens
                        pd = u.get("prompt_tokens_details") or {}
                        res["cached_tokens"] = pd.get("cached_tokens", 0)
    except Exception as e:
        res["error"] = repr(e)
    return res


def metric(name, m):
    for line in m.splitlines():
        if line.startswith(name + "{") and "_created" not in line:
            i = line.rfind("}")
            if i >= 0:
                return float(line[i + 2:])
    return None


def wave(prompts, seeds):
    results = [None] * N_LANES
    def w(i):
        results[i] = run_lane(i, prompts[i][0], seeds[i])
    ts = [threading.Thread(target=w, args=(i,)) for i in range(N_LANES)]
    for t in ts: t.start()
    for t in ts: t.join()
    return results


def main():
    print(f"[repro-26] N={N_LANES} target_in={TARGET_IN} out={OUT}")
    prompts = [build_unique_prompt(TARGET_IN, f"lane{i}-{int(time.time())}")
               for i in range(N_LANES)]
    for i, (_, c) in enumerate(prompts):
        print(f"  lane{i} prompt tokens = {c}")
    seeds = [SEED_BASE + i for i in range(N_LANES)]

    for wname in ("cold", "warm1", "warm2"):
        m_before = urllib.request.urlopen(API + "/metrics").read().decode()
        print(f"\n=== {wname} ===")
        t0 = time.perf_counter()
        res = wave(prompts, seeds)
        wall = time.perf_counter() - t0
        time.sleep(1)
        m_after = urllib.request.urlopen(API + "/metrics").read().decode()
        d_hits = (metric("vllm:prefix_cache_hits_total", m_after) or 0) - \
                 (metric("vllm:prefix_cache_hits_total", m_before) or 0)
        d_cached = (metric("vllm:prompt_tokens_cached_total", m_after) or 0) - \
                   (metric("vllm:prompt_tokens_cached_total", m_before) or 0)
        d_prompt = (metric("vllm:prompt_tokens_total", m_after) or 0) - \
                   (metric("vllm:prompt_tokens_total", m_before) or 0)
        cached_per_lane = [r["cached_tokens"] if r else -1 for r in res]
        hits_lanes = sum(1 for c in cached_per_lane if c > 0)
        total_in = sum(p[1] for p in prompts)
        print(f"  wave wall = {wall:.2f}s")
        print(f"  per-lane cached_tokens = {cached_per_lane}")
        print(f"  lanes with a hit: {hits_lanes}/{N_LANES}")
        print(f"  response-usage cached total = {sum(cached_per_lane)}")
        print(f"  server prefix_cache_hits_total delta = {d_hits:+.0f}")
        print(f"  server prompt_tokens_cached_total delta = {d_cached:+.0f}")
        print(f"  server prompt_tokens_total delta        = {d_prompt:+.0f}")
        print(f"  aggregate hit ratio = {(d_hits/d_prompt) if d_prompt else 0:.4f}")
        if wname == "cold":
            time.sleep(3)


if __name__ == "__main__":
    main()