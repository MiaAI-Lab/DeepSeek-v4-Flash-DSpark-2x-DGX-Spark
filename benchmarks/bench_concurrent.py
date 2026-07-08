#!/usr/bin/env python3
"""Concurrent (static-batch) benchmark: fire N simultaneous streams, measure
per-stream decode tok/s + server-side aggregate tok/s + draft acceptance."""
import json, time, urllib.request, sys, re, threading

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://10.100.10.2:8888"
MODEL= sys.argv[3] if len(sys.argv) > 3 else "MiMo-V2.5-NVFP4"
MAXT = 256

PROMPT = ("Complete this Python module with many conventional CRUD methods, "
          "helper functions, getters/setters and exhaustive docstrings. Keep "
          "going until complete.\n\n```python\nclass InventoryService:\n"
          "    \"\"\"CRUD service for inventory items.\"\"\"\n")

def metrics():
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
            t = r.read().decode()
    except Exception:
        return {}
    g = lambda n: sum(float(x) for x in re.findall(r"^%s(?:\{[^}]*\})?\s+([0-9.eE+-]+)$" % re.escape(n), t, re.M)) or 0.0
    return {"gen": g("vllm:generation_tokens_total"),
            "acc": g("vllm:spec_decode_num_accepted_tokens_total"),
            "draft": g("vllm:spec_decode_num_draft_tokens_total")}

def stream(idx, results):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0.0, "max_tokens": MAXT, "stream": True,
            "ignore_eos": True, "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); tf = None; n = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"): continue
            p = line[5:].strip()
            if p == "[DONE]": break
            try: o = json.loads(p)
            except: continue
            d = o.get("choices", [{}])[0].get("delta", {})
            if d.get("content"):
                if tf is None: tf = time.perf_counter()
                n += 1
    te = time.perf_counter()
    dec = (n - 1) / (te - tf) if tf and n > 1 else 0
    results[idx] = (n, dec, te)

def run_round(n_conc):
    m0 = metrics(); wall0 = time.perf_counter()
    results = {}; threads = [threading.Thread(target=stream, args=(i, results)) for i in range(n_conc)]
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.perf_counter() - wall0; m1 = metrics()
    gd = m1.get("gen",0) - m0.get("gen",0)
    ad = m1.get("acc",0) - m0.get("acc",0); dd = m1.get("draft",0) - m0.get("draft",0)
    per = [results[i] for i in sorted(results)]
    print(f"  concurrency={n_conc}: per-stream decode {[round(p[1],1) for p in per]} tok/s | "
          f"server-agg {gd/wall:.1f} tok/s | acceptance {ad/dd if dd else float('nan'):.3f}")
    return gd/wall

print(f"target {BASE}  model {MODEL}")
print("warmup...");
wr={}; stream(0, wr); wr={}; stream(0, wr)
print("warmup done")
SWEEP = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else "1,2,4,8,16".split(","))]
best = {}
for c in SWEEP:
    print(f"=== concurrency {c} ===")
    vals = [run_round(c) for _ in range(2)]
    best[c] = max(vals)
print("\n=== AGGREGATE tok/s curve (best of 2) ===")
for c in SWEEP:
    print(f"  x{c:<2}: {best[c]:.1f} tok/s agg  ({best[c]/c:.1f}/stream)")
