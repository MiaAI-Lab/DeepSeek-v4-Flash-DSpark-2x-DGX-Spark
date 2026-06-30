#!/usr/bin/env python3
"""Realistic STAGGERED-arrival concurrency test (exercises ragged mixed
prefill+decode = the Patch-2 path). N comparable requests launched with a
delay between each so they overlap at different phases. Measures server-side
aggregate tok/s + draft acceptance, and verifies every request succeeds (200)."""
import json, time, urllib.request, sys, re, threading

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8888"
N    = int(sys.argv[2]) if len(sys.argv) > 2 else 4
STAGGER = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
MODEL = "deepseek-v4-flash-dspark"
PROMPT = ("Write a complete, idiomatic Python implementation of a binary search "
          "tree with insert, delete, search, in-order traversal, and height; "
          "include docstrings and a small test harness. Be thorough.")

def metrics():
    try:
        with urllib.request.urlopen(BASE+"/metrics",timeout=10) as r: t=r.read().decode()
    except: return (0,0,0)
    g=lambda n: sum(float(x) for x in re.findall(r"^%s(?:\{[^}]*\})?\s+([0-9.eE+-]+)$"%re.escape(n),t,re.M)) or 0.0
    return (g("vllm:generation_tokens_total"),
            g("vllm:spec_decode_num_accepted_tokens_total"),
            g("vllm:spec_decode_num_draft_tokens_total"))

def run(idx, res):
    body={"model":MODEL,"messages":[{"role":"user","content":PROMPT}],
          "temperature":0.0,"max_tokens":256,"stream":False,"ignore_eos":True,
          "chat_template_kwargs":{"thinking":False}}
    req=urllib.request.Request(BASE+"/v1/chat/completions",
        data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    t0=time.perf_counter()
    try:
        with urllib.request.urlopen(req,timeout=600) as r:
            o=json.loads(r.read().decode())
        ct=o["choices"][0]["message"]["content"]
        res[idx]=("ok", len(ct), time.perf_counter()-t0)
    except Exception as e:
        res[idx]=("ERR", str(e)[:60], time.perf_counter()-t0)

print(f"target {BASE}  N={N}  stagger={STAGGER}s")
print("warmup..."); w={}; run(0,w)
g0,a0,d0=metrics(); wall0=time.perf_counter()
res={}; threads=[]
for i in range(N):
    t=threading.Thread(target=run,args=(i,res)); t.start(); threads.append(t)
    time.sleep(STAGGER)   # staggered arrival -> mixed prefill+decode
for t in threads: t.join()
wall=time.perf_counter()-wall0
g1,a1,d1=metrics()
ok=sum(1 for v in res.values() if v[0]=="ok"); err=N-ok
acc=(a1-a0)/(d1-d0) if d1>d0 else float('nan')
print(f"requests: {ok}/{N} ok, {err} errors")
for i in sorted(res): print(f"  req{i}: {res[i][0]} len={res[i][1]} {res[i][2]:.1f}s")
print(f"server aggregate gen tok/s over window: {(g1-g0)/wall:.1f}")
print(f"draft acceptance (all reqs): {acc:.3f}")
print("VERDICT:", "PASS (no errors, ragged/staggered works)" if err==0 else "FAIL (errors under staggered load)")
