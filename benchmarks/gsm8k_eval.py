#!/usr/bin/env python3
"""GSM8K quality eval — compare sequential (single-stream) vs concurrent
(patched batching path) on the same patched DSpark server. Greedy (temp 0):
if the patch is quality-neutral, accuracy matches AND per-question answers agree."""
import json, os, re, sys, time, urllib.request, threading

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8888"
N    = int(sys.argv[2]) if len(sys.argv) > 2 else 200
CONC = int(sys.argv[3]) if len(sys.argv) > 3 else 1
OUT  = sys.argv[4] if len(sys.argv) > 4 else "/tmp/gsm8k_run.json"
MODEL= "deepseek-v4-flash-dspark"
DATA = os.environ.get("GSM8K_DATA", "/home/keyspark/logs/he_runs/gsm8k_test.jsonl")

qs = [json.loads(l) for l in open(DATA)][:N]

def gold(ans):
    m = re.search(r"####\s*([-\d,\.]+)", ans)
    return m.group(1).replace(",", "").rstrip(".") if m else None

def pred(text):
    m = re.findall(r"answer is\s*\$?\s*([-\d,\.]+)", text, re.I)
    if not m:
        m = re.findall(r"(-?[\d,]+(?:\.\d+)?)", text)
    return m[-1].replace(",", "").rstrip(".") if m else None

def ask(idx, res):
    q = qs[idx]["question"]
    body = {"model": MODEL, "messages": [{"role": "user",
            "content": q + "\nSolve step by step. End with 'The answer is <number>'."}],
            "temperature": 0.0, "max_tokens": 400, "stream": False,
            "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(BASE+"/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                txt = json.loads(r.read().decode())["choices"][0]["message"]["content"]
            res[idx] = {"pred": pred(txt), "gold": gold(qs[idx]["answer"])}
            return
        except Exception as e:
            if attempt == 2: res[idx] = {"pred": None, "gold": gold(qs[idx]["answer"]), "err": str(e)[:50]}
            else: time.sleep(2)

res = {}
t0 = time.perf_counter()
from queue import Queue
work = Queue()
for i in range(len(qs)): work.put(i)
def worker():
    while not work.empty():
        try: i = work.get_nowait()
        except: return
        ask(i, res); work.task_done()
threads = [threading.Thread(target=worker) for _ in range(CONC)]
for t in threads: t.start()
for t in threads: t.join()
dt = time.perf_counter() - t0

correct = sum(1 for i in res if res[i]["pred"] is not None and res[i]["pred"] == res[i]["gold"])
acc = correct/len(qs)
json.dump({i: res[i] for i in sorted(res)}, open(OUT,"w"))
print(f"BASE={BASE} N={len(qs)} CONC={CONC}: accuracy={acc:.4f} ({correct}/{len(qs)}) in {dt:.0f}s -> {OUT}")
