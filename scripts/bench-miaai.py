#!/usr/bin/env python3
"""Faithful replica of MiaAI's 2026-08-14 matrix methodology:
unique cold prefix per request, thinking=false, min_tokens=max_tokens=128,
ignore_eos, numbered-word instruction. Reports median per-stream decode tok/s
after first token (their cell value) + acceptance-ish stats from server logs are
read separately.  Usage: bench_miaai.py --base-url ... --prompt 256 --concurrency 1
"""
import argparse, asyncio, json, statistics, time, urllib.error, urllib.request

def request_json(url, body):
    for attempt in range(4):
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                return json.load(resp)
        except urllib.error.URLError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)

def tokenize_url(base_url):
    return base_url.removesuffix("/v1") + "/tokenize"

def build_prompt(base_url, model, target, nonce):
    unit = "benchmark context datum "
    text = f"unique request {nonce} " + unit * max(1, target // 3)
    while True:
        count = request_json(tokenize_url(base_url), {"model": model, "prompt": text})["count"]
        if count >= target:
            return text
        text += unit * max(1, (target - count) // 3)

def stream_one(base_url, model, prompt):
    instruction = "\nReturn exactly 128 numbered lowercase English words, then stop."
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt + instruction}],
        "stream": True, "stream_options": {"include_usage": True},
        "temperature": 0.6, "top_p": 0.95,
        "max_tokens": 128, "min_tokens": 128, "ignore_eos": True,
        "chat_template_kwargs": {"thinking": False},
    }
    req = urllib.request.Request(f"{base_url}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    first = None
    usage = None
    output = []
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            choices = event.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            if first is None and (delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")):
                first = time.perf_counter()
            output.append(delta.get("content") or "")
            if event.get("usage"):
                usage = event["usage"]
    finished = time.perf_counter()
    output_tokens = (usage or {}).get("completion_tokens", 0)
    ttft = (first or finished) - started
    return {"ttft_s": ttft, "elapsed_s": finished - started,
            "output_tokens": output_tokens,
            "output_tok_s": output_tokens / max(0.001, finished - (first or finished)),
            "prompt_tokens": (usage or {}).get("prompt_tokens", 0)}

async def run_case(base_url, model, target_prompt_tokens, concurrency, nonce_base):
    prompts = await asyncio.gather(*[
        asyncio.to_thread(build_prompt, base_url, model, target_prompt_tokens,
                          f"p{target_prompt_tokens}-c{concurrency}-{nonce_base}-r{index}")
        for index in range(concurrency)
    ])
    started = time.perf_counter()
    results = await asyncio.gather(*[asyncio.to_thread(stream_one, base_url, model, p) for p in prompts])
    elapsed = time.perf_counter() - started
    total = sum(r["output_tokens"] for r in results)
    return {"concurrency": concurrency, "elapsed_s": elapsed,
            "aggregate_tok_s": total / max(0.001, elapsed),
            "median_ttft_s": statistics.median(r["ttft_s"] for r in results),
            "median_output_tok_s": statistics.median(r["output_tok_s"] for r in results),
            "requests": results}

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    ap.add_argument("--model", default="deepseek-v4-flash-0731")
    ap.add_argument("--prompt", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=5, help="how many sequential trials (unique nonces)")
    args = ap.parse_args()
    rows = []
    for rep in range(args.repeat):
        case = await run_case(args.base_url, args.model, args.prompt, args.concurrency, rep)
        rows.append(case)
        med = case["median_output_tok_s"]
        print(f"trial {rep}: c={case['concurrency']} p={args.prompt} "
              f"median_decode={med:.1f} tok/s agg={case['aggregate_tok_s']:.1f} "
              f"ttft={case['median_ttft_s']*1000:.0f}ms n={[r['output_tokens'] for r in case['requests']]}", flush=True)
    if args.repeat > 1:
        meds = sorted(r["median_output_tok_s"] for r in rows)
        print(f"\nFINAL: median-of-trials decode = {statistics.median(meds):.1f} tok/s "
              f"(min {meds[0]:.1f} max {meds[-1]:.1f})")

asyncio.run(main())
