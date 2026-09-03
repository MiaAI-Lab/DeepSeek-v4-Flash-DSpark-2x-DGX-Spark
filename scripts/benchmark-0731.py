import argparse
import asyncio
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


def request_json(url, body):
    for attempt in range(4):
        request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=3600) as response:
                return json.load(response)
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


def stream_one(base_url, model, prompt, thinking_token_budget=None, max_tokens=4096):
    instruction = "\nReturn exactly 128 numbered lowercase English words, then stop."
    # max_tokens is mandatory here: the server's DEFAULT_THINKING can put reasoning
    # on by default, and reasoning is not obliged to reach EOS. Without a cap a
    # single lane can generate for hours (observed: >200k tokens, ~75 tok/s, no
    # error) and stall the whole sweep behind it.
    body = {"model": model, "messages": [{"role": "user", "content": prompt + instruction}], "stream": True, "stream_options": {"include_usage": True}, "temperature": 0.6, "top_p": 0.95, "max_tokens": max_tokens}
    if thinking_token_budget is not None:
        body["thinking_token_budget"] = thinking_token_budget
    request = urllib.request.Request(f"{base_url}/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    first = None
    usage = None
    output = []
    finish_reason = None
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            choices = event.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            if choices and choices[0].get("finish_reason"):
                finish_reason = choices[0]["finish_reason"]
            if first is None and (delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")):
                first = time.perf_counter()
            reasoning = delta.get("reasoning") or delta.get("reasoning_content") or ""
            content = delta.get("content") or ""
            output.extend((reasoning, content))
            if event.get("usage"):
                usage = event["usage"]
    finished = time.perf_counter()
    measured = None if usage else request_json(tokenize_url(base_url), {"model": model, "prompt": "".join(output)})["count"]
    output_tokens = (usage or {}).get("completion_tokens", measured or 0)
    ttft = (first or finished) - started
    prompt_tokens = (usage or {}).get("prompt_tokens", 0)
    # truncated == the cap bound this request, so output_tokens is a floor rather
    # than the model's natural length. Rates stay valid; total-length claims do not.
    return {"ttft_s": ttft, "elapsed_s": finished - started, "prompt_tokens": prompt_tokens, "prefill_tok_s": prompt_tokens / max(0.001, ttft), "output_tokens": output_tokens, "output_tok_s": output_tokens / max(0.001, finished - (first or finished)), "finish_reason": finish_reason, "truncated": finish_reason == "length"}


async def run_case(base_url, model, target_prompt_tokens, concurrency, thinking_token_budget=None, max_tokens=4096):
    prompts = await asyncio.gather(*[
        asyncio.to_thread(build_prompt, base_url, model, target_prompt_tokens, f"p{target_prompt_tokens}-c{concurrency}-r{index}")
        for index in range(concurrency)
    ])
    started = time.perf_counter()
    results = await asyncio.gather(*[
        asyncio.to_thread(stream_one, base_url, model, prompt, thinking_token_budget, max_tokens)
        for prompt in prompts
    ])
    elapsed = time.perf_counter() - started
    total = sum(item["output_tokens"] for item in results)
    return {"concurrency": concurrency, "elapsed_s": elapsed, "aggregate_tok_s": total / max(0.001, elapsed), "median_ttft_s": statistics.median(item["ttft_s"] for item in results), "median_prefill_tok_s": statistics.median(item["prefill_tok_s"] for item in results), "median_output_tok_s": statistics.median(item["output_tok_s"] for item in results), "truncated_requests": sum(1 for item in results if item["truncated"]), "requests": results}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--prompt-lengths", default="256,2048,8192,32768,131072")
    parser.add_argument("--concurrency", default="1,2,4,6")
    parser.add_argument("--thinking-token-budget", type=int)
    parser.add_argument("--max-tokens", type=int, default=4096, help="Per-request output cap. Guards against unbounded reasoning under DEFAULT_THINKING=high/max; raise it if truncated_requests is non-zero and you need natural lengths.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"model": args.model, "base_url": args.base_url, "max_tokens": args.max_tokens, "cases": []}
    for prompt_length in [int(value) for value in args.prompt_lengths.split(",")]:
        for concurrency in [int(value) for value in args.concurrency.split(",")]:
            case = await run_case(
                args.base_url,
                args.model,
                prompt_length,
                concurrency,
                args.thinking_token_budget,
                args.max_tokens,
            )
            case["target_prompt_tokens"] = prompt_length
            case["thinking_token_budget"] = args.thinking_token_budget
            case["max_tokens"] = args.max_tokens
            report["cases"].append(case)
            path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            print(json.dumps(case, sort_keys=True), flush=True)


asyncio.run(main())
