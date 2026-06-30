#!/usr/bin/env python3
"""Needle-in-haystack + DSpark acceptance sweep over context length.

For each target context/depth:
- Build long haystack with a unique code at a specified depth.
- Ask model to retrieve the code and continue with enough deterministic text to
  create a useful speculative-decoding acceptance sample.
- Read vLLM metrics before/after to compute draft acceptance deltas.
- Preserve full JSONL traces and a summary JSON.
"""
import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_SENTENCES = [
    "The cartographer annotated river deltas, mountain passes, trade routes, and forgotten observatories with careful measurements and cautious notes.",
    "An archive of botanical surveys described moss, cedar, lichen, fern, and mycelium across coastal valleys and high desert basins.",
    "Engineers reviewing suspension bridges compared catenary curves, cable fatigue, wind shear, rivet tolerances, and maintenance schedules.",
    "A museum catalog listed bronze tools, ceramic fragments, woven textiles, pigments, coins, maps, letters, and instruments in chronological order.",
    "Marine biologists recorded octopus behavior, coral bleaching patterns, plankton blooms, current velocity, salinity, and water temperature.",
    "The observatory log summarized eclipses, variable stars, telescope calibrations, atmospheric seeing, mirror alignment, and photometric errors.",
    "Linguists documented vowel shifts, case endings, idioms, loan words, inscriptions, dialect boundaries, and regional storytelling traditions.",
    "A compiler note explained register allocation, loop unrolling, cache locality, branch prediction, vector lanes, and profiling counters.",
    "The expedition journal mentioned lantern fuel, rope length, weather changes, basalt cliffs, glacier melt, and coordinates copied twice.",
    "The mathematics appendix reviewed prime gaps, Fourier coefficients, Markov chains, graph cuts, convexity, eigenvectors, and modular forms.",
]


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def http_json(url, payload=None, timeout=600):
    headers = {"Content-Type": "application/json", "Authorization": "Bearer local"}
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="GET" if payload is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_text(url, timeout=20):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def metric_sum(text, name):
    pat = re.compile(r"^" + re.escape(name) + r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", re.M)
    return sum(float(m.group(1)) for m in pat.finditer(text))


def metric_pos(text):
    out = {}
    pat = re.compile(r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"[^}]*\}\s+([0-9.eE+-]+)$', re.M)
    for m in pat.finditer(text):
        out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def read_metrics(base):
    text = http_text(base.rstrip('/') + '/metrics')
    return {
        "raw_time": now_iso(),
        "prompt": metric_sum(text, "vllm:prompt_tokens_total"),
        "generation": metric_sum(text, "vllm:generation_tokens_total"),
        "drafts": metric_sum(text, "vllm:spec_decode_num_drafts_total"),
        "draft_tokens": metric_sum(text, "vllm:spec_decode_num_draft_tokens_total"),
        "accepted_tokens": metric_sum(text, "vllm:spec_decode_num_accepted_tokens_total"),
        "accepted_by_pos": metric_pos(text),
    }


def delta_metrics(a, b):
    d = {k: b.get(k, 0) - a.get(k, 0) for k in ["prompt", "generation", "drafts", "draft_tokens", "accepted_tokens"]}
    pos_keys = set(a.get("accepted_by_pos", {})) | set(b.get("accepted_by_pos", {}))
    d["accepted_by_pos"] = {k: b.get("accepted_by_pos", {}).get(k, 0) - a.get("accepted_by_pos", {}).get(k, 0) for k in sorted(pos_keys, key=int)}
    d["acceptance_rate"] = (d["accepted_tokens"] / d["draft_tokens"]) if d["draft_tokens"] else None
    d["accepted_per_draft"] = (d["accepted_tokens"] / d["drafts"]) if d["drafts"] else None
    return d


def make_haystack(target_chars, needle, depth):
    rng = random.Random(12345)
    chunks = []
    total = 0
    i = 0
    while total < target_chars:
        s = BASE_SENTENCES[i % len(BASE_SENTENCES)]
        # Low-entropy enough for deterministic tokenization, varied enough to avoid pathological repetition.
        chunk = f"[Record {i:06d}] {s} Reference number {rng.randrange(10**8, 10**9)}.\n"
        chunks.append(chunk)
        total += len(chunk)
        i += 1
    hay = ''.join(chunks)
    split = int(len(hay) * depth)
    return hay[:split] + f"\n\nNEEDLE FACT: The DSV4 DSpark needle code is {needle}.\n\n" + hay[split:]


def run_one(base, model, target_tokens, depth, chars_per_token, max_tokens, outdir, strict_code_only=False):
    needle = f"CTX-{target_tokens//1000}K-D{int(depth*100):02d}-R7Q9"
    # Leave room for system/question overhead. Calibration is approximate; actual usage is captured.
    target_chars = int(target_tokens * chars_per_token)
    doc = make_haystack(target_chars, needle, depth)
    if strict_code_only:
        prompt = (
            "TASK: Retrieve one exact code from the document.\n"
            "The answer is on a line that starts exactly with: NEEDLE FACT:\n"
            "Output ONLY the code token after 'is'. No prose. No quotes. No explanation.\n\n"
            "<document>\n" + doc + "\n</document>\n\n"
            "Output ONLY the DSV4 DSpark needle code now:"
        )
        system_prompt = (
            "You are an exact substring retrieval engine. Never summarize. "
            "Never continue the document. Output only the requested code."
        )
    else:
        prompt = (
            "You will receive a long document. Find the DSV4 DSpark needle code.\n"
            "Return the code on the first line exactly. Then write five concise bullet points explaining how you found it.\n"
            "Do not invent a different code.\n\n"
            "<document>\n" + doc + "\n</document>\n\n"
            "Question: What is the DSV4 DSpark needle code?"
        )
        system_prompt = "You are a precise long-context retrieval system. Answer deterministically."
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"thinking": False},
    }
    if strict_code_only:
        payload["stop"] = ["\n", " "]
    label = f"{target_tokens//1000:03d}k_d{int(depth*100):02d}"
    m0 = read_metrics(base)
    t0 = time.perf_counter()
    error = None
    resp_json = None
    try:
        resp_json = http_json(base.rstrip('/') + '/v1/chat/completions', payload, timeout=1200)
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', errors='replace')[:2000]
        error = f"HTTP {e.code}: {body}"
    except Exception as e:
        error = repr(e)
    elapsed = time.perf_counter() - t0
    # Give metrics exporter a moment to flush.
    time.sleep(0.5)
    m1 = read_metrics(base)
    md = delta_metrics(m0, m1)

    content = ""
    usage = {}
    if resp_json:
        usage = resp_json.get("usage", {}) or {}
        try:
            msg = resp_json["choices"][0]["message"]
            content = (msg.get("content") or "") + "\n" + (msg.get("reasoning_content") or "")
        except Exception:
            content = json.dumps(resp_json)[:1000]
    passed = (needle.upper() in content.upper()) if not error else False
    rec = {
        "label": label,
        "timestamp": now_iso(),
        "target_tokens": target_tokens,
        "needle_depth": depth,
        "target_chars": target_chars,
        "needle": needle,
        "passed": passed,
        "error": error,
        "elapsed_sec": round(elapsed, 3),
        "usage": usage,
        "metrics_delta": md,
        "response_preview": content.strip()[:1000],
    }
    (outdir / f"{label}.json").write_text(json.dumps({"record": rec, "response": resp_json}, indent=2), encoding="utf-8")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8888")
    ap.add_argument("--model", default="deepseek-v4-flash-dspark")
    ap.add_argument("--tokens", nargs="+", type=int, default=[8000, 32000, 64000, 128000, 180000])
    ap.add_argument("--depths", nargs="+", type=float, default=[0.1, 0.5, 0.9])
    ap.add_argument("--chars-per-token", type=float, default=3.35)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--strict-code-only", action="store_true", help="Use exact-code-only retrieval prompt and stop after the code; recommended for 1M sweeps.")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    stamp = datetime.now().strftime("needle_acceptance_%Y%m%d_%H%M%S")
    outdir = Path(args.outdir or f"benchmark-results/{stamp}").resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"started_at": now_iso(), "outdir": str(outdir)}
    (outdir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Health/model metadata
    try:
        models = http_json(args.base.rstrip('/') + "/v1/models", None, timeout=10)
    except Exception as e:
        models = {"error": repr(e)}
    (outdir / "models.json").write_text(json.dumps(models, indent=2), encoding="utf-8")

    results = []
    jsonl = (outdir / "results.jsonl").open("w", encoding="utf-8")
    print(f"DSpark needle acceptance sweep -> {outdir}", flush=True)
    print(f"base={args.base} model={args.model} tokens={args.tokens} depths={args.depths}", flush=True)
    for tok in args.tokens:
        for depth in args.depths:
            print(f"RUN target={tok} depth={depth:.2f}", flush=True)
            rec = run_one(args.base, args.model, tok, depth, args.chars_per_token, args.max_tokens, outdir, args.strict_code_only)
            results.append(rec)
            jsonl.write(json.dumps(rec) + "\n"); jsonl.flush()
            ar = rec["metrics_delta"].get("acceptance_rate")
            pt = rec.get("usage", {}).get("prompt_tokens")
            ct = rec.get("usage", {}).get("completion_tokens")
            print(f"  {'PASS' if rec['passed'] else 'FAIL'} prompt={pt} completion={ct} accept={ar if ar is not None else 'NA'} elapsed={rec['elapsed_sec']}s", flush=True)
    jsonl.close()

    summary = {
        "config": config,
        "models": models,
        "completed_at": now_iso(),
        "n": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "results": results,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Markdown summary
    lines = ["# DSpark Needle Acceptance Sweep", "", f"Started: {config['started_at']}", f"Endpoint: `{args.base}`", f"Model: `{args.model}`", "", "| target | actual prompt | depth | retrieval | acceptance | accepted/draft | gen tok | elapsed |", "|---:|---:|---:|---|---:|---:|---:|---:|"]
    for r in results:
        md = r["metrics_delta"]
        usage = r.get("usage", {})
        lines.append(f"| {r['target_tokens']:,} | {usage.get('prompt_tokens','?')} | {int(r['needle_depth']*100)}% | {'PASS' if r['passed'] else 'FAIL'} | {md.get('acceptance_rate') if md.get('acceptance_rate') is not None else 'NA'} | {md.get('accepted_per_draft') if md.get('accepted_per_draft') is not None else 'NA'} | {md.get('generation')} | {r['elapsed_sec']}s |")
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"DONE {summary['passed']}/{summary['n']} passed; summary={outdir/'README.md'}", flush=True)

if __name__ == "__main__":
    main()
