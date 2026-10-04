#!/usr/bin/env python3
"""Decode decomposition from vLLM's Prometheus counters.

Reads /metrics (a URL or a saved dump) and prints what the engine actually spent its time on since the
last restart: prefill vs decode share, how often decode steps carried one sequence, step-time and
per-token quantiles, request shapes, finish reasons, prefix-cache hit rate, and the drafter's
per-position acceptance with the expected tokens per step for every draft depth.

Usage:
    python3 scripts/metrics-decode-report.py                         # http://127.0.0.1:8888/metrics
    python3 scripts/metrics-decode-report.py http://HEAD_NODE_IP:8888/metrics
    python3 scripts/metrics-decode-report.py results/metrics.txt     # a saved dump

Everything here is cumulative since engine start; save a dump before a restart if you want to keep it.
"""
import collections
import re
import sys
import urllib.request

SRC = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8888/metrics"
if SRC.startswith("http"):
    TEXT = urllib.request.urlopen(SRC, timeout=15).read().decode()
else:
    TEXT = open(SRC, encoding="utf-8").read()

LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")
counters = {}
hist = collections.defaultdict(dict)
hsum, hcount = {}, {}


def labels_of(s):
    return dict(re.findall(r'(\w+)="([^"]*)"', s or ""))


for line in TEXT.splitlines():
    if not line.startswith("vllm:"):
        continue
    m = LINE.match(line)
    if not m:
        continue
    name, lab, val = m.group(1), labels_of(m.group(2)), float(m.group(3))
    if name.endswith("_bucket"):
        hist[name[:-7]][lab.get("le")] = val
    elif name.endswith("_sum"):
        hsum[name[:-4]] = val
    elif name.endswith("_count"):
        hcount[name[:-6]] = val
    else:
        key = tuple(sorted((k, v) for k, v in lab.items() if k not in ("engine", "model_name")))
        counters[(name, key)] = val


def c(name, **lab):
    return counters.get((name, tuple(sorted(lab.items()))), 0.0)


def quantiles(base, qs=(0.5, 0.9, 0.99)):
    b = hist.get(base)
    if not b:
        return None
    items = sorted(((float("inf") if le == "+Inf" else float(le)), n) for le, n in b.items())
    total = items[-1][1]
    if total == 0:
        return None
    out = {}
    for q in qs:
        target, prev_le, prev_n = q * total, 0.0, 0.0
        for le, n in items:
            if n >= target:
                out[q] = prev_le if le == float("inf") else prev_le + (target - prev_n) / max(n - prev_n, 1e-9) * (le - prev_le)
                break
            prev_le, prev_n = le, n
    return out, total


def show(base, title, unit="", scale=1.0):
    r = quantiles(base)
    if not r:
        print(f"  {title}: (no data)")
        return
    q, total = r
    mean = hsum.get(base, 0.0) / max(hcount.get(base, 1.0), 1.0)
    first_le, first_n = sorted(((float("inf") if le == "+Inf" else float(le)), n) for le, n in hist[base].items())[0]
    if first_n >= total:   # everything sits in the first bucket: quantiles are not resolvable, only the bound is
        print(f"  {title}: n={int(total):,}  mean={mean*scale:.3g}{unit}  all observations <= {first_le*scale:.3g}{unit} (first bucket)")
        return
    print(f"  {title}: n={int(total):,}  mean={mean*scale:.3g}{unit}  p50={q[0.5]*scale:.3g}{unit}  p90={q[0.9]*scale:.3g}{unit}  p99={q[0.99]*scale:.3g}{unit}  (quantiles interpolated within buckets)")


def table(base, title):
    b = hist.get(base)
    if not b:
        return
    items = sorted(((float("inf") if le == "+Inf" else float(le)), n) for le, n in b.items())
    total = items[-1][1]
    print(f"  {title} (n={int(total):,}):")
    prev, prev_le = 0.0, 0.0
    for le, n in items:
        d = n - prev
        if d > 0:
            lab = f"<= {le:g}" if le != float("inf") else f"> {prev_le:g}"
            print(f"    {lab:>12}: {int(d):>9,}  ({100*d/total:5.1f}%)")
        prev, prev_le = n, le


print("=== REQUESTS (since engine start)")
for reason in ("stop", "length", "abort", "error"):
    print(f"  finished {reason:6}: {int(c('vllm:request_success_total', finished_reason=reason)):,}")
print(f"  prompt tokens: {c('vllm:prompt_tokens_total'):,.0f}   generated: {c('vllm:generation_tokens_total'):,.0f}")
pq, ph = c("vllm:prefix_cache_queries_total"), c("vllm:prefix_cache_hits_total")
print(f"  prefix cache: {ph:,.0f} / {pq:,.0f} = {100*ph/max(pq,1):.1f}% hit")
print(f"  preemptions: {c('vllm:num_preemptions_total'):,.0f}")

print("\n=== PER-REQUEST TIME (summed over requests; overlapping requests double-count)")
ps, ds = hsum.get("vllm:request_prefill_time_seconds"), hsum.get("vllm:request_decode_time_seconds")
if ps is not None and ds is not None:
    print(f"  prefill {ps/3600:.2f} h   decode {ds/3600:.2f} h   -> decode share {100*ds/max(ps+ds,1e-9):.1f}%")
show("vllm:request_queue_time_seconds", "queue wait", " s")
show("vllm:inter_token_latency_seconds", "step time (one observation per decode step)", " ms", 1000)
show("vllm:request_time_per_output_token_seconds", "per-request time per output token", " ms", 1000)
show("vllm:time_to_first_token_seconds", "TTFT", " s")
show("vllm:e2e_request_latency_seconds", "end-to-end", " s")
table("vllm:iteration_tokens_total", "tokens generated per engine step (plus computed prompt tokens on prefill steps)")
# sequences per decode step: inter_token_latency is observed once per sequence per decode step, so its count divided
# by the number of decode steps is the mean concurrency during decode. Decode steps are estimated as engine steps
# minus one pure-prefill step per request. With r sequences per step the single-sequence share is at least 2-r
# (every multi-sequence step a pair); the upper bound comes from the histogram: a step that generated 9-16 tokens
# cannot be one sequence at k=5 (at most k+1 = 6 tokens), so those steps are certainly multi-sequence.
itl_n = hcount.get("vllm:inter_token_latency_seconds")
steps_n = hcount.get("vllm:iteration_tokens_total")
reqs = sum(c("vllm:request_success_total", finished_reason=r) for r in ("stop", "length", "abort", "error"))
if itl_n and steps_n and steps_n > reqs:
    decode_steps = steps_n - reqs
    r = itl_n / decode_steps
    it = {float(le): n for le, n in hist.get("vllm:iteration_tokens_total", {}).items() if le != "+Inf"}
    certainly_multi = it.get(16.0, 0.0) - it.get(8.0, 0.0)
    lower = max(2 - r, 0.0)
    upper = max(1 - certainly_multi / decode_steps, 0.0)
    print(f"  sequences per decode step ~ {r:.2f}  (ITL observations {itl_n:,.0f} / decode steps ~{decode_steps:,.0f}) "
          f"-> single-sequence steps between {100*lower:.0f}% (all extra sequences pairwise) and {100*upper:.0f}% "
          f"(steps that generated 9-16 tokens are certainly multi-sequence)")

print("\n=== REQUEST SHAPES")
show("vllm:request_prompt_tokens", "prompt tokens", " tok")
show("vllm:request_prefill_kv_computed_tokens", "prompt tokens actually computed (cache misses)", " tok")
show("vllm:request_generation_tokens", "generated tokens", " tok")
show("vllm:request_params_max_tokens", "max_tokens requested", " tok")
table("vllm:request_params_max_tokens", "max_tokens requested")
table("vllm:request_prompt_tokens", "prompt tokens per request")

print("\n=== DRAFTER")
drafts = c("vllm:spec_decode_num_drafts_total")
dtok = c("vllm:spec_decode_num_draft_tokens_total")
acc = c("vllm:spec_decode_num_accepted_tokens_total")
if drafts:
    print(f"  drafts {drafts:,.0f}  draft tokens {dtok:,.0f}  accepted {acc:,.0f}  -> acceptance {100*acc/max(dtok,1):.1f}%  "
          f"k={dtok/drafts:.2f}  accepted/draft {acc/drafts:.2f}  tokens/step {1+acc/drafts:.2f}")
    per_pos = {}
    for (name, key), v in counters.items():
        if name in ("vllm:spec_decode_num_accepted_tokens_per_pos", "vllm:spec_decode_num_accepted_tokens_per_pos_total"):
            per_pos[int(dict(key).get("position", -1))] = v
    curve = [per_pos[i] / drafts for i in sorted(per_pos)]
    cum = 0.0
    print("  per-position acceptance P[pos i accepted] and expected tokens/step if k were i+1:")
    for i, p in enumerate(curve):
        cum += p
        cond = f"  P[pos {i} | pos {i-1}] = {100*p/max(curve[i-1],1e-9):5.1f}%" if i else ""
        print(f"    pos {i}: {100*p:5.1f}%   tokens/step at k={i+1}: {1+cum:.3f}{cond}")
    print("  (a flat conditional rate means the drafter's quality does not decay along the draft; "
          "the marginal position is worth its conditional rate times everything before it)")
else:
    print("  no speculative-decoding counters (spec decode off?)")
