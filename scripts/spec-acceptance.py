#!/usr/bin/env python3
"""spec-acceptance.py — capture DSpark speculative-decoding acceptance from /metrics.

Why: acceptance rate is THE canonical spec-decode health metric (per vLLM docs and
Snowflake/RedHat benchmarks). It measures how often the in-checkpoint DSpark drafter
is accepted by the target — directly determines decode tok/s. Watch it drift after
model swaps (abliteration drains it) or image changes.

Method (verified 2026-08-16):
  1. Read /metrics counters (drafted / accepted totals, per-position accepted)
  2. Run a short MiaAI-methodology burst (unique cold prefixes, min=max=128,
     ignore_eos, thinking=false) so the counters move
  3. Read again; report delta acceptance + per-position curve

Usage:
  python3 spec-acceptance.py [--base-url http://127.0.0.1:8888/v1] [--model deepseek-v4-flash-0731]
      [--trials 5] [--prompt 256]
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.request


def get_metrics(base_url: str) -> dict:
    url = base_url.removesuffix("/v1") + "/metrics"
    with urllib.request.urlopen(url, timeout=30) as r:
        txt = r.read().decode()
    out = {"drafted": None, "accepted": None, "per_pos": {}}
    for line in txt.splitlines():
        if line.startswith("vllm:spec_decode_num_draft_tokens_total"):
            out["drafted"] = float(line.split()[-1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            out["accepted"] = float(line.split()[-1])
        elif "accepted_tokens_per_pos" in line and "{" in line:
            try:
                pos = int(line.split("position=")[1].split(",")[0].strip('"'))
                out["per_pos"][pos] = float(line.split()[-1])
            except Exception:
                pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    ap.add_argument("--model", default="deepseek-v4-flash-0731")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--prompt", type=int, default=256)
    ap.add_argument("--bench-script", default="scripts/bench-miaai.py",
                    help="MiaAI-methodology bench that moves the counters")
    args = ap.parse_args()

    m1 = get_metrics(args.base_url)
    print(f"before: drafted={m1['drafted']:.0f} accepted={m1['accepted']:.0f}", file=sys.stderr)

    cmd = [sys.executable, args.bench_script, "--base-url", args.base_url,
           "--model", args.model, "--prompt", str(args.prompt),
           "--concurrency", "1", "--repeat", str(args.trials)]
    subprocess.run(cmd, capture_output=True)

    m2 = get_metrics(args.base_url)
    if m1["drafted"] is None or m2["drafted"] is None:
        print("NO draft counters in /metrics — is spec-decode on? (check MTP_NUM_TOKENS)")
        return 0
    d = m2["drafted"] - (m1["drafted"] or 0)
    a = (m2["accepted"] or 0) - (m1["accepted"] or 0)
    print(f"after:  drafted={m2['drafted']:.0f} accepted={m2['accepted']:.0f}")

    print(f"\nDELTA over {args.trials} trials: drafted={d:.0f} accepted={a:.0f}")
    if d > 0:
        rate = a / d * 100
        print(f"OVERALL ACCEPTANCE = {rate:.1f}%")
        print(f"tokens accepted per draft = {a/d:.3f} (k=5 -> max 5)")
        # per-position acceptance curve
        print("\nper-position acceptance (pos0..pos4):")
        for pos in sorted(m2["per_pos"]):
            v = m2["per_pos"][pos]
            if v:
                print(f"  pos{pos}: {v:.0f} accepted")
    else:
        print("NO draft activity in window — is spec-decode on? (check MTP_NUM_TOKENS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
