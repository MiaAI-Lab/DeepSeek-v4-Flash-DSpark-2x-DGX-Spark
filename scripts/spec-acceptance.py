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
import re
import statistics
import subprocess
import sys
import time
import urllib.request

POSITION_RE = re.compile(r'position="(\d+)"')


def get_metrics(base_url: str) -> dict:
    url = base_url.removesuffix("/v1") + "/metrics"
    with urllib.request.urlopen(url, timeout=30) as r:
        txt = r.read().decode()
    out = {"drafted": None, "accepted": None, "drafts": None, "per_pos": {}}
    for line in txt.splitlines():
        if line.startswith("vllm:spec_decode_num_draft_tokens_total"):
            out["drafted"] = float(line.split()[-1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            out["accepted"] = float(line.split()[-1])
        elif line.startswith("vllm:spec_decode_num_drafts_total"):
            out["drafts"] = float(line.split()[-1])
        # Anchor on _total: the sibling _created gauge carries a Unix timestamp.
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total"):
            # Match the label directly. Splitting on "," assumes position= is
            # followed by another label, but this build emits it last, so the old
            # parse produced '0"} 40263.0' and the bare except dropped every
            # position — leaving the curve permanently empty.
            hit = POSITION_RE.search(line)
            if hit:
                out["per_pos"][int(hit.group(1))] = float(line.split()[-1])
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
    if m1["drafted"] is None or m1["accepted"] is None:
        print("NO draft counters in /metrics — is spec-decode on? (check MTP_NUM_TOKENS)")
        return 0
    print(f"before: drafted={m1['drafted']:.0f} accepted={m1['accepted']:.0f}", file=sys.stderr)

    cmd = [sys.executable, args.bench_script, "--base-url", args.base_url,
           "--model", args.model, "--prompt", str(args.prompt),
           "--concurrency", "1", "--repeat", str(args.trials)]
    subprocess.run(cmd, capture_output=True)

    m2 = get_metrics(args.base_url)
    if m2["drafted"] is None or m2["accepted"] is None:
        print("NO draft counters in /metrics — is spec-decode on? (check MTP_NUM_TOKENS)")
        return 0
    d = m2["drafted"] - (m1["drafted"] or 0)
    a = (m2["accepted"] or 0) - (m1["accepted"] or 0)
    print(f"after:  drafted={m2['drafted']:.0f} accepted={m2['accepted']:.0f}")

    print(f"\nDELTA over {args.trials} trials: drafted={d:.0f} accepted={a:.0f}")
    if d > 0:
        rate = a / d * 100
        print(f"OVERALL ACCEPTANCE = {rate:.1f}%")
        n_drafts = (m2["drafts"] or 0.0) - (m1["drafts"] or 0.0)
        # Per-position curve over THIS window. Reporting m2 alone would print the
        # container's lifetime totals, which no longer describe the burst above.
        print("\nper-position acceptance (this window):")
        for pos in sorted(m2["per_pos"]):
            v = m2["per_pos"][pos] - m1["per_pos"].get(pos, 0.0)
            if n_drafts > 0:
                print(f"  pos{pos}: {v / n_drafts:.3f}  ({v:.0f}/{n_drafts:.0f})")
            elif v:
                print(f"  pos{pos}: {v:.0f} accepted")
    else:
        print("NO draft activity in window — is spec-decode on? (check MTP_NUM_TOKENS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
