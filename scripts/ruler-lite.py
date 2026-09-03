#!/usr/bin/env python3
"""RULER-lite: synthetic long-context eval for DS4 DSpark serving.

Reference: NVIDIA RULER (arXiv:2404.06654) task generators, adapted to run LIVE
against an OpenAI-compatible endpoint with tokenize-verified context lengths.
Prompts follow the canonical RULER templates including the ANSWER PREFIX (the
"Answer: ... they are:" continuation that primes the model to emit the answer)
and canonical formats (5-char uppercase var names, numbered word lists).

Tasks (3 families beyond shallow NIAH):
  1. sniah / mkniah — single & multi-key retrieval ("magic number" needles)
  2. vartrack       — multi-hop variable-tracking (coreference chains)
  3. cwe            — common-words extraction (aggregation)

Usage:
  python3 ruler-lite.py [--base-url http://127.0.0.1:8888/v1] [--model deepseek-v4-flash-0731]
      [--lengths 8192,32768,131072,262144] [--request-timeout 3600]
      [--output results/ruler-lite.json]
Exit 0 = all tasks at all lengths pass, 1 = any failure (CI-able).
"""
import argparse
import json
import math
import random
import re
import string
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

TOK_URL_SUFFIX = "/tokenize"
CHAT_URL_SUFFIX = "/chat/completions"

HAYSTACK_SENTENCE = ("The grass is green. The sky is blue. The sun is yellow. "
                     "Here we go. There and back again.")

# CWE vocabulary (wonderwords-style; synthetic, knowledge-free)
CWE_WORDS = ["apple", "banana", "cherry", "dragon", "eagle", "forest", "garden",
             "harbor", "island", "jungle", "kettle", "lantern", "mountain",
             "needle", "ocean", "piano", "quartz", "river", "silver", "temple",
             "umbrella", "valley", "winter", "yellow", "zephyr", "anchor",
             "bridge", "candle", "dolphin", "ember", "feather", "glacier",
             "harbor", "ivory", "jasmine", "kingdom", "lagoon", "mirror",
             "night", "orange", "prairie", "quiver", "rainbow", "sunset",
             "thunder", "utopia", "violet", "willow", "xenon", "yonder"]


# Client-side HTTP timeout for every /tokenize and /chat/completions call,
# overridable with --request-timeout. It only covers the depths this script
# defaults to. Recorded on this cluster (docs/DEEPSEEK_V4_FLASH_0731.md,
# results/RESULTS-2026-08-14.md): an 899,994-token prompt took 1,028.85 s to
# first token at ~874.8 prefill tok/s. At that rate 900 s buys ~787k tokens of
# prefill, so a ~900k case hangs up mid-prefill and the harness scores its own
# client limit as a model FAIL.
REQUEST_TIMEOUT = 900.0


def request_json(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return json.load(resp)


def tokenize(base_url: str, model: str, text: str) -> int:
    return request_json(base_url.removesuffix("/v1") + TOK_URL_SUFFIX,
                        {"model": model, "prompt": text})["count"]


def haystack_reps(needed_tokens: int, unit_tokens: int) -> int:
    """How many haystack copies to append to close `needed_tokens`."""
    unit = max(1, int(unit_tokens))
    return max(1, int(needed_tokens) // unit + 1)


def pad_to_length(base_url: str, model: str, text: str, target: int,
                  tokenize_fn=None) -> str:
    """Pad with haystack noise until /tokenize reports >= target tokens.

    Appends in bulk. A one-sentence-per-loop cap of 200 used to ceiling every
    prompt at ~4.8k tokens (issue #81), so 32k/262k RULER-lite cells were fake.
    """
    tok = tokenize_fn or (lambda t: tokenize(base_url, model, t))
    unit = tok(HAYSTACK_SENTENCE) or 1
    n = tok(text)
    guard = 0
    while n < target and guard < 40:
        text += (" " + HAYSTACK_SENTENCE) * haystack_reps(target - n, unit)
        n = tok(text)
        guard += 1
    return text


def chat(base_url: str, model: str, prompt: str, max_tokens: int = 512,
         temperature: float = 0.0, thinking_key: str = "thinking") -> tuple[str, float]:
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature,
            "chat_template_kwargs": {thinking_key: False}}
    t0 = time.perf_counter()
    d = request_json(base_url + CHAT_URL_SUFFIX, body)
    wall = time.perf_counter() - t0
    return (d["choices"][0]["message"].get("content") or "").strip(), wall


# ---------------------------------------------------------------- tasks ----
def task_sniah(rng: random.Random) -> tuple[str, list[str], str, str]:
    key = rng.choice(CWE_WORDS)
    value = str(rng.randint(10000, 99999))
    needle = f"One of the special magic numbers for {key} is: {value}."
    prompt = ("Below is a collection of statements. "
              f"What is the special magic number for {key} mentioned in the text? "
              "Answer with the number only.\n\n" + needle)
    return prompt, [value], "", f"sniah(key={key})"


def task_mkniah(rng: random.Random, n_keys: int = 3) -> tuple[str, list[str], str, str]:
    keys = rng.sample(CWE_WORDS, n_keys)
    values = {k: str(rng.randint(10000, 99999)) for k in keys}
    query_key = rng.choice(keys)
    needles = " ".join(f"One of the special magic numbers for {k} is: {v}."
                       for k, v in values.items())
    prompt = ("Below is a collection of statements. "
              f"What is the special magic number for {query_key} mentioned in the text? "
              "Answer with the number only.\n\n" + needles)
    return prompt, [values[query_key]], "", f"mkniah(keys={n_keys},query={query_key})"


def task_vartrack(rng: random.Random, n_chains: int = 1, n_hops: int = 4) -> tuple[str, list[str], str, str]:
    """RULER canonical variable tracking (from NVIDIA/RULER variable_tracking.py).

    One chain of n_hops+1 vars initialized to a random 5-digit value, then hops
    'VAR X = VAR Y'. Answer prefix primes: "... N variables are assigned the
    value V, they are:". Gold = all var names in the chain.
    """
    k = 5
    num_vars = (n_hops + 1) * n_chains
    vars_all = []
    while len(set(vars_all)) < num_vars:
        vars_all = [''.join(rng.choices(string.ascii_uppercase, k=k)) for _ in range(num_vars)]

    chains = []
    for i in range(0, len(vars_all), n_hops + 1):
        this_vars = vars_all[i:i + n_hops + 1]
        chain = [f"VAR {this_vars[0]} = {rng.randint(10000, 99999)}"]
        for j in range(n_hops):
            chain.append(f"VAR {this_vars[j + 1]} = VAR {this_vars[j]} ")
        chains.append(chain)

    # query value = first chain's initial value (all chains share in RULER? no:
    # each chain has its own value; gold = the queried chain's vars)
    value = chains[0][0].split("=")[-1].strip()
    gold_vars = vars_all[:n_hops + 1]

    # interleave chains with noise sentences, canonical shuffle
    sentences = [HAYSTACK_SENTENCE] * 20
    flat = []
    for chain in chains:
        flat.extend(chain)
    positions = sorted(rng.sample(range(len(sentences) + len(flat)), len(flat)))
    si = 0
    out = []
    pi = 0
    for i in range(len(sentences) + len(flat)):
        if pi < len(positions) and i == positions[pi]:
            out.append(flat[pi])
            pi += 1
        else:
            out.append(sentences[si])
            si += 1
    context = "\n".join(out)

    template = ("Memorize and track the chain(s) of variable assignment hidden in the "
                "following text.\n\n{context}\nQuestion: Find all variables that are "
                "assigned the value {query} in the text above.")
    answer_prefix = (f"Answer: According to the chain(s) of variable assignment in the "
                     f"text above, {n_hops + 1} variables are assigned the value {value}, "
                     f"they are: ")
    prompt = template.format(context=context, query=value)
    return prompt, gold_vars, answer_prefix, f"vartrack(chains={n_chains},hops={n_hops})"


def task_cwe(rng: random.Random, n_common: int = 5, freq_cw: int = 10,
             freq_ucw: int = 3) -> tuple[str, list[str], str, str]:
    """RULER canonical common-words extraction (numbered list, answer prefix)."""
    words = list(CWE_WORDS)
    rng.shuffle(words)
    common = words[:n_common]
    uncommon = words[n_common:]
    word_list = common * freq_cw + uncommon * freq_ucw
    rng.shuffle(word_list)
    context = ' '.join(f"{i + 1}. {w}" for i, w in enumerate(word_list))
    template = ("Below is a numbered list of words. In these words, some appear more "
                "often than others. Memorize the ones that appear most often.\n"
                "{context}\nQuestion: What are the {num_cw} most common words in the "
                "above list?")
    answer_prefix = (f"Answer: The top {n_common} words that appear most often in the "
                     "list are: ")
    prompt = template.format(context=context, num_cw=n_common)
    return prompt, common, answer_prefix, f"cwe(common={n_common})"


# ------------------------------------------------------------ scoring ------
def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", " ", s.lower())


def score_answer(pred: str, golds: list[str], task: str) -> bool:
    p = norm(pred)
    if task.startswith("sniah") or task.startswith("mkniah"):
        nums = re.findall(r"\d{4,5}", p)
        return any(g in nums for g in golds)
    if task.startswith("vartrack"):
        gp = {norm(g) for g in golds}
        pp = set(p.split())
        return gp.issubset(pp)
    if task.startswith("cwe"):
        gp = {norm(g) for g in golds}
        pp = set(p.split())
        return gp.issubset(pp)
    return any(norm(g) == p for g in golds)


def run_case(base_url: str, model: str, length: int, make_task, rng: random.Random,
             thinking_key: str = "thinking", max_tokens: int = 512):
    result = make_task(rng)
    if len(result) == 4:
        prompt, golds, answer_prefix, task = result
        prompt = prompt + answer_prefix  # prime continuation
    else:
        prompt, golds, task = result
    padded = pad_to_length(base_url, model, prompt, length)
    actual = tokenize(base_url, model, padded)
    if actual < int(length * 0.97):
        raise RuntimeError(
            f"padding fell short: {actual} tokens for a target of {length}")
    pred, wall = chat(base_url, model, padded, thinking_key=thinking_key, max_tokens=max_tokens)
    ok = score_answer(pred, golds, task)
    return {"task": task, "target": length, "actual_tokens": actual,
            "ok": ok, "prediction": pred[:100], "gold": golds, "secs": round(wall, 1)}


def positive_timeout(value: str) -> float:
    """argparse type for --request-timeout: finite and strictly positive.

    Plain `float` would accept 0, negatives, `nan` and `inf`; a socket timeout
    trips instantly on the first three and OverflowErrors on the last, which
    would resurface as a per-case FAIL rather than a usage error.
    """
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError(
            f"expected a finite number of seconds > 0, got {value!r}")
    return seconds


def main() -> int:
    global REQUEST_TIMEOUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    ap.add_argument("--model", default="deepseek-v4-flash-0731")
    ap.add_argument("--lengths", default="8192,32768,131072,262144")
    ap.add_argument("--tasks", default="sniah,mkniah,vartrack,cwe")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="")
    ap.add_argument("--thinking-key", default="thinking",
                    help="chat_template_kwargs key to disable thinking: 'thinking' (DeepSeek) "
                         "or 'enable_thinking' (Qwen3). Qwen3 ignores 'thinking' and burns the "
                         "budget on reasoning -> empty content -> false FAILs.")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="generation budget; verbose models (Qwen3.8-27B) write prose before "
                         "the answer, so 256 truncates the word list -> false FAIL. 512 covers it.")
    ap.add_argument("--request-timeout", type=positive_timeout,
                    default=REQUEST_TIMEOUT, metavar="SECONDS",
                    help="client HTTP timeout in seconds (default: %(default)s). A cold "
                         "prefill near the 1M ceiling outruns it: a recorded 899,994-token "
                         "prompt needed 1,028.85 s to first token at ~874.8 prefill tok/s, "
                         "and 900 s only buys ~787k tokens, so the client hangs up "
                         "mid-prefill and the harness scores its own limit as a model FAIL. "
                         "Raise it for --lengths past ~790k.")
    args = ap.parse_args()
    REQUEST_TIMEOUT = args.request_timeout

    lengths = [int(x) for x in args.lengths.split(",")]
    task_map = {"sniah": task_sniah, "mkniah": task_mkniah,
                "vartrack": task_vartrack, "cwe": task_cwe}
    tasks = [task_map[t] for t in args.tasks.split(",") if t in task_map]
    rng = random.Random(args.seed)
    out = Path(args.output or f"results/ruler-lite-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    report = {"model": args.model, "base_url": args.base_url, "seed": args.seed, "cases": []}
    failures = 0
    for length in lengths:
        for make_task in tasks:
            try:
                case = run_case(args.base_url, args.model, length, make_task, rng, args.thinking_key, args.max_tokens)
            except urllib.error.HTTPError as e:
                case = {"task": make_task.__name__, "target": length, "ok": False,
                        "error": f"HTTP {e.code}: {e.read().decode()[:200]}"}
            except Exception as e:
                case = {"task": make_task.__name__, "target": length, "ok": False,
                        "error": str(e)[:200]}
            report["cases"].append(case)
            out.write_text(json.dumps(report, indent=2) + "\n")
            tag = "PASS" if case.get("ok") else "FAIL"
            if not case.get("ok"):
                failures += 1
            detail = case.get("prediction", case.get("error", ""))
            print(f"{tag:4s} {case['task']:30s} ctx={case.get('actual_tokens', '?'):>8} "
                  f"({case.get('secs', 0):>5.0f}s) gold={case.get('gold')} pred={detail!r}",
                  flush=True)

    print(f"\n=== RULER-lite: {len(report['cases']) - failures}/{len(report['cases'])} passed ===")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
