#!/usr/bin/env python3
"""Tool-calling battery for DS4 Keys — verifies the recipe's tool fixes work live.
Usage: tool_battery.py [base_url] [model]
Covers: single call, complex schema, multi-turn, parallel calls, thinking+tool,
and the issue55 truncation scenario (max_tokens cut mid-tool-call).
"""
import json, sys, time, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8888/v1/chat/completions"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "deepseek-v4-flash-0731"

TOOLS = [
    {"type": "function", "function": {"name": "search_web", "description": "Search the web for up-to-date information. Use when the answer may have changed recently or you need facts beyond your training.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "send_email", "description": "Send an email to a recipient",
        "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "subject", "body"]}}},
    {"type": "function", "function": {"name": "complex_query", "description": "Run a complex structured query with nested parameters",
        "parameters": {"type": "object", "properties": {"filters": {"type": "object", "properties": {"date_range": {"type": "object", "properties": {"start": {"type": "string"}, "end": {"type": "string"}}}, "tags": {"type": "array", "items": {"type": "string"}}, "limit": {"type": "integer"}}}, "sort": {"type": "string", "enum": ["asc", "desc"]}}, "required": ["filters", "sort"]}}},
]

def call(messages, tools=None, tool_choice=None, max_tokens=800, thinking=None):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    if tools is not None:
        body["tools"] = tools
        body["tool_choice"] = tool_choice or "auto"
    if thinking is not None:
        body["chat_template_kwargs"] = {"thinking": thinking}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    return d, time.perf_counter() - t0

def check_tool_calls(d, expect_n):
    ch = d["choices"][0]; m = ch["message"]
    tcs = m.get("tool_calls") or []
    fr = ch.get("finish_reason")
    ok = len(tcs) == expect_n
    details = []
    for tc in tcs:
        fn = (tc.get("function") or {}).get("name", "?")
        args = (tc.get("function") or {}).get("arguments", "")
        try:
            parsed = json.loads(args)
            details.append(f"{fn}:VALID_JSON")
        except Exception:
            details.append(f"{fn}:BAD_JSON({args[:50]})")
            ok = False
    return ok, fr, details, m.get("content") or ""

print(f"=== Tool-calling battery: {MODEL} ===")
results = {}

# 1. Single tool call
d, w = call([{"role": "user", "content": "What's the weather in Tokyo right now? Use the tool."}], TOOLS)
ok, fr, det, c = check_tool_calls(d, 1)
results["single_call"] = (ok, fr, det)
print(f"1. single_call      {'PASS' if ok else 'FAIL'} finish={fr} {det} ({w:.1f}s)")

# 2. Complex nested schema
d, w = call([{"role": "user", "content": "Query records with tag 'urgent' and 'finance', sorted descending, limit 10, from 2026-01-01 to 2026-08-16."}], TOOLS)
ok, fr, det, c = check_tool_calls(d, 1)
results["complex_schema"] = (ok, fr, det)
print(f"2. complex_schema   {'PASS' if ok else 'FAIL'} finish={fr} {det} ({w:.1f}s)")

# 3. Multi-turn: tool call -> result -> final answer
d, w = call([{"role": "user", "content": "What's the weather in Paris? Use get_weather in celsius."}], TOOLS)
tcs = d["choices"][0]["message"].get("tool_calls") or []
if not tcs:
    results["multiturn"] = (False, d["choices"][0].get("finish_reason"), ["no tool_calls"])
    print(f"3. multiturn        FAIL no tool_calls ({w:.1f}s)")
else:
    tc = tcs[0]
    fn = tc["function"]["name"]; args = json.loads(tc["function"]["arguments"])
    msg2 = {"role": "user", "content": "The weather tool returned: 22C, partly cloudy. Now tell me if I should wear a jacket."}
    d2, w2 = call([{"role": "user", "content": "What's the weather in Paris? Use get_weather in celsius."},
                   {"role": "assistant", "content": None, "tool_calls": [tc]},
                   {"role": "tool", "tool_call_id": tc["id"], "name": fn, "content": "22C partly cloudy"},
                   msg2], TOOLS)
    c2 = d2["choices"][0]["message"].get("content") or ""
    multiturn_ok = bool(c2.strip()) and not (d2["choices"][0]["message"].get("tool_calls"))
    results["multiturn"] = (multiturn_ok, d2["choices"][0].get("finish_reason"), [c2[:60]])
    print(f"3. multiturn        {'PASS' if multiturn_ok else 'FAIL'} content={c2[:60]!r} ({w+w2:.1f}s)")

# 4. Parallel tool calls
d, w = call([{"role": "user", "content": "I need weather for both Berlin and Madrid right now — call get_weather for both."}], TOOLS)
ok, fr, det, c = check_tool_calls(d, 2)
results["parallel"] = (ok, fr, det)
print(f"4. parallel         {'PASS' if ok else 'FAIL'} finish={fr} {det} ({w:.1f}s)")

# 5. Thinking + tool call
d, w = call([{"role": "user", "content": "Search the web for 'DGX Spark inference benchmarks 2026'."}], TOOLS, thinking=True)
ok, fr, det, c = check_tool_calls(d, 1)
results["thinking_tool"] = (ok, fr, det)
print(f"5. thinking+tool    {'PASS' if ok else 'FAIL'} finish={fr} {det} reas={(d['choices'][0]['message'].get('reasoning') or '')[:30]!r} ({w:.1f}s)")

# 6. ISSUE55: max_tokens truncation mid-tool-call — must NOT emit broken tool_calls
d, w = call([{"role": "user", "content": "Search the web for a very long detailed query about the history of distributed computing systems and their evolution over the past fifty years, including all major milestones and contributors, and then email the results."}], TOOLS, max_tokens=40)
ch = d["choices"][0]; m = ch["message"]
tcs = m.get("tool_calls") or []
fr = ch.get("finish_reason")
broken = False
for tc in tcs:
    args = (tc.get("function") or {}).get("arguments", "")
    try:
        json.loads(args)
    except Exception:
        broken = True
# issue55 fix: truncated calls report finish_reason=length and drop unparseable args
issue55_ok = (fr == "length") or (not broken)
if tcs and broken:
    issue55_ok = False
results["issue55_trunc"] = (issue55_ok, fr, [f"{len(tcs)} tool_calls, broken={broken}"])
print(f"6. issue55_trunc    {'PASS' if issue55_ok else 'FAIL'} finish={fr} n_tool_calls={len(tcs)} broken_json={broken} ({w:.1f}s)")

# 7. Tool choice forced
d, w = call([{"role": "user", "content": "Say hello, no tools needed."}], TOOLS, tool_choice={"type": "function", "function": {"name": "get_weather"}})
ok, fr, det, c = check_tool_calls(d, 1)
results["forced_choice"] = (ok, fr, det)
print(f"7. forced_choice    {'PASS' if ok else 'FAIL'} finish={fr} {det} ({w:.1f}s)")

print("\n=== SUMMARY ===")
for k, (ok, fr, det) in results.items():
    print(f"  {k:15s} {'PASS' if ok else 'FAIL'}  {det}")
npass = sum(1 for v in results.values() if v[0])
print(f"\n{ npass}/{len(results)} tool-calling checks PASSED")
sys.exit(0 if npass == len(results) else 1)
