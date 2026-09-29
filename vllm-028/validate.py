#!/usr/bin/env python3
"""Small API acceptance suite. Writes only synthetic prompts and results."""
import concurrent.futures
import json
import pathlib
import time
import urllib.request

BASE = "http://127.0.0.1:8888"
MODEL = "deepseek-v4-flash-0731"
results = []

def request(path, payload=None, timeout=240):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)

def chat(prompt, **overrides):
    payload = dict(model=MODEL, messages=[dict(role="user", content=prompt)],
                   max_tokens=128, temperature=0, chat_template_kwargs={"thinking": False})
    payload.update(overrides)
    started = time.monotonic()
    data = request("/v1/chat/completions", payload)
    return dict(seconds=round(time.monotonic()-started, 2),
                message=data["choices"][0]["message"],
                finish_reason=data["choices"][0]["finish_reason"], usage=data.get("usage"))

def record(name, result):
    results.append(dict(test=name, **result))
    print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    pathlib.Path("validation.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))

version=request("/version")["version"]
assert version == "0.28.0", version
models=request("/v1/models")
record("version_models", dict(version=version, models=models))

r=chat("What is 37 times 43? Reply only with the integer.")
assert "1591" in (r["message"].get("content") or ""), r
record("arithmetic", r)

r=chat("请用中文回答：法国的首都是哪里？只回答城市名。")
assert "巴黎" in (r["message"].get("content") or ""), r
record("chinese", r)

r=chat("Call get_weather for Seoul.", tools=[dict(type="function", function=dict(
    name="get_weather", description="Get weather in a city", parameters=dict(
        type="object", properties=dict(city=dict(type="string")), required=["city"])))], tool_choice="required")
calls=r["message"].get("tool_calls") or []
assert calls and calls[0]["function"]["name"] == "get_weather", r
assert json.loads(calls[0]["function"]["arguments"])["city"].lower() == "seoul", r
record("tool_call", r)

r=chat("Compute 17 plus 25. Give the result briefly.", max_tokens=512,
       chat_template_kwargs={"thinking":True,"reasoning_effort":"low"})
assert "42" in (r["message"].get("content") or ""), r
record("reasoning", r)

payload=dict(model=MODEL, messages=[dict(role="user",content="Reply with exactly STREAM_OK")],
             stream=True, max_tokens=32, temperature=0, chat_template_kwargs={"thinking":False})
req=urllib.request.Request(BASE+"/v1/chat/completions",data=json.dumps(payload).encode(),
                           headers={"Content-Type":"application/json"})
parts=[]
done=False
with urllib.request.urlopen(req,timeout=120) as response:
    for raw in response:
        line=raw.decode().strip()
        if line == "data: [DONE]":
            done=True
            break
        if line.startswith("data: "):
            event=json.loads(line[6:])
            for choice in event.get("choices",[]):
                parts.append(choice.get("delta",{}).get("content") or "")
assert done and "STREAM_OK" in "".join(parts), parts
record("streaming",dict(content="".join(parts),done=done))

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    values=list(pool.map(chat,["Reply with exactly ALPHA", "Reply with exactly BETA"]))
for target,r in zip(["ALPHA","BETA"],values):
    assert target in (r["message"].get("content") or ""), r
    record("concurrent_"+target, r)

for repeats in [4000, 18000]:
    prompt="The secret code is 73915.\n" + "This is ordinary background text.\n"*repeats + "\nWhat is the secret code? Reply only with its five digits."
    r=chat(prompt,max_tokens=32)
    assert "73915" in (r["message"].get("content") or ""), r
    record("long_context_"+str(repeats),r)
print("ALL_CHECKS_PASSED",flush=True)
