#!/usr/bin/env python3
"""Generate docker-compose.lmcache.yml from docker-compose.dspark.yml.

Every LMCache-specific change is applied inside one entrypoint branch that is
taken only when DSPARK_ENABLE_LMCACHE is exactly "1" (repo convention, same
shape as DSPARK_ENABLE_ISSUE31_GPU_HOTFIX / DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX).
Inside that branch, immediately before `exec vllm serve`:
  - the LMCacheMPConnector --kv-transfer-config (hardcoded escaped JSON in the
    compose command: .env values are bash-sourced by the launcher, which
    strips quotes, so JSON cannot ride an env var in this stack)
  - export PYTHONHASHSEED=0 (chunk keys use Python's randomized hash();
    unpinned, every restart invalidates the whole cache)
  - unset PYTORCH_CUDA_ALLOC_CONF (vLLM rejects KV connectors alongside
    expandable_segments:True)

With the flag unset or 0 the branch is not taken: the stock
PYTORCH_CUDA_ALLOC_CONF service env entry is left untouched and no
PYTHONHASHSEED entry is added, so the engine keeps stock allocator/hash
behaviour and stock argv. The rendered config is NOT byte-identical to stock:
two inert deltas are always present — the DSPARK_ENABLE_LMCACHE pass-through
(default "0") in the service env, and an empty $${KVT_ARGS} expansion in the
serve argv (empty unquoted expansion adds no argument).

server-urls is strictly parsed (see MP_URL): the value is embedded in the
compose entrypoint inside a double-quoted shell string, so only
comma-separated tcp://host:port over an IPv4 literal or RFC1123 hostname is
accepted; anything else — shell punctuation ($ ` ; & | ( ) < > [ ] etc.),
quotes, spaces, control characters, bad schemes, leading-zero or
out-of-range ports — is refused at generation time. IPv6 literals are not
accepted: this recipe's fabric is IPv4-only (the server launcher's own
preflight binds IPv4 addresses).

Usage:
  patch-compose-lmcache.py <src> <dst> <server-urls>
  server-urls: comma-separated, e.g.
    tcp://192.168.104.10:6667,tcp://192.168.104.11:6667
"""
import re
import sys

if len(sys.argv) != 4:
    sys.exit(__doc__)
src, dst, urls = sys.argv[1], sys.argv[2], sys.argv[3]

# Strict grammar: tcp://<ipv4-or-hostname>:<1-65535>, comma-separated, nothing
# else. Whitelist-style — anything outside it (including every shell
# metacharacter and control character) fails fullmatch and is refused.
MP_URL = re.compile(
    r"tcp://"
    r"(?P<host>[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*)"
    r":(?P<port>[1-9][0-9]{0,4})"
)


def check_server_urls(raw):
    if not raw:
        sys.exit("error: server-urls must list at least one tcp://host:port")
    for part in raw.split(","):
        m = MP_URL.fullmatch(part)  # fullmatch: also rejects embedded newlines
        if not m:
            sys.exit(
                "error: server-urls entry " + repr(part) + " must match "
                "tcp://host:port — tcp scheme, IPv4 or hostname, numeric port "
                "1-65535; no spaces, quotes, or shell metacharacters"
            )
        if int(m.group("port")) > 65535:
            sys.exit("error: server-urls entry " + repr(part) + ": port must be <= 65535")
        host = m.group("host")
        if len(host) > 253:
            sys.exit("error: server-urls entry " + repr(part) + ": host too long")
        # A host made only of digits and dots is an intended IPv4 literal, so it
        # must be a valid dotted quad — the RFC1123 hostname branch would
        # otherwise swallow typos like 999.999.999.999 or 01.2.3.4 verbatim,
        # deferring the failure to connect time. (Mixed hosts like
        # "10.server.example" stay valid hostnames.)
        if "." in host and all(c.isdigit() or c == "." for c in host):
            octets = host.split(".")
            if len(octets) != 4 or any(
                not o or not o.isdigit() or len(o) > 3 or int(o) > 255
                or (len(o) > 1 and o[0] == "0")
                for o in octets
            ):
                sys.exit(
                    "error: server-urls entry " + repr(part) + ": host looks like "
                    "an IPv4 address but is not a valid dotted quad"
                )


check_server_urls(urls)
s = open(src).read()

a1 = 'if [ -n "$${DSPARK_REVISION:-}" ]; then REVISION_ARGS="--revision $${DSPARK_REVISION}"; fi;'
assert a1 in s, "anchor (REVISION_ARGS) not found — compose layout changed?"
kvt = (
    '{\\"kv_connector\\":\\"LMCacheMPConnector\\",\\"kv_role\\":\\"kv_both\\",'
    '\\"kv_connector_extra_config\\":{\\"lmcache.mp.server_urls\\":\\"' + urls + '\\"}}'
)
s = s.replace(
    a1,
    a1
    + '\n        KVT_ARGS="";'
    + '\n        if [ "$${DSPARK_ENABLE_LMCACHE:-0}" = "1" ]; then'
    + ' KVT_ARGS="--kv-transfer-config ' + kvt + '";'
    + ' export PYTHONHASHSEED=0;'
    + ' unset PYTORCH_CUDA_ALLOC_CONF; fi;',
    1,
)

a2 = "        $${VLLM_QUANTIZATION_ARGS}\n"
assert a2 in s, "anchor (serve args) not found"
s = s.replace(a2, a2 + "        $${KVT_ARGS}\n", 1)

a3 = 'PYTORCH_CUDA_ALLOC_CONF: "${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"'
assert a3 in s, "anchor (alloc conf) not found — compose layout changed?"

a4 = "      HF_HOME: /cache/huggingface\n"
assert a4 in s, "anchor (env block) not found"
s = s.replace(
    a4,
    a4 + '      DSPARK_ENABLE_LMCACHE: "${DSPARK_ENABLE_LMCACHE:-0}"\n',
    1,
)

# Opt-in boundary, enforced here so a future edit cannot quietly reintroduce an
# unconditional env change: with the flag off the engine must see stock env.
assert a3 in s, "generated compose must leave the stock PYTORCH_CUDA_ALLOC_CONF entry intact"
assert "\n      PYTHONHASHSEED:" not in s, "PYTHONHASHSEED must be exported inside the gate, not set in the service env"
assert s.count("DSPARK_ENABLE_LMCACHE") == 3, "expected exactly one gate plus one env pass-through"

open(dst, "w").write(s)
print("wrote", dst)
