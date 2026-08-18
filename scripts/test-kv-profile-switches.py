#!/usr/bin/env python3
"""CPU gates for the optional KV / execution profile switches (PR #63).

Covers, with no GPU, no image, no container and no docker daemon:

* **default-off proof** — with `KV_CACHE_DTYPE`, `ENABLE_DSPARK_SPECULATIVE`
  and `ENFORCE_EAGER` all unset, the rendered `vllm serve` argv is word-for-word
  identical to `BASELINE_DEFAULT_ARGV`, captured from `docker-compose.dspark.yml`
  at main `8997d417` (the commit before the switches existed);
* **render matrix** — 3 dtypes x speculation {0,1} x eager {0,1}, including the
  decode-query-length sizing of `--max-cudagraph-capture-size`
  (`MAX_NUM_SEQS * (MTP_NUM_TOKENS + 1)` with speculation on,
  `MAX_NUM_SEQS * 1` with it off, flag absent under `--enforce-eager`);
* **rejection matrix** — `start-deepseek-v4-flash-dspark.sh` and
  `validate-dspark-config.sh` both `exit 2` on non-whitelisted values, before
  any docker or ssh call, and both accept the same set;
* **two-rank sync** — every compose invocation that renders the serve argv
  (head funnel plus both worker `remote_compose` sites) carries all three
  variables, so rank 0 and rank 1 can never render a different KV layout;
* **resolved profile** — the operator-facing summary is executed for real, so a
  reported "mtp speculative tokens: inactive" can never sit next to an
  MTP-derived capture size;
* **operator rationale** — `.env.dspark.example` states the measured
  584-byte-record reality and carries none of the withdrawn pool/precision
  trade-off claims, with all three switches still commented out (defaults).

Compose host-side interpolation is reproduced in-process. It is byte-exact:
`DockerComposeFidelityTest` re-verifies the rendered command and environment
against `docker compose config` whenever that binary exists. The container-side
command is then executed by `/bin/sh` with `exec /usr/local/bin/vllm` swapped
for an argv dumper, which is what makes the argv assertions real rather than
textual.

Run: `python3 scripts/test-kv-profile-switches.py -q`
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.dspark.yml"
START = ROOT / "start-deepseek-v4-flash-dspark.sh"
VALIDATE = ROOT / "validate-dspark-config.sh"
ENV_EXAMPLE = ROOT / ".env.dspark.example"

SERVICE = "vllm-dspark"
VLLM_BIN = "/usr/local/bin/vllm"

# Host-side environment the start script exports on the head rank. Addresses are
# from the RFC 5737 documentation range: this file never carries real fabric IPs.
HOST_ENV = {
    "NODE_RANK": "0",
    "HEADLESS": "",
    "MASTER_ADDR": "192.0.2.10",
    "MASTER_PORT": "25000",
    "NCCL_IB_HCA": "mlx5_0",
    "NCCL_SOCKET_IFNAME": "eth0",
    "VLLM_HOST_IP": "192.0.2.10",
}

# The serve argv main rendered at 8997d417 ("fix(eval): bulk-pad RULER-lite so
# 32k/262k cells are real"), i.e. before these switches existed, under HOST_ENV.
# The three switches are default-off, so the candidate must reproduce this list
# exactly. A deliberate future change to the serve argv must update this list in
# the same commit -- that is the point of freezing it.
BASELINE_DEFAULT_ARGV = (
    "serve",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
    "--served-model-name",
    "deepseek-v4-flash-dspark",
    "--host",
    "127.0.0.1",
    "--port",
    "8888",
    "--trust-remote-code",
    "--tensor-parallel-size",
    "2",
    "--pipeline-parallel-size",
    "1",
    "--kv-cache-dtype",
    "nvfp4_ds_mla",
    "--block-size",
    "256",
    "--max-model-len",
    "1048576",
    "--max-num-seqs",
    "6",
    "--max-num-batched-tokens",
    "8192",
    "--long-prefill-token-threshold",
    "1024",
    "--max-cudagraph-capture-size",
    "36",
    "--gpu-memory-utilization",
    "0.80",
    "--enable-prefix-caching",
    "--enable-prompt-tokens-details",
    "--async-scheduling",
    "--enable-chunked-prefill",
    "--speculative-config",
    '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}',
    "--tokenizer-mode",
    "deepseek_v4",
    "--distributed-executor-backend",
    "mp",
    "--moe-backend",
    "flashinfer_b12x",
    "--tool-call-parser",
    "deepseek_v4",
    "--enable-auto-tool-choice",
    "--reasoning-parser",
    "deepseek_v4",
    "--reasoning-config",
    '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>",'
    '"reasoning_end_str":"</think>"}',
    "--default-chat-template-kwargs",
    '{"thinking":true,"reasoning_effort":"low"}',
    "--generation-config",
    "vllm",
    "--enable-flashinfer-autotune",
    "--nnodes",
    "2",
    "--node-rank",
    "0",
    "--master-addr",
    "192.0.2.10",
    "--master-port",
    "25000",
)

SPEC_JSON = ('{"method":"dspark","num_speculative_tokens":%s,'
             '"draft_sample_method":"probabilistic"}')

DTYPES = ("nvfp4_ds_mla", "fp8_ds_mla", "fp8")

# ---------------------------------------------------------------------------
# compose readers (stdlib only; no PyYAML dependency in the CPU gate)
# ---------------------------------------------------------------------------


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").split("\n")


def read_command_scalar(path: Path = COMPOSE) -> str:
    """Return the folded (`>`) command scalar of the single compose service.

    Folding rules implemented: a break between two lines at the block
    indentation folds to one space; a break next to a more-indented or empty
    line stays literal; clip chomping adds one trailing newline. Verified
    byte-identical to `docker compose config` by DockerComposeFidelityTest.
    """
    lines = _lines(path)
    services = [i for i, ln in enumerate(lines) if re.fullmatch(r"  [A-Za-z0-9_.-]+:", ln)]
    assert len(services) == 1, f"expected exactly one compose service, got {len(services)}"
    folded = [i for i, ln in enumerate(lines) if ln.strip() == "- >"]
    assert len(folded) == 1, f"expected exactly one folded command scalar, got {len(folded)}"
    start = folded[0] + 1
    base = len(lines[start]) - len(lines[start].lstrip(" "))
    body: list[str] = []
    for ln in lines[start:]:
        if ln.strip() == "":
            body.append("")
            continue
        if len(ln) - len(ln.lstrip(" ")) < base:
            break
        body.append(ln[base:])
    while body and body[-1] == "":
        body.pop()
    out: list[str] = []
    for idx, ln in enumerate(body):
        if idx:
            prev = body[idx - 1]
            keep = ln == "" or prev == "" or ln.startswith(" ") or prev.startswith(" ")
            out.append("\n" if keep else " ")
        out.append(ln)
    return "".join(out) + "\n"


def read_environment(path: Path = COMPOSE) -> dict[str, str]:
    """Return the service `environment:` map, values un-interpolated."""
    lines = _lines(path)
    heads = [i for i, ln in enumerate(lines) if ln.rstrip() == "    environment:"]
    assert len(heads) == 1, f"expected exactly one environment block, got {len(heads)}"
    env: dict[str, str] = {}
    for ln in lines[heads[0] + 1:]:
        if ln.strip() == "" or ln.lstrip().startswith("#"):
            continue
        if len(ln) - len(ln.lstrip(" ")) <= 4:
            break
        key, _, value = ln.strip().partition(":")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            value = value[1:-1]
        env[key] = value
    return env


# ---------------------------------------------------------------------------
# compose host-side interpolation
# ---------------------------------------------------------------------------

_NAME = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)")


def _match_brace(text: str, start: int) -> int:
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unbalanced ${ } in compose file")


def interpolate(text: str, env: dict[str, str]) -> str:
    """Compose host-side interpolation: `$$` escapes, `${VAR}` and its
    `:-` / `-` / `:+` / `+` forms, nested defaults included."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "$" and i + 1 < n:
            if text[i + 1] == "$":
                out.append("$")
                i += 2
                continue
            if text[i + 1] == "{":
                end = _match_brace(text, i + 1)
                out.append(_expand(text[i + 2:end], env))
                i = end + 1
                continue
            m = _NAME.match(text, i + 1)
            if m:
                out.append(env.get(m.group(1), ""))
                i = m.end()
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _expand(inner: str, env: dict[str, str]) -> str:
    m = _NAME.match(inner)
    if not m:
        raise ValueError(f"unsupported interpolation: ${{{inner}}}")
    name, rest = m.group(1), inner[m.end():]
    value = env.get(name)
    if rest.startswith(":-"):
        return value if value else interpolate(rest[2:], env)
    if rest.startswith("-"):
        return value if value is not None else interpolate(rest[1:], env)
    if rest.startswith(":+"):
        return interpolate(rest[2:], env) if value else ""
    if rest.startswith("+"):
        return interpolate(rest[1:], env) if value is not None else ""
    if rest:
        raise ValueError(f"unsupported interpolation: ${{{inner}}}")
    return value or ""


# ---------------------------------------------------------------------------
# container-side evaluation -> argv
# ---------------------------------------------------------------------------

# Everything before this reset only patches the image; skipping it keeps the
# simulation hermetic (no /opt hotfixes, no HF cache probing) while preserving
# every statement that contributes to the serve argv.
ARGV_ANCHOR = 'VLLM_QUANTIZATION_ARGS=""'
_DUMPER = '__argv_dump() { for a in "$@"; do printf "%s\\0" "$a"; done; }\n'


def argv_from_script(script: str, container_env: dict[str, str]) -> list[str]:
    tail = script[script.rindex(ARGV_ANCHOR):]
    assert tail.count(f"exec {VLLM_BIN}") == 1, "expected exactly one exec of the vllm binary"
    tail = tail.replace(f"exec {VLLM_BIN}", "__argv_dump")
    run = subprocess.run(
        ["sh", "-c", _DUMPER + tail],
        capture_output=True,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), **container_env},
    )
    if run.returncode != 0:
        raise AssertionError(
            f"container-side render failed (rc={run.returncode}): "
            f"{run.stderr.decode(errors='replace')[-800:]}"
        )
    words = run.stdout.split(b"\0")
    assert words and words[-1] == b"", "argv dump not NUL-terminated"
    return [w.decode() for w in words[:-1]]


def render(**overrides: str) -> list[str]:
    """Render the serve argv for one host-side environment."""
    env = {**HOST_ENV, **overrides}
    script = interpolate(read_command_scalar(), env)
    container_env = {k: interpolate(v, env) for k, v in read_environment().items()}
    return argv_from_script(script, container_env)


def value_after(argv: list[str], flag: str) -> str | None:
    if flag not in argv:
        return None
    return argv[argv.index(flag) + 1]


def as_flags(argv: list[str]) -> dict[str, object]:
    """`['--a', '1', '--b']` -> `{'--a': '1', '--b': True}`, positionals under ''."""
    flags: dict[str, object] = {}
    i = 0
    while i < len(argv):
        word = argv[i]
        if word.startswith("--"):
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                flags[word] = argv[i + 1]
                i += 2
            else:
                flags[word] = True
                i += 1
        else:
            flags.setdefault("", []).append(word)  # type: ignore[union-attr]
            i += 1
    return flags


# ---------------------------------------------------------------------------
# start-script / validator execution
# ---------------------------------------------------------------------------


def run_gate(script: Path, env_file_body: str, **env: str) -> subprocess.CompletedProcess:
    """Run a config gate against a scratch env file. Both gates validate the
    three switches before their first docker/ssh call, so nothing is started."""
    with tempfile.TemporaryDirectory() as tmp:
        env_file = Path(tmp) / "env.dspark"
        env_file.write_text(env_file_body, encoding="utf-8")
        return subprocess.run(
            ["bash", str(script)],
            capture_output=True,
            text=True,
            cwd=tmp,
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": os.environ.get("HOME", tmp),
                "ENV_FILE": str(env_file),
                **env,
            },
        )


class DefaultOffArgvTest(unittest.TestCase):
    """The switches are inert: default argv equals pre-switch main."""

    def test_default_argv_matches_pre_switch_baseline(self):
        self.assertEqual(render(), list(BASELINE_DEFAULT_ARGV))

    def test_explicit_defaults_match_unset(self):
        self.assertEqual(
            render(KV_CACHE_DTYPE="nvfp4_ds_mla", ENABLE_DSPARK_SPECULATIVE="1",
                   ENFORCE_EAGER="0"),
            list(BASELINE_DEFAULT_ARGV),
        )

    def test_empty_is_treated_as_unset_by_both_layers(self):
        # bash `:-` in the start script and compose `:-` agree: empty = default.
        self.assertEqual(
            render(KV_CACHE_DTYPE="", ENABLE_DSPARK_SPECULATIVE="", ENFORCE_EAGER=""),
            list(BASELINE_DEFAULT_ARGV),
        )
        accepted = run_gate(START, "KV_CACHE_DTYPE=\nENABLE_DSPARK_SPECULATIVE=\nENFORCE_EAGER=\n")
        self.assertNotIn("KV_CACHE_DTYPE must be", accepted.stderr)
        self.assertNotEqual(accepted.returncode, 2)


class RenderMatrixTest(unittest.TestCase):
    """3 dtypes x speculation x eager, argv-level."""

    def test_matrix(self):
        for dtype in DTYPES:
            for spec in ("1", "0"):
                for eager in ("0", "1"):
                    with self.subTest(dtype=dtype, spec=spec, eager=eager):
                        argv = render(KV_CACHE_DTYPE=dtype,
                                      ENABLE_DSPARK_SPECULATIVE=spec,
                                      ENFORCE_EAGER=eager)
                        self.assertEqual(argv.count("--kv-cache-dtype"), 1)
                        self.assertEqual(value_after(argv, "--kv-cache-dtype"), dtype)

                        if spec == "1":
                            self.assertEqual(value_after(argv, "--speculative-config"),
                                             SPEC_JSON % 5)
                        else:
                            self.assertNotIn("--speculative-config", argv)

                        self.assertEqual("--enforce-eager" in argv, eager == "1")

                        cap = value_after(argv, "--max-cudagraph-capture-size")
                        if eager == "1":
                            # Nothing is captured under eager execution, so a
                            # capture size would be inert and misleading.
                            self.assertIsNone(cap)
                        else:
                            self.assertEqual(cap, "36" if spec == "1" else "6")

                        self.assertNotIn("", argv, "empty argv word rendered")
                        self.assertEqual(argv[:2],
                                         ["serve", "deepseek-ai/DeepSeek-V4-Flash-0731"])

    def test_capture_size_tracks_decode_query_length(self):
        on = render(MAX_NUM_SEQS="4", MTP_NUM_TOKENS="7")
        self.assertEqual(value_after(on, "--max-cudagraph-capture-size"), "32")  # 4*(7+1)
        self.assertEqual(value_after(on, "--speculative-config"), SPEC_JSON % 7)

        off = render(MAX_NUM_SEQS="4", MTP_NUM_TOKENS="7", ENABLE_DSPARK_SPECULATIVE="0")
        self.assertEqual(value_after(off, "--max-cudagraph-capture-size"), "4")  # 4*1

    def test_only_the_switched_flags_move(self):
        """Each arm changes exactly its own flags -- nothing else in the argv."""
        base = as_flags(render())
        for overrides, expected_delta in (
            ({"KV_CACHE_DTYPE": "fp8_ds_mla"},
             {"--kv-cache-dtype": "fp8_ds_mla"}),
            ({"ENABLE_DSPARK_SPECULATIVE": "0"},
             {"--speculative-config": None, "--max-cudagraph-capture-size": "6"}),
            ({"ENFORCE_EAGER": "1"},
             {"--max-cudagraph-capture-size": None, "--enforce-eager": True}),
        ):
            with self.subTest(**overrides):
                arm = as_flags(render(**overrides))
                delta = {flag: arm.get(flag)
                         for flag in set(base) | set(arm)
                         if base.get(flag) != arm.get(flag)}
                self.assertEqual(delta, expected_delta)

    def test_compose_has_no_ungated_capture_size(self):
        text = COMPOSE.read_text(encoding="utf-8")
        occurrences = [ln for ln in text.splitlines() if "--max-cudagraph-capture-size" in ln]
        self.assertEqual(len(occurrences), 1, occurrences)
        self.assertIn('CUDAGRAPH_ARGS="--max-cudagraph-capture-size', occurrences[0].strip())
        self.assertIn("$${CUDAGRAPH_ARGS}", text)
        # The eager gate must clear the capture args, not just add the flag.
        self.assertIn('if [ "${ENFORCE_EAGER:-0}" = "1" ]; then '
                      'EXECUTION_ARGS="--enforce-eager"; CUDAGRAPH_ARGS=""; fi;', text)
        # Exact-1 gates, fail-closed style used across this compose file.
        self.assertIn('if [ "${ENABLE_DSPARK_SPECULATIVE:-1}" = "1" ]; then', text)
        self.assertIn("--kv-cache-dtype ${KV_CACHE_DTYPE:-nvfp4_ds_mla}", text)


class ValidationRejectionTest(unittest.TestCase):
    """Fail-closed whitelist, mirrored by both config gates, pre-docker."""

    BAD_DTYPES = ("fp16", "fp8_e4m3", "nvfp4", "bfloat16", "auto", "FP8", '"fp8 "', "fp8_dsmla")
    BAD_FLAGS = ("2", "true", "yes", "on", "01", "-1")

    def _assert_rejected(self, script, body, needle):
        got = run_gate(script, body)
        self.assertEqual(got.returncode, 2, f"{body!r} -> rc={got.returncode}\n{got.stderr}")
        self.assertIn(needle, got.stderr)

    def test_bad_dtype_exits_2(self):
        for script in (START, VALIDATE):
            for value in self.BAD_DTYPES:
                with self.subTest(script=script.name, value=value):
                    self._assert_rejected(script, f"KV_CACHE_DTYPE={value}\n",
                                          "KV_CACHE_DTYPE must be")

    def test_bad_boolean_exits_2(self):
        for script in (START, VALIDATE):
            for value in self.BAD_FLAGS:
                with self.subTest(script=script.name, value=value):
                    self._assert_rejected(script, f"ENABLE_DSPARK_SPECULATIVE={value}\n",
                                          "must be 0 or 1")
                    self._assert_rejected(script, f"ENFORCE_EAGER={value}\n",
                                          "must be 0 or 1")

    def test_whitelisted_values_pass_validation(self):
        # Accepted values fall through to the first hard requirement
        # (WORKER_HOST), which proves validation did not fire.
        for dtype in DTYPES:
            for spec in ("0", "1"):
                for eager in ("0", "1"):
                    with self.subTest(dtype=dtype, spec=spec, eager=eager):
                        got = run_gate(
                            START,
                            f"KV_CACHE_DTYPE={dtype}\n"
                            f"ENABLE_DSPARK_SPECULATIVE={spec}\n"
                            f"ENFORCE_EAGER={eager}\n",
                        )
                        self.assertNotEqual(got.returncode, 2, got.stderr)
                        self.assertIn("WORKER_HOST", got.stderr)
                        self.assertNotIn("must be", got.stderr.split("WORKER_HOST")[0])

    def test_validation_precedes_any_docker_or_ssh_call(self):
        text = START.read_text(encoding="utf-8")
        whitelist = text.index("KV_CACHE_DTYPE must be")
        for command in ("need_cmd docker", "docker compose", "ssh "):
            self.assertLess(whitelist, text.index(command),
                            f"{command} appears before the whitelist")


class TwoRankSyncTest(unittest.TestCase):
    """Both ranks must render one identical KV layout / execution profile."""

    SWITCHES = ("KV_CACHE_DTYPE", "ENABLE_DSPARK_SPECULATIVE", "ENFORCE_EAGER")

    def setUp(self):
        self.start = START.read_text(encoding="utf-8")

    def test_head_funnel_exports_all_switches(self):
        body = self.start.split("compose_base() {", 1)[1].split("\n}", 1)[0]
        for name in self.SWITCHES:
            self.assertIn(f'{name}="${name}" \\', body)

    def test_every_worker_render_site_carries_all_switches(self):
        sites = [ln for ln in self.start.splitlines()
                 if "remote_compose " in ln and COMPOSE.name in ln
                 and ("up -d" in ln or "config --quiet" in ln)]
        self.assertEqual(len(sites), 2, sites)  # worker validate + worker start
        for site in sites:
            for name in self.SWITCHES:
                self.assertIn(f"{name}='${name}'", site)

    def test_vl_sidecar_is_a_separate_namespace(self):
        # The sidecar renders its own compose file and its own
        # VL_SIDECAR_KV_CACHE_DTYPE; it must not read these switches.
        sidecar = [ln for ln in self.start.splitlines() if "vl-sidecar.yml" in ln]
        self.assertTrue(sidecar)
        for line in sidecar:
            self.assertNotIn(COMPOSE.name, line)
            for name in self.SWITCHES:
                self.assertNotIn(name, line)

    def test_switches_are_exported_before_use(self):
        text = self.start
        export = text.index("export ENABLE_DSPARK_SPECULATIVE ENFORCE_EAGER")
        self.assertLess(text.index("KV_CACHE_DTYPE"), export)
        self.assertLess(export, text.index("compose_base() {"))


class ResolvedProfileTest(unittest.TestCase):
    """The printed profile must never contradict the rendered argv."""

    @staticmethod
    def _profile(**env: str) -> str:
        text = START.read_text(encoding="utf-8")
        start = text.index("print_resolved_profile() {")
        end = text.index("\n}\n", start) + 3
        with tempfile.TemporaryDirectory() as tmp:
            fn = Path(tmp) / "profile.sh"
            fn.write_text(text[start:end], encoding="utf-8")
            run = subprocess.run(
                ["bash", "-c", f'source "{fn}"; print_resolved_profile'],
                capture_output=True, text=True,
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                     "SCRIPT_DIR": str(ROOT), "KV_CACHE_DTYPE": "nvfp4_ds_mla",
                     "ENABLE_DSPARK_SPECULATIVE": "1", "ENFORCE_EAGER": "0", **env},
            )
        assert run.returncode == 0, run.stderr
        return run.stdout

    def test_defaults_report_mtp_derived_capture(self):
        out = self._profile()
        self.assertIn("mtp speculative tokens: 5", out)
        self.assertIn("cudagraph capture size: 36", out)

    def test_speculation_off_reports_query_length_one(self):
        out = self._profile(ENABLE_DSPARK_SPECULATIVE="0")
        self.assertIn("mtp speculative tokens: inactive", out)
        self.assertIn("cudagraph capture size: 6", out)
        self.assertNotIn("capture size: 36", out)

    def test_eager_reports_no_capture(self):
        for spec in ("0", "1"):
            with self.subTest(spec=spec):
                out = self._profile(ENFORCE_EAGER="1", ENABLE_DSPARK_SPECULATIVE=spec)
                self.assertIn("cudagraph capture size: none", out)
                self.assertNotIn("capture size: 36", out)

    def test_printed_capture_matches_rendered_argv(self):
        for spec in ("0", "1"):
            for eager in ("0", "1"):
                with self.subTest(spec=spec, eager=eager):
                    out = self._profile(ENABLE_DSPARK_SPECULATIVE=spec, ENFORCE_EAGER=eager)
                    printed = re.search(r"cudagraph capture size: (\S+)", out).group(1)
                    rendered = value_after(
                        render(ENABLE_DSPARK_SPECULATIVE=spec, ENFORCE_EAGER=eager),
                        "--max-cudagraph-capture-size")
                    self.assertEqual(printed, rendered if rendered else "none")

    def test_dtype_line_states_pool_neutrality(self):
        out = self._profile(KV_CACHE_DTYPE="fp8_ds_mla")
        self.assertIn("KV cache dtype: fp8_ds_mla", out)
        self.assertIn("584B/token", out)


class OperatorRationaleTest(unittest.TestCase):
    """`.env.dspark.example` ships the measured reality, defaults untouched."""

    WITHDRAWN = (
        "~2x the latent bytes",
        "largest KV token pool",
        "smaller token pool",
        "higher precision",
        "better long-context recall",
    )

    def setUp(self):
        self.example = ENV_EXAMPLE.read_text(encoding="utf-8")

    def test_no_withdrawn_memory_or_precision_claim(self):
        for claim in self.WITHDRAWN:
            for path, text in ((ENV_EXAMPLE, self.example),
                               (START, START.read_text(encoding="utf-8")),
                               (COMPOSE, COMPOSE.read_text(encoding="utf-8"))):
                self.assertNotIn(claim, text, f"{path.name} still claims {claim!r}")

    def test_states_the_584_byte_record(self):
        self.assertIn("584-byte-per-token", self.example)
        self.assertIn("448 B fp8 NoPE + 128 B bf16 RoPE + 8 B ue8m0 scales", self.example)
        self.assertIn("the KV token pool is unchanged", self.example)

    def test_all_three_dtypes_documented_and_commented_out(self):
        for dtype in DTYPES:
            self.assertIn(f"# KV_CACHE_DTYPE={dtype}\n", self.example)
        for name in ("KV_CACHE_DTYPE", "ENABLE_DSPARK_SPECULATIVE", "ENFORCE_EAGER"):
            live = [ln for ln in self.example.splitlines()
                    if ln.startswith(f"{name}=")]
            self.assertEqual(live, [], f"{name} must ship commented out: {live}")

    def test_documents_capture_size_consequences(self):
        self.assertIn("MAX_NUM_SEQS * 1 (decode query length 1)", self.example)
        self.assertIn("--max-cudagraph-capture-size argv is then dropped", self.example)


@unittest.skipUnless(shutil.which("docker"), "docker binary not present")
class DockerComposeFidelityTest(unittest.TestCase):
    """Pin the in-process interpolation against the real renderer.

    `docker compose config` needs no daemon and starts nothing.
    """

    @classmethod
    def setUpClass(cls):
        probe = subprocess.run(["docker", "compose", "version"], capture_output=True)
        if probe.returncode != 0:
            raise unittest.SkipTest("docker compose plugin unavailable")

    def _docker(self, overrides):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / "scratch.env"
            env_file.write_text("", encoding="utf-8")
            run = subprocess.run(
                ["docker", "compose", "--env-file", str(env_file), "-f", str(COMPOSE),
                 "config", "--format", "json"],
                capture_output=True, cwd=str(ROOT),
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                     "HOME": os.environ.get("HOME", tmp),
                     "COMPOSE_DISABLE_ENV_FILE": "1", **HOST_ENV, **overrides},
            )
        self.assertEqual(run.returncode, 0, run.stderr.decode(errors="replace")[-1500:])
        service = json.loads(run.stdout)["services"][SERVICE]
        # compose re-escapes `$` as `$$` in `config` output; the container gets `$`.
        script = service["command"][2].replace("$$", "$")
        env = {k: ("" if v is None else str(v))
               for k, v in (service.get("environment") or {}).items()}
        return script, env

    def test_interpolation_and_argv_match_docker(self):
        for overrides in ({}, {"KV_CACHE_DTYPE": "fp8_ds_mla"},
                          {"ENABLE_DSPARK_SPECULATIVE": "0"}, {"ENFORCE_EAGER": "1"}):
            with self.subTest(**overrides):
                script, env = self._docker(overrides)
                host = {**HOST_ENV, **overrides}
                self.assertEqual(interpolate(read_command_scalar(), host), script)
                self.assertEqual(
                    {k: interpolate(v, host) for k, v in read_environment().items()}, env)
                self.assertEqual(argv_from_script(script, env), render(**overrides))


if __name__ == "__main__":
    unittest.main()
