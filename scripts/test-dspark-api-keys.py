#!/usr/bin/env python3
"""CPU gates for DSPARK_API_KEYS multi-key auth (behavioral, stdlib only).

These tests execute the REAL auth code — extracted from docker-compose.dspark.yml
(the single-line entrypoint block) and from the three probe scripts (the
`# DSPARK_API_KEYS auth ...` marker blocks) — through a shell, and assert on
observable behavior rather than source text:

- unset / empty / whitespace-only DSPARK_API_KEYS adds no auth anywhere;
- a parsed value becomes EXACTLY ONE `--api-key` flag carrying every key
  (order preserved, separators collapsed, duplicates allowed). Repeating the
  flag overwrites instead of appends in vLLM (nargs with last-wins), so a
  per-key loop would silently leave only the last key valid;
- literal glob characters survive as literal argv/header tokens (no pathname
  expansion, no word-splitting inside an element);
- CR/LF/VT/FF and backslashes are rejected before empty/conflict
  classification in all four contexts; a token starting with `-` is rejected
  with exit 2 and a fixed diagnostic that never echoes the token bytes;
- controls placed before a dash still report the single-line failure, proving
  validation precedence is identical in the entrypoint and all three probes;
- VLLM_API_KEY and DSPARK_API_KEYS both meaningful => exit 2 naming BOTH
  variables, in the compose entrypoint and in all three probe scripts. The
  server must never silently choose one (vLLM's CLI would win) and the probes
  must never guess which variable the server honoured;
- VLLM_API_KEY alone keeps the legacy single-key path unchanged;
- the three probe blocks are byte-identical (consistent parsing);
- Compose interpolates the key variables container-side (`$$`), so key text is
  never host-interpolated into shell source by Compose.

A ComposedHandoff layer additionally drives the REAL `docker compose
--env-file <stub> config --format json` render of docker-compose.dspark.yml
when Docker Compose is available, extracts the auth block, fail-closed
redaction gate, and exec tail from rendered `command[2]`, runs them under the
rendered `environment`, and requires exactly one `--api-key` flag. It includes
two negative controls — the compose DSPARK_API_KEYS passthrough line removed,
and `$$` regressed to `$` in the entrypoint block — that must FAIL the chain.
Hosts without Docker Compose skip the layer; once the CLI is available,
render and JSON failures are loud test failures.

run_bash is hermetic: the base env excludes VLLM_API_KEY and DSPARK_API_KEYS
(and any variable whose name contains "API_KEY"), so a hostile parent carrying
either/both variables cannot change any outcome; a case opts a variable in
first. The docker invocation is scrubbed the same way, and the handoff env is
the rendered compose `environment` alone.

The committed-key scan covers every PR-changed text artifact: compose, env
example, probes/launcher, docs, changelog, CI wiring, redaction patch, both
auth test modules, and the env-normalization harness.

No GPU, no serve, stdlib only.
"""

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
ENV_EXAMPLE = ROOT / ".env.dspark.example"
ENVS_DOC = ROOT / "docs" / "ENVS.md"
CHANGELOG = ROOT / "CHANGELOG.md"
SELF = Path(__file__)
REDACTION_PATCH = ROOT / "patches" / "hotfix-vllm-redact-api-key-log.sh"
CI_VALIDATE = ROOT / "scripts" / "ci-validate.sh"
REDACTION_TEST = ROOT / "scripts" / "test-redact-api-key-log.py"
ENV_NORMALISATION_TEST = ROOT / "scripts" / "test-env-normalisation.py"
SOURCE = START.read_text(encoding="utf-8")
PROBES = (
    ROOT / "start-deepseek-v4-flash-dspark.sh",
    ROOT / "smoke-deepseek-v4-flash-dspark.sh",
    ROOT / "status-deepseek-v4-flash-dspark.sh",
)

PROBE_BEGIN = "# DSPARK_API_KEYS auth (begin)"
PROBE_END = "# DSPARK_API_KEYS auth (end)"

# Exact contract messages (identical in all four contexts, exit code 2).
BOTH_SET_MSG = (
    "error: VLLM_API_KEY and DSPARK_API_KEYS are both set; "
    "set exactly one of them"
)
SINGLE_LINE_MSG = "error: DSPARK_API_KEYS must be a single-line space-separated list"
BACKSLASH_MSG = "error: DSPARK_API_KEYS must not contain backslashes"
DASH_REJECT_MSG = "error: DSPARK_API_KEYS contains a token beginning with '-'"
AMBIENT_GUARD_MSG = (
    "error: DSPARK_API_KEYS is set in the environment but does not match "
    ".env.dspark; set it only in .env.dspark"
)
PREFLIGHT_MSG = (
    "error: API keys are configured but "
    "patches/hotfix-vllm-redact-api-key-log.sh is missing; keyed starts "
    "require the startup-log redaction hotfix"
)

# Compose `$$` escapes stay `$$` in the rendered `command[2]` text; the regexes
# below tolerate both `$` and `$$` so one extractor serves source and render.
_AUTH_START = re.compile(
    r'API_KEY_ARGS=\(\);[ \t]*case "\$+\{DSPARK_API_KEYS:-\}" in'
)
_AUTH_END = re.compile(r'API_KEY_ARGS=\(--api-key "\$+\{_dspark_keys\[@\]\}"\); fi;')
_REDACTION_GATE = re.compile(
    r'if \[ "\$+\{_dspark_keys_set\}" = "1" \] \|\| '
    r'\[ -n "\$+\{VLLM_API_KEY:-\}" \]; then '
    r'bash /opt/dspark-patches/hotfix-vllm-redact-api-key-log\.sh \|\| exit 1; '
    r'bash /opt/dspark-patches/hotfix-vllm-redact-api-key-log\.sh --status \|\| exit 1; fi;'
)

DOCKER = shutil.which("docker")


# --------------------------------------------------------------------------
# Extraction: get the real code out of the real files.
# --------------------------------------------------------------------------


def find_auth_block(text: str) -> str:
    """The entrypoint auth block inside `text` (compose source or rendered).

    Requires the block to be ONE physical line (ask #3: a folded-scalar-safe
    single compose line), and returns it verbatim (`$$` form in the source,
    `$$` form in the docker render — both decode to container-side `$`).
    """
    start = _AUTH_START.search(text)
    if not start:
        raise AssertionError("entrypoint auth block not found")
    end = _AUTH_END.search(text, start.start())
    if not end:
        raise AssertionError("entrypoint auth block end not found")
    block = text[start.start():end.end()]
    # The whole block must survive folding: no real newline inside.
    if "\n" in block.strip("\n"):
        raise AssertionError("entrypoint auth block must stay ONE physical line")
    return block


def worker_sync_loop() -> str:
    """The start script's `_hf_sync` worker patch sync loop, extracted verbatim."""
    text = START.read_text(encoding="utf-8")
    m = re.search(r'(for _hf_sync in [\s\S]*?^\s*done$)', text, re.MULTILINE)
    if not m:
        raise AssertionError("_hf_sync loop not found in start script")
    return m.group(1)


def entrypoint_auth_block() -> str:
    """The compose entrypoint's auth line, still carrying Compose `$$` escapes."""
    return find_auth_block(COMPOSE.read_text(encoding="utf-8"))


def entrypoint_exec_tail() -> str:
    """The `exec vllm serve ...` tail as the container sees it ($$ already `$`)."""
    text = COMPOSE.read_text(encoding="utf-8")
    start = text.index("exec /usr/local/bin/vllm serve")
    line_start = text.rindex("\n", 0, start) + 1
    tail_lines = []
    for line in text[line_start:].splitlines():
        if not line.strip():
            break
        if len(line) - len(line.lstrip(" ")) < 8:
            break
        tail_lines.append(line.strip())
    if not tail_lines:
        raise AssertionError("exec vllm serve tail not found in compose")
    return " ".join(tail_lines).replace("$$", "$")


def tail_from_text(text: str) -> str:
    """The `exec vllm serve ...` tail from a one-line rendered `command[2]`."""
    start = text.index("exec /usr/local/bin/vllm serve")
    return text[start:]


def probe_auth_block(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    begin = text.index(PROBE_BEGIN) + len(PROBE_BEGIN)
    end = text.index(PROBE_END, begin)
    return text[begin:end]


def launcher_slice(start: str, end: str) -> str:
    """Return an exact launcher chunk bounded by stable marker text."""
    begin = SOURCE.index(start)
    return SOURCE[begin:SOURCE.index(end, begin)]


LAUNCHER_ENV_BLOCK = launcher_slice("_dspark_env_clean=", "# GPU util comes from GPU_MEMORY_UTILIZATION_TEXT")
PREFLIGHT_BLOCK = launcher_slice(
    "# DSPARK redaction pre-flight (begin)",
    "# DSPARK redaction pre-flight (end)",
)


def entrypoint_redaction_gate(text: str = None) -> str:
    """Extract the exact fail-closed redaction gate from source or render."""
    if text is None:
        text = COMPOSE.read_text(encoding="utf-8")
    match = _REDACTION_GATE.search(text)
    if not match:
        raise AssertionError("entrypoint redaction gate not found")
    return match.group(0)


def middle_segment(text: str) -> str:
    """Entrypoint text skipped by the auth-block + exec-tail harness."""
    block = find_auth_block(text)
    start = text.index(block) + len(block)
    end = text.index("exec /usr/local/bin/vllm serve", start)
    return text[start:end]


# --------------------------------------------------------------------------
# Shell helpers.
# --------------------------------------------------------------------------


def base_env(ambient=None):
    """Hermetic bash env: parent env minus ALL key material.

    VLLM_API_KEY and DSPARK_API_KEYS never reach a case through the parent
    (or through an injected hostile ambient), so a case must opt them in
    explicitly via env_kwargs. Anything whose name contains "API_KEY" is
    scrubbed (covers both variables plus any future variant).
    """
    env = dict(os.environ)
    if ambient:
        env.update(ambient)
    for name in [k for k in env if "API_KEY" in k]:
        del env[name]
    env.setdefault("PATH", "/usr/bin:/bin")
    env.setdefault("HOME", "/")
    return env


def compose_cli_available(docker_path, ambient=None):
    """True only when `docker compose version` succeeds under a scrubbed env."""
    if not docker_path:
        return False
    proc = subprocess.run(
        [docker_path, "compose", "version"],
        capture_output=True,
        text=True,
        env=base_env(ambient),
    )
    return proc.returncode == 0


COMPOSE_CLI_OK = compose_cli_available(DOCKER)


def run_bash(body: str, env_kwargs, cwd=None, ambient=None):
    """Run `body` in bash under a hermetic env.

    env_kwargs is the ONLY way key variables enter: a value sets it, None
    removes it (explicit absence even under a hostile parent).
    """
    env = base_env(ambient)
    for key, value in env_kwargs.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return subprocess.run(
        ["bash", "-c", body], capture_output=True, text=True, env=env, cwd=cwd
    )


def run_entrypoint(env_kwargs, cwd=None, ambient=None, block=None):
    """Run the real entrypoint auth block and print the resulting API_KEY_ARGS."""
    if block is None:
        block = entrypoint_auth_block()
    body = (
        block.replace("$$", "$") + "\n"
        + "printf 'COUNT=%s\\n' \"${#API_KEY_ARGS[@]}\";\n"
        + 'printf "<%s>\\n" "${API_KEY_ARGS[@]}"\n'
    )
    return run_bash(body, env_kwargs, cwd, ambient)


def run_probe(path: Path, env_kwargs, ambient=None):
    """Run a probe's real auth block and print the resulting AUTH_HEADER_ARGS."""
    body = (
        probe_auth_block(path) + "\n"
        + "printf 'COUNT=%s\\n' \"${#AUTH_HEADER_ARGS[@]}\";\n"
        + 'printf "<%s>\\n" "${AUTH_HEADER_ARGS[@]}"\n'
    )
    return run_bash(body, env_kwargs, ambient=ambient)


_VLLM_DUMPER = "python3 -c 'import sys; [print(\"A[\"+a+\"]\") for a in sys.argv[1:]]' "


def _decode_block_tail(block: str, tail: str):
    block = block.replace("$$", "$")
    tail = tail.replace("$$", "$").replace(
        "exec /usr/local/bin/vllm serve", _VLLM_DUMPER
    )
    return block, tail


def run_launcher_env(env_text: str, ambient_marker=None):
    """Execute the launcher's real normalized-snapshot block."""
    with tempfile.TemporaryDirectory() as td:
        env_file = Path(td, ".env.dspark")
        env_file.write_text(env_text, encoding="utf-8")
        env = base_env()
        env["ENV_FILE"] = str(env_file)
        env["TMPDIR"] = td
        if ambient_marker is not None:
            env["DSPARK_API_KEYS"] = ambient_marker
        body = (
            "set -euo pipefail\n" + LAUNCHER_ENV_BLOCK
            + "printf 'EFFECTIVE=<%s>\n' \"${DSPARK_API_KEYS:-}\"\n"
        )
        return subprocess.run(
            ["bash", "-c", body], capture_output=True, text=True, env=env
        )


def run_preflight(keys_set: str, vllm_key=None, patch_present=False):
    """Execute the launcher-only redaction preflight marker block."""
    with tempfile.TemporaryDirectory() as td:
        script_dir = Path(td)
        patch_dir = script_dir / "patches"
        patch_dir.mkdir()
        if patch_present:
            (patch_dir / "hotfix-vllm-redact-api-key-log.sh").write_text("")
        env = {
            "TEST_SCRIPT_DIR": str(script_dir),
            "TEST_KEYS_SET": keys_set,
            "VLLM_API_KEY": vllm_key,
        }
        body = (
            "set -euo pipefail\n"
            "SCRIPT_DIR=\"$TEST_SCRIPT_DIR\"\n"
            "_dspark_keys_set=\"$TEST_KEYS_SET\"\n"
            + PREFLIGHT_BLOCK
            + "printf 'FELL-THROUGH\n'\n"
        )
        return run_bash(body, env)


def run_redaction_gate(env_kwargs, *, patch_present=False,
                       apply_rc=0, status_rc=0):
    """Run the real auth block + redaction gate + vLLM argv dumper."""
    with tempfile.TemporaryDirectory() as td:
        patch_dir = Path(td, "patches")
        patch_dir.mkdir()
        log = Path(td, "calls.log")
        if patch_present:
            stub = patch_dir / "hotfix-vllm-redact-api-key-log.sh"
            stub.write_text(
                "#!/usr/bin/env bash\n"
                "if [ \"${1:-}\" = --status ]; then\n"
                "  printf '%s\\n' --status >>\"$PATCH_LOG\"\n"
                "  exit \"${STATUS_RC:-0}\"\n"
                "fi\n"
                "printf '%s\\n' apply >>\"$PATCH_LOG\"\n"
                "exit \"${APPLY_RC:-0}\"\n",
                encoding="utf-8",
            )
        block, tail = _decode_block_tail(
            entrypoint_auth_block(), entrypoint_exec_tail()
        )
        gate = entrypoint_redaction_gate().replace(
            "/opt/dspark-patches", str(patch_dir)
        ).replace("$$", "$")
        env = dict(env_kwargs)
        env.update({
            "PATCH_LOG": str(log),
            "APPLY_RC": str(apply_rc),
            "STATUS_RC": str(status_rc),
        })
        result = run_bash(block + "\n" + gate + "\n" + tail, env)
        calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
        return result, calls


def argv_of(env_kwargs, cwd=None, ambient=None, block=None, tail=None):
    """Run the real block + exec tail and return the argv handed to vllm serve."""
    if block is None:
        block = entrypoint_auth_block()
    if tail is None:
        tail = entrypoint_exec_tail()
    block, tail = _decode_block_tail(block, tail)
    r = run_bash(block + "\n" + tail, env_kwargs, cwd, ambient)
    if r.returncode != 0:
        raise AssertionError(f"argv dump failed:\n{r.stderr}")
    return [ln[2:-1] for ln in r.stdout.splitlines()
            if ln.startswith("A[") and ln.endswith("]")]


_SINGLE_DOLLAR = re.compile(
    r'(?<!\$)\$\{(DSPARK_API_KEYS|VLLM_API_KEY|_dspark_keys_set'
    r'|_dspark_keys\[@\]|_dspark_keys|_dspark_key)(:-[^}]*)?\}'
)


def single_dollar_refs(text: str):
    """Compose host-side interpolation refs (a `$` that is not `$$`)."""
    return _SINGLE_DOLLAR.findall(text)


# --------------------------------------------------------------------------
# Compose entrypoint: argv construction.
# --------------------------------------------------------------------------


class EntrypointArgv(unittest.TestCase):
    def test_unset_empty_whitespace_only_add_no_args(self):
        for env in ({}, {"DSPARK_API_KEYS": ""}, {"DSPARK_API_KEYS": "   \t "}):
            r = run_entrypoint(env)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(r.stdout.startswith("COUNT=0"), (env, r.stdout))

    def test_single_key_is_one_flag(self):
        r = run_entrypoint({"DSPARK_API_KEYS": "k1"})
        self.assertEqual(r.stdout, "COUNT=2\n<--api-key>\n<k1>\n", r.stderr)

    def test_many_keys_one_flag_exactly_order_kept(self):
        r = run_entrypoint({"DSPARK_API_KEYS": "k1 k2 k3"})
        elems = r.stdout.splitlines()
        self.assertEqual(elems[0], "COUNT=4")
        self.assertEqual(elems[1:], ["<--api-key>", "<k1>", "<k2>", "<k3>"])

    def test_separators_collapsed_order_kept(self):
        for value in ("k1  k2   k3", "  k1 k2\tk3  ", "k1   \t  k2 k3"):
            r = run_entrypoint({"DSPARK_API_KEYS": value})
            elems = r.stdout.splitlines()
            self.assertEqual(elems[1:], ["<--api-key>", "<k1>", "<k2>", "<k3>"], value)

    def test_duplicates_allowed(self):
        r = run_entrypoint({"DSPARK_API_KEYS": "k1 k1 k1"})
        elems = r.stdout.splitlines()
        self.assertEqual(elems[1:], ["<--api-key>", "<k1>", "<k1>", "<k1>"])

    def test_literal_glob_chars_not_expanded(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td, "k2tail").write_text("x")
            r = run_entrypoint({"DSPARK_API_KEYS": "k1 k2*"}, cwd=td)
        elems = r.stdout.splitlines()
        self.assertEqual(elems[1:], ["<--api-key>", "<k1>", "<k2*>"])

    def test_dash_leading_token_rejected_without_echoing_token(self):
        for value in ("-bad", "k1 -bad", "k1  -bad  k3"):
            r = run_entrypoint({"DSPARK_API_KEYS": value})
            self.assertEqual(r.returncode, 2, value)
            self.assertIn(DASH_REJECT_MSG, r.stderr, value)
            self.assertNotIn("-bad", r.stderr + r.stdout, value)

    def test_newline_cr_dash_after_newline_exit2_single_line_msg(self):
        # ask #4: newline/CR anywhere in the value, and a dash placed after one,
        # must exit 2 with the single-line message (guard runs before read).
        for value in ("k1\nk2", "k1\rk2", "k1\n-bad", "k1\r-bad"):
            r = run_entrypoint({"DSPARK_API_KEYS": value})
            self.assertEqual(r.returncode, 2, (value, r.stderr))
            self.assertIn(SINGLE_LINE_MSG, r.stderr, (value, r.stderr))
            if value.endswith("-bad"):
                self.assertNotIn(DASH_REJECT_MSG, r.stderr, (value, r.stderr))

    def test_both_vars_named_and_exit_2(self):
        r = run_entrypoint({"VLLM_API_KEY": "vk", "DSPARK_API_KEYS": "k1 k2"})
        self.assertEqual(r.returncode, 2)
        self.assertIn(BOTH_SET_MSG, r.stderr)

    def test_vllm_only_adds_no_flag(self):
        # VLLM_API_KEY is vLLM's own env var (served natively); the entrypoint
        # must not duplicate it into --api-key, where the CLI would silently
        # override the env value.
        r = run_entrypoint({"VLLM_API_KEY": "vk"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(r.stdout.startswith("COUNT=0"), r.stdout)

    def test_full_argv_exactly_one_api_key_flag(self):
        argv = argv_of({"DSPARK_API_KEYS": "sk-a sk-b sk-c"})
        self.assertEqual(argv.count("--api-key"), 1, argv)
        i = argv.index("--api-key")
        self.assertEqual(argv[i + 1:i + 4], ["sk-a", "sk-b", "sk-c"])

    def test_full_argv_no_api_key_when_unset(self):
        argv = argv_of({})
        self.assertNotIn("--api-key", argv)


class ControlCharacters(unittest.TestCase):
    """The entrypoint and all probes retain one fail-closed value grammar."""

    @staticmethod
    def outcomes(env):
        yield "entrypoint", run_entrypoint(env)
        for probe in PROBES:
            yield probe.name, run_probe(probe, env)

    def test_controls_rejected_before_classification_in_all_contexts(self):
        values = (
            "\n", "\r", "\r\n", " \r ", "\t\n\t",
            "\v", "\f", "k1\vk2", "k1\nk2", "k1\n-bad",
        )
        for value in values:
            for context, result in self.outcomes({"DSPARK_API_KEYS": value}):
                with self.subTest(context=context, value=repr(value)):
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn(SINGLE_LINE_MSG, result.stderr)
                    self.assertNotIn(DASH_REJECT_MSG, result.stderr)

    def test_control_rejection_precedes_both_set_check(self):
        env = {"VLLM_API_KEY": "vk", "DSPARK_API_KEYS": "\n"}
        for context, result in self.outcomes(env):
            with self.subTest(context=context):
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(SINGLE_LINE_MSG, result.stderr)
                self.assertNotIn(BOTH_SET_MSG, result.stderr)

    def test_backslashes_rejected_in_all_contexts(self):
        for value in ("a\\tb", "k1\\\\k2"):
            for context, result in self.outcomes({"DSPARK_API_KEYS": value}):
                with self.subTest(context=context, value=value):
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn(BACKSLASH_MSG, result.stderr)

    def test_unset_empty_and_space_tab_only_remain_keyless(self):
        for env in ({}, {"DSPARK_API_KEYS": ""},
                    {"DSPARK_API_KEYS": "   \t "}):
            for context, result in self.outcomes(env):
                with self.subTest(context=context, env=env):
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertTrue(result.stdout.startswith("COUNT=0"), result.stdout)

    def test_non_ascii_whitespace_is_not_treated_as_empty(self):
        value = "\u2003"
        for context, result in self.outcomes({"DSPARK_API_KEYS": value}):
            with self.subTest(context=context):
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(result.stdout.startswith("COUNT=2"), result.stdout)

    def test_every_rejection_diagnostic_is_secret_free(self):
        cases = (
            (
                {"DSPARK_API_KEYS": "zz-sentinel-control\nrest"},
                SINGLE_LINE_MSG,
                ("zz-sentinel-control",),
            ),
            (
                {"DSPARK_API_KEYS": "zz-sentinel-backslash\\rest"},
                BACKSLASH_MSG,
                ("zz-sentinel-backslash",),
            ),
            (
                {"DSPARK_API_KEYS": "-zz-sentinel-dash"},
                DASH_REJECT_MSG,
                ("zz-sentinel-dash",),
            ),
            (
                {
                    "VLLM_API_KEY": "zz-sentinel-vllm",
                    "DSPARK_API_KEYS": "zz-sentinel-dspark",
                },
                BOTH_SET_MSG,
                ("zz-sentinel-vllm", "zz-sentinel-dspark"),
            ),
        )
        for env, message, sentinels in cases:
            for context, result in self.outcomes(env):
                with self.subTest(context=context, message=message):
                    output = result.stdout + result.stderr
                    self.assertEqual(result.returncode, 2, output)
                    self.assertIn(message, result.stderr)
                    for sentinel in sentinels:
                        self.assertNotIn(sentinel, output)


class AmbientGuard(unittest.TestCase):
    """The normalized .env snapshot is the sole DSPARK_API_KEYS carrier."""

    def test_ambient_only_and_mismatch_are_rejected_secret_free(self):
        cases = (
            ("WORKER_HOST=worker\n", "zz-sentinel-ambient-only"),
            (
                'DSPARK_API_KEYS="file-value"\n',
                "zz-sentinel-ambient-mismatch",
            ),
        )
        for env_text, ambient in cases:
            result = run_launcher_env(env_text, ambient)
            with self.subTest(ambient=ambient):
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(AMBIENT_GUARD_MSG, result.stderr)
                self.assertNotIn(ambient, result.stdout + result.stderr)

    def test_equal_ambient_file_only_and_no_key_paths_pass(self):
        value = "file-one file-two"
        equal = run_launcher_env(
            f'DSPARK_API_KEYS="{value}"\n', ambient_marker=value
        )
        self.assertEqual(equal.returncode, 0, equal.stderr)
        self.assertIn(f"EFFECTIVE=<{value}>", equal.stdout)

        file_only = run_launcher_env(f'DSPARK_API_KEYS="{value}"\n')
        self.assertEqual(file_only.returncode, 0, file_only.stderr)
        self.assertIn(f"EFFECTIVE=<{value}>", file_only.stdout)

        no_key = run_launcher_env("WORKER_HOST=worker\n")
        self.assertEqual(no_key.returncode, 0, no_key.stderr)
        self.assertIn("EFFECTIVE=<>", no_key.stdout)

    def test_keyed_preflight_requires_local_patch_for_either_variable(self):
        for keys_set, vllm_key in (("1", None), ("0", "legacy-key")):
            missing = run_preflight(keys_set, vllm_key, patch_present=False)
            with self.subTest(keys_set=keys_set, vllm_key=vllm_key):
                self.assertEqual(missing.returncode, 1, missing.stderr)
                self.assertIn(PREFLIGHT_MSG, missing.stderr)
                self.assertNotIn("FELL-THROUGH", missing.stdout)

                present = run_preflight(keys_set, vllm_key, patch_present=True)
                self.assertEqual(present.returncode, 0, present.stderr)
                self.assertIn("FELL-THROUGH", present.stdout)

    def test_keyless_preflight_does_not_require_patch(self):
        result = run_preflight("0", None, patch_present=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("FELL-THROUGH", result.stdout)


class RedactionGate(unittest.TestCase):
    """Keyed entrypoints cannot reach exec unless apply and status succeed."""

    @staticmethod
    def dumped_argv(result):
        return [line[2:-1] for line in result.stdout.splitlines()
                if line.startswith("A[") and line.endswith("]")]

    def test_keyed_missing_patch_never_reaches_exec_for_either_key_variable(self):
        for env in (
            {"DSPARK_API_KEYS": "key-one key-two"},
            {"VLLM_API_KEY": "legacy-key"},
        ):
            result, calls = run_redaction_gate(env, patch_present=False)
            with self.subTest(env=env):
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(self.dumped_argv(result), [])
                self.assertEqual(calls, [])

    def test_apply_or_status_failure_never_reaches_exec(self):
        cases = ((1, 0, ["apply"]), (0, 1, ["apply", "--status"]))
        for apply_rc, status_rc, expected_calls in cases:
            result, calls = run_redaction_gate(
                {"DSPARK_API_KEYS": "key-one key-two"},
                patch_present=True,
                apply_rc=apply_rc,
                status_rc=status_rc,
            )
            with self.subTest(apply_rc=apply_rc, status_rc=status_rc):
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(self.dumped_argv(result), [])
                self.assertEqual(calls, expected_calls)

    def test_success_reaches_exec_once_even_when_performance_hotfixes_skip(self):
        for skip in (None, "1"):
            env = {"DSPARK_API_KEYS": "key-one key-two"}
            if skip is not None:
                env["DSPARK_SKIP_HOTFIX"] = skip
            result, calls = run_redaction_gate(env, patch_present=True)
            with self.subTest(skip=skip):
                self.assertEqual(result.returncode, 0, result.stderr)
                argv = self.dumped_argv(result)
                self.assertEqual(argv.count("--api-key"), 1, argv)
                index = argv.index("--api-key")
                self.assertEqual(argv[index + 1:index + 3], ["key-one", "key-two"])
                self.assertEqual(calls, ["apply", "--status"])

    def test_keyless_missing_patch_reaches_exec_without_invocation(self):
        result, calls = run_redaction_gate({}, patch_present=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = self.dumped_argv(result)
        self.assertTrue(argv, result.stdout)
        self.assertNotIn("--api-key", argv)
        self.assertEqual(calls, [])

    def test_gate_shape_is_unskippable_and_outside_optional_loop(self):
        gate = entrypoint_redaction_gate()
        self.assertNotIn("DSPARK_SKIP_HOTFIX", gate)
        self.assertNotIn("-f ", gate)
        self.assertNotIn("|| true", gate)
        self.assertIn("hotfix-vllm-redact-api-key-log.sh || exit 1", gate)
        self.assertIn("hotfix-vllm-redact-api-key-log.sh --status || exit 1", gate)
        loop = next(
            line for line in COMPOSE.read_text(encoding="utf-8").splitlines()
            if "for _hf in " in line
        )
        self.assertNotIn("hotfix-vllm-redact-api-key-log.sh", loop)


class EntrypointInterpolation(unittest.TestCase):
    """Compose `$$` vs `$` on the key variables (container-side expansion)."""

    def test_entrypoint_uses_double_dollar(self):
        block = entrypoint_auth_block()
        self.assertIn("$${DSPARK_API_KEYS", block)
        self.assertIn("$${VLLM_API_KEY", block)
        self.assertEqual(single_dollar_refs(block), [],
                         "host-side interpolation refs in entrypoint block")

    def test_single_dollar_regression_is_caught(self):
        regressed = entrypoint_auth_block().replace("$${", "${")
        self.assertTrue(single_dollar_refs(regressed),
                        "regression gate did not detect `$` in place of `$$`")

    def test_repeated_flag_regression_is_caught(self):
        # Happy path: exactly one --api-key for N keys.
        happy = run_entrypoint({"DSPARK_API_KEYS": "k1 k2"})
        self.assertEqual(happy.stdout.count("--api-key"), 1, happy.stdout)
        # Regress: per-key loop (would leave only the last key valid in vLLM).
        regressed = entrypoint_auth_block().replace(
            'API_KEY_ARGS=(--api-key "$${_dspark_keys[@]}"); fi;',
            'for _rk in "$${_dspark_keys[@]}"; do API_KEY_ARGS+=(--api-key "$${_rk}"); done; fi;',
        ).replace("$$", "$")
        rr = run_bash(
            regressed + "\n" + 'printf "%s\\n" "${API_KEY_ARGS[@]}"',
            {"DSPARK_API_KEYS": "k1 k2"},
        )
        self.assertGreater(rr.stdout.count("--api-key"), 1,
                           "regression gate missed repeated --api-key flags")


# --------------------------------------------------------------------------
# Probe scripts: header selection and conflict handling.
# --------------------------------------------------------------------------


class ProbeAuth(unittest.TestCase):
    def test_probes_consistent_identical_blocks(self):
        texts = {probe_auth_block(p) for p in PROBES}
        self.assertEqual(len(texts), 1, "probe auth blocks must stay identical")

    def test_probes_no_auth_when_unset_empty_whitespace(self):
        for path in PROBES:
            for env in ({}, {"DSPARK_API_KEYS": ""}, {"DSPARK_API_KEYS": "  \t "}):
                r = run_probe(path, env)
                self.assertEqual(r.returncode, 0, (path.name, env))
                self.assertTrue(r.stdout.startswith("COUNT=0"), (path.name, env))

    def test_probes_use_first_parsed_key(self):
        for path in PROBES:
            for value, want in (("k1", "k1"), (" k2  k1 ", "k2"),
                                ("k3 k2 k1", "k3"), ("\tk4 k1", "k4")):
                r = run_probe(path, {"DSPARK_API_KEYS": value})
                self.assertEqual(r.returncode, 0, (path.name, value))
                self.assertIn(f"<Authorization: Bearer {want}>", r.stdout,
                              (path.name, value))

    def test_probes_literal_glob_first_key(self):
        for path in PROBES:
            r = run_probe(path, {"DSPARK_API_KEYS": "g* k1"})
            self.assertIn("<Authorization: Bearer g*>", r.stdout, path.name)

    def test_probes_reject_dash_leading_token(self):
        for path in PROBES:
            r = run_probe(path, {"DSPARK_API_KEYS": "k1 -bad"})
            self.assertEqual(r.returncode, 2, path.name)
            self.assertIn(DASH_REJECT_MSG, r.stderr, path.name)
            self.assertNotIn("-bad", r.stderr + r.stdout, path.name)

    def test_probes_newline_cr_dash_after_newline_exit2(self):
        # ask #4 in the probe contexts: exit 2 + single-line message, and the
        # dash-token message never fires for `\n-bad` (single-line guard wins).
        for path in PROBES:
            for value in ("k1\nk2", "k1\rk2", "k1\n-bad", "k1\r-bad"):
                r = run_probe(path, {"DSPARK_API_KEYS": value})
                self.assertEqual(r.returncode, 2, (path.name, value))
                self.assertIn(SINGLE_LINE_MSG, r.stderr, (path.name, value))
                if value.endswith("-bad"):
                    self.assertNotIn(DASH_REJECT_MSG, r.stderr,
                                     (path.name, value))

    def test_probes_exit_2_naming_both_vars(self):
        for path in PROBES:
            r = run_probe(path, {"VLLM_API_KEY": "vk", "DSPARK_API_KEYS": "k1"})
            self.assertEqual(r.returncode, 2, path.name)
            self.assertIn(BOTH_SET_MSG, r.stderr, path.name)

    def test_probes_vllm_only_unchanged(self):
        for path in PROBES:
            r = run_probe(path, {"VLLM_API_KEY": "vk"})
            self.assertEqual(r.returncode, 0, path.name)
            self.assertEqual(r.stdout,
                             "COUNT=2\n<-H>\n<Authorization: Bearer vk>\n",
                             path.name)


# --------------------------------------------------------------------------
# ComposedHandoff: real docker compose render + the real auth code.
# --------------------------------------------------------------------------


def render_compose(compose_text: str, env_text: str, ambient=None,
                   docker_path=DOCKER, cli_ok=None):
    """Render through Docker Compose, skipping only when its CLI is absent.

    CLI availability is probed once for normal calls. Once available, any
    nonzero config result or invalid JSON raises AssertionError with stderr;
    central handoff coverage must never silently turn green on render failure.
    """
    if cli_ok is None:
        cli_ok = (COMPOSE_CLI_OK if docker_path == DOCKER
                  else compose_cli_available(docker_path, ambient))
    if not cli_ok:
        return None
    with tempfile.TemporaryDirectory() as td:
        compose_copy = Path(td, "docker-compose.dspark.yml")
        stub_env = Path(td, "stub.env")
        compose_copy.write_text(compose_text, encoding="utf-8")
        stub_env.write_text(env_text, encoding="utf-8")
        proc = subprocess.run(
            [docker_path, "compose", "--env-file", str(stub_env), "-f",
             str(compose_copy), "config", "--format", "json"],
            capture_output=True, text=True, env=base_env(ambient), cwd=td,
        )
    if proc.returncode != 0:
        raise AssertionError(
            f"docker compose config failed ({proc.returncode}):\n{proc.stderr}"
        )
    try:
        return json.loads(proc.stdout)
    except ValueError as exc:
        raise AssertionError(
            f"docker compose config returned invalid JSON: {exc}\n"
            f"stderr:\n{proc.stderr}"
        ) from exc

def rendered_env(render) -> dict:
    return render["services"]["vllm-dspark"]["environment"]


def rendered_command(render) -> list:
    return render["services"]["vllm-dspark"]["command"]


def env_from_render(render) -> dict:
    """The rendered compose `environment` as a runnable bash env.

    This is the ONLY carrier for handoff runs: exactly what the container
    would see, plus a usable PATH/HOME (the image defines PATH itself).
    """
    env = {}
    for key, value in rendered_env(render).items():
        if not isinstance(key, str):
            continue
        env[key] = "" if value is None else str(value)
    env.setdefault("PATH", os.environ.get("PATH", "/usr/bin:/bin"))
    env.setdefault("HOME", os.environ.get("HOME", "/"))
    return env


def handoff_entrypoint(block: str, env: dict):
    """Run the rendered container-side auth block under the rendered env."""
    body = (
        block.replace("$$", "$") + "\n"
        + "printf 'COUNT=%s\\n' \"${#API_KEY_ARGS[@]}\";\n"
        + 'printf "<%s>\\n" "${API_KEY_ARGS[@]}"\n'
    )
    return subprocess.run(["bash", "-c", body],
                          capture_output=True, text=True, env=env)


def handoff_argv(block: str, tail: str, env: dict):
    """Rendered block + exec tail under the rendered env → vllm argv."""
    block, tail = _decode_block_tail(block, tail)
    r = subprocess.run(["bash", "-c", block + "\n" + tail],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise AssertionError(f"handoff argv dump failed:\n{r.stderr}")
    return [ln[2:-1] for ln in r.stdout.splitlines()
            if ln.startswith("A[") and ln.endswith("]")]


class ComposeRendering(unittest.TestCase):
    """Docker Compose absence skips; an available-but-broken CLI fails loud."""

    def test_absent_cli_returns_none(self):
        self.assertIsNone(
            render_compose("services: {}\n", "", docker_path=None, cli_ok=False)
        )

    def test_config_failure_raises_with_stderr(self):
        with tempfile.TemporaryDirectory() as td:
            docker = Path(td, "docker")
            docker.write_text(
                "#!/usr/bin/env bash\n"
                "if [ \"${1:-} ${2:-}\" = \"compose version\" ]; then exit 0; fi\n"
                "echo compose-render-sentinel >&2\n"
                "exit 41\n",
                encoding="utf-8",
            )
            docker.chmod(0o755)
            with self.assertRaisesRegex(AssertionError, "compose-render-sentinel"):
                render_compose("services: {}\n", "", docker_path=str(docker))


class ComposedHandoff(unittest.TestCase):
    """Drive the REAL `docker compose config` render plus the REAL auth code.

    Skips only when Docker Compose is absent. Once its version probe succeeds,
    every source/variant render and JSON decode is required to succeed.
    """

    HOSTILE = {"VLLM_API_KEY": "ambient-vk", "DSPARK_API_KEYS": "ambient-k1 ambient-k2"}
    STUB_ENV = (
        "DSPARK_API_KEYS=sk-alice sk-bob\n"
        "VLLM_API_KEY=\n"
        "DSPARK_MODEL=deepseek-ai/DeepSeek-V4-Flash-0731\n"
        "NCCL_SOCKET_IFNAME=eth0\n"
        "NCCL_IB_HCA=mlx5_0\n"
    )

    @classmethod
    def setUpClass(cls):
        cls.render = render_compose(COMPOSE.read_text(encoding="utf-8"),
                                    cls.STUB_ENV)
        if cls.render is not None:
            cls.command2 = rendered_command(cls.render)[2]
            cls.block = find_auth_block(cls.command2)
            cls.tail = tail_from_text(cls.command2)
            cls.env = env_from_render(cls.render)
        else:
            cls.command2 = cls.block = cls.tail = cls.env = None

    def setUp(self):
        if self.render is None:
            self.skipTest("Docker Compose is unavailable")

    def test_rendered_command_contains_fail_closed_redaction_gate(self):
        gate = entrypoint_redaction_gate(self.command2)
        self.assertIn("hotfix-vllm-redact-api-key-log.sh --status || exit 1", gate)

    def test_handoff_env_reaches_container(self):
        # The stub keys are what compose hands the container.
        self.assertEqual(self.env.get("VLLM_API_KEY"), "")
        self.assertEqual(self.env.get("DSPARK_API_KEYS"), "sk-alice sk-bob")

    def test_handoff_exactly_one_flag_carries_every_key(self):
        r = handoff_entrypoint(self.block, self.env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout, "COUNT=3\n<--api-key>\n<sk-alice>\n<sk-bob>\n",
                         r.stderr)
        argv = handoff_argv(self.block, self.tail, self.env)
        self.assertEqual(argv.count("--api-key"), 1, argv)
        i = argv.index("--api-key")
        self.assertEqual(argv[i + 1:i + 3], ["sk-alice", "sk-bob"])

    def test_handoff_passthrough_line_dropped_breaks_chain(self):
        # Negative control (a): remove the DSPARK_API_KEYS environment
        # passthrough line from the compose render; the YAML source's command
        # still contains the auth block, but the chain must detect NO flag.
        source = COMPOSE.read_text(encoding="utf-8")
        lines = [ln for ln in source.splitlines()
                 if not re.match(r'^\s*DSPARK_API_KEYS:\s*"\$\{DSPARK_API_KEYS:-\}"\s*$', ln)]
        variant = "\n".join(lines) + "\n"
        self.assertNotEqual(variant, source, "passthrough drop produced no edit")
        dropped = render_compose(variant, self.STUB_ENV)
        dropped_env = env_from_render(dropped)
        # The stub still set the key; compose just never passed it through.
        self.assertNotIn("DSPARK_API_KEYS", dropped_env)
        argv = handoff_argv(self.block, self.tail, dropped_env)
        self.assertNotIn("--api-key", argv)

    def test_handoff_single_dollar_regression_caught(self):
        # Negative control (b): `$$` → `$` in the entrypoint block. Compose
        # then host-interpolates the stub keys straight into the shell source,
        # so the auth code no longer reads DSPARK_API_KEYS from the env —
        # the interpolation gate must detect the inlining.
        source = COMPOSE.read_text(encoding="utf-8")
        variant = source.replace("$${DSPARK_API_KEYS", "${DSPARK_API_KEYS")
        # Static first: the single-dollar refs are visible in the raw copy.
        variant_block = find_auth_block(variant)
        self.assertTrue(single_dollar_refs(variant_block),
                        "interpolation gate missed `$` in the compose block")
        # Rendered: compose inlines the stub keys; the shell `$` ref is gone.
        inline = render_compose(variant, self.STUB_ENV)
        seek = "${DSPARK_API_KEYS"
        self.assertIn(seek, self.command2,
                      "happy-path block lost its env ref (render changed?)")
        inline_command = rendered_command(inline)[2]
        self.assertIn("sk-alice", inline_command,
                      "variant render did not inline stub keys (gate vacuous)")
        self.assertNotIn(seek, inline_command,
                         "interpolation gate failed: keys inlined into shell source")

    def test_handoff_hermetic_under_hostile_parent(self):
        # Re-render with a hostile parent carrying BOTH key vars: the render
        # (and hence every outcome) must be byte-identical.
        hostile = render_compose(COMPOSE.read_text(encoding="utf-8"),
                                 self.STUB_ENV, ambient=self.HOSTILE)
        self.assertIsNotNone(hostile)
        self.assertEqual(rendered_env(hostile), rendered_env(self.render))
        self.assertEqual(rendered_command(hostile), rendered_command(self.render))
        self.assertEqual(env_from_render(hostile), self.env)


# --------------------------------------------------------------------------
# The handoff harness may splice auth directly to exec only if state is stable.
# --------------------------------------------------------------------------


class MiddleSegmentGuard(unittest.TestCase):
    """No skipped middle statement may reassign auth state or argv."""

    def assert_no_auth_reassignment(self, segment):
        self.assertIsNone(re.search(r"API_KEY_ARGS\s*=", segment), segment)
        self.assertIsNone(re.search(r"_dspark_keys_set\s*=", segment), segment)

    def test_compose_source_middle_does_not_reassign(self):
        source = COMPOSE.read_text(encoding="utf-8")
        self.assert_no_auth_reassignment(middle_segment(source))

    def test_rendered_command_middle_does_not_reassign(self):
        render = render_compose(
            COMPOSE.read_text(encoding="utf-8"), ComposedHandoff.STUB_ENV
        )
        if render is None:
            self.skipTest("Docker Compose is unavailable")
        command = rendered_command(render)[2]
        self.assert_no_auth_reassignment(middle_segment(command))


# --------------------------------------------------------------------------
# Ambient-parent env: hostile parent variants change nothing.
# --------------------------------------------------------------------------


class AmbientParent(unittest.TestCase):
    """A parent env carrying either/both key vars must change no outcome."""

    HOSTILE = ComposedHandoff.HOSTILE

    def test_entrypoint_outcomes_unchanged(self):
        cases = (
            {},
            {"DSPARK_API_KEYS": "k1"},
            {"DSPARK_API_KEYS": "k1 k2"},
            {"DSPARK_API_KEYS": "-bad"},
            {"DSPARK_API_KEYS": "k1\nk2"},
            {"DSPARK_API_KEYS": "\v"},
            {"VLLM_API_KEY": "vk"},
            {"VLLM_API_KEY": "vk", "DSPARK_API_KEYS": "k1"},
        )
        for env in cases:
            clean = run_entrypoint(env)
            hostile = run_entrypoint(env, ambient=self.HOSTILE)
            self.assertEqual(
                (hostile.returncode, hostile.stdout, hostile.stderr),
                (clean.returncode, clean.stdout, clean.stderr),
                env,
            )

    def test_probe_outcomes_unchanged(self):
        cases = (
            {},
            {"DSPARK_API_KEYS": "k1 k2"},
            {"DSPARK_API_KEYS": "-bad"},
            {"DSPARK_API_KEYS": "k1\nk2"},
            {"DSPARK_API_KEYS": "\v"},
            {"VLLM_API_KEY": "vk"},
            {"VLLM_API_KEY": "vk", "DSPARK_API_KEYS": "k1"},
        )
        for path in PROBES:
            for env in cases:
                clean = run_probe(path, env)
                hostile = run_probe(path, env, ambient=self.HOSTILE)
                self.assertEqual(
                    (hostile.returncode, hostile.stdout, hostile.stderr),
                    (clean.returncode, clean.stdout, clean.stderr),
                    (path.name, env),
                )

    def test_argv_outcomes_unchanged(self):
        for env in ({}, {"DSPARK_API_KEYS": "k1 k2"}, {"VLLM_API_KEY": "vk"}):
            self.assertEqual(argv_of(env, ambient=self.HOSTILE), argv_of(env), env)


# --------------------------------------------------------------------------
# Docs + hygiene.
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Worker patch sync covers the redaction hotfix too.
# --------------------------------------------------------------------------


class WorkerSync(unittest.TestCase):
    """The start script's `_hf_sync` loop must actually cp the redaction patch.

    A two-node compose mounts the same `patches/` directory on the worker, and
    start-deepseek-v4-flash-dspark.sh scps a whitelist there before the worker
    container starts. If the redaction hotfix is not in that whitelist, the
    worker rank still prints --api-key values to the startup log.
    """

    def test_whitelist_contains_redact_and_every_file_exists(self):
        sync = re.search(r'for _hf_sync in ([^\n;]+);', START.read_text(encoding="utf-8"))
        self.assertTrue(sync, "_hf_sync whitelist not found in start script")
        names = sync.group(1).split()
        self.assertIn("hotfix-vllm-redact-api-key-log.sh", names)
        missing = [n for n in names
                   if not (ROOT / "patches" / n).is_file()]
        self.assertEqual(missing, [], f"sync whitelists files absent from patches/: {missing}")

    def test_loop_runs_and_ships_every_whitelisted_patch(self):
        with tempfile.TemporaryDirectory() as td:
            stubdir = Path(td, "stubbin")
            stubdir.mkdir()
            log = Path(td, "ssh-scp.log")
            for name in ("ssh", "scp"):
                f = stubdir / name
                f.write_text('#!/bin/bash\n' + f'echo "{name} $*" >>"{log}"\n')
                f.chmod(0o755)
            env = dict(os.environ)
            env["PATH"] = f"{stubdir}:{env.get('PATH','')}"
            env["WORKER_HOST"] = "w"
            env["REMOTE_WORKER_DIR"] = "/wdir"
            env["SCRIPT_DIR"] = str(ROOT)
            chunk = worker_sync_loop()
            r = subprocess.run(["bash", "-c", chunk + "\n"
                                + f'touch "{log}"'],
                               capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            out = log.read_text(encoding="utf-8")
            # every whitelisted patch must be scp'd to the worker dir
            names = re.search(r'for _hf_sync in ([^\n;]+);',
                              START.read_text(encoding="utf-8")).group(1).split()
            for n in names:
                self.assertIn(f"/wdir/patches/{n}", out, n)


class Documented(unittest.TestCase):
    ROUTE_SCOPE = (
        "Every route outside the guarded prefixes `/v1`, `/v2`, `/inference` "
        "is keyless."
    )
    ROUTE_DETAIL = (
        "On the pinned runtime that includes `POST /invocations` and `POST "
        "/generative_scoring` (both run inference unauthenticated) and the "
        "`/tokenize` / `/detokenize` utility routes, besides `/health`, "
        "`/metrics`, `/version`, `/ping`; a keyed deployment still needs "
        "network-level access control on the server port."
    )

    @staticmethod
    def normalized_prose(path):
        parts = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#"):
                line = line[1:].strip()
            parts.append(line)
        return " ".join(" ".join(parts).split())

    def assert_documented_contract(self, path):
        text = path.read_text(encoding="utf-8")
        prose = self.normalized_prose(path)
        for required in (
            "/generative_scoring", "/invocations", "/tokenize",
            "/detokenize", "network-level access control",
            "fails the container before exec",
        ):
            self.assertIn(required, text, (path.name, required))
        self.assertIn(self.ROUTE_SCOPE, prose, path.name)
        self.assertIn(self.ROUTE_DETAIL, prose, path.name)

    def test_env_example_documents_exact_contract(self):
        text = ENV_EXAMPLE.read_text(encoding="utf-8")
        self.assertIn("DSPARK_API_KEYS", text)
        self.assertIn("VLLM_API_KEY", text)
        self.assertIn("exit 2", text)
        self.assertIn("hotfix-vllm-redact-api-key-log.sh", text)
        self.assert_documented_contract(ENV_EXAMPLE)

    def test_envs_doc_documents_exact_contract(self):
        text = ENVS_DOC.read_text(encoding="utf-8")
        self.assertIn("`DSPARK_API_KEYS`", text)
        self.assertIn("`VLLM_API_KEY`", text)
        self.assertIn("hotfix-vllm-redact-api-key-log.sh", text)
        self.assert_documented_contract(ENVS_DOC)

    def test_no_key_material_committed_in_any_changed_text_artifact(self):
        allow = (
            "sk-dspark-alice", "sk-dspark-bob", "sk-single-key",
            "sk-probe-a", "sk-probe-b",
        )
        paths = (
            COMPOSE, ENV_EXAMPLE, *PROBES, ENVS_DOC, CHANGELOG, SELF,
            REDACTION_PATCH, CI_VALIDATE, REDACTION_TEST,
            ENV_NORMALISATION_TEST,
        )
        for changed_path in paths:
            changed_text = changed_path.read_text(encoding="utf-8")
            for match in re.findall(r"sk-[A-Za-z0-9_-]{6,}", changed_text):
                self.assertIn(
                    match,
                    allow,
                    f"possible real key in {changed_path.name}: {match}",
                )

if __name__ == "__main__":
    unittest.main()
