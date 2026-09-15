#!/usr/bin/env python3
"""Behavioral tests for the issue #82 loop-breaker hotfix (CPU only).

vLLM is not importable here, so the tests apply the patch to a synthetic module
that carries the exact production anchors and then *exec* the patched module:
firing thresholds, the counting-window bound, the fence/indent/structural
false-positive controls, the DSML hold and the runtime knob parsing are
exercised as code. CLI tests cover apply/--check/--status, the default-OFF and
skip gates, complete injected-block classification, atomic same-directory
writes with mode preservation, the fail-closed restore paths (including a
post-write re-read/decode failure) and the bounded knob parsing. Consumer cases
run the launcher's real executable slices - patcher admission, resolved
controls, remote Compose wrappers, worker sync statements and the pre-flight and
boot call sites - inside a temporary sandbox whose ssh, scp and docker are
recording, inert local transports. The ssh stub runs each generated remote
command with a login-like environment, the docker stub resolves the Compose
mount to the file the rank actually holds and executes it, and every rank's
target is a synthetic detokenizer carrying the production anchors. The cases
therefore assert the patcher's own behavior (applied / skipped / disabled /
fail-closed, the knobs it resolved and the bytes it consumed) rather than
launcher argv or source text, and never touch a host or a container.

    python3 scripts/test-loop-breaker.py -q
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import logging
import os
import shlex
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
HOTFIX = ROOT / "patches" / "hotfix-dsv4-loop-breaker.py"

# The patched module warns through its own logger when a knob is malformed (it
# runs inside the import block, before the module logger exists). Keep that out
# of the test output; test_malformed_knob_warning_names_the_variable asserts on
# the record itself.
logging.getLogger("vllm.loop-breaker").addHandler(logging.NullHandler())


def _load():
    spec = importlib.util.spec_from_file_location("hotfix_loop_breaker", HOTFIX)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_source(mod, header: str = "stock") -> str:
    """Synthetic detokenizer carrying the exact production anchors.

    The class mirrors the pinned ``BaseIncrementalDetokenizer`` shape the patch
    anchors on (``__init__`` generation block, ``update()`` stop-string tail,
    module import block), so applying the patch here exercises the same
    transformations the container boot performs.
    """
    import_block = (
        mod.IMPORT_OLD_STOCK
        if header == "stock"
        else mod.IMPORT_OLD_WITH_SUPPRESS
    )
    return (
        'import logging\n\n'
        'logger = logging.getLogger("loop-breaker-fixture")\n\n'
        + import_block
        + '\nclass DetokenizerFixture:\n'
        '    def __init__(self):\n'
        '        self.tokens = 0\n'
        + mod.INIT_OLD
        + '\n'
        '    def num_output_tokens(self):\n'
        '        return self.tokens\n'
        '\n'
        '    def update(self, stop=None):\n'
        '        stop_string = None\n'
        '        if stop is not None:\n'
        + mod.RET_OLD
        + '\n'
        '    def feed(self, text, tokens=10):\n'
        '        self.output_text += text\n'
        '        self.tokens += tokens\n'
    )


LAUNCHER = ROOT / "start-deepseek-v4-flash-dspark.sh"
COMPOSE = ROOT / "docker-compose.dspark.yml"
HOTFIX_NAME = "hotfix-dsv4-loop-breaker.py"
ADMISSION_BEGIN = "# Issue #82 loop-breaker patcher admission (begin)."
ADMISSION_END = "# Issue #82 loop-breaker patcher admission (end)."
PARITY_BEGIN = "# Issue #82 loop-breaker rank parity (begin)."
PARITY_END = "# Issue #82 loop-breaker rank parity (end)."
FORWARD_BEGIN = "remote_compose() {"
FORWARD_END = "log_since() {"
SYNC_BEGIN = "# Issue #82 loop-breaker patcher sync (begin)."
SYNC_END = "# Issue #82 loop-breaker patcher sync (end)."
WORKER2_SYNC_BEGIN = '  if [ -f "$DSPARK_C128A_PREFILL_CACHE_HOTFIX" ]'
WORKER2_SYNC_END = '  sync_tp3_patch_dir "$WORKER2_HOST" "$REMOTE_WORKER2_DIR"'
PREFLIGHT_BEGIN = "# Issue #82 loop-breaker (opt-in): validate the enabled knob set and the"
PREFLIGHT_END = 'echo "Starting DSpark worker on ${WORKER_HOST}..."'
BOOT_END = 'echo "Waiting for DSpark vLLM API..."'

# The worker copy of .env.dspark every consumer case pushes: file-backed values
# that disagree with the head environment, so a forwarded control has to be what
# the container actually runs with.
STALE_WORKER_ENV = """\
# Worker copy of .env.dspark: every loop-breaker value here is stale.
DSPARK_LOOP_BREAKER=0
DSPARK_SKIP_LOOP_BREAKER_HOTFIX=0
DSPARK_LOOP_BREAKER_REPEATS=99
DSPARK_LOOP_BREAKER_SHORT_REPEATS=99
DSPARK_LOOP_BREAKER_MIN_TOKENS=999999
DSPARK_LOOP_BREAKER_HOTFIX=/opt/stale-loop-breaker.py
"""

SSH_STUB = """#!/usr/bin/env bash
# Inert remote transport: record the generated command, then run it locally in
# the environment a login shell would present. None of the head's ambient
# DSPARK_* values survive here, so a control reaches a rank only when the
# launcher put it on the remote command line.
host="$1"
shift
printf '%s\\t%s\\n' "$host" "$*" >> "$SSH_RECORD"
exec env -i PATH="$SANDBOX_PATH" HOME="$HOME" TMPDIR="${TMPDIR:-/tmp}" \\
  SSH_RECORD="$SSH_RECORD" SCP_RECORD="$SCP_RECORD" DOCKER_RECORD="$DOCKER_RECORD" \\
  WORKER_ROOT_ONE="$WORKER_ROOT_ONE" WORKER_ROOT_TWO="$WORKER_ROOT_TWO" \\
  bash -c "$*"
"""

SCP_STUB = """#!/usr/bin/env bash
# Inert transfer: record the (source, destination) pair and materialize the
# bytes at the sandbox worker root that the destination host token names. The
# remote path is already rooted at that worker, so an absolute destination is
# used as-is.
printf '%s\\t%s\\n' "$1" "${@: -1}" >> "$SCP_RECORD"
destination="${@: -1}"
host="${destination%%:*}"
path="${destination#*:}"
case "$host" in
  worker-one) root="$WORKER_ROOT_ONE" ;;
  worker-two) root="$WORKER_ROOT_TWO" ;;
  *) echo "scp stub: unknown host $host" >&2; exit 1 ;;
esac
target="$root$path"
case "$path" in "$root"*) target="$path" ;; esac
mkdir -p "$(dirname "$target")"
cp -- "$1" "$target"
"""

# Inert container runtime: resolves the Compose mount for the loop-breaker
# patcher inside the sandbox and executes that file (the canonical synced copy
# on a worker, the selected mount on the head). The container environment is
# built from the real Compose service ``environment:`` block with Compose's
# documented precedence (shell environment over --env-file over the file
# default), so a key the Compose file does not declare never reaches the
# patcher. The rank's synthetic detokenizer is passed as the patcher's
# positional TARGET. Each run appends one JSON record: the resolved patcher path
# and sha256, the argv, the loop-breaker slice of the container environment, the
# exit status and the captured output.
FAKE_DOCKER = r'''#!/usr/bin/env python3
"""Inert container runtime for the loop-breaker consumer cases."""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

LOOP_BREAKER = "/opt/hotfix-dsv4-loop-breaker.py"
TARGET = Path.cwd() / ".container" / "detokenizer.py"


def fail(message):
    print(f"fake docker: {message}", file=sys.stderr)
    raise SystemExit(2)


def main(argv):
    if len(argv) < 3 or argv[1] != "compose":
        fail(f"unsupported invocation: {argv[1:]!r}")
    env_files, files = [], []
    subcommand = entrypoint = service = None
    container = []
    tokens = argv[2:]
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--env-file":
            env_files.append(tokens[index + 1]); index += 2; continue
        if token in ("-f", "--file"):
            files.append(tokens[index + 1]); index += 2; continue
        if token in ("-p", "--project-name"):
            index += 2; continue
        if subcommand is None:
            subcommand = token; index += 1; continue
        if subcommand == "run":
            if service is None:
                if token == "--entrypoint":
                    entrypoint = tokens[index + 1]; index += 2; continue
                if token.startswith("-"):
                    index += 1; continue
                service = token; index += 1; continue
            container.append(token); index += 1; continue
        index += 1

    compose = Path(files[0]) if files else Path("docker-compose.dspark.yml")
    text = compose.read_text()
    file_env = {}
    for name in env_files:
        for line in Path(name).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                file_env[key.strip()] = value.strip().strip('"').strip("'")

    def lookup(key, default=""):
        # Compose precedence: shell environment over --env-file over the default.
        value = os.environ.get(key)
        return value if value else file_env.get(key, default)

    def interpolate(spec):
        match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}", spec)
        if match is None:
            return spec
        return lookup(match.group(1), match.group(2) or "")

    def mounted(container_path):
        match = re.search(
            r"^\s*-\s+(\$\{[^}]+}|[^:\s]+):"
            + re.escape(container_path)
            + r"(?::ro)?\s*$",
            text,
            re.M,
        )
        if match is None:
            return None
        source = interpolate(match.group(1))
        if source.startswith("/"):
            return Path(source)
        return Path.cwd() / source.lstrip("./")

    service_body = re.search(r"^  vllm-dspark:\n(.*?)(?=\n\S|\Z)", text, re.M | re.S)
    if service_body is None:
        fail("the Compose file has no vllm-dspark service")
    environment = {}
    block = re.search(
        r"^    environment:\n(.*?)(?=\n    \S|\n\S|\Z)",
        service_body.group(1),
        re.M | re.S,
    )
    if block is None:
        fail("the vllm-dspark service declares no environment block")
    for line in block.group(1).splitlines():
        key, separator, value = line.strip().partition(":")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        environment[key] = interpolate(value.strip().strip('"'))

    if subcommand == "run":
        if not container:
            fail("run without a container command")
        patcher = mounted(container[0])
        if patcher is None:
            fail(f"no Compose mount backs {container[0]}")
        command = [entrypoint or "python3", str(patcher), *container[1:], str(TARGET)]
        kind = "check" if "--check" in container else "apply"
    elif subcommand == "up":
        patcher = mounted(LOOP_BREAKER)
        if patcher is None:
            fail(f"no Compose mount backs {LOOP_BREAKER}")
        command = ["python3", str(patcher), str(TARGET)]
        kind = "boot"
    else:
        fail(f"unsupported compose subcommand {subcommand!r}")

    child_env = {
        **environment,
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
    }
    if patcher.is_file():
        proc = subprocess.run(command, env=child_env, capture_output=True, text=True)
        rc, stdout, stderr, ran = proc.returncode, proc.stdout, proc.stderr, True
    elif kind == "boot" and not (
        environment.get("DSPARK_LOOP_BREAKER") == "1"
        and environment.get("DSPARK_SKIP_LOOP_BREAKER_HOTFIX") != "1"
    ):
        # A short-syntax bind mount without a source leaves an empty directory
        # and the entrypoint gate never reaches the patcher.
        rc, stdout, stderr, ran = 0, "", "", False
    else:
        fail(f"{command[0]} would run {patcher} but no file backs it")

    record = {
        "kind": kind,
        "cwd": os.getcwd(),
        "patcher": str(patcher),
        "sha256": (
            hashlib.sha256(patcher.read_bytes()).hexdigest() if ran else None
        ),
        "command": command,
        "env": {
            key: value
            for key, value in environment.items()
            if key.startswith("DSPARK_LOOP_BREAKER")
            or key == "DSPARK_SKIP_LOOP_BREAKER_HOTFIX"
        },
        "ran": ran,
        "rc": rc,
        "stdout": stdout,
        "stderr": stderr,
    }
    with open(os.environ["DOCKER_RECORD"], "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
'''


def _launcher_region(begin: str, end: str) -> str:
    """Executable launcher slice between two anchors that must be unique."""
    text = LAUNCHER.read_text()
    for anchor in (begin, end):
        if text.count(anchor) != 1:
            raise AssertionError(f"launcher anchor is not unique: {anchor!r}")
    start = text.index(begin)
    return text[start : text.index(end, start)]


class PatchTextTest(unittest.TestCase):
    def setUp(self):
        self.mod = _load()

    def test_patched_source_from_both_header_states_is_executable(self):
        for header in ("stock", "suppress"):
            with self.subTest(header=header):
                patched, status = self.mod.apply_text(
                    fixture_source(self.mod, header)
                )
                self.assertEqual(status, "applied")
                compile(patched, "<fixture>", "exec")

    def test_second_apply_is_skipped_and_byte_identical(self):
        once, status = self.mod.apply_text(fixture_source(self.mod))
        self.assertEqual(status, "applied")
        twice, status = self.mod.apply_text(once)
        self.assertEqual(status, "skipped")
        self.assertEqual(once, twice)

    def test_partial_patch_is_not_an_idempotent_hit(self):
        once, _ = self.mod.apply_text(fixture_source(self.mod))
        partial = once.replace("        if stop_string is None and _LB_ENABLED:\n", "")
        self.assertNotEqual(partial, once)
        new, status = self.mod.apply_text(partial)
        self.assertEqual(status, "partial")
        self.assertEqual(new, partial)
        self.assertEqual(self.mod.classify(partial), "partial")
        self.assertEqual(self.mod.classify(once), "applied")

    def test_missing_anchors_are_named_and_never_written(self):
        drifted = 'print("not a detokenizer")\n'
        new, status = self.mod.apply_text(drifted)
        self.assertTrue(status.startswith("missing:"), status)
        self.assertEqual(new, drifted)
        for part in ("import", "init", "return"):
            self.assertIn(part, status)


class CliTest(unittest.TestCase):
    def setUp(self):
        self.mod = _load()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _fixture(self, mode: int = 0o640) -> Path:
        target = self.root / "detokenizer.py"
        target.write_text(fixture_source(self.mod))
        target.chmod(mode)
        return target

    def _run(self, argv, env) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, env, clear=True):
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = self.mod.main(["hotfix-dsv4-loop-breaker.py", *argv])
        return code, out.getvalue(), err.getvalue()

    def _leftovers(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_file())

    def test_disabled_boot_ignores_knobs_and_writes_nothing(self):
        target = self._fixture()
        original = target.read_bytes()
        for value in (None, "0", "true", "2", ""):
            with self.subTest(value=value):
                env = {"DSPARK_LOOP_BREAKER_REPEATS": "not-an-int"}
                if value is not None:
                    env["DSPARK_LOOP_BREAKER"] = value
                code, out, _ = self._run([str(target)], env)
                self.assertEqual(code, 0, out)
                self.assertIn("disabled", out)
        code, out, _ = self._run(["--check", str(target)], {"DSPARK_LOOP_BREAKER_REPEATS": "abc"})
        self.assertEqual(code, 0, out)
        self.assertIn("disabled", out)
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(self._leftovers(), ["detokenizer.py"])

    def test_skip_flag_skips_apply_and_check_but_not_status(self):
        target = self._fixture()
        original = target.read_bytes()
        env = {"DSPARK_LOOP_BREAKER": "1", "DSPARK_SKIP_LOOP_BREAKER_HOTFIX": "1"}
        code, out, _ = self._run([str(target)], env)
        self.assertEqual(code, 0, out)
        self.assertIn("DSPARK_SKIP_LOOP_BREAKER_HOTFIX=1", out)
        self.assertEqual(target.read_bytes(), original)
        code, out, _ = self._run(["--check", str(target)], env)
        self.assertEqual(code, 0, out)
        self.assertIn("skipped", out)
        self.assertEqual(target.read_bytes(), original)

        # --status stays a byte-state query: the skip flag must not mask it.
        code, out, _ = self._run(["--status", str(target)], env)
        self.assertEqual(code, 1, out)
        self.assertIn("STOCK", out)
        applied, _ = self.mod.apply_text(target.read_text())
        target.write_text(applied)
        code, out, _ = self._run(["--status", str(target)], env)
        self.assertEqual(code, 0, out)
        self.assertIn("APPLIED", out)

    def test_enabled_apply_writes_atomically_and_preserves_mode(self):
        target = self._fixture(0o640)
        seen_dir = []
        real_mkstemp = tempfile.mkstemp

        def spy(*args, **kwargs):
            seen_dir.append(kwargs.get("dir"))
            return real_mkstemp(*args, **kwargs)

        with mock.patch.object(tempfile, "mkstemp", spy):
            code, out, err = self._run([str(target)], {"DSPARK_LOOP_BREAKER": "1"})
        self.assertEqual(code, 0, out + err)
        self.assertEqual(seen_dir, [str(target.parent)])
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o640)
        self.assertEqual(self.mod.classify(target.read_text()), "applied")
        self.assertEqual(self._leftovers(), ["detokenizer.py"])
        compile(target.read_text(), str(target), "exec")

    def test_write_failure_leaves_target_untouched(self):
        target = self._fixture()
        original = target.read_bytes()
        with mock.patch.object(
            tempfile, "mkstemp", side_effect=OSError("read-only filesystem")
        ):
            code, _, err = self._run([str(target)], {"DSPARK_LOOP_BREAKER": "1"})
        self.assertEqual(code, 1)
        self.assertIn("FAIL-CLOSED", err)
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(self._leftovers(), ["detokenizer.py"])

    def test_uncompilable_patch_never_reaches_the_target(self):
        target = self._fixture()
        original = target.read_bytes()
        try:
            self.mod.RET_NEW = "        return (\n"
            code, _, err = self._run([str(target)], {"DSPARK_LOOP_BREAKER": "1"})
        finally:
            reloaded = _load()
            self.mod.RET_NEW = reloaded.RET_NEW
        self.assertEqual(code, 1)
        self.assertIn("does not compile", err)
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(self._leftovers(), ["detokenizer.py"])

    def test_failed_postcondition_restores_the_original(self):
        target = self._fixture()
        original = target.read_bytes()
        real_write = self.mod.write_atomically
        calls = []

        def tampering_write(path, payload, mode):
            calls.append(payload)
            if len(calls) == 1:
                # Simulate a corrupted first write: the postcondition must see
                # it, restore the original and fail closed.
                real_write(path, payload.replace("def _lb_check(self)", "def _lb_check("), mode)
            else:
                real_write(path, payload, mode)

        with mock.patch.object(self.mod, "write_atomically", tampering_write):
            code, _, err = self._run([str(target)], {"DSPARK_LOOP_BREAKER": "1"})
        self.assertEqual(code, 1)
        self.assertIn("post-apply verification failed", err)
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(self._leftovers(), ["detokenizer.py"])

    def test_post_write_read_failure_restores_the_original(self):
        failures = (
            ("OSError", OSError("simulated re-read failure")),
            (
                "UnicodeDecodeError",
                UnicodeDecodeError("utf-8", b"\xff", 0, 1, "simulated decode failure"),
            ),
        )
        for name, failure in failures:
            with self.subTest(name=name):
                target = self._fixture()
                original = target.read_bytes()
                real_read_text = Path.read_text
                reads = []

                def failing_read_text(path, *args, **kwargs):
                    reads.append(path)
                    if len(reads) == 2:
                        raise failure
                    return real_read_text(path, *args, **kwargs)

                with mock.patch.object(Path, "read_text", failing_read_text):
                    code, _, err = self._run(
                        [str(target)], {"DSPARK_LOOP_BREAKER": "1"}
                    )
                self.assertEqual(code, 1)
                self.assertIn("FAIL-CLOSED", err)
                self.assertIn("original restored", err)
                self.assertEqual(target.read_bytes(), original)
                self.assertEqual(self._leftovers(), ["detokenizer.py"])

    def test_post_write_read_failure_reports_a_failed_restore(self):
        target = self._fixture()
        real_read_text = Path.read_text
        real_write = self.mod.write_atomically
        reads, writes = [], []

        def failing_read_text(path, *args, **kwargs):
            reads.append(path)
            if len(reads) == 2:
                raise OSError("simulated re-read failure")
            return real_read_text(path, *args, **kwargs)

        def failing_restore(path, payload, mode):
            writes.append(payload)
            if len(writes) == 2:
                raise OSError("simulated restore failure")
            real_write(path, payload, mode)

        with mock.patch.object(Path, "read_text", failing_read_text), \
                mock.patch.object(self.mod, "write_atomically", failing_restore):
            code, _, err = self._run([str(target)], {"DSPARK_LOOP_BREAKER": "1"})
        self.assertEqual(code, 1)
        self.assertIn("cannot restore", err)
        # The restore failed, so the written patch is still on disk; the CLI
        # must have said so instead of reporting success.
        self.assertEqual(self.mod.classify(target.read_text()), "applied")

    def test_huge_digit_knobs_fail_closed_without_conversion_errors(self):
        target = self._fixture()
        original = target.read_bytes()
        huge = "9" * 5000
        for name in (
            "DSPARK_LOOP_BREAKER_REPEATS",
            "DSPARK_LOOP_BREAKER_SHORT_REPEATS",
            "DSPARK_LOOP_BREAKER_MIN_TOKENS",
        ):
            with self.subTest(name=name):
                code, _, err = self._run(
                    [str(target)], {"DSPARK_LOOP_BREAKER": "1", name: huge}
                )
                self.assertEqual(code, 1)
                self.assertIn("FAIL-CLOSED", err)
                self.assertIn(name, err)
                self.assertEqual(target.read_bytes(), original)

    def test_zero_padded_in_range_knob_still_applies(self):
        target = self._fixture()
        code, out, err = self._run(
            [str(target)],
            {
                "DSPARK_LOOP_BREAKER": "1",
                "DSPARK_LOOP_BREAKER_REPEATS": "0" * 5000 + "6",
            },
        )
        self.assertEqual(code, 0, out + err)
        self.assertEqual(self.mod.classify(target.read_text()), "applied")

    def test_enabled_rejects_malformed_and_out_of_range_knobs(self):
        target = self._fixture()
        original = target.read_bytes()
        rows = (
            ("DSPARK_LOOP_BREAKER_REPEATS", "abc"),
            ("DSPARK_LOOP_BREAKER_REPEATS", "-3"),
            ("DSPARK_LOOP_BREAKER_REPEATS", "1"),
            ("DSPARK_LOOP_BREAKER_REPEATS", "1025"),
            ("DSPARK_LOOP_BREAKER_REPEATS", " 6"),
            ("DSPARK_LOOP_BREAKER_REPEATS", "6.0"),
            ("DSPARK_LOOP_BREAKER_SHORT_REPEATS", "0"),
            ("DSPARK_LOOP_BREAKER_MIN_TOKENS", "1000001"),
            ("DSPARK_LOOP_BREAKER_MIN_TOKENS", "\uff16"),
        )
        for name, value in rows:
            with self.subTest(name=name, value=value):
                code, _, err = self._run(
                    [str(target)], {"DSPARK_LOOP_BREAKER": "1", name: value}
                )
                self.assertEqual(code, 1)
                self.assertIn("FAIL-CLOSED", err)
                self.assertIn(name, err)
                self.assertEqual(target.read_bytes(), original)

    def test_enabled_accepts_documented_defaults_and_bounds(self):
        rows = (
            {},
            {"DSPARK_LOOP_BREAKER_REPEATS": "2"},
            {"DSPARK_LOOP_BREAKER_SHORT_REPEATS": "1024"},
            {"DSPARK_LOOP_BREAKER_MIN_TOKENS": "0"},
            {"DSPARK_LOOP_BREAKER_MIN_TOKENS": "1000000"},
        )
        for extra in rows:
            with self.subTest(extra=extra):
                target = self._fixture()
                code, out, err = self._run(
                    [str(target)], {"DSPARK_LOOP_BREAKER": "1", **extra}
                )
                self.assertEqual(code, 0, out + err)
                self.assertEqual(self.mod.classify(target.read_text()), "applied")
        # A second run over a fully patched target stays a byte-identical no-op.
        target = self._fixture()
        self.assertEqual(self._run([str(target)], {"DSPARK_LOOP_BREAKER": "1"})[0], 0)
        once = target.read_bytes()
        code, out, err = self._run([str(target)], {"DSPARK_LOOP_BREAKER": "1"})
        self.assertEqual(code, 0, out + err)
        self.assertIn("skipped", out)
        self.assertEqual(target.read_bytes(), once)

    def test_check_mode_reports_ready_drift_and_partial(self):
        ready = self._fixture()
        code, out, err = self._run(["--check", str(ready)], {"DSPARK_LOOP_BREAKER": "1"})
        self.assertEqual(code, 0, out + err)
        self.assertIn("READY", out)
        self.assertIn("repeats=6", out)
        self.assertIn("min_tokens=64", out)

        applied_text, _ = self.mod.apply_text(fixture_source(self.mod))
        ready.write_text(applied_text)
        code, out, _ = self._run(["--check", str(ready)], {"DSPARK_LOOP_BREAKER": "1"})
        self.assertEqual(code, 0, out)
        self.assertIn("already applied", out)

        drifted = self.root / "drifted.py"
        drifted.write_text('print("not a detokenizer")\n')
        code, _, err = self._run(["--check", str(drifted)], {"DSPARK_LOOP_BREAKER": "1"})
        self.assertEqual(code, 1)
        self.assertIn("expected anchors", err)

        partial = self.root / "partial.py"
        partial.write_text(
            applied_text.replace("        if stop_string is None and _LB_ENABLED:\n", "")
        )
        code, _, err = self._run(["--check", str(partial)], {"DSPARK_LOOP_BREAKER": "1"})
        self.assertEqual(code, 1)
        self.assertIn("partial patch", err)
        missing = self.root / "absent.py"
        code, out, _ = self._run(["--status", str(missing)], {"DSPARK_LOOP_BREAKER": "1"})
        self.assertEqual(code, 1)
        self.assertIn("NOT APPLIED", out)

    def test_status_is_nonzero_on_partial_and_missing_state(self):
        target = self._fixture()
        code, out, _ = self._run(["--status", str(target)], {})
        self.assertEqual(code, 1)
        self.assertIn("STOCK", out)

        applied_text, _ = self.mod.apply_text(target.read_text())
        target.write_text(applied_text)
        code, out, _ = self._run(["--status", str(target)], {})
        self.assertEqual(code, 0, out)
        self.assertIn("APPLIED", out)

        target.write_text(
            applied_text.replace("    def _lb_check(self) -> str | None:\n", "")
        )
        code, out, _ = self._run(["--status", str(target)], {})
        self.assertEqual(code, 1)
        self.assertIn("PARTIAL", out)

    def test_incomplete_injected_code_outside_the_markers_is_refused(self):
        target = self._fixture()
        applied_text, _ = self.mod.apply_text(target.read_text())
        # Executable injected state is deleted while every substring the old
        # classifier counted would still match: the module raises on first use.
        gutted = applied_text
        for line in (
            "        self._lb_counts: dict[str, int] = {}\n",
            "        self._lb_hist: deque[str] = deque()\n",
            "        self._lb_scan: int = 0\n",
        ):
            gutted = gutted.replace(line, "")
        self.assertNotEqual(gutted, applied_text)
        target.write_text(gutted)

        code, out, _ = self._run(["--status", str(target)], {})
        self.assertEqual(code, 1)
        self.assertIn("PARTIAL", out)

        code, _, err = self._run(["--check", str(target)], {"DSPARK_LOOP_BREAKER": "1"})
        self.assertEqual(code, 1)
        self.assertIn("partial patch", err)

        code, _, err = self._run([str(target)], {"DSPARK_LOOP_BREAKER": "1"})
        self.assertEqual(code, 1)
        self.assertIn("FAIL-CLOSED", err)
        self.assertEqual(target.read_text(), gutted)

    def test_uncompilable_applied_source_is_not_reported_applied(self):
        target = self._fixture()
        applied_text, _ = self.mod.apply_text(target.read_text())
        target.write_text(applied_text + "def broken(:\n")

        code, out, _ = self._run(["--status", str(target)], {})
        self.assertEqual(code, 1)
        self.assertIn("PARTIAL", out)

        code, _, err = self._run([str(target)], {"DSPARK_LOOP_BREAKER": "1"})
        self.assertEqual(code, 1)
        self.assertIn("FAIL-CLOSED", err)


class DetectorTest(unittest.TestCase):
    """Exec the patched fixture and drive the detector through update()."""

    def setUp(self):
        self.mod = _load()
        self.patched, status = self.mod.apply_text(fixture_source(self.mod))
        self.assertEqual(status, "applied")

    def _exec(self, env: dict[str, str]) -> dict:
        with mock.patch.dict(os.environ, env, clear=True):
            namespace: dict = {"__name__": "loop_breaker_fixture"}
            exec(compile(self.patched, "<fixture>", "exec"), namespace)
        return namespace

    def _detector(self, env: dict[str, str] | None = None):
        namespace = self._exec({"DSPARK_LOOP_BREAKER": "1", **(env or {})})
        return namespace["DetokenizerFixture"]()

    @staticmethod
    def _repeat(fixture, line: str, times: int, tokens: int = 20) -> str | None:
        result = None
        for _ in range(times):
            fixture.feed(line + "\n", tokens)
            result = fixture.update()
        return result

    def test_prose_repeat_fires_exactly_at_the_configured_bound(self):
        for count, expected in ((5, None), (6, "loop-breaker")):
            with self.subTest(count=count):
                fixture = self._detector()
                self.assertEqual(self._repeat(fixture, "Doing it.", count), expected)

    def test_short_sentence_tier_needs_its_own_bound(self):
        fixture = self._detector()
        self.assertEqual(self._repeat(fixture, "Now.", 14), None)
        self.assertEqual(self._repeat(fixture, "Now.", 1), "loop-breaker")
        # A six-repeat short line is prose noise, not a loop.
        self.assertEqual(self._repeat(self._detector(), "Yes.", 6), None)
        # Below the short tier's length floor and never sentence-terminated.
        self.assertEqual(self._repeat(self._detector(), "ok", 40), None)

    def test_token_floor_defers_fire_until_the_bound_is_passed(self):
        fixture = self._detector()
        self.assertEqual(self._repeat(fixture, "Doing it.", 6, tokens=5), None)
        fixture.tokens = 64
        self.assertEqual(self._repeat(fixture, "Doing it.", 1), "loop-breaker")

    def test_configured_bounds_and_window_follow_the_knobs(self):
        fixture = self._detector(
            {"DSPARK_LOOP_BREAKER_REPEATS": "2", "DSPARK_LOOP_BREAKER_MIN_TOKENS": "0"}
        )
        self.assertEqual(self._repeat(fixture, "Doing it.", 2), "loop-breaker")

    def test_fenced_code_repetition_never_fires(self):
        for fence in ("```", "~~~", "```python"):
            with self.subTest(fence=fence):
                fixture = self._detector()
                fixture.feed(fence + "\n", 5)
                # Unindented, so only the fence state can suppress the count.
                self.assertIsNone(
                    self._repeat(fixture, "return value_that_repeats()", 40)
                )
                fixture.feed(fence + "\n", 5)
                fixture.feed("Closing prose line.\n", 5)
                self.assertIsNone(fixture.update())

    def test_indented_code_repetition_never_fires(self):
        for line in ("    value = compute(queue)", "\tvalue = compute(queue)"):
            with self.subTest(line=line.split("=")[0].strip() or "tab"):
                fixture = self._detector()
                self.assertIsNone(self._repeat(fixture, line, 40))

    def test_structural_and_marker_lines_are_never_counted(self):
        rows = (
            '{"tool": "call", "name": "bash"}',
            "const x = `template ${value}`;",
            "<\uff5cDSML\uff5cfunction_calls>",
            "---",
            "| column a | column b |",
            "x" * 200,
        )
        for line in rows:
            with self.subTest(line=line[:24]):
                fixture = self._detector()
                self.assertIsNone(self._repeat(fixture, line, 40))

    def test_dsml_tail_holds_fire_until_the_tool_block_closes(self):
        fixture = self._detector()
        self.assertEqual(self._repeat(fixture, "Doing it.", 6), "loop-breaker")
        fixture.feed("<\uff5cDSML\uff5cinvoke name=\"bash\">\n", 10)
        self.assertIsNone(fixture.update())
        fixture.feed("x" * 500 + "\n", 20)
        self.assertEqual(self._repeat(fixture, "Doing it.", 1), "loop-breaker")

    def test_counting_window_decays_and_stays_bounded(self):
        knobs = {
            "DSPARK_LOOP_BREAKER_REPEATS": "2",
            "DSPARK_LOOP_BREAKER_SHORT_REPEATS": "2",
            "DSPARK_LOOP_BREAKER_MIN_TOKENS": "0",
        }
        within = self._detector(knobs)
        within.feed("alpha repeated line\n", 5)
        within.feed("".join(f"filler {i}\n" for i in range(60)), 100)
        within.feed("alpha repeated line\n", 5)
        self.assertEqual(within.update(), "loop-breaker")

        decayed = self._detector(knobs)
        decayed.feed("alpha repeated line\n", 5)
        decayed.feed("".join(f"filler {i}\n" for i in range(200)), 300)
        decayed.feed("alpha repeated line\n", 5)
        self.assertIsNone(decayed.update())

        long_generation_ns = self._exec({"DSPARK_LOOP_BREAKER": "1", **knobs})
        window = long_generation_ns["_LB_WINDOW"]
        long_generation = long_generation_ns["DetokenizerFixture"]()
        for i in range(500):
            long_generation.feed(f"line {i} of a long answer\n", 3)
            long_generation.update()
        self.assertLessEqual(len(long_generation._lb_counts), window)
        self.assertLessEqual(len(long_generation._lb_hist), window)

    def test_alternating_two_line_babble_fires(self):
        fixture = self._detector()
        result = None
        for _ in range(6):
            fixture.feed("Doing it.\n", 10)
            result = fixture.update()
            fixture.feed("Running.\n", 10)
            result = fixture.update()
        self.assertEqual(result, "loop-breaker")

    def test_healthy_verbose_turn_is_never_touched(self):
        fixture = self._detector()
        transcript = [
            "# Result",
            "",
            "The run completed with the expected shape.",
            "",
            "```python",
            'print("value")',
            "print(value)",
            "```",
            "",
            '{"tool": "bash", "command": "ls"}',
            "",
            "Next step:",
            "- Inspect the log.",
            "- Inspect the log.",
            "- Rerun the sampler.",
            "",
            "The sampler produced the expected sequence.",
        ]
        for line in transcript:
            fixture.feed(line + "\n", 20)
            self.assertIsNone(fixture.update(), line)
        # Four prose repeats of the same line, inside the bound.
        self.assertIsNone(self._repeat(fixture, "Retrying the request now.", 4))

    def test_disabled_module_is_inert_and_parses_no_knob(self):
        for value in (None, "0", "true", "2", ""):
            with self.subTest(value=value):
                env = {"DSPARK_LOOP_BREAKER_REPEATS": "not-an-int"}
                if value is not None:
                    env["DSPARK_LOOP_BREAKER"] = value
                namespace = self._exec(env)
                self.assertFalse(namespace["_LB_ENABLED"])
                fixture = namespace["DetokenizerFixture"]()
                self.assertIsNone(self._repeat(fixture, "Doing it.", 50))

    def test_malformed_knobs_disarm_instead_of_aborting_import(self):
        rows = ("abc", "1", "0", "-3", "1025", " 6", "6.0", "\uff16")
        for value in rows:
            with self.subTest(value=value):
                namespace = self._exec(
                    {"DSPARK_LOOP_BREAKER": "1", "DSPARK_LOOP_BREAKER_REPEATS": value}
                )
                self.assertFalse(namespace["_LB_ENABLED"])
                fixture = namespace["DetokenizerFixture"]()
                self.assertIsNone(self._repeat(fixture, "Doing it.", 40))

    def test_malformed_knob_warning_names_the_variable(self):
        with self.assertLogs("vllm.loop-breaker", level="WARNING") as captured:
            namespace = self._exec(
                {
                    "DSPARK_LOOP_BREAKER": "1",
                    "DSPARK_LOOP_BREAKER_SHORT_REPEATS": "many",
                }
            )
        self.assertFalse(namespace["_LB_ENABLED"])
        self.assertEqual(len(captured.records), 1)
        message = captured.records[0].getMessage()
        self.assertIn("DSPARK_LOOP_BREAKER_SHORT_REPEATS", message)
        self.assertIn("'many'", message)
        self.assertIn("[loop-breaker]", message)

    def test_huge_digit_knobs_disarm_the_import(self):
        huge = "9" * 5000
        for name in (
            "DSPARK_LOOP_BREAKER_REPEATS",
            "DSPARK_LOOP_BREAKER_SHORT_REPEATS",
            "DSPARK_LOOP_BREAKER_MIN_TOKENS",
        ):
            with self.subTest(name=name):
                with self.assertLogs("vllm.loop-breaker", level="WARNING") as captured:
                    namespace = self._exec({"DSPARK_LOOP_BREAKER": "1", name: huge})
                self.assertFalse(namespace["_LB_ENABLED"])
                self.assertEqual(len(captured.records), 1)
                self.assertIn(name, captured.records[0].getMessage())
                fixture = namespace["DetokenizerFixture"]()
                self.assertIsNone(self._repeat(fixture, "Doing it.", 40))

    def test_zero_padded_knob_arms_with_the_normalized_value(self):
        namespace = self._exec(
            {"DSPARK_LOOP_BREAKER": "1", "DSPARK_LOOP_BREAKER_REPEATS": "0" * 5000 + "6"}
        )
        self.assertTrue(namespace["_LB_ENABLED"])
        self.assertEqual(namespace["_LB_REPEATS"], 6)

    def test_cli_and_runtime_knob_bounds_agree(self):
        specs = self.mod.KNOB_SPECS
        defaults = {name: default for name, default, _, _ in specs}
        for name, _, low, high in specs:
            for value in (str(low), str(high), str(low - 1), str(high + 1)):
                with self.subTest(name=name, value=value):
                    namespace = self._exec({"DSPARK_LOOP_BREAKER": "1", name: value})
                    runtime_armed = namespace["_LB_ENABLED"]
                    in_range = low <= int(value) <= high
                    self.assertEqual(runtime_armed, in_range, (name, value))
                    try:
                        self.mod.resolve_knobs({name: value})
                        cli_accepted = True
                    except self.mod.LoopBreakerConfigError:
                        cli_accepted = False
                    self.assertEqual(cli_accepted, runtime_armed, (name, value))
                    observed = tuple(
                        namespace[key]
                        for key in (
                            "_LB_REPEATS",
                            "_LB_SHORT_REPEATS",
                            "_LB_MIN_TOKENS",
                        )
                    )
                    if runtime_armed:
                        expected = dict(defaults)
                        expected[name] = int(value)
                        self.assertEqual(
                            observed,
                            tuple(expected[spec[0]] for spec in specs),
                        )
                    else:
                        self.assertEqual(observed, (0, 0, 0))


class _ConsumerSandbox:
    """Temporary head + two workers whose transports are recording and inert.

    The head and each worker hold the repository patcher copy, the stale worker
    ``.env.dspark``, the Compose file and a synthetic stock detokenizer. The
    selected override is a byte-distinct copy of the shipped patcher, so a case
    can tell the bytes a rank consumed from the bytes it merely held.
    """

    def __init__(self, root, mod, ambient, override="selected"):
        self.root = root
        self.mod = mod
        self.head = root / "head"
        self.worker_roots = {"one": root / "one", "two": root / "two"}
        self.bin = root / "bin"
        self.records = {
            name: root / f"{name}-record.txt" for name in ("ssh", "scp", "docker")
        }
        for path in self.records.values():
            path.touch()
        self.bin.mkdir()
        for executable, body in (("ssh", SSH_STUB), ("scp", SCP_STUB), ("docker", FAKE_DOCKER)):
            stub = self.bin / executable
            stub.write_text(body)
            stub.chmod(0o755)
        self.stock = fixture_source(mod).encode()
        for directory in (self.head, *self.worker_roots.values()):
            (directory / "patches").mkdir(parents=True)
            (directory / ".container").mkdir()
            (directory / ".env.dspark").write_text(STALE_WORKER_ENV)
            (directory / "docker-compose.dspark.yml").write_bytes(COMPOSE.read_bytes())
            (directory / ".container" / "detokenizer.py").write_bytes(self.stock)
        self.repository = HOTFIX.read_bytes() + b"\n# sandbox: repository copy bytes\n"
        self.selected = HOTFIX.read_bytes() + b"\n# sandbox: selected override bytes\n"
        for directory in (self.head, *self.worker_roots.values()):
            (directory / "patches" / HOTFIX_NAME).write_bytes(self.repository)
        self.selected_path = root / "selected-loop-breaker.py"
        self.selected_path.write_bytes(self.selected)
        if override == "selected":
            self.override = self.selected_path
        elif override == "missing":
            self.override = root / "absent-loop-breaker.py"
        elif override == "directory":
            self.override = root / "selected-loop-breaker-dir"
            self.override.mkdir()
        elif override == "symlink":
            self.override = root / "selected-loop-breaker-link.py"
            self.override.symlink_to(self.selected_path)
        else:
            self.override = Path(override)
        self.sha = {
            "repository": hashlib.sha256(self.repository).hexdigest(),
            "selected": hashlib.sha256(self.selected).hexdigest(),
        }
        self.proc = self._run(ambient)

    def _run(self, ambient):
        one = self.worker_roots["one"]
        two = self.worker_roots["two"]
        responses_store = (
            "DSPARK_RESPONSES_STORE_REMOTE_ENV="
            "VLLM_ENABLE_RESPONSES_API_STORE='0' "
            "DSPARK_RESPONSES_STORE_MAX_ENTRIES='256'"
        )
        absent_c128a = shlex.quote(str(self.root / "absent-c128a-prefill-cache.py"))
        exports = [
            f"export {key}={shlex.quote(value)}"
            for key, value in sorted(ambient.items())
        ]
        exports.append(
            f"export DSPARK_LOOP_BREAKER_HOTFIX={shlex.quote(str(self.override))}"
        )
        script = "\n".join(
            [
                "set -euo pipefail",
                f"SCRIPT_DIR={shlex.quote(str(self.head))}",
                "PROJECT_NAME=dspark",
                "COMPOSE_FILE=docker-compose.dspark.yml",
                f"COMPOSE_ENV_FILE={shlex.quote(str(self.head / '.env.dspark'))}",
                "WORKER_HOST=worker-one",
                "WORKER2_HOST=worker-two",
                f"WORKER_DIR={shlex.quote(str(one))}",
                f"WORKER2_DIR={shlex.quote(str(two))}",
                f"REMOTE_WORKER_DIR={shlex.quote(str(one))}",
                f"REMOTE_WORKER2_DIR={shlex.quote(str(two))}",
                "DSPARK_TP3=1",
                "TP_SIZE=2",
                "NNODES=2",
                "REMOTE_C128A_PREFILL_CACHE=0",
                "REMOTE_ISSUE136_ENABLE=0",
                "REMOTE_ISSUE191_ENABLE=0",
                "REMOTE_ISSUE191_RETRIES=2",
                "REMOTE_ISSUE191_MODE=failclosed",
                "REMOTE_ISSUE191_THINKOFF=1",
                "REMOTE_ASYNC_SCHEDULING=1",
                "REMOTE_DSPARK_BLOCK_K=0",
                "REMOTE_ROPE_SWA_FIX=0",
                "REMOTE_DSPARK_SWA_PREFIX=0",
                "REMOTE_DSML_RECOVERY=0",
                "REMOTE_MXFP4_INDEXER=0",
                "REMOTE_ISSUE144_EFFORT_ALIGN=0",
                "WORKER_HF_COMPOSE_ENV=HF_CACHE='/cache/huggingface'",
                "WORKER2_HF_COMPOSE_ENV=HF_CACHE='/cache/huggingface'",
                'WORKER_COMPOSE_FILES="-f docker-compose.dspark.yml"',
                'WORKER2_COMPOSE_FILES="-f docker-compose.dspark.yml"',
                "DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT=0",
                "DSPARK_ENABLE_CODEX_AGENT_MESSAGE_COMPAT=0",
                responses_store,
                "WORKER_VLLM_HOST_IP=10.0.0.21",
                "WORKER2_VLLM_HOST_IP=10.0.0.22",
                "GPU_MEMORY_UTILIZATION=0.835",
                "DSPARK_MODEL=deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
                "DSPARK_REVISION=main",
                "DSPARK_ISSUE141_EFFECTIVE=0",
                "DSPARK_SP_INDEXER_EFFECTIVE=0",
                "DSPARK_DEEPGEMM_ALIAS_EFFECTIVE=0",
                "ENABLE_VLLM_GB10_PATCH=0",
                "VLLM_GB10_PATCH_DIR=./vllm_patch_gb10",
                f"DSPARK_C128A_PREFILL_CACHE_HOTFIX={absent_c128a}",
                "remote_nccl_env() { printf \"NCCL_IB_HCA='rocep1s0f0'\"; }",
                "remote_nccl_env2() { printf \"NCCL_IB_HCA='rocep2s0f0'\"; }",
                "REMOTE_COMPOSE="
                + shlex.quote(
                    f"cd {one} && env -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK "
                    "-u HEADLESS COMPOSE_DISABLE_ENV_FILE=1"
                ),
                "REMOTE_COMPOSE2="
                + shlex.quote(
                    f"cd {two} && env -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK "
                    "-u HEADLESS COMPOSE_DISABLE_ENV_FILE=1"
                ),
                'compose_base() { (cd "$SCRIPT_DIR" && docker compose -p "$PROJECT_NAME" '
                '--env-file "$COMPOSE_ENV_FILE" -f "$COMPOSE_FILE" "${@:3}"); }',
                *exports,
                _launcher_region(ADMISSION_BEGIN, ADMISSION_END),
                _launcher_region(PARITY_BEGIN, PARITY_END),
                _launcher_region(FORWARD_BEGIN, FORWARD_END),
                _launcher_region(SYNC_BEGIN, SYNC_END),
                _launcher_region(WORKER2_SYNC_BEGIN, WORKER2_SYNC_END),
                _launcher_region(PREFLIGHT_BEGIN, PREFLIGHT_END),
                _launcher_region(PREFLIGHT_END, BOOT_END),
            ]
        )
        path = f"{self.bin}:/usr/bin:/bin"
        env = {
            "PATH": path,
            "SANDBOX_PATH": path,
            "HOME": os.environ.get("HOME", "/tmp"),
            "SSH_RECORD": str(self.records["ssh"]),
            "SCP_RECORD": str(self.records["scp"]),
            "DOCKER_RECORD": str(self.records["docker"]),
            "WORKER_ROOT_ONE": str(self.worker_roots["one"]),
            "WORKER_ROOT_TWO": str(self.worker_roots["two"]),
        }
        return subprocess.run(
            ["bash", "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    @property
    def ssh_lines(self):
        return self.records["ssh"].read_text().splitlines()

    @property
    def scp_lines(self):
        return self.records["scp"].read_text().splitlines()

    @property
    def runs(self):
        return [
            json.loads(line)
            for line in self.records["docker"].read_text().splitlines()
        ]

    def target(self, rank):
        directory = self.head if rank == "head" else self.worker_roots[rank]
        return directory / ".container" / "detokenizer.py"

    def canonical(self, rank):
        return self.worker_roots[rank] / "patches" / HOTFIX_NAME

    def patcher_copy(self, rank):
        directory = self.head if rank == "head" else self.worker_roots[rank]
        return directory / "patches" / HOTFIX_NAME


class WorkerConsumerTest(unittest.TestCase):
    """Consumer guard: the generated remote commands run the real patcher.

    scripts/test-python-hotfix-failclosed.py executes the real Compose command
    line on the head; nothing there runs the remote commands the launcher
    generates. Each case below assembles the launcher's executable slices
    (patcher admission, resolved controls, remote Compose wrappers, worker sync
    statements and the real pre-flight/boot call sites) in a temporary sandbox
    whose ssh, scp and docker are recording, inert local transports. The ssh
    stub runs the generated command with a login-like environment (no inherited
    DSPARK_* ambient), docker resolves the Compose-mounted
    /opt/hotfix-dsv4-loop-breaker.py to the file the rank holds and executes it,
    and every rank's target is a synthetic detokenizer carrying the production
    anchors. The cases assert what the patcher did - applied, skipped, disabled
    or fail-closed, the knob values it resolved and the bytes it consumed - not
    the launcher's argv or source text.
    """

    ENABLED = {
        "DSPARK_LOOP_BREAKER": "1",
        "DSPARK_LOOP_BREAKER_REPEATS": "9",
        "DSPARK_LOOP_BREAKER_SHORT_REPEATS": "2",
        "DSPARK_LOOP_BREAKER_MIN_TOKENS": "0",
    }
    RANKS = ("one", "two", "head")

    def setUp(self):
        self.mod = _load()

    def _sandbox(self, ambient, override="selected"):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        return _ConsumerSandbox(root, self.mod, ambient, override)

    def _run_for(self, sandbox, kind, rank):
        matches = [
            run
            for run in sandbox.runs
            if run["kind"] == kind and Path(run["cwd"]).name == rank
        ]
        self.assertEqual(len(matches), 1, sandbox.runs)
        return matches[0]

    def _assert_targets(self, sandbox, state):
        for rank in self.RANKS:
            with self.subTest(rank=rank):
                if state == "stock":
                    self.assertEqual(
                        sandbox.target(rank).read_bytes(), sandbox.stock, rank
                    )
                else:
                    text = sandbox.target(rank).read_text()
                    self.assertEqual(self.mod.classify(text), state, rank)

    def test_enabled_boot_consumes_the_selected_override_over_the_stale_env_file(self):
        sandbox = self._sandbox(self.ENABLED)
        self.assertEqual(sandbox.proc.returncode, 0, sandbox.proc.stderr)
        self.assertEqual(
            [(run["kind"], Path(run["cwd"]).name) for run in sandbox.runs],
            [
                ("check", "one"),
                ("check", "two"),
                ("check", "head"),
                ("boot", "one"),
                ("boot", "two"),
                ("boot", "head"),
            ],
        )
        # The worker .env.dspark says 0/99/99/999999 and an absolute stale patch
        # path; only the forwarded controls can produce this run and these knobs.
        for run in sandbox.runs:
            self.assertEqual(run["sha256"], sandbox.sha["selected"], run)
            self.assertEqual(run["rc"], 0, run)
        for rank in ("one", "two"):
            check = self._run_for(sandbox, "check", rank)
            self.assertEqual(check["patcher"], str(sandbox.canonical(rank)), check)
            self.assertIn("READY", check["stdout"])
            self.assertIn("repeats=9 short_repeats=2 min_tokens=0", check["stdout"])
            boot = self._run_for(sandbox, "boot", rank)
            self.assertIn("[loop-breaker] applied:", boot["stdout"])
            self.assertEqual(sandbox.canonical(rank).read_bytes(), sandbox.selected)
        head_check = self._run_for(sandbox, "check", "head")
        self.assertEqual(head_check["patcher"], str(sandbox.selected_path))
        self.assertIn("repeats=9 short_repeats=2 min_tokens=0", head_check["stdout"])
        head_boot = self._run_for(sandbox, "boot", "head")
        self.assertIn("[loop-breaker] applied:", head_boot["stdout"])
        self._assert_targets(sandbox, "applied")

    def test_skip_flag_leaves_every_rank_and_the_preflight_inert(self):
        sandbox = self._sandbox(
            {**self.ENABLED, "DSPARK_SKIP_LOOP_BREAKER_HOTFIX": "1"}
        )
        self.assertEqual(sandbox.proc.returncode, 0, sandbox.proc.stderr)
        # The launcher's own guard skips the pre-flight entirely.
        self.assertEqual([run["kind"] for run in sandbox.runs], ["boot"] * 3)
        for run in sandbox.runs:
            self.assertEqual(run["rc"], 0, run)
            self.assertEqual(run["env"]["DSPARK_SKIP_LOOP_BREAKER_HOTFIX"], "1", run)
            self.assertIn(
                "skipped via DSPARK_SKIP_LOOP_BREAKER_HOTFIX=1", run["stdout"]
            )
        self._assert_targets(sandbox, "stock")

    def test_disabled_boot_is_inert_even_with_a_malformed_knob(self):
        sandbox = self._sandbox({"DSPARK_LOOP_BREAKER_REPEATS": "abc"})
        self.assertEqual(sandbox.proc.returncode, 0, sandbox.proc.stderr)
        self.assertEqual([run["kind"] for run in sandbox.runs], ["boot"] * 3)
        for run in sandbox.runs:
            self.assertEqual(run["rc"], 0, run)
            self.assertEqual(run["env"]["DSPARK_LOOP_BREAKER"], "0", run)
            self.assertIn("disabled", run["stdout"])
        self._assert_targets(sandbox, "stock")

    def test_malformed_knob_fails_closed_before_any_rank_starts(self):
        sandbox = self._sandbox(
            {**self.ENABLED, "DSPARK_LOOP_BREAKER_REPEATS": "abc"}
        )
        self.assertNotEqual(sandbox.proc.returncode, 0, sandbox.proc.stdout)
        self.assertEqual([run["kind"] for run in sandbox.runs], ["check"])
        run = sandbox.runs[0]
        self.assertEqual(Path(run["cwd"]).name, "one")
        self.assertEqual(run["rc"], 1, run)
        self.assertIn("DSPARK_LOOP_BREAKER_REPEATS", run["stderr"])
        self.assertIn("'abc'", run["stderr"])
        self._assert_targets(sandbox, "stock")

    def test_missing_or_non_regular_selected_patcher_is_refused_before_any_host_touch(self):
        for scenario in ("missing", "directory", "symlink"):
            with self.subTest(scenario=scenario):
                sandbox = self._sandbox(self.ENABLED, override=scenario)
                self.assertNotEqual(sandbox.proc.returncode, 0, sandbox.proc.stdout)
                self.assertIn("missing or not a regular file", sandbox.proc.stderr)
                self.assertIn(str(sandbox.override), sandbox.proc.stderr)
                self.assertEqual(sandbox.ssh_lines, [], sandbox.ssh_lines)
                self.assertEqual(sandbox.scp_lines, [], sandbox.scp_lines)
                self.assertEqual(sandbox.runs, [])
                for rank in self.RANKS:
                    self.assertEqual(
                        sandbox.patcher_copy(rank).read_bytes(),
                        sandbox.repository,
                        rank,
                    )
                self._assert_targets(sandbox, "stock")

    def test_skipped_or_disabled_boot_needs_no_selected_patcher(self):
        cases = (
            (
                "skipped",
                {**self.ENABLED, "DSPARK_SKIP_LOOP_BREAKER_HOTFIX": "1"},
                "skipped via DSPARK_SKIP_LOOP_BREAKER_HOTFIX=1",
            ),
            ("disabled", {"DSPARK_LOOP_BREAKER_REPEATS": "abc"}, "disabled"),
        )
        for label, ambient, marker in cases:
            with self.subTest(label=label):
                sandbox = self._sandbox(ambient, override="missing")
                self.assertEqual(sandbox.proc.returncode, 0, sandbox.proc.stderr)
                self.assertEqual([run["kind"] for run in sandbox.runs], ["boot"] * 3)
                by_rank = {Path(run["cwd"]).name: run for run in sandbox.runs}
                self.assertEqual(sorted(by_rank), ["head", "one", "two"])
                # Workers mount the forwarded canonical copy and report the
                # inert state themselves; nothing was copied there because the
                # selected file never existed.
                for rank in ("one", "two"):
                    self.assertTrue(by_rank[rank]["ran"], by_rank[rank])
                    self.assertEqual(by_rank[rank]["rc"], 0, by_rank[rank])
                    self.assertIn(marker, by_rank[rank]["stdout"])
                    self.assertEqual(
                        sandbox.canonical(rank).read_bytes(), sandbox.repository
                    )
                # The head mounts the missing selection; its entrypoint gate is
                # closed, so the boot proceeds without running a patcher.
                self.assertFalse(by_rank["head"]["ran"], by_rank["head"])
                expected_enable = "1" if label == "skipped" else "0"
                self.assertEqual(
                    by_rank["head"]["env"]["DSPARK_LOOP_BREAKER"], expected_enable
                )
                # The boot only ran the commands; nothing was ever copied.
                self.assertEqual(sandbox.scp_lines, [], sandbox.scp_lines)
                self._assert_targets(sandbox, "stock")


if __name__ == "__main__":
    unittest.main()
