#!/usr/bin/env python3
"""Exercise the shipped GB10 install block without installing or serving.

The source check needs only stdlib and Bash. The rendered check additionally
uses Docker Compose's daemon-free config command, or reports an explicit skip.
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

PYTHON_STUB = """#!/bin/bash
printf 'install\\n' >> "$INVOCATIONS"
exit "$INSTALL_STATUS"
"""
SERVE_STUB = """#!/bin/bash
printf 'serve\\n' >> "$INVOCATIONS"
"""


def plugin_block(command):
    match = re.search(
        r'if \[ [^\n;]*ENABLE_VLLM_GB10_PATCH[^\n;]*; then\s.*?\bfi;',
        command,
        re.DOTALL,
    )
    if match is None:
        raise AssertionError("GB10 optional boot block not found")
    # Compose escapes container-side dollars; no host substitutions occur in
    # this block. The existing auth handoff test uses the same decoding.
    return match.group().replace("$$", "$")


class GB10InstallTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = COMPOSE.read_text(encoding="utf-8")
        # YAML's folded scalar preserves newlines around more-indented lines.
        # Require that narrow layout rather than invent a general YAML folder.
        match = re.search(
            r'^        if \[ [^\n]*ENABLE_VLLM_GB10_PATCH[^\n]*\n'
            r'(?:          [^\n]*\n)+        fi;',
            source,
            re.MULTILINE,
        )
        if match is None:
            raise AssertionError("GB10 folded-scalar layout changed; review extraction")
        cls.block = plugin_block(match.group())

    def run_block(self, enabled, install_status):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name, body in (("python3", PYTHON_STUB), ("serve-recorder", SERVE_STUB)):
                stub = root / name
                stub.write_text(body)
                stub.chmod(0o755)
            invocations = root / "invocations"
            invocations.touch()
            env = {
                "PATH": str(root),
                "INVOCATIONS": str(invocations),
                "INSTALL_STATUS": str(install_status),
            }
            if enabled is not None:
                env["ENABLE_VLLM_GB10_PATCH"] = enabled
            # Do not add errexit: the shipped bash -lc relies on explicit guards.
            proc = subprocess.run(
                ["/bin/bash", "-c", self.block + "\nexec serve-recorder\n"],
                env=env, cwd=td, capture_output=True, text=True, timeout=30,
            )
            return proc, invocations.read_text().splitlines()

    def test_install_success_continues(self):
        proc, events = self.run_block("1", 0)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(events, ["install", "serve"])

    def test_install_failure_stops_before_serving(self):
        proc, events = self.run_block("1", 7)
        self.assertNotEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(events, ["install"])

    def test_disabled_path_skips_install_and_continues(self):
        for enabled in (None, "0"):
            with self.subTest(enabled=enabled):
                proc, events = self.run_block(enabled, 7)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(events, ["serve"])


class RenderedGB10InstallTest(GB10InstallTest):
    @classmethod
    def setUpClass(cls):
        docker = shutil.which("docker")
        env = {"PATH": os.defpath}
        if docker is None or subprocess.run(
            [docker, "compose", "version"], env=env,
            capture_output=True, text=True, timeout=30,
        ).returncode != 0:
            raise unittest.SkipTest("Docker Compose CLI unavailable; rendered boot not checked")
        with tempfile.TemporaryDirectory() as td:
            compose = Path(td) / COMPOSE.name
            compose.write_text(COMPOSE.read_text(encoding="utf-8"))
            env_file = Path(td) / "stub.env"
            env_file.touch()
            proc = subprocess.run(
                [docker, "compose", "--env-file", str(env_file), "-f", str(compose),
                 "config", "--format", "json"],
                env=env, cwd=td, capture_output=True, text=True, timeout=30,
            )
        if proc.returncode != 0:
            raise AssertionError(f"Compose config failed: {proc.stderr}")
        command = json.loads(proc.stdout)["services"]["vllm-dspark"]["command"]
        cls.block = plugin_block(command[2])


if __name__ == "__main__":
    unittest.main()
