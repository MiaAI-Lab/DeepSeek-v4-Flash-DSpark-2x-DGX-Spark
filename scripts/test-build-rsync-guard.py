#!/usr/bin/env python3
"""Host-only build guard regressions; no real SSH, Docker, or rsync is run.

Run the shipped build script in a temporary checkout. SSH's command string is
interpreted by a local shell without reconstructing or unescaping source text.
The rsync recorder stops the build at permission to synchronize, without copying
or deleting anything. A compose filename recognizes a recipe, not its ownership.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build-dspark-vllm-runtime.sh"

RECORDER = """import json
import os
import subprocess
import sys
from pathlib import Path

command = Path(sys.argv[0]).name
with open(os.environ['COMMAND_LOG'], 'a') as log:
    log.write(json.dumps({'command': command, 'args': sys.argv[1:]}) + '\\n')
if command == 'ssh':
    if sys.argv[1] != 'fixture-worker':
        sys.exit('unexpected SSH destination')
    if os.environ.get('FAIL_SSH'):
        sys.exit(255)
    sys.exit(subprocess.run(['/bin/sh', '-c', ' '.join(sys.argv[2:])]).returncode)
if command == 'rsync':
    # Stop before the remote build as well as before any synchronization.
    sys.exit(73)
if command in ('find', 'ls'):
    if os.environ.get('FAIL_INSPECTION'):
        sys.stdout.write(os.environ.get('INSPECTION_OUTPUT', ''))
        sys.exit(1)
    real_command = os.environ['REAL_' + command.upper()]
    os.execv(real_command, [real_command, *sys.argv[1:]])
if command != 'docker':
    sys.exit('unexpected recorder command')
"""


def write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


class GuardBehavior(unittest.TestCase):
    def setUp(self):
        self.workdir = Path(tempfile.mkdtemp(prefix="build guard "))
        self.addCleanup(shutil.rmtree, self.workdir)
        self.checkout = self.workdir / "worker checkout"
        self.repo = self.workdir / "local checkout"
        (self.repo / "scripts").mkdir(parents=True)
        shutil.copyfile(BUILD, self.repo / BUILD.name)
        write_executable(self.repo / "scripts/verify-overlay-sources.sh",
                         "#!/bin/sh\nexit 0\n")
        bindir = self.workdir / "bin"
        bindir.mkdir()
        for command in ("ssh", "rsync", "docker", "find", "ls"):
            write_executable(bindir / command, f"#!{sys.executable}\n{RECORDER}")
        self.log = self.workdir / "commands.jsonl"
        self.env = {
            "PATH": f"{bindir}:/usr/bin:/bin",
            "HOME": str(self.workdir),
            "LC_ALL": "C",
            "ENV_FILE": str(self.workdir / "absent.env"),
            "WORKER_BUILD": "1",
            "WORKER_HOST": "fixture-worker",
            "COMMAND_LOG": str(self.log),
            "REAL_FIND": shutil.which("find") or "/usr/bin/find",
            "REAL_LS": shutil.which("ls") or "/usr/bin/ls",
        }

    def run_build(self):
        self.env["WORKER_CHECKOUT"] = str(self.checkout)
        result = subprocess.run(
            ["bash", str(self.repo / BUILD.name)], env=self.env,
            cwd=self.workdir, capture_output=True, text=True,
        )
        commands = [json.loads(line) for line in self.log.read_text().splitlines()]
        return result, commands

    def assert_allowed(self):
        result, commands = self.run_build()
        self.assertEqual(result.returncode, 73, result.stderr)
        syncs = [event for event in commands if event["command"] == "rsync"]
        self.assertEqual(len(syncs), 1)
        self.assertEqual(syncs[0]["args"][-2:],
                         [f"{self.repo}/", f"fixture-worker:{self.checkout}/"])
        self.assertTrue(self.checkout.is_dir())

    def assert_refused(self):
        result, commands = self.run_build()
        self.assertNotEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any(event["command"] == "rsync" for event in commands))
        return result

    def foreign_file(self, name="important-data.txt"):
        self.checkout.mkdir()
        data = self.checkout / name
        data.write_bytes(b"retain this foreign data\n")
        return data

    def test_missing_directory_allowed(self):
        self.assert_allowed()

    def test_empty_directory_allowed(self):
        self.checkout.mkdir()
        self.assert_allowed()

    def test_recipe_filename_allows_even_unrelated_data(self):
        data = self.foreign_file()
        (self.checkout / "docker-compose.dspark.yml").write_text("services: {}\n")
        self.assert_allowed()
        # The recorder preserves data; a real --delete can remove this file.
        self.assertEqual(data.read_bytes(), b"retain this foreign data\n")

    def test_foreign_directory_refused(self):
        data = self.foreign_file()
        self.assert_refused()
        self.assertEqual(data.read_bytes(), b"retain this foreign data\n")

    def test_hidden_foreign_file_refused(self):
        data = self.foreign_file(".hidden")
        self.assert_refused()
        self.assertEqual(data.read_bytes(), b"retain this foreign data\n")

    def test_newline_filename_refused(self):
        data = self.foreign_file("\n")
        self.assert_refused()
        self.assertEqual(data.read_bytes(), b"retain this foreign data\n")

    def test_apostrophe_path_allowed(self):
        self.checkout = self.workdir / "worker's checkout"
        self.assert_allowed()

    def test_apostrophe_foreign_path_refused(self):
        self.checkout = self.workdir / "worker's checkout"
        data = self.foreign_file()
        self.assert_refused()
        self.assertEqual(data.read_bytes(), b"retain this foreign data\n")

    def test_non_directory_refused(self):
        self.checkout.write_bytes(b"not a directory\n")
        self.assert_refused()
        self.assertEqual(self.checkout.read_bytes(), b"not a directory\n")

    def test_unreadable_directory_refused(self):
        if os.geteuid() == 0:
            self.skipTest("root bypasses directory read permissions")
        data = self.foreign_file()
        self.checkout.chmod(0o300)
        try:
            self.assert_refused()
        finally:
            self.checkout.chmod(0o700)
        self.assertEqual(data.read_bytes(), b"retain this foreign data\n")

    def test_unsearchable_directory_refused(self):
        if os.geteuid() == 0:
            self.skipTest("root bypasses directory search permissions")
        data = self.foreign_file()
        self.checkout.chmod(0o600)
        try:
            self.assert_refused()
        finally:
            self.checkout.chmod(0o700)
        self.assertEqual(data.read_bytes(), b"retain this foreign data\n")

    def test_failed_inspection_refused(self):
        data = self.foreign_file()
        self.env["FAIL_INSPECTION"] = "1"
        self.assert_refused()
        self.assertEqual(data.read_bytes(), b"retain this foreign data\n")

    def test_partial_failed_inspection_refuses_recognized_directory(self):
        data = self.foreign_file()
        (self.checkout / "docker-compose.dspark.yml").write_text("services: {}\n")
        self.env.update(FAIL_INSPECTION="1", INSPECTION_OUTPUT="x")
        self.assert_refused()
        self.assertEqual(data.read_bytes(), b"retain this foreign data\n")

    def test_ssh_failure_refused(self):
        data = self.foreign_file()
        self.env["FAIL_SSH"] = "1"
        self.assert_refused()
        self.assertEqual(data.read_bytes(), b"retain this foreign data\n")


if __name__ == "__main__":
    unittest.main()
