#!/usr/bin/env python3
"""Host-only shutdown selection checks; Docker, SSH and rm are recorders.

The actual stop script runs with a private PATH and an empty operator env.
Worker command strings are parsed by a local POSIX shell. Python re models
Docker's name-filter subset here; this does not test Docker/Compose itself.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STOP = ROOT / "stop-deepseek-v4-flash-dspark.sh"

RECORDER = r'''
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

# grep -q may close a resource query's pipe after the first result.
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

command = Path(sys.argv[0]).name
args = sys.argv[1:]
host = os.environ.get("RECORDER_HOST", "head")
entry = {"command": command, "args": args, "host": host}
selected = []
if command == "docker" and args[:1] == ["ps"]:
    filters = [args[i + 1] for i, arg in enumerate(args) if arg == "--filter"]
    names = [value[5:] for value in filters if value.startswith("name=")]
    labels = [value[6:] for value in filters if value.startswith("label=")]
    for item in json.loads(Path(os.environ["RECORDER_FIXTURE"]).read_text()):
        if (not names or any(re.search(pattern, item["name"]) for pattern in names)) and (
            not labels or any(label == "com.docker.compose.project=" + item["label"]
                              for label in labels)
        ):
            selected.append(item["id"])
    entry["selected"] = selected
with open(os.environ["RECORDER_LOG"], "a") as log:
    log.write(json.dumps(entry) + "\n")
if command == "ssh":
    while args and args[0] == "-o":
        args = args[2:]
    env = dict(os.environ, RECORDER_HOST=args[0])
    sys.exit(subprocess.run(["sh", "-c", " ".join(args[1:])], env=env).returncode)
if selected:
    print("\n".join(selected))
'''


def run_stop(source, projects, fixture):
    with tempfile.TemporaryDirectory(prefix="stop-name-filter-") as tmp:
        workdir = Path(tmp)
        bindir = workdir / "bin"
        bindir.mkdir()
        # No inherited PATH: an unstubbed Docker/SSH command cannot fall through.
        for command in ("bash", "sh", "dirname", "basename", "tr", "sed", "cat",
                        "awk", "sort", "grep", "xargs", "env"):
            executable = shutil.which(command)
            if executable is None:
                raise RuntimeError(f"required host tool missing: {command}")
            (bindir / command).symlink_to(executable)
        for command in ("docker", "ssh", "rm"):
            path = bindir / command
            path.write_text(f"#!{sys.executable}\n" + RECORDER)
            path.chmod(0o755)
        script = workdir / STOP.name
        script.write_text(source)
        fixture_path = workdir / "containers.json"
        fixture_path.write_text(json.dumps(fixture))
        log = workdir / "commands.jsonl"
        env = {
            "PATH": str(bindir), "HOME": tmp, "LC_ALL": "C", "USER": "recorder",
            "ENV_FILE": str(workdir / "absent.env"),
            "PROJECT_NAME": projects[0], "LEGACY_PROJECT_NAME": projects[-1],
            "WORKER_HOST": "worker", "WORKER2_HOST": "worker2",
            "WORKER_DIR": tmp, "WORKER2_DIR": tmp,
            "RECORDER_FIXTURE": str(fixture_path), "RECORDER_LOG": str(log),
        }
        result = subprocess.run(
            [str(bindir / "bash"), str(script)], cwd=tmp, env=env,
            capture_output=True, text=True, timeout=30,
        )
        entries = [json.loads(line) for line in log.read_text().splitlines()]
        return result, entries


def containers(project):
    own = []
    foreign = []
    for service in ("vllm-dspark", "vl-sidecar"):
        own.extend((f"{project}-{service}", f"{project}-{service}-1",
                    f"{project}-{service}-12", f"{project}_{service}_1"))
        foreign.extend((f"{project}-{service}-old", f"{project}-{service}-1-backup",
                        f"{project}-{service}-1a", f"{project}-{service}-1-2",
                        f"my-{project}-{service}-1", f"{project}-{service}2-1",
                        f"{project}-x{service}-1", f"{project}-{service}x-1"))
    return own, foreign


class StopNameFilters(unittest.TestCase):
    def check_selection(self, projects, own, foreign, labelled=False):
        fixture = []
        for name in own + foreign:
            fixture.append({"id": f"container{len(fixture)}", "name": name, "label": "foreign"})
        if labelled:
            for project in projects:
                fixture.append({"id": f"container{len(fixture)}", "name": "unrelated-service",
                                "label": project})
        result, entries = run_stop(STOP.read_text(), projects, fixture)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        expected_removed = {item["id"] for item in fixture
                            if item["name"] in own or item["label"] in projects}
        for host in ("head", "worker", "worker2"):
            with self.subTest(host=host):
                calls = [entry for entry in entries if entry["host"] == host
                         and entry["command"] == "docker"]
                removed = {arg for entry in calls if entry["args"][:2] == ["rm", "-f"]
                           for arg in entry["args"][2:]}
                self.assertEqual(removed, expected_removed)
                compose_projects = {entry["args"][entry["args"].index("-p") + 1]
                                    for entry in calls if entry["args"][:1] == ["compose"]}
                self.assertEqual(compose_projects, set(projects) if own or labelled else set())
                for entry in calls:
                    if entry["args"][:1] != ["ps"]:
                        continue
                    args = entry["args"]
                    filters = [args[i + 1] for i, arg in enumerate(args) if arg == "--filter"]
                    name_filters = [value for value in filters if value.startswith("name=")]
                    if name_filters:
                        # Check each query, not just the union: a later sweep must
                        # not hide a broken worker or head filter.
                        services = {service for service in ("vllm-dspark", "vl-sidecar")
                                    if any(service in value for value in name_filters)}
                        candidates = [{item["id"] for item in fixture if item["name"] in own
                                       and any(item["name"] in (
                                           f"{project}-{service}", f"{project}-{service}-1",
                                           f"{project}-{service}-12", f"{project}_{service}_1")
                                               for service in services)} for project in projects]
                        self.assertIn(set(entry["selected"]), candidates, entry)
                    else:
                        expected = {item["id"] for item in fixture
                                    if "label=com.docker.compose.project=" + item["label"] in filters}
                        self.assertEqual(set(entry["selected"]), expected, entry)

    def test_current_and_legacy_projects_with_label_owned_resources(self):
        projects = ("deepseek-v4-flash", "deepseek.v4.foo")
        names = [containers(project) for project in projects]
        self.check_selection(projects, [name for own, _ in names for name in own],
                             [name for _, foreign in names for name in foreign]
                             + ["deepseekXv4Xfoo-vllm-dspark-1", "deepseek-v4.foo-vl-sidecar-1"],
                             labelled=True)

    def test_punctuation_survives_actual_shell_commands(self):
        # Synthetic directory-derived projects exercise quoting, not Compose's
        # project-name validation. No shell commands are embedded in the names.
        for project, lookalike in (("ab+c[d]", "abbbbc[d]"), ("deepseek.$USER", "deepseek.recorder"),
                                   ('deepseek"quoted', "deepseekquoted"),
                                   ("deepseek'quoted", "deepseekquoted"),
                                   (r"deepseek\path", "deepseekpath")):
            with self.subTest(project=project):
                own, foreign = containers(project)
                foreign.extend((f"{lookalike}-vllm-dspark-1", f"{lookalike}-vl-sidecar-1"))
                self.check_selection((project,), own, foreign, labelled=True)

    def test_unlabelled_own_names_trigger_compose_cleanup(self):
        own, foreign = containers("deepseek-v4-flash")
        self.check_selection(("deepseek-v4-flash",), own, foreign)

    def test_foreign_names_alone_do_not_trigger_cleanup(self):
        _, foreign = containers("deepseek-v4-flash")
        self.check_selection(("deepseek-v4-flash",), [], foreign)


if __name__ == "__main__":
    unittest.main()
