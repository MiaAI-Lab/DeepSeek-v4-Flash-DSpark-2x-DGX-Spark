#!/usr/bin/env python3
"""CPU regressions for the NFS export hardening.

The DSpark NFS exporter shares the head's HuggingFace cache root (which also
holds the 0600 `huggingface-cli` token file) on a privileged, host-networked
container. Three defaults weakened that:

1. `no_root_squash` mapped client-side root to server root, so the token file
   was readable regardless of its mode. Default is now root_squash: remote
   root maps to nobody — world-readable blobs stay readable, the token does not.
2. `insecure` allowed mounts from unprivileged client ports; dropped (the
   worker's kernel NFS client in the docker volume mounts privileged).
3. `NFS_CLIENTS:-*` (entrypoint) and `${worker_ip:-*}` (nfs_clients fallback)
   wrote a WORLD export when configuration or CIDR detection failed. Both now
   fail closed with a clear message.

Tests run the shipped guard block and the shipped nfs_clients() against stub
binaries; they do not need nfsd.

The CheckNfsExportCompat class runs the shipped live-evidence script in a
temporary checkout whose docker, ssh and files/nfs-share.sh are stubs, against
a disposable /etc/exports and a disposable populated HF cache; it proves the
root_squash assertion is an exact export option (a `no_root_squash` substring
cannot pass) and that the model check reads the snapshots/<revision>/ layout
rather than a config.json at the model root.
"""
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = (ROOT / "files" / "nfs-server" / "entrypoint.sh").read_text()
SHARE = (ROOT / "files" / "nfs-share.sh").read_text()
CHECKER = ROOT / "scripts" / "check-nfs-export-compat.sh"
BASH = shutil.which("bash") or "/bin/bash"

# The guard block: variable resolution + empty-client refusal + opts default,
# stopping before the first nfsd filesystem touch.
GUARD_BLOCK = ENTRYPOINT[ENTRYPOINT.index("EXPORT_DIR="):ENTRYPOINT.index("\nmkdir ")]

_fn_start = SHARE.index("nfs_clients() {")
CLIENTS_FN = SHARE[_fn_start:SHARE.index("\n}", _fn_start) + 2]

FAKE_IP = """#!/usr/bin/env bash
if [ -n "${IP_CIDR:-}" ]; then
  echo "3: dev inet ${IP_CIDR} brd 10.0.22.255 scope global dev"
fi
exit 0
"""


def run_guard(env_over: dict) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("NFS_CLIENTS", None)
    env.pop("NFS_OPTS", None)
    env.update(env_over)
    return subprocess.run(["bash", "-c", GUARD_BLOCK], capture_output=True,
                          text=True, env=env)


def run_clients(cidr: str, worker_ip: str) -> subprocess.CompletedProcess:
    workdir = Path(tempfile.mkdtemp())
    ip = workdir / "ip"
    ip.write_text(FAKE_IP)
    ip.chmod(0o755)
    script = f"""set -euo pipefail
PATH={shlex.quote(str(workdir))}:/usr/bin:/bin
export IP_CIDR={shlex.quote(cidr)}
IFACE=enp1s0f1np1
WORKER_IP={shlex.quote(worker_ip)}
WORKER_HOST=worker.example
nfs_err() {{ echo "$*" >&2; exit 1; }}
{CLIENTS_FN}
nfs_clients
"""
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    shutil.rmtree(workdir)
    return r


class EntrypointGuard(unittest.TestCase):
    def test_empty_clients_refused(self):
        r = run_guard({})
        self.assertEqual(r.returncode, 1)
        self.assertIn("refusing to export to '*'", r.stderr)
        self.assertIn("NFS_CLIENTS", r.stderr)

    def test_explicit_clients_pass(self):
        r = run_guard({"NFS_CLIENTS": "10.0.22.2,10.0.22.0/24"})
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_default_opts_are_squashed_and_secure(self):
        for source, name in ((ENTRYPOINT, "entrypoint.sh"), (SHARE, "nfs-share.sh")):
            with self.subTest(file=name):
                for line in source.splitlines():
                    if "NFS_OPTS:-" in line:
                        self.assertIn("root_squash", line)
                        self.assertNotIn("no_root_squash", line)
                        self.assertNotIn("insecure", line)

    def test_no_wildcard_default_remains(self):
        for source, name in ((ENTRYPOINT, "entrypoint.sh"), (SHARE, "nfs-share.sh")):
            code = "\n".join(ln for ln in source.splitlines()
                             if not ln.strip().startswith("#"))
            with self.subTest(file=name):
                self.assertNotIn('NFS_CLIENTS:-*}', code)
                self.assertNotIn('${worker_ip:-*}', code)


# ── check-nfs-export-compat.sh fixtures ─────────────────────────────────────

MODEL_DIR = "models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp"

# The checker's repo-side dependencies are satisfied by recorders; the live
# properties under test live in the stubs below and the disposable fixtures.
NFS_SHARE_STUB = """\
nfs_ensure_server() { return 0; }
nfs_ensure_worker_volume() { return 0; }
"""

# docker: `image inspect` succeeds (build is skipped); `run --entrypoint
# /entrypoint.sh` replays the real fail-closed guard semantics for the -e
# NFS_CLIENTS= probe; `exec <c> sh -c '<payload>'` replays the shipped remote
# pipeline against the disposable exports file, so whatever the checker
# actually sends — including its grep -o — is what runs.
DOCKER_STUB = """\
#!/usr/bin/env bash
printf 'docker %s\\n' "$*" >> "$STUB_LOG"
case "$1" in
  image) exit 0 ;;
  run)
    clients="" want=0
    for a in "$@"; do
      if [ "$want" = 1 ]; then case "$a" in NFS_CLIENTS=*) clients="${a#NFS_CLIENTS=}" ;; esac; fi
      [ "$a" = "-e" ] && want=1 || want=0
    done
    if [ -z "$clients" ]; then
      echo "FATAL: NFS_CLIENTS is empty; refusing to export to '*'." >&2
      exit 1
    fi
    exit 0 ;;
  exec)
    for last; do :; done
    eval "${last//\\/etc\\/exports/$STUB_EXPORTS}"
    exit $? ;;
esac
exit 0
"""

# ssh: the last argument is the remote payload, always of the form
# `docker run … sh -c 'test -X /hf/<path>'`. The payload is mapped onto the
# disposable export root; a squashed client reads as nobody, so `test -r` is
# only true when the file's other-read bit is set (o+r), not when the file is
# readable by the test user.
SSH_STUB = """\
#!/usr/bin/env bash
printf 'ssh %s\\n' "$*" >> "$STUB_LOG"
for last; do :; done
payload="$last"
case "$payload" in
  *"sh -c '"*) payload="${payload##*sh -c \\'}"; payload="${payload%\\'}" ;;
esac
set -- $payload
[ "${1:-}" = "test" ] || exit 2
p="$STUB_EXPORT_ROOT${3#/hf}"
case "$2" in
  -d) [ -d "$p" ] ;;
  -r)
    [ -f "$p" ] || exit 1
    mode="$(stat -f %Lp "$p" 2>/dev/null || stat -c %a "$p" 2>/dev/null)" || exit 1
    [ $(( ${mode: -1} & 4 )) -ne 0 ] ;;
  *) exit 2 ;;
esac
"""

GOOD_EXPORTS = f"/export 10.0.22.2(ro,sync,no_subtree_check,root_squash,fsid=0)\n"


def write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


class CheckNfsExportCompat(unittest.TestCase):
    """Runs the shipped check-nfs-export-compat.sh with stub docker/ssh."""

    def setUp(self):
        self.workdir = Path(tempfile.mkdtemp(prefix="nfs-compat-"))
        self.addCleanup(shutil.rmtree, self.workdir)
        self.checkout = self.workdir / "checkout"
        (self.checkout / "scripts").mkdir(parents=True)
        (self.checkout / "files").mkdir(parents=True)
        shutil.copyfile(CHECKER, self.checkout / "scripts" / CHECKER.name)
        (self.checkout / "files" / "nfs-share.sh").write_text(NFS_SHARE_STUB)
        bindir = self.workdir / "bin"
        bindir.mkdir()
        write_executable(bindir / "docker", DOCKER_STUB)
        write_executable(bindir / "ssh", SSH_STUB)

    def hf_cache(self, revision=None, refs_main=None, root_config=False,
                 config_mode=0o644, token_mode=None) -> Path:
        hf = self.workdir / "hf-cache"
        model = hf / "hub" / MODEL_DIR
        if revision is not None:
            cfg = model / "snapshots" / revision / "config.json"
            cfg.parent.mkdir(parents=True, exist_ok=True)
            cfg.write_text("{}")
            cfg.chmod(config_mode)
        if refs_main is not None:
            (model / "refs").mkdir(parents=True, exist_ok=True)
            (model / "refs" / "main").write_text(refs_main + "\n")
        if root_config:
            model.mkdir(parents=True, exist_ok=True)
            (model / "config.json").write_text("{}")
        if token_mode is not None:
            tok = hf / "token"
            tok.write_text("hf_secret")
            tok.chmod(token_mode)
        return hf

    def run_checker(self, hf: Path, exports_text: str, extra_env=None):
        exports = self.workdir / "exports.txt"
        exports.write_text(exports_text)
        env = {
            "PATH": f"{self.workdir / 'bin'}:/usr/bin:/bin",
            "HOME": str(self.workdir),
            "LC_ALL": "C",
            "ENV_FILE": str(self.workdir / "absent-env"),
            "WORKER_HOST": "worker.example",
            "NFS_SERVER_IP": "10.0.22.1",
            "HF_CACHE": str(hf),
            "DSPARK_MODEL": "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
            "STUB_LOG": str(self.workdir / "stub.log"),
            "STUB_EXPORTS": str(exports),
            "STUB_EXPORT_ROOT": str(hf),
        }
        env.update(extra_env or {})
        result = subprocess.run(
            [BASH, str(self.checkout / "scripts" / CHECKER.name)],
            env=env, capture_output=True, text=True, timeout=60)
        log = (self.workdir / "stub.log").read_text()
        return result, result.stdout + result.stderr, log

    def assert_no_loosening_advice(self, out: str):
        self.assertNotIn("without root_squash", out)
        self.assertNotIn("no_root_squash\"", out)
        self.assertNotIn("chmod a+r", out)

    def test_happy_path(self):
        hf = self.hf_cache(revision="abc123", token_mode=0o600)
        r, out, log = self.run_checker(hf, GOOD_EXPORTS, {"DSPARK_REVISION": "abc123"})
        self.assertEqual(r.returncode, 0, out)
        self.assertIn("exact root_squash", out)
        self.assertIn("snapshots/abc123/config.json", out)
        self.assertIn("/hf/hub/" + MODEL_DIR + "/snapshots/abc123/config.json", log)
        self.assertNotIn("/hf/hub/" + MODEL_DIR + "/config.json", log)
        self.assertIn("NOT readable", out)
        self.assert_no_loosening_advice(out)

    def test_no_root_squash_cannot_pass(self):
        hf = self.hf_cache(revision="abc123")
        for opts in ("rw,sync,no_root_squash,fsid=0",
                     "rw,sync,no_subtree_check,fsid=0"):
            with self.subTest(opts=opts):
                r, out, _ = self.run_checker(
                    hf, f"/export 10.0.22.2({opts})\n", {"DSPARK_REVISION": "abc123"})
                self.assertEqual(r.returncode, 1, out)
                self.assertIn("exact root_squash", out)
                self.assertNotIn("PASS: all", out)

    def test_any_unsquashed_group_fails(self):
        hf = self.hf_cache(revision="abc123")
        exports = ("/export 10.0.22.2(ro,sync,root_squash,fsid=0)\n"
                   "/export 10.0.22.0/24(ro,sync,no_root_squash,fsid=0)\n")
        r, out, _ = self.run_checker(hf, exports, {"DSPARK_REVISION": "abc123"})
        self.assertEqual(r.returncode, 1, out)
        self.assertIn("exact root_squash", out)

    def test_revision_from_refs_main(self):
        hf = self.hf_cache(revision="def456", refs_main="def456")
        r, out, log = self.run_checker(hf, GOOD_EXPORTS, {"DSPARK_REVISION": ""})
        self.assertEqual(r.returncode, 0, out)
        self.assertIn("snapshots/def456/config.json", out)
        self.assertIn("/hf/hub/" + MODEL_DIR + "/snapshots/def456/config.json", log)

    def test_revision_from_lone_snapshot(self):
        hf = self.hf_cache(revision="aaa999")
        r, out, log = self.run_checker(hf, GOOD_EXPORTS, {"DSPARK_REVISION": ""})
        self.assertEqual(r.returncode, 0, out)
        self.assertIn("snapshots/aaa999/config.json", out)

    def test_root_level_config_does_not_satisfy(self):
        hf = self.hf_cache(root_config=True)
        r, out, log = self.run_checker(hf, GOOD_EXPORTS, {"DSPARK_REVISION": ""})
        self.assertEqual(r.returncode, 1, out)
        self.assertIn("snapshots", out)
        self.assertIn("not a populated HF hub snapshot", out)
        self.assertNotIn("/hf/hub/" + MODEL_DIR + "/config.json", log)
        self.assert_no_loosening_advice(out)

    def test_pinned_revision_absent_reports_head_gap(self):
        hf = self.hf_cache(revision="abc123")
        r, out, _ = self.run_checker(hf, GOOD_EXPORTS, {"DSPARK_REVISION": "missing9"})
        self.assertEqual(r.returncode, 1, out)
        self.assertIn("absent from the head cache", out)
        self.assertIn("snapshots/missing9", out)
        self.assert_no_loosening_advice(out)

    def test_config_not_other_readable_reports_export_gap(self):
        hf = self.hf_cache(revision="abc123", config_mode=0o600)
        r, out, _ = self.run_checker(hf, GOOD_EXPORTS, {"DSPARK_REVISION": "abc123"})
        self.assertEqual(r.returncode, 1, out)
        self.assertIn("NOT readable through the export", out)
        self.assertIn("snapshots/abc123/config.json", out)
        self.assert_no_loosening_advice(out)

    def test_world_readable_token_fails(self):
        hf = self.hf_cache(revision="abc123", token_mode=0o644)
        r, out, _ = self.run_checker(hf, GOOD_EXPORTS, {"DSPARK_REVISION": "abc123"})
        self.assertEqual(r.returncode, 1, out)
        self.assertIn("world-readable", out)


class NfsClients(unittest.TestCase):
    def test_cidr_and_worker_ip(self):
        r = run_clients("10.0.22.1/24", "10.0.22.2")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "10.0.22.2,10.0.22.0/24")

    def test_cidr_only(self):
        r = run_clients("10.0.22.1/24", "")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "10.0.22.0/24")

    def test_worker_ip_only(self):
        r = run_clients("", "10.0.22.2")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "10.0.22.2")

    def test_nothing_detected_fails_closed(self):
        r = run_clients("", "")
        self.assertEqual(r.returncode, 1)
        self.assertIn("Refusing to export to '*'", r.stderr)
        self.assertNotIn("*", r.stdout)


if __name__ == "__main__":
    unittest.main()
