#!/usr/bin/env python3
"""CPU regressions for the open-API and env-permission warnings in start.

Two warning-only additions to the launcher (no behavior change):

1. Wildcard bind (0.0.0.0/::) with no API key configured prints a loud
   UNAUTHENTICATED warning naming the keyless routes and the remedies. The
   default example ships this way for bring-up; a quiet operator should not
   discover the open endpoint from a scanner.
2. .env.dspark is only ever sourced, so nothing enforced its mode: with
   secrets in it (VLLM_API_KEY / DSPARK_API_KEYS / HF_TOKEN) and a
   group/other-readable mode, the launcher now warns with chmod guidance.

Both blocks are extracted from the shipped launcher and run against a fake
`stat` (deterministic modes on any platform).
"""
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "start-deepseek-v4-flash-dspark.sh"
SOURCE = LAUNCHER.read_text()


def extract(begin: str, end: str) -> str:
    i = SOURCE.index(begin)
    return SOURCE[i:SOURCE.index(end, i) + len(end)]


OPEN_API_BLOCK = extract("# Open-API warning (begin)", "# Open-API warning (end)")
PERMS_BLOCK = extract("# Secret-file permission check (begin)",
                      "# Secret-file permission check (end)")

FAKE_STAT = """#!/usr/bin/env bash
printf '%s\\n' "${FAKE_MODE:-600}"
exit 0
"""


def run_block(block: str, env: dict, fake_mode: str | None = None):
    workdir = Path(tempfile.mkdtemp())
    bindir = workdir / "bin"
    bindir.mkdir()
    if fake_mode is not None:
        f = bindir / "stat"
        f.write_text(FAKE_STAT)
        f.chmod(0o755)
    env_file = workdir / ".env.dspark"
    env_file.write_text("VLLM_API_KEY=sk-x\n")
    lines = [f"PATH={shlex.quote(str(bindir))}:/usr/bin:/bin"]
    if fake_mode is not None:
        lines.append(f"export FAKE_MODE={shlex.quote(fake_mode)}")
    lines.append(f"ENV_FILE={shlex.quote(str(env_file))}")
    for k, v in env.items():
        lines.append(f"{k}={shlex.quote(v)}")
    lines.append(block)
    r = subprocess.run(["bash", "-c", "\n".join(lines)],
                       capture_output=True, text=True)
    shutil.rmtree(workdir)
    return r


class OpenApiWarning(unittest.TestCase):
    def run_open(self, host: str, keys: str = "", api_key: str = ""):
        return run_block(OPEN_API_BLOCK, {
            "VLLM_HOST": host, "VLLM_PORT": "8888",
            "_dspark_keys_set": keys, "VLLM_API_KEY": api_key,
        })

    def test_wildcard_no_key_warns(self):
        r = self.run_open("0.0.0.0")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("UNAUTHENTICATED API on 0.0.0.0:8888", r.stderr)
        self.assertIn("/invocations", r.stderr)  # keyless routes named

    def test_wildcard_ipv6_warns(self):
        r = self.run_open("::")
        self.assertIn("UNAUTHENTICATED", r.stderr)

    def test_wildcard_with_key_is_quiet(self):
        r = self.run_open("0.0.0.0", api_key="sk-x")
        self.assertNotIn("UNAUTHENTICATED", r.stderr)

    def test_wildcard_with_multikey_is_quiet(self):
        r = self.run_open("0.0.0.0", keys="1")
        self.assertNotIn("UNAUTHENTICATED", r.stderr)

    def test_loopback_is_quiet(self):
        r = self.run_open("127.0.0.1")
        self.assertNotIn("UNAUTHENTICATED", r.stderr)

    def test_lan_ip_is_quiet(self):
        # Deliberate scope: the warning targets wildcard binds; a LAN-IP bind
        # without a key is the operator's explicit choice of exposure.
        r = self.run_open("10.0.0.1")
        self.assertNotIn("UNAUTHENTICATED", r.stderr)


class EnvPermsWarning(unittest.TestCase):
    def run_perms(self, mode: str, api_key: str = "sk-x", keys: str = "",
                  hf_token: str = ""):
        return run_block(PERMS_BLOCK, {
            "VLLM_API_KEY": api_key, "_dspark_keys_set": keys,
            "HF_TOKEN": hf_token,
        }, fake_mode=mode)

    def test_600_with_secret_is_quiet(self):
        r = self.run_perms("600")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("chmod 600", r.stderr)

    def test_644_with_secret_warns(self):
        r = self.run_perms("644")
        self.assertIn("mode 644", r.stderr)
        self.assertIn("chmod 600", r.stderr)

    def test_640_with_secret_warns(self):
        r = self.run_perms("640")
        self.assertIn("mode 640", r.stderr)

    def test_644_without_secret_is_quiet(self):
        r = self.run_perms("644", api_key="")
        self.assertNotIn("chmod 600", r.stderr)

    def test_777_with_hf_token_warns(self):
        r = self.run_perms("777", api_key="", hf_token="hf_x")
        self.assertIn("chmod 600", r.stderr)

    def test_unreadable_mode_is_quiet(self):
        # stat failure / non-numeric mode must never break a launch.
        r = self.run_perms("")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("chmod 600", r.stderr)


class SourceShape(unittest.TestCase):
    def test_blocks_present_and_ordered(self):
        auth_end = SOURCE.index("# DSPARK_API_KEYS auth (end)")
        open_begin = SOURCE.index("# Open-API warning (begin)")
        perms_begin = SOURCE.index("# Secret-file permission check (begin)")
        self.assertLess(auth_end, open_begin)
        self.assertLess(open_begin, perms_begin)

    def test_warnings_never_fail_the_launch(self):
        for block in (OPEN_API_BLOCK, PERMS_BLOCK):
            self.assertNotIn("exit 1", block)
            self.assertNotIn("exit 2", block)


if __name__ == "__main__":
    unittest.main()
