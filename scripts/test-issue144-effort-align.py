#!/usr/bin/env python3
"""CPU tests for patches/hotfix-dsv4-issue144-effort-align.py (no vLLM, no GPU, no network).

Covers both legitimate pre-states of the co-owned encoder module:
- the pristine 48095b345 checkpoint snapshot, and
- the live entrypoint chain product (snapshot + issue21 + vision-exp +
  assistant-final).

The Blocks class simulates vLLM-v1 prefix caching (256-token blocks, each
block hash chained on the parent hash) over a deterministic special-token
aware byte tokenizer, proving the issue #144 acceptance property: with the
fix, cross-effort-bucket shared-prefix blocks hash identically; stock shares
zero. Set DSPARK_I144_TOKENIZER_JSON to a HuggingFace tokenizer.json to run
the same proof with the real BPE tokenizer (skipped otherwise; the BPE
junction merge may cost at most one token at the divergence point).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXDIR = ROOT / "scripts" / "fixtures" / "issue144-effort-align"
FIX_SNAPSHOT = FIXDIR / "encoding_dsv4-48095b345-snapshot.py"
FIX_LIVE = FIXDIR / "deepseek_v4_encoding-48095b345-live-chain.py"
PATCHER = ROOT / "patches" / "hotfix-dsv4-issue144-effort-align.py"
COMPOSE = ROOT / "docker-compose.dspark.yml"
START = ROOT / "start-deepseek-v4-flash-dspark.sh"
ENV_EXAMPLE = ROOT / ".env.dspark.example"
CI = ROOT / "scripts" / "ci-validate.sh"

BLOCK = 256


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


HF = _load(PATCHER, "dspark_issue144")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]
SYS = (
    "You are a precise coding agent for the dspark lane.\n"
    "Follow the workspace rules.\n- keep diffs minimal\n- never fabricate output"
)
BIG_SYS = SYS + "\n\n## Workspace rules\n" + "\n".join(
    f"- rule {i}: keep operation {i} deterministic, fail closed on drift, and "
    f"log the outcome of step {i} before advancing." for i in range(160)
)


def _module_pair(fixture: Path, tmpdir: str, tag: str):
    stock_src = fixture.read_bytes()
    patched_src = HF.transform(stock_src)
    stock = HF._load_module(stock_src, f"enc_stock_{tag}", tmpdir)
    patched = HF._load_module(patched_src, f"enc_patched_{tag}", tmpdir)
    return stock, patched


def _render(mod, msgs, effort, mode="thinking", **kw):
    return mod.encode_messages([dict(m) for m in msgs], mode, reasoning_effort=effort, **kw)


class Transform(unittest.TestCase):
    def test_fixtures_match_identity_pins(self):
        snap = FIX_SNAPSHOT.read_bytes()
        live = FIX_LIVE.read_bytes()
        self.assertEqual(hashlib.sha256(snap).hexdigest(), HF.SNAPSHOT_SHA256)
        self.assertEqual(len(snap), 36707)
        self.assertEqual(hashlib.sha256(live).hexdigest(), HF.LIVE_CHAIN_SHA256)
        self.assertEqual(len(live), 39960)

    def test_region_constants_match_self_pins(self):
        HF._verify_self()
        self.assertEqual(
            hashlib.sha256(HF.REGION_OLD.encode()).hexdigest(), HF.REGION_OLD_SHA256
        )
        self.assertEqual(
            hashlib.sha256(HF.REGION_NEW.encode()).hexdigest(), HF.REGION_NEW_SHA256
        )

    def test_transform_is_pinned_single_site_and_compiles(self):
        for fixture, patched_sha in (
            (FIX_SNAPSHOT, HF.PATCHED_SNAPSHOT_SHA256),
            (FIX_LIVE, HF.PATCHED_LIVE_CHAIN_SHA256),
        ):
            stock = fixture.read_bytes()
            self.assertEqual(stock.count(HF.REGION_OLD.encode()), 1)
            patched = HF.transform(stock)
            self.assertEqual(hashlib.sha256(patched).hexdigest(), patched_sha)
            self.assertEqual(patched, stock.replace(HF.REGION_OLD.encode(), HF.REGION_NEW.encode(), 1))
            self.assertEqual(patched.count(HF.MARK.encode()), 1)
            compile(patched, "deepseek_v4_encoding.py", "exec")

    def test_transform_refuses_foreign_or_patched_bytes(self):
        with self.assertRaises(HF.HotfixError):
            HF.transform(b"def nothing():\n    pass\n")
        patched = HF.transform(FIX_SNAPSHOT.read_bytes())
        with self.assertRaises(HF.HotfixError):
            HF.transform(patched)

    def test_self_check_accepts_both_patched_pre_states(self):
        for fixture in (FIX_SNAPSHOT, FIX_LIVE):
            ok, why = HF._self_check(HF.transform(fixture.read_bytes()))
            self.assertTrue(ok, why)

    def test_self_check_rejects_a_broken_relocation(self):
        # Sabotage the relocation: drop the tail append from the patched bytes.
        patched = HF.transform(FIX_SNAPSHOT.read_bytes())
        sabotaged = patched.replace(
            b'        if _effort_directive_tail:\n'
            b'            prompt += ("\\n\\n" if prompt else "") + _effort_directive_tail\n',
            b"",
            1,
        )
        self.assertNotEqual(sabotaged, patched)
        ok, why = HF._self_check(sabotaged)
        self.assertFalse(ok)
        self.assertTrue(why)


class Rendering(unittest.TestCase):
    """Byte-level rendering contract on both pre-states."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="i144-render-")
        cls.pairs = {
            "snapshot": _module_pair(FIX_SNAPSHOT, cls.tmp, "snap"),
            "live": _module_pair(FIX_LIVE, cls.tmp, "live"),
        }

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _fixtures(self):
        return {
            "sys+user": [{"role": "system", "content": SYS}, {"role": "user", "content": "hello"}],
            "toolsys+sys+user": [
                {"role": "system", "tools": TOOLS},
                {"role": "system", "content": SYS},
                {"role": "user", "content": "run it"},
            ],
            "sys(tools)+user": [
                {"role": "system", "content": SYS, "tools": TOOLS},
                {"role": "user", "content": "run it"},
            ],
            "user-only": [{"role": "user", "content": "no system here"}],
        }

    def test_low_default_and_chat_renders_are_byte_identical(self):
        for tag, (stock, patched) in self.pairs.items():
            for name, msgs in self._fixtures().items():
                for effort, mode in ((None, "thinking"), ("low", "thinking"), ("high", "chat")):
                    self.assertEqual(
                        _render(stock, msgs, effort, mode),
                        _render(patched, msgs, effort, mode),
                        f"{tag}/{name}/effort={effort}/mode={mode}",
                    )

    def test_high_and_max_relocate_directive_to_system_region_tail(self):
        for tag, (stock, patched) in self.pairs.items():
            bos = stock.bos_token
            prompts = stock.REASONING_EFFORT_PROMPTS
            for name, msgs in self._fixtures().items():
                k = 0
                while k < len(msgs) and msgs[k].get("role") == "system":
                    k += 1
                low = _render(stock, msgs, "low")
                sysregion = "".join(
                    stock.render_message(i, [dict(m) for m in msgs], "thinking", True, "low")
                    for i in range(k)
                )
                rest = low[len(bos) + len(sysregion):]
                for effort in ("high", "max"):
                    got = _render(patched, msgs, effort)
                    ref = _render(stock, msgs, effort)
                    if k == 0:
                        self.assertEqual(got, ref, f"{tag}/{name}/{effort}")
                        continue
                    self.assertEqual(
                        ref, bos + prompts[effort] + sysregion + rest,
                        f"{tag}/{name}/{effort}: stock structure",
                    )
                    self.assertEqual(
                        got, bos + sysregion + "\n\n" + prompts[effort] + rest,
                        f"{tag}/{name}/{effort}: relocation",
                    )
                    self.assertTrue(got.startswith(bos + sysregion))

    def test_context_continuations_stay_stock(self):
        ctx = [{"role": "system", "content": SYS}, {"role": "user", "content": "q1"}]
        tail = [{"role": "assistant", "content": "a1"}, {"role": "user", "content": "q2"}]
        for tag, (stock, patched) in self.pairs.items():
            for effort in (None, "low", "high", "max"):
                a = stock.encode_messages(
                    [dict(m) for m in tail], "thinking",
                    context=[dict(m) for m in ctx], reasoning_effort=effort,
                )
                b = patched.encode_messages(
                    [dict(m) for m in tail], "thinking",
                    context=[dict(m) for m in ctx], reasoning_effort=effort,
                )
                self.assertEqual(a, b, f"{tag}/effort={effort}")

    def test_assistant_final_coexistence_on_live_chain(self):
        stock, patched = self.pairs["live"]
        speaker, think = patched.ASSISTANT_SP_TOKEN, patched.thinking_start_token
        trailing = [
            {"role": "system", "content": SYS},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "A finished answer."},
        ]
        reminded = trailing + [{"role": "latest_reminder", "content": "Fresh context."}]
        for msgs in (trailing, reminded):
            got = _render(patched, msgs, "high")
            self.assertTrue(got.endswith(speaker + think), "generation header lost")
            self.assertTrue(
                got.startswith(patched.bos_token + SYS + "\n\n" + patched.REASONING_EFFORT_PROMPTS["high"]),
                "directive not at system-region tail",
            )


class Blocks(unittest.TestCase):
    """Issue #144 acceptance: cross-bucket shared-prefix blocks hash identically."""

    EFFORTS = ("low", "high", "max")
    PAIRS = (("low", "high"), ("low", "max"), ("high", "max"))

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="i144-blocks-")
        cls.stock, cls.patched = _module_pair(FIX_LIVE, cls.tmp, "blocks")
        cls.specials = [
            cls.stock.bos_token, cls.stock.eos_token,
            cls.stock.thinking_start_token, cls.stock.thinking_end_token,
            cls.stock.USER_SP_TOKEN, cls.stock.ASSISTANT_SP_TOKEN,
        ]
        cls.msgs = [
            {"role": "system", "content": BIG_SYS, "tools": TOOLS},
            {"role": "user", "content": "please fix the flaky test in scripts/test_example.py"},
        ]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _byte_tokenize(self, text):
        """Deterministic, prefix-stable surrogate: specials -> one id, else per byte."""
        tokens, i = [], 0
        while i < len(text):
            for si, sp in enumerate(self.specials):
                if text.startswith(sp, i):
                    tokens.append(0x110000 + si)
                    i += len(sp)
                    break
            else:
                tokens.extend(text[i].encode())
                i += 1
        return tokens

    @staticmethod
    def _chained_block_hashes(tokens):
        hashes, parent = [], b""
        for i in range(len(tokens) // BLOCK):
            parent = hashlib.sha256(
                parent + json.dumps(tokens[i * BLOCK:(i + 1) * BLOCK]).encode()
            ).digest()
            hashes.append(parent)
        return hashes

    @staticmethod
    def _shared_prefix(a, b):
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    def _tokenizer(self):
        path = os.environ.get("DSPARK_I144_TOKENIZER_JSON")
        if not path:
            return self._byte_tokenize, True
        from tokenizers import Tokenizer  # noqa: PLC0415

        tk = Tokenizer.from_file(path)
        return (lambda text: tk.encode(text).ids), False

    def _proof(self, mod):
        tokenize, _ = self._tokenizer()
        toks = {e: tokenize(_render(mod, self.msgs, e)) for e in self.EFFORTS}
        out = {}
        for a, b in self.PAIRS:
            ta, tb = toks[a], toks[b]
            shared_tokens = self._shared_prefix(ta, tb)
            shared_blocks = self._shared_prefix(
                self._chained_block_hashes(ta), self._chained_block_hashes(tb)
            )
            out[(a, b)] = (shared_tokens, shared_blocks, len(ta) // BLOCK)
        return out

    def test_stock_shares_zero_cross_bucket_blocks(self):
        for pair, (shared_tokens, shared_blocks, total) in self._proof(self.stock).items():
            self.assertGreater(total, 4, "fixture must span multiple blocks")
            self.assertEqual(shared_blocks, 0, f"stock {pair} must share no blocks")
            self.assertLess(shared_tokens, BLOCK, f"stock {pair} diverges inside block 0")

    def test_aligned_shares_every_full_block_of_the_system_region(self):
        tokenize, exact = self._tokenizer()
        proofs = self._proof(self.patched)
        low = _render(self.patched, self.msgs, "low")
        high = _render(self.patched, self.msgs, "high")
        shared_str = low[: len(os.path.commonprefix([low, high]))]
        expect = len(tokenize(shared_str))
        for pair, (shared_tokens, shared_blocks, total) in proofs.items():
            self.assertEqual(
                shared_blocks, shared_tokens // BLOCK,
                f"aligned {pair}: every full block of the shared prefix must hash identically",
            )
            self.assertGreater(shared_blocks, 0, f"aligned {pair} must share blocks")
        lo_hi = proofs[("low", "high")]
        if exact:
            self.assertEqual(lo_hi[0], expect, "byte surrogate is prefix-stable")
        else:
            self.assertGreaterEqual(lo_hi[0], expect - 1, "BPE junction merge costs at most 1 token")
        # the diverging tail is only the directive + junction + user turn
        self.assertGreaterEqual(lo_hi[1], lo_hi[2] - 1, "at most the junction block is lost")


class Patcher(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="i144-patcher-"))
        self.target = self.tmp / "deepseek_v4_encoding.py"
        shutil.copyfile(FIX_LIVE, self.target)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_apply_then_idempotent(self):
        self.assertEqual(HF.apply(self.target), "applied")
        data = self.target.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), HF.PATCHED_LIVE_CHAIN_SHA256)
        self.assertEqual(HF.inspect(self.target)[0], "patched")
        self.assertEqual(HF.apply(self.target), "already-patched")
        self.assertEqual(self.target.read_bytes(), data)
        self.assertEqual([q.name for q in self.tmp.iterdir()], ["deepseek_v4_encoding.py"])

    def test_refuses_foreign_bytes(self):
        self.target.write_bytes(b"x = 1\n")
        with self.assertRaises(HF.HotfixError):
            HF.inspect(self.target)
        with self.assertRaises(HF.HotfixError):
            HF.apply(self.target)
        self.assertEqual(self.target.read_bytes(), b"x = 1\n")

    def test_refuses_symlink(self):
        link = self.tmp / "link.py"
        os.symlink(self.target, link)
        with self.assertRaises(HF.HotfixError):
            HF.inspect(link)

    def test_cli_check_and_status_do_not_write(self):
        before = self.target.read_bytes()
        for flag in ("--check", "--status"):
            proc = subprocess.run(
                [sys.executable, str(PATCHER), flag, "--target", str(self.target)],
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("stock", proc.stdout)
            self.assertEqual(self.target.read_bytes(), before)

    def test_cli_check_resolves_dspark_encoding_file(self):
        env = dict(os.environ, DSPARK_ENCODING_FILE=str(self.target))
        proc = subprocess.run(
            [sys.executable, str(PATCHER), "--check"],
            capture_output=True, text=True, env=env,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn(str(self.target), proc.stdout)

    def test_cli_apply_and_status_roundtrip(self):
        proc = subprocess.run(
            [sys.executable, str(PATCHER), "--target", str(self.target)],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("applied", proc.stdout)
        proc = subprocess.run(
            [sys.executable, str(PATCHER), "--status", "--target", str(self.target)],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("patched", proc.stdout)


class Wiring(unittest.TestCase):
    def test_compose_gate_default_off_fail_closed(self):
        compose = COMPOSE.read_text()
        self.assertIn(
            'DSPARK_ENABLE_ISSUE144_EFFORT_ALIGN: "${DSPARK_ENABLE_ISSUE144_EFFORT_ALIGN:-0}"',
            compose,
        )
        self.assertIn(
            'if [ "$${DSPARK_ENABLE_ISSUE144_EFFORT_ALIGN:-0}" = "1" ]; then '
            "python3 /opt/hotfix-dsv4-issue144-effort-align.py || exit 1; fi;",
            compose,
        )
        self.assertIn(
            "${DSPARK_ISSUE144_EFFORT_ALIGN_HOTFIX:-./patches/hotfix-dsv4-issue144-effort-align.py}"
            ":/opt/hotfix-dsv4-issue144-effort-align.py:ro",
            compose,
        )
        # must run after the last encoder co-patcher (assistant-final)
        self.assertLess(
            compose.index("/opt/hotfix-dsv4-assistant-final-continuation.py || exit 1"),
            compose.index("/opt/hotfix-dsv4-issue144-effort-align.py || exit 1"),
        )

    def test_launcher_passthrough_sync_and_preflight(self):
        start = START.read_text()
        self.assertIn(
            "DSPARK_ISSUE144_EFFORT_ALIGN_HOTFIX='./patches/hotfix-dsv4-issue144-effort-align.py'",
            start,
        )
        self.assertIn("DSPARK_ENABLE_ISSUE144_EFFORT_ALIGN=$REMOTE_ISSUE144_EFFORT_ALIGN", start)
        self.assertIn("/opt/hotfix-dsv4-issue144-effort-align.py --check", start)
        self.assertIn('patches/hotfix-dsv4-issue144-effort-align.py"', start)
        self.assertIn(
            "export DSPARK_ISSUE144_EFFORT_ALIGN_HOTFIX DSPARK_ENABLE_ISSUE144_EFFORT_ALIGN",
            start,
        )

    def test_env_example_and_ci(self):
        env = ENV_EXAMPLE.read_text()
        self.assertIn("DSPARK_ENABLE_ISSUE144_EFFORT_ALIGN=0", env)
        ci = CI.read_text()
        self.assertIn("scripts/test-issue144-effort-align.py", ci)
        self.assertIn("hotfix-dsv4-issue144-effort-align.py", ci)


if __name__ == "__main__":
    unittest.main()
