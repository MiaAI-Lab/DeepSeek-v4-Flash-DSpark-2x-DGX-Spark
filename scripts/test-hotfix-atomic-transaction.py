#!/usr/bin/env python3
"""CPU behavioral tests for the whole-script transaction in the DSV4 shell
hotfix chain and the fail-closed shell-hotfix wiring in docker-compose.dspark.yml.

Fixtures are synthetic vLLM trees built from the exact anchors embedded in each
script (parsed out of the unquoted PYEOF patch heredoc). Every assertion is
behavioral — exit codes, resulting bytes, permission bits, invocation records —
with two deliberate structural guards that supplement (never replace) those
behavioral cases: the frozen INVENTORY below pins each production patch()
call's target/label/expect plus SHA-256 digests of its exact old and new hunk
bodies so removal or alteration of any transformation fails independently of
the self-derived fixtures, and test_compose_hotfix_block_precedes_real_exec
pins Compose placement.
"""
import ast
import hashlib
import os
import json
import shutil
import stat
import subprocess
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHES = ROOT / "patches"
COMPOSE = ROOT / "docker-compose.dspark.yml"

# The DSV4 perf-hotfix chain, in compose invocation order. #50004 (adaptive
# C128A topk width) was removed from the chain: upstream vLLM reverted it in
# PR #51318 because the packed row stride diverges between CUDA graph capture
# and replay, corrupting sparse-attention slot ids for rows >= 1.
CHAIN = [
    "hotfix-dsv4-mtp-buffer-50312.sh",
    "hotfix-dsv4-skip-topk-49486.sh",
    "hotfix-dsv4-dense-prefill-indexer-48407.sh",
    "hotfix-dsv4-skip-empty-c128-48957.sh",
    "hotfix-dsv4-flashmla-workspace-50298.sh",
    "hotfix-dsv4-grammar-advance.sh",
]

# Compose invocation order: issue22 + spin-wait first, then the DSV4 loop.
FULL_ORDER = ["hotfix-nvfp4-ds-mla-issue22.sh", "hotfix-gb10-spin-wait.sh"] + CHAIN

SKIP_VARS = (
    "DSPARK_SKIP_HOTFIX",
    "DSPARK_SKIP_ISSUE22_HOTFIX",
    "DSPARK_SKIP_SPIN_WAIT_HOTFIX",
)

SEP = "\n\n# ---- synthetic fixture piece boundary ----\n\n"
# Independent hunk inventory: the frozen contract each script must keep. This
# is declared here, not derived from the scripts at assertion time, so deleting
# or altering a production patch() call cannot shrink or reshape its own test
# coverage. Each entry freezes (target path, label, expect, SHA-256 of the old
# anchor, SHA-256 of the new replacement) as literals committed beside this
# test — the digests pin the exact intended transformation bytes so a hunk body
# cannot drift while its metadata stays constant.
INVENTORY = {
    "hotfix-dsv4-mtp-buffer-50312.sh": [
        ("models/deepseek_v4/nvidia/model.py", "model.py: conditional _mtp_hidden_buffer allocation (dspark -> None)", 1,
         "954a9f1960ac0ace8553416ad73b8006df3602117660cdd0820610fe3ad5bcd4",
         "db50cb54e326ac8fd8fedc4983233459c572b661ec127ca17e27325d361dde98"),
        ("models/deepseek_v4/nvidia/model.py", "model.py: skip copy_ when buffer is None", 1,
         "7733325241c93fc28f68c51f56c452d4de037f9586b643324bbcf7794e56d06b",
         "c885eb7c48f81a55e8d81624766a38bd2662131be067047c22c368cc039e1e35"),
        ("v1/worker/gpu/model_runner.py", "model_runner.py: None-guard at both get_mtp_target_hidden_states sites", 2,
         "b6c3f0087322cbe6b99d120ad9d9500ce032d92e8c9e9e4f3b4e9d310694841f",
         "8da97a0130198c5c2fe6238ee31ea59b0b47b36500cbef994fc27ea22fc89855"),
    ],
    "hotfix-dsv4-skip-topk-49486.sh": [
        ("models/deepseek_v4/attention.py", "attention.py: import tl, triton (upstream #49486)", 1,
         "68f76f8373bf48b339c5a48c2afbc113ce1fa9db3f14d901f44019cf2f1e15c6",
         "79296981ccc7b3826e52939b603a1f35b25e1c57068fc5e4eec4e6bf252bd49b"),
        ("models/deepseek_v4/attention.py", "attention.py: _fill_short_context_topk_indices kernel (upstream #49486)", 1,
         "186780ff9046c9181b189aa8423d86548abf6b76000f1aad0a1a491faf9a44a4",
         "1249c66030483c1275978189d4f17eceb678963f0c440bbec6937dd84529e335"),
        ("models/deepseek_v4/attention.py", "attention.py: short-context early return (upstream #49486)", 1,
         "898b7b950f9ddc78e823e08edc9ecaa7be11ab6bf3b2a73c15a8eac65f8f182a",
         "b3bbeccfc040c8b9c6144de89802b3605c4dc2bacf029188884a7aafbd7be8a4"),
    ],
    "hotfix-dsv4-dense-prefill-indexer-48407.sh": [
        ("model_executor/layers/sparse_attn_indexer.py", "sparse_attn_indexer.py: import CUDAGraphMode", 1,
         "378518abbc91f1f3e0a7ef18f168cf93f77d3a9d2dab84112bad7ec4ae3c04fd",
         "5cff944b744afa233b696f3ddeabc7ccb118a19c79dae39e6f34e5b2a5bd73de"),
        ("model_executor/layers/sparse_attn_indexer.py", "sparse_attn_indexer.py: sparse_attn_indexer() signature param", 1,
         "fbd1c122a0a0fadd5eb5b86ee73d96fbf48995b2742f3896450ab16b7801e441",
         "2de40901ad53ad324b09d9e1d106035183d9d4d0267831d2e65daaf7be2bcae9"),
        ("model_executor/layers/sparse_attn_indexer.py", "sparse_attn_indexer.py: capture forward_context", 1,
         "44c82d8629352c63bee8f14ade091c02ed9d9f585753af7d3374b45bd5499fc0",
         "8c25c2fec2ecef433a1687263c62379ed7147fc87109543d01c35ff942c77de5"),
        ("model_executor/layers/sparse_attn_indexer.py", "sparse_attn_indexer.py: pass param into sparse_attn_indexer_fake()", 1,
         "2ff55f89560329c31c757e8374dd2bcf759b4956da80667c8ecd42ec18b6b081",
         "9cbac087f6a8abe4210f8136e09b7ffee1c654efc5b585cb7cbc4d34f8b22944"),
        ("model_executor/layers/sparse_attn_indexer.py", "sparse_attn_indexer.py: sparse_attn_indexer_fake() signature param", 1,
         "20b0ad8732f8111100a068056403dab55489391d5605a320e74c5bc29d6a8ec3",
         "b0534c9ff23ba2b00b6a18eb56d579df4f30c25a690fb999ea5aa40fbbd004a9"),
        ("model_executor/layers/sparse_attn_indexer.py", "sparse_attn_indexer.py: dormant scoring-skip gate (mainline #48407)", 1,
         "a34ec14880899b0369d226b98bf8a914d6c1e4222a96102188bdfe073e4c1de6",
         "f7b8f895b2b229a5f0834eaf7a2657f4e5bcc528d5094b85557666995c044845"),
        ("model_executor/layers/sparse_attn_indexer.py", "sparse_attn_indexer.py: __init__ dormant binding field", 1,
         "a7dcce9e76c4cbf4c11acd194f1fafe1cf15cf4c12bb4d95b820d8051cb7c9d6",
         "edd2a7fd31e89404499c9f67d679cc195c169084c825045669e01388780037ec"),
        ("model_executor/layers/sparse_attn_indexer.py", "sparse_attn_indexer.py: forward_cuda passes binding", 1,
         "e07725e5255eb970a883cd7549e4d07459a4decce3e21e425328e21ebe8db68d",
         "606cda42c7fd780a824b876c796c353f92ff26ab2a1c88c442f787d464d992ec"),
        ("models/deepseek_v4/sparse_mla.py", "sparse_mla.py: DeepseekV4FlashMLAMetadata.num_decode_tokens field", 1,
         "765f9ec310a8f1be9afb979b1e0b9ad1868d85f66af65b9f658fa58d2a722aea",
         "acd97e5912b432296f5396e5e5dfe9518918dcd0d11946dca2dd99edfb56ecd5"),
        ("models/deepseek_v4/sparse_mla.py", "sparse_mla.py: build() populates num_decode_tokens", 1,
         "9d607a433a360562face62b60253f9933edb09bb6406635f625500accaafd897",
         "9129358d752f2c629d5f0f388f3ec1801c8ce27dc89f78be2ee872b0bf4e7fec"),
        ("models/deepseek_v4/attention.py", "attention.py: DeepseekV4Indexer dormant binding \"\"", 1,
         "122728f85fc35d091f492f3de52926253874a6dc019abcd706fdf4ec2c2d0516",
         "63bced08ed073a199a095c5f46cc94658f2fb25204a33b4b880d00fdcedca0b9"),
        ("models/deepseek_v32/nvidia/attention.py", "deepseek_v32/nvidia/attention.py: keyword args + dormant binding", 1,
         "9b84d615a1ff63caee7e5b341cbbbc0da10aa37788c9989a7087df3435def981",
         "961e909de5887da2d5d973a1204267b96e6897a7b2b366d06a3b1e8232987eca"),
    ],
    "hotfix-dsv4-skip-empty-c128-48957.sh": [
        ("models/deepseek_v4/compressor.py", "compressor.py: import CUDAGraphMode (upstream #48957)", 1,
         "4cde2795795100dada128de5bba2f34ffa8cd895d860e9d98eaf1e076565742b",
         "caad0c0dd2cfef20094b6ee8e3ad5b9eca110f71b1066e5635ff4734091be7e0"),
        ("models/deepseek_v4/compressor.py", "compressor.py: _get_c128_boundary helper (upstream #48957)", 1,
         "1d821e0a17fc518176a81902c2159d42ddca8d3621664a4bd4c351c4eef4d9e1",
         "6db65eb868f4266a83dcb1739ab4a859b38c99c6c1f46c681f627f20b87a4201"),
        ("models/deepseek_v4/compressor.py", "compressor.py: CompressorMetadata.c128_boundary field (upstream #48957)", 1,
         "e8fbdcde89c445eb2a4bc0dba1823255c648ec5db58295bee8b8c7e651483fee",
         "9258a9d397b7b55411fdfdecb88b1661cb3f056b9aabbf45a58eab99b0e21c2d"),
        ("models/deepseek_v4/compressor.py", "compressor.py: build() populates c128_boundary (upstream #48957)", 1,
         "b5733aead940d07ac053488edbe0803ed117beff08892ace183e30bae4326209",
         "9e031186a1e2975a509745e2d99cb573d69bb9c35564761af2a1c9cd7629c98a"),
        ("models/deepseek_v4/compressor.py", "compressor.py: capture forward_context (upstream #48957)", 1,
         "f9df191696976a238b68c839ea59df1cfae5de2c9656f5568dad066441ae7de5",
         "770c40be85f26b4695a114afcd14eac27de2356acfe4a3625091bd9462f2ccca"),
        ("models/deepseek_v4/compressor.py", "compressor.py: skip empty C128 launch gate (upstream #48957)", 1,
         "c4bdcc39eb576603712cff9059a2f9ea46d71ac9408ef7aef5bf56463c2ab5c5",
         "6f01973635212da61a6f28b088ea75282c25720613fde0b0219199a1e4b15214"),
    ],
    "hotfix-dsv4-flashmla-workspace-50298.sh": [
        ("models/deepseek_v4/nvidia/flashmla.py", "flashmla.py: import round_up (upstream #50298)", 1,
         "f4f19fa3fad7030bdbfd2b3daecabe06c279bfbeee96650416b8d58b700b729f",
         "9865cad3c99d802f18212347263e09c7ef69a7bd9c94f622d8d814cd240ff3b2"),
        ("models/deepseek_v4/nvidia/flashmla.py", "flashmla.py: dummy path reserves combined-topk buffers (upstream #50298)", 1,
         "4bd06bca2c1b9a5f5f7b00b0e04b6b70545f9240f4b8b7336d0b3a9a9642053c",
         "df1fbeae100f475eb1ee3d45ed70cebcfbf9d7a3e7506f1401fb534f9da58739"),
        ("models/deepseek_v4/nvidia/flashmla.py", "flashmla.py: _forward_prefill workspace request (upstream #50298)", 1,
         "6ba0115d0b97d9472f7a024e05334cef0942da1179da66d42e978913f65475d2",
         "ef7a43f4821889e6b0fc223580c777263a458aced9a33479046211ffc7524d9a"),
        ("models/deepseek_v4/nvidia/flashmla.py", "flashmla.py: slice reused buffers per chunk (upstream #50298)", 1,
         "89fd4b3f3375d9cd76f230c81d9f0b0866005463c9f5bc687d9ce3dfa98042f5",
         "606b21e39029f5a7a3910569ce09c38efea926410ba8cbe914dad0e2d2a57b7b"),
        ("models/deepseek_v4/nvidia/flashmla.py", "flashmla.py: combine_topk_swa_indices out= kwarg (upstream #50298)", 1,
         "b9b5b5b4deef92d917cdf92d25d972ff1b16982bb8dae2ae3e9a51fba865c399",
         "a77dec9520ad96628281acc1c3e755ec12a5467335f6b8b86f3d9aba6877cfc0"),
        ("models/deepseek_v4/common/ops/cache_utils.py", "cache_utils.py: combine_topk_swa_indices out= param (upstream #50298)", 1,
         "716ae4433494bcabd21f6d249b524d585da73b70e0b348be86b6719148959235",
         "6025b832ee0226204c716127359eb9601ba7449341fd535b335f54031a3f7e1e"),
    ],
    "hotfix-dsv4-grammar-advance.sh": [
        ("v1/structured_output/__init__.py", "__init__.py: should_advance new_token_ids param (upstream #44993)", 1,
         "02bb2a2812fdcaa454765995718146738468c525efeb23446f65befe6621fa58",
         "1b2166796e8c735a7220ab1c4d29cffc8465d186ba71f6c004c121f41e3c762b"),
        ("v1/structured_output/__init__.py", "__init__.py: delta window + boundary record, all constraint types (upstream #44993)", 1,
         "e7d8c9db971216d3c2296024d0297b790b13fe3a04c039068efa7ca1df24c57f",
         "e1d612959060b54651a13c58df83df17aae5551e9e97ef6814cd6c24137cba8b"),
        ("v1/core/sched/scheduler.py", "scheduler.py: pass new_token_ids into should_advance (upstream #44993)", 1,
         "0ae271f9107ba474984cd318604f6a7c0b406ea3ee19322386ed88965ce5cd7f",
         "c75567a86d1f1cb5c2861091186d90c4cdac25573c4519ca7665585349e769f6"),
    ],}


# Test-only sitecustomize: fail the Nth os.replace call of the patched python
# process (one-shot) so commit-failure rollback runs against real writes. No
# production hooks involved; injected purely via PYTHONPATH from the tests.
SITECUSTOMIZE = """import json
import os
import pathlib

_fail = {x for x in os.environ.get("HOTFIX_TEST_FAIL_REPLACE_CALLS", "").split(",") if x}
_after = {x for x in os.environ.get("HOTFIX_TEST_FAIL_REPLACE_AFTER_CALLS", "").split(",") if x}
_corrupt = {x for x in os.environ.get("HOTFIX_TEST_CORRUPT_AFTER_CALLS", "").split(",") if x}
_interrupt = {x for x in os.environ.get("HOTFIX_TEST_INTERRUPT_AFTER_CALLS", "").split(",") if x}
_read_suffix = os.environ.get("HOTFIX_TEST_FAIL_READ_SUFFIX", "")
_read_budget = int(os.environ.get("HOTFIX_TEST_FAIL_READ_MAX", "0"))
_interrupt_read_suffix = os.environ.get("HOTFIX_TEST_INTERRUPT_READ_SUFFIX", "")
_interrupt_read_budget = int(os.environ.get("HOTFIX_TEST_INTERRUPT_READ_MAX", "0"))
_record = os.environ.get("HOTFIX_TEST_RECORD_RENAMES", "")

_state = {"n": 0}
_real_replace = os.replace


def _log(call, src, dst):
    if _record:
        with open(_record, "a") as fh:
            fh.write(json.dumps({"call": call, "src": str(src), "dst": str(dst)}) + "\\n")


def _patched_replace(src, dst):
    _state["n"] += 1
    tag = str(_state["n"])
    if tag in _fail:
        raise OSError(tag, "injected os.replace failure (test)")
    _real_replace(src, dst)
    _log(tag, src, dst)
    if tag in _after:
        raise OSError(tag, "injected post-rename failure (test)")
    if tag in _corrupt:
        with open(dst, "wb") as fh:
            fh.write(b"CORRUPTED-BY-TEST\\n")
    if tag in _interrupt:
        raise KeyboardInterrupt("injected interrupt after publish (test)")


os.replace = _patched_replace

if _read_suffix:
    _real_read = pathlib.Path.read_bytes
    _read_state = {"n": 0}

    def _patched_read(self):
        if _state["n"] > 0 and str(self).endswith(_read_suffix):
            _read_state["n"] += 1
            if _read_state["n"] <= _read_budget:
                raise OSError("injected verification read failure (test)")
        return _real_read(self)

    pathlib.Path.read_bytes = _patched_read

if _interrupt_read_suffix:
    # Verification-time interrupt: raised from Path.read_bytes AFTER at least
    # one real publish (gated on _state["n"] > 0) and entirely outside the
    # patched os.replace wrapper, so only a transaction-wide BaseException
    # boundary can catch it.
    _ireal_read = pathlib.Path.read_bytes
    _iread_state = {"n": 0}

    def _interrupting_read(self):
        if _state["n"] > 0 and str(self).endswith(_interrupt_read_suffix):
            _iread_state["n"] += 1
            if _iread_state["n"] <= _interrupt_read_budget:
                raise KeyboardInterrupt(
                    "injected verification-time interrupt (test)"
                )
        return _ireal_read(self)

    pathlib.Path.read_bytes = _interrupting_read
"""


@dataclass
class Hunk:
    path: str
    old: str
    new: str
    label: str
    expect: int


_hunk_cache = {}


def hunks(script):
    """Parse the patch(...) calls out of a script's PYEOF heredoc."""
    if script not in _hunk_cache:
        text = (PATCHES / script).read_text()
        body = text.split("python3 <<PYEOF\n", 1)[1].split("\nPYEOF", 1)[0]
        tree = ast.parse(body.replace("$VLLM_ROOT", "."))
        found = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "patch"
            ):
                path, old, new, label = (a.value for a in node.args)
                expect = 1
                for kw in node.keywords:
                    assert kw.arg == "expect"
                    expect = kw.value.value
                found.append(Hunk(path, old, new, label, expect))
        _hunk_cache[script] = found
    return _hunk_cache[script]


def corrupt(anchor):
    """Break an anchor so its count no longer matches expect."""
    if "\n" in anchor:
        return anchor.replace("\n", " \n", 1)
    return anchor[:-1] + ("X" if anchor[-1] != "X" else "Z")


def materialize(entries):
    """entries: [(hunk, broken?)] in apply order -> {rel path: bytes}."""
    per_file = {}
    for h, broken in entries:
        piece = h.old * h.expect
        if broken:
            piece = corrupt(h.old) + h.old * (h.expect - 1)
        per_file.setdefault(h.path, []).append(piece)
    return {rel: (SEP.join(pieces) + "\n").encode() for rel, pieces in per_file.items()}


def build_tree(scripts):
    """Synthetic vLLM root plus the byte-exact post-apply expectation."""
    entries = [(h, False) for s in scripts for h in hunks(s)]
    files = materialize(entries)
    exp_per_file = {}
    for h, _ in entries:
        exp_per_file.setdefault(h.path, []).append(h.new * h.expect)
    expected = {
        rel: (SEP.join(pieces) + "\n").encode() for rel, pieces in exp_per_file.items()
    }
    return files, expected


def make_root(files, parent=None, mode=None, modes=None):
    root = Path(tempfile.mkdtemp(prefix="vllm-root-", dir=parent))
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        m = (modes or {}).get(rel, mode)
        if m is not None:
            p.chmod(m)
    return root


def snap(root):
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def temps(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob(".*.tmp"))


def snap_modes(root):
    return {
        str(p.relative_to(root)): stat.S_IMODE(p.stat().st_mode)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def read_renames(path):
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def run_hotfix(script, root, extra_env=None):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["VLLM_ROOT"] = str(root)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(PATCHES / script)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )

def compose_hotfix_lines():
    text = COMPOSE.read_text()
    found = []
    for var in SKIP_VARS:
        token = "$${" + var + ":-0}"
        matches = [(i, ln.strip()) for i, ln in enumerate(text.splitlines()) if token in ln]
        if len(matches) != 1:
            raise AssertionError(
                f"expected exactly one compose line for {var}, got {len(matches)}"
            )
        found.extend(matches)
    found.sort(key=lambda item: item[0])
    return [ln for _, ln in found]


STUB_TEMPLATE = (
    "#!/usr/bin/env bash\n"
    "printf '%s\\n' \"$(basename \"$0\")\" >> \"$INVOCATIONS\"\n"
    "exit {code}\n"
)


class TransactionTests(unittest.TestCase):
    def test_inventory_contract(self):
        # Parsed production patch() calls must match the frozen inventory
        # exactly: count, target paths in order, labels, expect values, and
        # the SHA-256 digests of every old anchor and new replacement. The
        # digests are frozen literals above, so any edit to a hunk body —
        # not just its removal or retargeting — fails this guard even though
        # fixtures and oracles are derived from the same parse.
        self.assertEqual(sorted(INVENTORY), sorted(CHAIN))
        for script in CHAIN:
            with self.subTest(script=script):
                parsed = [
                    (
                        h.path,
                        h.label,
                        h.expect,
                        hashlib.sha256(h.old.encode()).hexdigest(),
                        hashlib.sha256(h.new.encode()).hexdigest(),
                    )
                    for h in hunks(script)
                ]
                self.assertEqual(parsed, INVENTORY[script])

    def test_mixed_partial_hunk_fails_closed(self):
        # MTP expect=2 with one callsite already new and one still old: the
        # hunk is partial, so the script must never report success while the
        # file stays half-applied, and nothing may be written.
        script = "hotfix-dsv4-mtp-buffer-50312.sh"
        files, _ = build_tree([script])
        guard = next(h for h in hunks(script) if h.expect == 2)
        mixed = {rel: data for rel, data in files.items()}
        mixed[guard.path] = (
            guard.new + SEP + guard.old + "\n"
        ).encode()
        root = make_root(mixed)
        try:
            before = snap(root)
            r = run_hotfix(script, root)
            self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertEqual(snap(root), before)
            self.assertEqual(temps(root), [])
            self.assertNotIn("Committed", r.stdout)
            self.assertNotIn("Hotfix applied", r.stdout)
            self.assertNotIn("already applied", r.stdout)
        finally:
            shutil.rmtree(root)

    def test_missing_target_file_zero_writes(self):
        # A missing vLLM target while other hunks validate: nonzero exit and
        # zero writes anywhere.
        script = "hotfix-dsv4-grammar-advance.sh"
        files, _ = build_tree([script])
        victim = "v1/core/sched/scheduler.py"
        del files[victim]
        root = make_root(files)
        try:
            before = snap(root)
            r = run_hotfix(script, root)
            self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertEqual(snap(root), before)
            self.assertEqual(temps(root), [])
            self.assertNotIn("Committed", r.stdout)
        finally:
            shutil.rmtree(root)

    def test_all_chain_clean_apply_byte_exact_mode_kept(self):
        for script in CHAIN:
            with self.subTest(script=script):
                files, expected = build_tree([script])
                root = make_root(files, mode=0o640)
                try:
                    r = run_hotfix(script, root)
                    self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                    for rel, want in expected.items():
                        self.assertEqual((root / rel).read_bytes(), want, rel)
                        self.assertEqual(
                            stat.S_IMODE((root / rel).stat().st_mode), 0o640, rel
                        )
                    self.assertIn("[stage]", r.stdout)
                    self.assertIn("Committed", r.stdout)
                    self.assertEqual(temps(root), [])
                finally:
                    shutil.rmtree(root)

    def test_idempotent_rerun_makes_no_changes(self):
        for script in CHAIN:
            with self.subTest(script=script):
                files, _ = build_tree([script])
                root = make_root(files)
                try:
                    first = run_hotfix(script, root)
                    self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
                    before = snap(root)
                    second = run_hotfix(script, root)
                    self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
                    self.assertIn("already applied", second.stdout)
                    self.assertEqual(snap(root), before)
                    self.assertEqual(temps(root), [])
                finally:
                    shutil.rmtree(root)

    def test_each_hunk_position_failure_is_atomic_zero_writes(self):
        # Every hunk position, including the final anchor: a validation failure
        # anywhere must exit nonzero with EVERY target byte-identical.
        cases = 0
        for script in CHAIN:
            hs = hunks(script)
            for k in range(len(hs)):
                with self.subTest(script=script, hunk=k, label=hs[k].label):
                    files = materialize([(h, i == k) for i, h in enumerate(hs)])
                    root = make_root(files)
                    try:
                        before = snap(root)
                        r = run_hotfix(script, root)
                        self.assertNotEqual(r.returncode, 0, r.stdout)
                        self.assertEqual(snap(root), before)
                        self.assertNotIn("Committed", r.stdout)
                        self.assertNotIn("Hotfix applied", r.stdout)
                        self.assertEqual(temps(root), [])
                    finally:
                        shutil.rmtree(root)
                    cases += 1
        self.assertGreater(cases, 0)

    def test_final_anchor_replacement_is_byte_exact(self):
        for script in CHAIN:
            with self.subTest(script=script):
                last = hunks(script)[-1]
                files, _ = build_tree([script])
                root = make_root(files)
                try:
                    r = run_hotfix(script, root)
                    self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                    data = (root / last.path).read_bytes()
                    self.assertEqual(data.count(last.new.encode()), last.expect)
                    self.assertEqual(data.count(last.old.encode()), 0)
                finally:
                    shutil.rmtree(root)

    def setup_fault_run(self, script, modes=None):
        files, _ = build_tree([script])
        workdir = Path(tempfile.mkdtemp(prefix="hotfix-fault-"))
        self.addCleanup(shutil.rmtree, workdir, ignore_errors=True)
        site = workdir / "site"
        site.mkdir()
        (site / "sitecustomize.py").write_text(SITECUSTOMIZE)
        root = make_root(files, parent=workdir, modes=modes)
        return root, snap(root), snap_modes(root), workdir

    @staticmethod
    def fault_env(workdir, **vars):
        env = {"PYTHONPATH": str(workdir / "site")}
        env.update({k: v for k, v in vars.items() if v})
        return env

    def test_commit_fault_in_each_engine_rolls_back_verified(self):
        # One pre-replace commit fault per duplicated engine; multi-file
        # engines fail mid-commit so published targets must roll back with
        # bytes AND permission bits restored exactly.
        cases = [
            ("hotfix-dsv4-mtp-buffer-50312.sh", "2"),
            ("hotfix-dsv4-skip-topk-49486.sh", "1"),
            ("hotfix-dsv4-dense-prefill-indexer-48407.sh", "3"),
            ("hotfix-dsv4-skip-empty-c128-48957.sh", "1"),
            ("hotfix-dsv4-flashmla-workspace-50298.sh", "2"),
            ("hotfix-dsv4-grammar-advance.sh", "2"),
        ]
        distinct_modes = [0o600, 0o640, 0o750, 0o644]
        for script, fail_calls in cases:
            with self.subTest(script=script, fail_calls=fail_calls):
                files, _ = build_tree([script])
                modes = {
                    rel: distinct_modes[i % len(distinct_modes)]
                    for i, rel in enumerate(sorted(files))
                }
                root, before, before_modes, workdir = self.setup_fault_run(
                    script, modes=modes
                )
                r = run_hotfix(script, root, extra_env=self.fault_env(
                    workdir, HOTFIX_TEST_FAIL_REPLACE_CALLS=fail_calls))
                self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
                self.assertIn("commit failed", r.stderr)
                self.assertNotIn("Committed", r.stdout)
                self.assertEqual(snap(root), before)
                self.assertEqual(snap_modes(root), before_modes)
                self.assertEqual(temps(root), [])

    def test_post_rename_failure_rolls_back_current_target(self):
        # The rename lands, THEN the fault raises: the current target must be
        # tracked as published and restored together with earlier targets.
        for script, after_call in (
            ("hotfix-dsv4-mtp-buffer-50312.sh", "1"),
            ("hotfix-dsv4-dense-prefill-indexer-48407.sh", "4"),
        ):
            with self.subTest(script=script, after_call=after_call):
                root, before, before_modes, workdir = self.setup_fault_run(script)
                r = run_hotfix(script, root, extra_env=self.fault_env(
                    workdir, HOTFIX_TEST_FAIL_REPLACE_AFTER_CALLS=after_call))
                self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
                self.assertIn("commit failed", r.stderr)
                self.assertIn("[restored]", r.stderr)
                self.assertNotIn("Committed", r.stdout)
                self.assertEqual(snap(root), before)
                self.assertEqual(snap_modes(root), before_modes)
                self.assertEqual(temps(root), [])

    def test_verification_read_failure_rolls_back(self):
        script = "hotfix-dsv4-mtp-buffer-50312.sh"
        root, before, before_modes, workdir = self.setup_fault_run(script)
        r = run_hotfix(script, root, extra_env=self.fault_env(
            workdir,
            HOTFIX_TEST_FAIL_READ_SUFFIX="model_runner.py",
            HOTFIX_TEST_FAIL_READ_MAX="1",
        ))
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("post-commit verification failed", r.stderr)
        self.assertIn("[restored]", r.stderr)
        self.assertNotIn("Committed", r.stdout)
        self.assertEqual(snap(root), before)
        self.assertEqual(snap_modes(root), before_modes)
        self.assertEqual(temps(root), [])

    def test_post_write_corruption_detected_and_rolled_back(self):
        script = "hotfix-dsv4-mtp-buffer-50312.sh"
        root, before, before_modes, workdir = self.setup_fault_run(script)
        r = run_hotfix(script, root, extra_env=self.fault_env(
            workdir, HOTFIX_TEST_CORRUPT_AFTER_CALLS="1"))
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("post-commit verification failed", r.stderr)
        self.assertIn("[restored]", r.stderr)
        self.assertNotIn("Committed", r.stdout)
        self.assertEqual(snap(root), before)
        self.assertEqual(snap_modes(root), before_modes)
        self.assertEqual(temps(root), [])

    def test_interrupt_after_publish_rolls_back(self):
        script = "hotfix-dsv4-mtp-buffer-50312.sh"
        root, before, before_modes, workdir = self.setup_fault_run(script)
        r = run_hotfix(script, root, extra_env=self.fault_env(
            workdir, HOTFIX_TEST_INTERRUPT_AFTER_CALLS="1"))
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("[restored]", r.stderr)
        self.assertNotIn("Committed", r.stdout)
        self.assertEqual(snap(root), before)
        self.assertEqual(snap_modes(root), before_modes)
        self.assertEqual(temps(root), [])

    def test_verification_interrupt_rolls_back_and_reraises(self):
        # KeyboardInterrupt raised during the post-commit verification READ —
        # after real publishes have landed and entirely outside os.replace.
        # Only a transaction-wide BaseException boundary can catch this timing:
        # every published target must roll back (bytes AND modes), no temp
        # file may survive, and the interrupt must propagate (nonzero exit).
        script = "hotfix-dsv4-mtp-buffer-50312.sh"
        root, before, before_modes, workdir = self.setup_fault_run(script)
        renames = workdir / "renames.jsonl"
        r = run_hotfix(script, root, extra_env=self.fault_env(
            workdir,
            HOTFIX_TEST_INTERRUPT_READ_SUFFIX="model.py",
            HOTFIX_TEST_INTERRUPT_READ_MAX="1",
            HOTFIX_TEST_RECORD_RENAMES=str(renames),
        ))
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        # The injection point is deterministic: the first verification read of
        # model.py happens only after its real rename (and any earlier ones).
        records = read_renames(renames)
        self.assertGreaterEqual(len(records), 1)
        self.assertTrue(
            records[0]["dst"].endswith("models/deepseek_v4/nvidia/model.py")
        )
        self.assertIn("transaction interrupted", r.stderr)
        self.assertIn("[restored]", r.stderr)
        self.assertIn("KeyboardInterrupt", r.stderr)
        self.assertNotIn("Committed", r.stdout)
        self.assertEqual(snap(root), before)
        self.assertEqual(snap_modes(root), before_modes)
        self.assertEqual(temps(root), [])

    def test_rollback_failure_is_loud_exit_2(self):
        # Publish file 1, fail committing file 2, then fail restoring file 1:
        # rollback failure must be loud with exit status 2.
        script = "hotfix-dsv4-mtp-buffer-50312.sh"
        root, _, _, workdir = self.setup_fault_run(script)
        r = run_hotfix(script, root, extra_env=self.fault_env(
            workdir, HOTFIX_TEST_FAIL_REPLACE_CALLS="2,3"))
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("FATAL", r.stderr)

    def test_commit_order_and_same_dir_temp_sources(self):
        script = "hotfix-dsv4-dense-prefill-indexer-48407.sh"
        files, expected = build_tree([script])
        workdir = Path(tempfile.mkdtemp(prefix="hotfix-order-"))
        self.addCleanup(shutil.rmtree, workdir, ignore_errors=True)
        site = workdir / "site"
        site.mkdir()
        (site / "sitecustomize.py").write_text(SITECUSTOMIZE)
        renames = workdir / "renames.jsonl"
        root = make_root(files, parent=workdir)
        r = run_hotfix(script, root, extra_env={
            "PYTHONPATH": str(site),
            "HOTFIX_TEST_RECORD_RENAMES": str(renames),
        })
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        records = read_renames(renames)
        sorted_rels = sorted(files)
        self.assertEqual(
            [rec["dst"] for rec in records],
            [str(root / rel) for rel in sorted_rels],
        )
        for rec in records:
            src, dst = Path(rec["src"]), Path(rec["dst"])
            self.assertEqual(src.parent, dst.parent)
            self.assertTrue(src.name.startswith("."))
            self.assertTrue(src.name.endswith(".tmp"))
        self.assertEqual(temps(root), [])

    def test_reverse_rollback_order_recorded(self):
        script = "hotfix-dsv4-dense-prefill-indexer-48407.sh"
        files, _ = build_tree([script])
        workdir = Path(tempfile.mkdtemp(prefix="hotfix-order-"))
        self.addCleanup(shutil.rmtree, workdir, ignore_errors=True)
        site = workdir / "site"
        site.mkdir()
        (site / "sitecustomize.py").write_text(SITECUSTOMIZE)
        renames = workdir / "renames.jsonl"
        root = make_root(files, parent=workdir)
        r = run_hotfix(script, root, extra_env={
            "PYTHONPATH": str(site),
            "HOTFIX_TEST_RECORD_RENAMES": str(renames),
            "HOTFIX_TEST_FAIL_REPLACE_CALLS": "3",
        })
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        records = read_renames(renames)
        sorted_rels = sorted(files)
        published = [str(root / rel) for rel in sorted_rels[:2]]
        self.assertEqual(
            [rec["dst"] for rec in records],
            published + list(reversed(published)),
        )

    def test_normal_caller_order_shared_tree_converges(self):
        # Compose applies the whole chain sequentially over one live tree; the shared
        # fixture must converge byte-exactly and then be fully idempotent.
        files, expected = build_tree(CHAIN)
        root = make_root(files)
        try:
            for script in CHAIN:
                r = run_hotfix(script, root)
                self.assertEqual(r.returncode, 0, f"{script}: {r.stdout}{r.stderr}")
            for rel, want in expected.items():
                self.assertEqual((root / rel).read_bytes(), want, rel)
            for script in CHAIN:
                r = run_hotfix(script, root)
                self.assertEqual(r.returncode, 0, f"{script}: {r.stdout}{r.stderr}")
                self.assertIn("already applied", r.stdout)
            self.assertEqual(temps(root), [])
        finally:
            shutil.rmtree(root)


class ComposeFailClosedWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lines = compose_hotfix_lines()

    def run_block(self, env_extra=None, stub_exits=None, remove=(), real=(), vllm_root=None):
        workdir = Path(tempfile.mkdtemp(prefix="compose-wiring-"))
        self.addCleanup(shutil.rmtree, workdir)
        patches = workdir / "dspark-patches"
        patches.mkdir()
        exits = dict(stub_exits or {})
        for name in FULL_ORDER:
            if name in remove:
                continue
            stub = patches / name
            if name in real:
                # Thin logging wrapper that execs the REAL repo script.
                stub.write_text(
                    "#!/usr/bin/env bash\n"
                    f'printf \'%s\\n\' "{name}" >> "$INVOCATIONS"\n'
                    f'exec bash "{PATCHES / name}"\n'
                )
            else:
                stub.write_text(STUB_TEMPLATE.format(code=exits.get(name, 0)))
            stub.chmod(0o755)
        invocations = workdir / "invocations.txt"
        invocations.touch()
        reached = workdir / "reached.txt"
        block = "\n".join(self.lines)
        block = block.replace("$$", "$").replace("/opt/dspark-patches", str(patches))
        script = block + f'\nprintf x >> "{reached}"\n'
        env = dict(os.environ)
        for var in SKIP_VARS:
            env.pop(var, None)
        env["INVOCATIONS"] = str(invocations)
        if vllm_root is not None:
            env["VLLM_ROOT"] = str(vllm_root)
        if env_extra:
            env.update(env_extra)
        proc = subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        hit = reached.exists()
        inv = invocations.read_text().split()
        return proc, inv, hit

    def test_default_boot_runs_all_shell_hotfixes_in_compose_order(self):
        proc, inv, reached = self.run_block()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(inv, FULL_ORDER)
        self.assertTrue(reached)

    def test_each_skip_switch_skips_only_its_group(self):
        for var, excluded in (
            ("DSPARK_SKIP_HOTFIX", tuple(CHAIN)),
            ("DSPARK_SKIP_ISSUE22_HOTFIX", ("hotfix-nvfp4-ds-mla-issue22.sh",)),
            ("DSPARK_SKIP_SPIN_WAIT_HOTFIX", ("hotfix-gb10-spin-wait.sh",)),
        ):
            with self.subTest(var=var):
                proc, inv, reached = self.run_block(env_extra={var: "1"})
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                self.assertTrue(reached)
                self.assertEqual(inv, [s for s in FULL_ORDER if s not in excluded])

    def test_missing_enabled_script_fails_closed_before_exec(self):
        victim = "hotfix-dsv4-skip-empty-c128-48957.sh"
        proc, inv, reached = self.run_block(remove=(victim,))
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertFalse(reached)
        self.assertEqual(inv, FULL_ORDER[: FULL_ORDER.index(victim)])
        proc, inv, reached = self.run_block(remove=("hotfix-gb10-spin-wait.sh",))
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertFalse(reached)
        self.assertEqual(inv, ["hotfix-nvfp4-ds-mla-issue22.sh"])
        proc, inv, reached = self.run_block(
            env_extra={"DSPARK_SKIP_HOTFIX": "1"}, remove=tuple(CHAIN)
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertTrue(reached)
        self.assertEqual(
            inv, ["hotfix-nvfp4-ds-mla-issue22.sh", "hotfix-gb10-spin-wait.sh"]
        )

    def test_nonzero_script_exit_propagates_and_blocks_exec(self):
        for name, code in (
            ("hotfix-dsv4-grammar-advance.sh", 7),
            ("hotfix-gb10-spin-wait.sh", 1),
            ("hotfix-nvfp4-ds-mla-issue22.sh", 3),
        ):
            with self.subTest(script=name, code=code):
                proc, inv, reached = self.run_block(stub_exits={name: code})
                self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
                self.assertFalse(reached)
                self.assertEqual(inv, FULL_ORDER[: FULL_ORDER.index(name) + 1])

    def test_real_spin_wait_drift_blocks_exec(self):
        # The REAL hotfix-gb10-spin-wait.sh runs through the extracted Compose
        # block against a VLLM_ROOT whose shm_broadcast.py contains NEITHER
        # anchor: the script must fail verification, Compose must propagate
        # nonzero before exec, and the fixture must stay byte-identical.
        workdir = Path(tempfile.mkdtemp(prefix="spin-drift-"))
        self.addCleanup(shutil.rmtree, workdir)
        vllm_root = workdir / "vllm"
        shm = vllm_root / "distributed" / "device_communicators"
        shm.mkdir(parents=True)
        drifted = (
            b"# drifted fixture: neither production anchor present\n"
            b"class SpinCondition:\n"
            b"    busy_loop_s: float = 30\n"
        )
        shm_file = shm / "shm_broadcast.py"
        shm_file.write_bytes(drifted)
        proc, inv, reached = self.run_block(
            real=("hotfix-gb10-spin-wait.sh",),
            vllm_root=vllm_root,
        )
        self.assertNotEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertFalse(reached)
        self.assertIn("hotfix-gb10-spin-wait.sh", inv)
        self.assertEqual(shm_file.read_bytes(), drifted)

    def test_compose_hotfix_block_precedes_real_exec(self):
        # Supplemental placement guard: all three shell-hotfix lines must sit
        # before the real `exec vllm serve` in the Compose command.
        text = COMPOSE.read_text()
        exec_pos = text.index("exec /usr/local/bin/vllm serve")
        for var in SKIP_VARS:
            self.assertLess(text.index("$${" + var + ":-0}"), exec_pos)


if __name__ == "__main__":
    unittest.main()
