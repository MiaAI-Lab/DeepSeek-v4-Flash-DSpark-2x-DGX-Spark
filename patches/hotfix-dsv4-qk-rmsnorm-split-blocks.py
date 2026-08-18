#!/usr/bin/env python3
"""hotfix-dsv4-qk-rmsnorm-split-blocks.py — per-task Q/KV RMSNorm block widths
(backport of upstream vLLM PR #49283).

Rewrites, in place and idempotently,
`vllm/models/deepseek_v4/common/ops/fused_qk_rmsnorm.py` inside the Anemll
dspark-vllm-gx10 0.1.1 image (vLLM 0.25.2.dev0+g752a3a504.d20260714).

WHAT IT DOES
The fused kernel normalises two rows per token on a `(num_tokens, 2)` grid: the
Q latent row and the KV latent row. Stock code launches both with one shared
width, `BLOCK_SIZE = next_power_of_2(max(q_size, kv_size))`. The backport
extracts a reusable `_rmsnorm_row` helper and gives each task its own
compile-time width (`Q_BLOCK = next_power_of_2(q_size)`,
`KV_BLOCK = next_power_of_2(kv_size)`).

The text transform is byte-faithful to upstream: applying it to upstream's
pre-image (`vllm-project/vllm` @ `8688a06`, sha256 `8ea5fd82ab09db66...`)
reproduces upstream's post-image (`419d610a`, sha256 `cb5262282376c5c4...`)
byte for byte. Both blobs are frozen under `tests/fixtures/vllm-49283/` and the
equivalence is re-asserted on every CPU CI run by
`scripts/test-qk-rmsnorm-split-blocks.py`.

WHY — for DSV4-Flash-0731 specifically, which is NOT upstream's reason
The caller splits `qr_kv` into `[q_lora_rank, head_dim]`, so `q_size =
q_lora_rank` and `kv_size = head_dim`. `deepseek-ai/DeepSeek-V4-Flash-0731`
(`config.json`) sets `q_lora_rank = 1024` and `head_dim = 512`:

    task | size | stock BLOCK_SIZE      | patched          | change
    Q    | 1024 | next_pow2(1024) = 1024 | Q_BLOCK  = 1024 | none
    KV   |  512 | 1024                   | KV_BLOCK =  512 | 1024 -> 512

On this checkpoint the **KV** row is the narrow one. `Q_BLOCK` is unchanged,
and because 1024 is already a power of two the Q row never had a single masked
lane, so the Q path is bit-identical and gains nothing. The entire effect here
is halving the KV task's reduction width. Upstream's stated rationale (narrow Q
rows paying for KV-wide reductions) and its 1.19-1.34x kernel microbenchmark
are for `q_size, kv_size = 192, 576` on an Intel B60 — the opposite asymmetry
on a different backend — and do not transfer to this deployment.

NUMERICS
Masked lanes load exact `0.0` and contribute exact `0.0` to `sum(x * x)`, so
the mathematical variance is unchanged and the Q row is bitwise unchanged. The
KV row's fp32 summation *order* can change with the block width, because Triton
derives the per-thread element layout from `BLOCK`; bitwise equality on the KV
row is therefore NOT claimed. The bound is ~1 ulp fp32 in `variance`, hence
<= 1 ulp in the bf16 store. Do not build a bitwise A/B gate on the KV row.

EVIDENCE STATUS — no end-to-end measurement is claimed
Verified (CPU, in CI): byte-equivalence with upstream's post-image, anchor
uniqueness, idempotence with re-validation, fail-closed refusal on anchor
drift, and a post-write self-check that restores the original bytes on failure.
NOT verified: any serving-level effect. Per token per layer this kernel touches
1024 + 512 bf16 inputs and as many outputs (~6 KB in, ~6 KB out; ~0.5 MB/token
across 43 layers). Decode on this MoE is weight-bandwidth bound by orders of
magnitude, the launch count is unchanged (2 CTAs per layer per step), and only
one of the two tasks narrows, so the expected end-to-end effect is at or below
the noise floor of this repo's probes. Earlier revisions of this file carried
marginal and stacked decode percentages; they were not attributable to this
patch and have been deleted rather than restated. The CHANGELOG entry dated
2026-08-19 records exactly which figures were withdrawn and why.
ALSO NOT verified: that these anchors match the *deployed* image. They are
proven against upstream `8688a06` (2026-07-21); the pinned image tree is
`g752a3a504` (2026-07-14). Run `--dry-run` inside the container to close that
gap without mutating anything.

USAGE — both nodes, every boot
The rewrite lives in the container filesystem
(`/usr/local/lib/python3.12/dist-packages`), which is not a volume: `docker
compose up` recreates the container and discards it. It must be re-applied at
every boot, and on BOTH ranks — a 2x DGX-Spark TP=2 deployment runs one
container per node, and `fused_wqa_wkv` is built with `disable_tp=True`, so
this latent is TP-replicated and one-rank application would leave the ranks
computing ulp-divergent values.

The supported path is the compose entrypoint, default OFF:

    DSPARK_ENABLE_QK_RMSNORM_SPLIT=1 ./start-deepseek-v4-flash-dspark.sh

`start-deepseek-v4-flash-dspark.sh` scp's this file to the worker, and
`docker-compose.dspark.yml` invokes it on each rank only when the flag is
exactly `1`, chained with `|| exit 1` — fail-closed: a boot that cannot apply
and verify the patch does not serve. The default `0` keeps the stock kernel and
never invokes the patcher.

Inspect a live container without mutating it:

    docker exec <c> python3 /opt/hotfix-dsv4-qk-rmsnorm-split-blocks.py --status
    docker exec <c> python3 /opt/hotfix-dsv4-qk-rmsnorm-split-blocks.py --dry-run

MODES AND EXIT CODES
    (default)  apply. Idempotent: an already-applied target is re-validated,
               never double-patched. 0 = applied and verified; 1 = refused
               (anchors drifted / target missing) or self-check failed, in
               which case the original bytes are restored before exiting.
    --dry-run  check all five anchors; never writes. 0 = would apply cleanly,
               or already applied and valid; 2 = drifted / unreadable.
    --status   report applied / not-applied / drifted, plus the target sha256
               and the image's vLLM version; never writes. 0 = applied and
               valid; 1 = not applied (stock, appliable); 2 = drifted /
               unreadable.

The self-check is a structural (AST) validation of the rewritten module, not an
import: it asserts the helper is defined and `@triton.jit`-decorated, that the
kernel takes `Q_BLOCK`/`KV_BLOCK` and no longer takes `BLOCK_SIZE`, that it
calls the helper exactly twice, that the host assigns `q_block`/`kv_block` and
passes them at launch, and that no reference to the removed shared width
survives anywhere in the file. Importing the module instead would pull `torch`
and the whole `vllm` package into the boot chain for a weaker guarantee.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
import re
import sys
from pathlib import Path

MARK = "[qk-rmsnorm-split]"

DEFAULT_VLLM_ROOT = Path(
    os.environ.get("VLLM_ROOT", "/usr/local/lib/python3.12/dist-packages/vllm")
)
TARGET_REL = "models/deepseek_v4/common/ops/fused_qk_rmsnorm.py"

# The five upstream anchors, byte-faithful to vllm-project/vllm#49283. Each OLD
# must occur exactly once in the target; anything else is drift and is refused.
OLD1 = "@triton.jit\ndef _fused_q_kv_rmsnorm_kernel("
NEW1 = """@triton.jit
def _rmsnorm_row(
    row_in,
    weight_ptr,
    row_out,
    eps,
    SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # fp32 throughout, single cast at store. BLOCK is next_pow2(SIZE) so narrow
    # Q rows avoid KV-wide reductions that waste ALU on masked lanes.
    block = tl.arange(0, BLOCK)
    mask = block < SIZE
    x = tl.load(row_in + block, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / SIZE
    rrms = tl.rsqrt(variance + eps)
    w = tl.load(weight_ptr + block, mask=mask, other=0.0).to(tl.float32)
    y = x * rrms * w
    tl.store(row_out + block, y.to(row_out.dtype.element_ty), mask=mask)


@triton.jit
def _fused_q_kv_rmsnorm_kernel("""

OLD2 = "    Q_SIZE: tl.constexpr,\n    KV_SIZE: tl.constexpr,\n    BLOCK_SIZE: tl.constexpr,\n"
NEW2 = "    Q_SIZE: tl.constexpr,\n    KV_SIZE: tl.constexpr,\n    Q_BLOCK: tl.constexpr,\n    KV_BLOCK: tl.constexpr,\n"

OLD3 = """    pid_task = tl.program_id(1)

    if pid_task == 0:
        SIZE = Q_SIZE
        row_in = q_ptr + token_idx * q_in_stride
        weight_ptr = q_weight_ptr
        row_out = q_out_ptr + token_idx * q_out_stride
    else:
        SIZE = KV_SIZE
        row_in = kv_ptr + token_idx * kv_in_stride
        weight_ptr = kv_weight_ptr
        row_out = kv_out_ptr + token_idx * kv_out_stride

    # RMSNorm in fp32 throughout — matches csrc/layernorm_kernels.cu's
    # `(scalar_t)(x * s_variance * w)` and DeepseekV4's compressor kernel, which
    # keep x, rrms, and w all in fp32 and perform a single cast at store.
    block = tl.arange(0, BLOCK_SIZE)
    mask = block < SIZE
    x = tl.load(row_in + block, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / SIZE
    rrms = tl.rsqrt(variance + eps)
    w = tl.load(weight_ptr + block, mask=mask, other=0.0).to(tl.float32)
    y = x * rrms * w
    tl.store(row_out + block, y.to(row_out.dtype.element_ty), mask=mask)"""
NEW3 = """    pid_task = tl.program_id(1)

    if pid_task == 0:
        _rmsnorm_row(
            q_ptr + token_idx * q_in_stride,
            q_weight_ptr,
            q_out_ptr + token_idx * q_out_stride,
            eps,
            Q_SIZE,
            Q_BLOCK,
        )
    else:
        _rmsnorm_row(
            kv_ptr + token_idx * kv_in_stride,
            kv_weight_ptr,
            kv_out_ptr + token_idx * kv_out_stride,
            eps,
            KV_SIZE,
            KV_BLOCK,
        )"""

OLD4 = "    block_size = triton.next_power_of_2(max(q_size, kv_size))"
NEW4 = "    q_block = triton.next_power_of_2(q_size)\n    kv_block = triton.next_power_of_2(kv_size)"

OLD5 = "        BLOCK_SIZE=block_size,"
NEW5 = "        Q_BLOCK=q_block,\n        KV_BLOCK=kv_block,"

HUNKS: tuple[tuple[str, str], ...] = (
    (OLD1, NEW1),
    (OLD2, NEW2),
    (OLD3, NEW3),
    (OLD4, NEW4),
    (OLD5, NEW5),
)

# setuptools_scm emits `__version__ = version = '...'`, so allow the chain.
_VERSION_RE = re.compile(
    r"^__version__\s*=\s*(?:\w+\s*=\s*)*['\"]([^'\"]+)['\"]", re.M
)


def _dotted(node: ast.expr) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _module_functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def structure_errors(src: str) -> list[str]:
    """Every property the five-hunk transform must produce, checked on the AST.

    Token presence (`"Q_BLOCK" in src`) is spoofable by a comment and cannot
    tell a partially applied file from a finished one; the AST can.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError as err:
        return [f"patched source does not parse: {err}"]

    errors: list[str] = []
    funcs = _module_functions(tree)

    helper = funcs.get("_rmsnorm_row")
    if helper is None:
        errors.append("_rmsnorm_row helper is not defined at module level")
    else:
        if not any(_dotted(d) == "triton.jit" for d in helper.decorator_list):
            errors.append("_rmsnorm_row is not decorated with @triton.jit")
        tail = [a.arg for a in helper.args.args][-2:]
        if tail != ["SIZE", "BLOCK"]:
            errors.append(f"_rmsnorm_row trailing params are {tail}, expected ['SIZE', 'BLOCK']")

    kernel = funcs.get("_fused_q_kv_rmsnorm_kernel")
    if kernel is None:
        errors.append("_fused_q_kv_rmsnorm_kernel is not defined at module level")
    else:
        params = [a.arg for a in kernel.args.args]
        for want in ("Q_BLOCK", "KV_BLOCK"):
            if want not in params:
                errors.append(f"kernel signature is missing {want}")
        if "BLOCK_SIZE" in params:
            errors.append("kernel signature still takes the shared BLOCK_SIZE")
        helper_params = len(helper.args.args) if helper is not None else 6
        calls = [
            n
            for n in ast.walk(kernel)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_rmsnorm_row"
        ]
        if len(calls) != 2:
            errors.append(
                f"kernel calls _rmsnorm_row {len(calls)}x, expected 2 (Q and KV)"
            )
        for call in calls:
            if call.keywords or len(call.args) != helper_params:
                errors.append(
                    f"_rmsnorm_row call on line {call.lineno} passes "
                    f"{len(call.args)} positional args, expected {helper_params}"
                )

    host = funcs.get("fused_q_kv_rmsnorm")
    if host is None:
        errors.append("fused_q_kv_rmsnorm host function is not defined at module level")
    else:
        assigned = {
            t.id
            for n in ast.walk(host)
            if isinstance(n, ast.Assign)
            for t in n.targets
            if isinstance(t, ast.Name)
        }
        for want in ("q_block", "kv_block"):
            if want not in assigned:
                errors.append(f"host function never assigns {want}")
        launch_kwargs = {
            kw.arg
            for n in ast.walk(host)
            if isinstance(n, ast.Call)
            for kw in n.keywords
        }
        for want in ("Q_BLOCK", "KV_BLOCK"):
            if want not in launch_kwargs:
                errors.append(f"kernel launch does not pass {want}=")
        if "BLOCK_SIZE" in launch_kwargs:
            errors.append("kernel launch still passes BLOCK_SIZE=")

    dangling = sorted(
        {
            n.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Name) and n.id in ("block_size", "BLOCK_SIZE")
        }
    )
    if dangling:
        errors.append(
            "removed shared-block names are still referenced: " + ", ".join(dangling)
        )
    return errors


def anchor_counts(src: str) -> list[int]:
    return [src.count(old) for old, _ in HUNKS]


def classify(src: str) -> tuple[str, list[str], list[int]]:
    """`applied` (structurally valid), `stock` (all five anchors unique), or
    `drifted` (everything else, including partially applied targets)."""
    errors = structure_errors(src)
    counts = anchor_counts(src)
    if not errors:
        return "applied", errors, counts
    if all(count == 1 for count in counts):
        return "stock", errors, counts
    return "drifted", errors, counts


def apply_hunks(src: str) -> str:
    for old, new in HUNKS:
        src = src.replace(old, new, 1)
    return src


def _anchor_report(counts: list[int]) -> str:
    return " ".join(f"anchor{i}={c}" for i, c in enumerate(counts, 1))


def _vllm_version(root: Path) -> str:
    """Read the image's vLLM version textually; importing vllm here would cost
    the whole dependency graph for a diagnostic line."""
    for name in ("_version.py", "version.py"):
        try:
            text = (root / name).read_text(encoding="utf-8")
        except OSError:
            continue
        found = _VERSION_RE.search(text)
        if found:
            return found.group(1)
    return "unknown"


def _read(target: Path) -> str | None:
    try:
        return target.read_text(encoding="utf-8")
    except OSError as err:
        print(f"[FAIL] {MARK} cannot read {target}: {err}", file=sys.stderr)
        return None


def _explain(errors: list[str]) -> None:
    for err in errors:
        print(f"[FAIL] {MARK}   {err}", file=sys.stderr)


def cmd_status(target: Path, root: Path) -> int:
    src = _read(target)
    if src is None:
        return 2
    state, errors, counts = classify(src)
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()
    print(f"{MARK} target   {target}")
    print(f"{MARK} sha256   {digest}")
    print(f"{MARK} vllm     {_vllm_version(root)}")
    print(f"{MARK} anchors  {_anchor_report(counts)}")
    if state == "applied":
        print(f"{MARK} status   applied")
        return 0
    if state == "stock":
        print(f"{MARK} status   not-applied (stock; all five anchors unique)")
        return 1
    print(f"{MARK} status   drifted", file=sys.stderr)
    _explain(errors)
    return 2


def cmd_dry_run(target: Path) -> int:
    src = _read(target)
    if src is None:
        return 2
    state, errors, counts = classify(src)
    print(f"{MARK} dry-run  {target}")
    print(f"{MARK} anchors  {_anchor_report(counts)}")
    if state == "applied":
        print(f"[OK] {MARK} dry-run: already applied and valid, nothing to do")
        return 0
    if state == "stock":
        # A real prediction, not just an anchor tally: run the transform in
        # memory and self-check the result, so a target that would trip the
        # post-write restore is reported here instead of at boot.
        would_be = structure_errors(apply_hunks(src))
        if not would_be:
            print(f"[OK] {MARK} dry-run: all five anchors unique, would apply cleanly")
            return 0
        print(
            f"[FAIL] {MARK} dry-run: anchors match but the result fails the "
            "self-check, apply would refuse and restore",
            file=sys.stderr,
        )
        _explain(would_be)
        return 2
    print(f"[FAIL] {MARK} dry-run: drifted, would refuse to patch", file=sys.stderr)
    _explain(errors)
    return 2


def cmd_apply(target: Path) -> int:
    src = _read(target)
    if src is None:
        # Invoked == gated ON: an unreadable target is a prerequisite failure,
        # not a skip (compose chains this with `|| exit 1`).
        return 1
    state, errors, counts = classify(src)
    if state == "applied":
        print(f"[OK] {MARK} already applied and re-validated: {target}")
        return 0
    if state == "drifted":
        print(
            f"[FAIL] {MARK} refusing to patch {target} "
            f"({_anchor_report(counts)}); nothing was written",
            file=sys.stderr,
        )
        _explain(errors)
        return 1

    patched = apply_hunks(src)
    target.write_text(patched, encoding="utf-8")
    errors = structure_errors(patched)
    if not errors:
        print(f"[OK] {MARK} patched and verified: {target}")
        return 0

    # Fail closed: never leave a written-but-unverified kernel behind.
    target.write_text(src, encoding="utf-8")
    print(
        f"[FAIL] {MARK} self-check failed, original bytes restored ({target}):",
        file=sys.stderr,
    )
    _explain(errors)
    return 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="hotfix-dsv4-qk-rmsnorm-split-blocks.py",
        description=(
            "Give the fused Q/KV RMSNorm kernel per-task block widths "
            "(backport of upstream vLLM PR #49283). Default action applies the "
            "patch; --status and --dry-run never write."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--status",
        action="store_true",
        help="report applied/not-applied/drifted, sha256 and vLLM version; exit 0/1/2",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="check all five anchors without writing; exit 0/2",
    )
    where = parser.add_mutually_exclusive_group()
    where.add_argument(
        "--root",
        type=Path,
        help=f"vLLM package root (default {DEFAULT_VLLM_ROOT}); target is <root>/{TARGET_REL}",
    )
    where.add_argument(
        "--target",
        type=Path,
        help="operate on this file directly instead of deriving it from --root",
    )
    args = parser.parse_args(argv[1:])

    root = args.root if args.root is not None else DEFAULT_VLLM_ROOT
    target = args.target if args.target is not None else root / TARGET_REL

    if args.status:
        return cmd_status(target, root)
    if args.dry_run:
        return cmd_dry_run(target)
    return cmd_apply(target)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
