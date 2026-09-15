#!/usr/bin/env python3
"""Opt-in workaround for issue #141 sparse-MLA verify-decode hangs.

The pinned Anemll 0.1.1 SM120 adapter sends every verify row through one
FlashInfer call. FlashInfer dispatches calls of at most 64 rows to its
standalone DSv4 decode kernel and larger calls to the paged/prefill
orchestrator implicated by the stochastic TP=2 stalls reported in issue #141.

This startup patch source-locks the complete Anemll adapter method and the
relevant pinned FlashInfer call/dispatch contracts. It leaves the <=64-row
call unchanged in semantics and splits only larger calls into ordered views of
at most 64 rows. It is a workaround, not a root-cause fix.

Production usage takes no path arguments. Tests may use the repository's
established explicit positional-target convention:

    hotfix-dsv4-issue141-sparse-mla-decode-chunk.py [TARGET CORE SPARSE]
    hotfix-dsv4-issue141-sparse-mla-decode-chunk.py --status [TARGET CORE SPARSE]
"""
from __future__ import annotations

import os
import stat
import sys
import tempfile
from pathlib import Path

DEFAULT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/"
    "nvidia/flashinfer_sparse.py"
)
DEFAULT_CORE = Path(
    "/usr/local/lib/python3.12/dist-packages/flashinfer/mla/_core.py"
)
DEFAULT_SPARSE = Path(
    "/usr/local/lib/python3.12/dist-packages/flashinfer/mla/"
    "_sparse_mla_sm120.py"
)
MARK = "# [issue141-hotfix] SM120 DSv4 decode/prefill cutoff is 64 rows."

# Complete _forward_decode method from Anemll/dspark-vllm-gx10 revision
# 47503f8e38dadd4dededca798150db2619594fce. The Anemll-only
# _pad_decode_sparse_indices call is deliberately part of the lock.
OLD_METHOD = '''    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        extra_sparse_indices = None
        extra_sparse_lengths = None
        if not swa_only:
            if attn_metadata is None:
                raise RuntimeError(
                    "Sparse MLA metadata is required for compressed layers."
                )
            if swa_metadata.is_valid_token is None:
                raise RuntimeError(
                    "SWA validity metadata is required for compressed layers."
                )
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                if self.topk_indices_buffer is None:
                    raise RuntimeError(
                        "C4A decode requires top-k indices from the indexer."
                    )
                block_size = attn_metadata.block_size // self.compress_ratio
                global_indices, extra_sparse_lengths = (
                    compute_global_topk_indices_and_lens(
                        self.topk_indices_buffer[:num_decode_tokens],
                        swa_metadata.token_to_req_indices,
                        attn_metadata.block_table[:num_decodes],
                        block_size,
                        is_valid,
                    )
                )
                extra_sparse_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                extra_sparse_indices = attn_metadata.c128a_global_decode_topk_indices
                extra_sparse_lengths = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None
        assert swa_lens is not None
        swa_indices = self._pad_decode_sparse_indices(swa_indices)
        q = self._prepare_query(q, output)
        swa_cache = self._as_sparse_cache(self.swa_cache_layer.kv_cache)
        extra_cache = self._as_sparse_cache(kv_cache) if kv_cache is not None else None
        if extra_cache is not None and extra_sparse_indices is None:
            raise RuntimeError(
                "Compressed sparse MLA decode requires compressed sparse indices."
            )
        flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
            query=q,
            swa_kv_cache=swa_cache,
            workspace_buffer=self._get_workspace(q.device),
            sparse_indices=swa_indices,
            compressed_kv_cache=extra_cache,
            out=output,
            bmm1_scale=self.scale,
            sinks=self.attn_sink,
            kv_layout="NHD",
            swa_topk_lens=swa_lens,
            extra_sparse_indices=extra_sparse_indices,
            extra_sparse_topk_lens=extra_sparse_lengths,
        )
'''

NEW_METHOD = '''    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        extra_sparse_indices = None
        extra_sparse_lengths = None
        if not swa_only:
            if attn_metadata is None:
                raise RuntimeError(
                    "Sparse MLA metadata is required for compressed layers."
                )
            if swa_metadata.is_valid_token is None:
                raise RuntimeError(
                    "SWA validity metadata is required for compressed layers."
                )
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                if self.topk_indices_buffer is None:
                    raise RuntimeError(
                        "C4A decode requires top-k indices from the indexer."
                    )
                block_size = attn_metadata.block_size // self.compress_ratio
                global_indices, extra_sparse_lengths = (
                    compute_global_topk_indices_and_lens(
                        self.topk_indices_buffer[:num_decode_tokens],
                        swa_metadata.token_to_req_indices,
                        attn_metadata.block_table[:num_decodes],
                        block_size,
                        is_valid,
                    )
                )
                extra_sparse_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                extra_sparse_indices = attn_metadata.c128a_global_decode_topk_indices
                extra_sparse_lengths = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None
        assert swa_lens is not None
        swa_indices = self._pad_decode_sparse_indices(swa_indices)
        q = self._prepare_query(q, output)
        swa_cache = self._as_sparse_cache(self.swa_cache_layer.kv_cache)
        extra_cache = self._as_sparse_cache(kv_cache) if kv_cache is not None else None
        if extra_cache is not None and extra_sparse_indices is None:
            raise RuntimeError(
                "Compressed sparse MLA decode requires compressed sparse indices."
            )
        workspace = self._get_workspace(q.device)
        if num_decode_tokens <= 64:
            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
                query=q,
                swa_kv_cache=swa_cache,
                workspace_buffer=workspace,
                sparse_indices=swa_indices,
                compressed_kv_cache=extra_cache,
                out=output,
                bmm1_scale=self.scale,
                sinks=self.attn_sink,
                kv_layout="NHD",
                swa_topk_lens=swa_lens,
                extra_sparse_indices=extra_sparse_indices,
                extra_sparse_topk_lens=extra_sparse_lengths,
            )
            return

        # [issue141-hotfix] SM120 DSv4 decode/prefill cutoff is 64 rows.
        for row_start in range(0, num_decode_tokens, 64):
            rows = slice(row_start, min(row_start + 64, num_decode_tokens))
            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
                query=q[rows],
                swa_kv_cache=swa_cache,
                workspace_buffer=workspace,
                sparse_indices=swa_indices[rows],
                compressed_kv_cache=extra_cache,
                out=output[rows],
                bmm1_scale=self.scale,
                sinks=self.attn_sink,
                kv_layout="NHD",
                swa_topk_lens=swa_lens[rows],
                extra_sparse_indices=(
                    extra_sparse_indices[rows]
                    if extra_sparse_indices is not None
                    else None
                ),
                extra_sparse_topk_lens=(
                    extra_sparse_lengths[rows]
                    if extra_sparse_lengths is not None
                    else None
                ),
            )
'''

# Exact load-bearing source locks from FlashInfer revision
# 0472b9b3f2fba11b463f8526f390297d52a8aad7: the 64-row cutoff in both files,
# the dispatch predicate, the sliced call signature, and the custom-op
# mutation contract that excludes both KV caches. Each must occur once.
CORE_GUARDS = (
    (
        "decode-workspace-64-cutoff",
        '''def _sparse_mla_decode_workspace(
    workspace_buffer: torch.Tensor,
    *,
    num_tokens: int,
    num_heads: int,
    d_v: int,
    topk: int,
    extra_topk: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if num_tokens > 64:
        return None, None
    split_tile = 64
''',
    ),
    (
        "sm120-adapter-call-shape",
        '''def _trtllm_batch_decode_sparse_mla_dsv4_sm120(
    *,
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    sparse_indices: torch.Tensor,
    compressed_kv_cache: Optional[torch.Tensor],
    swa_topk_lens: torch.Tensor,
    extra_sparse_indices: Optional[torch.Tensor],
    extra_sparse_topk_lens: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    bmm1_scale: float,
    bmm2_scale: float,
    sinks: Optional[torch.Tensor],
    kv_layout: Literal["HND", "NHD"],
) -> torch.Tensor:
''',
    ),
)

SPARSE_GUARDS = (
    (
        "decode-cutoff-definition",
        '''# Decode/prefill cutoff: num_tokens > _DECODE_MAX_TOKENS routes to the
# prefill orchestrator; otherwise to the standalone decode kernels.
_DECODE_MAX_TOKENS = 64
''',
    ),
    (
        "dsv4-dispatch-predicate",
        '''def _decode_dsv4_dispatchable(
    num_tokens: int,
    num_heads: int,
    topk: int,
    d_qk: int,
    page_block_size: int,
    extra_topk: int = 0,
) -> bool:
    """True iff decode-dsv4 supports this shape configuration.

    The split count only affects scratch size; the merge kernel stores per-split
    LSE in dynamic shared memory.
    """
    return (
        num_tokens <= _DECODE_MAX_TOKENS
        and d_qk == 512
        and page_block_size == _DECODE_DSV4_PAGE_BLOCK_SIZE
        and (num_heads, topk) in _DECODE_DSV4_DISPATCH
    )
''',
    ),
    (
        "custom-op-mutation-contract",
        '''    @register_custom_op(
        "flashinfer::sparse_mla_sm120_paged_attention",
        mutates_args=("output", "out_lse", "mid_out", "mid_lse"),
    )
''',
    ),
)


class PatchError(RuntimeError):
    """An enabled patch cannot safely accept or publish the installed source."""


def _read_source(path: Path, label: str) -> tuple[bytes, str]:
    if not path.is_file():
        raise PatchError(f"missing {label}: {path}")
    try:
        raw = path.read_bytes()
        return raw, raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise PatchError(f"cannot read {label} {path}: {exc}") from exc


def _check_guards(source: str, guards: tuple[tuple[str, str], ...], label: str) -> None:
    failures = []
    offsets = []
    for name, fragment in guards:
        count = source.count(fragment)
        if count != 1:
            failures.append(f"{name}={count}")
        else:
            offsets.append((name, source.index(fragment)))
    if failures:
        raise PatchError(f"{label} source drift ({', '.join(failures)})")
    # Ordered inventories pin that each fragment still sits at its pinned
    # relative position, rather than merely existing somewhere in the file.
    if any(left[1] >= right[1] for left, right in zip(offsets, offsets[1:])):
        raise PatchError(f"{label} guard order drift")


def validate_flashinfer(core_source: str, sparse_source: str) -> None:
    _check_guards(core_source, CORE_GUARDS, "FlashInfer _core.py")
    _check_guards(sparse_source, SPARSE_GUARDS, "FlashInfer _sparse_mla_sm120.py")
    cutoff_assignments = sparse_source.count("_DECODE_MAX_TOKENS =")
    if cutoff_assignments != 1:
        raise PatchError(
            "FlashInfer _sparse_mla_sm120.py source drift "
            f"(decode-cutoff-assignments={cutoff_assignments})"
        )


def method_state(source: str) -> str:
    old_count = source.count(OLD_METHOD)
    new_count = source.count(NEW_METHOD)
    mark_count = source.count(MARK)
    if (old_count, new_count, mark_count) == (1, 0, 0):
        return "old"
    if (old_count, new_count, mark_count) == (0, 1, 1):
        return "new"
    raise PatchError(
        "adapter source drift "
        f"(old={old_count}, new={new_count}, marker={mark_count})"
    )


def _compile_source(source: str, target: Path) -> None:
    try:
        compile(source, str(target), "exec")
    except (SyntaxError, ValueError, TypeError) as exc:
        raise PatchError(f"adapter source does not compile: {exc}") from exc


def _stage_bytes(target: Path, data: bytes, mode: int, tag: str) -> Path:
    fd = -1
    staged: Path | None = None
    try:
        fd, name = tempfile.mkstemp(
            prefix=f".{target.name}.{tag}.",
            dir=target.parent,
        )
        staged = Path(name)
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        return staged
    except BaseException:
        if fd >= 0:
            os.close(fd)
        if staged is not None:
            staged.unlink(missing_ok=True)
        raise


def _restore_original(target: Path, original: bytes, mode: int) -> None:
    staged = _stage_bytes(target, original, mode, "rollback")
    try:
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)
    if target.read_bytes() != original:
        raise PatchError("rollback byte verification failed")
    if stat.S_IMODE(target.stat().st_mode) != mode:
        raise PatchError("rollback mode verification failed")


def _verify_published(target: Path, expected: bytes, mode: int) -> None:
    committed = target.read_bytes()
    if committed != expected:
        raise PatchError("post-write byte verification failed")
    if stat.S_IMODE(target.stat().st_mode) != mode:
        raise PatchError("post-write mode verification failed")
    try:
        source = committed.decode("utf-8")
    except UnicodeError as exc:
        raise PatchError(f"post-write UTF-8 verification failed: {exc}") from exc
    if method_state(source) != "new":
        raise PatchError("post-write method verification failed")
    _compile_source(source, target)


def _publish(target: Path, original: bytes, updated: bytes, mode: int) -> None:
    staged: Path | None = None
    publish_attempted = False
    try:
        staged = _stage_bytes(target, updated, mode, "issue141")
        publish_attempted = True
        os.replace(staged, target)
        staged = None
        _verify_published(target, updated, mode)
    except BaseException as exc:
        if staged is not None:
            staged.unlink(missing_ok=True)
        restore_needed = publish_attempted
        if restore_needed:
            try:
                restore_needed = (
                    target.read_bytes() != original
                    or stat.S_IMODE(target.stat().st_mode) != mode
                )
            except BaseException:
                restore_needed = True
        rollback_error: BaseException | None = None
        if restore_needed:
            try:
                _restore_original(target, original, mode)
            except BaseException as rollback_exc:
                rollback_error = rollback_exc
        detail = f"atomic publication failed: {type(exc).__name__}: {exc}"
        if rollback_error is not None:
            detail += (
                "; rollback failed: "
                f"{type(rollback_error).__name__}: {rollback_error}"
            )
        raise PatchError(detail) from exc


def patch_paths(target: Path, core: Path, sparse: Path) -> str:
    original, target_source = _read_source(target, "adapter target")
    _, core_source = _read_source(core, "FlashInfer core guard source")
    _, sparse_source = _read_source(sparse, "FlashInfer sparse guard source")
    validate_flashinfer(core_source, sparse_source)

    state = method_state(target_source)
    if state == "new":
        _compile_source(target_source, target)
        return "already applied and verified"

    updated_source = target_source.replace(OLD_METHOD, NEW_METHOD, 1)
    if method_state(updated_source) != "new":
        raise PatchError("internal replacement did not produce the exact new method")
    _compile_source(updated_source, target)
    mode = stat.S_IMODE(target.stat().st_mode)
    _publish(target, original, updated_source.encode("utf-8"), mode)
    return "applied and verified"


def _targets(argv: list[str], offset: int) -> tuple[Path, Path, Path] | None:
    remaining = argv[offset:]
    if not remaining:
        return DEFAULT_TARGET, DEFAULT_CORE, DEFAULT_SPARSE
    if len(remaining) == 3:
        return Path(remaining[0]), Path(remaining[1]), Path(remaining[2])
    return None


def main(argv: list[str]) -> int:
    status_only = len(argv) > 1 and argv[1] == "--status"
    targets = _targets(argv, 2 if status_only else 1)
    if targets is None:
        print(
            f"usage: {argv[0]} [--status] [TARGET CORE SPARSE]",
            file=sys.stderr,
        )
        return 2
    target, core, sparse = targets

    try:
        if status_only:
            _, target_source = _read_source(target, "adapter target")
            _, core_source = _read_source(core, "FlashInfer core guard source")
            _, sparse_source = _read_source(sparse, "FlashInfer sparse guard source")
            validate_flashinfer(core_source, sparse_source)
            state = method_state(target_source)
            _compile_source(target_source, target)
            label = "APPLIED + VERIFIED" if state == "new" else "NOT APPLIED (compatible)"
            print(f"issue141 sparse-MLA decode chunk : {label}")
            return 0

        result = patch_paths(target, core, sparse)
    except PatchError as exc:
        print(f"[issue141-hotfix] refusing enabled boot: {exc}", file=sys.stderr)
        return 1
    print(f"[issue141-hotfix] {result}: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
