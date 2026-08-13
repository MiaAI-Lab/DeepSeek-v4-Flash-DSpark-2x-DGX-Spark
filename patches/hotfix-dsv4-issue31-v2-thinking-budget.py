#!/usr/bin/env python3
"""Hotfix: enable thinking_token_budget on the V2 runner (issue #31).

Upstream vLLM 0.25.2.dev0 rejects thinking_token_budget on the V2 model
runner (HTTP 400). DSpark only exists on V2, so this recipe had no way to
close a reasoning block before max_tokens.

This patch:
  1. Drops the V2 400 in the input processor.
  2. Installs a V2 sampler hook that counts tokens after the last
     reasoning-start marker and, once the budget is exhausted, forces the
     next sampled token(s) to the reasoning-end sequence (</think>).
  3. Threads ReasoningConfig into the V2 Sampler so start/end ids are known.

When the request omits thinking_token_budget, DEFAULT_THINKING_TOKEN_BUDGET
supplies a server-side fallback; unset (the default) keeps the request-only
behaviour. DSpark rejection sampling still runs; forcing is applied to
processed logits before top-k.

Idempotent. Patches files under
/usr/local/lib/python3.12/dist-packages/vllm/
"""
from pathlib import Path

VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARK = "# [issue31-hotfix] v2 thinking_token_budget"

THINKING_BUDGET_PY = r'''# SPDX-License-Identifier: Apache-2.0
# [issue31-hotfix] V2 thinking_token_budget (not upstream)
"""Force reasoning-end tokens when thinking_token_budget is exhausted."""

from __future__ import annotations

import os

import numpy as np
import torch

from vllm.sampling_params import SamplingParams


def _default_budget() -> int | None:
    """Server-side fallback for clients that cannot send the field.

    The OpenAI SDK can pass thinking_token_budget via extra_body, but many
    agent harnesses and most chat UIs never expose it, so a request-only knob
    leaves those deployments unprotected.
    Empty, unparseable, or <= 0 means no default, i.e. previous behaviour.

    A request can still opt out explicitly with a negative budget, which
    apply() already treats as unlimited. JSON null is indistinguishable from
    an omitted field and therefore takes the default.
    """
    raw = os.environ.get("DEFAULT_THINKING_TOKEN_BUDGET", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


DEFAULT_BUDGET = _default_budget()


def _last_subseq(seq: list[int], pat: list[int]) -> int:
    if not pat or len(seq) < len(pat):
        return -1
    last = -1
    plen = len(pat)
    for i in range(0, len(seq) - plen + 1):
        if seq[i : i + plen] == pat:
            last = i
    return last


class ThinkingBudgetState:
    def __init__(self, req_states, reasoning_config, max_num_reqs: int):
        self.req_states = req_states
        self.budget = np.full(max_num_reqs, -1, dtype=np.int32)
        start = None
        end = None
        if reasoning_config is not None:
            start = reasoning_config.reasoning_start_token_ids
            end = reasoning_config.reasoning_end_token_ids
        self.start_ids = list(start or [])
        self.end_ids = list(end or [])
        self.enabled = bool(self.end_ids)

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        b = getattr(sampling_params, "thinking_token_budget", None)
        if b is None:
            b = DEFAULT_BUDGET
        self.budget[req_idx] = -1 if b is None else int(b)

    def apply(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        if not self.enabled or not np.any(self.budget[idx_mapping_np] >= 0):
            return

        req_rows = expanded_idx_mapping.detach().to("cpu").tolist()
        local_pos = expanded_local_pos.detach().to("cpu").tolist()
        token_buf = self.req_states.all_token_ids
        if hasattr(token_buf, "_uva_buf"):
            token_cpu = token_buf._uva_buf.cpu
        else:
            token_cpu = None

        force_rows: list[int] = []
        force_toks: list[int] = []
        for row, (req_idx, lpos) in enumerate(zip(req_rows, local_pos)):
            req_idx = int(req_idx)
            if req_idx < 0:
                continue
            budget = int(self.budget[req_idx])
            if budget < 0:
                continue
            prefill_len = int(self.req_states.prefill_len.np[req_idx])
            n = int(self.req_states.num_computed_tokens_np[req_idx])
            if n < prefill_len:
                continue
            if token_cpu is not None:
                seq = token_cpu[req_idx, :n].tolist()
            else:
                seq = token_buf.gpu[req_idx, :n].detach().to("cpu").tolist()
            seq = [int(x) for x in seq]
            last_start = _last_subseq(seq, self.start_ids)
            last_end = _last_subseq(seq, self.end_ids)
            if last_start < 0:
                continue
            if last_end > last_start:
                continue
            think_count = n - (last_start + len(self.start_ids))
            # This logit is the next generated token at offset lpos in the
            # draft+bonus window (0 = first new token this step).
            if think_count + int(lpos) < budget:
                continue
            overflow = think_count + int(lpos) - budget
            end_i = min(overflow, len(self.end_ids) - 1)
            force_rows.append(row)
            force_toks.append(int(self.end_ids[end_i]))

        if not force_rows:
            return
        # Wipe the forced rows then pin the end token so later top-k cannot
        # drop it.
        rows = torch.tensor(force_rows, device=logits.device, dtype=torch.long)
        toks = torch.tensor(force_toks, device=logits.device, dtype=torch.long)
        logits[rows] = float("-inf")
        logits[rows, toks] = 1.0e9
'''


def _patch_file(path: Path, old: str, new: str, what: str) -> None:
    src = path.read_text()
    # Must test the full replacement, not just MARK: several hunks share one
    # file, and MARK is written by the first hunk.
    if new in src:
        print(f"[issue31-hotfix] already applied: {what} ({path})")
        return
    if old not in src:
        raise SystemExit(f"[issue31-hotfix] anchor not found for {what} in {path}")
    path.write_text(src.replace(old, new, 1))
    print(f"[issue31-hotfix] patched {what} in {path}")


def main() -> None:
    dest = VLLM / "v1/worker/gpu/sample/thinking_budget.py"
    dest.write_text(THINKING_BUDGET_PY)
    print(f"[issue31-hotfix] wrote {dest}")

    _patch_file(
        VLLM / "v1/engine/input_processor.py",
        """                if self.use_v2_model_runner:
                    raise ValueError(
                        "thinking_token_budget is not yet supported by the V2 "
                        "model runner. Run vLLM with VLLM_USE_V2_MODEL_RUNNER=0 "
                        "to use thinking_token_budget."
                    )""",
        f"""                if self.use_v2_model_runner:
                    # {MARK}: implemented in V2 Sampler (DSpark).
                    pass""",
        "input_processor V2 gate",
    )

    sampler = VLLM / "v1/worker/gpu/sample/sampler.py"
    _patch_file(
        sampler,
        """from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS, SamplingStates
from vllm.v1.worker.gpu.states import RequestState""",
        f"""from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS, SamplingStates
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState  # {MARK}""",
        "sampler import",
    )
    _patch_file(
        sampler,
        """        num_speculative_tokens: int = 1,
        use_fp64_gumbel: bool = False,
    ):""",
        f"""        num_speculative_tokens: int = 1,
        use_fp64_gumbel: bool = False,
        reasoning_config=None,  # {MARK}
    ):""",
        "sampler init signature",
    )
    _patch_file(
        sampler,
        """        self.num_speculative_tokens = num_speculative_tokens
        self.use_flashinfer = flashinfer_sampler_supported()""",
        f"""        self.num_speculative_tokens = num_speculative_tokens
        self.use_flashinfer = flashinfer_sampler_supported()
        self.thinking_budget_state = ThinkingBudgetState(  # {MARK}
            req_states, reasoning_config, max_num_reqs
        )""",
        "sampler thinking state",
    )
    _patch_file(
        sampler,
        """        self.sampling_states.add_request(req_idx, sampling_params)
        self.penalties_state.add_request(req_idx, sampling_params)""",
        f"""        self.sampling_states.add_request(req_idx, sampling_params)
        self.thinking_budget_state.add_request(req_idx, sampling_params)  # {MARK}
        self.penalties_state.add_request(req_idx, sampling_params)""",
        "sampler add_request",
    )
    _patch_file(
        sampler,
        """        # Apply min_p in place.
        self.sampling_states.apply_min_p(logits, expanded_idx_mapping, idx_mapping_np)

        if skip_top_k_top_p:
            return logits""",
        f"""        # Apply min_p in place.
        self.sampling_states.apply_min_p(logits, expanded_idx_mapping, idx_mapping_np)

        # {MARK}: force </think> before top-k so the token cannot be dropped.
        self.thinking_budget_state.apply(
            logits, expanded_idx_mapping, expanded_local_pos, idx_mapping_np
        )

        if skip_top_k_top_p:
            return logits""",
        "sampler apply",
    )

    cfg = VLLM / "config/vllm.py"
    _patch_file(
        cfg,
        """        if self.reasoning_config is not None:
            logger.warning_once(
                "Model Runner V2 does not yet support the thinking_token_budget "
                "request parameter. Set VLLM_USE_V2_MODEL_RUNNER=0 if this is required."
            )""",
        f"""        if self.reasoning_config is not None:
            # {MARK}: budget is implemented in the V2 Sampler.
            pass""",
        "vllm.py stale V2 warning",
    )

    runner = VLLM / "v1/worker/gpu/model_runner.py"
    _patch_file(
        runner,
        """            self.sampler = Sampler(
                max_num_reqs=self.max_num_reqs,
                vocab_size=self.vocab_size,
                device=self.device,
                req_states=self.req_states,
                logprobs_mode=self.model_config.logprobs_mode,
                num_speculative_tokens=self.decode_query_len,
                use_fp64_gumbel=self.model_config.use_fp64_gumbel,
            )""",
        f"""            self.sampler = Sampler(
                max_num_reqs=self.max_num_reqs,
                vocab_size=self.vocab_size,
                device=self.device,
                req_states=self.req_states,
                logprobs_mode=self.model_config.logprobs_mode,
                num_speculative_tokens=self.decode_query_len,
                use_fp64_gumbel=self.model_config.use_fp64_gumbel,
                reasoning_config=self.vllm_config.reasoning_config,  # {MARK}
            )""",
        "model_runner Sampler kwarg",
    )
    print("[issue31-hotfix] done")


if __name__ == "__main__":
    main()
