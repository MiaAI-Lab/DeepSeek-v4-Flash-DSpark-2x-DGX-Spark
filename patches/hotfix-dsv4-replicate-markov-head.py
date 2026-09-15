#!/usr/bin/env python3
"""Replicate DSpark Markov w1/w2 so the sequential draft loop is TP-local.

Stock Anemll 0.1.1 shards ``DSparkMarkovHead`` with ``VocabParallelEmbedding``
(embed all-reduce) and ``ParallelLMHead`` (logits all-gather via
``LogitsProcessor``). The V2 speculator's six-step sequential loop therefore
pays 12 serialized collectives per decode step. Both matrices are 129,280 x 256
bf16 = 66 MB — cheap to replicate. This is the Anemll-image equivalent of
Stage-C ``VLLM_DSPARK_REPLICATE_MARKOV_W1``.

Idempotent. Patches
``vllm/model_executor/models/qwen3_dspark.py`` (shared by the DSV4 draft in
``models/deepseek_v4/nvidia/dspark.py``).
"""
from pathlib import Path
import sys

P = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dspark.py"
)
MARK = "# [dspark-replicate-markov]"
if len(sys.argv) > 1 and sys.argv[1] == "--status":
    status_src = P.read_text() if P.is_file() else ""
    print("dspark replicate Markov head       :",
          "APPLIED" if MARK in status_src else "NOT APPLIED")
    raise SystemExit(0)
src = P.read_text()
if MARK in src:
    print(f"[dspark-replicate-markov] already applied to {P}")
    raise SystemExit(0)

IMPORT_OLD = (
    "from vllm.model_executor.layers.logits_processor import LogitsProcessor\n"
    "from vllm.model_executor.layers.vocab_parallel_embedding import (\n"
    "    ParallelLMHead,\n"
    "    VocabParallelEmbedding,\n"
    ")\n"
)
INIT_OLD = (
    "        super().__init__()\n"
    "        # TODO(ben): profile for which (if any) it makes sense to replicate or TP-shard\n"
    "        self.markov_w1 = VocabParallelEmbedding(\n"
    "            vocab_size, markov_rank, prefix=maybe_prefix(prefix, \"markov_w1\")\n"
    "        )\n"
    "        self.markov_w2 = ParallelLMHead(\n"
    "            draft_vocab_size, markov_rank, prefix=maybe_prefix(prefix, \"markov_w2\")\n"
    "        )\n"
)
BIAS_OLD = (
    "    def bias(self, markov_embed: torch.Tensor, logits_processor) -> torch.Tensor:\n"
    "        \"\"\"Vocab-size transition bias from a Markov embedding ([B, r] -> [B, V]).\"\"\"\n"
    "        return logits_processor(self.markov_w2, markov_embed)\n"
)
assert IMPORT_OLD in src, "dspark-replicate-markov: import anchor not found; refusing to patch"
assert INIT_OLD in src, "dspark-replicate-markov: init anchor not found; refusing to patch"
assert BIAS_OLD in src, "dspark-replicate-markov: bias anchor not found; refusing to patch"

IMPORT_NEW = (
    "from vllm.model_executor.layers.linear import ReplicatedLinear\n"
    + IMPORT_OLD
)
INIT_NEW = (
    "        super().__init__()\n"
    "        # [dspark-replicate-markov] replicate w1/w2 (66 MB each) to drop\n"
    "        # 12 serialized TP collectives per decode step (embed all-reduce +\n"
    "        # logits all-gather x k). Stage-C VLLM_DSPARK_REPLICATE_MARKOV_W1.\n"
    "        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)\n"
    "        self.markov_w2 = ReplicatedLinear(\n"
    "            markov_rank,\n"
    "            draft_vocab_size,\n"
    "            bias=False,\n"
    "            return_bias=False,\n"
    "            prefix=maybe_prefix(prefix, \"markov_w2\"),\n"
    "        )\n"
)
BIAS_NEW = (
    "    def bias(self, markov_embed: torch.Tensor, logits_processor) -> torch.Tensor:\n"
    "        \"\"\"Vocab-size transition bias from a Markov embedding ([B, r] -> [B, V]).\"\"\"\n"
    "        # [dspark-replicate-markov] local F.linear; skip LogitsProcessor all-gather.\n"
    "        out = self.markov_w2(markov_embed)\n"
    "        scale = getattr(logits_processor, \"scale\", 1.0)\n"
    "        if scale != 1.0:\n"
    "            out = out * scale\n"
    "        return out\n"
)
src = src.replace(IMPORT_OLD, IMPORT_NEW, 1)
src = src.replace(INIT_OLD, INIT_NEW, 1)
src = src.replace(BIAS_OLD, BIAS_NEW, 1)
compile(src, "qwen3_dspark.py", "exec")
P.write_text(src)
print(f"[dspark-replicate-markov] patched {P}")
