# Frozen upstream evidence for vLLM PR #49283

Two byte-frozen blobs of
`vllm/models/deepseek_v4/common/ops/fused_qk_rmsnorm.py` from
`vllm-project/vllm`. They exist so
`scripts/test-qk-rmsnorm-split-blocks.py` can prove, offline and
deterministically, that `patches/hotfix-dsv4-qk-rmsnorm-split-blocks.py`
reproduces upstream's own post-image byte for byte instead of an
approximation of it.

| file | upstream ref | role | lines | sha256 |
|---|---|---|---:|---|
| `fused_qk_rmsnorm.base.py` | `8688a06d676bda6e9e3ec3a87233dce91c40bb4f` (PR #49283 base, 2026-07-21) | pre-image the five anchors are matched against | 96 | `8ea5fd82ab09db66872be1fbd5e830022bef97f75719a009fef5ed2a9f70fbb8` |
| `fused_qk_rmsnorm.head.py` | `419d610a97f8fe17369a0308f860aa324505d8aa` (PR #49283 head) | post-image the patcher must reproduce exactly | 116 | `cb5262282376c5c4d51e6cc423ff0fb5f4068ea406d8c0b61e2f700764909fb8` |

Retrieved with
`gh api repos/vllm-project/vllm/contents/vllm/models/deepseek_v4/common/ops/fused_qk_rmsnorm.py?ref=<sha>`.
Both hashes are asserted by the test, so silent edits fail CI.

**These files are data, never imported.** They are upstream vLLM sources under
Apache-2.0 and carry their original SPDX headers; do not reformat, relint, or
"fix" them.

Two caveats the test cannot close:

- Upstream deleted this file from `main` on 2026-07-30 (`aeeb36b1f171`,
  "[New model] Kimi K3 (#50000)"), so `main` no longer serves either blob;
  the pinned SHAs above still do.
- The pre-image is upstream at 2026-07-21. The deployed Anemll image tree is
  `0.25.2.dev0+g752a3a504.d20260714` (2026-07-14). Anchor equality against the
  **image** is therefore unproven from this repository — run
  `python3 /opt/hotfix-dsv4-qk-rmsnorm-split-blocks.py --dry-run` inside the
  container to check it without mutating anything.
