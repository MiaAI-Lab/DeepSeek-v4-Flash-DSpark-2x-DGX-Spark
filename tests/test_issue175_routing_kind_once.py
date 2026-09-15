"""Issue #175: classify routing once per model forward, not once per MoE gate.

Run inside the container:
  python3 tests/test_issue175_routing_kind_once.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patches"))

from vision_exp import apply as ap  # noqa: E402
from vision_exp import image_processor as ip  # noqa: E402
from vllm.forward_context import override_forward_context  # noqa: E402

ID = ip.IMAGE_TOKEN_ID
N_MOE_LAYERS = 43
_real_item = torch.Tensor.item
_counter = {"n": 0}
FAILURES = []


def _counting_item(self, *args, **kwargs):
    _counter["n"] += 1
    return _real_item(self, *args, **kwargs)


torch.Tensor.item = _counting_item


def reset():
    _counter["n"] = 0


def syncs():
    return _counter["n"]


def check(name, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
    if not ok:
        FAILURES.append(name)


def raises(name, exc_type, fn):
    try:
        fn()
    except exc_type as exc:
        print(f"  [PASS] {name}: raised {type(exc).__name__}")
        return
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {name}: raised {type(exc).__name__}, want {exc_type.__name__}")
        FAILURES.append(name)
        return
    print(f"  [FAIL] {name}: no exception, want {exc_type.__name__}")
    FAILURES.append(name)


def fake_ctx():
    return types.SimpleNamespace(tag="forward-context")


class FakeLM:
    def __init__(self):
        self._dspark_routing_kind_cell = [None]


def embed_fn():
    def orig(self, input_ids):
        return torch.zeros((int(input_ids.numel()), 4), dtype=torch.float32)

    return ap.make_embed_input_ids(orig)


def gate_reads(ids, kind_cell=None, n=N_MOE_LAYERS):
    return {ip.current_routing_kind(ids, kind_cell=kind_cell) for _ in range(n)}


print("1. text batch: embed + 43 gate reads")
reset()
embed, lm = embed_fn(), FakeLM()
ids = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)
with override_forward_context(None):
    out = embed(lm, ids, multimodal_embeddings=None, is_multimodal=None)
with override_forward_context(fake_ctx()):
    kinds = gate_reads(ids, lm._dspark_routing_kind_cell)
check("text kind", kinds, {"text"})
check("text .item() count", syncs(), 0)
check("text embed shape", tuple(out.shape), (6, 4))

print("2. all-image batch: one existing merge sync, no gate syncs")
reset()
embed, lm = embed_fn(), FakeLM()
ids = torch.full((8,), ID, dtype=torch.long)
with override_forward_context(None):
    embed(
        lm,
        ids,
        multimodal_embeddings=[torch.ones((8, 4))],
        is_multimodal=torch.ones(8, dtype=torch.bool),
    )
with override_forward_context(fake_ctx()):
    kinds = gate_reads(ids, lm._dspark_routing_kind_cell)
check("image kind", kinds, {"image"})
check("image .item() count", syncs(), 1)

print("3. mixed batch: one existing merge sync, no gate syncs")
reset()
embed, lm = embed_fn(), FakeLM()
ids = torch.tensor([7, ID, ID, ID, 9], dtype=torch.long)
with override_forward_context(None):
    embed(
        lm,
        ids,
        multimodal_embeddings=[torch.ones((3, 4))],
        is_multimodal=torch.tensor([False, True, True, True, False]),
    )
with override_forward_context(fake_ctx()):
    kinds = gate_reads(ids, lm._dspark_routing_kind_cell)
check("mixed kind", kinds, {"mixed"})
check("mixed .item() count", syncs(), 1)

print("4. empty multimodal output is text")
reset()
embed, lm = embed_fn(), FakeLM()
ids = torch.tensor([1, 2, 3], dtype=torch.long)
with override_forward_context(None):
    embed(lm, ids, multimodal_embeddings=[], is_multimodal=None)
with override_forward_context(fake_ctx()):
    kinds = gate_reads(ids, lm._dspark_routing_kind_cell)
check("empty-mm kind", kinds, {"text"})
check("empty-mm .item() count", syncs(), 0)

print("5. no ForwardContext keeps token scan behavior")
for name, token_ids in [
    ("text", torch.tensor([1, 2, 3])),
    ("image", torch.full((4,), ID)),
    ("mixed", torch.tensor([1, ID, 2])),
    ("empty", torch.tensor([], dtype=torch.long)),
    ("none", None),
    ("list", [1, ID, 3]),
]:
    reset()
    with override_forward_context(None):
        got = ip.current_routing_kind(token_ids)
    check(f"fallback {name}", got, ip.token_routing_kind(token_ids))

print("6. unpublished warmup/dummy forward scans once")
reset()
ids = torch.tensor([1, ID, 2], dtype=torch.long)
with override_forward_context(fake_ctx()):
    kinds = gate_reads(ids)
check("unpublished kind", kinds, {"mixed"})
check("unpublished syncs", syncs(), 1)

print("7. fresh contexts do not reuse the prior step")
reset()
embed, lm = embed_fn(), FakeLM()
image_ids = torch.full((4,), ID, dtype=torch.long)
with override_forward_context(None):
    embed(
        lm,
        image_ids,
        multimodal_embeddings=[torch.ones((4, 4))],
        is_multimodal=torch.ones(4, dtype=torch.bool),
    )
ctx_a = fake_ctx()
with override_forward_context(ctx_a):
    image_kind = ip.current_routing_kind(image_ids, kind_cell=lm._dspark_routing_kind_cell)
text_ids = torch.tensor([1, 2, 3], dtype=torch.long)
with override_forward_context(None):
    embed(lm, text_ids, multimodal_embeddings=None, is_multimodal=None)
ctx_b = fake_ctx()
with override_forward_context(ctx_b):
    text_kind = ip.current_routing_kind(text_ids, kind_cell=lm._dspark_routing_kind_cell)
check("step A kind", image_kind, "image")
check("step B kind", text_kind, "text")
check("ctx A retained image", getattr(ctx_a, ip._ROUTING_KIND_CTX_ATTR), "image")
check("ctx B retained text", getattr(ctx_b, ip._ROUTING_KIND_CTX_ATTR), "text")

print("8. publication cell is consumed once")
reset()
cell = ["image"]
with override_forward_context(fake_ctx()):
    first = ip.current_routing_kind(torch.full((2,), ID), kind_cell=cell)
with override_forward_context(fake_ctx()):
    second = ip.current_routing_kind(torch.tensor([1, 2]), kind_cell=cell)
check("first consumes image", first, "image")
check("cell empty after consume", cell[0], None)
check("next forward scans its own text", second, "text")
check("only fallback scanned", syncs(), 1)

print("9. model-local cells cannot cross target/drafter or target instances")
reset()
target_a, target_b = FakeLM(), FakeLM()
target_a._dspark_routing_kind_cell[0] = "image"
target_b._dspark_routing_kind_cell[0] = "text"
with override_forward_context(fake_ctx()):
    drafter = ip.current_routing_kind(torch.tensor([1, 2]))
check("untagged drafter scans text", drafter, "text")
check("drafter did not consume target A", target_a._dspark_routing_kind_cell[0], "image")
with override_forward_context(fake_ctx()):
    a = ip.current_routing_kind(torch.full((2,), ID), kind_cell=target_a._dspark_routing_kind_cell)
with override_forward_context(fake_ctx()):
    b = ip.current_routing_kind(torch.tensor([1, 2]), kind_cell=target_b._dspark_routing_kind_cell)
check("target A receives image", a, "image")
check("target B receives text", b, "text")

print("10. CUDA-graph capture exits before consuming the cell")
reset()
calls = {"orig": 0}


class _Gate:
    e_score_correction_bias_vl = torch.zeros(4)


class _Router(nn.Module):
    scoring_func = "sigmoid"
    top_k = 2
    renormalize = True
    routed_scaling_factor = 1.0
    num_fused_shared_experts = 0

    def _compute_routing(self, hidden_states, router_logits, indices_type, *, input_ids=None):
        calls["orig"] += 1
        return ("orig", input_ids)


lm, router = FakeLM(), _Router()
ap._wrap_router_compute_routing(router, _Gate())
router._dspark_routing_kind_cell = lm._dspark_routing_kind_cell
lm._dspark_routing_kind_cell[0] = "image"
real_capturing = torch.cuda.is_current_stream_capturing
torch.cuda.is_current_stream_capturing = lambda: True
try:
    with override_forward_context(fake_ctx()):
        result = router._compute_routing(
            torch.zeros((4, 4)),
            torch.zeros((4, 4)),
            None,
            input_ids=torch.full((4,), ID, dtype=torch.long),
        )
    check("capturing uses original path", result[0], "orig")
    check("capturing does not sync", syncs(), 0)
    check("capturing preserves publication", lm._dspark_routing_kind_cell[0], "image")
finally:
    torch.cuda.is_current_stream_capturing = real_capturing

print("11. non-capturing text wrapper uses original route with zero syncs")
reset()
calls["orig"] = 0
lm._dspark_routing_kind_cell[0] = None
with override_forward_context(None):
    embed_fn()(lm, text_ids, multimodal_embeddings=None, is_multimodal=None)
torch.cuda.is_current_stream_capturing = lambda: False
try:
    with override_forward_context(fake_ctx()):
        for _ in range(N_MOE_LAYERS):
            router._compute_routing(
                torch.zeros((4, 4)), torch.zeros((4, 4)), None, input_ids=text_ids
            )
    check("text original-route calls", calls["orig"], N_MOE_LAYERS)
    check("text wrapper syncs", syncs(), 0)
finally:
    torch.cuda.is_current_stream_capturing = real_capturing

print("12. precomputed fused-topk kind does not re-resolve")
reset()
import vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router as ftb_mod

seen = {}


def fake_ftb(**kwargs):
    seen.update(kwargs)
    return ("w", "ids")


real_ftb, real_current = ftb_mod.fused_topk_bias, ap.current_routing_kind
ftb_mod.fused_topk_bias = fake_ftb
ap.current_routing_kind = lambda *args, **kwargs: (_ for _ in ()).throw(
    AssertionError("precomputed kind must not be resolved")
)
common = dict(
    hidden_states=torch.zeros((2, 2)),
    gating_output=torch.zeros((2, 2)),
    scoring_func="sigmoid",
    e_score_correction_bias=torch.ones(4),
    e_score_correction_bias_vl=torch.zeros(4),
    topk=1,
    renormalize=False,
    indices_type=None,
    hash_indices_table="HASH",
    routed_scaling_factor=1.0,
)
try:
    ap.fused_topk_bias_split_vl(input_tokens=torch.tensor([ID, ID]), kind="image", **common)
    check("image uses Vision-Exp bias", seen["e_score_correction_bias"].data_ptr(), common["e_score_correction_bias_vl"].data_ptr())
    check("image skips hash table", seen["hash_indices_table"], None)
    seen.clear()
    ap.fused_topk_bias_split_vl(input_tokens=torch.tensor([1, 2]), kind="text", **common)
    check("text keeps hash table", seen["hash_indices_table"], "HASH")
    check("precomputed paths do not sync", syncs(), 0)
finally:
    ftb_mod.fused_topk_bias = real_ftb
    ap.current_routing_kind = real_current

print("13. kind=None resolves from a model cell")
reset()
ftb_mod.fused_topk_bias = fake_ftb
try:
    seen.clear()
    with override_forward_context(fake_ctx()):
        ap.fused_topk_bias_split_vl(
            input_tokens=torch.full((2,), ID), kind_cell=["image"], **common
        )
    check("cell image skips hash table", seen["hash_indices_table"], None)
    check("cell resolution does not sync", syncs(), 0)
finally:
    ftb_mod.fused_topk_bias = real_ftb

print("14. hard failures prevent silent performance regressions")
reset()


class _Slotted:
    __slots__ = ()


with override_forward_context(_Slotted()):
    raises(
        "locked ForwardContext",
        AttributeError,
        lambda: ip.current_routing_kind(torch.tensor([1, 2]), kind_cell=["text"]),
    )
raises("unsized token input", TypeError, lambda: ap._num_tokens(object()))
raises("zero-token multimodal classification", ValueError, lambda: ip.routing_kind_from_mm(3, 0))
saved_hooks = ip._FORWARD_CONTEXT_HOOKS
ip._FORWARD_CONTEXT_HOOKS = None
import vllm.forward_context as fc_mod

saved_fn = fc_mod.is_forward_context_available
del fc_mod.is_forward_context_available
try:
    raises("missing vLLM context symbol", AttributeError, ip._forward_context_or_none)
finally:
    fc_mod.is_forward_context_available = saved_fn
    ip._FORWARD_CONTEXT_HOOKS = saved_hooks

print("15. cell attachment follows the model graph, not names")


class _Experts(nn.Module):
    def __init__(self, router):
        super().__init__()
        self.router = router


class _MoE(nn.Module):
    def __init__(self, with_vl):
        super().__init__()
        self.gate = nn.Module()
        self.gate.e_score_correction_bias_vl = torch.zeros(4) if with_vl else None
        self.experts = _Experts(_Router())


class _Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_MoE(True), _MoE(True), _MoE(False)])


stack, cell = _Stack(), [None]
attached = ap.attach_routing_kind_cell(stack, cell)
check("attached only Vision-Exp gates", attached, 2)
check("MoE shares exact cell", stack.layers[0]._dspark_routing_kind_cell is cell, True)
check("router shares exact cell", stack.layers[0].experts.router._dspark_routing_kind_cell is cell, True)
check("non-Vision gate remains untagged", hasattr(stack.layers[2], "_dspark_routing_kind_cell"), False)

print()
if FAILURES:
    print(f"RESULT: {len(FAILURES)} FAILED -> {FAILURES}")
    sys.exit(1)
print("RESULT: ALL PASS")
