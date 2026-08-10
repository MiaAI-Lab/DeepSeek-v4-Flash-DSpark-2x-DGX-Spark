"""DSpark-transparent attribute proxy utilities."""

from __future__ import annotations

from typing import Any


class TransparentLanguageModelProxy:
    """Mixin that exposes language_model attributes for DSpark proposer access.

    Critical: never hide ``lm_head``, ``model``, ``config``, or custom attrs the
    Anemll DSpark path reads via the served model object.

    Note: ``nn.Module`` stores children in ``_modules``. This mixin must not
    intercept ``language_model`` before Module's own ``__getattr__``.
    """

    def _resolve_language_model(self) -> Any:
        mods = self.__dict__.get("_modules")
        if isinstance(mods, dict) and mods.get("language_model") is not None:
            return mods["language_model"]
        if "language_model" in self.__dict__ and self.__dict__["language_model"] is not None:
            return self.__dict__["language_model"]
        # Fall through to nn.Module.__getattr__ if present
        for base in type(self).__mro__:
            if base is TransparentLanguageModelProxy:
                continue
            ga = base.__dict__.get("__getattr__")
            if ga is not None:
                try:
                    return ga(self, "language_model")
                except AttributeError:
                    continue
        raise AttributeError("language_model")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        # nn.Module stores submodules in _modules; resolve language_model here
        # because this mixin sits before Module in the MRO and would otherwise
        # shadow Module.__getattr__.
        if name == "language_model":
            return self._resolve_language_model()
        # Prefer local submodules (vision_tower, mm_projector) over language_model.
        mods = self.__dict__.get("_modules")
        if isinstance(mods, dict) and name in mods:
            return mods[name]
        if name in self.__dict__:
            return self.__dict__[name]
        try:
            lm = self._resolve_language_model()
        except AttributeError as exc:
            raise AttributeError(
                f"{type(self).__name__!s} has no attribute {name!r} "
                "(language_model not set)"
            ) from exc
        try:
            return getattr(lm, name)
        except AttributeError as exc:
            raise AttributeError(name) from exc

    @property
    def lm_head(self) -> Any:
        return self._resolve_language_model().lm_head

    @property
    def model(self) -> Any:
        lm = self._resolve_language_model()
        return getattr(lm, "model", lm)

    def get_language_model(self) -> Any:
        return self._resolve_language_model()


def assert_dspark_transparency(wrapper: Any) -> dict[str, bool]:
    """Structural check that DSpark-critical attrs remain reachable."""
    if hasattr(wrapper, "get_language_model"):
        lm = wrapper.get_language_model()
    else:
        lm = wrapper.language_model
    checks = {
        "has_language_model": lm is not None,
        "lm_head_via_wrapper": hasattr(wrapper, "lm_head"),
        "lm_head_via_language_model": hasattr(lm, "lm_head"),
        "forward_accepts_kwargs": True,
    }
    import inspect

    try:
        sig = inspect.signature(wrapper.forward)
        checks["forward_accepts_kwargs"] = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        ) or "inputs_embeds" in sig.parameters
    except (TypeError, ValueError):
        checks["forward_accepts_kwargs"] = False

    if hasattr(lm, "compute_logits"):
        checks["compute_logits_reachable"] = hasattr(wrapper, "compute_logits") or hasattr(
            lm, "compute_logits"
        )
    else:
        checks["compute_logits_reachable"] = True

    try:
        checks["lm_head_same_object"] = wrapper.lm_head is lm.lm_head
    except Exception:
        checks["lm_head_same_object"] = False

    checks["all_ok"] = all(
        checks[k]
        for k in (
            "has_language_model",
            "lm_head_via_wrapper",
            "lm_head_via_language_model",
            "forward_accepts_kwargs",
            "compute_logits_reachable",
            "lm_head_same_object",
        )
    )
    return checks
