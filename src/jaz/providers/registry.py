"""LLM backend registration.

A decorator-based registry for :class:`~jaz.providers.llm.LLM` implementations, mirroring the
REPL registry (see :mod:`jaz.repl.registry`). Registering is what lets *authored data* — an
eval YAML, a settings file — name the backend as ``llm={"tag": "mybackend", ...}``, which
:func:`~jaz.instantiate.build_config` compiles into an instance. Code that already has the
instance passes it straight to ``configure`` and needs no tag.
"""

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import LLM


# Global mapping from backend tags to LLM classes. One namespace, one lookup: a tag names
# exactly one class, and `LLM_REGISTRY[tag].from_dict(params)` is the whole instantiation path.
LLM_REGISTRY: dict[str, type["LLM"]] = {}

# Backends whose module is imported only on first use, because their dependency is optional
# (RLM needs the `rlm` package, which importing jaz must not require). Materialized into
# `LLM_REGISTRY` by `resolve_llm`.
#
# This is NOT the old `_NON_PROVIDER_LLM_TAGS`, which was a *second registry with different
# semantics* — a separate lookup, holding import strings instead of classes, consulted by
# separate branches in `create_llm`. This is deferred materialization
# within the one namespace: `known_llm_tags` counts these, `resolve_llm` returns the same class
# object either way, and no caller branches on which side a tag came from.
_LAZY_LLM_MODULES: dict[str, str] = {
    "rlm": "jaz.llm_client_rlm:RLMClient",
}

# Valid backend tags — the same character class as the model "tag/" prefix.
LLM_TAG_RE = re.compile(r"^[a-zA-Z0-9_]+$")


def registered_llm_tags() -> frozenset[str]:
    """Every tag that names a backend, including ones not yet imported."""
    return frozenset(LLM_REGISTRY) | frozenset(_LAZY_LLM_MODULES)


def resolve_llm(tag: str) -> "type[LLM] | None":
    """The class registered under ``tag``, importing an optional backend's module on demand.

    ``None`` when the tag names no backend. An optional backend whose dependency is missing
    raises its own ``ImportError`` — naming the missing package, which is more useful than
    reporting the tag as unknown.
    """
    cls = LLM_REGISTRY.get(tag)
    if cls is not None:
        return cls
    target = _LAZY_LLM_MODULES.get(tag)
    if target is None:
        return None
    import importlib

    # Importing the module is the whole job: a lazy target MUST carry its own `@register_llm`,
    # so the import populates `LLM_REGISTRY` through the decorator. Assigning here as well would
    # be a write to global state from a *lookup*, bypassing `register_llm`'s already-registered
    # guard — dead in the case it exists for (RLMClient self-registers) and load-bearing only for
    # a target that does not, which is exactly when that guard should have fired.
    module_name, _, attr = target.partition(":")
    cls = getattr(importlib.import_module(module_name), attr)
    registered = LLM_REGISTRY.get(tag)
    if registered is None:
        raise RuntimeError(
            f"{target} did not register itself under {tag!r}; a lazy backend must carry "
            f"@register_llm({tag!r})."
        )
    return registered


def register_llm(name: str):
    """Decorator registering an :class:`LLM` subclass under the tag ``name``.

    Args:
        name: The backend tag (e.g. ``"openai"``, ``"mybackend"``). Must match
            ``[a-zA-Z0-9_]+``. It is also the model-string prefix this backend *rejects*: the
            model id is sent verbatim, so ``mybackend/some-model`` is an error naming
            ``some-model`` as the fix — see :meth:`~jaz.providers.llm.LLM.validate_model`.

    Returns:
        A decorator that registers the ``LLM`` subclass and returns it.

    Example::

        import os
        from jaz.providers import LLM, register_llm

        @register_llm("mybackend")
        class MyLLM(LLM):
            def __init__(self, base_url: str | None = None, **retry):
                super().__init__(**retry)
                self.base_url = base_url or os.environ["MYBACKEND_API_BASE"]

            def complete(self, model, messages, **kwargs): ...

        # Nameable in authored data as {"tag": "mybackend", "base_url": ...} — `base_url`
        # reaches `__init__` because it is declared there. In code, just
        # `jaz.configure(llm=MyBackend(base_url=...))`.

    Raises:
        ValueError: If ``name`` is malformed or already registered.
        TypeError: If the decorated class is not an ``LLM`` subclass.
    """
    if not LLM_TAG_RE.match(name):
        raise ValueError(
            f"Invalid LLM tag: {name!r}. LLM tags must match the regex [a-zA-Z0-9_]+"
        )

    def decorator[C: type["LLM"]](llm_class: C) -> C:
        """Register the LLM class in the global registry."""
        # Imported here to avoid a circular dependency with llm.py, which imports nothing from
        # this module (the registry knows about backends; backends do not know about it).
        from .llm import LLM

        if not issubclass(llm_class, LLM):
            raise TypeError(f"LLM class {llm_class.__name__} must inherit from LLM")
        if name in LLM_REGISTRY:
            raise ValueError(
                f"LLM tag {name!r} is already registered "
                f"({LLM_REGISTRY[name].__name__}). Reconfigure a built-in backend by "
                "constructing it — jaz.configure(llm=OpenAILLM(...)) — instead of "
                "re-registering."
            )
        LLM_REGISTRY[name] = llm_class
        # Recorded on the class so the error can tell "your own prefix" apart from "another
        # backend's prefix" — `validate_model` still consults the registry to decide whether a
        # leading segment is a tag at all; this only selects which hint to give. A subclass
        # inherits it, which is the intent: a subclass of `OpenAILLM` still speaks OpenAI.
        llm_class.registry_tag = name
        return llm_class

    return decorator
