"""Interaction-protocol registration system.

Decorator-based registry mirroring :mod:`jaz.repl.registry` (REPLs) and
``create_llm`` (LLM backends): a protocol is selected via
``Config.interaction_protocol`` as either a registered *name* (for YAML/env
configurability) or a direct instance.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import InteractionProtocol

# Global mapping from protocol names to InteractionProtocol classes.
INTERACTION_PROTOCOL_MAP: dict[str, type[InteractionProtocol]] = {}

# Valid protocol-name character class, mirroring REPL_LANG_NAME_PAT.
PROTOCOL_NAME_PAT = r"[a-zA-Z0-9_]+"
PROTOCOL_NAME_RE = re.compile(rf"^{PROTOCOL_NAME_PAT}$")


def register_protocol(name: str):
    """Decorator registering an :class:`InteractionProtocol` under ``name``.

    Example::

        from jaz.protocol import InteractionProtocol, register_protocol

        @register_protocol("my_protocol")
        class MyProtocol(InteractionProtocol):
            def parse(self, response_content, repl_language): ...

        # Now nameable in authored data as {"name": "my_protocol"}; in code, pass the
        # instance: jaz.configure(protocol=MyProtocol())
    """
    if not PROTOCOL_NAME_RE.match(name):
        raise ValueError(
            f"Invalid interaction-protocol name: {name!r}. "
            "Protocol names must match the regex [a-zA-Z0-9_]+"
        )

    def decorator[C: type["InteractionProtocol"]](protocol_class: C) -> C:
        # Import here to mirror register_repl; base.py does not import registry, so this
        # is cycle-free either way (the import guards against future import-order shifts).
        from .base import InteractionProtocol

        if not issubclass(protocol_class, InteractionProtocol):
            raise TypeError(
                f"Protocol class {protocol_class.__name__} must inherit from "
                "InteractionProtocol base class"
            )
        if name in INTERACTION_PROTOCOL_MAP:
            raise ValueError(
                f"Interaction protocol '{name}' is already registered "
                f"({INTERACTION_PROTOCOL_MAP[name].__name__})"
            )
        INTERACTION_PROTOCOL_MAP[name] = protocol_class
        return protocol_class

    return decorator


def validate_protocol_name(name: str, field: str = "interaction_protocol") -> None:
    """Raise ``ValueError`` if ``name`` is not a registered protocol.

    Surfaces the error eagerly (e.g. at ``configure`` / ``config_override``) rather than
    deferring to :func:`create_protocol`, mirroring ``validate_llm_tag``.
    """
    if name not in INTERACTION_PROTOCOL_MAP:
        known = ", ".join(sorted(INTERACTION_PROTOCOL_MAP)) or "(none)"
        raise ValueError(f"Unknown {field}: {name!r}. Registered protocols: {known}.")


def create_protocol(
    name: str, params: Mapping[str, Any] | None = None
) -> InteractionProtocol:
    """Instantiate the registered protocol ``name``, configured from ``params``.

    ``REGISTRY[tag].from_dict(params)`` — the same shape as ``create_llm``. Params it does not
    declare as constructor arguments are ignored rather than raising, so a config carrying a
    setting a *custom* protocol doesn't implement still builds.
    """
    validate_protocol_name(name)
    cls = INTERACTION_PROTOCOL_MAP[name]
    # Passed through unfiltered. This used to drop any key the selected protocol did not declare,
    # because `protocol.params` was populated from the config's *full* set of protocol settings —
    # every leaf, whether or not anyone set it — so a custom protocol implementing only some of
    # them would have died on the rest.
    #
    # That is no longer the shape: params are what a caller actually wrote for *this* protocol.
    # So an undeclared key is a typo, and the filter's only remaining effect was to swallow it —
    # `protocol={"max_input_lenth": 10}` was silently ignored rather than rejected.
    return cls.from_dict(params or {})
