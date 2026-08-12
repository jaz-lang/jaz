"""Interaction protocols — the codec between the LLM and the REPL.

One of the three pluggable components of an agent loop (REPL × InteractionProtocol ×
LLM; see ``design/design_features/basics.md`` and issue #566). Selected via
``Config.interaction_protocol`` (a registered name or a direct instance).

Phase 1a introduces the object and the ``parse`` (decode) seam; encode primitives
follow in later PRs.
"""

from .base import InteractionProtocol, ParsedCode
from .default import DefaultProtocol
from .registry import (
    INTERACTION_PROTOCOL_MAP,
    create_protocol,
    register_protocol,
    validate_protocol_name,
)

__all__ = [
    "InteractionProtocol",
    "ParsedCode",
    "DefaultProtocol",
    "INTERACTION_PROTOCOL_MAP",
    "create_protocol",
    "register_protocol",
    "validate_protocol_name",
]
