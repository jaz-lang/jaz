"""Interaction protocols — the codec between the LLM and the REPL.

One of the three pluggable components of an agent loop (BaseREPL × BaseProtocol ×
BaseLLM; see ``design/design_features/basics.md``). Selected via
``Config.interaction_protocol`` (a registered name or a direct instance).

A protocol both decodes an LLM turn (``parse``) and encodes the messages sent back to
it (``render_observation`` / ``render_initial_message_list``).
"""

# Design/history: the seam was introduced in #566, ``parse`` (decode) first and the encode
# primitives in later PRs of the same stack.

# Public surface: base + default concrete protocol only (see the uniform-surface note in
# ``jaz.repl.__init__``). The registration seam and ``ParsedCode`` are re-exported for
# reachability (internal call sites import them from the package) but kept OUT of ``__all__``
# via the ruff-blessed ``X as X`` re-export idiom — not part of the v1 public promise.
from .base import BaseProtocol
from .base import ParsedCode as ParsedCode
from .code_only import CodeOnlyProtocol
from .registry import INTERACTION_PROTOCOL_MAP as INTERACTION_PROTOCOL_MAP
from .registry import create_protocol as create_protocol
from .registry import register_protocol as register_protocol
from .registry import validate_protocol_name as validate_protocol_name

__all__ = [
    "BaseProtocol",
    "CodeOnlyProtocol",
]
