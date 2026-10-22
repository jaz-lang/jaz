"""Interaction protocols — the codec between the LLM and the REPL.

One of the three pluggable components of an agent loop (BaseREPL × BaseProtocol ×
BaseLLM). Selected by configuring an instance: ``jaz.configure(protocol=...)``.

A protocol both decodes an LLM turn (``parse``) and encodes the messages sent back to
it (``render_observation`` / ``render_initial_message_list``).
"""

# Design/history: the three-component decomposition is specified in
# ``design/design_features/basics.md``. The seam was introduced in #566, ``parse`` (decode)
# first and the encode primitives in later PRs of the same stack.

# Public surface: base + default concrete protocol only (see the uniform-surface note in
# ``jaz.repl.__init__``).
from ._base import BaseProtocol
from ._code_only import CodeOnlyProtocol

__all__ = [
    "BaseProtocol",
    "CodeOnlyProtocol",
]
