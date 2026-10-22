"""The chat-message alias, in a module with no ``jaz`` imports.

Deliberately dependency-free. ``jaz._tokens`` annotates with this name and is itself imported
by ``jaz.llm``, so a home under ``jaz.llm`` would have made a plain import a cycle. Here there
is no cycle to dodge — and the alias is not LLM-specific in the first place.
"""

from __future__ import annotations

from typing import Any

#: One chat message in the OpenAI wire shape — ``{"role": ..., "content": ...}`` plus whatever
#: else a provider accepts. An alias rather than a ``TypedDict`` because the shape is the
#: provider's, not JAZ's, and JAZ neither validates nor closes it. Internal: public signatures
#: spell it out as ``dict[str, Any]``.
MessageDict = dict[str, Any]
