"""LLM backends — one class per backend (:class:`BaseLLM`), selected by configuring an instance.

The ``litellm`` backend routes through the LiteLLM library and is the default, reaching every
provider LiteLLM supports. ``GitHubCopilotLLM`` and ``SGLangLLM`` ship alongside it for the two cases it
does not serve well — completions billed to GitHub AI credits, and a self-hosted SGLang server
addressed token-in/token-out. ``MockLLM`` answers from a callable you supply, for deterministic
tests with no provider. Add your own by subclassing :class:`BaseLLM` — see :class:`BaseLLM` for
what a backend owns.
"""

from __future__ import annotations

# Public surface: the authoring contract, plus every backend that ships.
#
# ``BaseLLM`` is the class you subclass and ``LLMResponse`` is what its one abstract method
# returns, so a backend author cannot write ``complete`` without both — an unadvertised return
# type on a public ABC is a boundary nobody can respect.
#
# The provider-backed backends are public by the same rule, stated in ``jaz.repl.__init__``: a
# caller is handed one. ``LiteLLM`` is the default; ``GitHubCopilotLLM`` and ``SGLangLLM`` cover
# the two cases it cannot reach. Each is constructed by name in user code and named by a console
# tag (``litellm``, ``copilot``, ``sglang``), so shipping them unadvertised would only mean
# callers depended on them without a promise. ``MockLLM`` is public on a different rule: no
# authored config can name it (it is built from a callable) and it has no console tag, but it
# exists so a test needs no provider — and a test author has to be able to import it.
#
# ``ChatFormat``/``HFChatFormat`` follow from ``SGLangLLM``: they type its ``chat_format``
# parameter and its docstring tells the caller to construct one, so promoting the backend without
# them would leave a public constructor taking a type nobody can name.
#
# Nothing else is exported for reachability. There is no message/usage envelope type: every
# backend constructs ``LLMResponse`` directly (``SGLangLLM`` says why in a comment — its wire
# envelope is not OpenAI-shaped), and ``MessageDict`` lives in ``jaz._messages``. Pricing is
# LiteLLM's job (``litellm.completion_cost``), which is what a caller wanting cost should reach
# for. The exception taxonomy is public in ``jaz.exceptions`` only; a second path here would be a
# duplicate nothing imports.
# ``TokenStamp``/``TurnRecord`` are here rather than in a module of their own because
# ``LLMResponse`` is the type that originates one: a token-native backend builds the record
# inside ``complete()`` and hands it back on the response (a caller also meets one on
# ``Rollout.turns``, and a hook may construct one for ``SupplyLLMResponse.tokens``, but the
# response is where it comes from). Publishing the
# response type while leaving the type of one of its fields unnameable was the gap
# (EXECUTIVE CALL, user, 2026-09-13).
from jaz._tokens import TokenStamp, TurnRecord

from ._base import BaseLLM
from ._chat_format import ChatFormat, HFChatFormat
from ._copilot import GitHubCopilotLLM
from ._litellm import LiteLLM
from ._mock import MockLLM
from ._sglang import SGLangLLM
from ._types import LLMResponse as LLMResponse
from ._types import ModelInfo as ModelInfo
from ._types import RetryPolicy as RetryPolicy

__all__ = [
    "BaseLLM",
    "ChatFormat",
    "GitHubCopilotLLM",
    "HFChatFormat",
    "LLMResponse",
    "LiteLLM",
    "MockLLM",
    # The two value types a backend author constructs: what it reports about its model, and
    # how the framework retries its calls.
    "ModelInfo",
    "RetryPolicy",
    "SGLangLLM",
    "TokenStamp",
    "TurnRecord",
]
