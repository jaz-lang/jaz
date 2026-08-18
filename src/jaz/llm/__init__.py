"""LLM backends — one class per backend (:class:`BaseLLM`), selected by one tag.

The ``litellm`` backend routes through the LiteLLM library; the ``OpenAILLM``/``AnthropicLLM``
classes post directly to a vendor's HTTP API. Add your own with ``@register_llm("mybackend")``;
see :mod:`jaz.llm.llm` for what a backend owns and why the former ``LLMClient``/``Provider``
split was merged away.
"""

from __future__ import annotations

# Public surface: base + default concrete backend only — ``BaseLLM`` and ``LiteLLM`` (the sole
# v1 backend). Everything else (the retired OpenAILLM/AnthropicLLM backends, ``LLMResponse`` and
# the message/usage types, the LLM exception taxonomy, pricing helpers, the extras-merge
# registry, and the backend registry/registration) is re-exported for reachability — internal
# call sites import these from the package — but kept OUT of ``__all__`` via the ruff-blessed
# ``X as X`` re-export idiom, so it is not part of the v1 public promise. The three
# pluggable-component packages (llm/protocol/repl) are uniform on this: each exposes only its
# base + default concrete (see the note in ``jaz.repl.__init__``).
#
# ``ChatFormat``/``HFChatFormat`` are demoted the same way, by deliberate call (PR #1128): the
# caller-owned chat-format seam stays reachable as ``jaz.llm.HFChatFormat`` but is NOT pinned
# into the v1 promise — it gets the identical treatment to the OpenAILLM/AnthropicLLM backends
# rather than a sanctioned exception to the base+default-concrete pin.
from ._litellm import LiteLLM
from .anthropic import AnthropicLLM as AnthropicLLM
from .base import Choice as Choice
from .base import CompletionResponse as CompletionResponse
from .base import Message as Message
from .base import MessageDict as MessageDict
from .base import Usage as Usage
from .base import declared_init_keys as declared_init_keys
from .chat_format import ChatFormat as ChatFormat
from .chat_format import HFChatFormat as HFChatFormat
from .exceptions import APIError as APIError
from .exceptions import AuthenticationError as AuthenticationError
from .exceptions import ContentPolicyViolationError as ContentPolicyViolationError
from .exceptions import ContextWindowExceededError as ContextWindowExceededError
from .exceptions import LLMError as LLMError
from .exceptions import NotFoundError as NotFoundError
from .exceptions import PermissionDeniedError as PermissionDeniedError
from .exceptions import RateLimitError as RateLimitError
from .exceptions import UnsupportedParamsError as UnsupportedParamsError
from .extras_merge import MergeRule as MergeRule
from .extras_merge import get_extra_merge_rule as get_extra_merge_rule
from .extras_merge import merge_extras as merge_extras
from .extras_merge import register_extra_merge_rule as register_extra_merge_rule
from .llm import BaseLLM
from .llm import LLMResponse as LLMResponse
from .openai import OpenAILLM as OpenAILLM
from .pricing import compute_cost as compute_cost
from .pricing import cost_per_token as cost_per_token
from .pricing import get_all_models as get_all_models
from .pricing import get_model_info as get_model_info
from .registry import LLM_REGISTRY as LLM_REGISTRY
from .registry import register_llm as register_llm
from .sglang import SGLangLLM as SGLangLLM

__all__ = [
    "BaseLLM",
    "LiteLLM",
]
