"""Legacy import path for :mod:`jaz._llm_client` — reachable, but not public API.

``LLM`` / ``LLMResponse`` / ``MockLLMClient`` and the backend-construction helpers moved to
the ``_``-prefixed sibling so that this path *warns* instead of resolving silently; see
:func:`jaz._warnings.make_private_module_shim`.

``LLMClient`` no longer exists under any name: it was merged with ``Provider`` into the single
``LLM`` class (:mod:`jaz.providers.llm`), whose public home is ``jaz.providers``. #984 tracks
the seam's final public shape. Configure the backend by constructing it and passing it in —
``jaz.configure(llm=OpenAILLM(model=...))``.

Internal code imports from ``jaz._llm_client`` directly; going through this shim warns.
"""

from ._warnings import make_private_module_shim

__getattr__, __dir__ = make_private_module_shim(__name__, "jaz._llm_client")
