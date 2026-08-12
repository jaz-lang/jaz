"""Legacy import path for :mod:`jaz._agent` — reachable, but not public API.

``Agent`` moved to the ``_``-prefixed sibling so that this path *warns* instead of
resolving silently; see :func:`jaz._warnings.make_private_module_shim` for why relocating
the definition is what makes ``from jaz.agent import Agent`` observable at all.

The supported surface is ``jaz.__all__`` — an agent is run via ``jaz.invoke()``.

Internal code imports from ``jaz._agent`` directly; going through this shim warns.
"""

from ._warnings import make_private_module_shim

__getattr__, __dir__ = make_private_module_shim(__name__, "jaz._agent")
