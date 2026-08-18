"""Legacy import path for :mod:`jaz._library` — reachable, but not public API.

``Library`` moved to the ``_``-prefixed sibling so that this path *warns* instead of
resolving silently; see :func:`jaz._warnings.make_private_module_shim` for why relocating
the definition is what makes ``from jaz.library import Library`` observable at all.

``Library`` is experimental in v1 (``# TODO`` to stabilize): it is intentionally left off
``jaz.__all__`` and every route through this public-named module warns. Internal code
imports from ``jaz._library`` directly; going through this shim warns.
"""

from ._warnings import make_private_module_shim

__getattr__, __dir__ = make_private_module_shim(__name__, "jaz._library")
