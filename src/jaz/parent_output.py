"""Legacy import path for :mod:`jaz._parent_output` — reachable, but not public API.

``parent_print`` and the output-channel accessors moved to the ``_``-prefixed sibling so
that this path *warns* instead of resolving silently; see
:func:`jaz._warnings.make_private_module_shim`.

There is no supported replacement yet; prefer a name in ``jaz.__all__`` where one fits.

Internal code imports from ``jaz._parent_output`` directly; going through this shim warns.
"""

from ._warnings import make_private_module_shim

__getattr__, __dir__ = make_private_module_shim(__name__, "jaz._parent_output")
