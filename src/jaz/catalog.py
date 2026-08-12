"""Legacy import path for :mod:`jaz._catalog` — reachable, but not public API.

``Catalog`` / ``as_catalog`` moved to the ``_``-prefixed sibling so that this path *warns*
instead of resolving silently; see :func:`jaz._warnings.make_private_module_shim`.

The supported way to control how a value is rendered to an agent is :func:`jaz.describe`.
``jaz.Display`` controls rendering for a single call, but it is experimental and outside
``jaz.__all__`` — reachable, and it warns on use.

Internal code imports from ``jaz._catalog`` directly; going through this shim warns.
"""

from ._warnings import make_private_module_shim

__getattr__, __dir__ = make_private_module_shim(__name__, "jaz._catalog")
