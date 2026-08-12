"""Legacy import path for :mod:`jaz._display` — reachable, but not public API.

``Display`` moved to the ``_``-prefixed sibling so that this path *warns* instead of
resolving silently; see :func:`jaz._warnings.make_private_module_shim`.

Per-call display control is **experimental**: the mechanism may change or be removed once
``jaz.describe`` is shown to cover its use cases (a hide-capable ``describe`` would leave
only the t-string render override, which is itself experimental). Until then it is
reachable, unsupported, and outside ``jaz.__all__``.

Internal code imports from ``jaz._display`` directly; going through this shim warns.
"""

from ._warnings import make_private_module_shim

__getattr__, __dir__ = make_private_module_shim(__name__, "jaz._display")
