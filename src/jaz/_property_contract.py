"""One check behind the "a no-argument reader is a property" convention.

The convention is in CLAUDE.md under Code Style: a reader that takes no arguments is a
property with a noun name, a reader that needs an argument is a ``get_*`` method, and anything
that acts is a verb. This module holds the guard for the classes third parties subclass.
"""

from __future__ import annotations

import inspect


def require_properties(cls: type, names: tuple[str, ...]) -> None:
    """Raise ``TypeError`` if ``cls`` resolves any of ``names`` to a callable rather than a property.

    Call from ``__init_subclass__`` on a base whose readers are properties.
    """
    # Why this exists at all, when every other rename in this family simply breaks loudly: a
    # method where a property is expected does NOT break loudly. The attribute resolves, to a
    # bound method, and the caller uses it as a value — rendering "<bound method ...>" into a
    # prompt, or failing three frames away with "'method' object has no attribute 'items'".
    # The ABC machinery cannot catch it either: an abstract property is satisfied, as far as
    # `abstractmethod` is concerned, by any attribute of that name. So the check is here, at
    # class definition, where the message can name the fix.
    #
    # Resolved through the MRO (`getattr_static`), not read off `cls.__dict__`: a mixin listed
    # before the base (`class H(Mixin, Hook)`) supplies the attribute without defining it on
    # `cls`, and a dict-only check would pass it. `getattr_static` is what makes that case
    # visible without invoking the descriptor — a plain `getattr` would CALL a property here,
    # on a half-built class, which is exactly the side effect a validator must not have.
    #
    # `classmethod` and `staticmethod` are rejected rather than waved through: neither is a
    # reader of instance state, a `staticmethod` reaches the same bound-object-used-as-a-value
    # bug this module exists to prevent, and chaining `classmethod` onto `property` stopped
    # working in 3.13, so nothing legitimate is being excluded.
    for name in names:
        try:
            member = inspect.getattr_static(cls, name)
        except (
            AttributeError
        ):  # not defined anywhere on the MRO — the ABC's problem, not ours
            continue
        if isinstance(member, property) or type(member).__name__ == "cached_property":
            continue
        if callable(member) or isinstance(member, (classmethod, staticmethod)):
            raise TypeError(
                f"{cls.__name__}.{name} must be a property, not a "
                f"{type(member).__name__}: it takes no arguments, and this package spells a "
                f"no-argument reader as a property. Add @property above the definition "
                f"(@functools.cached_property if it is expensive). Left as it is, reading "
                f"{name!r} gives back the callable — a bound method, or the function itself for "
                f"a staticmethod — which the framework then uses as a value, with no error at "
                f"that point."
            )
