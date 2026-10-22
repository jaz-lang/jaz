"""The framework's single "no value supplied" sentinel.

Defined here, in a module with no ``jaz`` imports, so that both the effect layer and the
config layer can share one object without an import cycle between them. The public path is
:data:`jaz.UNSET` (with :class:`jaz.UnsetType`); this module is private and callers never
import from it.
"""

from __future__ import annotations

import enum


class UnsetType(enum.Enum):
    """The "no value supplied" sentinel's type — a marker distinct from ``None``.

    A named type rather than a bare ``object()`` so it can be written in annotations
    (``T | UnsetType``), and so ``x is UNSET`` narrows the type for a type checker.
    """

    UNSET = enum.auto()

    def __repr__(self) -> str:
        return "UNSET"


# A single-member enum with a custom `__repr__`, chosen over a plain sentinel class after
# measuring both (EXECUTIVE CALL, user, 2026-09-13).
#
# PEP 661's objection to the enum idiom is its repr — `<UnsetType.UNSET: 1>` — and a two-line
# `__repr__` answers exactly that. What the plain-class form cannot recover is the other two
# properties, both measured on this code:
#
#   * `is` NARROWING. `if c is not UNSET: c.upper()` type-checks against an enum member and is
#     a pyright error against a class instance, because only literal-like types narrow on
#     identity. `@typing.final` does not change this.
#   * SINGLETON IDENTITY UNDER PICKLE. A `__new__`-based singleton is defeated at pickle
#     protocols 0 and 1, where `copyreg._reconstructor` calls `object.__new__` directly: the
#     result reprs as `UNSET` but `is not UNSET`, so it survives a keep-check and gets written
#     into a typed field. Enum members pickle by name and hold identity at every protocol.
#
# Enums with members are also final by construction, so there is no second instance to create.
# PEP 661's `sentinel()` supersedes all of this, but it lands in 3.15 and this package targets
# 3.12.

#: The sentinel itself. Two layers share it: effects use it as ``ModifyLLMResponse``'s
#: "leave this field untouched" marker, and config uses it as the "caller did not pass this
#: group" marker. Both mean "no value supplied", and ``None`` is a real value in both.
UNSET = UnsetType.UNSET
