"""REPL registration system.

This module provides a decorator-based system for registering REPL implementations.
Users can register custom REPLs that will be available to the agent system.
"""

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseREPL


# Global mapping from language names to REPL classes
REPL_LANGUAGE_MAP: dict[str, type["BaseREPL"]] = {}

# Character class for valid REPL language names. No longer shared with the protocol layer: the
# The response is raw code with no wrapper tag or ``lang=`` attribute, so the protocol has no language
# to validate and this pattern is used only for registry-name checking.
REPL_LANG_NAME_PAT = r"[a-zA-Z0-9_]+"
REPL_LANGUAGE_NAME_RE = re.compile(rf"^{REPL_LANG_NAME_PAT}$")


def register_repl(language: str):
    """Decorator to register a REPL implementation for a specific language.

    This decorator allows users to register custom REPL implementations that can
    be used by the agent. The registered REPL will be available via the language
    name in REPL_LANGUAGE_MAP.

    Args:
        language: The language name for this REPL (e.g., "python", "javascript")

    Returns:
        Decorator function that registers the REPL class

    A registered REPL must follow two contracts, neither of which this decorator can check
    (it verifies only that the class subclasses ``BaseREPL``):

    * ``__init__`` takes **configuration only** and must leave the instance inert — no
      namespace, no session, nothing per-run. Its declared parameters are the REPL's config
      surface, discovered by signature the same way a backend's are.
    * ``initialize`` takes **invoke-time arguments only** and returns a **new** instance,
      leaving the receiver a reusable template. Binding onto ``self`` and returning it makes
      the object single-use; one configured REPL is expected to serve many invokes, each with
      its own state.

    Examples:
        from jaz.repl.base import BaseREPL
        from jaz.repl.registry import register_repl

        @register_repl("javascript")
        class JavaScriptREPL(BaseREPL):
            def __init__(self, exec_timeout=30.0):
                self.exec_timeout = exec_timeout  # configuration only

            def initialize(self, inputs, invoke_tool, ...):
                new = copy.copy(self)             # never bind onto `self`
                new.namespace = dict(inputs)      # per-invoke state on the copy
                return new

            def exec(self, src, input_id, ...):
                # Implementation
                ...

        # Now "javascript" is available in REPL_LANGUAGE_MAP
        # and can be selected via jaz.configure(repl="javascript")
    """
    # The `initialize` contract changed: it used to be documented as single-use ("construct a
    # fresh REPL per invoke, as `Agent` does"), so an implementation written against that text
    # binds onto `self` and returns it. That keeps working while every caller builds a REPL per
    # invoke, and starts bleeding state between invokes the moment one is held and reused —
    # silently, as wrong output rather than an error. There is no dispatch seam around
    # `initialize` to enforce it from, so it is stated here, where an implementer looks, and on
    # the `BaseREPL` ABC. TODO(#1057) tracks detecting it; TODO(#1058) tracks removing the need for
    # it by making `BaseREPL` stateless with an explicit `REPLState`, which is the intended end
    # state — there is then no `self` to bind to and the contract cannot be violated.

    if not REPL_LANGUAGE_NAME_RE.match(language):
        raise ValueError(
            f"Invalid REPL language name: '{language}'. "
            "REPL language names must match the regex [a-zA-Z0-9_]+"
        )

    def decorator[C: type["BaseREPL"]](repl_class: C) -> C:
        """Register the REPL class in the global map."""
        # Import here to avoid circular dependency
        from .base import BaseREPL

        if not issubclass(repl_class, BaseREPL):
            raise TypeError(
                f"REPL class {repl_class.__name__} must inherit from BaseREPL base class"
            )
        if language in REPL_LANGUAGE_MAP:
            raise ValueError(
                f"REPL for language '{language}' is already registered "
                f"({REPL_LANGUAGE_MAP[language].__name__})"
            )
        REPL_LANGUAGE_MAP[language] = repl_class
        return repl_class

    return decorator
