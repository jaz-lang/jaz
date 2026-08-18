from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from types import ModuleType
from typing import Self

from .._catalog import render_catalog


def _parse_dotted_path(dotted_path: str) -> list[str]:
    dotted_path_parts = dotted_path.split(".")
    if len(dotted_path_parts) == 0 or any(part == "" for part in dotted_path_parts):
        raise ValueError(
            "Invalid dotted path. A dotted path must be a non-empty dot-separated "
            "string."
        )
    return dotted_path_parts


def _unparse_dotted_path(dotted_path_parts: list[str]) -> str:
    return ".".join(dotted_path_parts)


class Library:
    """A convenience builder for a nested, dotted tool namespace.

    A library is a collection of tools (functions or other objects with ``__name__``
    / ``__doc__``) assembled into a hierarchical namespace, like a Python package.
    Its construction ergonomics — dotted-path :meth:`~Library.add`, the
    :meth:`~Library.register` decorator, nested submodules with docstrings — are the reason
    it still exists; ``SimpleNamespace`` covers the flat case, ``Library`` is nicer
    for nested namespaces built at runtime (tools that close over per-episode state).

    Role in the framework (see ``design/design_features/library_as_input.md``):
    a ``Library`` is now an **ordinary input**, not a privileged kwarg. There is no
    ``libraries=`` on ``jaz.invoke`` — you pass a library as a normal keyword
    argument and it binds into the REPL under that name. Two protocol methods make
    this work:

    - :meth:`__jaz_get__` returns the library's single root module, so the agent
      interacts with the module of tools rather than this bookkeeping object (payload
      substitution — the same ``__jaz_get__`` protocol that ``jaz.Display`` uses).
    - :meth:`__jaz_description__` renders the tool catalog rooted at the *bound name*
      (the kwarg), so ``jaz.invoke(tools=lib)`` documents ``tools.foo(...)`` regardless
      of the root module's internal name.

    Constraint: a ``Library`` used as an input must have **exactly one root module**
    (every real library does); see :meth:`_single_root_module`. The privileged
    multi-name binding (``add_self_to_program_state``) survives only to support eval
    harnesses with their own REPLs.

    Methods:
    - add(tool_path: str, tool: object): Add a tool to the library at the
      path specified by tool_path.
    """

    # ``Library`` is now purely a *user*-facing builder. The framework used to eat its own
    # dog food here — the recursive sub-invoke tool shipped as a one-member ``Library`` named
    # "JAZ" whose root module was ``jaz``, which is where ``add_self_to_program_state`` and
    # ``render_prompt_description`` came from. That wrapper went away with the ``jaz.``
    # prefix (the primitive is now a bare callable — see ``jaz._invoke_tool``), so both
    # methods are kept for out-of-core callers only, and no core path constructs a ``Library``.

    def __init__(
        self,
        name: str,
        desc: str | None,
        modules: Iterable[tuple[str, str]],
        tools: Iterable[tuple[str, object]],
    ) -> None:
        self._name = name
        self._desc = desc
        self._root_modules: dict[str, ModuleType] = {}
        for module_path, module_doc in modules:
            self.create_module(module_path, module_doc)
        for tool_path, tool in tools:
            self.add(tool_path, tool)

    @classmethod
    def from_python_module(cls, module: ModuleType) -> Self:
        library = cls(module.__name__, module.__doc__, [], [])
        library._root_modules[module.__name__] = module
        return library

    def get(self, path: str):
        path_parts = _parse_dotted_path(path)
        if path_parts[0] not in self._root_modules:
            raise AttributeError(
                f"Module {path_parts[0]} not found in library {self._name}."
            )
        current_module = self._root_modules[path_parts[0]]
        for i in range(1, len(path_parts)):
            part = path_parts[i]
            if not hasattr(current_module, part):
                raise AttributeError(
                    f"Name {part} not found in module "
                    f"{_unparse_dotted_path(path_parts[:i])}."
                )
            current_module = getattr(current_module, part)
        return current_module

    def create_module(self, module_path: str, doc: str, exist_ok: bool = False):
        module_path_parts = _parse_dotted_path(module_path)
        if len(module_path_parts) == 1:
            if module_path in self._root_modules and not exist_ok:
                raise AttributeError(
                    f"module {module_path} already exists in library {self._name}. "
                    "To replace it, set the option `exist_ok = True` when calling "
                    "`Library.add()`."
                )
            new_module = ModuleType(module_path, doc)
            self._root_modules[module_path] = new_module
            return
        if module_path_parts[0] not in self._root_modules:
            raise AttributeError(
                f"Module {module_path_parts[0]} not found in library {self._name}. "
                "Please create it with Library.create_module() first."
            )
        current_module = self._root_modules[module_path_parts[0]]
        for part in module_path_parts[1:-1]:
            if not hasattr(current_module, part):
                raise AttributeError(
                    f"Module {part} not found in library {self._name}. "
                    "Please create it with Library.create_module() first."
                )
            current_module = getattr(current_module, part)
        if hasattr(current_module, module_path_parts[-1]) and not exist_ok:
            raise AttributeError(
                f"module {module_path} already exists in library {self._name}. "
                "To replace it, set the option `exist_ok = True` when calling "
                "`Library.add()`."
            )
        new_module = ModuleType(module_path_parts[-1], doc)
        setattr(current_module, module_path_parts[-1], new_module)

    def add(self, tool_path: str, tool: object, exist_ok: bool = False):
        tool_path_parts = _parse_dotted_path(tool_path)
        if tool_path_parts[0] not in self._root_modules:
            raise AttributeError(
                f"Module {tool_path_parts[0]} not found in library {self._name}. "
                "Please create it with Library.create_module() first."
            )
        current_module = self._root_modules[tool_path_parts[0]]
        for part in tool_path_parts[1:-1]:
            if not hasattr(current_module, part):
                raise AttributeError(
                    f"Module {part} not found in library {self._name}. "
                    "Please create it with Library.create_module() first."
                )
            current_module = getattr(current_module, part)
        if hasattr(current_module, tool_path_parts[-1]) and not exist_ok:
            raise AttributeError(
                f"Tool {tool_path} already exists in library {self._name}. "
                "To replace it, set the option `exist_ok = True` when calling "
                "`Library.add()`."
            )
        setattr(current_module, tool_path_parts[-1], tool)

    def register(self, tool_path: str):
        def tool_decorator[T](tool: T) -> T:
            self.add(tool_path, tool)
            return tool

        return tool_decorator

    def add_self_to_program_state(
        self, program_state: MutableMapping[str, object]
    ) -> None:
        for module_name, module in self._root_modules.items():
            program_state[module_name] = module

    def _single_root_module(self) -> ModuleType:
        """Return the library's one root module, enforcing the single-root invariant.

        A ``Library`` used as a normal ``jaz.invoke`` input binds under exactly one
        kwarg name, so it must expose exactly one root module (the design constrains
        ``Library`` to a single root — see ``library_as_input.md``). The invariant is
        checked lazily here rather than at construction so the builder API (empty
        libraries, incremental ``create_module``) keeps working; it only has to hold
        at the point the library is consumed as an input.
        """
        if len(self._root_modules) != 1:
            raise ValueError(
                f"Library {self._name!r} must have exactly one root module to be used "
                f"as an input (has {len(self._root_modules)}: "
                f"{sorted(self._root_modules)}). A Library binds under a single kwarg "
                "name; nest additional namespaces as sub-modules of the one root."
            )
        (module,) = self._root_modules.values()
        return module

    def __jaz_get__(self) -> ModuleType:
        """Return the root module to bind into the REPL under this input's kwarg name.

        Payload substitution (the ``__jaz_get__`` protocol): the
        ``Library`` is bookkeeping; the agent interacts with the module of tools. So
        ``jaz.invoke(env=lib)`` binds the root module as ``env``, not the ``Library``.
        """
        return self._single_root_module()

    def __jaz_description__(self, bound_name: str | None) -> str:
        """Render this library's tool catalog rooted at ``bound_name``.

        This is the generalized form of the old ``<tool_libraries>`` rendering: the
        catalog renders against the *bound name* (the kwarg the agent sees), so
        ``jaz.invoke(tools=lib)`` produces ``tools.bash(...)`` regardless of the root
        module's internal name.

        The body is delegated to :meth:`~Library.default_description` so a ``jaz.describe``
        override can *compose* with the default catalog rather than replace it
        wholesale: ``jaz.get_description`` routes back through this method
        (which the override has replaced), so the default has to be reachable by a
        path that does not — that path is ``default_description``.
        """
        return self.default_description(bound_name)

    def default_description(self, bound_name: str | None = None) -> str:
        """Render the default tool catalog for this library, rooted at ``bound_name``.

        Factored out of :meth:`__jaz_description__` so a caller that overrides this
        library's description via ``jaz.describe`` can still reach the default
        rendering and append to it, e.g.::

            describe(
                lib, lambda l, name: f"Custom note.\\n\\n{l.default_description(name)}"
            )

        Going through ``jaz.get_description`` instead would recurse — it resolves
        ``__jaz_description__``, which the override has replaced.
        """
        root = self._single_root_module()
        return render_catalog(root, bound_name)

    def render_prompt_description(self, *, full_docstrings: bool = True) -> str:
        """Render the full library card (name + description + tool catalog).

        The per-object catalog walk is shared with the general ``jaz.Catalog`` render mode
        via :func:`jaz._catalog.render_catalog`, so the two cannot drift. Each root module is
        rendered rooted at its own name (real libraries have exactly one root; see
        ``_single_root_module``).

        ``full_docstrings`` defaults to ``True`` — render comprehensively; pass
        ``False`` to opt into compact (first-line-only) rendering when size matters.
        See :func:`jaz._catalog.render_catalog` for the rationale.
        """

        def backtickify(text: str) -> str:
            return f"`{text}`"

        sections = [
            f"**Library name:** {self._name}",
            f"**Description:** {self._desc or '(no description available)'}",
            f"**Top-level module(s):** {', '.join(map(backtickify, self._root_modules.keys()))}",
        ]

        # Each root is rendered independently, so the per-catalog entry cap
        # (_MAX_PROMPT_OBJECT_ENTRIES) now applies per root rather than once across
        # all roots — a multi-root library could emit up to N*cap entries and N
        # truncation markers. Harmless because the _single_root_module invariant
        # means real libraries have exactly one root.
        catalogs = [
            render_catalog(module, module_name, full_docstrings=full_docstrings)
            for module_name, module in sorted(self._root_modules.items())
        ]
        catalog_text = "\n".join(c for c in catalogs if c)
        if catalog_text:
            sections.append("**Available objects:**\n" + catalog_text)

        return "\n\n".join(sections) + "\n"

    def __str__(self) -> str:
        return self.render_prompt_description()

    def __repr__(self) -> str:
        return str(self)
