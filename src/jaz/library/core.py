from __future__ import annotations

import inspect
from collections.abc import Iterable, MutableMapping
from types import ModuleType
from typing import Self

# TODO: Have this be configurable
_MAX_PROMPT_OBJECT_ENTRIES = 100


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
    """
    A library is a collection of tools that can be called by the language model
    in a REPL session. Tools are functions or other Python objects that have
    __name__ and __doc__ attributes. A library structured as a hierarchical namespace,
    similar to Python packages and modules.

    Methods:
    - add(tool_path: str, tool: object): Add a tool to the library at the
      path specified by tool_path.
    """

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

    @staticmethod
    def _summarize_doc(obj: object, *, full: bool = False) -> str | None:
        doc = inspect.getdoc(obj)
        if not doc:
            return None
        doc = doc.strip()
        return doc if full else doc.splitlines()[0]

    @staticmethod
    def _format_signature(obj: object) -> str:
        if not callable(obj):
            return ""
        try:
            return str(inspect.signature(obj))
        except (TypeError, ValueError):
            return "()"

    @classmethod
    def _format_object_entry(
        cls, dotted_path: str, obj: object, *, full_docstrings: bool = False
    ) -> str:
        if isinstance(obj, ModuleType):
            doc_summary = (
                cls._summarize_doc(obj, full=full_docstrings)
                or "(no description available)"
            )
            return f"- `{dotted_path}` [module]: {doc_summary}"

        if callable(obj):
            signature = cls._format_signature(obj)
            doc_summary = (
                cls._summarize_doc(obj, full=full_docstrings)
                or "(no description available)"
            )
            return f"- `{dotted_path}{signature}`: {doc_summary}"

        type_name = type(obj).__name__
        return f"- `{dotted_path}` [{type_name}]"

    @classmethod
    def _render_module_entries(
        cls,
        module: ModuleType,
        dotted_path: str,
        *,
        seen_modules: set[int],
        remaining_entries: list[int],
        full_docstrings: bool = False,
    ) -> list[str]:
        if remaining_entries[0] <= 0:
            return []

        module_id = id(module)
        if module_id in seen_modules:
            return [f"- `{dotted_path}` [module]: (already documented above)"]

        seen_modules.add(module_id)
        entries = [
            cls._format_object_entry(
                dotted_path, module, full_docstrings=full_docstrings
            )
        ]
        remaining_entries[0] -= 1

        for name, value in sorted(vars(module).items()):
            if remaining_entries[0] <= 0:
                break
            if name.startswith("_"):
                continue
            child_path = f"{dotted_path}.{name}"
            if isinstance(value, ModuleType):
                entries.extend(
                    cls._render_module_entries(
                        value,
                        child_path,
                        seen_modules=seen_modules,
                        remaining_entries=remaining_entries,
                        full_docstrings=full_docstrings,
                    )
                )
            else:
                entries.append(
                    cls._format_object_entry(
                        child_path, value, full_docstrings=full_docstrings
                    )
                )
                remaining_entries[0] -= 1
        return entries

    def render_prompt_description(self, *, full_docstrings: bool = False) -> str:
        def backtickify(text: str) -> str:
            return f"`{text}`"

        sections = [
            f"**Library name:** {self._name}",
            f"**Description:** {self._desc or '(no description available)'}",
            f"**Top-level module(s):** {', '.join(map(backtickify, self._root_modules.keys()))}",
        ]

        exported_entries: list[str] = []
        seen_modules: set[int] = set()
        remaining_entries = [_MAX_PROMPT_OBJECT_ENTRIES]
        for module_name, module in sorted(self._root_modules.items()):
            exported_entries.extend(
                self._render_module_entries(
                    module,
                    module_name,
                    seen_modules=seen_modules,
                    remaining_entries=remaining_entries,
                    full_docstrings=full_docstrings,
                )
            )

        if exported_entries:
            if remaining_entries[0] <= 0:
                exported_entries.append(
                    f"- ... truncated after {_MAX_PROMPT_OBJECT_ENTRIES} objects"
                )
            sections.append("**Available objects:**\n" + "\n".join(exported_entries))

        return "\n\n".join(sections) + "\n"

    def __str__(self) -> str:
        return self.render_prompt_description()

    def __repr__(self) -> str:
        return str(self)
