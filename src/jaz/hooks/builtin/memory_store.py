"""MemoryStoreHook — injects a MemoryStore into each invoke() session."""

from __future__ import annotations

from jaz.hooks.base import Event
from jaz.hooks.dispatcher import Hook
from jaz.hooks.effects import AddInstructionPrompt, AddReplInput, Effect
from jaz.hooks.events import InvokeEnter
from jaz.memory_store import MemoryStore


class MemoryStoreHook(Hook):
    """Hook that injects a :class:`~jaz.memory_store.MemoryStore` into each invoke.

    At the start of every ``jaz.invoke()`` call, this hook injects:

    - ``store``: the :class:`~jaz.memory_store.MemoryStore` instance,
      with ``get``, ``get_code``, ``insert``, and ``delete`` methods.
    - ``__store_catalog__``: a ``{name: description}`` dict of all stored items.

    The agent sees a summary of the store's contents in the instruction prompt
    and can retrieve, add, or remove items using plain Python code.

    Usage::

        store = MemoryStore()
        with MemoryStoreHook(store):
            result = jaz.invoke("solve this task", return_type=str)

    The agent can then write, for example::

        # Retrieve a relevant tool
        fn = store.get("parse_nested_json")
        result = fn(raw_text)

        # Save a new discovery
        store.insert(
            "retry_on_timeout",
            "Decorator that retries a function up to 3 times on TimeoutError.",
            '''
        def retry_on_timeout(fn):
            def wrapper(*args, **kwargs):
                for _ in range(3):
                    try:
                        return fn(*args, **kwargs)
                    except TimeoutError:
                        pass
                return fn(*args, **kwargs)
            return wrapper
        ''',
        )

        # Improve an existing item
        old = store.get_code("parse_nested_json")
        new = jaz.invoke(f"Extend this to handle JSONL:\\n{old}", return_type=str)
        store.delete("parse_nested_json")
        store.insert("parse_nested_json", "Handles JSON and JSONL embedded in text.", new)
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def on_event(self, event: Event) -> list[Effect]:
        if isinstance(event, InvokeEnter):
            return [
                AddReplInput("store", self.store, reason="memory store"),
                AddReplInput(
                    "__store_catalog__", self.store.catalog, reason="store catalog"
                ),
                AddInstructionPrompt(self._build_prompt()),
            ]
        return []

    def _build_prompt(self) -> str:
        idx = self.store.catalog
        lines = [
            "## Memory Store",
            "",
            "You have a memory store for this episode only.",
            "**At the start of your task:** check `__store_catalog__` and retrieve anything relevant with `store.get(name)`.",
            "**Before returning:** you MUST call `store.insert()` at least once to save something useful.",
            "Store the most reusable helper function or strategy you developed. If the store already has a",
            "related item you improved upon, delete the old one and insert the new version.",
            "",
            "Good candidates: helper functions, chunking/parsing strategies, observations about data format,",
            "patterns that generalize beyond this specific task.",
            "",
            "Interface (available as `store` in the REPL):",
            "  store.get(name)                        → live object (callable, str, dict, ...)",
            "  store.get_code(name)                   → raw source code string",
            "  store.insert(name, description, code)  → add a new item (code is a Python string)",
            "                                           code must define a top-level symbol named `name`",
            "  store.delete(name)                     → remove an item",
            "  __store_catalog__                      → {name: description} dict of all items",
            "",
        ]
        if idx:
            lines.append(f"Current store ({len(idx)} item(s)):")
            for name, desc in idx.items():
                lines.append(f"  - {name}: {desc}")
        else:
            lines.append("The store is currently empty.")
        return "\n".join(lines)
