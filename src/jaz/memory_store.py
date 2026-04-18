"""In-memory store for reusable Python snippets.

Items are stored as named code snippets with associated descriptions.
This class is process-local and has no disk persistence.
"""

from __future__ import annotations

import threading


class MemoryStore:
    """Memory store for named Python snippets.

    Items are stored as named code snippets. Each snippet has:

    - A **name**: unique identifier used for retrieval.
    - A **description**: shown in the catalog; used by the agent to decide
      what to retrieve without loading the full code.
    - **Code**: any valid Python source — a function definition, a variable
      assignment, a string literal, etc.
      The code must define a top-level symbol with the same name as ``name``.

    Stored functions are callable, stored strings are readable, and stored
    dicts are usable directly. The agent can also call ``get_code()`` to
    inspect or edit any item.

    Lifecycle
    ---------
    The store lifetime is the Python object lifetime:

    - Reuse the same ``MemoryStore`` instance to accumulate memory over
      multiple invoke calls.
    - Create a new instance to start from an empty store.

    In the evaluation harness, ``jaz.memory_store: true`` creates a fresh
    ``MemoryStore`` per episode, so memory is isolated per episode and is not
    shared across tasks/episodes.

    Example::

        store = MemoryStore()
        store.insert(
            "double",
            "Doubles a number.",
            "def double(x):\\n    return x * 2",
        )
        fn = store.get("double")
        fn(3)  # → 6

        store.insert(
            "API_NOTE",
            "Observed quirk: search endpoint paginates with offset param.",
            'API_NOTE = "search returns max 10 results; use offset= to paginate"',
        )
        note = store.get("API_NOTE")  # → the string
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._catalog: dict[str, str] = {}  # name → description
        self._sources: dict[str, str] = {}  # name → raw code string
        self._namespace: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, name: str) -> object:
        """Return the live object stored under *name*.

        The returned object may be a callable, string, dict, or any other
        Python value depending on what was inserted.

        Raises:
            KeyError: If *name* is not in the store.
        """
        if name not in self._catalog:
            raise KeyError(f"'{name}' not found in store")
        return self._namespace[name]

    def get_code(self, name: str) -> str:
        """Return the raw code string for *name*.

        Useful for inspecting or editing a stored item before re-inserting it.

        Raises:
            KeyError: If *name* is not in the store.
        """
        if name not in self._sources:
            raise KeyError(f"'{name}' not found in store")
        return self._sources[name]

    def insert(self, name: str, description: str, code: str) -> None:
        """Add a new item to the store.

        Args:
            name: Unique identifier. Must not already exist in the store.
            description: Short description shown in the catalog. The agent uses
                this to decide whether to retrieve the item without loading
                the full code.
            code: Valid Python source code. May define a function, assign a
                variable, or contain any other valid Python statement(s).
                It must define a top-level symbol named exactly ``name``.

        Raises:
            ValueError: If *name* already exists, or if *code* does not define
                a top-level symbol named ``name``.
            SyntaxError: If *code* does not compile cleanly.
        """
        compile(code, "<store_insert>", "exec")
        with self._lock:
            if name in self._catalog:
                raise ValueError(
                    f"'{name}' already exists in store. Call delete() first."
                )
            self._catalog[name] = description
            self._sources[name] = code
            try:
                self._rebuild_namespace()
                if name not in self._namespace:
                    raise ValueError(
                        f"Inserted code for '{name}' must define "
                        f"a top-level symbol named '{name}'."
                    )
            except Exception:
                # Keep insert atomic if code execution fails.
                self._catalog.pop(name, None)
                self._sources.pop(name, None)
                self._rebuild_namespace()
                raise

    def delete(self, name: str) -> None:
        """Remove the item stored under *name*.

        Raises:
            KeyError: If *name* is not in the store.
        """
        with self._lock:
            if name not in self._catalog:
                raise KeyError(f"'{name}' not found in store")
            del self._catalog[name]
            del self._sources[name]
            self._rebuild_namespace()

    @property
    def catalog(self) -> dict[str, str]:
        """Return a ``{name: description}`` mapping for all stored items."""
        return dict(self._catalog)

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _rebuild_namespace(self) -> None:
        namespace: dict[str, object] = {}
        for item_name, code in self._sources.items():
            exec(compile(code, f"<memory_store:{item_name}>", "exec"), namespace)
        self._namespace = namespace
