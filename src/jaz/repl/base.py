import asyncio
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any, Self

from jaz._invoke_tool import InvokeTool

from .types import ExecResult


class BaseREPL(ABC):
    """Base contract for REPL implementations used by LM agents.

    The REPL knows nothing about the return type or its validation. Return-*type* checking,
    return-*value* validation, and REPL-*input* validation all live in hooks (``ReturnType`` /
    ``ValidateReturn`` / ``ValidateREPLInput``) that act on the effect path; the REPL only
    produces a ``Return`` carrying the value and the hooks decide whether it's acceptable.
    """

    # The hooks-on-the-effect-path split is #528/#568.

    # REPL-specific description shown in the system prompt
    # Subclasses should override this with their specific instructions
    description: str

    # Whether this REPL type surfaces ``__history__`` in its namespace. A static
    # *capability* (not session state): the core agent loop only builds and tracks a
    # history list for REPLs that advertise support, so a REPL that keeps no history
    # (e.g. a non-Python REPL) is skipped cleanly rather than
    # accumulating a list nothing can read.
    maintains_repl_history: bool = False

    @classmethod
    def get_description(cls, config: Mapping[str, Any] | None = None) -> str:
        """Get the REPL description for the given configuration.

        Subclasses can override this to return different descriptions based on
        the configuration (e.g., PythonREPL returns different descriptions for
        single-statement vs multi-statement mode).

        Args:
            config: Optional REPL-specific configuration dict.

        Returns:
            The REPL description string to show in the system prompt.
        """
        return cls.description

    @classmethod
    def from_dict(cls: type[Self], params: Mapping[str, Any] | None = None) -> Self:
        """Build a configured (not yet initialized) REPL — ``REGISTRY[tag].from_dict(params)``.

        Params this REPL does not declare as constructor arguments are ignored rather than
        raising. Override for a REPL whose config shape does not match its ``__init__``.
        """
        # Filtered because `repl.configs[language]` is an open, host-authored bag: an eval YAML
        # may carry a key meant for a different REPL, or one whose feature has since been
        # removed. Raising TypeError on those would make an unrelated stale key fatal at invoke
        # time. The keys a REPL *does* declare still reach it, which is the contract that
        # matters.
        known = cls.construction_keys()
        return cls(**{k: v for k, v in (params or {}).items() if k in known})

    @classmethod
    def construction_keys(cls) -> frozenset[str]:
        """Which ``repl.params`` keys configure this REPL at construction.

        Derived from ``__init__``'s declared parameters, exactly as for an LLM backend or a
        protocol — so a REPL declares its config simply by declaring its constructor.
        """
        from ..llm.base import declared_init_keys

        return declared_init_keys(cls)

    @abstractmethod
    def initialize(
        self,
        inputs: dict[str, object],
        invoke_tool: InvokeTool | None,
        allowed_builtins: dict[str, object] | None = None,
        session_id: str = "",
        initial_repl_history: list[object] | None = None,
    ) -> Self:
        """Return a **new** REPL carrying this one's configuration plus one invoke's state.

        Takes **only invoke-time arguments**. Everything that configures the REPL —
        ``exec_timeout``, the sandbox allow-lists, the finishing rules — is a constructor
        parameter, so a REPL is *configured* by construction and *initialized* per run.

        **The receiver is a reusable template and is left untouched**; the returned instance is
        the one to ``exec`` against. Call it once per invoke on the same configured REPL::

            template = PythonREPL(exec_timeout=30.0)
            first = template.initialize(inputs=..., session_id="a")
            second = template.initialize(inputs=..., session_id="b")  # independent of `first`

        Implementations must not bind per-invoke state onto ``self``. Returning ``self`` would
        make the object single-use — a second call would silently replace the first run's
        namespace, invoke-tool binding and session id — which rules out holding one configured REPL
        and running many invokes from it.

        Arguments:
            inputs: A dictionary of initial inputs to the REPL.
            invoke_tool: The recursive sub-invoke primitive to bind in the REPL under the
                bare name ``invoke``, or None to withhold it (recursion disabled).
            allowed_builtins: A dictionary of allowed built-in functions and variables.
            session_id: Unique session identifier for this REPL instance.
            initial_repl_history: An empty list container the driver passes in to own
                ``__history__``, or ``None`` when history is disabled. A REPL that
                supports history surfaces this exact list object in its namespace as
                ``__history__`` (by reference); the core agent loop then retains the
                reference and is the single writer of subsequent (iteration) entries. REPLs
                without a Python namespace (a non-Python REPL) ignore it (leaving it empty),
                which signals to the driver that the REPL keeps no history.
        Returns:
            This REPL, initialized.
        """

    @abstractmethod
    def exec(
        self,
        src: str,
        input_id: str,
        exec_timeout_override: float | None = None,
    ) -> ExecResult[Any]:
        # TODO: Fix static types
        """Execute REPL input and mutate state in place.

        Errors in the executed code should be captured and returned as a recoverable
        :class:`~jaz.repl.Continue` (its ``exception`` set, its ``output`` shown to the
        agent) — with one carve-out: an exception that must escape the invoke (a
        :class:`~jaz.exceptions.FatalError`-category error, a non-``Exception``
        signal, or an enclosing exec's timeout/memory breach) **re-raises** out of
        this call instead of becoming feedback, so it propagates past the agent.

        Arguments:
            src: The REPL input to execute.
            input_id: The unique identifier for the REPL input.
        Returns:
            An :class:`~jaz.repl.ExecResult` — exactly one of :class:`~jaz.repl.Continue`
            (ran, session continues; carries ``output`` and an optional recoverable
            ``exception``), :class:`~jaz.repl.Return` (finished by returning
            ``return_value``), or :class:`~jaz.repl.Raise` (finished by raising
            ``exception``). The REPL state is mutated in place.
        """

    async def aexec(
        self,
        src: str,
        input_id: str,
        exec_timeout_override: float | None = None,
    ) -> ExecResult[Any]:
        """Async version of exec(). Runs exec() in a thread pool to avoid blocking
        the event loop.

        Contextvar propagation: asyncio.to_thread copies the current context
        snapshot into the thread (reads see the caller's values), but any
        contextvar a hook rebinds *inside* the thread is not visible back on
        the event loop — propagation is one-way (inward only).

        Concurrency note: to_thread uses the default ThreadPoolExecutor
        (bounded to roughly min(32, cpu_count+4) threads). Under a wide
        asyncio.gather of agents that are all in REPL exec simultaneously,
        excess tasks queue behind the thread limit rather than running in
        true parallel — real concurrency is capped at the executor size.
        """
        return await asyncio.to_thread(
            self.exec,
            src,
            input_id,
            exec_timeout_override,
        )

    def get_finish_command_hint(self) -> str:
        """Text like '`return ...`' or '`return ...` or `raise ...`'."""
        return "`return ...`"

    @abstractmethod
    def add_inputs(self, inputs: dict[str, object]) -> None:
        """Add inputs to the REPL environment after initialization.

        This method allows dynamically adding variables, functions, or other
        Python objects to the REPL state after the REPL has been initialized.
        This is useful for hooks to inject context or utilities into the REPL
        at different stages of execution (invoke, iteration, or execution level).

        Args:
            inputs: Dictionary of variable name -> value to add to REPL state

        Note:
            This method should be idempotent - calling it multiple times
            with the same key should update the value (last write wins).
            The behavior is implementation-specific - some REPLs may not
            support this operation (e.g., a non-Python REPL).

            Implementations that bind Python objects (e.g. PythonREPL) MUST apply the
            same ``__jaz_get__`` payload substitution that initialization does, so an
            injected ``Library``/``jaz.wrap`` binds its payload rather than the wrapper.
            Hook authors can therefore pass the wrapper directly to ``AddInputs``.
        """

    @abstractmethod
    def add_variables(self, variables: dict[str, object]) -> None:
        """Bind ``variables`` **raw** into the REPL namespace (the per-turn ``AddVariables``).

        The namespace-level counterpart of ``add_inputs``: where ``add_inputs`` is an
        *input*-level operation that applies ``__jaz_get__`` payload substitution (so an
        injected wrapper binds its payload) and is conceptually part of the prompt, this binds
        the given objects **verbatim** into the REPL namespace and leaves the prompt untouched.
        Applied at ``REPLExecEnter`` before each turn's code runs, so a hook may re-bind a name
        every turn. The sandbox key ``__builtins__`` must never be bound this way (enforced at
        effect composition and again in implementations, defensively).

        Args:
            variables: Dictionary of variable name -> value to bind verbatim.

        Note:
            Implementation-specific — a REPL that doesn't maintain a mutable namespace may
            treat this as a no-op.
        """

    @abstractmethod
    def drop_variables(
        self, names: Iterable[str], allow_missing: Iterable[str] = ()
    ) -> None:
        """Remove ``names`` from the REPL namespace (the inverse of ``add_inputs``).

        Applies a ``DropVariables`` effect: each name is unbound so the agent's next
        turn sees it as undefined (referencing it raises ``NameError``). Dropping a name
        that isn't currently bound raises ``MissingDropTargetError`` (a hook bug — a drop
        of an absent name), UNLESS the name is in ``allow_missing`` (the set of names whose
        ``DropVariables`` opted into tolerating absence), in which case it is skipped. The
        compiler-sandbox key ``__builtins__`` is never dropped even if named (enforced
        upstream at effect composition and again here, defensively), so import/attribute/
        builtins policy can't be unbound.

        Args:
            names: The variable names to remove from REPL state.
            allow_missing: Names exempt from the missing-target check (from
                ``DropVariables(..., allow_missing=True)``); an absent one is skipped, not raised.

        Note:
            Implementation-specific — a REPL that doesn't maintain a mutable namespace
            may treat this as a no-op.
        """
