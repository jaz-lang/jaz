import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from jaz._property_contract import require_properties
from jaz.llm._types import LLMResponse

from ._types import ExecResult


class BaseREPL[StateT](ABC):
    """Base contract for REPL implementations used by LM agents.

    Generic in ``StateT``, the per-invoke state this REPL works in terms of: :meth:`initialize`
    returns one and :meth:`exec` / :meth:`aexec` / :meth:`add_to_namespace` / :meth:`drop_from_namespace`
    / :meth:`namespace` take it back. An implementation declares its own state type and writes
    ``class MyREPL(BaseREPL[MyREPLState])``; nothing constrains what that type is, so a REPL whose
    namespace lives in another process or another machine holds whatever it actually needs (a
    connection, a container handle) rather than a namespace dict it would leave empty.

    The REPL knows nothing about the return type or its validation. Return-*type* checking,
    return-*value* validation, and REPL-*code* validation all live in hooks (``ReturnType`` /
    ``ValidateReturn`` / ``ValidateREPLCode``) that act on the effect path; the REPL only
    produces a ``Return`` carrying the value and the hooks decide whether it's acceptable.

    The ``__history__`` record is the REPL's: :meth:`initialize` binds it, the agent loop
    hands each turn to :meth:`update_repl_history`, and :attr:`~BaseREPL.repl_history_description`
    describes it to the agent, or is ``None`` for a REPL that binds no such name. The system
    prompt describes the record exactly when the invoke binds it — the description is not ``None``
    and no hook withheld it with :class:`~jaz.hooks.effects.WithholdREPLHistory` — so the prompt
    never advertises a name the agent cannot reference.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        require_properties(cls, ("description", "repl_history_description"))

    # The hooks-on-the-effect-path split is #528/#568.

    # EXECUTIVE CALL (user, 2026-09-13): the per-invoke state is a TYPE PARAMETER, not a shared
    # base class. The alternative on the table — and the one briefly taken — was to keep a single
    # concrete `REPLState` in `jaz.repl.__all__`, on the reading that a type a public ABC's own
    # abstract methods hand an implementer must carry a promise. That reading is right about the
    # obligation and wrong about how to discharge it: a *concrete* shared state forces every
    # implementation through one class's fields, and the fields were the host Python REPL's. An
    # out-of-tree container-backed REPL was the proof — it inherited a namespace dict it
    # documented as unused, and had to give every one of its own fields a `| None` default because
    # the inherited frozen base had already defaulted its own. Making the state a type parameter
    # discharges the obligation instead by *naming nothing*: the implementer's own state type is
    # the one in their signatures, so `jaz.repl` advertises no type it does not own.
    #
    # `StateT` is deliberately UNBOUNDED — not even a `Protocol`. The framework never reads a
    # field off the state: `Agent` only threads it from `initialize` into `exec`/`add_to_namespace`/
    # `drop_from_namespace`/`namespace`, hooks never see it, and `subinvoke`/`session_id` are passed
    # *into* `initialize` as parameters, so nothing needs them read back out. `session_id` is
    # read again only by `PythonREPL` itself, off its own state, which a bound would not help
    # with. A bound with nothing behind it is a constraint implementers must satisfy for no
    # reader's benefit.
    #
    # EXECUTIVE CALL (user, 2026-09-18): hooks DO see the namespace — as a read-only snapshot on
    # `REPLExecSend.namespace`, produced by `namespace()` below — but still never the state
    # object. This refines rather than reverses the call above: every `*Send` event carries the
    # result of committing its input, and for the REPL the result is the
    # namespace the committed code runs against, which only the REPL can read off its own state.
    # Three shapes were weighed. The opaque state object itself on the event was rejected: it
    # would hand hooks `PythonREPLState.exec_globals`, a LIVE dict, through a supposedly read-only
    # event, and force every hook to downcast to a concrete state type. Names only (a
    # `frozenset[str]`) was rejected as too little: it answers "is `x` bound" but not what an
    # observer recording the turn wants, which is what `x` was. A method returning a mapping the
    # framework snapshots keeps the state opaque and the type parameter unbounded at the cost of
    # one more method every REPL must answer — and a REPL whose namespace lives elsewhere may
    # answer with less.

    @property
    @abstractmethod
    def description(self) -> str:
        """This REPL's instructions for the system prompt, rendered from its own settings.

        A REPL is its own configuration, so what it tells the agent comes from the object that
        enforces it — ``PythonREPL`` offers ``raise`` as a finish only when it allows one.

        Must be a non-empty string: the system prompt refuses an empty or non-string value with
        ``ValueError`` rather than rendering an empty ``<repl_spec>`` element.
        """
        # A property on the instance rather than a classmethod taking a config mapping: the
        # mapping form had no schema (a misspelled key rendered the default, silently — telling
        # an agent it may `raise` from a REPL that forbids it) and had to reconstruct every
        # constructor default for an absent key, while the instance already holds the effective
        # values. The one caller with no instance (a bare prompt preview) renders an unconfigured
        # `PythonREPL()`, which is cheap: the REPL is stateless, so constructing one is
        # constructing a config.
        ...

    @property
    @abstractmethod
    def repl_history_description(self) -> str | None:
        """The agent-facing description of this REPL's ``__history__`` record, or ``None``.

        Returns the prompt text telling the agent what ``__history__`` holds and what an entry's
        fields are; the system-prompt template splices it in verbatim and owns the framing. A
        REPL with an in-process Python namespace binds the name and describes it; one whose
        namespace lives elsewhere cannot bind it and returns ``None``, and the system prompt then
        omits its ``__history__`` section. Otherwise the text must be non-empty: the prompt
        refuses ``""`` or a non-string with ``ValueError``, as it does every description seam,
        and drops the section on ``None`` alone.
        """
        # One property answers both "is it bound" and "what is it", so the two cannot disagree:
        # the opener renders this text exactly when the invoke binds the name (not `None`, and no
        # `WithholdREPLHistory`), and `initialize` binds the name on the same two facts. Abstract
        # rather than defaulted: a REPL that binds no such name is the case that must state
        # itself, and a concrete default, text or `None`, would let a REPL inherit a lie in one
        # direction or the other.
        #
        # EXECUTIVE CALL (user, 2026-09-18): the prompt follows the REPL rather than a template a
        # config author pairs by hand. This REVERSED an earlier call that kept the prompt/REPL
        # pairing a documented assumption and left suppression to a template block
        # (`repl_history`, since removed), weighed against removing the block outright (which
        # would have left only copied templates and prompt-rewriting hooks) and against leaving
        # both. Taken because the tree already held a REPL that cannot bind the name (the eval
        # harness's Docker REPL), whose pairing nothing enforced. Why the words and the fact are
        # one property rather than a bool beside a protocol-side description is in
        # `jaz.repl._repl_history`.
        ...

    @abstractmethod
    def initialize(
        self,
        inputs: dict[str, object],
        subinvoke: Callable[..., object] | None,
        session_id: str = "",
        repl_history_withheld: bool = False,
    ) -> StateT:
        """Build one invoke's state — the REPL itself stays pure configuration.

        Takes **only invoke-time arguments**. Everything that configures the REPL —
        ``exec_timeout``, the sandbox allow-lists, the finishing rules — is a constructor
        parameter, so a REPL is *configured* by construction and produces a fresh ``StateT``
        per run. The REPL holds no per-invoke state, so one configured REPL serves many invokes::

            repl = PythonREPL(exec_timeout=30.0)
            first = repl.initialize(inputs=..., subinvoke=None, session_id="a")   # this REPL's state type
            second = repl.initialize(inputs=..., subinvoke=None, session_id="b")  # an independent one
            repl.exec(first, "x = 1", "0")                                        # pass the state back to exec

        The returned state is threaded back into :meth:`exec` / :meth:`add_to_namespace` /
        :meth:`drop_from_namespace`; the REPL never mutates ``self``, so two invokes cannot bleed into
        each other by construction. An implementation chooses what the state holds — the shipped
        :class:`~jaz.repl.PythonREPL` keeps the exec namespace in it, a REPL executing elsewhere
        keeps whatever connects it to that elsewhere.

        Arguments:
            inputs: A dictionary of initial inputs to the REPL.
            subinvoke: The recursive sub-invoke primitive to bind in the REPL under the
                bare name ``invoke``, or None to withhold it (recursion disabled).
            session_id: Unique session identifier for this REPL instance.
            repl_history_withheld: Whether a hook withheld ``__history__`` from this invoke with
                :class:`~jaz.hooks.effects.WithholdREPLHistory`. A REPL that binds the name
                (:attr:`~BaseREPL.repl_history_description` is not ``None``) then binds none; a REPL
                that binds no such name ignores it. :meth:`update_repl_history` is still called
                on every turn either way.
        Returns:
            A fresh instance of this REPL's own state type, holding everything the run needs.
        """
        # A `describe_repl_history` switch on `CodeOnlyProtocol` was rejected earlier (it made
        # the codec answer a rendering question); the per-invoke fact comes from the hook effect
        # and the REPL's fact from `repl_history_description`, and the loop passes only the former.

    @abstractmethod
    def exec(
        self,
        state: StateT,
        src: str,
        code_id: str,
    ) -> ExecResult[Any]:
        # TODO: Fix static types
        """Execute REPL code against ``state``, mutating it in place.

        Errors in the executed code should be captured and returned as a recoverable
        :class:`~jaz.repl.Continue` (its ``exception`` set, its ``output`` shown to the
        agent) — with one carve-out: an exception that must escape the invoke (a
        :class:`~jaz.exceptions.FatalError`-category error, a non-``Exception``
        signal, or an enclosing exec's timeout/memory breach) **re-raises** out of
        this call instead of becoming feedback, so it propagates past the agent.

        Arguments:
            src: The REPL code to execute.
            code_id: The unique identifier for the REPL code.
        Returns:
            An :class:`~jaz.repl.ExecResult` — exactly one of :class:`~jaz.repl.Continue`
            (ran, session continues; carries ``output`` and an optional recoverable
            ``exception``), :class:`~jaz.repl.Return` (finished by returning
            ``return_value``), or :class:`~jaz.repl.Raise` (finished by raising
            ``exception``). The REPL state is mutated in place.
        """

    async def aexec(
        self,
        state: StateT,
        src: str,
        code_id: str,
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
            state,
            src,
            code_id,
        )

    @abstractmethod
    def update_repl_history(
        self,
        state: StateT,
        llm_response: LLMResponse,
        exec_result: ExecResult[Any],
    ) -> None:
        """Update the REPL history held in ``state`` with one turn.

        Called by the agent loop once per iteration, after the turn's result is final: the
        hooks' exit-time overrides are applied, so the result is what the agent was told
        rather than what the REPL alone saw. Every turn the model answered is handed here,
        including one a hook resolved without running ``exec``.

        It is called on every turn even when the invoke withheld ``__history__``
        (``repl_history_withheld`` at :meth:`initialize`), and on a REPL that binds no such
        name, so a REPL that keeps a history for some use other than the agent's can still do
        so. Withholding decides what the agent can reach, not what the REPL keeps.
        :class:`~jaz.repl.PythonREPL` keeps nothing on a withheld invoke.

        Arguments:
            state: This REPL's per-invoke state, as returned by :meth:`initialize`.
            llm_response: The whole response that produced the turn, so an implementation
                may record token counts, cost or provider ``extra`` as well as the text.
            exec_result: The turn's finalized :class:`~jaz.repl.ExecResult`.
        """
        # A separate call rather than a write inside `exec`: recording from `exec`'s own view
        # would show the agent one story in its observation and another in `__history__` (#448).
        # Why the loop has what the entry needs and the REPL owns it: `jaz.repl._repl_history`.
        #
        # Named `update`, not `record`/`add`/`append` or anything `*_entry` (the user's call,
        # 2026-10-21): the method hands the REPL one turn and says nothing about what it keeps.
        # A REPL may keep nothing (`DockerPythonREPL`), append an entry per turn (`PythonREPL`),
        # keep a rolling window or a running summary; the accumulating verbs and `entry` each
        # name one of those. `record_repl_history`, the earlier name, also read as recording the
        # whole history rather than one turn of it.
        #
        # Called unconditionally, not skipped by the loop on a withheld invoke (the user's call,
        # 2026-10-21). Skipping would let the loop enforce that nothing is kept, but it would
        # widen `WithholdREPLHistory` from "the agent cannot reach it" to "the REPL keeps none",
        # and gating on withholding without also gating on a `None` description is
        # inconsistent, while gating on both puts the described-iff-bound logic back in the
        # loop. The cost is that a REPL honouring the flag must carry it from `initialize` in
        # its state, as `PythonREPL` does with `repl_history=None`. Pinned, for both a withheld
        # invoke and a REPL with a `None` description, by
        # `test_the_turn_is_handed_over_when_the_agent_has_no_history`.

    # No `add_inputs` here: a post-init seam for injecting *inputs* that nothing could reach.
    # `AddInputs` is valid at `InvokeEnter` only, so the loop folds it into the input dicts
    # before `initialize` and emitting it later raises `InvalidEffectError` — the seam was
    # unreachable, not merely uncalled. Removed rather than deprecated because JAZ has no
    # stable release yet (EXECUTIVE CALL, user, 2026-09-12). The `AddInputs` *effect* is
    # untouched; it is the supported path.

    @abstractmethod
    def add_to_namespace(self, state: StateT, entries: dict[str, object]) -> None:
        """Bind ``entries`` **raw** into the REPL namespace (the per-turn ``AddToNamespace``).

        Binds the given objects **verbatim** into the REPL namespace and leaves the prompt
        untouched — a *variable* is not an *input*, so nothing renders a prompt block for it.
        Applied at ``REPLExecEnter`` before each turn's code runs, so a hook may re-bind a name
        every turn. Naming the sandbox key ``__builtins__`` raises ``KeyEditDeniedError`` (at effect
        composition, and again in implementations on a direct call).

        Args:
            entries: Dictionary of name -> value to bind verbatim.

        Note:
            Implementation-specific — a REPL that doesn't maintain a mutable namespace may
            treat this as a no-op.
        """

    @abstractmethod
    def get_namespace(self, state: StateT) -> Mapping[str, object]:
        """The names currently bound in the REPL namespace, with their values.

        Read twice per turn: at ``REPLExecEnter`` (before that turn's ``AddToNamespace`` /
        ``DropFromNamespace``, populating ``REPLExecEnter.namespace`` — the proposal the effects
        edit) and at ``REPLExecSend`` (after them and before the code runs, populating
        ``REPLExecSend.namespace`` — the namespace the committed code is about to run
        against). Returns the names the agent's code is run against at the top level: invoke
        inputs, hook-bound names and the agent's assignments, plus, for a REPL with a builtins
        layer (``PythonREPL``'s ``__builtins__``), that layer read-only. The top-level names other
        than that layer are exactly what :meth:`drop_from_namespace` can edit.

        The returned mapping is copied by the caller before anything else touches the
        namespace, so an implementation may return a live view; the variables' values are never
        copied, so a caller holding the snapshot holds the live objects (``PythonREPL`` shows its
        builtins layer as a copy, so it does not change under a kept snapshot). A REPL whose namespace it
        cannot read cheaply (one in another process) may return a partial or empty mapping,
        and should document which.

        Args:
            state: This REPL's per-invoke state, as returned by :meth:`initialize`.

        Returns:
            A name-to-value mapping of the bound names.
        """

    @abstractmethod
    def drop_from_namespace(
        self,
        state: StateT,
        names: Iterable[str],
        allow_missing: Iterable[str] = (),
    ) -> None:
        """Remove ``names`` from the REPL namespace — the namespace counterpart of
        ``add_to_namespace``.

        Applies a ``DropFromNamespace`` effect: each name is unbound so the agent's next
        turn sees it as undefined (referencing it raises ``NameError``). Dropping a name
        that isn't currently bound raises ``MissingDropTargetError`` (a hook bug — a drop
        of an absent name), UNLESS the name is in ``allow_missing`` (the set of names whose
        ``DropFromNamespace`` opted into tolerating absence), in which case it is skipped. Naming
        the compiler-sandbox key ``__builtins__`` raises ``KeyEditDeniedError`` (at effect
        composition, and again in implementations on a direct call), so import/attribute/
        builtins policy can't be unbound. An allowed builtin (``len``) that no variable shadows
        is policy, not a bound name, so naming one is a missing target, as ``del len`` raises
        ``NameError`` in Python. An implementation checks every name before removing any, so a
        call that raises removes nothing.

        Args:
            names: The variable names to remove from REPL state.
            allow_missing: Names a drop may find unbound (from
                ``DropFromNamespace(..., allow_missing=True)``); an absent one is skipped, not
                raised.

        Note:
            Implementation-specific — a REPL that doesn't maintain a mutable namespace
            may treat this as a no-op.
        """
