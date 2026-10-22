"""The interaction protocol: the codec between the LLM and the REPL.

A :class:`BaseProtocol` is one of the three independently-pluggable components
of an agent loop (BaseREPL × BaseProtocol × BaseLLM — see
``design/design_features/basics.md``). It is the codec between the LLM and the REPL:
it *decodes* an LLM response into the code to run (``parse``), *encodes* each turn's
observation back to the model (``render_observation``), and opens the session
(``render_initial_message_list``).
"""

# Design/history (#566): landed as a small stack — 1a the object + decode seam (``parse``), 1b
# observation rendering, 1c the session opener — each a behavior-preserving lift of logic that
# previously lived inline in ``Agent``. The signatures are deliberately fat where today's coupling
# demands it (hook prompt additions, recursion/scoping context) and are transitional, not the target
# contract: they slim in later phases per ``basics.md``. The whole-``LLMResponse`` input landed for
# ``parse`` in #1146, so no method now takes pre-extracted content; the remaining #566
# generalization (a ``meta`` axis, the
# ``CodeT``/``MetaT``/``OutputT`` generics, and dropping the per-message ``lang=`` tag) stays
# tracked under #566.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from jaz._property_contract import require_properties

from .._value_rendering import _get_doc, _get_signature

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from ..llm._types import LLMResponse
    from ..repl._base import BaseREPL
    from ..repl._types import Continue


class BaseProtocol(ABC):
    """Codec between the LLM and the REPL — encode the session, decode the response.

    Pluggable like the LLM backend: a protocol is selected by configuring an instance
    (``jaz.configure(protocol=...)``). See :class:`~CodeOnlyProtocol` for the built-in
    raw-code protocol.

    Defines the decoder (:meth:`~BaseProtocol.parse`), the encode primitives
    (:meth:`~BaseProtocol.render_observation`, :meth:`~BaseProtocol.render_initial_message_list`), and the
    prompt-text seam for the wire format (:attr:`~BaseProtocol.response_format_description`).
    All four are abstract. The ``__history__`` record is the REPL's, not the protocol's: see
    :meth:`~jaz.repl.BaseREPL.update_repl_history`.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        require_properties(cls, ("response_format_description",))

    @abstractmethod
    def parse(self, llm_response: LLMResponse) -> str:
        """Decode an LLM response into the code to execute.

        Returns the extracted code.

        ``llm_response`` is the whole :class:`~jaz.llm.LLMResponse`, not a pre-extracted
        content string: a protocol reads ``.content`` itself (``str | None`` — ``None`` on a
        content-less completion) and can also reach ``.raw_response`` for ``refusal`` /
        ``finish_reason`` / ``tool_calls`` — what a tool-calling or refusal-aware protocol needs
        to decode a turn the content string alone cannot express.

        **Write this method so that it never raises.** There is no recoverable parse failure: the
        loop catches nothing here, so any exception is terminal and ends the invoke — an agent
        that emitted one bad message loses the whole run. Return your best reading instead,
        including the message verbatim when you have none, and let the REPL judge it; that is
        where malformed code surfaces anyway, as a ``SyntaxError`` inside a recoverable
        ``Continue`` the model can react to. An empty message is empty *code*, i.e. a no-op turn.
        """
        ...

    @property
    @abstractmethod
    def response_format_description(self) -> str:
        """The description of the response format — the *request* half of the codec.

        Returns the model-facing text that tells the LLM how to shape its response so that
        :meth:`~BaseProtocol.parse` can decode it (for the default: "respond with ONLY code, no fences").
        The system-prompt template splices this in verbatim as an opaque
        ``{{ response_format_description }}`` block, so the accept half (:meth:`~BaseProtocol.parse`) and the
        request half (this) are owned by the *same* protocol and move together — a protocol
        cannot instruct one wire format while parsing another. The template still owns the
        surrounding wording and placement; only the format block itself belongs to the codec.

        Must return a non-empty string: :class:`CodeOnlyProtocol` refuses an empty or non-string
        return with ``ValueError``, as it does for every description seam, since the template
        renders this block unconditionally.
        """
        ...

    # The seam takes no ``config``. It used to, because the rendering settings were read
    # ambiently off ``config.protocol.*`` at each call; once they became constructor params a
    # protocol reads its own state and the argument was dead weight that still *looked* live —
    # the next reader would reasonably assume something consulted it.
    #
    # EXECUTIVE CALL (user, 2026-08-08): removed outright rather than retained for implementers.
    # It is an ABC signature break for an out-of-tree protocol, taken deliberately: a parameter
    # kept "in case someone wants it" is indistinguishable from one that is load-bearing, and
    # this stack's whole argument is that a component should be configured by construction. A
    # protocol that genuinely needs invoke-time context should receive exactly that context as a
    # named argument, not the whole config.
    #
    # ``repl`` was removed the same way (#1055): render_observation never read it — the
    # truncation advice that once justified the param moved onto the protocol before being removed
    # outright — and a dead param that *looks* live invites the same wrong assumption. This also lets
    # TrajectoryReplay reproduce an observation by *calling* this seam (which owns whatever transform,
    # abbreviation included, the protocol applies) instead of re-deriving the truncation from a
    # CodeOnlyProtocol attribute the ABC never declared; the seam must render from
    # ``exec_result`` alone for that to be driveable without a live REPL.
    @abstractmethod
    def render_observation(self, exec_result: Continue) -> list[dict[str, Any]]:
        """Render a (continuing) REPL execution result into the next turn's message(s).

        Returns the observation as one or more ready messages — the seam speaks the LLM's
        own message shape directly rather than a bare ``str`` body. A message is an
        OpenAI-shaped mapping, ``{"role": ..., "content": ...}``, plus whatever extra keys
        the configured provider accepts; JAZ neither validates nor closes the shape. The
        common case is a single user message carrying the text body, which is the ``CodeOnlyProtocol``
        single-REPL path and the only shape any in-tree protocol returns. The list is
        load-bearing for protocols whose observation is inherently *multiple* messages or a
        multi-block message: one ``tool`` message per parallel tool call (OpenAI requires one
        result per ``tool_call_id``), or a single message carrying several ``tool_result`` /
        image content blocks (Anthropic returns all parallel results together in one user
        message). A ``str`` return could express neither.

        The driver appends the returned messages verbatim — persisted history is pure protocol
        output, with no per-turn footer folded in. Hook prompt additions, when present, are the
        *driver's* concern, not the protocol's.
        """
        # The list return is the shape specified in ``design/design_features/basics.md``, landed
        # while ``CodeOnlyProtocol`` is still the only implementer rather than re-breaking this
        # ABC once a parallel-tool-call protocol needs it. #634 removed the per-turn footer (the
        # budget-status / "Enter your next REPL input" block) that used to be folded in here,
        # which is what makes the verbatim-append guarantee true. ``basics.md`` is also where
        # "per-turn warnings are hook additions the driver concatenates" is settled.
        ...

    @abstractmethod
    def render_initial_message_list(
        self,
        inputs: Mapping[str, object],
        scope: Mapping[str, object],
        repl: BaseREPL[Any],
        *,
        subinvoke: Callable[..., object] | None,
        repl_history_withheld: bool,
    ) -> list[dict[str, Any]]:
        """Encode the session opening into the initial message list (system + user).

        Rendered once per session: the REPL self-description, safety guidance, and the
        input blocks. There is no distinguished ``task`` argument — the task prompt is an
        ordinary entry in ``inputs`` (conventionally named ``task``), rendered as one
        ``<name type="...">`` block like any other input.

        The **response-format description** (the "respond with ONLY code" block that is the
        *request* half of what :meth:`~BaseProtocol.parse` accepts) is owned by the protocol via
        :meth:`~BaseProtocol.response_format_description` and spliced into the system-prompt template as an opaque
        ``{{ response_format_description }}`` block. The template still owns the surrounding wording and
        layout, but the codec's accept half (``parse``) and request half live in the same
        protocol and cannot silently desync.

        Hook prompt additions do not flow through this seam — the driver owns them, the same
        symmetry :meth:`~BaseProtocol.render_observation` has. The arguments are the genuine
        rendering context the opener needs to render (not hook-effect outputs).

        Args:
            inputs: The invoke's inputs, each rendered as an input block.
            scope: Ambient scoped values (``jaz.scope``) in effect for this session.
            repl: The configured REPL, for its self-description.
            subinvoke: The bound sub-invoke function, or ``None`` when this invoke may not
                spawn sub-invokes. Its presence is the single signal for both the ``invoke``
                description and the delegation guidance.
            repl_history_withheld: Whether a hook withheld ``__history__`` from this invoke with
                :class:`~jaz.hooks.effects.WithholdREPLHistory`. The history section renders
                when it is ``False`` and the REPL describes a record
                (:attr:`~jaz.repl.BaseREPL.repl_history_description` is not ``None``), and is
                omitted otherwise.

        As with :meth:`~BaseProtocol.render_observation`, the driver folds any hook ``AddInputs`` /
        ``DropInputs`` into the REPL and prompt — not the protocol's job.
        """
        ...

    # Two removals here were genuinely public breaks — the required kw-only
    # ``input_display_overrides`` parameter, and ``parse``'s old ``ParsedCode`` tuple return:
    # ``BaseProtocol`` is in ``jaz.protocol.__all__``, so an out-of-tree protocol declaring the
    # dead parameter still *defines* fine and dies at the driver's call site, and one still
    # returning the old tuple dies in ``Agent`` as ``AttributeError: 'tuple' object has no
    # attribute 'splitlines'``. Deliberately shipped bare, with no ``__init_subclass__`` tripwire: #1367 settled
    # the rule for this stack — the ``a`` in ``0.2.0a3`` is for alpha, alphas carry no
    # compatibility promise, and a guard whose only job is to raise a better error for a caller
    # that should not exist goes. (That rule covers changes out of an alpha; what the betas from
    # 0.2.0b1 on promise is not yet decided.) #1363/#1364/#1367 removed the last three such guards; adding a
    # fourth here would contradict the rule they just established. The revival pin lives in
    # ``tests/protocol/test_code_only_protocol.py`` instead, where it costs nothing at runtime.

    def get_subinvoke_description(self, subinvoke: Callable[..., object]) -> str:
        """The agent-facing description of the recursive sub-invoke tool (bound as ``invoke``).

        Rendered into the system prompt as the ``invoke`` catalog entry. Must return a non-empty
        string; :class:`CodeOnlyProtocol` refuses an empty or non-string return with ``ValueError``,
        since a bound tool described as nothing would render an empty block. The default introspects
        the bound closure — its signature and docstring ARE the description, so the two cannot
        drift. Not abstract, unlike :attr:`response_format_description`: this default is *derived
        from the tool itself* rather than chosen, so there is no decision for an author to make
        and forcing one would buy nothing.

        A protocol overrides this to author the description explicitly — typically to advertise a
        NARROWER surface than the closure actually supports. The motivating case is a
        ``restrict_invoke_args``-style guard that vetoes non-literal / positional-hook ``invoke``
        calls at runtime: the prompt should then describe an inputs-only signature even though the
        closure still accepts hooks. Owning the words here — rather than swapping the executable
        closure — keeps "what the tool can do" separable from "what the agent is told it can do";
        an overriding author owns keeping the two in step with the guard that motivates the
        override.
        """
        # One formatted line, not a namespace walk: the only value this ever receives is the
        # subinvoke closure, so a recursive tool-catalog renderer (entry budget, cycle guard,
        # compact mode) would be unreachable machinery around a single entry (#1393 holds the
        # collapse and the byte-identity test).
        #
        # That single entry rests on a precondition, not on "a callable is not a namespace": a
        # function has a `__dict__`, so a closure carrying a public attribute would be a second
        # catalog line (`- \`invoke.helper()\`: ...`) that this one-liner silently drops.
        # `vars(subinvoke)` is EMPTY, and `test_the_subinvoke_carries_no_public_attributes` pins
        # that, so attaching one to the tool fails there rather than vanishing from the prompt.
        return f"- `invoke{_get_signature(subinvoke)}`: {_get_doc(subinvoke)}"
