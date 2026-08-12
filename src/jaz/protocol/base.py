"""The interaction protocol: the codec between the LLM and the REPL.

An :class:`InteractionProtocol` is one of the three independently-pluggable components
of an agent loop (REPL × InteractionProtocol × LLM — see
``design/design_features/basics.md`` and issue #566). It is the codec between the LLM
and the REPL: it *decodes* an LLM response into the code to run, and (in later phases)
*encodes* the session into messages.

This is **Phase 1** of the refactor, landed as a small stack: 1a introduced the object
and the **decode** seam (``parse``), 1b the per-turn **observation** rendering
(``render_observation``), and 1c the session **opener** (``render_initial_message_list``).
Each is a behavior-preserving lift of logic that previously lived inline in ``Agent``.
The signatures are deliberately fat where today's coupling demands it (hook prompt
additions, recursion/scoping context) and are transitional, not the target contract:
they slim in later phases per ``basics.md``, with the generalization (whole-``LLMResponse``
input + a ``meta`` axis, the ``CodeT``/``MetaT``/``OutputT`` generics, and dropping the
per-message ``lang=`` tag) tracked under #566.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .._display import DisplayText
    from .._llm_client import LLMResponse
    from ..library import Library
    from ..providers.base import MessageDict
    from ..repl.base import REPL
    from ..repl.types import Continue, ExecResult


class ParsedCode(NamedTuple):
    """The success result of :meth:`InteractionProtocol.parse` — the decoded input.

    A named type rather than a bare ``tuple``: the slots are self-documenting, call sites
    aren't coupled to position, and a new field can be added without silently breaking
    every custom ``parse``. This is the seam where ``basics.md``'s eventual
    ``ParsedCode[CodeT, MetaT]`` lands — the ``meta`` axis (tool-calling / ReAct payload)
    slots in here as a later field without another signature break.
    """

    code: str
    exec_timeout_override: float | None


class InteractionProtocol(ABC):
    """Codec between the LLM and the REPL — encode the session, decode the response.

    Pluggable like ``llm_client``: a protocol is selected via
    ``Config.interaction_protocol`` (a registered name or a direct instance). See
    :class:`~jaz.protocol.default.DefaultProtocol` for the built-in XML
    raw-code protocol.

    Defines the decoder (:meth:`parse`), the encode primitives
    (:meth:`render_observation`, :meth:`render_initial_message_list`), and the
    ``__repl_history__`` record projection (:meth:`build_history_entry`).
    """

    @classmethod
    def from_dict(
        cls, params: Mapping[str, object] | None = None
    ) -> InteractionProtocol:
        """Build this protocol from its params — ``REGISTRY[tag].from_dict(params)``.

        The default maps params to constructor kwargs. Override for a protocol whose config
        shape does not match its ``__init__``.
        """
        return cls(**(params or {}))

    @abstractmethod
    def parse(self, response_content: str | None, repl_language: str) -> ParsedCode:
        """Decode an LLM response into a :class:`ParsedCode` ``(code, exec_timeout_override)``.

        On success returns the extracted ``code`` (already HTML-unescaped) and an
        optional per-exec timeout. The response carries no language attribute — an
        agent has a single REPL, so ``repl_language`` is passed for protocols that want it but
        the default no longer validates against it. On malformed output (empty response, no
        block, ambiguous multiple blocks) **raises**
        :class:`~jaz.exceptions.LLMResponseParseError` — the recoverable parse error
        (basics.md's "ParseError"). The loop catches it *narrowly* and turns it into
        feedback for the model; any *other* exception (e.g. a bug in a custom protocol)
        is terminal and propagates. Raising — rather than returning an error value —
        keeps custom protocols trivial (return the happy path, raise on bad input) and
        puts the recoverable/terminal split on the exception *type*.
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
    @abstractmethod
    def render_observation(
        self,
        exec_result: Continue,
        iteration: int,
        repl: REPL,
    ) -> list[MessageDict]:
        """Render a (continuing) REPL execution result into the next turn's message(s).

        Returns the observation as one or more ready messages — the seam speaks the LLM's
        own message shape directly rather than a bare ``str`` body. The common case is a
        single user message carrying the text body, which is the ``DefaultProtocol`` / XML
        single-REPL path; that same protocol returns **two** messages when the output was
        truncated, emitting the truncation advice separately so the observation stays the raw
        output (#928) — so the list return is exercised by the default implementer, not only by
        the hypothetical protocols below. The
        list is load-bearing for protocols whose observation is inherently *multiple*
        messages or a multi-block message: one ``tool`` message per parallel tool call
        (OpenAI requires one result per ``tool_call_id``), or a single message carrying
        several ``tool_result`` / image content blocks (Anthropic returns all parallel
        results together in one user message). A ``str`` return could express neither, so
        the final ``basics.md`` shape is landed now, while ``DefaultProtocol`` is the only
        implementer, rather than re-breaking this ABC once such a protocol lands.

        The per-turn footer and hook warnings (budget status, "Enter your next REPL
        input", must-exit / iteration notices) are the *driver's* concern, not the
        protocol's (``basics.md``: "per-turn warnings are hook additions the driver
        concatenates"). The driver appends the returned messages and folds the footer into
        the last message when it is a plain user-text message (the single-observation
        case), else appends the footer as its own trailing message.
        """
        ...

    @abstractmethod
    def render_initial_message_list(
        self,
        inputs: Mapping[str, object],
        scope: Mapping[str, object],
        repl: REPL,
        *,
        jaz_library: Library | None,
        depth: int,
        recursion_available: bool,
        repl_language: str,
        input_display_overrides: Mapping[str, DisplayText] | None,
    ) -> list[MessageDict]:
        """Encode the session opening into the initial message list (system + user).

        Rendered once per session: the REPL self-description, safety guidance, and the
        input blocks. There is no longer a distinguished ``task`` argument — the prompt
        that used to be ``task`` is now an ordinary entry in ``inputs`` (conventionally
        named ``task``), rendered as one ``<name type="...">`` block like any other input
        (#538). Phase 1c originally moved the inline assembly from ``Agent.invoke`` here.

        One coupling remains transitional, tracked for a later phase:

        - The **wire-format instructions** (the "respond with ONLY code" block
          that is the *request* half of what :meth:`parse` accepts) are NOT rendered here
          — they live in ``config.protocol.system_prompt_template``, which this method just passes
          through. So the codec's accept half (``parse``) and request half (the template)
          can desync; making the protocol own that block is tracked in #639.

        Hook prompt additions no longer flow through this seam: ``AddInstructionPrompt`` was
        removed in favour of ``AddMessages`` folded at query time (#660), and ``AddSystemPrompt``
        was removed together with ``ParentUpdatesHook``, its sole emitter (#703). That restored
        symmetry with :meth:`render_observation` — the driver owns hook additions, not the
        protocol — and closed #640. The signature stays wider than the illustrative sketch in
        basics.md because it carries genuine rendering context (``cur_recursion_depth`` /
        ``max_recursion_depth`` for delegation guidance, ``scoped_keys`` /
        ``input_display_overrides`` for which inputs to show and how, ``repl_config`` /
        ``jaz_library`` for the REPL/tool self-descriptions). Unlike the removed hook
        additions, these are inputs the opener needs to *render*, not hook-effect outputs that
        belong on the driver — so no further slimming is pending here.

        As with :meth:`render_observation`, the driver folds any hook ``AddInputs`` /
        ``DropInputs`` into the REPL and prompt — not the protocol's job. (There is no longer
        an iteration-0 footer: #634
        removed the per-turn budget-status / next-input footer entirely.)
        """
        ...

    @abstractmethod
    def build_history_entry(
        self, llm_response: LLMResponse | None, exec_result: ExecResult
    ) -> object:
        """Project one turn into its ``__repl_history__`` entry — the **record** seam.

        The return type is ``object`` on purpose: the **entry shape is the protocol's**, not the
        core's — each protocol records whatever it likes, and the driver only holds a
        ``list[object]`` and surfaces it as ``__repl_history__``. The default is
        :class:`~jaz.protocol.default.REPLHistoryEntry`; a custom protocol may return anything.

        Sibling to :meth:`render_observation` — both encode a turn into an agent-facing surface —
        differing in destination (the in-band next-turn observation vs. the persistent
        ``__repl_history__`` projection) and in input breadth: ``render_observation`` takes the
        recoverable ``Continue`` (a continuing turn always has one), while this records the broader
        ``ExecResult`` (a turn's finalized result may be terminal). Keeping both on the protocol puts
        "how a turn is represented to the agent" in one owner, so a protocol that customizes error
        rendering can't silently desync its history representation.

        Part of the required contract like :meth:`parse` / :meth:`render_observation` (the base
        stays fully abstract — no default projection lives here); :class:`~jaz.protocol.default.
        DefaultProtocol` implements the standard entry. ``llm_response`` is the whole
        ``LLMResponse`` rather than the pre-extracted content string — the #566 "whole-``LLMResponse``
        input" target — so an implementation may fold in token counts / cost / provider ``extra``
        without another signature break. The ``| None`` is an **external-only affordance** (a caller
        that has no response object): the in-loop caller ``_record_history`` guards on the
        response being present and then always synthesizes an ``LLMResponse``, so the framework
        never passes ``None`` here — a future reader need not hunt for a loop path that does.

        The core agent loop is the single writer of ``__repl_history__`` and appends one entry per
        iteration from the finalized (post-effect-composition) result via this method — see
        ``agent.do_one_repl_iteration`` / ``_record_history``.
        """
        ...
