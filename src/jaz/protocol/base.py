"""The interaction protocol: the codec between the LLM and the REPL.

A :class:`BaseProtocol` is one of the three independently-pluggable components
of an agent loop (BaseREPL × BaseProtocol × BaseLLM — see
``design/design_features/basics.md``). It is the codec between the LLM and the REPL:
it *decodes* an LLM response into the code to run (``parse``), *encodes* each turn's
observation back to the model (``render_observation``), and opens the session
(``render_initial_message_list``).
"""

# Design/history (#566): landed as a small stack — 1a the object + decode seam (``parse``),
# 1b observation rendering, 1c the session opener — each a behavior-preserving lift of logic
# that previously lived inline in ``Agent``. The signatures are deliberately fat where today's
# coupling demands it (hook prompt additions, recursion/scoping context) and are transitional,
# not the target contract: they slim in later phases per ``basics.md``. The whole-``LLMResponse``
# input landed for ``parse`` in #1146 (the record seam ``build_history_entry`` already took it), so
# no method now takes pre-extracted content; the remaining #566 generalization (a ``meta`` axis,
# the ``CodeT``/``MetaT``/``OutputT`` generics, and dropping the per-message ``lang=`` tag) stays
# tracked under #566.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, NamedTuple

from .._catalog import render_catalog

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .._display import DisplayText
    from .._invoke_tool import InvokeTool
    from .._llm_client import LLMResponse
    from ..llm.base import MessageDict
    from ..repl.base import BaseREPL
    from ..repl.types import Continue, ExecResult


class ParsedCode(NamedTuple):
    """The success result of :meth:`BaseProtocol.parse` — the decoded input.

    A named type rather than a bare ``tuple``: the slots are self-documenting, call sites
    aren't coupled to position, and a new field can be added without silently breaking
    every custom ``parse``. This is the seam where ``basics.md``'s eventual
    ``ParsedCode[CodeT, MetaT]`` lands — the ``meta`` axis (tool-calling / ReAct payload)
    slots in here as a later field without another signature break.
    """

    code: str
    exec_timeout_override: float | None


class BaseProtocol(ABC):
    """Codec between the LLM and the REPL — encode the session, decode the response.

    Pluggable like ``llm_client``: a protocol is selected via
    ``Config.interaction_protocol`` (a registered name or a direct instance). See
    :class:`~CodeOnlyProtocol` for the built-in
    raw-code protocol.

    Defines the decoder (:meth:`~BaseProtocol.parse`), the encode primitives
    (:meth:`~BaseProtocol.render_observation`, :meth:`~BaseProtocol.render_initial_message_list`), and the
    ``__history__`` record projection (:meth:`~BaseProtocol.build_history_entry`).
    """

    @classmethod
    def from_dict(cls, params: Mapping[str, object] | None = None) -> BaseProtocol:
        """Build this protocol from its params — ``REGISTRY[tag].from_dict(params)``.

        The default maps params to constructor kwargs. Override for a protocol whose config
        shape does not match its ``__init__``.
        """
        return cls(**(params or {}))

    @abstractmethod
    def parse(self, llm_response: LLMResponse, repl_language: str) -> ParsedCode:
        """Decode an LLM response into a :class:`ParsedCode` ``(code, exec_timeout_override)``.

        Returns the extracted ``code`` and an optional per-exec timeout. The response carries no
        language attribute — an agent has a single REPL, so ``repl_language`` is passed for
        protocols that want it but the default does not validate against it.

        ``llm_response`` is the whole :class:`~jaz._llm_client.LLMResponse`, not a pre-extracted
        content string: a protocol reads ``.content`` itself (``str | None`` — ``None`` on a
        content-less completion) and can also reach ``.raw_response`` for ``refusal`` /
        ``finish_reason`` / ``tool_calls`` — what a tool-calling or refusal-aware protocol needs
        to decode a turn the content string alone cannot express.

        **Write this method so that it never raises.** There is no recoverable parse failure: the
        loop catches nothing here, so any exception is terminal and ends the invoke — an agent
        that emitted one bad message loses the whole run. Return your best reading instead,
        including the message verbatim when you have none, and let the REPL judge it; that is
        where malformed input surfaces anyway, as a ``SyntaxError`` inside a recoverable
        ``Continue`` the model can react to. An empty message is empty *code*, i.e. a no-op turn.
        """
        ...

    @abstractmethod
    def format_instructions(self) -> str:
        """The wire-format instruction block — the *request* half of the codec.

        Returns the model-facing text that tells the LLM how to shape its response so that
        :meth:`~BaseProtocol.parse` can decode it (for the default: "respond with ONLY code, no fences").
        The system-prompt template splices this in verbatim as an opaque
        ``{{ format_instructions }}`` block, so the accept half (:meth:`~BaseProtocol.parse`) and the
        request half (this) are owned by the *same* protocol and move together — a protocol
        cannot instruct one wire format while parsing another. The template still owns the
        surrounding wording and placement; only the format block itself belongs to the codec.
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
    # truncation advice moved onto the protocol, which renders from its own settings, not the
    # REPL — and a dead param that *looks* live invites the same wrong assumption. This also lets
    # Replay reproduce an observation by *calling* this seam (which owns whatever transform,
    # abbreviation included, the protocol applies) instead of re-deriving the truncation from a
    # CodeOnlyProtocol attribute the ABC never declared; the seam must render from
    # ``(exec_result, iteration)`` alone for that to be driveable without a live REPL.
    @abstractmethod
    def render_observation(
        self,
        exec_result: Continue,
        iteration: int,
    ) -> list[MessageDict]:
        """Render a (continuing) REPL execution result into the next turn's message(s).

        Returns the observation as one or more ready messages — the seam speaks the LLM's
        own message shape directly rather than a bare ``str`` body. The common case is a
        single user message carrying the text body, which is the ``CodeOnlyProtocol``
        single-REPL path; that same protocol returns **two** messages when the output was
        truncated, emitting the truncation advice separately so the observation stays the raw
        output — so the list return is exercised by the default implementer, not only by
        the hypothetical protocols below. The
        list is load-bearing for protocols whose observation is inherently *multiple*
        messages or a multi-block message: one ``tool`` message per parallel tool call
        (OpenAI requires one result per ``tool_call_id``), or a single message carrying
        several ``tool_result`` / image content blocks (Anthropic returns all parallel
        results together in one user message). A ``str`` return could express neither, so
        the final ``basics.md`` shape is landed now, while ``CodeOnlyProtocol`` is the only
        implementer, rather than re-breaking this ABC once such a protocol lands.

        The driver appends the returned messages verbatim — persisted history is pure protocol
        output, with no per-turn footer folded in (#634 removed the budget-status / "Enter your
        next REPL input" footer). Hook prompt additions, when present, are the *driver's* concern,
        not the protocol's (``basics.md``: "per-turn warnings are hook additions the driver
        concatenates").
        """
        ...

    @abstractmethod
    def render_initial_message_list(
        self,
        inputs: Mapping[str, object],
        scope: Mapping[str, object],
        repl: BaseREPL,
        *,
        invoke_tool: InvokeTool | None,
        depth: int,
        recursion_available: bool,
        repl_language: str,
        input_display_overrides: Mapping[str, DisplayText] | None,
    ) -> list[MessageDict]:
        """Encode the session opening into the initial message list (system + user).

        Rendered once per session: the REPL self-description, safety guidance, and the
        input blocks. There is no distinguished ``task`` argument — the task prompt is an
        ordinary entry in ``inputs`` (conventionally named ``task``), rendered as one
        ``<name type="...">`` block like any other input.

        The **wire-format instructions** (the "respond with ONLY code" block that is the
        *request* half of what :meth:`~BaseProtocol.parse` accepts) are owned by the protocol via
        :meth:`~BaseProtocol.format_instructions` and spliced into the system-prompt template as an opaque
        ``{{ format_instructions }}`` block. The template still owns the surrounding wording and
        layout, but the codec's accept half (``parse``) and request half live in the same
        protocol and cannot silently desync.

        Hook prompt additions do not flow through this seam — the driver owns them, the same
        symmetry :meth:`~BaseProtocol.render_observation` has. The arguments are the genuine
        rendering context the opener needs to render (not hook-effect outputs).

        Args:
            inputs: The invoke's inputs, each rendered as an input block.
            scope: Ambient scoped values (``jaz.scope``) in effect for this session.
            repl: The initialized REPL, for its self-description and language.
            invoke_tool: The bound sub-invoke tool, or ``None`` when recursion is disabled.
            depth: This invoke's recursion depth (0 at the top level).
            recursion_available: Whether sub-invokes are still permitted at this depth,
                for the delegation guidance.
            repl_language: The REPL's language tag for the rendered input/observation blocks.
            input_display_overrides: Per-input display text (relabel/hide), or ``None``.

        As with :meth:`~BaseProtocol.render_observation`, the driver folds any hook ``AddInputs`` /
        ``DropInputs`` into the REPL and prompt — not the protocol's job.
        """
        ...

    # History (#566 stack): the ``task``-as-ordinary-input move is #538; ``format_instructions``
    # ownership is #639; hook prompt additions left this seam when ``AddInstructionPrompt`` gave way
    # to query-time ``AddMessages`` (#660) and ``AddSystemPrompt`` was removed with its sole emitter
    # ``ParentUpdatesHook`` (#703), closing #640; the iteration-0 footer was removed by #634.

    @abstractmethod
    def build_history_entry(
        self, llm_response: LLMResponse | None, exec_result: ExecResult
    ) -> object:
        """Project one turn into its ``__history__`` entry — the **record** seam.

        The return type is ``object`` on purpose: the **entry shape is the protocol's**, not the
        core's — each protocol records whatever it likes, and the driver only holds a
        ``list[object]`` and surfaces it as ``__history__``. The default is
        :class:`~jaz.protocol.code_only.REPLHistoryEntry`; a custom protocol may return anything.

        Sibling to :meth:`~BaseProtocol.render_observation` — both encode a turn into an agent-facing surface —
        differing in destination (the in-band next-turn observation vs. the persistent
        ``__history__`` projection) and in input breadth: ``render_observation`` takes the
        recoverable ``Continue`` (a continuing turn always has one), while this records the broader
        ``ExecResult`` (a turn's finalized result may be terminal). Keeping both on the protocol puts
        "how a turn is represented to the agent" in one owner, so a protocol that customizes error
        rendering can't silently desync its history representation.

        Part of the required contract like :meth:`~BaseProtocol.parse` / :meth:`~BaseProtocol.render_observation` (the base
        stays fully abstract — no default projection lives here); :class:`~jaz.protocol.code_only.
        CodeOnlyProtocol` implements the standard entry. ``llm_response`` is the whole
        ``LLMResponse`` rather than the pre-extracted content string, so an implementation may
        fold in token counts / cost / provider ``extra`` without another signature break. The ``| None`` is an **external-only affordance** (a caller
        that has no response object): in the loop, ``_llm_response_to_record`` returns an
        ``LLMResponse`` by construction — its input is a ``str``, not ``str | None`` — so the
        framework never passes ``None`` here, and a future reader need not hunt for a loop path
        that does.

        The core agent loop is the single writer of ``__history__`` and appends one entry per
        iteration from the finalized (post-effect-composition) result via this method — see
        ``agent.do_one_repl_iteration`` / ``_record_history``.
        """
        ...

    def describe_history_entry(self) -> str:
        """The model-facing description of the ``__history__`` record — the *describe* half.

        Returns the prompt text telling the agent what ``__history__`` holds and what an entry's
        fields are, or ``""`` to say nothing. The system-prompt template splices it in verbatim,
        exactly like :meth:`~BaseProtocol.format_instructions`; the template owns the placement,
        the protocol owns the words. Not abstract: the default says nothing, so a protocol that
        does not implement it describes no record.
        """
        # Concrete-with-""-default rather than abstract, for two reasons: a protocol whose record
        # is not agent-readable (or that prefers to leave it undocumented) is a legitimate
        # implementation, and this is a new method on a shipped ABC — an abstract one would break
        # every out-of-tree protocol on upgrade for no gain.
        #
        # Why this exists at all: the entry SHAPE is the protocol's (see build_history_entry —
        # `object` return, "a custom protocol may return anything"), but until this seam the prose
        # naming `.llm_response` / `.repl_output` / `.repl_exception` lived in the REPL description
        # template, which knows nothing about the configured protocol. Swapping the protocol
        # therefore left the prompt describing another object's fields, and nothing could catch it.
        # This is the same accept/request pairing `format_instructions` documents for the wire
        # format ("a protocol cannot instruct one wire format while parsing another") — the record
        # seam now has both halves on one owner too.
        #
        # What stays OUTSIDE this seam: that the name `__history__` is bound at all is the REPL's
        # (``BaseREPL.maintains_repl_history``), and the list's one-entry-per-iteration ordering is
        # the core loop's (the single writer). CodeOnlyProtocol's block states all three because a
        # coherent prompt paragraph has to, and the caller gates it on the REPL capability — but
        # this method is only *authoritative* for the entry shape.
        #
        # TODO(#1171): the seam covers the SYSTEM PROMPT only. `repl_output_truncation_advice*
        # .jinja2` still names `__history__[-1].repl_output` — a `REPLHistoryEntry` field — from a
        # core template, gated on neither capability, so the observation path can still describe a
        # record the configured protocol does not produce.
        return ""

    def get_subinvoke_description(self, invoke_tool: InvokeTool) -> str:
        """The agent-facing description of the recursive sub-invoke tool (bound as ``invoke``).

        Rendered into the system prompt as the ``invoke`` catalog entry. The default introspects
        the bound closure — its signature and docstring ARE the description, so the two cannot
        drift. Not abstract, for the same reasons as :meth:`describe_history_entry`: the
        introspection default fits every protocol, and adding an abstract method to a shipped ABC
        would break out-of-tree protocols on upgrade.

        A protocol overrides this to author the description explicitly — typically to advertise a
        NARROWER surface than the closure actually supports. The motivating case is a
        ``restrict_invoke_args``-style guard that vetoes non-literal / positional-hook ``invoke``
        calls at runtime: the prompt should then describe an inputs-only signature even though the
        closure still accepts hooks. Owning the words here — rather than swapping the executable
        closure — keeps "what the tool can do" separable from "what the agent is told it can do";
        an overriding author owns keeping the two in step with the guard that motivates the
        override.
        """
        return render_catalog(invoke_tool, "invoke", full_docstrings=True)
