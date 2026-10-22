"""The interaction protocol: the assistant message **is** the code.

Registered as ``"code_only"`` and the only protocol JAZ ships. There is no XML framing, no
``<repl_code>`` tag, and no delimiter of any kind — the whole assistant message is executed
verbatim as REPL code, and reasoning goes in leading comments.

Why there is no delimiter
-------------------------
Every delimiter-based framing (``<repl_input lang=...>``, ``<code>``, ``<repl_code>``, markdown
fences, CDATA, a nonce fence) shares one defect: the wire format gives the model no way to say
*"this occurrence of the delimiter is data, not structure."* When the code legitimately contains
the delimiter, the two readings are indistinguishable from the bytes alone and the parser is
guessing. Under the former XML protocol::

    <repl_code>
    html = "<repl_code>hi</repl_code>"
    print(html)
    </repl_code>

the non-greedy body stopped at the *inner* closing tag, so the REPL received
``html = "<repl_code>hi`` — an unterminated string literal — and, because the truncated body was a
lone match, the multi-block check did not fire. The failure was silent.

Deleting the delimiter removes that class of failure by construction: nothing in the byte stream
is structural, so there is nothing for content to imitate. Code that *contains* fence or tag
characters — an agent authoring a prompt, which JAZ's own meta-harnesses do — is taken verbatim.

**This fixes delimiter collision specifically, NOT over-escaping in general.** A controlled repro
(150-line script, 10 trials/arm) showed this protocol producing top-level ``f\"...\"`` — the same
escape-your-own-output failure the XML protocol showed on SWE-bench — numerically more often than
XML did there (8.9% vs 4.7% of turns; Fisher p=0.51, not significant, but plainly not zero).
Over-escaping is provoked by the task shape (writing code that contains code); removing the
delimiter narrows which shapes provoke it rather than eliminating it. See the comment above
:class:`CodeOnlyProtocol` for the measurement with intervals and ``logs/protocol_repro/`` for
traces.

Known costs (accepted)
----------------------
- **A chatty preamble is a hard failure.** ``Sure! Let me compute that.`` followed by bare code is
  not valid Python. The XML protocol tolerated surrounding prose by construction (it *searched*
  for a block); this cannot. The failure is loud and recoverable rather than silent, which is the
  intended trade, but it falls hardest on smaller models.
- **No per-exec timeout attribute.** ``<repl_code timeout="30">``'s successor is not a protocol
  concern at all: the REPL reads a ``# timeout: <seconds>`` comment among the lines that open
  the code (see :meth:`PythonREPL.exec`), so the request travels *inside* the code rather than
  in framing this protocol no longer has.
- **No plan/code split.** Consumers that extracted ``<plan>`` separately see only code — though
  the reasoning survives, since comments travel with the code into ``llm_response``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jaz._component import repr_call

from .._string_utils import abbreviate_string
from ..exceptions import _JazInternalError
from ..repl._types import Continue
from ._base import BaseProtocol
from ._prompts import (
    get_system_prompt,
    get_user_prompt,
    render_input_section,
    require_description,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from ..llm._types import LLMResponse
    from ..repl._base import BaseREPL


# The response-format description — the *request* half of this codec, owned here beside `parse`
# (its accept half) so the two cannot desync (#639): the text must stay in step with what
# `parse` accepts (raw code, no fences, prose as comments). The XML framing lives in the
# template (`<response_format>` around `{{ response_format_description }}`), not here, so the
# protocol supplies only the text.
_RESPONSE_FORMAT_DESCRIPTION = """\
- Respond with ONLY code. Your entire response will be sent verbatim to the REPL and executed as code.
  Anything in your response that is not valid code is a syntax error.
  - All natural language prose MUST be written as *comments* in your code.
  - Do NOT wrap your code in markdown fences or XML tags.
- Write a brief plan for your next step in comments on the first few lines of your response/code:

    # <brief plan for next step>
    code_for_next_step
"""


# EXECUTIVE CALL (user, 2026-08-03): raw-code framing is the ONLY protocol — the XML
# ``<repl_code>`` codec was deleted rather than kept as an option, and its prompt templates were
# rewritten in place. Decided on a measured A/B (gpt-5-mini; SWE-bench, StuLife, Recall). The
# evidence is narrower than the change looks, so it is recorded with intervals rather than as a
# slogan. (This writeup lived on `ProtocolConfig`, which no longer exists; it belongs with the
# component whose settings it justifies.)
#
# **What the A/B established.** On SWE-bench, the rate of turns producing no runnable code. The
# unit of independence is the *instance*, not the turn — turns within one instance share a task,
# a repo and a conversation, so pooling them inflates n. With n=10 instances per arm:
#
#     XML        8.0% per-instance  [95% CI -0.6%, 16.6%]
#     pure_code  0.8% per-instance  [95% CI -0.1%,  1.8%]
#     permutation test over instances, 20k resamples: p = 0.016
#
# The direction replicated across three sweeps (turn-level 7.9/1.1, 16.1/2.7, 6.2/1.4). But the
# effect is **heavy-tailed, not uniform**: per-instance XML rates were 0, 9, 46, 0, 8, 0, 4, 8, 2,
# 3 percent — drop the 46% instance and the mean falls to ~3.8%. The defensible claim is "XML
# occasionally fails badly on an instance", NOT "XML is uniformly noisier per turn". A naive
# turn-level Fisher test reports p<0.001; that is wrong, since it treats 1312 clustered turns as
# independent.
#
# **Zero parse failures in either arm across ~2000 turns.** The XML protocol's *nominal* failure
# modes (missing block, duplicate block) never fired once. Every failure was downstream, at
# execution — which is why this was invisible until measured.
#
# **Task scores did not favor it, and are not significant either way.** SWE-bench resolved 20.0%
# [9.5%, 37.3%] for BOTH arms (Fisher p=1.0). StuLife 93.5% vs 92.3% and Recall 100% vs 100% are
# single episodes: n=1, no interval, not estimates. This default rests on robustness, NOT on
# performance.
#
# **A controlled repro did NOT reproduce the gap.** Holding the task fixed and asking for a large
# script (10 trials/arm), neither cell separates:
#
#     50-line script   XML 2.1% [0.6%,  7.4%]   pure 2.7% [0.7%,  9.2%]   Fisher p = 1.00
#     150-line script  XML 4.7% [1.6%, 12.9%]   pure 8.9% [4.4%, 17.2%]   Fisher p = 0.51
#
# So the advantage is specific to SWE-bench-shaped work, or needs more power to see elsewhere. It
# is not a demonstrated general property of the framing.
#
# **The mechanism, and a correction.** In SWE-bench traces the XML failures were the model
# escaping its own output — a whole program emitted as one line of literal ``\n`` sequences
# ("unexpected character after line continuation character" dominated 106:23). Two further
# XML-only corruptions were reproduced but were not the dominant cause: a ``</repl_code>`` inside
# a string literal truncates the body, and ``html.unescape`` turns a legitimate ``&quot;`` into a
# bare quote.
#
# Do NOT conclude that raw-code framing removes over-escaping. The 150-line repro shows
# ``pure_code`` inducing the same class of failure — top-level ``f"..."``, backslash before quote
# — numerically more often than XML there (not significantly). Over-escaping is provoked by the
# TASK (writing code that contains code), not only by the wire format; the protocol shifts which
# shapes provoke it, not whether it can happen.
class CodeOnlyProtocol(BaseProtocol):
    """The protocol: the assistant message is the code (see the module docstring).

    Its rendering settings are constructor parameters — the constructor is the only declaration
    of them, and of their defaults:

    - ``*_template``: which Jinja template renders the system prompt and the user prompt.
    - ``max_invoke_input_length`` / ``max_repl_output_length``: truncation ceilings for rendered inputs and
      for REPL output. A truncated value keeps an even split of head and tail. The REPL appends
      a recoverable error's traceback at the end of the turn's output, so the retained tail
      carries the exception line; what a long turn can lose is the top of that traceback, and
      any error message the agent's own code printed early (see ``render_observation``).
    - ``subinvoke_description``: an authored description of the recursive ``invoke`` tool in
      place of the one derived from its closure (see :meth:`get_subinvoke_description`).
    """

    # These were previously read *ambiently*, off `config.protocol.*` at each use site, while
    # the protocol builder took no arguments at all — so the protocol was "a tag beside a flat bag
    # that something else consumes" rather than a `{tag, params}` component like the LLM family.
    #
    # EXECUTIVE CALL (user, 2026-08-07): make them constructor params (option (a) of the design
    # doc's open question 4), over leaving them ambient (b) or splitting the difference (c). The
    # doc left this undecided because ctor params "would be fixed at construction, so a scoped
    # ConfigOverride would no longer take effect on an already-built protocol."
    #
    # That concern does not arise, which is what makes this behaviour-preserving. Every call site
    # already received `self.config` — the Agent's *construction-time* config — and this class
    # never called `get_config()`. The protocol was already built once per Agent, from exactly the
    # config these params are now read from, so binding at construction changes nothing about when
    # a value is observed.
    #
    # The one genuine behaviour change: a caller who passes a pre-built protocol *instance* as
    # `protocol.name` now gets the params that instance was built with, rather than having
    # `config.protocol.*` silently override them. That matches how a `BaseLLM` instance tag already
    # behaves ("you built it, you own it") and is the more predictable of the two.
    #
    # `max_repl_output_length` moved here from `REPLConfig` in the same pass: truncation is a
    # *rendering* decision and rendering is what the codec owns. It was never read by the REPL —
    # only by this class and by `TrajectoryReplay`, which reproduces this class's abbreviation.
    # These defaults are literals here. They used to be spelled `= ProtocolConfig.<field>` (a
    # config class that no longer exists), which meant this module imported the config layer at
    # runtime purely to learn what it itself defaults to — the dependency ran backwards. It also
    # made a *registered* protocol's own defaults unreachable: `Config.protocol_params` passed
    # every leaf explicitly, including ones nobody set, so `ProtocolConfig`'s value always won and
    # only `CodeOnlyProtocol` could express a default at all. That was documented as a known
    # limitation on `protocol_params`; owning the values here removes it.
    def __init__(
        self,
        *,
        system_prompt_template: str = "jaz/system_prompt.jinja2",
        user_prompt_template: str = "jaz/user_prompt.jinja2",
        max_invoke_input_length: int = 50000,
        max_repl_output_length: int = 50000,
        # An explicit agent-facing description of the recursive ``invoke`` tool, overriding the
        # default (which introspects the bound closure's signature/docstring — see
        # ``BaseProtocol.get_subinvoke_description``). Set it to advertise a NARROWER surface than
        # the closure supports — e.g. an inputs-only signature under a ``restrict_invoke_args``
        # guard that vetoes positional-hook ``invoke`` calls at runtime. ``None`` derives the
        # description from the closure; an override is the author's to keep in step with that guard.
        # Kept although nothing sets it today (executive call, user, 2026-09-18): the codec's
        # settings are its complete rendering surface, and authoring this text is a rendering
        # setting like the templates beside it, not something a host should have to subclass for.
        subinvoke_description: str | None = None,
    ) -> None:
        # Range checks live with the settings they constrain, for the same reason the values
        # do: `Config` used to make these on the protocol's behalf, so a protocol constructed
        # directly was unchecked and a *registered* protocol got the default's rules applied to
        # settings it may not even have.
        for name, value in (
            ("max_invoke_input_length", max_invoke_input_length),
            ("max_repl_output_length", max_repl_output_length),
        ):
            if value <= 0:
                raise ValueError(f"protocol.{name} must be positive")
        for name, value in (
            ("system_prompt_template", system_prompt_template),
            ("user_prompt_template", user_prompt_template),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        # Same predicate as `require_description`, written out because the message differs: a
        # setting is *set*, not returned, and it sits beside the two template-name checks above
        # whose wording it shares. The seam's own check (in `render_initial_message_list`) is
        # what names the class and method when an override returns the bad value.
        if subinvoke_description is not None and (
            not isinstance(subinvoke_description, str) or not subinvoke_description
        ):
            raise ValueError("subinvoke_description must be a non-empty string or None")
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template
        self.max_invoke_input_length = max_invoke_input_length
        self.max_repl_output_length = max_repl_output_length
        self.subinvoke_description = subinvoke_description

    def __repr__(self) -> str:
        return repr_call(
            "CodeOnlyProtocol",
            ("system_prompt_template", self.system_prompt_template),
            ("user_prompt_template", self.user_prompt_template),
            ("max_invoke_input_length", self.max_invoke_input_length),
            ("max_repl_output_length", self.max_repl_output_length),
            ("subinvoke_description", self.subinvoke_description),
        )

    def parse(self, llm_response: LLMResponse) -> str:
        """Take the response as code, verbatim.

        The message *is* the code — what the format instructions ask for and what the REPL
        receives. Nothing is unwrapped, extracted or repaired.

        ``parse`` never fails. A truncated response, one with a prose preamble, or one carrying no
        content at all reaches the REPL and surfaces there — a ``SyntaxError`` inside a
        recoverable ``Continue`` for the first two, and a no-op turn for an empty one. Failing
        here instead would move the failure from exec-time to parse-time and change which history
        fields record it, for no gain to the agent.

        An empty or whitespace-only message is therefore empty *code*, not an error. The agent is
        not told its turn produced nothing; it sees a blank observation and continues, and the turn
        still costs an iteration. That is the accepted trade for a protocol with no failure mode;
        a content-less completion is rare in practice.
        """
        # This used to attempt a markdown-fence rescue: when the message did not parse as Python,
        # a fenced block whose contents did parse was unwrapped instead. Measured over 335,652
        # assistant turns in the eval corpus, that rescue would have fired ONCE — models told not
        # to fence essentially never do. Of the 2,115 responses there that did not parse, 2,114
        # were ordinary broken code no fence handling could repair.
        #
        # It was not free. The oracle it needed was a hardcoded Python parser, so `parse` had to
        # take the REPL's language to decide whether it could ask — which made a language string a
        # protocol input, and denied the rescue to `docker_python` (which runs Python) purely by
        # how its name is spelled.
        #
        # CodeOnlyProtocol treats a content-less completion as empty code, so it flattens
        # ``content is None`` to ``""`` here — the coercion the loop used to do before parse
        # (#1141). Now that parse sees the whole response (#1146), a protocol that wants to tell a
        # refusal from empty code reads ``llm_response.content`` / ``.raw_response`` instead.
        return (llm_response.content or "").strip("\n\r")

    @property
    def response_format_description(self) -> str:
        """Return the response-format description (see :meth:`BaseProtocol.response_format_description`).

        A constant: this protocol's request format does not vary with its rendering settings. It
        is the request half of the codec whose accept half is :meth:`~CodeOnlyProtocol.parse` — the two are edited
        together here so an instruction to "respond with ONLY code" can never drift from what
        ``parse`` actually accepts.
        """
        return _RESPONSE_FORMAT_DESCRIPTION

    def render_observation(self, exec_result: Continue) -> list[dict[str, Any]]:
        """Render a continuing REPL result into the next turn's user message(s).

        Returns a single user message whose content is the (abbreviated) ``output`` and **nothing
        else**: no ``<repl_output>`` wrapper, no separate ``<error>`` block, no ``**REPL output
        (iteration #N):**`` header, no error-flag line, and no truncation advice.

        The framing is gone entirely rather than merely simplified because ``output`` already holds
        the whole agent-facing text (stdout on success; stdout + traceback, or the rendered
        parse error, on a recoverable error). Every wrapper around it was restating in markup what
        the text says on its own — including the success/error split, which the traceback in
        ``output`` already announces, so ``exception`` stays pure metadata with no rendered surface
        at all. The iteration number went with the header: the turn's position is implicit in the
        conversation, and ``__history__`` is the addressable form for an agent that needs it.
        """
        output = exec_result.output
        if not isinstance(output, str):
            raise _JazInternalError("REPL output should be a string")
        # The truncation flag is dropped: nothing reports truncation any more (see below), and the
        # cut announces itself inline via `[...N characters omitted...]`.
        abbreviated_output, _ = abbreviate_string(
            output, max_length=self.max_repl_output_length
        )

        # The observation IS the output — the message carries no framing of its own (#928,
        # executive call). Two consequences worth knowing before adding any back:
        #
        # 1. No error branch. Dropping the ``*An error occurred:*`` flag (and, before it, the
        #    ``<error>`` block) leaves the error text to reach the agent as ordinary output, which
        #    means truncation can take it. The REPL's own error text is always the tail: a
        #    recoverable exception ends the turn and `_format_error` returns `stdout + traceback`,
        #    so the retained suffix carries the exception line and the frames nearest it; what
        #    is cut is whatever lies between head and tail — the traceback's top on a long turn,
        #    its middle frames when stdout is short. The text that CAN
        #    sit early is the agent's: code that catches an error, prints it, and keeps going
        #    puts that message near the start, where only a retained prefix sees it. No head/tail
        #    split covers both — measured on a 3129-char turn whose printed error lands 2% in,
        #    only an all-head split kept it, and an all-head split would drop every traceback —
        #    so the even split is the one that never loses the REPL's own report. Reinstating a
        #    flag, or making the error text exempt from truncation, is the fix if this bites.
        # 2. No parsing anchor. With both the tags and the header gone, an observation message is
        #    not self-identifying: consumers that read outputs back out of a transcript
        #    (``TrajectoryReplay`` divergence detection, ``trace_to_directory``) can only locate it
        #    *positionally* — it is the first user message following an assistant message. Any
        #    future change that emits a second user message before the observation, or that reuses
        #    that slot for something else, silently breaks both.
        # The observation is the ONLY message this turn produces. There used to be a second one:
        # `repl_output_truncation_advice_template` named a template core rendered and appended
        # whenever the output was truncated, telling the agent that `__history__[-1].repl_output`
        # holds the whole string.
        #
        # Nothing replaces it, and the pointer it carried is not restated anywhere: the obvious
        # home would be `PythonREPL.repl_history_description`, but that is agent-facing text under the
        # DEFAULT config and changing it would invalidate published eval results (see the comment
        # there). A host that wants the agent told about `.repl_output` says so in its own
        # `system_prompt_template`. The default prompts are byte-identical to before this PR,
        # because the knob defaulted to `None` — no default run ever saw this message.
        #
        # The knob repeated `input_truncation_advice_template`'s failure mode: offered as a name,
        # it was set by 87 eval configs that wanted the default, and its `_subagents` variant
        # smuggled a whole decompose-and-delegate strategy into what read like a truncation notice.
        # Both variants are recoverable from git history if an arm wants to own that prose.
        #
        return [{"role": "user", "content": abbreviated_output}]

    def get_subinvoke_description(self, subinvoke: Callable[..., object]) -> str:
        # Author-supplied override wins; otherwise fall back to the base's closure introspection.
        # An override advertises a narrower surface than the closure supports (see the
        # `subinvoke_description` constructor param) without swapping the executable tool.
        if self.subinvoke_description is not None:
            return self.subinvoke_description
        return super().get_subinvoke_description(subinvoke)

    def render_initial_message_list(
        self,
        inputs: Mapping[str, object],
        scope: Mapping[str, object],
        repl: BaseREPL[Any],
        *,
        subinvoke: Callable[..., object] | None,
        repl_history_withheld: bool,
    ) -> list[dict[str, Any]]:
        """Build the [system, user] opener.

        ``inputs`` (explicit ``**inputs`` kwargs) and ``scope`` (resolved ``jaz.scope``)
        arrive as SEPARATE mappings, so the two sections render directly from their own
        dict — no merged namespace, no ``scoped_keys`` re-split. There is no separate
        ``task`` argument anymore: the prompt is an ordinary input."""

        # Separate inputs/scope mappings is #727; ``task`` as an ordinary input is #538.
        # Scoped inputs render in the SYSTEM prompt (ambient/global, like tool
        # libraries); render the whole `scope` mapping here and hand it to
        # get_system_prompt — the same blocks-then-sentence shape the user prompt uses for
        # explicit inputs.
        scoped_inputs = render_input_section(
            scope, max_invoke_input_length=self.max_invoke_input_length
        )
        # The protocol owns how the recursive `invoke` function is described to the agent
        # (default: introspect the closure; override via `subinvoke_description`). Rendered here
        # and handed to get_system_prompt as a STRING, so that helper stays codec-agnostic.
        # `None` when there is no subinvoke.
        subinvoke_description: str | None = None
        if subinvoke is not None:
            # The same check the constructor applies to the `subinvoke_description` setting,
            # applied to an override's return value — including `None`, since the tool IS bound
            # here and "described as nothing" is a bug in the override, not an absence. Refusing
            # it is what lets both templates gate on the description being PRESENT and have that
            # mean exactly "the agent is shown an `invoke`".
            subinvoke_description = require_description(
                self.get_subinvoke_description(subinvoke),
                seam=f"{type(self).__name__}.get_subinvoke_description",
            )
        # The `__history__` description, gated the way the `invoke` entry is: on the fact that
        # decides whether the agent can reach the thing. The REPL says whether it binds the name
        # at all (`repl_history_description`, `None` for a REPL whose namespace is elsewhere), and
        # the driver says whether a `WithholdREPLHistory` withheld it from this invoke; the
        # section renders only when neither absence holds, so the prompt never describes a name
        # the agent cannot reference. Checked under the rule every description seam gets.
        repl_history_description = (
            None if repl_history_withheld else repl.repl_history_description
        )
        if repl_history_description is not None:
            require_description(
                repl_history_description,
                seam=f"{type(repl).__name__}.repl_history_description",
            )
        user_prompt = get_user_prompt(
            user_prompt_template=self.user_prompt_template,
            inputs=inputs,
            max_invoke_input_length=self.max_invoke_input_length,
            # #568: the closing inputs sentence's wording is owned by the (config-selectable)
            # user_prompt_template, not by a REPL capability — a withheld-inputs variant flips it.
            # The two sections state opposite halves of the propagation rule (scoped values ride
            # along, inputs do not), so they must appear or vanish together. Both templates are
            # handed this ONE value and gate on it being present; the check above is what makes
            # "present" and "the agent is shown an `invoke`" the same thing. Presence rather
            # than truthiness (executive call, user, 2026-09-18): the question is whether a
            # subinvoke is bound, not whether a string is non-empty, and an empty string is
            # refused rather than read as "absent".
            subinvoke_description=subinvoke_description,
        )
        system_prompt = get_system_prompt(
            system_prompt_template=self.system_prompt_template,
            subinvoke_description=subinvoke_description,
            repl=repl,
            scoped_inputs=scoped_inputs,
            # The protocol owns the wire-format block and hands it to the template (#639);
            # the template only decides where it lands. No `None` arm, unlike the two optional
            # descriptions: a codec always has a request format.
            response_format_description=require_description(
                self.response_format_description,
                seam=f"{type(self).__name__}.response_format_description",
            ),
            # The `__history__` block is gated in the template on presence, like the `invoke`
            # block, and the protocol has no switch of its own for it: whether the section renders
            # is the REPL's description and the invoke's withholding, resolved above.
            repl_history_description=repl_history_description,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
