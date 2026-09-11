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
  in framing this protocol no longer has. ``parse`` therefore still returns
  ``exec_timeout_override=None`` — the seam stays open for a protocol that does carry one
  out-of-band, and the pragma outranks it when both are present.
- **No plan/code split.** Consumers that extracted ``<plan>`` separately see only code — though
  the reasoning survives, since comments travel with the code into ``llm_response``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..exceptions import _JazInternalError
from ..repl.types import Continue
from ..string_utils import abbreviate_string
from .base import BaseProtocol, ParsedCode
from .prompts import get_system_prompt, get_user_prompt, render_input_section

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .._display import DisplayText
    from .._invoke_tool import InvokeTool
    from .._llm_client import LLMResponse
    from ..llm.base import MessageDict
    from ..repl.base import BaseREPL
    from ..repl.types import ExecResult


@dataclass(frozen=True)
class REPLHistoryEntry:
    """One ``__history__`` entry — **this** protocol's history-record shape.

    The entry type belongs to the protocol, not to the core or the REPL: it is exactly what
    :meth:`CodeOnlyProtocol.build_history_entry` projects a turn into, and a custom protocol may
    record an entirely different shape. (Core stays schema-agnostic — the driver holds a
    ``list[object]`` and appends whatever ``build_history_entry`` returns; the REPL only *surfaces*
    that list as ``__history__`` by reference. Per ``design/design_features/basics.md``'s
    Harness-as-Language / FAC view, a REPL session's own history is a first-class, agent-visible
    projection of the loop history into the namespace, but the projection is this seam's call.)

    ``__history__`` is a plain list of one entry per iteration in order — ``[0]`` first, ``[-1]``
    most recent — with no index-0 init snapshot, so ``__history__[0]`` is the first iteration.
    """

    # No index-0 init snapshot: the old ``REPLHistoryInit(inputs)`` (removed in #568) duplicated
    # ``__inputs__`` and would have leaked withheld inputs back through history; callers that used
    # ``__history__[1:]`` now use ``__history__`` directly.

    # The full LLM response that produced this turn — the raw model output, NOT the parsed code.
    # This is the executive call (was ``repl_input``): the real consumers (sweb/appworld
    # meta-harnesses) print each step for an LLM to read, and the highest-value signal — the
    # model's stated reasoning — was being discarded by storing only the parsed code.
    #
    # Under raw-code framing the two are the same string up to a trailing-newline strip: the
    # response *is* the code, and nothing is unwrapped or recovered. It mattered more under the
    # old XML protocol, where the parsed code dropped the whole ``<plan>`` block.
    # Cost tradeoff (accepted): a chatty agent's comments can be many KB/turn, so a meta-agent
    # printing a long history verbatim pays more than for the old code-only field — the intended
    # price for preserving the reasoning, bounded the same way any large history is (the consuming
    # harness slices/summarizes). Note ``repl_output`` itself is NOT bounded:
    # ``max_repl_output_length`` abbreviates the observation MESSAGE, never the record. The agent
    # is not told this — see the comment above ``_REPL_HISTORY_DESCRIPTION``.
    llm_response: str
    # Always a ``str``, never ``None``: a *terminal* turn (``Return``/``Raise``, which carries no
    # ``output`` at all) records ``""``.
    #
    # EXECUTIVE CALL (user): this reverses #903, which used ``None`` for the terminal turn so a
    # meta-agent could tell "the finishing block printed nothing" from "output is not a concept for
    # this turn". That distinction lost to simplicity, on two grounds. (1) The field is
    # *agent-facing* — every consumer is model-written code inside the REPL, where a one-type field
    # needs no ``or ""`` guard and cannot ``TypeError`` on a string operation. (2) The shipped
    # system prompt already described it as ``str`` (see ``_REPL_HISTORY_DESCRIPTION``), so ``None``
    # was a documented-away edge case rather than a surface anyone was told to handle.
    #
    # KNOWN LIMITATION (the accepted cost). A terminal entry *is* readable, so this is a trade, not
    # a free win: ``_record_history`` appends to the same list the REPL binds as ``__history__``, so
    # a subagent finishing with ``return __history__`` hands its parent a list the terminal entry
    # then lands in by reference — which is what the ultrahorizon/sweb meta-prompts iterate. To that
    # reader ``""`` now asserts "this turn printed nothing", and a finishing block that printed
    # before returning makes that false (its stdout is captured and dropped — see
    # ``test_repl_history_appended_on_return_with_preceding_code``). Such a consumer must key off
    # POSITION, the last entry, rather than the value. An agent reading its *own* history is
    # unaffected: its terminal entry is appended after the turn ends, so it is never there while
    # that agent is reading. Recording the finishing block's real stdout would dissolve both the
    # ambiguity and the loss, but "a terminal turn carries a value/exception, not agent-facing
    # text" is #903's deliberate invariant; reversing it is a separate change.
    repl_output: str
    # The exception from a recoverable ``Continue`` (``None`` for a clean success, a terminal
    # ``Raise``, or a ``Return``). Stored as the **raw exception object**, not a summarized string
    # (was ``error``). Rationale: the agent-facing *text* of the error is already fully contained in
    # ``repl_output`` — an error ``Continue``'s ``output`` is ``stdout + traceback`` (see
    # ``PythonREPL._format_error``), and ``traceback.format_exception`` always ends with the
    # ``Type: message`` line that ``summarize_exception`` would have produced. So the summary string
    # was redundant with ``repl_output``; keeping the object instead gives consumers structured
    # access (type, args, the ``__cause__``/``__context__`` chain), and any consumer that still
    # wants the old text calls ``summarize_exception(repl_exception)`` itself — matching the #590
    # decision that ``Continue`` reports the raw exception and each consumer owns its serialization.
    # (One non-textual nuance: for a ``BaseExceptionGroup`` the traceback renders children with
    # ``+---``/``| `` markup rather than ``summarize_exception``'s indented form, but every child's
    # type+message is still present in ``repl_output``.)
    #
    # The invariant holds *uniformly*, including a turn whose code never ran: the REPL's own
    # did-not-execute paths populate ``output`` with the rendered error (``summarize_exception(e)``)
    # rather than leaving it empty, precisely so ``repl_output`` is a valid pointer for every entry
    # (see the ``Continue`` contract). ``parse`` no longer contributes such a turn — it has no
    # failure mode, and an empty message is empty code. A meta-agent that
    # instead wants to re-derive the text from the raw ``repl_exception`` — e.g. from *inside* its own
    # sandboxed REPL, where ``from jaz.string_utils import summarize_exception`` is usually denied by
    # the default ``allowed_imports`` — can reconstruct the one-liner import-free as
    # ``f"{type(e).__name__}: {e}"``.
    repl_exception: BaseException | None


# The wire-format instruction block — the *request* half of this codec, owned here beside `parse`
# (its accept half) so the two cannot desync (#639). This was previously hardcoded in
# `system_prompt.jinja2`; the shipped template now splices it in as `{{ format_instructions }}`.
# The text must stay in step with what `parse` accepts: raw code, no fences, prose as comments.
# Kept byte-identical to the block it replaced so the rendered system prompt is unchanged.
# The wire-format instructions the protocol owns. The XML framing lives in the template
# (``<response_format>`` around ``{{ format_instructions }}``), not here, so the protocol
# supplies only the instruction text (#639).
_FORMAT_INSTRUCTIONS = """\
- Respond with ONLY code. Your entire response will be sent verbatim to the REPL and executed as code.
  Anything in your response that is not valid code is a syntax error.
  - All natural language prose MUST be written as *comments* in your code.
  - Do NOT wrap your code in markdown fences or XML tags.
- Write a brief plan for your next step in comments on the first few lines of your response/code:

    # <brief plan for next step>
    code_for_next_step
"""


# The ``__history__`` description — the *describe* half of this codec's record seam, owned here
# beside `build_history_entry` (its write half) for the same reason `_FORMAT_INSTRUCTIONS` sits
# beside `parse`: the field names below are exactly `REPLHistoryEntry`'s, so the two must be edited
# together. It previously lived in `python_repl_description.jinja2`, where a swapped protocol left
# the REPL describing a record it does not produce.
#
# The text states only the record's SHAPE. Its framing — a lead-in sentence and the enclosing
# `<__history__ type="list">` element — belongs to `system_prompt.jinja2`, the same division
# `{{ format_instructions }}` already had: the template owns placement and framing, the protocol
# owns the words that have to track `REPLHistoryEntry`. Hence no opening announcement of the
# variable here, and a flat field list rather than one nested under such a bullet.
#
# EXECUTIVE CALL (user): the block renders in its own element AFTER `</repl_spec>`, not inside it.
# It previously sat at the tail of `<repl_spec>`, on the reasoning that it documents a REPL
# *namespace* variable and so belongs with the REPL's spec. That was reversed: as its own element
# the block is a peer of `<invoke type="function">` — the other named thing the agent is handed —
# and `<repl_spec>` is left as purely the REPL's own description. No behavioural evidence either
# way was collected. Nor does any measurement in this repo bear on it: the `return`-framing result
# in python_repl_description.jinja2 compares two WORDINGS at a fixed position, not two positions,
# so do not cite it as a prompt-placement effect (an earlier draft of this comment did).
#
# `repl_output` is described as `str` and genuinely is one — the terminal turn records `""` rather
# than `None`, precisely so this agent-facing description needs no two-state caveat.
#
# NOT EXTENDED to say `repl_output` holds the untruncated text, though it does: the observation
# message is abbreviated while `build_history_entry` records `exec_result.output` whole, and the
# deleted `repl_output_truncation_advice*.jinja2` was the only thing that ever told the agent so.
# Stating it here would have been the natural home — this seam owns the entry's shape. It is not
# added because THIS STRING IS AGENT-FACING TEXT UNDER THE DEFAULT CONFIG, and published eval
# results have to reproduce under the exact prompts that produced them (executive call, the user).
# Changing the default system prompt is out of bounds regardless of how much better the wording
# would be; a host that wants the pointer states it in its own `system_prompt_template`.
#
# Two further reasons the sentence would have needed caveats anyway: a terminal turn records `""`
# (see the field comment on `REPLHistoryEntry`), and a REPL that truncates host-side before the
# protocol sees the result — `evals/sweb/docker_repl.py::_truncate` — makes the entry lossy too.
_REPL_HISTORY_DESCRIPTION = """\
`__history__` is a list with one entry per REPL iteration, in order (`__history__[0]` is your first
iteration and `__history__[-1]` the most recent). Each entry has:
- `.llm_response (str)`: Your full response for that iteration containing your code
- `.repl_output (str)`: The printed output from that iteration, including any error traceback
- `.repl_exception (BaseException | None)`: The exception object raised if that iteration hit
  a recoverable error, else None
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
      for REPL output.
    - ``truncation_prefix_ratio``: how a truncated value is split between its retained head and
      tail, from 0.0 (all tail) to 1.0 (all head). A truncated turn's error text can be lost at
      any setting — it sits wherever the failure happened in the output, so a late one survives
      only in the tail and an early one only in the head (see ``render_observation``).
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
    # These defaults are literals here, and `ProtocolConfig`'s leaves are unset. They used to be
    # spelled `= ProtocolConfig.<field>`, which meant this module imported the config layer at
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
        # 0.5 (an even head/tail split), not the 0.8 this defaulted to historically: 64 of the 65
        # eval configs that set it override to exactly 0.5 and the 65th to 0.0, so nothing in the
        # repo wanted 0.8. Weighting the head assumes the informative part of an over-long value
        # is at the front — true for prose and stack traces, false for REPL output, where the
        # result usually lands at the end.
        truncation_prefix_ratio: float = 0.5,
        # An explicit agent-facing description of the recursive ``invoke`` tool, overriding the
        # default (which introspects the bound closure's signature/docstring — see
        # ``BaseProtocol.get_subinvoke_description``). Set it to advertise a NARROWER surface than
        # the closure supports — e.g. an inputs-only signature under a ``restrict_invoke_args``
        # guard that vetoes positional-hook ``invoke`` calls at runtime. ``None`` derives the
        # description from the closure; an override is the author's to keep in step with that guard.
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
        # Inclusive at both ends: no ratio is safe for error text, so none is privileged.
        #
        # A REPL turn's error text sits wherever the failure happened in the stream, not at a
        # fixed end. An uncaught traceback closes the output, so a *suffix* carries it; code that
        # catches, prints, and keeps going puts it early, and then only a *prefix* carries it.
        # Measured on a 3129-char turn whose error text lands 2% in: at 1.0 the text survives, at
        # 0.5 and 0.0 it does not — the reverse of the case that motivated bounding this.
        #
        # So the ratio cannot guarantee the error text reaches the agent, at any value. Rejecting
        # 1.0 would forbid the one setting that works for early failures while leaving the
        # default, 0.5, dropping late ones at a small ceiling. The docstrings claiming "must stay
        # below 1.0" were the thing that was wrong; the range check is what it always was.
        if not 0.0 <= truncation_prefix_ratio <= 1.0:
            raise ValueError(
                "truncation_prefix_ratio must be between 0.0 and 1.0 "
                f"(got {truncation_prefix_ratio!r})"
            )
        for name, value in (
            ("system_prompt_template", system_prompt_template),
            ("user_prompt_template", user_prompt_template),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if subinvoke_description is not None and (
            not isinstance(subinvoke_description, str) or not subinvoke_description
        ):
            raise ValueError("subinvoke_description must be a non-empty string or None")
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template
        self.max_invoke_input_length = max_invoke_input_length
        self.max_repl_output_length = max_repl_output_length
        self.truncation_prefix_ratio = truncation_prefix_ratio
        self.subinvoke_description = subinvoke_description

    def parse(self, llm_response: LLMResponse) -> ParsedCode:
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
        still costs an iteration. That is the accepted trade for a protocol with no failure mode —
        see ``Agent._record_llm_response`` for how rare a content-less completion is in practice.
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
        return ParsedCode((llm_response.content or "").strip("\n\r"), None)

    def format_instructions(self) -> str:
        """Return the wire-format instruction block (see :meth:`BaseProtocol.format_instructions`).

        A constant: this protocol's request format does not vary with its rendering settings. It
        is the request half of the codec whose accept half is :meth:`~CodeOnlyProtocol.parse` — the two are edited
        together here so an instruction to "respond with ONLY code" can never drift from what
        ``parse`` actually accepts.
        """
        return _FORMAT_INSTRUCTIONS

    def get_history_description(self) -> str:
        """Return the ``__history__`` description (see :meth:`BaseProtocol.get_history_description`).

        Describes the :class:`REPLHistoryEntry` records this protocol writes. A constant: this
        protocol always writes that record, so it always describes it.

        To suppress the block — because the REPL cannot bind ``__history__``, or a hook unbinds it
        at runtime — override the rendering instead of the codec: point ``system_prompt_template``
        at a template that empties the ``repl_history`` block (``{% extends "jaz/system_prompt.jinja2"
        %}`` plus an empty ``{% block repl_history %}``), or install a hook that rewrites the
        system message before the query.
        """
        # The describe half of the seam whose write half is `build_history_entry`: the field list
        # below is exactly `REPLHistoryEntry`'s, and the two are edited together here so the prompt
        # cannot name a field the record does not carry.
        #
        # EXECUTIVE CALL (user): no `describe_repl_history` constructor switch. It was a
        # prompt-only knob on the codec, and suppressing a section of the system prompt is the
        # TEMPLATE's job — core already ships the two seams that do it (a `system_prompt_template`
        # naming a vendored template, and the hook path that rewrites messages at `LLMQueryEnter`,
        # as `jaz_evals.hooks.CodeAct` does). Rejected because it made the codec answer a question
        # about rendering, and because one boolean cannot express the cases that actually arise:
        # the evals description templates state a RICHER history block of their own, which needs
        # the core block gone rather than the record undescribed. The `repl_history` block exists
        # in the shipped template so that vendoring is two lines, not a copy that drifts.
        return _REPL_HISTORY_DESCRIPTION

    def render_observation(
        self,
        exec_result: Continue,
        iteration: int,
    ) -> list[MessageDict]:
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
            output,
            max_length=self.max_repl_output_length,
            prefix_ratio=self.truncation_prefix_ratio,
        )

        # The observation IS the output — the message carries no framing of its own (#928,
        # executive call). Two consequences worth knowing before adding any back:
        #
        # 1. No error branch. Dropping the ``*An error occurred:*`` flag (and, before it, the
        #    ``<error>`` block) leaves the error text to reach the agent as ordinary output, which
        #    means truncation can take it. Where it sits is not fixed: an uncaught traceback ends
        #    the output, so a retained *suffix* carries it, while code that catches an error,
        #    prints it, and keeps going puts it early, where only a retained *prefix* does. No
        #    ``truncation_prefix_ratio`` covers both — 0.5 drops an early one, 1.0 drops a late
        #    one — so on a truncated turn the agent may see no error text at all. Reinstating a
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
        # home would be `_REPL_HISTORY_DESCRIPTION`, but that is agent-facing text under the
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
        # `iteration` stays in the signature because `BaseProtocol.render_observation` specifies it
        # and out-of-tree protocols use it; this one no longer does.
        return [{"role": "user", "content": abbreviated_output}]

    def build_history_entry(
        self, llm_response: LLMResponse | None, exec_result: ExecResult
    ) -> REPLHistoryEntry:
        """Project one turn into its ``__history__`` entry (the **record** seam).

        The default entry records the response *text* (``llm_response.content`` — a missing
        response object, or an empty completion, projects to ``""``), the REPL ``output``, and
        the recoverable exception. ``llm_response`` arrives as the whole ``LLMResponse`` (not the
        pre-extracted content string), so a protocol that wants a richer record can fold in
        token counts / cost / provider ``extra`` from the same object — this default just drops
        them.

        ``repl_output`` is always a ``str``: the REPL ``output`` for a ``Continue``, and ``""`` for
        a terminal ``Return``/``Raise`` (which carry no ``output`` at all).

        ``repl_exception`` is populated only for a recoverable ``Continue`` — the raw exception
        object passed straight through (``Continue`` carries the exception, not a pre-baked
        summary); a plain success, a terminal ``Return``, and a terminal ``Raise``
        all record ``repl_exception=None`` (the raise's exception travels out of the loop, not
        into history). The object is kept rather than a summarized string because that text is
        already fully contained in ``repl_output`` (an error ``Continue``'s ``output`` is
        ``stdout + traceback``, whose final line is exactly what ``summarize_exception`` would
        emit) — mirroring :meth:`~CodeOnlyProtocol.render_observation`, which is the sibling seam that *does* render
        the exception to text (via ``summarize_exception``) for the in-band observation. Keeping
        both on this protocol is why a custom protocol can't desync its two representations.
        """
        content = llm_response.content if llm_response is not None else None
        repl_exception = (
            exec_result.exception
            if isinstance(exec_result, Continue) and exec_result.exception is not None
            else None
        )
        return REPLHistoryEntry(
            llm_response=content or "",
            # Only a ``Continue`` carries ``output`` (#903 dropped it from the terminal ``Return`` /
            # ``Raise``, whose value/exception — not agent-facing text — is the payload); a terminal
            # turn therefore records ``""`` — see the field comment on :class:`REPLHistoryEntry`
            # for why the empty string rather than ``None``.
            repl_output=exec_result.output if isinstance(exec_result, Continue) else "",
            repl_exception=repl_exception,
        )

    def get_subinvoke_description(self, invoke_tool: InvokeTool) -> str:
        # Author-supplied override wins; otherwise fall back to the base's closure introspection.
        # An override advertises a narrower surface than the closure supports (see the
        # `subinvoke_description` constructor param) without swapping the executable tool.
        if self.subinvoke_description is not None:
            return self.subinvoke_description
        return super().get_subinvoke_description(invoke_tool)

    def render_initial_message_list(
        self,
        inputs: Mapping[str, object],
        scope: Mapping[str, object],
        repl: BaseREPL,
        *,
        invoke_tool: InvokeTool | None,
        depth: int,
        recursion_available: bool,
        input_display_overrides: Mapping[str, DisplayText] | None,
    ) -> list[MessageDict]:
        """Build the [system, user] opener.

        ``inputs`` (explicit ``**inputs`` kwargs) and ``scope`` (resolved ``jaz.scope``)
        arrive as SEPARATE mappings, so the two sections render directly from their own
        dict — no merged namespace, no ``scoped_keys`` re-split. There is no separate
        ``task`` argument anymore: the prompt is an ordinary input."""

        # Separate inputs/scope mappings is #727; ``task`` as an ordinary input is #538.
        # Scoped inputs render in the SYSTEM prompt (ambient/global, like tool
        # libraries); render the whole `scope` mapping here and hand it to
        # get_system_prompt, along with their shown-name list for that section's closing
        # sentence — the same blocks-then-sentence shape the user prompt uses for explicit inputs.
        scoped_input_blocks, scoped_shown = render_input_section(
            scope,
            max_invoke_input_length=self.max_invoke_input_length,
            prefix_ratio=self.truncation_prefix_ratio,
        )
        user_prompt = get_user_prompt(
            user_prompt_template=self.user_prompt_template,
            inputs=inputs,
            max_invoke_input_length=self.max_invoke_input_length,
            # #568: the closing inputs sentence's wording is owned by the (config-selectable)
            # user_prompt_template, not by a REPL capability — a withheld-inputs variant flips it.
            prefix_ratio=self.truncation_prefix_ratio,
            # Same flag the system prompt gets: the two sections state opposite halves of the
            # propagation rule, so they have to appear or vanish together.
            recursion_available=recursion_available,
        )
        # The protocol owns how the recursive `invoke` tool is described to the agent (default:
        # introspect the closure; override via `subinvoke_description`). Render it here and hand
        # the STRING to get_system_prompt, which stays codec-agnostic. `None` when recursion is off.
        subinvoke_description = (
            self.get_subinvoke_description(invoke_tool)
            if invoke_tool is not None
            else None
        )
        system_prompt = get_system_prompt(
            system_prompt_template=self.system_prompt_template,
            subinvoke_description=subinvoke_description,
            depth=depth,
            recursion_available=recursion_available,
            repl=repl,
            scoped_input_blocks=scoped_input_blocks,
            scoped_names=scoped_shown,
            # The protocol owns the wire-format block and hands it to the template (#639);
            # the template only decides where it lands.
            format_instructions=self.format_instructions(),
            # Same arrangement for the `__history__` block, and with no gate at all: whether the
            # REPL binds the name is not checked here, and there is no switch to say it does not.
            # See `get_history_description` for what a host suppresses the block with instead.
            history_description=self.get_history_description(),
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
