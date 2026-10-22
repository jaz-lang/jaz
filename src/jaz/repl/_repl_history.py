"""The ``__history__`` record :class:`~jaz.repl.PythonREPL` keeps: the entry type and the prose
that describes it to the agent.

The two live in one module because the prose names the entry's fields and must change with them
(``tests/repl/test_repl_history.py`` pins that every field is described).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class REPLHistoryEntry:
    """One ``__history__`` entry — what :meth:`PythonREPL.update_repl_history` appends per turn.

    ``__history__`` is a plain list of one entry per iteration in order — ``[0]`` first, ``[-1]``
    most recent — with no init snapshot, so ``__history__[0]`` is the first iteration. Another
    REPL may record an entirely different shape, or none.

    Attributes:
        llm_response: The full model output that produced the turn — the raw response text,
            not the parsed code; ``""`` for an empty completion.
        repl_output: What the turn's code printed (stdout, plus the traceback on a recoverable
            error). Always a ``str``: a terminal ``Return``/``Raise`` turn records ``""``.
        repl_exception: The raw exception object from a recoverable error, or ``None`` on a
            clean success and on every terminal turn.
    """

    # The entry type is the REPL's, as the list is: `PythonREPL.initialize` creates the list and
    # binds it as `__history__`, `PythonREPL.update_repl_history` appends to it, and this module
    # states what it appends. The core loop holds no list and knows no entry shape; it hands
    # the REPL each turn's response and finalized result (see `Agent._finalize_iteration`),
    # because only the loop has them: the REPL's `exec` sees code, not the model's text, and a
    # hook may override the result after `exec` returned (a `SupplyExecResult` skips `exec`
    # altogether). Per `design/design_features/basics.md`'s Harness-as-Language / FAC
    # view, a REPL session's own history is a first-class, agent-visible projection of the loop
    # history into the namespace; the REPL that surfaces the namespace owns the projection.
    #
    # It used to be the protocol's (`BaseProtocol.build_repl_history_entry`, with the prose on
    # `BaseProtocol.repl_history_description`), on the argument that a protocol with a richer
    # wire format might record a richer entry. No second protocol exists, and the split put the
    # record's shape and words on one component and its binding on another, so the loop had to
    # compose "described" with "bound" and nothing made a protocol that returned `None` keep an
    # in-process REPL from binding an undescribed name. One owner makes "described iff bound"
    # hold by construction (the user's call, 2026-10-20).

    # No index-0 snapshot of the inputs: ``__history__`` is one entry per iteration, and a copy of
    # the inputs there would be a second path to values the REPL already binds by name — the same
    # second-door problem that rules out an ``__inputs__`` builtin (see ``PythonREPL.initialize``).

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
    # a free win: ``PythonREPL.update_repl_history`` appends to the same list the REPL binds as
    # ``__history__``, so a subagent finishing with ``return __history__`` hands its parent a list
    # the terminal entry then lands in by reference — which is what the ultrahorizon/sweb
    # meta-prompts iterate. To that reader ``""`` now asserts "this turn printed nothing", and a
    # finishing block that printed before returning makes that false (its stdout is captured and
    # dropped — see ``test_repl_history_appended_on_return_with_preceding_code``). Such a consumer
    # must key off POSITION, the last entry, rather than the value. An agent reading its *own*
    # history is unaffected: its terminal entry is appended after the turn ends, so it is never
    # there while that agent is reading. Recording the finishing block's real stdout would dissolve
    # both the ambiguity and the loss, but "a terminal turn carries a value/exception, not
    # agent-facing text" is #903's deliberate invariant; reversing it is a separate change.
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
    # sandboxed REPL, where ``from jaz._string_utils import summarize_exception`` is usually denied by
    # the default ``allowed_imports`` — can reconstruct the one-liner import-free as
    # ``f"{type(e).__name__}: {e}"``.
    repl_exception: BaseException | None


# The ``__history__`` description `PythonREPL.repl_history_description` returns, beside the entry
# whose fields it names so the two are edited together. It is not in
# `python_repl_description.jinja2` with the rest of the REPL's prose because its framing — a
# lead-in sentence and the enclosing `<__history__ type="list">` element — belongs to
# `system_prompt.jinja2`, the same division `{{ response_format_description }}` has: the template
# owns placement and framing, the REPL owns the words that have to track `REPLHistoryEntry`.
# Hence no opening announcement of the variable here, and a flat field list rather than one
# nested under such a bullet.
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
# message is abbreviated while `PythonREPL.update_repl_history` keeps `exec_result.output` whole, and
# the deleted `repl_output_truncation_advice*.jinja2` was the only thing that ever told the agent
# so. Stating it here would have been the natural home — this is where the entry's shape is
# described. It is not added because THIS STRING IS AGENT-FACING TEXT UNDER THE DEFAULT CONFIG, and
# published eval results have to reproduce under the exact prompts that produced them (executive
# call, the user). Changing the default system prompt is out of bounds regardless of how much
# better the wording would be; a host that wants the pointer states it in its own
# `system_prompt_template`.
#
# Two further reasons the sentence would have needed caveats anyway: a terminal turn records `""`
# (see the field comment on `REPLHistoryEntry`), and a REPL that truncates host-side before
# recording makes the entry lossy too.
_REPL_HISTORY_DESCRIPTION = """\
`__history__` is a list with one entry per REPL iteration, in order (`__history__[0]` is your first
iteration and `__history__[-1]` the most recent). Each entry has:
- `.llm_response (str)`: Your full response for that iteration containing your code
- `.repl_output (str)`: The printed output from that iteration, including any error traceback
- `.repl_exception (BaseException | None)`: The exception object raised if that iteration hit
  a recoverable error, else None
"""
