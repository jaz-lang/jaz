"""Compaction hook: summarize old turns to keep the conversation within the window.

When the buffer approaches the model's context window, ``Compaction`` replaces a span
of old turns with an LLM-generated summary — *persistently*, so the compaction carries
forward. It reads each message's provenance to pin the load-bearing seed
(``MessageKind.SEED`` — system prompt + task) and to roll any prior summary into the new
one, then emits a persistent ``DropMessages`` + ``AddMessages``.

Contrast with the other context hooks:
- ``SlidingWindow`` *drops* old turns (transient; the information is simply lost).
- ``ContextWindow`` only *warns* the agent.
- ``Compaction`` *replaces* a span with a summary — lossy but information-preserving,
  and it persists into the carried buffer.

**Summary generation.** By default the hook formats the dropped turns into a task and runs
it as a nested ``jaz.invoke`` (``jaz.ReturnType(str)``) — *not* a direct ``client.complete``.
This is deliberate: a nested invoke fires the normal ``LLMQueryExit`` events,
so the summary's tokens are counted and enforced by ``BudgetPool`` like any other invoke,
whereas a raw client call escaped budget accounting entirely (the summary spend was
invisible on exactly the long runs where budgets matter most). The nested invoke uses the
invoke's effective ``llm_client`` / ``model_config`` and a forced ``repl="python"`` so the
fixed ``return \"\"\"...\"\"\"`` output contract in the summary prompt is always valid,
independent of the parent's REPL. Recursion is prevented by an ``_in_summarization``
re-entrancy guard: while the summary invoke runs, this hook is a no-op, so a large summary
task can't re-trigger compaction on itself. Pass ``summarizer=`` to override with a custom
strategy (or a deterministic one for tests) — a custom summarizer bypasses the invoke, so
budgeting/tracking of its calls is the caller's responsibility.

**Known limitations:**
- The summary invoke is synchronous (``_dispatch_event`` is sync), so under ``ainvoke`` it blocks
  the event loop for the summarization. Acceptable for now; an ``acomplete``-via-thread
  variant is a possible follow-up.
- If the pinned prefix + ``keep_recent`` tail already exceed the window there is no middle
  to compact → no-op. Pair with ``SlidingWindow`` or rely on the provider's own
  context limit as the backstop.
- A compaction turn pays one extra summary invoke over the dropped span (reusing
  ``model_config`` — a cheaper-model knob is a future addition); its cost IS tracked and
  budgeted (see *Summary generation*).
- The summary invoke is a *nested* invoke, so it consumes one level of recursion depth. If a
  ``RecursionLimit`` is installed, an agent compacting at its recursion cap has no spare level
  for the summary, so the summary invoke fails (caught → compaction skipped that turn, never
  fatal). With no ``RecursionLimit`` (the default) recursion is unbounded and this never bites.
- ``summarizer`` / ``summary_prompt`` don't round-trip through config serialization; use
  this as a scoped/``local_hooks`` hook, not a serialized baseline.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextvars import ContextVar
from typing import TYPE_CHECKING

from jaz.provenance import MessageKind, provenance_of
from jaz.providers import MessageDict

from ...repl.python_repl import PythonREPL
from ..dispatcher import Effect, Hook
from ..effects import AddMessages, DropMessages
from ..events import LLMQueryEnter
from .sliding_window import _estimate_tokens

if TYPE_CHECKING:
    from jaz.config import Config

logger = logging.getLogger(__name__)

# Re-entrancy guard: True while the built-in summarizer's nested jaz.invoke is running, so
# Compaction.on_llm_query_enter no-ops for the summary invoke's own queries and a large
# summary task can't re-trigger compaction on itself. A ContextVar (not an instance flag) so
# it scopes to the running context and is seen by the nested invoke, which runs in it.
_in_summarization: ContextVar[bool] = ContextVar(
    "compaction_in_summarization", default=False
)

_DEFAULT_SUMMARY_PROMPT = (
    "You are compacting an agent's conversation to fit its context window. Summarize the "
    "earlier turns below concisely but faithfully — preserve the key decisions, facts, "
    "results, and any state needed to continue the task. Do NOT execute any other code or "
    "use any tools. Respond with EXACTLY one REPL input returning the summary as a string, "
    "in this exact form (triple-quoted so multi-line summaries are valid):\n\n"
    'return """<your summary here>"""'
)


class Compaction(Hook):
    """Summarize old conversation turns when the buffer nears the context window.

    Once a turn's prompt is estimated to exceed ``fraction`` of the model's context window,
    the middle of the conversation is replaced with a summary of it: the task and system
    prompt are always kept, so are the most recent ``keep_recent`` messages, and everything
    between them is summarized into one message. Re-compacting folds the previous summary
    into the new one, so summaries do not accumulate however long the run goes.

    The replacement is carried forward: later turns see the summary, not the messages it
    replaced.

    Compaction is best-effort: if the summarizer fails, or the span is smaller than
    ``min_compact``, the turn proceeds uncompacted rather than failing the invoke. The
    provider's own context limit remains the backstop, so a run can still hit it.

    Under ``with``, every invoke in scope compacts its own conversation. Passed positionally,
    only that invoke's is compacted — this hook does not compact the invokes it nests.

    Args:
        fraction: Compact once the estimated prompt exceeds this fraction of the context
            window (``0 < fraction <= 1``).
        keep_recent: Number of most-recent messages kept verbatim, never summarized.
            Default 6.
        summarizer: ``(span_messages) -> summary_text`` override. ``None`` (default) uses
            the built-in LLM summarizer, which runs on the same model as the invoke.
        summary_prompt: Instruction text for the built-in summarizer. Ignored when
            ``summarizer`` is set.
        min_compact: Skip compaction when the span is smaller than this, so a handful of
            messages is not churned into a summary. Default 2.

    Raises:
        ValueError: If ``fraction`` is outside ``(0, 1]``, ``keep_recent`` is negative, or
            ``min_compact`` is less than 1.

    Example::

        with Compaction(fraction=0.7):
            jaz.invoke(...)
    """

    # Mechanics kept out of the docstring: handles `LLMQueryEnter`, and emits a **persistent**
    # `DropMessages` + `AddMessages` pair anchored to that enter snapshot — persistent so the
    # replacement is carried into the buffer rather than applying to one query, and anchored so
    # the list shown this turn is already compacted.
    #
    # The kept prefix is pinned by *provenance* (`MessageKind.SEED`), not by role. See
    # `_compaction_span` for why there is deliberately no role-based fallback for an unstamped
    # buffer. The span is everything non-pinned outside the recent tail, which includes any
    # prior `ADDED` compaction summary — that is what makes re-compaction *roll* rather than
    # accumulate summaries.
    #
    # The built-in summarizer runs as a nested `jaz.invoke`, so it is tracked and budgeted like
    # any other invoke, and is guarded by `_in_summarization` so its own long prompt cannot
    # re-trigger compaction.
    #
    # A failed summarization is swallowed (logged at warning) rather than aborting: robustness
    # first — losing compaction degrades the run, aborting ends it.

    def __init__(
        self,
        *,
        fraction: float,
        keep_recent: int = 6,
        summarizer: Callable[[list[MessageDict]], str] | None = None,
        summary_prompt: str | None = None,
        min_compact: int = 2,
    ) -> None:
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if keep_recent < 0:
            raise ValueError(f"keep_recent must be >= 0, got {keep_recent}")
        if min_compact < 1:
            raise ValueError(f"min_compact must be >= 1, got {min_compact}")
        self.fraction = fraction
        self.keep_recent = keep_recent
        self.summarizer = summarizer
        self.summary_prompt = summary_prompt or _DEFAULT_SUMMARY_PROMPT
        self.min_compact = min_compact

    def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Effect]:
        # Re-entrancy guard: don't compact the built-in summarizer's own nested invoke
        # (its task carries the whole span, so it could itself cross the threshold).
        if _in_summarization.get():
            return []
        messages = event.messages
        window = self._max_input_tokens(event.config)
        if window is None or _estimate_tokens(messages) <= self.fraction * window:
            return []

        span_indices = self._compaction_span(messages)
        if len(span_indices) < self.min_compact:
            return []  # nothing worth compacting (or prefix + recent already fill it)

        span_messages = [messages[i] for i in span_indices]
        try:
            summary_text = self._summarize(span_messages, event.config)
        except Exception:
            # Robustness-first: a failed summary must not abort the invoke. Skip this
            # turn's compaction — the provider's own context limit remains the backstop.
            logger.warning(
                "Compaction: summarization failed; skipping compaction this turn",
                exc_info=True,
            )
            return []
        if not summary_text:
            return []

        summary_msg: MessageDict = {
            "role": "user",
            "content": f"[Summary of earlier turns]\n{summary_text}",
        }
        # Persistent: the agent folds these into the carried buffer AND stamps the summary
        # with MessageKind.ADDED (reason="compaction") — the hook does not stamp itself.
        # Both anchor to this enter snapshot, so the shown list this turn is compacted too.
        return [
            DropMessages(indices=set(span_indices), persistent=True),
            AddMessages(
                [summary_msg],
                index=min(span_indices),
                persistent=True,
            ),
        ]

    # --- span selection -------------------------------------------------------

    def _compaction_span(self, messages: list[MessageDict]) -> list[int]:
        """Indices to compact: everything between the pinned seed and the recent tail.

        Pins the seed (system prompt + task, plus any protocol-seeded prefix) by
        *provenance* (``MessageKind.SEED``) — never summarized. No role-based fallback for
        an un-stamped buffer: the core agent loop always stamps the seed before any query
        (``Agent._stamp_seed_provenance``), so every buffer reaching this hook via the loop
        is stamped; an empty pinned set means the seed was deliberately dropped (or a hook
        hand-built an unstamped buffer), where fabricating a pin by role would be
        wrong. Matches ``SlidingWindow`` (#606). Keeps the last ``keep_recent``
        non-pinned messages verbatim. The remaining middle — which includes any prior
        ``ADDED`` compaction summary, so re-compaction *rolls* it in — is the span.
        """
        n = len(messages)
        pinned = {
            i
            for i, m in enumerate(messages)
            if (p := provenance_of(m)) is not None and p.kind is MessageKind.SEED
        }
        non_pinned = [i for i in range(n) if i not in pinned]
        recent = set(non_pinned[-self.keep_recent :]) if self.keep_recent else set()
        return [i for i in non_pinned if i not in recent]

    # --- summary generation ---------------------------------------------------

    def _summarize(self, span: list[MessageDict], config: Config) -> str:
        if self.summarizer is not None:
            return self.summarizer(span)
        # Imported lazily: this module is loaded during jaz package init, so a top-level
        # `import jaz` would be circular.
        import jaz

        transcript = "\n\n".join(
            f"{m.get('role', '?')}: {_message_text(m)}" for m in span
        )
        prompt = f"{self.summary_prompt}\n\n{transcript}"
        # Run the summary as a nested invoke (tracked/budgeted), guarded so it can't re-fire
        # compaction. Force a Python REPL so the summary prompt's ``return`` contract is valid
        # regardless of the parent's REPL; reuse the effective backend so the summary runs on
        # the same model the invoke is using.
        token = _in_summarization.set(True)
        # The session's own REPL when it is already Python, a default one otherwise. The intent
        # here is "force Python", NOT "reset the sandbox" — and with groups replacing rather than
        # merging, a bare `PythonREPL()` would express the second: a session running
        # `PythonREPL(allowed_imports=["json"], exec_timeout=600)` would silently get a
        # deny-all, 30 s summariser. Reusing the instance keeps the summariser sandboxed exactly
        # like the agent it summarises, which is what the old merging override did.
        repl = config.repl if isinstance(config.repl, PythonREPL) else PythonREPL()
        try:
            # The invoke's own backend, passed straight through — there is no bag to
            # reconstruct it from, and reusing the object means the summary provably runs on
            # the same model rather than on a rebuild that might differ.
            with jaz.ConfigOverride(llm=config.llm, repl=repl):
                summary = jaz.invoke(jaz.ReturnType(str), task=prompt)
        finally:
            _in_summarization.reset(token)
        return summary or ""

    @staticmethod
    def _max_input_tokens(config: Config) -> int | None:
        # Mirrors ContextWindow._compute_max_input_tokens: size the window from the
        # invoke's EFFECTIVE config (carried on the event), not ambient get_config().
        # TODO(#718): reaching max_input_tokens via get_model_info is still duplicated with
        # ContextWindow; expose it directly on LLM.
        info = config.llm.get_model_info()
        if not info:
            return None
        return info.get("max_input_tokens")


def _message_text(message: MessageDict) -> str:
    """Best-effort plain text of a message's content, for building the summary transcript.

    Assumes the cross-provider-common shape of ``content``: a plain ``str``, or a list of
    content-part dicts where text parts carry a ``"text"`` field (the OpenAI ∩ Anthropic
    format — both use ``[{"type": "text", "text": ...}, ...]``). Text parts are inlined
    verbatim.

    Non-text parts (image / tool-use / tool-result blocks, whose shapes differ between
    providers) are **not silently dropped**: each becomes a short ``[<type>: <ref>]``
    placeholder — an image source/url, a tool name, or a ``tool_use_id`` when one can be
    found. This follows the compaction convention (see #610 review research) of preserving
    the *signal* that content existed rather than omitting it, so the summarizer knows an
    image or tool step was there without ballooning the request with the raw block (dumping
    raw ``tool_use`` markup risks the post-compaction model pattern-completing it as text).

    Still lossy by design — a placeholder is not the content, so exact tool outputs / image
    pixels don't survive. Known gap: message-level OpenAI ``tool_calls`` live *outside*
    ``content`` and are not captured here. For jaz's default REPL protocol this is all moot —
    REPL I/O is plain text, so the transcript is effectively lossless.
    """
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(_part_text(part) for part in content)
    return str(content)


# Longest reference (image url/path) kept in a placeholder before truncation — enough to
# stay recognizable (e.g. a filename or a data-URL's mime prefix) without inlining a blob.
_MAX_REF_LEN = 48


def _part_text(part: object) -> str:
    """Text of one content part, or a short ``[type: ref]`` placeholder for a non-text part."""
    if not isinstance(part, dict):
        return str(part)
    if "text" in part:
        return str(part["text"])
    label = str(part.get("type", "content"))
    ref = _part_reference(part)
    return f"[{label}: {ref}]" if ref else f"[{label}]"


def _part_reference(part: dict) -> str | None:
    """A short human reference for a non-text part (image source, tool name, ...), if any."""
    # Image: OpenAI {"image_url": {"url": ...}}; Anthropic {"source": {"url"/path/...}}.
    image_url = part.get("image_url")
    if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
        return _abbrev_ref(image_url["url"])
    source = part.get("source")
    if isinstance(source, dict):
        for key in ("url", "path", "source_path", "file_id"):
            if isinstance(source.get(key), str):
                return _abbrev_ref(source[key])
    # Tool use / result: the tool name, else the id it correlates with.
    for key in ("name", "tool_use_id", "id"):
        if isinstance(part.get(key), str):
            return _abbrev_ref(part[key])
    return None


def _abbrev_ref(ref: str) -> str:
    return ref if len(ref) <= _MAX_REF_LEN else ref[:_MAX_REF_LEN] + "…"


#: Deprecated alias for the pre-rename spelling — see the rationale block in
#: ``jaz/hooks/__init__.py``. Every renamed hook carries this alias at its definition
#: site so the deep-path import keeps working and so the alias map stays checkable
#: (``test_every_renamed_hook_has_an_alias``).
CompactionHook = Compaction
