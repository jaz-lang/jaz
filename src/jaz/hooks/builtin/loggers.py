"""Logger hooks that write to files or terminal.

These hooks provide logging functionality similar to the old listener system,
but using the new hook architecture.
"""

import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from jaz.hooks.base import Event
from jaz.hooks.dispatcher import Hook
from jaz.hooks.events.base import Aborted, Failed
from jaz.llm import MessageDict
from jaz.provenance import MessageKind, provenance_of, to_wire_messages
from jaz.string_utils import abbrev_repr, summarize_exception

if TYPE_CHECKING:
    from jaz.hooks.events.llm_query import ResolvedMessageEdits


def _new_message_lines(
    *,
    messages: Sequence[MessageDict],
    invoke_id: str,
    logged_prompt_invokes: set[str],
    logged_message_ids: set[str],
) -> list[MessageDict]:
    """The wire messages to emit as `[Agent] message:` lines for this query's buffer —
    each buffer message exactly once.

    Messages are identified by their stable provenance id (#599), mirroring ATIFTrace's
    emission rule: a stamped message is logged the first time it appears in the buffer
    (whatever its position), and ASSISTANT messages are skipped here because the reply is
    logged at LLMQueryExit. This replaced a positional ``messages[-1]``-only increment,
    which silently dropped all but the last message appended between two queries (e.g.
    the truncation-advice observation CodeOnlyProtocol emits alongside a truncated
    output) and never logged a persistent hook add (a compaction summary) at all.

    This stream is the carried *buffer* content as of LLMQueryEnter; per-query edits are
    logged separately as ``[Agent] message added/dropped:`` lines from LLMQuerySend's
    pre-resolved records (see :func:`_edit_lines`) — together the two streams record
    everything the model was shown, mirroring ATIFTrace's step-stream/``extra.edits``
    split. ADDED-kind messages (persistent hook adds already in the buffer, e.g. a
    compaction summary) are skipped here for the same reason ATIF skips them: their
    content was logged by the add line on the turn they were sent, so re-logging them on
    buffer entry would double them. The ASSISTANT skip
    assumes this logger observed every assistant turn's LLMQueryExit: a logger attached
    mid-conversation, or a buffer seeded with prior-conversation messages that keep
    ASSISTANT provenance, never logs those earlier assistant turns (the old first-query
    full dump did) — accepted, since a partial-scope logger owes a partial transcript.
    (The same partial-scope trade applies to ADDED messages whose add line predates the
    logger's attachment.)

    Un-stamped messages (a hand-built buffer with no provenance) keep the legacy
    positional fallback: dump the whole buffer on the invoke's first query, then only
    ``messages[-1]`` on later ones.
    """
    first_query = invoke_id not in logged_prompt_invokes
    logged_prompt_invokes.add(invoke_id)
    to_log = []
    for position, msg in enumerate(messages):
        prov = provenance_of(msg) if isinstance(msg, dict) else None
        if prov is not None:
            if prov.kind is MessageKind.ASSISTANT:
                continue  # the reply is logged at LLMQueryExit
            if prov.kind is MessageKind.ADDED:
                continue  # content was logged by its `[Agent] message added:` line
            if prov.id in logged_message_ids:
                continue
            logged_message_ids.add(prov.id)
            to_log.append(msg)
        elif first_query or position == len(messages) - 1:
            to_log.append(msg)
    return to_wire_messages(to_log)


def _outcome_failure(outcome: Aborted | Failed) -> tuple[str, str]:
    """Render a non-``Completed`` outcome as ``(variant name, exception text)`` for a
    failure log line.

    The variant IS the status (no parallel enum), so its lowercased class name is the
    serialized form — the same spelling ATIFTrace and OTelTracing write, so "how did this
    span end" greps identically across all three records. ``summarize_exception`` rather
    than ``str``: multiple aborts at one boundary group into an ``ExceptionGroup`` by law,
    and ``str()`` on a group renders only a sub-exception count, dropping every child's
    text — on exactly the failure records these lines exist for.
    """
    exception = outcome.exception
    return type(outcome).__name__.lower(), summarize_exception(exception)


def _edit_lines(edits: "ResolvedMessageEdits") -> tuple[list[str], list[str]]:
    """Render one query's resolved edit records (``LLMQuerySend.edits``) as the payloads
    of ``[Agent] message added:`` / ``[Agent] message dropped:`` lines.

    Returns ``(added, dropped)`` payload strings. Adds carry their full wire content plus
    the resolved position and persistence — transient adds (per-turn warnings) appear on
    every turn they are actually sent, which is the truthful record; persistent adds
    appear once here and are then skipped by the buffer stream (ADDED provenance, see
    :func:`_new_message_lines`). Drops carry the dropped message's stable id + persistence
    rather than its content (the content was already logged when the message first
    appeared; ATIF's ``extra.edits`` makes the same id-not-content choice) — except an
    un-stamped dropped message, which has no id to reference, so its content is carried
    inline.
    """
    added = [
        str(
            {
                "position": record.position,
                "persistent": record.persistent,
                "message": to_wire_messages([msg])[0],
            }
        )
        for record in edits.adds
        for msg in record.messages
    ]
    dropped = []
    for record in edits.drops:
        prov = provenance_of(record.message)
        payload: dict = {"persistent": record.persistent}
        if prov is not None:
            payload["id"] = prov.id
        else:
            payload["message"] = to_wire_messages([record.message])[0]
        dropped.append(str(payload))
    return added, dropped


class PrintLogger(Hook):
    """Log every event to the terminal, with colors.

    A pure observer: it emits no effects, so it never influences the run.

    Under ``with`` it logs invokes nested inside too. Passed positionally, it logs only that
    invoke's events.

    Args:
        level: Logging level for the emitted records.
        max_field_length: Truncate any rendered field longer than this. ``None`` disables
            truncation.
    """

    def __init__(self, *, level: int = logging.INFO, max_field_length: int = 1000):
        self._max_field_length = max_field_length
        # Dedup state for the `[Agent] message:` stream (see _new_message_lines): stamped
        # messages are logged once by provenance id; the invoke set drives the legacy
        # fallback for un-stamped buffers.
        self._logged_prompt_invokes: set[str] = set()
        self._logged_message_ids: set[str] = set()
        name = f"jaz.hooks.print_logger.{id(self)}"
        self._logger = logging.Logger(name, level=level)
        self._logger.propagate = False
        # Bind to the host's live stdout captured *at construction* (`sys.stdout`), NOT
        # `sys.__stdout__`. Two reasons:
        #  - Capture-immunity: this is a saved reference distinct from the REPL's per-exec
        #    output capture, so diagnostics never land in an agent's observation buffer —
        #    including the subagent case, where a nested invoke's logger fires while the
        #    parent's exec-scoped capture is active. (Hooks are constructed before the
        #    invoke, so `sys.stdout` here is the host sink, not the capture proxy.)
        #  - Right destination: `sys.stdout` is where the human is actually looking — an
        #    ipykernel `OutStream` in a notebook, pytest's capture object under a test, the
        #    tty in a shell. `sys.__stdout__` is only "the console" in a bare terminal;
        #    empirically it is invisible in Jupyter under `capture_fd_output=False` and
        #    escapes pytest's `capsys`.
        #
        # Capture-immunity assumes construction happens *outside* a REPL exec — the normal
        # `with PrintLogger(): jaz.invoke(...)` case, and nested invokes (the instance is
        # reused via contextvars). The unenforced edge: an agent that constructs a
        # PrintLogger *inside* its own REPL exec binds to the live per-exec capture buffer,
        # reintroducing the leak this avoids. Also a forward seam: a ContextVar-routed
        # `sys.stdout` proxy (#649) that routes by contextvar *at write time* must keep this
        # binding pointed at the host sink (snapshot the underlying stream, or bind before
        # the proxy installs), or a logger write during an exec would route into the capture
        # buffer despite construction happening first.
        original_stdout = sys.stdout
        assert original_stdout is not None
        self._color_enabled = original_stdout.isatty()

        def _color(text, code: str) -> str:
            if not self._color_enabled:
                return str(text)
            return f"\033[{code}m{text}\033[0m"

        self._color = _color

        formatter = logging.Formatter(
            "%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        stream_handler = logging.StreamHandler(original_stdout)
        stream_handler.setFormatter(formatter)
        self._logger.addHandler(stream_handler)

    def on_any(self, event: Event):
        """Log event to terminal."""
        from jaz.hooks.events import (
            InvokeEnter,
            InvokeExit,
            InvokeSend,
            LLMQueryEnter,
            LLMQueryExit,
            LLMQueryRetry,
            LLMQuerySend,
            REPLExecEnter,
            REPLExecExit,
        )
        from jaz.hooks.events.base import Completed

        match event:
            # Non-completed exits (#892 outcome union) get dedicated failure lines: the
            # exits fire on every path now, and a log that goes silent on exactly the
            # runs it is read for was the loggers' last observability gap. Matched ahead
            # of the Completed arms below, which unwrap the happy-path payload.
            case LLMQueryExit(model=model) if not isinstance(event.outcome, Completed):
                outcome_name, error = _outcome_failure(event.outcome)
                self._logger.info(
                    f"{self._color('[LLM]', '35')} {self._color('query exit', '31')}: "
                    f"model={self._color(model, '36')}, "
                    f"outcome={self._color(outcome_name, '31')}, "
                    f"error={self._color(error, '31')}"
                )

            case REPLExecExit(iteration=i, depth=d) if not isinstance(
                event.outcome, Completed
            ):
                outcome_name, error = _outcome_failure(event.outcome)
                self._logger.info(
                    f"{self._color('[Agent]', '36')} "
                    f"{self._color('repl iteration', '31')} "
                    f"{self._color(i, '33')} "
                    f"{self._color('output', '31')}: "
                    f"outcome={self._color(outcome_name, '31')}, "
                    f"error={self._color(error, '31')}, "
                    f"depth={self._color(d, '36')}"
                )

            case InvokeExit(depth=d) if not isinstance(event.outcome, Completed):
                outcome_name, error = _outcome_failure(event.outcome)
                self._logger.info(
                    f"{self._color('[Agent]', '36')} "
                    f"{self._color('invoke exit', '31')}: "
                    f"outcome={self._color(outcome_name, '31')}, "
                    f"error={self._color(error, '31')}, "
                    f"depth={self._color(d, '36')}"
                )

            case InvokeEnter(
                inputs=inp,
                scope=sc,
                depth=d,
            ):
                # Explicit `**inputs` kwargs and resolved ambient `jaz.scope` are logged as SEPARATE
                # fields (#727); scope is shown only when active, to keep the common line clean.
                scope_frag = (
                    f"scope={self._color(abbrev_repr(dict(sc), self._max_field_length), '34')}, "
                    if sc
                    else ""
                )
                message = (
                    f"{self._color('[Agent]', '36')} "
                    f"{self._color('invoke enter', '32')}: "
                    # The task is an ordinary input now (#538) — it appears in `inputs=`
                    # below, so there is no separate `prompt=` field to log.
                    f"inputs={self._color(abbrev_repr(dict(inp), self._max_field_length), '34')}, "
                    f"{scope_frag}"
                    f"depth={self._color(d, '36')}, "
                    # Governance active for this invoke (incl. this logger), #727. `event.hooks` is
                    # the live active set; rendered at this edge. `abbrev_repr` reprs the list,
                    # which reprs each hook — mapping `repr` first would double-encode them
                    # into quoted, escaped strings.
                    f"hooks={self._color(abbrev_repr(list(event.hooks), self._max_field_length), '36')}"
                )
                self._logger.info(message)

            case InvokeSend(inputs=inp, added_inputs=added, dropped_inputs=dropped):
                # The COMMITTED input set (post AddInputs/DropInputs). The enter line
                # above records the proposal, which misses hook-injected inputs (the
                # input-side #906 pattern — e.g. ultrahorizon's per-child `env`); this
                # line is what the invoke actually received. Hook-added/dropped fragments
                # carry names only (the values are already in inputs=) and appear only
                # when hooks edited the set, so the common no-edit line stays short.
                added_frag = (
                    f"hook_added={self._color(sorted(added), '33')}, " if added else ""
                )
                dropped_frag = (
                    f"hook_dropped={self._color(sorted(dropped), '33')}, "
                    if dropped
                    else ""
                )
                self._logger.info(
                    f"{self._color('[Agent]', '36')} "
                    f"{self._color('invoke send', '32')}: "
                    f"inputs={self._color(abbrev_repr(dict(inp), self._max_field_length), '34')}, "
                    f"{added_frag}"
                    f"{dropped_frag}"
                    f"depth={self._color(event.depth, '36')}"
                )

            case InvokeExit(outcome=Completed(result=result), depth=d):
                message = (
                    f"{self._color('[Agent]', '36')} "
                    f"{self._color('invoke exit', '31')}: "
                    f"result={self._color(abbrev_repr(result, self._max_field_length), '35')}, "
                    f"depth={self._color(d, '36')}"
                )
                self._logger.info(message)

            case REPLExecEnter(iteration=i, code=code, depth=d):
                message = (
                    f"{self._color('[Agent]', '36')} "
                    f"{self._color('repl iteration', '32')} "
                    f"{self._color(i, '33')} "
                    f"{self._color('input', '32')}: "
                    f"code={self._color(repr(code), '34')}, "
                    f"depth={self._color(d, '36')}"
                )
                self._logger.info(message)

            case REPLExecExit(
                iteration=i,
                outcome=Completed(result=result),
                depth=d,
            ):
                message = (
                    f"{self._color('[Agent]', '36')} "
                    f"{self._color('repl iteration', '32')} "
                    f"{self._color(i, '33')} "
                    f"{self._color('output', '32')}: "
                    f"exec_result={self._color(abbrev_repr(result, self._max_field_length), '35')}, "
                    f"depth={self._color(d, '36')}"
                )
                self._logger.info(message)

            case LLMQueryEnter(invoke_id=invoke_id, messages=messages, model=model):
                # Emit `[Agent] message:` lines for conversation content — the source of
                # message-content logging now the Message event is gone, and what
                # log_to_markdown reconstructs the transcript from (every role, not just
                # assistant). Each buffer message is logged exactly once, keyed by its
                # provenance id; see _new_message_lines for the rule and its scope.
                new_wire = _new_message_lines(
                    messages=messages,
                    invoke_id=invoke_id,
                    logged_prompt_invokes=self._logged_prompt_invokes,
                    logged_message_ids=self._logged_message_ids,
                )
                for wire in new_wire:
                    self._logger.info(
                        f"{self._color('[Agent]', '36')} "
                        f"{self._color('message', '32')}: "
                        f"{self._color(wire, '33')}"
                    )
                message = (
                    f"{self._color('[LLM]', '35')} {self._color('query enter', '32')}: "
                    f"model={self._color(model, '36')}, "
                    f"messages={self._color(len(messages), '33')} messages"
                )
                self._logger.info(message)

            case LLMQuerySend(edits=edits):
                # This query's committed edits — what the buffer stream above cannot
                # carry (transient adds never enter the buffer; drops leave no trace in
                # it). See _edit_lines for the add/drop payload split.
                added, dropped = _edit_lines(edits)
                for payload in added:
                    self._logger.info(
                        f"{self._color('[Agent]', '36')} "
                        f"{self._color('message added', '32')}: "
                        f"{self._color(payload, '33')}"
                    )
                for payload in dropped:
                    self._logger.info(
                        f"{self._color('[Agent]', '36')} "
                        f"{self._color('message dropped', '31')}: "
                        f"{self._color(payload, '33')}"
                    )

            # Matching the Completed variant both narrows and unwraps the payload — the
            # non-completed arms were already skipped above.
            case LLMQueryExit(outcome=Completed(result=resp), model=model):
                cost = resp.cost_usd
                message = (
                    f"{self._color('[LLM]', '35')} {self._color('query exit', '32')}: "
                    f"model={self._color(model, '36')}, "
                    f"cost={self._color(f'${cost:.4f}' if cost is not None else 'N/A', '33')}"
                )
                self._logger.info(message)
                # Log the assistant reply as a message line — the source of assistant-content
                # logging now that the Message event is gone. Written in the
                # `[Agent] message: {dict}` form for human reading of the log.
                assistant_msg = to_wire_messages(
                    [{"role": "assistant", "content": resp.content}]
                )[0]
                self._logger.info(
                    f"{self._color('[Agent]', '36')} "
                    f"{self._color('message', '32')}: "
                    f"{self._color(assistant_msg, '33')}"
                )

            case LLMQueryRetry(
                model=model,
                attempt_number=attempt,
                exception=exc,
                wait_seconds=wait,
            ):
                message = (
                    f"{self._color('[LLM]', '35')} {self._color('retry', '33')}: "
                    f"model={self._color(model, '36')}, "
                    f"attempt={self._color(attempt, '33')}, "
                    f"error={self._color(f'{type(exc).__name__}: {exc}', '31')}, "
                    f"wait={self._color(f'{wait:.1f}s', '33')}"
                )
                self._logger.info(message)

        return []

    def teardown(self, exc: BaseException | None = None) -> None:
        # Release the dedup state (grows with invokes/messages). Unlike
        # FileLogger, don't touch the handler — it wraps the host's live stdout, which
        # this logger borrows but does not own.
        self._logged_prompt_invokes.clear()
        self._logged_message_ids.clear()


class FileLogger(Hook):
    """Log every event to a file, without colors.

    A pure observer: it emits no effects, so it never influences the run.

    Under ``with`` it logs invokes nested inside too. Passed positionally, it logs only that
    invoke's events.

    Args:
        file_path: Destination file; opened on activation and closed when the scope exits.
        level: Logging level for the emitted records.
        max_field_length: Truncate any rendered field longer than this. ``None`` disables
            truncation.
    """

    def __init__(
        self,
        file_path: str | Path,
        *,
        level: int = logging.INFO,
        max_field_length: int = 1000,
    ):
        self._max_field_length = max_field_length
        # Dedup state for the `[Agent] message:` stream (see _new_message_lines /
        # PrintLogger).
        self._logged_prompt_invokes: set[str] = set()
        self._logged_message_ids: set[str] = set()
        self.file_path = Path(file_path)
        name = f"jaz.hooks.file_logger.{id(self)}"
        self._logger = logging.Logger(name, level=level)
        self._logger.propagate = False

        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        formatter = logging.Formatter(
            "%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler = logging.FileHandler(self.file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

    def on_any(self, event: Event):
        """Log event to file."""
        from jaz.hooks.events import (
            InvokeEnter,
            InvokeExit,
            InvokeSend,
            LLMQueryEnter,
            LLMQueryExit,
            LLMQueryRetry,
            LLMQuerySend,
            REPLExecEnter,
            REPLExecExit,
        )
        from jaz.hooks.events.base import Completed

        match event:
            # Non-completed exits (#892 outcome union) get dedicated failure lines — see
            # PrintLogger.on_any for the rationale.
            case LLMQueryExit(model=model) if not isinstance(event.outcome, Completed):
                outcome_name, error = _outcome_failure(event.outcome)
                self._logger.info(
                    f"[LLM] query exit: model={model}, "
                    f"outcome={outcome_name}, error={error}"
                )

            case REPLExecExit(iteration=i, depth=d) if not isinstance(
                event.outcome, Completed
            ):
                outcome_name, error = _outcome_failure(event.outcome)
                self._logger.info(
                    f"[Agent] repl iteration {i} output: "
                    f"outcome={outcome_name}, error={error}, depth={d}"
                )

            case InvokeExit(depth=d) if not isinstance(event.outcome, Completed):
                outcome_name, error = _outcome_failure(event.outcome)
                self._logger.info(
                    f"[Agent] invoke exit: outcome={outcome_name}, "
                    f"error={error}, depth={d}"
                )

            case InvokeEnter(
                inputs=inp,
                scope=sc,
                depth=d,
            ):
                # Explicit `**inputs` kwargs and resolved ambient `jaz.scope` are logged as SEPARATE
                # fields (#727); scope is shown only when active, to keep the common line clean.
                scope_frag = (
                    f"scope={abbrev_repr(dict(sc), self._max_field_length)}, "
                    if sc
                    else ""
                )
                message = (
                    # The task is an ordinary input now (#538) — it appears in `inputs=`
                    # below, so there is no separate `prompt=` field to log.
                    f"[Agent] invoke enter: "
                    f"inputs={abbrev_repr(dict(inp), self._max_field_length)}, "
                    f"{scope_frag}"
                    f"depth={d}, "
                    # Governance active for this invoke (incl. this logger), #727. `event.hooks` is
                    # the live active set; rendered at this edge. `abbrev_repr` reprs the list,
                    # which reprs each hook — mapping `repr` first would double-encode them
                    # into quoted, escaped strings.
                    f"hooks={abbrev_repr(list(event.hooks), self._max_field_length)}"
                )
                self._logger.info(message)

            case InvokeSend(inputs=inp, added_inputs=added, dropped_inputs=dropped):
                # The COMMITTED input set — see PrintLogger's arm for the rationale.
                added_frag = f"hook_added={sorted(added)}, " if added else ""
                dropped_frag = f"hook_dropped={sorted(dropped)}, " if dropped else ""
                self._logger.info(
                    f"[Agent] invoke send: "
                    f"inputs={abbrev_repr(dict(inp), self._max_field_length)}, "
                    f"{added_frag}"
                    f"{dropped_frag}"
                    f"depth={event.depth}"
                )

            case InvokeExit(outcome=Completed(result=result), depth=d):
                message = f"[Agent] invoke exit: result={abbrev_repr(result, self._max_field_length)}, depth={d}"
                self._logger.info(message)

            case REPLExecEnter(iteration=i, code=code, depth=d):
                message = (
                    f"[Agent] repl iteration {i} input: code={repr(code)}, depth={d}"
                )
                self._logger.info(message)

            case REPLExecExit(
                iteration=i,
                outcome=Completed(result=result),
                depth=d,
            ):
                message = (
                    f"[Agent] repl iteration {i} output: "
                    f"exec_result={abbrev_repr(result, self._max_field_length)}, depth={d}"
                )
                self._logger.info(message)

            case LLMQueryEnter(invoke_id=invoke_id, messages=messages, model=model):
                # Each buffer message is logged exactly once (assistant replies at
                # LLMQueryExit), keyed by provenance id — which keeps observations in the
                # `[Agent] message:` stream log_to_markdown reconstructs from. See
                # _new_message_lines for the rule and its scope.
                new_wire = _new_message_lines(
                    messages=messages,
                    invoke_id=invoke_id,
                    logged_prompt_invokes=self._logged_prompt_invokes,
                    logged_message_ids=self._logged_message_ids,
                )
                for wire in new_wire:
                    self._logger.info(f"[Agent] message: {wire}")
                message = (
                    f"[LLM] query enter: "
                    f"model={model}, "
                    f"messages={len(messages)} messages"
                )
                self._logger.info(message)

            case LLMQuerySend(edits=edits):
                # This query's committed edits — what the buffer stream above cannot
                # carry (transient adds never enter the buffer; drops leave no trace in
                # it). See _edit_lines for the add/drop payload split.
                added, dropped = _edit_lines(edits)
                for payload in added:
                    self._logger.info(f"[Agent] message added: {payload}")
                for payload in dropped:
                    self._logger.info(f"[Agent] message dropped: {payload}")

            # Matching the Completed variant both narrows and unwraps the payload — the
            # non-completed arms were already skipped above.
            case LLMQueryExit(outcome=Completed(result=resp), model=model):
                cost = resp.cost_usd
                message = f"[LLM] query exit: model={model}, " + (
                    f"cost=${cost:.4f}" if cost is not None else "cost=N/A"
                )
                self._logger.info(message)
                # Log the assistant reply as a message line — the source of assistant-content
                # logging now that the Message event is gone. Written in the
                # `[Agent] message: {dict}` form for human reading of the log; replay no
                # longer parses these lines (it resumes from the ATIF trace instead).
                assistant_msg = to_wire_messages(
                    [{"role": "assistant", "content": resp.content}]
                )[0]
                self._logger.info(f"[Agent] message: {assistant_msg}")

            case LLMQueryRetry(
                model=model,
                attempt_number=attempt,
                exception=exc,
                wait_seconds=wait,
            ):
                message = (
                    f"[LLM] retry: model={model}, "
                    f"attempt={attempt}, "
                    f"error={type(exc).__name__}: {exc}, "
                    f"wait={wait:.1f}s"
                )
                self._logger.info(message)

        return []

    def teardown(self, exc: BaseException | None = None) -> None:
        """Flush and close file handlers."""
        # Release the dedup state — it only grows over a logger's lifetime, so drop it
        # once logging is done.
        self._logged_prompt_invokes.clear()
        self._logged_message_ids.clear()
        for handler in self._logger.handlers:
            handler.flush()
            handler.close()
