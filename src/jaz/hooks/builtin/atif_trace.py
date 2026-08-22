"""Hook that emits ATIF (Agent Trajectory Interchange Format) traces.

Produces Harbor-compatible trajectory JSON directly from JAZ execution events,
without an intermediate conversion step.

Usage::

    from jaz import invoke
    from jaz.hooks import ATIFTrace

    with ATIFTrace(output_path="./trace.atif.json"):
        result = invoke(task="Do something")

    # Creates: ./trace.atif.json validated by harbor.utils.trajectory_validator

ATIF is an external standard (Harbor), currently ``ATIF-v1.7``; we target that version and
do not invent our own. Its ``extra`` field — the spec's slot for custom information — carries
message provenance and compaction edits, letting it be the sole conversation-history
format:

- ``extra.provenance`` on system/user steps: the message's :class:`MessageProvenance`
  (``provenance_of(msg).to_dict()``) — its true origin plus a stable per-message id — and
  ``extra.index``, the message's slot in the buffer at the enter it was first seen.
- ``extra.edits`` on the **last message emitted the turn the edits were composed**: the compaction
  drops/adds for that query, ``{"iteration": N, "dropped": [...], "added": [...]}``. ``iteration`` is
  the query the edits shaped (block-level, once) — distinct from the anchor step's own
  ``extra.iteration``, since the edits are composed at enter N but hang on the last message of turn
  N-1's output. A drop is ``{id, index, persistent}`` (its content already lives in that message's
  own step); an added message is ``{role, content, index}`` (+ ``provenance`` when stamped).
  ``index`` is the edit's input coordinate (the resolved slot). See ``on_llm_query_send``.

See:
    https://www.harborframework.com/docs/agents/trajectory-format
    https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md
"""

# The provenance channel is #599 (stable per-message ids); riding ``extra`` to make ATIF
# the sole conversation-history format is #565's decision.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jaz.hooks.dispatcher import Hook
from jaz.hooks.events import Completed
from jaz.hooks.events.invoke import InvokeEnter, InvokeExit, InvokeSend
from jaz.hooks.events.llm_query import (
    LLMQueryEnter,
    LLMQueryExit,
    LLMQueryRetry,
    LLMQuerySend,
)
from jaz.llm import MessageDict
from jaz.provenance import PROVENANCE_KEY, MessageKind, provenance_of
from jaz.repl.types import Raise, Return
from jaz.string_utils import summarize_exception

SCHEMA_VERSION = "ATIF-v1.7"

#: ATIF v1.7 ``MetricsSchema`` fields that ``LLMResponse`` carries inside ``extra`` rather
#: than as fields of its own, and that the emitter therefore lifts back to the top level
#: of ``metrics`` on the way out.
ATIF_EXTRA_HOISTED_METRICS = ("prompt_token_ids", "completion_token_ids", "logprobs")
# These three are optional, provider-rare and per-token — large enough that promoting them
# to `LLMResponse` fields would write a retention cost into the contract every backend
# nominally implements (see the comment on `LLMResponse`). Backends spell them with these
# ATIF names in `extra` already, which is what `extras_merge` standardizes on, so the
# emitter is where the format's conformance is owed and cheapest to pay.


def hoist_atif_metrics(metrics: dict[str, Any], extra: dict[str, Any]) -> None:
    """Move ATIF's top-level token-id/logprob metrics out of ``extra`` into ``metrics``.

    Mutates both: ``extra`` is left holding only genuinely provider-specific keys, which is
    what ATIF's ``extra`` is specified to be.
    """
    for key in ATIF_EXTRA_HOISTED_METRICS:
        if key in extra:
            metrics[key] = extra.pop(key)


def _serialize_namespace(namespace: dict[str, Any]) -> dict[str, Any]:
    """Serialize a name→value namespace (inputs or scope) to the ATIF extra form.

    Each value becomes ``{"type": <class name>, "repr_prefix_10000": <repr, truncated>}``.
    """
    return {
        k: {"type": type(v).__name__, "repr_prefix_10000": repr(v)[:10000]}
        for k, v in namespace.items()
    }


def _serialize_message(message: MessageDict) -> dict[str, Any]:
    """Serialize a message to JSON-safe form for the ``extra.edits`` record.

    Minimal local serializer: strips the internal ``PROVENANCE_KEY`` and re-emits it
    as a clean ``"provenance"`` sub-object, and falls back to ``repr`` for any non-JSON value.
    """
    result: dict[str, Any] = {}
    for key, value in message.items():
        if key == PROVENANCE_KEY:
            # Emitted as a clean "provenance" sub-object below, not the raw magic key.
            continue
        try:
            json.dumps(value)
            result[key] = value
        except (TypeError, ValueError):
            result[key] = repr(value)
    prov = provenance_of(message)
    if prov is not None:
        result["provenance"] = prov.to_dict()
    return result


@dataclass
class _InvokeState:
    """Tracks per-invoke state for ATIF step building."""

    invoke_id: str
    parent_invoke_id: str | None
    steps: list[dict[str, Any]] = field(default_factory=list)
    step_counter: int = 0
    model_name: str | None = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int = 0
    total_cost: float = 0.0
    total_repl_iterations: int = 0
    current_iteration: int | None = None
    # Ids of messages already emitted as steps (SEED/OBSERVATION), so each is emitted once even
    # though it reappears in every later enter's buffer. Keyed by the stable per-message id (#599).
    emitted_ids: set[str] = field(default_factory=set)
    # The step the most recent turn's ``extra.edits`` hangs on: that turn's observation, or a seed
    # on the first turn. Chosen in ``on_llm_query_enter`` — see the anchor rule there.
    edits_anchor_step: dict[str, Any] | None = None
    # Agent steps by iteration. An agent step is emitted at exit *before* its assistant message is
    # stamped, so its provenance/id is filled in at the next enter (when the stamped assistant
    # appears) — letting a drop that names the assistant's id resolve to its step.
    agent_step_by_iteration: dict[int, dict[str, Any]] = field(default_factory=dict)
    children: list[_InvokeState] = field(default_factory=list)
    parent_repl_iteration: int | None = None
    depth: int | None = None
    # Metadata captured from the invoke events for ATIF extra fields. There is no separate
    # `prompt` field anymore (#538): `task` is an ordinary input, captured in `inputs` below.
    # Explicit `**inputs` kwargs and resolved ambient `jaz.scope`, kept as SEPARATE provenance
    # channels (#727) — emitted as distinct ``extra.inputs`` / ``extra.scope`` blocks.
    # ``inputs`` is snapshotted at InvokeSend (the committed set, post AddInputs/DropInputs),
    # NOT at InvokeEnter: a trace built from Enter mis-records what the child actually
    # received whenever a hook injects an input (the input-side #906 pattern — e.g.
    # ultrahorizon's per-child ``env`` library). ``scope`` stays an Enter snapshot: it is not
    # editable at InvokeEnter, so Enter's copy is already the committed one.
    inputs: dict[str, Any] = field(default_factory=dict)
    scope: dict[str, Any] = field(default_factory=dict)
    # Which of the committed input names hooks added / which caller inputs hooks dropped
    # (from InvokeSend.added_inputs / dropped_inputs) — the provenance split that separates
    # hook-injected inputs from caller-passed ones inside ``extra.inputs``.
    hook_added_inputs: list[str] = field(default_factory=list)
    hook_dropped_inputs: list[str] = field(default_factory=list)
    # Non-completed span records (#892 outcome union): per-query failures/aborts, and the
    # invoke's own outcome when it did not complete. ``outcome`` stays None on the normal
    # (COMPLETED) path so the emitted document is unchanged for successful runs.
    query_failures: list[dict[str, Any]] = field(default_factory=list)
    outcome: str | None = None
    # Per-attempt LLM retry records (LLMQueryRetry) — a call that failed and retried
    # leaves its failed attempts, errors, and backoff in the trace rather than showing
    # one clean call however many attempts it took.
    retries: list[dict[str, Any]] = field(default_factory=list)
    # ``repr()`` of the hooks active for this invoke (baseline ‖ propagating ‖ local), captured
    # at receipt from the live ``event.hooks`` (#727). Strings, not structured dicts: hooks have
    # no serialization contract, so what a hook chooses to show in its ``__repr__`` is the whole
    # record. Rendered here rather than at write time because ``event.hooks`` holds live
    # instances whose state moves on after the event.
    hooks: tuple[str, ...] = ()
    result_type: str | None = None
    result_value: Any = None
    result_exception: str | None = None
    result_exception_type: str | None = None

    def next_step_id(self) -> int:
        self.step_counter += 1
        return self.step_counter


class ATIFTrace(Hook):
    """Write the run as an ATIF trajectory JSON file.

    A pure observer: it emits no effects, so it never influences the run. The file is
    written once, when the hook's scope ends — a single trajectory object when the scope
    held one top-level ``invoke``, an array when it held several (nested invokes appear
    under their parent's ``subagent_trajectories``, not as extra roots). The resulting
    trace is also the input :class:`ATIFReplay` resumes from.

    Under ``with`` the trace covers invokes nested inside. Passed positionally, only that
    invoke is traced and its sub-invokes are absent from the output.

    Example::

        from jaz import invoke
        from jaz.hooks import ATIFTrace

        with ATIFTrace(output_path="./trace.atif.json"):
            result = invoke(task="Do something")

    Args:
        output_path: Where to write the trajectory JSON when the scope ends. ``None``
            (the default) writes no file — read the documents in-memory with
            :meth:`get_trajectories` instead.
        indent: Indentation for the written JSON; ``None`` writes compact single-line
            output.
    """

    # Builds ATIF steps directly as events fire, rather than converting an intermediate
    # per-iteration JSON in a second pass.

    def __init__(
        self,
        output_path: str | Path | None = None,
        *,
        indent: int | None = 2,
    ):
        self.output_path = Path(output_path) if output_path else None
        self.indent = indent
        self._invokes: dict[str, _InvokeState] = {}
        self._root_invokes: list[_InvokeState] = []

    def on_invoke_enter(self, event: InvokeEnter) -> list:
        state = _InvokeState(
            invoke_id=event.invoke_id,
            parent_invoke_id=event.parent_invoke_id,
            parent_repl_iteration=event.parent_repl_iteration,
            depth=event.depth,
            # ``inputs`` is NOT captured here — it is snapshotted at on_invoke_send, from the
            # committed post-add/drop set (see the field comment on _InvokeState.inputs).
            # ``scope`` is not editable at enter, so this copy is already the committed one.
            scope=dict(event.scope),
            # Render the live active-hook set here, at receipt, so the trace pins the governance
            # active AT InvokeEnter (not whatever a live peer mutates to later) (#727).
            hooks=tuple(repr(h) for h in event.hooks),
        )
        self._invokes[event.invoke_id] = state
        if event.parent_invoke_id and event.parent_invoke_id in self._invokes:
            self._invokes[event.parent_invoke_id].children.append(state)
        else:
            self._root_invokes.append(state)
        return []

    def on_invoke_send(self, event: InvokeSend) -> list:
        state = self._invokes.get(event.invoke_id)
        if not state:
            return []
        # The COMMITTED explicit inputs (post AddInputs/DropInputs) — what the prompt renders
        # and the REPL binds, so the trace records what the invoke actually received.
        # SHALLOW snapshot: dict() pins the key set / top-level bindings at receipt, but nested
        # MUTABLE values are shared with the live namespace, so a value the agent mutates during
        # execution is reflected in the trace serialized later (at teardown). This is the same
        # deep/shallow ceiling as event.inputs itself — pinning nested values would need eager
        # deep serialization here, which is costly and can fail on non-copyable inputs, so it
        # is a documented limitation (#831), not a bug.
        state.inputs = dict(event.inputs)
        # The hook-vs-caller provenance split within the committed set (#727's channel
        # discipline, extended to hook-injected inputs): names only — the values already live
        # in ``inputs`` above, so repeating them would just double the payload.
        state.hook_added_inputs = sorted(event.added_inputs)
        state.hook_dropped_inputs = sorted(event.dropped_inputs)
        return []

    def on_llm_query_send(self, event: LLMQuerySend) -> list:
        state = self._invokes.get(event.invoke_id)
        if not state:
            return []
        # Record this turn's compaction edits on the anchor step (the last message emitted this turn;
        # see on_llm_query_enter), not the agent step. The records arrive pre-resolved on the event
        # (LLMQuerySend.edits): each drop is {id, index, persistent} and each added message is
        # {role, content, index} (+ provenance when stamped), under a block-level `iteration` = the
        # query these edits shaped. Per-record iteration is not repeated. A dropped message's content
        # already lives in its own step (#599), so the id is a non-redundant
        # reference. This replaced the former exit-time resolution against a per-hook snapshot
        # of LLMQueryEnter.messages (this hook kept its own enter-buffer copy and reused the
        # effects layer's private index resolver): the dispatcher now resolves once against
        # the same snapshot the buffer fold used, so the trace agrees with the buffer by
        # construction and this hook keeps no index math of its own.
        # ``edits_anchor_step is None`` would drop this turn's edits, but no in-tree path
        # reaches it, so no fallback anchor is warranted — #565 review. Every buffer message
        # is stamped (_agent stamps SEED/ASSISTANT/OBSERVATION and persistent ADDED; transient
        # adds never enter the buffer, and LLMQueryEnter.messages is an enter-time
        # snapshot of the buffer — same membership, shared message dicts), so the
        # ``prov is None`` skip in on_llm_query_enter can't fire; iteration 1 always emits the
        # seed, and every continuing turn appends a fresh OBSERVATION that the next enter
        # emits. That observation cannot be pre-empted by a drop: edits composed at turn i are
        # applied to the buffer BEFORE turn i's assistant and observation are appended, so a
        # drop can never name a message that does not exist yet. Only a custom protocol
        # returning an empty render_observation could get here — which BaseProtocol's "one or
        # more ready messages" contract already forbids.
        edits = event.edits
        if (edits.drops or edits.adds) and state.edits_anchor_step is not None:
            dropped: list[dict[str, Any]] = []
            for record in edits.drops:
                drop_prov = provenance_of(record.message)
                # A drop is "which message" (id) + "where it sat when dropped" (index). No per-record
                # iteration: the turn the drop happens is the block's ``iteration`` (below), and the
                # dropped message's own origin turn lives on its own step (via provenance).
                dropped.append(
                    {
                        "id": drop_prov.id if drop_prov is not None else None,
                        "index": record.position,
                        "persistent": record.persistent,
                    }
                )
            added: list[dict[str, Any]] = []
            for record in edits.adds:
                messages: list[dict[str, Any]] = []
                for m in record.messages:
                    serialized = _serialize_message(
                        m
                    )  # role + content (+ provenance if stamped)
                    # index is ``record.position`` -- the add's input coordinate (the resolved
                    # insert-before slot), not a computed final slot: several adds landing at earlier
                    # positions shift the later ones, so ``position + offset`` would not be the true
                    # post-fold index. A stamped (persistent) add also keeps its ``provenance`` (with
                    # id); a transient add has none. Per-record iteration is not needed -- the whole
                    # block carries one (below).
                    serialized["index"] = record.position
                    messages.append(serialized)
                added.append({"persistent": record.persistent, "messages": messages})
            # ``iteration`` is the query these edits were composed for (``event.iteration``), carried
            # once for the whole block. It is NOT the anchor step's ``extra.iteration``: the edits are
            # composed at enter N but hang on the last message of turn N-1's output, so the step reads
            # N-1 while the edits belong to N -- recording it here is what makes that unambiguous.
            state.edits_anchor_step.setdefault("extra", {})["edits"] = {
                "iteration": event.iteration,
                "dropped": dropped,
                "added": added,
            }
        return []

    def on_llm_query_enter(self, event: LLMQueryEnter) -> list:
        state = self._invokes.get(event.invoke_id)
        if not state:
            return []

        # #481 removed the REPLIteration* events, and each iteration issues exactly one LLMQuery,
        # so LLMQueryEnter is the per-iteration boundary.
        state.current_iteration = event.iteration
        state.total_repl_iterations += 1

        # Emit the messages new this turn, identified by provenance -- not by a buffer diff. Every
        # buffer message carries a stable id (#599), so emit each SEED and OBSERVATION exactly once
        # (keyed by id) and skip ASSISTANT (emitted at LLMQueryExit as the agent step) and ADDED (a
        # compaction summary / hook injection -- recorded in extra.edits, not a conversation step).
        # The step list is thus append-only in spirit -- seed + one observation per turn + agent
        # responses -- so a persistent compaction that shrinks/reorders the buffer never perturbs
        # it; the compaction is recorded separately at LLMQuerySend. (Replaces a prefix/suffix buffer diff,
        # which was an unprincipled heuristic -- #565 review.)
        # Message steps are stamped with the ENTER event's emission time rather than a
        # per-hook clock read: one authoritative clock per event, so every observer of
        # this turn records identical times. Event stamps are aware UTC already.
        now = event.timestamp.isoformat()
        state.edits_anchor_step = None
        # The turn's edits hang on the LAST message emitted this enter -- the simplest rule and the
        # least surprising mental model: the edits are applied to the buffer as it stands right after
        # that last message, so they belong immediately after it. It is also chronologically correct
        # when a turn emits more than one observation (CodeOnlyProtocol appends a truncation-advice
        # observation after the output one): the edits act on the buffer after BOTH, so anchoring on
        # the last renders them after both, whereas the first-observation rule would wrongly slot them
        # between the two. On iteration 1 the last emitted step is a seed, which works the same way.
        last_emitted_step: dict[str, Any] | None = None
        for position, msg in enumerate(event.messages):
            if not isinstance(msg, dict):
                continue
            prov = provenance_of(msg)
            if prov is None:
                continue  # an un-stamped, hand-built message has no identity -- skip it
            if prov.kind is MessageKind.ASSISTANT:
                # The agent step was emitted at the previous exit, before this assistant message
                # was stamped, so fill its provenance/id -- and its index -- in now, so a drop that
                # names this message's id can resolve to its step. (An assistant always carries an
                # int iteration; the guard is for the type-checker.) NOTE the resulting cross-frame
                # pairing: extra.iteration is the assistant's turn N (stamped at exit), but the
                # backfilled index is its slot at THIS enter (N+1) -- the first snapshot it appears
                # in, since the turn-N assistant is appended after turn N's query. `setdefault` pins
                # that first-seen slot against the assistant's re-appearance in later enters.
                if prov.iteration is not None:
                    agent_step = state.agent_step_by_iteration.get(prov.iteration)
                    if agent_step is not None:
                        agent_extra = agent_step.setdefault("extra", {})
                        agent_extra.setdefault("provenance", prov.to_dict())
                        agent_extra.setdefault("index", position)
                continue
            if prov.kind is MessageKind.ADDED:
                continue  # recorded in extra.edits at send, not a conversation step of its own
            if prov.id in state.emitted_ids:
                continue
            # ``index`` is the message's slot in this enter snapshot, captured the first time it is
            # seen (right after it entered the buffer). A live buffer position, so it drifts as later
            # turns add/drop around it; provenance's id + iteration are the stable identity.
            #
            # ``iteration`` is the message's own provenance iteration -- the turn it was PRODUCED, not
            # the enter it is first shown. These differ for an OBSERVATION (produced at turn N-1, first
            # emitted at enter N), so stamping ``current_iteration`` here would tag it N, off by one;
            # ``prov.iteration`` keeps it N-1, matching how the assistant step (stamped at its own
            # exit) already agrees with its provenance. It is ``None`` for a seed, which belongs to no
            # REPL iteration (it exists before the loop) -- an honest absence, not a turn number.
            extra: dict[str, Any] = {
                "provenance": prov.to_dict(),
                "index": position,
                "iteration": prov.iteration,
            }
            step_data: dict[str, Any] = {
                "step_id": state.next_step_id(),
                "source": msg.get("role", "user"),
                "message": msg.get("content", ""),
                "timestamp": now,
                "llm_call_count": 0,
                "extra": extra,
            }
            state.steps.append(step_data)
            state.emitted_ids.add(prov.id)
            last_emitted_step = step_data
        state.edits_anchor_step = last_emitted_step
        return []

    def on_llm_query_retry(self, event: LLMQueryRetry) -> list:
        state = self._invokes.get(event.invoke_id)
        if state is not None:
            state.retries.append(
                {
                    "iteration": event.iteration,
                    "attempt_number": event.attempt_number,
                    "error_type": type(event.exception).__name__,
                    # summarize_exception, not str(): the group-expanding rendering used
                    # on every failure surface of this hook (see on_llm_query_exit).
                    "error": summarize_exception(event.exception),
                    "wait_seconds": event.wait_seconds,
                }
            )
        return []

    def on_llm_query_exit(self, event: LLMQueryExit) -> list:
        outcome = event.outcome
        # Non-completed arms (#892 outcome union): there is no response, no billed usage, and
        # no agent step to write — record the outcome and its exception instead, so a trace
        # whose turn vanished shows *why* (a failed provider call, an aborted turn) rather
        # than silently ending.
        if not isinstance(outcome, Completed):
            failed_state = self._invokes.get(event.invoke_id)
            if failed_state is not None:
                failed_state.query_failures.append(
                    {
                        "iteration": event.iteration,
                        # The variant IS the status (no parallel enum): its class name,
                        # lowercased ("aborted" / "failed"), is the serialized form.
                        "outcome": type(outcome).__name__.lower(),
                        "exception_type": type(outcome.exception).__name__,
                        # summarize_exception, not str(): multiple aborts at one boundary
                        # group into an ExceptionGroup by law, and str() on a group
                        # renders only a sub-exception count, dropping every child's
                        # text — on exactly the failure records this field exists for.
                        "exception": summarize_exception(outcome.exception),
                    }
                )
            return []
        response = outcome.result
        response_content = response.content
        state = self._invokes.get(event.invoke_id)
        if not state:
            return []
        # Empty/None content is NOT a reason to skip the turn. This point is only reached on
        # the Completed arm (a failed call took the arm above), so empty content means a
        # real, billed query that returned nothing usable -- a content filter, a reasoning-only
        # response, max_tokens spent on reasoning. Skipping it used to lose the agent step, this
        # turn's edits, AND the call's tokens/cost from the trajectory totals, while
        # ConversationHistory records the response unconditionally -- the one gap that falsified
        # "ATIF records everything ConversationHistory does" on the axis (cost) the rest of this
        # stack means to drive off ATIF (#565 review).

        if event.model:
            state.model_name = event.model

        prompt_tokens = response.prompt_tokens or 0
        completion_tokens = response.completion_tokens or 0
        # Keep the raw value (None = provider reported no caching) distinct from
        # the summing value (None coerced to 0).
        raw_cached_tokens = response.cached_tokens
        cached_tokens = raw_cached_tokens or 0
        cost = response.cost_usd or 0.0

        state.total_prompt_tokens += prompt_tokens
        state.total_completion_tokens += completion_tokens
        state.total_cached_tokens += cached_tokens
        state.total_cost += cost

        # The agent step's timestamp is the exit's emission time (Event.timestamp —
        # every event carries one; no interval fields exist), aware UTC already.
        ts = event.timestamp.isoformat()

        metrics: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost,
        }
        # Write an explicit 0 when the model reports caching but had no hit (raw
        # == 0); omit only when caching is unreported (raw is None).
        if raw_cached_tokens is not None:
            metrics["cached_tokens"] = raw_cached_tokens
        remaining_extra = dict(response.extra)
        hoist_atif_metrics(metrics, remaining_extra)
        if remaining_extra:
            metrics["extra"] = remaining_extra
        step: dict[str, Any] = {
            "step_id": state.next_step_id(),
            "source": "agent",
            # Coerced: content is ``str | None``, and an ATIF step's message is a string.
            "message": response_content or "",
            "timestamp": ts,
            "llm_call_count": 1,
            "metrics": metrics,
        }
        if state.current_iteration is not None:
            step["extra"] = {"iteration": state.current_iteration}
            # The assistant message this step reports is stamped only after this exit, so its
            # provenance/id is filled in at the next enter (see on_llm_query_enter). Track the
            # step by iteration so enter can find it.
            # Consequence: the FINAL iteration's agent step keeps no provenance, since no enter
            # follows it. Accepted, not overlooked -- the final assistant is never dropped, so no
            # drop ever has to resolve its id, and ConversationHistory records no assistant ids
            # either. Filling it at on_invoke_exit is not an option: InvokeExit carries only the
            # result, never the message buffer, so the stamped assistant is unreachable there.
            state.agent_step_by_iteration[state.current_iteration] = step
        state.steps.append(step)
        return []

    def on_invoke_exit(self, event: InvokeExit) -> list:
        outcome = event.outcome
        # Non-completed arms (#892 outcome union) carry no terminal result — record the
        # outcome and its exception instead; the per-invoke state is kept, so teardown still
        # writes everything traced up to the failure, now labelled with how it ended.
        if not isinstance(outcome, Completed):
            state = self._invokes.get(event.invoke_id)
            if state is not None:
                state.outcome = type(outcome).__name__.lower()
                # Group-expanding rendering, as on every failure field (see
                # on_llm_query_exit's non-completed arm).
                state.result_exception = summarize_exception(outcome.exception)
                state.result_exception_type = type(outcome.exception).__name__
            return []
        state = self._invokes.get(event.invoke_id)
        result = outcome.result
        if state:
            state.result_type = type(result).__name__
            # InvokeExit yields either Return (terminal RETURN) or Raise (terminal RAISE) — both
            # terminal, so neither carries ``output`` (#903); capture the value / exception instead.
            match result:
                case Return(return_value=rv) if rv is not None:
                    # Always uses repr (truncated) for safety/bounded size.
                    state.result_value = {
                        "type": type(rv).__name__,
                        "repr_prefix_10000": repr(rv)[:10000],
                    }
                case Raise(exception=exc):
                    # Group-expanding rendering, matching the failure fields above.
                    state.result_exception = summarize_exception(exc)
                    state.result_exception_type = type(exc).__name__
        return []

    def _build_atif(self, state: _InvokeState) -> dict[str, Any]:
        """Build an ATIF document from an invoke state."""
        atif: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "agent": {
                "name": "jaz",
                "version": __import__("jaz").__version__,
            },
            "steps": state.steps,
            "final_metrics": {
                "total_prompt_tokens": state.total_prompt_tokens,
                "total_completion_tokens": state.total_completion_tokens,
                "total_cached_tokens": state.total_cached_tokens,
                "total_cost_usd": round(state.total_cost, 6),
                "total_steps": state.total_repl_iterations,
            },
            "notes": (
                "total_steps is the number of REPL iterations (think-act-observe cycles), "
                "not len(steps). Each REPL iteration corresponds to one LLM call."
            ),
        }

        if state.model_name:
            atif["agent"]["model_name"] = state.model_name
        if state.invoke_id:
            atif["trajectory_id"] = state.invoke_id

        # Extra metadata not in core ATIF schema but needed for full coverage. The former
        # `extra["prompt"]` is gone (#538); the prompt lives in `extra["inputs"]` now.
        extra: dict[str, Any] = {}
        if state.depth is not None:
            extra["depth"] = state.depth
        if state.parent_repl_iteration is not None:
            # Which REPL iteration of the parent invoke spawned this subagent.
            extra["parent_repl_iteration"] = state.parent_repl_iteration
        if state.inputs:
            # Serialize inputs: try JSON-serializable value, fall back to repr
            # TODO: revisit — should we use the serialization shown to the agent?
            extra["inputs"] = _serialize_namespace(state.inputs)
        if state.hook_added_inputs:
            # Names of ``inputs`` entries injected by hooks (values live in ``inputs``).
            extra["hook_added_inputs"] = list(state.hook_added_inputs)
        if state.hook_dropped_inputs:
            # Caller inputs hooks un-passed — absent from ``inputs`` by design, not omission.
            extra["hook_dropped_inputs"] = list(state.hook_dropped_inputs)
        if state.scope:
            # Resolved ambient jaz.scope, logged as its own provenance channel (#727).
            extra["scope"] = _serialize_namespace(state.scope)
        if state.hooks:
            # Governance active for this invoke: the resolved active hooks' reprs (#727).
            extra["hooks"] = list(state.hooks)
        if state.outcome is not None:
            # Only set on the non-COMPLETED invoke arms (#892), so completed runs keep the
            # pre-outcome-union document shape unchanged.
            extra["outcome"] = state.outcome
        if state.query_failures:
            # Per-query non-completed closes (outcome + exception), keyed by iteration —
            # the turns whose agent step is absent above, with the reason why.
            extra["query_failures"] = list(state.query_failures)
        if state.retries:
            # Per-attempt LLM retry records, keyed by iteration — absent entirely on a
            # retry-free run, so the common document shape is unchanged.
            extra["retries"] = list(state.retries)
        if state.result_type:
            extra["result_type"] = state.result_type
        if state.result_value is not None:
            extra["result_value"] = state.result_value
        if state.result_exception is not None:
            extra["result_exception"] = state.result_exception
        if state.result_exception_type is not None:
            extra["result_exception_type"] = state.result_exception_type
        if extra:
            atif["extra"] = extra

        # Nested invokes: store in top-level subagent_trajectories[]
        if state.children:
            atif["subagent_trajectories"] = [
                self._build_atif(child) for child in state.children
            ]

        return atif

    def get_trajectories(self) -> list[dict[str, Any]]:
        """Return the ATIF trajectory documents traced so far, one per top-level invoke.

        Built fresh from the recorded state on each call, so it can be read mid-run or
        after the scope ends; nested invokes appear inside their parent's document.
        """
        return [self._build_atif(state) for state in self._root_invokes]

    def _write_output(self) -> None:
        """Write ATIF JSON to the output file."""
        if not self.output_path:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        trajectories = self.get_trajectories()
        # Single root → write as object; multiple → write as array
        data = trajectories[0] if len(trajectories) == 1 else trajectories
        self.output_path.write_text(json.dumps(data, indent=self.indent))

    def teardown(self, exc: BaseException | None = None) -> None:
        """Write the output file when the hook's scope ends."""
        # teardown, NOT __exit__: it runs on BOTH activation paths — `with` (via the base
        # __exit__) and positional/local_hooks (called directly by _activate_local_hooks,
        # which never touches __exit__). The write used to live in a __exit__ override,
        # so a positionally-passed ATIFTrace(output_path=...) silently never wrote its
        # file even though the docstring documents positional use.
        self._write_output()

    def reset(self) -> None:
        """Clear state for reuse."""
        self._invokes.clear()
        self._root_invokes.clear()


#: Deprecated alias for the pre-rename spelling — see the rationale block in
#: ``jaz/hooks/__init__.py``. Every renamed hook carries this alias at its definition
#: site so the deep-path import keeps working and so the alias map stays checkable
#: (``test_legacy_hook_names_still_importable``). Deep-path only: the old spelling predates
#: this hook's promotion to ``jaz.hooks.__all__``, so it was never importable from a
#: package namespace and gets no ``_DEMOTED`` entry there.
ATIFTraceHook = ATIFTrace
