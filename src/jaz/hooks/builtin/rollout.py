"""Record agent runs as training-grade rollouts: exact token ids, loss mask, logprobs.

:class:`RolloutRecorder` is an ordinary hook. Under a token-native backend (e.g.
``SGLangLLM``) every LLM response carries a frozen ``TurnRecord`` — the per-message token
stamps that were on the wire plus the newly sampled segment — and the recorder simply
accumulates them per invoke, freezing each invoke's log into a :class:`Rollout` at exit::

    with RolloutRecorder() as rec, jaz.ConfigOverride(llm=sglang):
        result = invoke(task="...")
    rollout = rec.rollouts[0]
    sample = rollout.to_flat()                      # flat ids + loss mask + logprobs
    write_jsonl({**sample.to_dict(), "reward": judge(result)})   # reward is driver-side

It propagates to nested invokes like any ``with``-activated hook; concurrent invokes
demultiplex by ``invoke_id``. An invoke served by a backend that produces no turn records
(a text backend, or a replay override) yields no rollout — the recorder warns once and
stands down for that invoke.

Export offers three projections, strictest to most general: :meth:`Rollout.to_flat` (one
flat masked sequence — raises :class:`~jaz.exceptions.NonMonotoneRolloutError` if the
context was ever edited mid-run), :meth:`Rollout.to_pieces` (flat samples per monotone
span), and :meth:`Rollout.to_turn_samples` (per-turn conditioning/sampled/logprobs
triples — always resolves). For reasoning models under ``SGLangLLM``'s hidden-trace
interface, multi-turn rollouts are non-monotone by design (each turn's trace is stripped
from the next turn's context), so ``to_turn_samples()`` is the export to reach for — the
quickstart's ``to_flat()`` fits single-turn runs and non-reasoning models.
"""

# Design rationale (maintainers; token-native rollouts design "Option C", PR #1099;
# executive call, user, 2026-08-13. The doc did not merge — jaz.tokens' module comment
# holds the option comparison; recorder-specific calls are recorded here):
#
# * The recorder is an ACCUMULATOR, NOT A VERIFIER — and export does not verify either.
#   The record is the GROUND TRUTH of what the model saw and produced: token truth lives
#   in the frozen stamps the messages own, the TurnRecord references only frozen data, and
#   nothing is ever re-derived — so nothing can drift. Fidelity is STRUCTURAL: the backend
#   mints each stamp from exactly the ids it sends (jaz.llm.sglang._prepare_stamps), and
#   the stale-stamp guard re-renders the instant a message's content stops matching its
#   stamp — so "recorded ids != passed ids" is not a reachable state, for any role.
#   (Executive call, user, 2026-08-14.) An earlier draft added an export-time "trainability
#   check" (guard 3) that refused any rollout containing an assistant-role stamp with
#   sampled_span=None, on the theory that such a stamp was re-tokenized text. It was
#   removed: sampled_span=None on an assistant stamp is NOT a corruption signal — it is the
#   normal render of an assistant message the policy did not sample (a few-shot exemplar in
#   the seed prompt, a hook-injected assistant example). That is faithful conditioning the
#   model genuinely saw, and it already carries loss mask 0, so training never scores it;
#   the guard conflated "not policy-sampled" with "corrupted" and could not structurally
#   tell the two apart. Loss can only ever land (mask 1) on a sampled_span, whose ids ARE
#   the verbatim sampled output, so drift cannot reach a training target regardless. A
#   genuine mid-run context edit is still represented — by the monotonicity split, not a
#   refusal.
# * The flat masked sequence is a projection computed at export — it does not exist
#   during the run. Monotonicity is decided by comparing stamp IDENTITIES (`is`), never
#   token or text values: the reference lists ARE the record of what was sent, and two
#   same-content renders (a stale-stamp re-render after a hook edit) must not chain.
# * Rollout deliberately has NO `reward` field (executive call, user, 2026-08-13). The
#   record holds facts about the trajectory; a reward is a judgment about it, and its
#   shape is trainer-specific (scalar for terminal-reward GRPO, per-turn for PPO,
#   vectors for multi-objective). Any shape here would privilege one algorithm. Rewards
#   live driver-side keyed by invoke_id (+ TurnRecord.iteration for per-turn); the
#   dividing line for future fields: facts about what happened (finished="aborted:...")
#   go on the record; evaluations of it stay with the driver.
# * Partial rollouts ARE emitted, with finished="aborted:<reason>", when an invoke
#   aborts (budget, exception) — chosen over dropping them: partial trajectories are
#   useful to some algorithms and trivially filterable, and the `finished` marker is a
#   fact, so it belongs on the record.

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from jaz.exceptions import NonMonotoneRolloutError
from jaz.hooks.dispatcher import Hook
from jaz.hooks.effects import Effect
from jaz.hooks.events import Completed, InvokeEnter, InvokeExit, LLMQueryExit
from jaz.repl.types import Raise
from jaz.tokens import TurnRecord

logger = logging.getLogger(__name__)


def _extends(prev: TurnRecord, cur: TurnRecord) -> bool:
    """Did ``cur``'s context extend ``prev``'s (same sent prefix + its sample)?"""
    # Compare by TokenStamp.message_id — the per-render UUID that is the stamp's identity key.
    # A reused message keeps its stamp (and id); a stale-stamp re-render mints a fresh id
    # (SGLangLLM._prepare_stamps), so equal ids mean "the same render was sent", which is
    # exactly the monotonicity question. In-process this equals object identity (`is`), but
    # unlike `is` it survives a TurnRecord being serialized and reloaded — needed the moment a
    # flat sample is built offline from persisted turns, where the original objects are gone.
    # (Full dataclass `==` would also discriminate here, since message_id participates in
    # equality — but it needlessly compares the whole ids tuple; the id alone is the intent.)
    expected = (*prev.sent, prev.sampled)
    # strict=False: cur.sent being LONGER is the normal case (new trailing messages).
    return len(cur.sent) >= len(expected) and all(
        a.message_id == b.message_id for a, b in zip(expected, cur.sent, strict=False)
    )


@dataclass(frozen=True)
class FlatSample:
    """One flat masked token sequence — the classic RL/SFT training sample.

    ``loss_mask[i] == 1`` exactly where ``ids[i]`` was sampled by the policy;
    ``logprobs[i]`` is the sampling logprob there and ``0.0`` elsewhere.
    """

    ids: tuple[int, ...]
    loss_mask: tuple[int, ...]
    logprobs: tuple[float, ...]
    invoke_id: str
    model: str
    finished: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible projection (ids/mask/logprobs materialized as lists)."""
        return {
            "ids": list(self.ids),
            "loss_mask": list(self.loss_mask),
            "logprobs": list(self.logprobs),
            "invoke_id": self.invoke_id,
            "model": self.model,
            "finished": self.finished,
        }


@dataclass(frozen=True)
class TurnSample:
    """One turn as a (conditioning, sampled, logprobs) triple — the fully general form.

    ``prefix_ids`` is everything the model conditioned on for this turn (context +
    generation prompt); ``sampled_ids`` are the verbatim sampled ids (loss on all of
    them); ``logprobs`` aligns with ``sampled_ids``. No cross-turn prefix sharing.
    """

    prefix_ids: tuple[int, ...]
    sampled_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    iteration: int
    model: str
    weight_version: str | None
    finish_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible projection."""
        return {
            "prefix_ids": list(self.prefix_ids),
            "sampled_ids": list(self.sampled_ids),
            "logprobs": list(self.logprobs),
            "iteration": self.iteration,
            "model": self.model,
            "weight_version": self.weight_version,
            "finish_reason": self.finish_reason,
        }


@dataclass(frozen=True)
class Rollout:
    """One invoke's frozen trajectory record: its turns plus terminal facts.

    ``finished`` is ``"completed"`` for a normal return, or ``"aborted:<ExceptionType>"``
    (e.g. ``"aborted:IterationLimitExhaustedError"``) for an aborted invoke — the marker
    carries the concrete exception class name. Partial rollouts are emitted, not dropped,
    and are trivially filterable on this marker.
    """

    # Known limitation (#906, inherited from InvokeExit): `finished` reflects the
    # PRE-transform terminal result; an InvokeExit-time ModifyExecResult downgrade is not
    # visible here.

    invoke_id: str
    depth: int
    turns: tuple[TurnRecord, ...]
    finished: str
    model: str
    # The invoke's boundary times: InvokeEnter/InvokeExit EMISSION stamps
    # (Event.timestamp — aware UTC, like every event stamp; not provider-call
    # boundaries). None when the recorder never saw the matching event.
    started_at: datetime | None
    ended_at: datetime | None

    def to_flat(self) -> FlatSample:
        """The whole trajectory as one flat masked sequence.

        Raises :class:`~jaz.exceptions.NonMonotoneRolloutError` if any turn's context was
        not a pure extension of the previous one — routine for reasoning models under
        ``SGLangLLM``'s hidden-trace interface (multi-turn rollouts are non-monotone by
        design there), or after a transient nudge, compaction, or edit — use
        :meth:`to_pieces` or :meth:`to_turn_samples` then.
        """
        for prev, cur in zip(self.turns, self.turns[1:], strict=False):
            if not _extends(prev, cur):
                raise NonMonotoneRolloutError(
                    "this trajectory's context differed between turns (hidden reasoning "
                    "traces stripped from later context, a transient nudge, or "
                    "compaction); no single flat sequence represents it — use "
                    "to_pieces() or to_turn_samples()"
                )
        return self._flatten(self.turns)

    def to_pieces(self) -> list[FlatSample]:
        """Flat samples per monotone span — split at each context break.

        A pre-break sample appearing in a later piece's *context* has loss mask 0 there:
        it was sampled against its own piece's prefix and gets its loss there.
        """
        pieces: list[FlatSample] = []
        start = 0
        for i in range(1, len(self.turns)):
            if not _extends(self.turns[i - 1], self.turns[i]):
                pieces.append(self._flatten(self.turns[start:i]))
                start = i
        if self.turns:
            pieces.append(self._flatten(self.turns[start:]))
        return pieces

    def to_turn_samples(self) -> list[TurnSample]:
        """Per-turn (conditioning, sampled, logprobs) triples. Always resolves.

        The unit a trainer's forward pass consumes anyway, just without cross-turn
        prefix sharing — inherent where the context genuinely differed between turns.
        """
        samples: list[TurnSample] = []
        for turn in self.turns:
            span = turn.sampled.sampled_span
            if span is None:
                # Defensive: a sampled stamp with no verbatim span (a hand-built record;
                # the backend always sets one on what it samples) yields empty
                # sampled_ids — faithful conditioning, nothing to train on this turn.
                span = (len(turn.sampled.ids), len(turn.sampled.ids))
            prefix: list[int] = list(turn.initial)
            for stamp in turn.sent:
                prefix.extend(stamp.ids)
            prefix.extend(turn.sampled.ids[: span[0]])  # the generation-prompt framing
            samples.append(
                TurnSample(
                    prefix_ids=tuple(prefix),
                    sampled_ids=tuple(turn.sampled.ids[span[0] : span[1]]),
                    logprobs=turn.sampled.logprobs or (),
                    iteration=turn.iteration,
                    model=turn.model,
                    weight_version=turn.sampled.weight_version,
                    finish_reason=turn.sampled.finish_reason,
                )
            )
        return samples

    def _flatten(self, turns: tuple[TurnRecord, ...]) -> FlatSample:
        """Flatten a monotone span: the final turn's context + sample, masked."""
        last = turns[-1]
        # Loss mask 1 only for segments sampled by THESE turns. In a whole-rollout
        # flatten that is every assistant sample; in a post-break piece, a sample from
        # before the break can appear in the context (sent) — it was sampled against a
        # *different* prefix (that is what the break means), so masking it here would
        # score it against tokens the model never saw. It gets loss in its own piece.
        # Keyed by message_id, not id(): same in-process semantics as object identity but
        # serialize/reload-safe, and it drops the id()-as-key caveat (that would be sound only
        # because self.turns keeps every stamp alive for the whole call). See _extends.
        own_samples = {t.sampled.message_id for t in turns}
        # The final turn's context contains every earlier turn's tokens (monotone), so
        # the sequence is just its sent stamps + its sample, with the preamble in front.
        ids: list[int] = list(last.initial)
        mask: list[int] = [0] * len(last.initial)
        logprobs: list[float] = [0.0] * len(last.initial)
        for stamp in (*last.sent, last.sampled):
            offset = len(ids)
            ids.extend(stamp.ids)
            mask.extend([0] * len(stamp.ids))
            logprobs.extend([0.0] * len(stamp.ids))
            if stamp.sampled_span is not None and stamp.message_id in own_samples:
                start, end = stamp.sampled_span
                for i, lp in zip(
                    range(offset + start, offset + end),
                    stamp.logprobs or (),
                    strict=True,  # span/logprob alignment is validated at construction
                ):
                    mask[i] = 1
                    logprobs[i] = lp
        return FlatSample(
            ids=tuple(ids),
            loss_mask=tuple(mask),
            logprobs=tuple(logprobs),
            invoke_id=self.invoke_id,
            model=self.model,
            finished=self.finished,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible record: terminal facts + per-turn samples (general form)."""
        return {
            "invoke_id": self.invoke_id,
            "depth": self.depth,
            "finished": self.finished,
            "model": self.model,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "turns": [s.to_dict() for s in self.to_turn_samples()],
        }


class RolloutRecorder(Hook):
    """Accumulate each invoke's ``TurnRecord``\\ s and freeze them into ``rollouts``.

    Use as a context manager (``with RolloutRecorder() as rec:``); after the block,
    ``rec.rollouts`` holds one :class:`Rollout` per recorded invoke, in completion
    order. Invokes whose backend produced no turn records (text backends, replay
    overrides) yield no rollout — a warning is logged once per invoke.
    """

    def __init__(self) -> None:
        self.rollouts: list[Rollout] = []
        self._log: dict[str, list[TurnRecord]] = {}
        self._started: dict[str, datetime] = {}
        self._stood_down: set[str] = set()
        # Concurrent invokes under one `with` demultiplex by invoke_id; the lock guards
        # the shared dicts (per-invoke ordering is free — each invoke's REPL loop is
        # sequential, so its LLMQueryExits arrive in turn order).
        self._lock = threading.RLock()

    def setup(self) -> None:
        # Fresh scope per activation, like BudgetPool: re-entering the hook must not
        # leak rollouts (or stand-downs) from a previous `with` block.
        self.rollouts = []
        self._log = {}
        self._started = {}
        self._stood_down = set()

    def on_invoke_enter(self, event: InvokeEnter) -> list[Effect]:
        with self._lock:
            # Event-stamped, not a hook-side clock read: Event.timestamp is the
            # emission time, one authoritative clock shared by every observer.
            self._started[event.invoke_id] = event.timestamp
        return []

    def on_llm_query_exit(self, event: LLMQueryExit) -> list[Effect]:
        outcome = event.outcome
        # LLMQueryExit fires on every path now that it is universal-close (#1149): the outcome
        # union is Completed|Aborted|Failed. Only a Completed query carries an LLMResponse; an
        # Aborted/Failed query has no token truth to record, so skip it here and let InvokeExit
        # freeze whatever turns already accumulated (the finished="aborted:..." partial rollout)
        # — guarding on the outcome, NOT on record's presence, is what keeps a failed turn from
        # standing the invoke down and discarding that partial.
        if not isinstance(outcome, Completed):
            return []
        record = outcome.result.tokens
        with self._lock:
            if event.invoke_id in self._stood_down:
                return []
            if record is None:
                # Token-less *completed* query (text backend or replay override): no token truth
                # exists for this invoke, and approximating with a local tokenizer would be silent
                # retokenization drift. Warn once, drop any partial log, stand down.
                logger.warning(
                    "RolloutRecorder: invoke %s produced a response without token "
                    "records (model %r — text backend or replay override); no rollout "
                    "will be recorded for it.",
                    event.invoke_id,
                    event.model,
                )
                self._stood_down.add(event.invoke_id)
                self._log.pop(event.invoke_id, None)
                return []
            # The backend cannot know the agent's iteration (complete() sees only
            # model/messages), so it records the -1 sentinel; this event is the
            # authoritative source. replace() shares the stamp references, so identity
            # chaining in _extends is unaffected.
            self._log.setdefault(event.invoke_id, []).append(
                replace(record, iteration=event.iteration)
            )
        return []

    def on_invoke_exit(self, event: InvokeExit) -> list[Effect]:
        with self._lock:
            started = self._started.pop(event.invoke_id, None)
            stood_down = event.invoke_id in self._stood_down
            self._stood_down.discard(event.invoke_id)
            turns = self._log.pop(event.invoke_id, [])
            if stood_down or not turns:
                return []
            # finished is a fact on the record, filterable trainer-side. Derived from the invoke's
            # span outcome union (#1149-#1151): a normal Return is "completed"; a terminal agent
            # Raise (Completed carrying Raise), an Aborted (this invoke's own hook Abort — budget /
            # iteration hard-stop), and a Failed (any other escaping error) all collapse to the
            # "aborted:<ExceptionType>" stem. The pipeline KEPT that "abort" stem (the design doc's
            # proposed Abort->CancelInvoke rename was dropped by user decision — see
            # jaz.hooks.effects.Abort), so a trainer-side filter on the prefix stays valid.
            outcome = event.outcome
            if isinstance(outcome, Completed):
                result = outcome.result  # Return | Raise
                finished = (
                    f"aborted:{type(result.exception).__name__}"
                    if isinstance(result, Raise)
                    else "completed"
                )
            else:  # Aborted | Failed — both carry .exception
                finished = f"aborted:{type(outcome.exception).__name__}"
            self.rollouts.append(
                Rollout(
                    invoke_id=event.invoke_id,
                    depth=event.depth,
                    turns=tuple(turns),
                    finished=finished,
                    model=turns[-1].model,
                    started_at=started,
                    # Event-stamped like started_at (see on_invoke_enter).
                    ended_at=event.timestamp,
                )
            )
        return []
