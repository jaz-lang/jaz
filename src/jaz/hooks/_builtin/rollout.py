"""Record agent runs as training-grade rollouts: exact token ids, loss mask, logprobs.

:class:`RolloutRecorder` is an ordinary hook. Under a token-native backend (e.g.
``SGLangLLM``) every LLM response carries a frozen ``TurnRecord`` — the per-message token
stamps that were on the wire plus the newly sampled segment — and the recorder simply
accumulates them per invoke, freezing each invoke's log into a :class:`Rollout` at exit::

    with RolloutRecorder() as rec, jaz.ConfigOverride(llm=sglang):
        result = invoke(task="...")
    rollout = rec.rollouts[0]
    for sample in rollout.to_spans():               # flat ids + loss mask + logprobs
        write_jsonl({**sample.to_dict(), "reward": judge(result)})   # reward is driver-side

It propagates to nested invokes like any ``with``-activated hook; concurrent invokes
demultiplex by ``invoke_id``. An invoke served by a backend that produces no turn records
(a text backend, or a replay override) yields no rollout — the recorder warns once and
stands down for that invoke.

Export is one total projection, :meth:`Rollout.to_spans`: a list of :class:`Sample`, one
per maximal prefix-monotone span of the trajectory — exactly one when the context only ever
grew, and one per break otherwise. Each ``Sample`` is a flat token sequence with a loss
mask, full-sequence logprobs, the prompt/response boundary (``prefix_len``), and the
provenance a batched consumer needs. For reasoning models under ``SGLangLLM``'s
hidden-trace interface, multi-turn rollouts break at every turn by design (each turn's trace
is stripped from the next turn's context), so ``to_spans()`` yields one sample per turn
there; nothing about the call changes.
"""

# Design rationale (maintainers; token-native rollouts design "Option C", PR #1099;
# executive call, user, 2026-08-13. The doc did not merge — jaz._tokens' module comment
# holds the option comparison; recorder-specific calls are recorded here):
#
# * The recorder is an ACCUMULATOR, NOT A VERIFIER — and export does not verify either.
#   The record is the GROUND TRUTH of what the model saw and produced: token truth lives
#   in the frozen stamps the messages own, the TurnRecord references only frozen data, and
#   nothing is ever re-derived — so nothing can drift. Fidelity is STRUCTURAL: the backend
#   mints each stamp from exactly the ids it sends (jaz.llm._sglang._prepare_stamps), and
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
# * ONE projection, `to_spans()`, and a mask-encoded `Sample` that carries its own
#   prompt/response boundary (design doc: #1459, `design/design_features/rollout_export_api.md`
#   on that branch — design-doc PRs are archived unmerged; executive call, user, 2026-09-15). The export unit is the ecosystem convention — a trajectory maps to
#   1..N flat masked sequences, split at each context break, never an exception the caller
#   catches to pick a projection: tinker-cookbook `trajectory_to_data`
#   (https://github.com/thinking-machines-lab/tinker-cookbook), SkyRL `merge_stepwise_output`
#   (https://github.com/NovaSky-AI/SkyRL), rLLM `_process_trajectory`
#   (https://github.com/rllm-org/rllm), Agent-Lightning `RolloutAdapter`
#   (https://github.com/microsoft/agent-lightning), ROLL `MessageTracker`
#   (https://github.com/alibaba/ROLL) and ART `_Branch` merging (https://github.com/OpenPipe/ART)
#   all implement that same greedy flush-and-split. An earlier shape shipped three projections in two encodings
#   (`to_flat`/`to_pieces`/`to_turn_samples`, `FlatSample`/`TurnSample`), one of which raised
#   `NonMonotoneRolloutError` to tell the caller to use another — on every multi-turn
#   reasoning rollout, since those are non-monotone by design. The split encoding went
#   because every trainer stores flat+mask (verl derives `prompt_ids`/`response_ids` by
#   length arithmetic from one growing list); `prefix_len` stayed because every split-camp
#   trainer still needs the boundary as an integer, and slime validates against it.
# * Rollout deliberately has NO `reward` field (executive call, user, 2026-08-13). The
#   record holds facts about the trajectory; a reward is a judgment about it, and its
#   shape is trainer-specific (scalar for terminal-reward GRPO, per-turn for PPO,
#   vectors for multi-objective). Any shape here would privilege one algorithm. Rewards
#   live driver-side keyed by invoke_id (+ the turn's index, for per-turn); the
#   dividing line for future fields: facts about what happened (finished="aborted:...")
#   go on the record; evaluations of it stay with the driver.
# * Partial rollouts ARE emitted, with finished="aborted:<reason>", when an invoke
#   aborts (budget, exception) — chosen over dropping them: partial trajectories are
#   useful to some algorithms and trivially filterable, and the `finished` marker is a
#   fact, so it belongs on the record.

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from jaz._tokens import TurnRecord
from jaz.hooks._dispatcher import Hook
from jaz.hooks.effects import Effect
from jaz.hooks.events import Completed, InvokeExit, LLMQueryExit
from jaz.repl._types import Raise

logger = logging.getLogger(__name__)


def _extends(prev: TurnRecord, cur: TurnRecord) -> bool:
    """Did ``cur``'s context extend ``prev``'s (same sent prefix + its sample)?"""
    # Compare by TokenStamp.message_id — the per-render UUID that is the stamp's identity key.
    # A reused message keeps its stamp (and id); a stale-stamp re-render mints a fresh id
    # (SGLangLLM._prepare_stamps), so equal ids mean "the same render was sent", which is
    # exactly the monotonicity question. In-process this equals object identity (`is`), but
    # unlike `is` it survives a TurnRecord being serialized and reloaded. (Full dataclass `==`
    # would also discriminate here, since message_id participates in equality — but it
    # needlessly compares the whole ids tuple; the id alone is the intent.)
    #
    # This is NOT the token-id prefix test the surveyed trainers run (tinker `_is_prefix`,
    # SkyRL `_is_prefix`, verifiers' longest-common-token-prefix fork). Identity is stricter:
    # a hook edit whose re-render happens to produce the same ids (the template normalized the
    # change away) splits here and would merge there. Same outcome for the case that dominates
    # — a stripped `<think>` block changes the ids — and a deliberate choice for that rare
    # same-ids re-render, because a history edit
    # is a real event in the trajectory and the identity rule is what keeps it representable.
    expected = (*prev.sent, prev.sampled)
    # strict=False: cur.sent being LONGER is the normal case (new trailing messages).
    return len(cur.sent) >= len(expected) and all(
        a.message_id == b.message_id for a, b in zip(expected, cur.sent, strict=False)
    )


@dataclass(frozen=True)
class Sample:
    """One flat masked token sequence — a training sample, plus what a batched consumer needs.

    Attributes:
        ids: The full token sequence.
        loss_mask: ``1`` exactly where ``ids[i]`` was sampled by the policy, else ``0``.
        logprobs: The sampling logprob where ``loss_mask[i] == 1``, ``0.0`` elsewhere, aligned
            to the full ``ids``.
        prefix_len: The prompt/response boundary: ``ids[:prefix_len]`` is everything the model
            conditioned on before the first sampled token, so ``len(ids) - prefix_len`` is a
            trainer's response length and ``logprobs[prefix_len:]`` its response-only alignment.
        invoke_id: As on :class:`Rollout`.
        span_index: This row's position among the rollout's spans; ``(invoke_id, span_index)``
            identifies it once samples are batched across rollouts.
    """

    # Fields are the payload plus the provenance the ecosystem carries per row (survey in
    # #1459): a rollout/group id (9 of 13 trainers) and a row index for exporters that emit
    # N>1 rows per trajectory (all of them). Dropped from the previous `FlatSample`: `model`
    # and `finished` — invoke-level facts a consumer reads off `Rollout` before projecting.
    # Deliberately absent, by one rule — a field ships when a consumer reads it, not when the
    # survey says one might (executive call, user, 2026-09-15): a policy version (the per-row
    # convention is a *range*, since a span straddles weight updates; per-turn truth stays on
    # `TokenStamp.weight_version`), an is-last flag (derivable from `span_index` and a count
    # the adapter keeps), and a truncation signal (the shipped SLIME adapter derives slime's
    # `status` from `Rollout.finished`, not from any per-row finish reason; a span-level value
    # would also have to pick one turn's, hiding a mid-span cut — per-turn truth is on
    # `TokenStamp.finish_reason`). #1133's rule applies to all three: add the field when the
    # trainer that needs it is adopted, sourced from the stamps.
    #
    # `prefix_len` is the FIRST trained turn's response start, read off that stamp's
    # `sampled_span` in `_flatten`. "Index of the first 1 in the mask" yields the same number
    # (no test can tell them apart, and none claims to); what the tests DO pin is that
    # `len(ids) - prefix_len`, the response length, is not `sum(loss_mask)` — the post-stop
    # separator and, on a multi-turn span, the later turns' observations all sit at mask 0
    # inside the response region — and that the boundary is the first turn's, not the last's.
    ids: tuple[int, ...]
    loss_mask: tuple[int, ...]
    logprobs: tuple[float, ...]
    prefix_len: int
    invoke_id: str
    span_index: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible projection — the JSONL unit (ids/mask/logprobs as lists)."""
        return {
            "ids": list(self.ids),
            "loss_mask": list(self.loss_mask),
            "logprobs": list(self.logprobs),
            "prefix_len": self.prefix_len,
            "invoke_id": self.invoke_id,
            "span_index": self.span_index,
        }


@dataclass(frozen=True)
class Rollout:
    """One invoke's frozen trajectory record: its turns plus terminal facts.

    Attributes:
        invoke_id: The invoke this trajectory belongs to.
        depth: Nesting depth of that invoke — ``1`` for a top-level call — which is what
            separates a sub-invoke's trajectory from its parent's.
        turns: The token record of every LLM query the invoke made, in order.
        finished: ``"completed"`` for a normal return, or ``"aborted:<ExceptionType>"``
            (e.g. ``"aborted:IterationLimitExhaustedError"``) carrying the concrete exception
            class name. Partial rollouts are emitted, not dropped, and are filterable on this.
    """

    # The fields here are the minimum a trainer needs (executive call, user, 2026-09-13): the
    # token payload (`turns`), the identity a sample is attributed to (`invoke_id`, `model`),
    # the nesting position that separates a sub-invoke's trajectory from its parent's
    # (`depth`), and the terminal marker partials are filtered on (`finished`). Each is either
    # the payload itself or a fact not recoverable from the payload — the admission test.
    # `started_at`/`ended_at` were here and failed it: wall-clock boundaries are neither, and
    # carrying them cost a permanent export key pair plus the `InvokeEnter` subscription whose
    # only job was to stash the enter stamp. Dropping the fields dropped that handler too.
    # Nothing became unmeasurable: both stamps were just `Event.timestamp` off
    # `InvokeEnter`/`InvokeExit`, which any hook can read for itself.

    invoke_id: str
    depth: int
    turns: tuple[TurnRecord, ...]
    finished: str

    def to_spans(self) -> list[Sample]:
        """The trajectory as flat masked samples, one per maximal prefix-monotone span.

        Exactly one sample when every turn's context extended the previous one; a new
        sample at each break — a stripped reasoning trace, a transient hook nudge, a
        compaction. Always resolves. A pre-break turn appearing in a later span's *context*
        has loss mask 0 there: it was sampled against its own span's prefix and gets its
        loss there, so each sampled token is trained exactly once across the list.
        """
        spans: list[Sample] = []
        start = 0
        for i in range(1, len(self.turns)):
            if not _extends(self.turns[i - 1], self.turns[i]):
                spans.append(self._flatten(self.turns[start:i], len(spans)))
                start = i
        if self.turns:
            spans.append(self._flatten(self.turns[start:], len(spans)))
        return spans

    def _flatten(self, turns: tuple[TurnRecord, ...], span_index: int) -> Sample:
        """Flatten a monotone span: the final turn's context + sample, masked."""
        last = turns[-1]
        # Loss mask 1 only for segments sampled by THESE turns. In a whole-rollout
        # flatten that is every assistant sample; in a post-break span, a sample from
        # before the break can appear in the context (sent) — it was sampled against a
        # *different* prefix (that is what the break means), so masking it here would
        # score it against tokens the model never saw. It gets loss in its own span.
        # Keyed by message_id, not id(): same in-process semantics as object identity but
        # serialize/reload-safe, and it drops the id()-as-key caveat (that would be sound only
        # because self.turns keeps every stamp alive for the whole call). See _extends.
        own_samples = {t.sampled.message_id for t in turns}
        # The final turn's context contains every earlier turn's tokens (monotone), so
        # the sequence is just its sent stamps + its sample, with the preamble in front.
        ids: list[int] = list(last.initial)
        mask: list[int] = [0] * len(last.initial)
        logprobs: list[float] = [0.0] * len(last.initial)
        prefix_len: int | None = None
        for stamp in (*last.sent, last.sampled):
            offset = len(ids)
            ids.extend(stamp.ids)
            mask.extend([0] * len(stamp.ids))
            logprobs.extend([0.0] * len(stamp.ids))
            if stamp.sampled_span is not None and stamp.message_id in own_samples:
                start, end = stamp.sampled_span
                if prefix_len is None:
                    # The FIRST trained turn's response start, read off its own sampled_span.
                    # In a multi-turn span the later turns' observations then sit inside the
                    # response region at mask 0 — slime's `response_length` shape exactly.
                    prefix_len = offset + start
                for i, lp in zip(
                    range(offset + start, offset + end),
                    stamp.logprobs or (),
                    strict=True,  # span/logprob alignment is validated at construction
                ):
                    mask[i] = 1
                    logprobs[i] = lp
        return Sample(
            ids=tuple(ids),
            loss_mask=tuple(mask),
            logprobs=tuple(logprobs),
            # A span with nothing sampled has no boundary — the whole sequence is prefix, so
            # the response length is 0. Reachable at runtime, not only from a hand-built record:
            # a hook can hand in a `TurnRecord` with `sampled_span=None` through
            # `SupplyLLMResponse.tokens` / `ModifyLLMResponse.tokens`. SGLang itself always sets
            # a span on what it samples (an empty `(k, k)` on a no-output abort).
            prefix_len=len(ids) if prefix_len is None else prefix_len,
            invoke_id=self.invoke_id,
            span_index=span_index,
        )


class RolloutRecorder(Hook):
    """Accumulate each invoke's :class:`~jaz.llm.TurnRecord`\\ s and freeze them into ``rollouts``.

    Use as a context manager (``with RolloutRecorder() as rec:``); after the block,
    ``rec.rollouts`` holds one :class:`Rollout` per recorded invoke, in completion
    order. Invokes whose backend produced no turn records (text backends, replay
    overrides) yield no rollout — a warning is logged once per invoke.
    """

    def __init__(self) -> None:
        self.rollouts: list[Rollout] = []
        self._log: dict[str, list[TurnRecord]] = {}
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
        self._stood_down = set()

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
                    event.config.llm.model_info.name,
                )
                self._stood_down.add(event.invoke_id)
                self._log.pop(event.invoke_id, None)
                return []
            # Stored unchanged, which keeps the stamp references identical — `_extends` chains
            # on them.
            self._log.setdefault(event.invoke_id, []).append(record)
        return []

    def on_invoke_exit(self, event: InvokeExit) -> list[Effect]:
        with self._lock:
            stood_down = event.invoke_id in self._stood_down
            self._stood_down.discard(event.invoke_id)
            logged = self._log.pop(event.invoke_id, [])
            if stood_down or not logged:
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
                    turns=tuple(logged),
                    finished=finished,
                )
            )
        return []
