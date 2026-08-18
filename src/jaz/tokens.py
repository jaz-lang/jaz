"""Per-message token records: the exact token ids each conversation message contributes.

Two ownership rules generate the whole token-recording design:

- **A message owns the tokens intrinsic to its own content.** For a sampled assistant turn
  that is the verbatim server output (plus logprobs, weight version, finish reason) — ground
  truth, never re-derived. For any other message it is the ids the chat template rendered for
  it. These ride *inside* the ``MessageDict`` as a :class:`TokenStamp` under
  :data:`TOKENS_KEY` (the same reserved-key sidecar mechanism as
  :data:`jaz.provenance.PROVENANCE_KEY`), so they follow the message by reference through
  every append, drop, edit and compaction.
- **A turn owns which messages it sent.** That is an *event*, intrinsic to no message, so it
  is logged when it happens: a :class:`TurnRecord` holding ordered references to the
  per-message stamps that were on the wire, plus the newly sampled segment. It is returned on
  ``LLMResponse.tokens`` by token-native backends and accumulated by
  ``jaz.hooks.builtin.rollout.RolloutRecorder``.

Only token-native backends (e.g. ``SGLangLLM``) produce or read stamps; text backends never
see them — the wire projection at egress strips the key unless the backend declares it
consumes it (``BaseLLM.consumes_internal_keys``). Like provenance, a stamp is deliberately not
JSON-serializable, so a missed strip may fail loudly at ``json.dumps`` rather than leaking an
internal key to a provider (best-effort backstop; see ``jaz.provenance``'s module docstring
for the caveats).
"""

# Design rationale (maintainers; "Option C" of the token-native rollouts design, PR #1099 —
# the design doc did not merge, so the full rationale is recorded here and at the sibling
# sites it names):
# tokens bind to the objects that own them instead of living in a parallel per-invoke ledger
# (rejected Option A: a second structure that re-derives — and must continually re-verify —
# what the message list already determines, and whose consumed-message index cannot represent
# non-monotone histories produced by routine hooks like IterationLimit) or in a retyped
# token-aware buffer (rejected Option B: big-bang migration of every list[MessageDict]
# surface, and a buffer is still a current view, not a history). Executive call: Option C
# (user, 2026-08-13). With ownership in place there is no fidelity invariant left to enforce:
# nothing is ever re-derived, so nothing can drift.
#
# The load-bearing invariant: exactly ONE TokenStamp object exists per message, ever. The
# backend creates it on the message's first send, the agent stamps it back onto the buffer
# message, and every subsequent wire copy and every TurnRecord share that object by
# reference. This is simultaneously the memory bound (no O(turns^2) duplication of prompt
# ids), the backend's re-render skip, and the recorder's immutability guarantee (a TurnRecord
# references only frozen data, so no later buffer mutation can rewrite what a past turn is
# recorded as having sent).
#
# This module must keep ZERO runtime jaz imports: jaz.llm.base / jaz.llm.llm import
# TurnRecord at runtime (LLMResponse.tokens), so a runtime edge back into jaz.llm would be a
# cycle. Hence MessageDict is imported under TYPE_CHECKING only — a deliberate deviation from
# provenance.py's runtime import, which has no such consumer.

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jaz.llm import MessageDict

# Reserved MessageDict key holding a TokenStamp. Stripped at egress (to_wire_messages) for
# every backend that does not declare it in `consumes_internal_keys`. Accessed only via the
# helpers below, never inline, so the magic string lives in exactly one place.
TOKENS_KEY = "__jaz_tokens__"


@dataclass(frozen=True)
class TokenStamp:
    """Frozen token record for one message; rides in the ``MessageDict`` under
    :data:`TOKENS_KEY`.

    For a sampled assistant turn, ``ids`` is the message's full contribution to a future
    context — the generation-prompt tokens that opened the turn plus the verbatim sampled
    ids (including the stop token) — and ``sampled_span`` marks the verbatim-sampled slice
    (the only part that gets loss mask 1). For any other message, ``ids`` is the chat
    format's render of that one message and ``sampled_span`` is ``None``.

    Raises ``ValueError`` at construction when the sampled-only fields are inconsistent:
    ``sampled_span`` and ``logprobs`` must be present together and aligned, and
    ``weight_version``/``finish_reason`` are only meaningful on sampled stamps.
    """

    # Frozen for the same reason MessageProvenance is (safe sharing by reference across
    # turns/edits) plus a stronger one: sampled ids are training ground truth; in-place
    # corruption must raise, not silently poison a rollout (runtime guard 1 of the design's
    # intermediate immutability contract — load-bearing until the typed-Message migration,
    # #1132).
    #
    # ids/logprobs are plain tuples, not int32 arrays: array.array is MUTABLE
    # (stamp.ids[0] = 5 would succeed), which would silently break that guard — a frozen
    # dataclass only freezes attribute rebinding, not the container. Cost: ~9x the 4 B/token
    # a true int32 array would take; still bounded and knob-free (per-invoke id storage is
    # ~ the final context length, stored once — no O(turns^2) prompt copies). Moving the
    # backing store to immutable bytes behind the same tuple-typed accessors is #1136.

    role: str  # mirrors the message's role at stamp time
    ids: tuple[int, ...]  # exact ids this message contributes to any context
    # Message text at stamp time — the stale-stamp guard: the backend re-renders when the
    # message's content no longer matches.
    content_text: str
    # Half-open [start, end) slice of `ids` sampled by the model (loss mask 1). Everything in
    # `ids` OUTSIDE this span — the generation-prompt tokens that open the turn, plus any trailing
    # template framing — is not model-generated and so carries loss mask 0. Keeping that framing in
    # the sequence but OUTSIDE the trained span is the convention of the trainers Rollout exports
    # to: verl marks such boundary tokens no-loss (https://github.com/volcengine/verl), and
    # slime/AReaL mask the inter-turn framing out (https://github.com/THUDM/slime,
    # https://github.com/inclusionAI/AReaL).
    sampled_span: tuple[int, int] | None = None
    logprobs: tuple[float, ...] | None = None  # aligned with sampled_span; iff sampled
    weight_version: str | None = None  # serving policy version (sampled only)
    # Per stamp, not per rollout: in a multi-turn trajectory one segment can end with
    # "length" while others end with "stop", and trainers mask those differently. The
    # invoke-level terminal marker lives on Rollout.finished.
    finish_reason: str | None = None  # "stop" | "length" | "abort" (sampled only)
    # Stable identity handle — the reference key by which turn records and export identify
    # a stamp. Unlike MessageProvenance.id this DOES participate in equality (no
    # compare=False): provenance wants value-equality so a surviving message can share one
    # instance across edits, while stamps want identity semantics — two same-content renders
    # of a message (e.g. after a stale-stamp re-render) must never compare equal, or export
    # would falsely chain across a history edit. It is also not the provenance id: stamps
    # exist for transient messages the agent never provenance-stamps, and the backend mints
    # stamps without touching provenance.
    #
    # Declared last (with a default) because dataclass rules forbid a defaulted field
    # before required ones; the position carries no meaning.
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        # "Required iff trainable", enforced at the write (both directions), plus span
        # bounds/alignment. Rejecting sampled-only metadata on rendered stamps is a choice
        # to enforce rather than merely document that those fields describe a sampling
        # event; trivially relaxable if a rendered-stamp use case appears.
        if self.sampled_span is not None and self.logprobs is None:
            raise ValueError(
                "a sampled TokenStamp (sampled_span set) requires logprobs"
            )
        if self.logprobs is not None and self.sampled_span is None:
            raise ValueError("logprobs require sampled_span (they align with it)")
        if self.sampled_span is not None:
            start, end = self.sampled_span
            if not (0 <= start <= end <= len(self.ids)):
                raise ValueError(
                    f"sampled_span {self.sampled_span} out of bounds for {len(self.ids)} ids"
                )
            assert self.logprobs is not None
            if len(self.logprobs) != end - start:
                raise ValueError(
                    f"logprobs length {len(self.logprobs)} != sampled_span length {end - start}"
                )
        elif self.weight_version is not None or self.finish_reason is not None:
            raise ValueError(
                "weight_version/finish_reason are sampled-only fields; this stamp has no "
                "sampled_span"
            )


@dataclass(frozen=True)
class TurnRecord:
    """What one generation call sent and sampled. Built inside ``complete()``; immutable.

    ``sent`` holds one stamp per wire message, in order — shared references to the same
    objects that ride in the messages, not copies. ``sampled`` is the newly sampled
    assistant turn's stamp. ``initial`` is the context preamble (BOS / template preamble)
    the backend prepended before any message's ids; no message owns it.

    Backend contract for ``sent`` that ``complete()`` must honor: exactly one stamp per wire
    message, in send order, with no merging of consecutive messages, reordering, or injected
    messages. The agent's stamp-back pairs ``sent[i]`` with wire message ``i`` positionally,
    so any deviation misaligns the stamps (a length change raises; a same-length reorder or
    merge is caught per-message by role, else surfaces later as a broken export).
    """

    # `sent` as shared references (pointer cost) rather than copied id sequences is what
    # keeps per-call observability O(1) in prompt length — the first draft's per-call
    # prompt_token_ids copy was the rejected O(turns^2) alternative. Built inside
    # complete() because the backend is the only party that ever holds the wire list; each
    # retry attempt builds its own record from its own server response and mutates nothing
    # shared, so a failed attempt's objects are simply dropped with it (no double-append
    # hazard).
    #
    # iteration: the backend cannot know the agent's iteration counter (complete() receives
    # only model/messages/request kwargs, and threading an iteration kwarg through would
    # leak into wire payloads on every backend), so token-native backends set the -1
    # sentinel and RolloutRecorder rewrites it from LLMQueryExit.iteration — the
    # authoritative source — via dataclasses.replace. Records reached any other way than
    # the recorder may still carry -1.
    #
    # initial is not a synthetic leading stamp in `sent` because stamp-back relies on the
    # positional contract sent[i] <-> wire[i].

    sent: tuple[TokenStamp, ...]  # one per wire message, in order — shared refs
    sampled: TokenStamp  # the newly sampled assistant turn's stamp
    iteration: int  # agent-loop iteration; -1 until rewritten by the recorder
    model: str
    initial: tuple[int, ...] = ()  # BOS / template preamble ids (loss mask 0)


def tokens_of(message: MessageDict) -> TokenStamp | None:
    """The message's token stamp, or ``None`` if it was never stamped (e.g. a message
    that has not yet been sent by a token-native backend)."""
    return message.get(TOKENS_KEY)


def set_tokens(message: MessageDict, stamp: TokenStamp) -> None:
    """Stamp ``message`` in place. Overwrites any existing stamp."""
    message[TOKENS_KEY] = stamp
