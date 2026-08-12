"""Per-message provenance: what each conversation message *is* and where it came from.

Provenance answers a question the raw message can't: is this the original system prompt,
the task, an assistant turn, a REPL observation, or a hook-inserted summary? Consumers read
it instead of guessing from role/position — ``ConversationHistory`` to log a message's
true origin, ``SlidingWindow`` to pin the load-bearing prefix (system + task). The guess
breaks the moment a persistent edit (compaction) reorders or inserts messages; explicit
provenance does not.

**Storage: an in-dict reserved key, stripped at the provider egress.** Provenance rides
*inside* the ``MessageDict`` under :data:`PROVENANCE_KEY`, so it carries through
:func:`jaz.hooks.effects.apply_message_edits` and every ``messages.append(...)`` for free
(the dict moves by reference) — there is no parallel list to keep aligned. It must never
reach a provider: the model-facing message list is projected to wire form via
:func:`to_wire_messages` at the single egress point (``Agent._compose_shown_messages``).
:class:`MessageProvenance` is deliberately a plain dataclass (**not** JSON-serializable by
default) — a *missed* strip **may** then raise at ``json.dumps`` rather than silently leaking
an internal key to the model. This is a best-effort backstop, not a guarantee: a serializer
that coerces unknown types (``json.dumps(..., default=str)`` — used elsewhere in this codebase
— or a provider/pydantic layer with arbitrary-type fallback) would stringify the stray
``MessageProvenance`` instead of raising. The actual safety property is structural, not
incidental: :func:`to_wire_messages` at the single egress point (``Agent._compose_shown_messages``)
is the *only* path from the loop buffer to a provider, so nothing downstream depends on
``json.dumps`` failing — the non-serializable dataclass is a defense-in-depth nicety on top of
that, not the load-bearing guarantee.

A type-safe wrapper alternative (``list[Message]`` binding dict + provenance, unwrapped before
send) was considered and deferred to a tracking issue: it is leak-proof by construction (the
provider signature forbids sending an un-unwrapped message) but retypes the buffer and the
``event.messages`` API everywhere, churn not yet justified.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from jaz.providers import MessageDict


class MessageKind(StrEnum):
    """The origin of a conversation message.

    ``SEED`` is *every* message in the initial rendered message list — the whole thing
    the protocol produces before any turn runs, not just the system + user prompt. For
    ``DefaultProtocol`` that is ``[system, user]`` (the user message being today's
    ``<task>`` + ``<inputs>`` + ``<return_type>``, still true once a pending refactor
    collapses that to just ``<inputs>``); a custom protocol may also seed extra context
    such as few-shot ``assistant`` examples, and those are seed too. This is the
    load-bearing prefix consumers protect. One kind, not several: every current and
    near-term consumer (``SlidingWindow``'s pinning, its eventual identity-based
    successor) treats the seed as a single protected unit, never distinguishing within
    it, and a consumer that does want a particular seed message (the system prompt, say)
    reads that message's own ``role`` — a finer kind would carry no information ``role``
    plus ``iteration is None`` doesn't already give. ``ASSISTANT``/``OBSERVATION`` are the
    normal turn-by-turn growth (``OBSERVATION`` is also ``role="user"`` but carries REPL
    *output* fed back, not part of the seed). ``ADDED`` is content a hook inserted via
    ``AddMessages`` (e.g. a compaction summary) — distinguishable from a genuine turn.
    """

    SEED = "seed"
    ASSISTANT = "assistant"
    OBSERVATION = "observation"
    ADDED = "added"


@dataclass(frozen=True)
class MessageProvenance:
    """Immutable origin record stamped onto a message.

    Frozen so a surviving message can share one instance across turns/edits by reference.
    Intentionally not JSON-serializable (see the module docstring): serialize it explicitly
    where a human-facing record is wanted (e.g. ``ConversationHistory``), and strip it
    via :func:`to_wire_messages` everywhere else.
    """

    kind: MessageKind
    iteration: int | None = None  # iteration when created; None for the seed
    persistent: bool | None = None  # for ADDED: persistent vs transient

    def to_dict(self) -> dict[str, Any]:
        """Project to a JSON-compatible dict for a human-facing record.

        Explicit opt-in, not automatic ``json.dumps`` support: this dataclass stays
        deliberately non-serializable by default (see the module docstring) so a missed
        strip stays visible; call this only where a serialized record is actually wanted
        (e.g. :class:`~jaz.hooks.builtin.conversation_history.ConversationHistory`).
        """
        return {
            "kind": self.kind.value,
            "iteration": self.iteration,
            "persistent": self.persistent,
        }


# Reserved MessageDict key holding a MessageProvenance. Never sent to a provider — projected
# out by to_wire_messages() at egress. Accessed only via the helpers below, never inline, so
# the magic string lives in exactly one place.
PROVENANCE_KEY = "__jaz_provenance__"


def provenance_of(message: MessageDict) -> MessageProvenance | None:
    """The message's provenance, or ``None`` if it was never stamped (e.g. a message a
    hook constructed by hand)."""
    return message.get(PROVENANCE_KEY)


def set_provenance(message: MessageDict, provenance: MessageProvenance) -> None:
    """Stamp ``message`` in place. Idempotent-friendly: overwrites any existing stamp."""
    message[PROVENANCE_KEY] = provenance


def to_wire_messages(messages: list[MessageDict]) -> list[MessageDict]:
    """Provider-ready copies with :data:`PROVENANCE_KEY` removed.

    Only messages that actually carry provenance are copied; the rest pass through by
    reference, so an un-stamped buffer is returned untouched. This is the single choke point
    that keeps the internal provenance key out of every provider payload.
    """
    return [
        {k: v for k, v in m.items() if k != PROVENANCE_KEY}
        if PROVENANCE_KEY in m
        else m
        for m in messages
    ]
