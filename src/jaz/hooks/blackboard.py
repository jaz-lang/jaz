"""The per-invoke blackboard: a generationally-versioned key-value store.

See ``design/design_features/blackboard.md`` for the full design. In brief: every
event carries a :class:`Blackboard` (``event.blackboard``, a sibling of
``event.config``) through which hooks receive per-invoke data and coordinate with one
another — *without* direct hook-to-hook coupling and *without* depending on hook
dispatch order.

Generational (double-buffered / BSP-superstep) semantics
--------------------------------------------------------
The dispatcher collects *all* hooks' effects for a single event before applying any
``BlackboardWrite`` to the board (see :meth:`HookDispatcher.emit`). So:

- **Within one event**, every hook reads the same pre-write snapshot — the board is
  mutated only *after* the hook loop completes, so two hooks reading cannot disagree
  based on who ran first.
- **Across events**, a write made during event *E* surfaces to event *E+1* (the next
  generation). The agent loop's event order — not hook order — sequences producers
  and consumers.

The collected effect list *is* the implicit write-buffer; no separate ``pending``
structure is needed. This module therefore holds only the live key-value state plus a
batch-apply method; the generational discipline lives in the dispatcher's emit
boundary.

Hooks treat a ``Blackboard`` as a **read-only** ``Mapping``. The only way to mutate it
is to emit a ``BlackboardWrite`` effect, which the dispatcher applies via the
underscore-prefixed :meth:`_apply_writes` at the event boundary.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from jaz.exceptions import BlackboardWriteConflictError


class Blackboard(Mapping[str, object]):
    """A per-invocation key-value store, read-only from a hook's perspective.

    Constructed once per invoke (seeded with generation 0; see
    ``HookDispatcher.seed_blackboard``) and carried on every event. Hooks read it
    like a dict — ``event.blackboard["task_name"]`` / ``event.blackboard.get(...)`` —
    and write to it only by emitting ``BlackboardWrite`` effects.

    Implements the read half of ``Mapping`` (``__getitem__``/``__iter__``/``__len__``,
    and the mixin-provided ``get``/``__contains__``/``keys``/``items``/``values``); it
    intentionally exposes **no** public mutation API. Mutation is the dispatcher's job
    via :meth:`_apply_writes`.

    Reading an absent key
    ---------------------
    A key that has not (yet) been produced is simply **absent**: ``board[key]`` raises
    ``KeyError`` and ``board.get(key)`` returns ``None`` (or a supplied default). This is
    unvalidated on purpose — a hook may declare it consumes a key (``blackboard_consumes``)
    that no active hook ever seeds or writes, and that is allowed, not an error. Crucially,
    the generational model makes *"not produced yet"* and *"never produced"* **read
    identically** (both absent): a consumer must not be able to tell whether the producer
    simply hasn't run yet or isn't present at all — that indistinguishability is what
    preserves order-independence. **Consumers should therefore read defensively** (prefer
    ``.get(key, default)`` over ``board[key]``) and treat absence as "no value this
    generation." See the design doc's *Generational read semantics*.
    """

    def __init__(self, seed: Mapping[str, object] | None = None) -> None:
        # Generation 0. Copied so the caller's dict can't alias the board's state.
        self._data: dict[str, object] = dict(seed) if seed is not None else {}

    # --- read interface (Mapping) ------------------------------------------------

    def __getitem__(self, key: str) -> object:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Blackboard({self._data!r})"

    # --- dispatcher-only mutation ------------------------------------------------

    def _apply_writes(self, writes: list[tuple[str, object]]) -> None:
        """Apply a batch of writes, minting the next generation.

        Called by ``HookDispatcher.emit`` *after* every hook for the current event
        has run, so it is invisible to that event's reads (it surfaces to the next
        event). Within the batch, write order across hooks is irrelevant by design,
        so a same-key conflict has no order-independent resolution: identical writes
        coalesce, distinct ones raise :class:`BlackboardWriteConflictError`. Across
        batches (i.e. across events) the later value simply overwrites — that is
        sequenced by the agent loop, not by hook order.
        """
        batch: dict[str, object] = {}
        for key, value in writes:
            if key in batch and batch[key] != value:
                raise BlackboardWriteConflictError(
                    f"Conflicting BlackboardWrite for key {key!r} in the same event: "
                    f"{batch[key]!r} != {value!r}. Within one event, write order is "
                    "irrelevant, so a same-key conflict cannot be resolved without "
                    "relying on hook order. Ensure only one hook writes a given key "
                    "per event, or have them write identical values."
                )
            batch[key] = value
        self._data.update(batch)
