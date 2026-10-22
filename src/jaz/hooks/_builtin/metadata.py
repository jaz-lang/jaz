"""MetaData: a generic carrier hook that seeds the per-invoke blackboard.

``MetaData`` is the entry point for putting caller-supplied per-invoke data onto the
blackboard (see ``Event.blackboard`` and ``design/design_features/blackboard.md``).
It is deliberately **generic** — it seeds whatever keys you hand it, rather than being one
hook per key (a ``TaskName`` / ``Label`` class per datum would fight the blackboard's
"generic, declared, validated keys" design). The first such key is ``label`` (consumed by
``TrajectoryDirectoryRecorder`` and the tracing hooks).

Per-call vs ambient
-------------------
Because seeding reads *active* hooks, ``MetaData`` works on both activation paths:

- **Per-call** (the common case for a distinct label): pass it as a leading positional
  local hook — ``invoke(MetaData(label="analyze"), task=...)`` — so it scopes to that
  one call and does not propagate to nested invokes.
- **Ambient**: ``with MetaData(label="analyze"): ...`` labels every invoke in the
  block.

Agent-facing note: the agent's ``invoke`` tool takes hooks positionally too, so an agent
can label its own sub-invokes by passing ``MetaData(label=...)``, provided the host
passed ``MetaData`` in as an input.

Validation note: seeding a key nothing active reads is not checked at all —
``MetaData(label="x")`` with no consumer runs silently, which is the point of a decoupled
board. In realistic use you only label when a consumer (e.g. ``TrajectoryDirectoryRecorder`` or
a tracing hook) is active, and *not* labeling is fine: each consumer falls back to a name of its
own (``"main"`` for the tracing hooks, the task text for ``TrajectoryDirectoryRecorder``). A *conflicting*
value for a key does still raise (``BlackboardSeedConflictError``) — an irresolvable ambiguity.
"""

# Agent-settable metadata is #545: a sub-invoke labels itself rather than the host guessing.

from __future__ import annotations

from jaz.hooks._dispatcher import Hook


class MetaData(Hook):
    """Attach key/value metadata to an invoke for other hooks to read.

    Under ``with``, the metadata is attached to every invoke in scope. Passed positionally,
    only to that invoke.

    Args:
        **metadata: Key/value pairs other hooks can read. Available from the very first
            event. Seeding a key no active hook reads only warns; seeding one key with two
            conflicting values raises.

    Examples:
        with TrajectoryDirectoryRecorder("./trace"):
            invoke(ReturnType(dict), MetaData(label="analyze"), task="Analyze data")
    """

    # The pairs go into the blackboard's generation 0, which is what makes them visible to
    # every event including ``InvokeEnter``.
    #
    # No key is rejected here, including spellings this framework has retired (EXECUTIVE CALL,
    # user, 2026-09-22). The point of the hook is that it is an open bag: which keys are
    # meaningful is the producers' and consumers' business, not this constructor's, and a
    # rejection list would make it one. The usual guard-and-test rule for removing a public
    # spelling does not bite — there are no users to migrate.

    def __init__(self, **metadata: object) -> None:
        self._metadata = dict(metadata)

    @property
    def blackboard_seed(self) -> dict[str, object]:
        # Pure function of own constructor state (no invoke context), as required of
        # a seed source. Copied so the returned dict can't alias internal state.
        return dict(self._metadata)
