"""MetaData: a generic carrier hook that seeds the per-invoke blackboard.

``MetaData`` is the entry point for putting caller-supplied per-invoke data onto the
blackboard (see ``hooks/blackboard.py``). It is deliberately **generic** — it seeds
whatever keys you hand it, rather than being one hook per key (a ``TaskName`` /
``Label`` class per datum would fight the blackboard's "generic, declared, validated
keys" design). The first such key is ``task_name`` (consumed by
``WorkflowReplay`` / tracing / ``ATIFTrace``).

Per-call vs ambient
-------------------
Because seeding reads *active* hooks, ``MetaData`` works on both activation paths:

- **Per-call** (the common case for a distinct label): pass it as a leading positional
  local hook — ``invoke(MetaData(task_name="analyze"), task=...)`` — so it scopes to that
  one call and does not propagate to nested invokes.
- **Ambient**: ``with MetaData(task_name="analyze"): ...`` labels every invoke in the
  block.

Agent-facing note: the synthesized agent-facing ``invoke`` exposes ``*local_hooks`` too
(#545), so an agent can label its own sub-invokes by passing ``MetaData(task_name=...)``
positionally, provided the host passed ``MetaData`` in as an input (it is no longer bound
under ``jaz.*``).

Validation note: seeding a key no active hook declares in ``blackboard_consumes`` is a
lint — it logs a **warning** (see ``HookDispatcher._warn_orphan_keys``), not an error, so
``MetaData(task_name="x")`` with no active consumer still runs. In realistic use you only
label when a consumer (e.g. ``WorkflowReplay`` or a tracing hook) is active, and *not*
labeling is fine (consumers default to ``"main"``). A *conflicting* value for a key does
still raise (``BlackboardSeedError``) — an irresolvable ambiguity, not an orphan.
"""

from __future__ import annotations

from jaz.hooks.dispatcher import Hook


class MetaData(Hook):
    """Attach key/value metadata to an invoke for other hooks to read.

    Under ``with``, the metadata is attached to every invoke in scope. Passed positionally,
    only to that invoke.

    Args:
        **metadata: Key/value pairs other hooks can read. Available from the very first
            event. Seeding a key no active hook reads only warns; seeding one key with two
            conflicting values raises.

    Examples:
        with WorkflowReplay(output_dir="./wf"):
            invoke(ReturnType(dict), MetaData(task_name="analyze"), task="Analyze data")
    """

    # The pairs go into the blackboard's generation 0, which is what makes them visible to
    # every event including ``InvokeEnter``. "A key no active hook reads" means one no hook
    # declares in ``blackboard_consumes`` — an orphan-seed lint, not an error.

    def __init__(self, **metadata: object) -> None:
        self._metadata = dict(metadata)

    def blackboard_seed(self) -> dict[str, object]:
        # Pure function of own constructor state (no invoke context), as required of
        # a seed source. Copied so the returned dict can't alias internal state.
        return dict(self._metadata)
