"""Hook system: observe and influence a running agent through events and effects.

The mental model in three sentences. As an agent runs, the framework fires typed
**events** at fixed points: once around the whole invoke, and per turn around the LLM
query and the REPL execution — each of those three spans walks the same four stages,
``Enter → Send → Complete → Exit`` (see :mod:`jaz.hooks.events` for the pipeline). A
**hook** is a class with a method per event it cares about (:class:`LLMQueryEnter` →
``on_llm_query_enter``); each handler receives the event and returns a list of
**effects** — the only way a hook changes anything (:mod:`jaz.hooks.effects` says which
effect is valid at which stage). Effects from all active hooks are composed
order-independently and applied by the framework; a hook that returns ``[]`` everywhere
is a pure observer.

Public surface (three parts):

- ``jaz.hooks`` — the authoring primitive :class:`Hook` and the ready-made
  "battery" hooks (:class:`PrintLogger`, :class:`BudgetPool`, :class:`RecursionLimit`,
  :class:`ReturnType`, ...).
- ``jaz.hooks.events`` — the typed events a hook handler *receives* (:class:`InvokeEnter`,
  :class:`REPLExecExit`, ...).
- ``jaz.hooks.effects`` — the typed effects a hook handler *returns* (:class:`ModifyExecResult`,
  :class:`AddMessages`, :class:`Abort`, ...).

A complete first hook — count the turns, nudge the agent, then stop it::

    from jaz import invoke
    from jaz.hooks import Hook
    from jaz.hooks.effects import Abort, AddMessages, Effect
    from jaz.hooks.events import LLMQueryEnter

    class TurnLimiter(Hook):
        def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Effect]:
            if event.iteration >= 10:
                return [Abort(error=RuntimeError("out of turns"))]
            if event.iteration >= 8:
                return [AddMessages([{"role": "user", "content": "Finish soon."}])]
            return []

    # `with` activation propagates to nested invokes:
    with TurnLimiter():
        result = invoke(task="...")

    # Passing a hook positionally scopes it to that one invoke — nested invokes
    # do NOT see it. See the Hook class for the difference.
    result = invoke(TurnLimiter(), task="...")

Before writing your own, check the battery hooks — limits, budgets, logging, tracing and
replay are already covered.

**Experimental.** The hook system is an experimental feature; its interfaces may change
in a future release.
"""

# The event/effect *vocabulary* deliberately lives in the ``events`` / ``effects``
# sub-namespaces, not flat on ``jaz.hooks`` — an Event or Effect only has meaning inside a
# hook, so it belongs to the hook system's grammar, scoped under it — the path is
# ``jaz.hooks.events.X`` / ``jaz.hooks.effects.X``.

# Public sub-namespaces — the hook vocabulary (events you receive, effects you return),
# advertised in __all__ (the docs generator discovers them there).
from . import effects, events

# Public authoring surface: the `Hook` primitive + the ready-made "battery" hooks. The
# hook *vocabulary* lives in the events/effects sub-namespaces (jaz.hooks.events /
# jaz.hooks.effects). The dispatch engine (`HookDispatcher`), the internal
# contract base (`ExecutionContext`), and `MetaData` are internal: they live
# in `_`-prefixed modules and are not re-exported here. The per-invoke blackboard needs no entry
# here at all — it is a plain read-only `Mapping` on every event, not a jaz-defined type.
#
# `TrajectoryRecorder`/`TrajectoryReplay` are public here as of 2026-08-15 (user call),
# superseding the review-of-#962 demotion of the replay hook and the recorder's
# deep-path-only status: ATIF is the canonical trace/replay/cost format (#813, #1159), so its
# write/read pair belongs on the offered surface rather than behind deep imports every consumer
# had to know.
#
# The pair has been renamed twice. At promotion the replay hook went `Replay` -> `ATIFReplay`
# (user call, same review): `RolloutRecorder` already makes token-native rollouts a second
# recorded format, so the bare `Replay` would collide with a future token-native one, and a
# shared prefix keeps the write/read pair adjacent in `__all__` and on the docs page. Later
# `ATIFTrace`/`ATIFReplay` became `TrajectoryRecorder`/`TrajectoryReplay`: the prefix argument
# survives (they still share one), but ATIF is the *format* these read and write, not what they
# are, and `Trace` named the artifact where an actor was meant. `Trajectory` rather than `Trace`
# because `OTelTracing` already owns "trace" in this namespace, in the industry sense.
# `Rollout` and `Sample` are public for the reason stated in `jaz.repl.__init__`: a caller is
# handed one. `RolloutRecorder.rollouts` is the whole point of the hook, and it hands back
# `Rollout`s whose one projection, `to_spans()`, returns `Sample`s — so publishing the recorder
# while leaving its output unnameable advertised a hook whose result no caller could type
# (EXECUTIVE CALL, user, 2026-09-13; collapsed to one sample type 2026-09-15).
from ._builtin import (
    BudgetForcing,
    BudgetPool,
    ContextWindowWarning,
    FileLogger,
    IterationLimit,
    PrintLogger,
    RecursionLimit,
    ReturnType,
    Rollout,
    RolloutRecorder,
    Sample,
    TrajectoryDirectoryRecorder,
    TrajectoryRecorder,
    TrajectoryReplay,
    ValidateREPLCode,
    ValidateReturn,
)
from ._dispatcher import Hook

try:
    from ._builtin import JaegerTracing, LangfuseTracing, OTelTracing  # noqa: F401

    _TRACING_AVAILABLE = True
except ImportError:
    _TRACING_AVAILABLE = False

__all__ = [
    # Public sub-namespaces (the hook vocabulary)
    "events",
    "effects",
    # Authoring primitive
    "Hook",
    # Concrete built-in hooks ("batteries")
    # Writers before the reader: the two hooks that WRITE a trajectory, then the one that
    # reads one back, which is the order the docs page renders. (This block was alphabetical
    # when the pair was `TrajectoryRecorder`/`TrajectoryReplay` and the two rules agreed;
    # `TrajectoryDirectoryRecorder` sorts before both, so only the write-before-read rule
    # survives.)
    "TrajectoryRecorder",
    "TrajectoryDirectoryRecorder",
    "TrajectoryReplay",
    "BudgetForcing",
    "BudgetPool",
    "ContextWindowWarning",
    "IterationLimit",
    "RecursionLimit",
    "PrintLogger",
    "FileLogger",
    "ReturnType",
    "RolloutRecorder",
    "Rollout",
    "Sample",
    "ValidateReturn",
    "ValidateREPLCode",
]

if _TRACING_AVAILABLE:
    # `OTelTracing` alongside its two presets: it is the base a caller subclasses to export to
    # any other OTLP collector, so the extension seam has to be reachable by the same route as
    # the batteries. It was previously deep-path-only, which meant the one tracing name a user
    # is expected to subclass was the only one they could not import from here — and silently,
    # since nothing flags a deep-path module read.
    __all__.append("OTelTracing")
    __all__.append("JaegerTracing")
    __all__.append("LangfuseTracing")

# Reachable-but-unsupported names (blessed homes noted). The hook
# vocabulary is public under jaz.hooks.events / jaz.hooks.effects — the flat
# `jaz.hooks.<name>` spelling of it is demoted; the dispatch engine / internal machinery
# has no public home.

# No pre-rename hook spellings are kept. #806 question 1 dropped the ``Hook`` suffix across the
# set, later renames added more (``ContextWindow`` -> ``ContextWindowWarning``,
# ``ValidateREPLInput`` -> ``ValidateREPLCode``, ``Replay``/``ATIFReplay`` ->
# ``TrajectoryReplay``, ``ATIFTrace`` -> ``TrajectoryRecorder``), and each left an alias here so
# an out-of-tree import kept working with a warning. They are all gone: the set had grown to
# sixteen dead spellings across four renames, which made the public surface read as larger and
# more settled than it is — ``dir(jaz.hooks)`` advertised three names for one replay hook. An old
# name is now an ``AttributeError`` naming the module, the same failure a typo gives. Pre-1.0 is
# when that trade is cheapest; ``ATIFTrace``/``ATIFReplay`` were announced in 0.2.0a2 and so do
# have released users, which is what the changelog entry is for rather than a reason to keep them.
