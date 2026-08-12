"""Shared must-exit-warning machinery for the baseline force-finish hooks.

Both :class:`~jaz.hooks.builtin.iteration_limit.IterationLimit` (last-iteration
notice) and :class:`~jaz.hooks.builtin.context_window.ContextWindow`
(context-window warning) surface the same agent-facing "you must finish now"
warning. The renderer machinery lives here so the two hooks share one definition
rather than coupling to each other.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TemplatedMustExitWarning:
    """Stock ``must_exit_warning`` renderer backed by a Jinja template.

    A callable suitable for ``IterationLimit(must_exit_warning=...)`` /
    ``ContextWindow(must_exit_warning=...)``: renders ``template_name`` with the one per-invoke
    *core* variable — ``can_delegate`` (whether this invoke can delegate to a subagent via
    ``jaz.invoke()``; sourced from ``LLMQueryEnter.can_recurse``) — and nothing else. The warning
    is deliberately agnostic to *how* the REPL finishes (return vs raise): telling the agent how to
    end the session is the REPL's own concern (its description), not this governance-layer nudge, so
    the text just says to finish. Core also names no prompt feature flags: whether the warning
    mentions REPL history, exposed inputs, etc. is an eval-configuration concern baked into the evals
    delegate meta-template at config time (the eval harness's ``bake_template``, #898/#912), leaving
    the runtime template needing only ``can_delegate``. The core default template
    (``must_exit_warning.jinja2``) is flag-free, so a direct, non-eval caller still gets a sensible
    default. There is deliberately no generic ``template_vars`` escape hatch: core must not take a
    template that requires additional (eval) render vars — that is what baking is for. A frozen
    dataclass rather than a closure so it is picklable and introspectable (round-trips through the
    eval harness's worker-config provenance).
    """

    template_name: str

    def __call__(self, can_delegate: bool) -> str:
        from jaz.template_loader import _jinja_env

        # can_delegate is the one per-invoke *core* variable (it isn't knowable at construction).
        # Every other var a delegate template needs is baked in at config time, so nothing else is
        # passed (and _jinja_env is StrictUndefined: a template referencing an unbaked toggle raises).
        return _jinja_env.get_template(self.template_name).render(
            can_delegate=can_delegate,
        )


def resolve_must_exit_warning(
    warning: str | Callable[[bool], str] | None,
    *,
    can_delegate: bool,
) -> str:
    """Resolve a ``must_exit_warning`` to text.

    ``None`` (the baseline default) renders the core ``must_exit_warning.jinja2`` — a
    finish-method-agnostic nudge (it says to finish the session, not *how*, since the finish
    mechanism is the REPL's own concern, described in its REPL description). A plain string is
    returned as-is; a callable is invoked with the single per-invoke fact it may depend on,
    ``can_delegate`` (the hook owns *when* the warning appears, the callable owns *what* it says).
    An empty string / empty render means "emit nothing".

    The callable contract is deliberately a bare ``can_delegate: bool`` rather than the
    ``LLMQueryEnter`` event or a curated context object. This is a conscious YAGNI tradeoff: a bare
    bool is a *breaking* bump to extend — a second per-invoke fact would change every custom
    ``must_exit_warning`` signature — whereas the event (already a public hook type) or a context
    object would extend without breaking callers. It is chosen anyway because the alternatives cost
    more today: passing the event pushes the ``can_recurse`` → ``can_delegate`` rename onto every
    callable and hands a text-tweaking author ``messages``/``model`` it has no business reading, and
    a context object is a second type to keep in sync. ``can_delegate`` is the *one* fact the
    warning text varies on, so that is all a callable receives; revisit only when a second fact is
    genuinely needed.
    """
    if warning is None:
        warning = TemplatedMustExitWarning(template_name="must_exit_warning.jinja2")
    if callable(warning):
        # TODO(#923): revisit the bare-bool callable contract (reintroduce a curated context
        # object) if a second per-invoke fact is ever needed, or if real external callables exist.
        return warning(can_delegate)
    return warning


def serialize_must_exit_warning(warning: object) -> object | None:
    """Serialize a hook's ``must_exit_warning`` for round-tripping (#441).

    Returns ``None`` (→ omit the key, reconstruction uses the hook default, i.e. the stock
    finish-agnostic warning) for the default ``None`` and for any
    non-``TemplatedMustExitWarning`` callable — an arbitrary callable fundamentally
    can't be JSON-serialized, a documented residual limitation (the eval harness
    round-trips its TemplatedMustExitWarning separately via
    ``must_exit_warning_template`` provenance). A custom plain string and a
    ``TemplatedMustExitWarning`` (serialized to a tagged dict) do round-trip.

    Lives here (not in config.py) so a force-finish hook's ``to_dict()`` can serialize
    its own ``must_exit_warning`` without a config→hook import cycle (#727).
    """
    if isinstance(warning, TemplatedMustExitWarning):
        # asdict (not a hand-listed dict) so new TemplatedMustExitWarning fields
        # round-trip automatically: deserialize splats every non-tag key straight
        # into the constructor, so a hand list here would be a second field list to
        # keep in sync — the exact silent param-loss this machinery exists to prevent.
        return {"__templated_must_exit_warning__": True, **asdict(warning)}
    if isinstance(warning, str):
        return warning
    return None


def deserialize_must_exit_warning(value: object) -> object:
    if isinstance(value, dict) and value.get("__templated_must_exit_warning__"):
        kwargs = {
            k: v for k, v in value.items() if k != "__templated_must_exit_warning__"
        }
        return TemplatedMustExitWarning(**kwargs)
    return value
