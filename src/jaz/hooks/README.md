# Hook System

The hook system provides a type-safe, composable way to observe and influence agent execution through events, effects, and contexts.

## Design Philosophy

The hook system was redesigned around **context-based hook management** to achieve several key goals:

### Why Context Variables?

1. **Automatic Propagation**: Hooks naturally propagate to nested `invoke()` calls without explicit parameter passing
2. **Scope-Based Lifecycle**: Hooks are active exactly within their `with` block - no manual cleanup needed
3. **Thread-Safe**: Python's `contextvars` module provides thread-safe and async-safe context management
4. **Composability**: Easy to add, remove, or temporarily disable hooks in nested scopes
5. **Cleaner API**: No need to pass `dispatcher` parameters through function calls

### Core Principles

1. **Hooks are order-independent** - Can run in parallel, composition is deterministic
2. **Type system prevents invalid operations** - Can't halt at InvokeExit, can't modify prompt in LLMQuery
3. **Composition rules are explicit** - No hidden behavior, easy to reason about
4. **Context managers for lifecycle** - Pythonic way to manage hook activation/deactivation
5. **Stateless dispatcher** - All state lives in context variables, dispatcher just orchestrates

### Design Trade-offs

**What we gained:**
- Automatic hook propagation to nested invokes
- No dispatcher parameter pollution
- Natural scoping via `with` statements
- Thread and async safety

**What we gave up:**
- Per-event-type hook registration (now all hooks see all events)
- Explicit hook ordering control (order is implicit via nesting)
- Multiple dispatchers (now singleton - all state in context)

The trade-offs were worth it for the cleaner API and better composability.

## Key Concepts

### Events
Discrete points in the agent's execution lifecycle where hooks can observe and influence behavior.

Each of the three spans (invoke, LLM query, REPL execution) fires the same four-stage
pipeline of value states (`span_event_lifecycle.md`):

```
proposal ──[edits]──▸ committed input ──[work or supply]──▸ raw result ──[modify]──▸ outcome

*Enter     observes the proposal          controls edits + abort      fires always
*Send      observes the committed input   controls supply + abort     fires iff the input committed
*Complete  observes the raw result        controls modify + abort     fires iff a raw result exists
*Exit      observes the outcome union     controls nothing (terminal) fires whenever the span opened*
```

Events mark **value states becoming fixed**, not phases of the machinery: an event's payload
is the commit of the previous stage's effects, so no event can observe its own composition's
output — only the next event can carry it. `*Send` therefore fires on the supplied/replayed
path too (the input commit is real even when no provider is called), and `*Exit` carries an
tagged outcome union (`Completed[...] | Aborted | Failed` — the variant IS the span status,
its payload the **post-transform** result or the terminating exception) — a span that opened
closes even when an exception unwinds through it (#892): `outcome=Aborted(exc)` when the
span's own invoke's `Abort` ended it, `outcome=Failed(exc)` for every other unwind.
(\*"Whenever the span opened" holds on every *execution* path — normal, abort, unwind.
The one non-firing state is a framework programming error: a span body that exits cleanly
without calling `complete()` trips a loud ERROR log and no exit, deliberately — see
`_log_incomplete_span` in the dispatcher.)

Anchor events:
- `LLMQueryEnter` - Before making an LLM API call (the always-present per-turn boundary)
- `REPLExecEnter` - Before executing parsed REPL code (conditional — only on runnable code)
- `InvokeEnter` - When `jaz.invoke()` is called

**Events are immutable — treat them as completely read-only.** An `Event` is a frozen dataclass (rebinding a field raises `FrozenInstanceError`), `inputs`/`scope` are read-only mappings, and `hooks` is an immutable tuple. The **only** way a hook influences execution — or communicates with another hook — is by **returning effects**. The same event object is dispatched to every hook, so mutating it would let an earlier-ordered hook silently change what a later-ordered hook sees at the same event; effects avoid this because the dispatcher applies them only *after* the whole hook loop, so one hook's output is invisible to the others at the same event. (Some referenced objects — `config`, the live `hooks` — are deliberately live so you can *read* real runtime state; reading is fine, mutating through them is unsupported.)

### Effects
Typed outputs from hooks that express how they want to influence execution. Effects are order-independent and composed by the dispatcher.

Examples:
- `Abort` - Abort the invoke: its carried exception raises out of `jaz.invoke()` and the invoke's spans close `Aborted` (loop/budget hard-stops). Valid at **every control event** (`*Enter`/`*Send`/`*Complete`; the `*Exit` events are observation-only).
- `SupplyExecResult` - Supply an `ExecResult` at `REPLExecSend`, skipping execution.
- `ModifyExecResult` - Transform the raw result at `REPLExecComplete` / `InvokeComplete` (e.g. budget-forcing a refusal).
- `AddMessages` - Add message(s) (e.g. instruction text) to the query at `LLMQueryEnter`
- `SupplyLLMResponse` - Supply a pre-computed LLM response at `LLMQuerySend`, skipping the API call

The two exec-result effects each carry a full `ExecResult`: `SupplyExecResult` *supplies* one at
`REPLExecSend` (execution is skipped; multiple **fold** among themselves), `ModifyExecResult`
*transforms* the raw result at the `*Complete` boundaries (multiple **fold** — carried `Continue`s
concat output + group exceptions). Suppliers live at `*Send` and transformers at `*Complete`
because each must see the value state its decision is about: a supplier decides on the committed
input (at `*Enter`, composition may still rewrite it), a transformer on the raw result. `Abort` is
*termination* — un-bundled from these (#481) — and is valid at **every
control event except `LLMQueryRetry`**, so loop/budget control has an always-present home. Emitting an
effect at an event that does not accept it raises `InvalidEffectError` — out-of-stage
effects are hook bugs and fail loudly.

At `LLMQueryComplete`, `Abort` is the *only* valid effect. The query has already completed and been
paid for, so an abort there does not un-do it — it stops the turn before the agent acts on the
response, meaning the code the model just proposed is never executed. That last part is the
whole difference from deferring to the next `LLMQueryEnter`, and it is *not* a saving in spend:
an abort at the next enter fires before the query, so no extra call is paid either way. What the
deferral costs is an **execution** — the code from the turn that prompted the decision runs in
between.

**Conditional vs unconditional events (a contract every enforcement hook must
know):** a span that opened always closes with its `*Exit`, on every path — an in-flight
exception fires `Exit(outcome=Failed(exc))` on the unwind (#892), and an `Abort` resolved
at any of the span's control stages fires `Exit(outcome=Aborted(exc))` before its carried
exception propagates. A pairing observer (e.g. a tracing span opened at `LLMQueryEnter`)
can therefore close unconditionally at its `*Exit`. Consumers that merely *read* `*Exit`
payloads (budget/cost, history, loggers) **must match
on `event.outcome`**: the payload lives on the `Completed` variant (`event.outcome.result`),
and the non-completed variants carry the terminating error (`event.outcome.exception`) —
there are no parallel payload fields. Child spans are
**not** guaranteed to open at all — REPL *execution* events only fire
when the LLM response parses to runnable code; a turn whose response fails to parse never
opens an execution span. So never use
execution events as a proxy for "once per turn" — a hard stop hung there silently never
fires for perpetually-unparseable output. Rule of thumb: **liveness / per-turn
enforcement → `Abort` at `LLMQueryEnter`** (the always-present per-turn boundary — it fires
unconditionally once per turn, before the query); **result-composed nudges → execution
*complete*** (the transform boundary — they intentionally compose with what the code
actually did, and the subsequent `*Exit` carries the post-transform result). This is how
`IterationLimit` and `BudgetPool` place their hard stops vs soft force-finish nudges.

### Contexts
Typed contracts between the dispatcher and agent containing composed effects. Each event type has a specific context type.

Examples:
- `REPLExecContext` - REPL execution *enter*: namespace edits (AddVariables/DropVariables) or abort (Abort)
- `REPLExecSendContext` - REPL execution *send*: supply a result (SupplyExecResult) or abort
- `REPLExecCompleteContext` - REPL execution *complete*: transform the result (ModifyExecResult) or abort
- `LLMQueryContext` - LLM query *enter*: prompt edits (AddMessages/DropMessages) or abort
- `LLMQuerySendContext` - LLM query *send*: override the response (SupplyLLMResponse) or abort
- `LLMQueryCompleteContext` - LLM query *complete*: abort only
- `InvokeContext` - Invoke *enter*: add/drop invoke inputs, disable recursion, or abort (Abort)
- `InvokeSendContext` - Invoke *send*: abort only (no invoke supplier effect exists)
- `InvokeCompleteContext` - Invoke *complete*: transform the terminal result (ModifyExecResult) or abort

### Hooks
Extension points that process events and return effects. Hooks inherit from the `Hook` base class and support the context manager protocol.

```python
from jaz.hooks import Hook, Effect, AddMessages
from jaz.hooks.events import LLMQueryEnter

class MyHook(Hook):
    # Override a typed per-event handler for just the events you care about — typed
    # attribute access, no isinstance/match, and unimplemented events default to no-op.
    def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Effect]:
        # Prompt text is added as an AddMessages at LLMQueryEnter (the context
        # where message edits compose).
        return [
            AddMessages(
                messages=[{"role": "user", "content": "Be concise."}],
                index=len(event.messages),
            )
        ]
```

A cross-cutting observer that treats **every** event uniformly (logging, tracing) overrides
`on_any(event)` instead — the dispatcher calls it *in addition to* the matched typed handler,
so it composes with (never replaces) the per-event dispatch, and one hook may use both.
Don't override `_dispatch_event`: it's the framework's private internal event router
(#597, renamed from `on_event` in #740 so the router no longer looks like an `on_<event>`
override point). Override the typed `on_<event>` handlers or `on_any` instead.

### What a hook shows in a trace (`__repr__`)

Observability consumers — `FileLogger`, `PrintLogger`, `ATIFTrace`, `OTelTracing` — render the
active hook set at `InvokeEnter` by calling `repr()` on each hook. That string is the whole
record of what governance applied, so a hook whose configuration constrains the run should say
so:

```python
class MyHook(Hook):
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def __repr__(self) -> str:
        return f"MyHook(threshold={self.threshold!r})"
```

The base `Hook.__repr__` returns `MyHook()` — the class name and nothing else. That is
deliberately minimal but *stable*: `object.__repr__` would embed the instance's memory address,
which is noise in a trace and differs between runs, defeating any diff of two traces. Override
when the configuration is worth recording.

**A dataclass hook needs nothing.** `@dataclass` generates a `__repr__` listing the fields,
which wins over the base by MRO:

```python
@dataclass(eq=False)          # eq=False: hooks are identity objects (deduped by `is`, not value)
class MyHook(Hook):
    threshold: float = 0.5
                              # repr -> MyHook(threshold=0.5)
```

Guidance for what to include:

- **Frozen construction params, not runtime state.** Two identically-configured hooks should
  render identically; a per-invoke counter in the repr makes the record depend on when it was
  taken. On a dataclass this needs `field(init=False, repr=False)` — `init=False` alone keeps
  a field out of the constructor but *not* out of the generated repr.
- **Render the value; the bar for hiding one is high.** Collapsing a field to a flag loses
  information, so it needs a reason beyond "this could be ugly". A bare callable rendering as
  `<function w at 0x102e523e0>` is *not* sufficient on its own — a field whose common values read
  fine as-is (`None`, a plain string, a named function) should show them. (The governance hooks
  sidestep the question now: `IterationLimit` / `ContextWindowWarning` / `BudgetPool` take a plain
  `warning_text: str | None`, not a callable.) The one field in-tree that is hidden is
  `Compaction.summary_prompt`, and only when it holds the
  default: a 440-character constant otherwise repeated in every record. It is *shown* when the
  caller overrode it. A repr whose field list varies by instance is fine.
- **Don't count on the consumer to truncate.** The loggers do (`abbrev_repr(...,
  max_field_length)`, default 1000), but `ATIFTrace` and `OTelTracing` write the string whole. A
  value that is large *and* near-constant is worth suppressing at the source; a merely long one
  that carries real information is not.
- **No JSON-safety requirement.** A repr is a string, so a value the old serializer had to encode
  or omit can simply be described.
- **No module qualifier.** The old serialized record carried a `qualified_name`; a repr does not,
  so two same-named hooks from different modules are indistinguishable in a trace. Rare enough
  not to pay for globally; a hook that expects to collide should qualify itself in its override.

**There is no hook serialization.** `Hook.to_dict`/`Hook.from_dict`, the `_to_dict_params`
escape hatch and the qualified-name subclass registry were removed (YAGNI): nothing in the
codebase ever reconstructed a hook from a dict, the only consumers were the four write-only
observability sinks above, and maintaining a round-trip contract — JSON-safety rules, per-field
encoder/decoder pairs, a registry that resolves without importing by name — cost more than the
debug string it actually produced. `repr()` is that debug string, honestly labelled.

### HookDispatcher
Stateless singleton that orchestrates hooks by:
1. Reading active hooks from the current context variable
2. Emitting events to all active hooks
3. Collecting effects from all hooks
4. Applying composition rules to create typed ExecutionContext

## Context-Based Hook Management

### Basic Usage

Hooks are activated using Python's `with` statement:

```python
from jaz import ReturnType, invoke
from jaz.hooks import PrintLogger, BudgetPool
import logging

# Simple hook activation
with PrintLogger(level=logging.INFO):
    result = invoke(ReturnType(int), task="Calculate 2+2")
    # PrintLogger is active during this invoke

# Nested hooks - both active
with PrintLogger(level=logging.INFO):
    with BudgetPool(cost_budget=1.0, calls_budget=50):
        result = invoke(ReturnType(str), task="Complex task")
        # Both PrintLogger and BudgetPool are active
```

### Automatic Propagation

Hooks automatically propagate to nested `invoke()` calls via `contextvars`:

```python
from jaz import ReturnType, invoke
from jaz.hooks import PrintLogger

with PrintLogger(level=logging.INFO):
    # This invoke has PrintLogger
    result = invoke(
        ReturnType(int),
        task="""
        Call invoke again with prompt: 'What is 5 * 5?'
        """,
    )
    # The nested invoke() call also has PrintLogger automatically!
```

This works across any level of nesting - hooks propagate through:
- Nested `invoke()` calls
- Function calls within the hook context
- Async functions (contextvars are async-safe)
- Threads started *inside* the hook context (each inherits a copy)

> **Thread pool caveat**: `ThreadPoolExecutor` workers do **not** inherit the
> calling thread's hooks — Python's `contextvars` only propagates to threads
> that are *started* inside the active context, not to pre-existing pool
> workers.
>
> Inside a **`jaz.invoke` sub-invoke** run in such a worker this is largely
> mitigated (#727): the ancestor hook chain is closure-threaded on the prehook and
> **re-based** onto the worker — composed *under* any hooks the worker activates
> locally (`ancestor + local`, so both fire), rather than the old either/or that
> dropped the ancestor whenever the worker had its own hook. What is **not**
> recovered automatically is a hook the agent activated **mid-REPL** and then a raw
> pool worker: `prehook.parent_hooks` was snapshotted before the REPL ran, and
> contextvars can't cross the pool boundary. To carry the full live context
> (including mid-REPL `with Hook()` activations) into a worker, wrap the call in
> `contextvars.copy_context()`:
>
> ```python
> import contextvars
> from concurrent.futures import ThreadPoolExecutor
>
> ctx = contextvars.copy_context()  # snapshots the live hook + config context
> with ThreadPoolExecutor() as ex:
>     ex.submit(ctx.run, lambda: jaz.invoke(jaz.ReturnType(str), task="subtask")).result()
> ```
>
> `copy_context()` carries the *established* context across, so the worker sees the
> full parent state and no re-base is needed. Outside a sub-invoke (a bare hook, no
> `jaz.invoke`), the caveat is unchanged — activate the hook *inside* the worker or
> use `copy_context()`.

### Default Global Context

The default hook context is **empty**. Hooks are opt-in: nothing runs until
you activate a hook via `with HookClass():`.

Future essential built-in behaviors (budget enforcement, permissions, etc.)
will be auto-installed by `jaz.invoke()`. Each such built-in will ship with
its own dedicated, behavior-specific opt-out (e.g., a hypothetical
`bypass_budget()`) — **not** a generic disable/clear mechanism.

There is deliberately **no** generic way to disable or clear hooks — neither a
class-keyed `disable_hook(SomeClass)` nor a blanket `clear_all_hooks()` (#466
removed both). Hook lifecycle is managed purely by lexical `with` scoping:

1. Multiple instances of the same `Hook` class can be active simultaneously
   (e.g., two `PrintLogger`s at different levels). A class-keyed disable
   conflates class with instance identity and silently kills all of them.
2. A blanket clear is a footgun: it silently strips governance / safety
   baseline hooks (e.g. the loop's only termination guarantee), and its effect
   can't even be honored in raw worker threads (they restore the ancestor
   context) — a false sense of a clean slate.
3. Named, behavior-specific opt-outs document intent at the call site and are
   far easier to audit than a generic kill switch.

### Helper Functions

```python
# Activate multiple hooks with native Python syntax
with Hook1(), Hook2(), Hook3():
    invoke(...)  # All three hooks active
```

## Architecture

### Context Variable Flow

```
┌─────────────────────────────────────────────────────────┐
│ Global Default Context                                   │
│ hooks: ()  # empty — hooks are opt-in                    │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
           ┌─────────────────────────────┐
           │ with MyHook():              │
           │ hooks: (MyHook,)            │
           └─────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────────────────────────┐
              │ invoke() calls                           │
              │ HookDispatcher reads current context     │
              │ and emits to active hooks                │
              └──────────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Nested invoke()      │
              │ Same hooks active!   │
              └──────────────────────┘
```

### Event Flow

```
┌─────────────┐
│   Agent     │
│             │
│  Emits ───────────┐
│  Events     │     │
└─────────────┘     │
                    ▼
              ┌────────────────────┐
              │  HookDispatcher    │ ◄── Reads hooks from
              │  (Singleton)       │     context variable
              │                    │
              │  emit() ───────────┼───► Hook 1 ───► [Effect, Effect]
              │                    │
              │                    ┼───► Hook 2 ───► [Effect]
              │                    │
              │                    │◄─── Hook 3 ───► [Effect, Effect]
              └────────────────────┘
                    │
                    │ Composition Rules:
                    │ • Abort → raises its carried error (supersedes ModifyExecResult)
                    │ • ModifyExecResult → transforms the ExecResult
                    │ • ALL prompts → concatenate
                    │ • MIN iteration limit
                    │ • ALL metrics → record
                    │
                    ▼
          ┌──────────────────────┐
          │ ExecutionContext     │
          │ (Typed)              │
          │                      │
          │ • error_effects: list│
          │ • raise_effects: list│
          │ • instruction_prompt │
          │ • max_iterations...  │
          └──────────────────────┘
                    │
                    ▼
              ┌─────────────┐
              │   Agent     │
              │  Uses       │
              │  Context    │
              └─────────────┘
```

## Usage Examples

### 1. Create a Hook

```python
from jaz.hooks import (
    Hook, Event, Effect, LLMQueryEnter, Abort, AddMessages,
)

class WrapUpHook(Hook):
    """Example custom hook: nudge the agent to finish after `soft` REPL
    iterations, then hard-stop after `hard`.

    Illustrates the two effect kinds you'll reach for most — a soft nudge via
    AddMessages (prompt modification, emitted at LLMQueryEnter) and a terminal abort
    via Abort — plus the fact that a hook may hold its own state across events.

    Both live on ``LLMQueryEnter``, the always-present per-turn boundary: the nudge
    because that's the only context where message edits compose, and the hard stop because
    ``Abort`` there fires once per turn *before* the query (so it never wastes a call and
    never silently skips a parse-failure turn the way a conditional execution event would).

    Note ``_iterations`` here is **tree-wide**: one hook instance propagates to
    nested ``invoke()`` calls via contextvars, so the count spans the whole
    invoke tree rather than each invoke separately. For per-invoke state, key your
    tracking by ``event.invoke_id`` — which is exactly why the built-in
    BudgetPool keys its trackers by invoke_id. For real LLM cost /
    call-count budgets, use that built-in rather than rolling your own.
    """

    def __init__(self, soft: int = 5, hard: int = 8):
        self.soft = soft
        self.hard = hard
        self._iterations = 0

    def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Effect]:
        # LLMQueryEnter fires once per turn, before the query — count here.
        self._iterations += 1

        # Once past `hard`, terminate the invoke: the Abort's carried error
        # propagates out of invoke() and the invoke's spans close Aborted.
        if self._iterations > self.hard:
            return [Abort(error=RuntimeError("Iteration limit exceeded"))]

        # Once past `soft`, nudge the agent to finish. The nudge is an AddMessages
        # appended to the end of this query.
        if self._iterations > self.soft:
            return [
                AddMessages(
                    messages=[{"role": "user", "content": "Wrap up and RETURN soon."}],
                    index=len(event.messages),
                )
            ]

        return []
```

### 2. Use Hook as Context Manager

```python
from jaz import ReturnType, invoke

# Activate hook for specific scope
with WrapUpHook():
    result = invoke(ReturnType(str), task="Expensive task")
    # WrapUpHook is active during this invoke
```

### 3. Combine Multiple Hooks

```python
from jaz.hooks import PrintLogger, BudgetPool
import logging

# All three hooks active; they compose and propagate together
with PrintLogger(level=logging.INFO):
    with BudgetPool(cost_budget=1.0, calls_budget=50):
        with WrapUpHook():
            result = invoke(ReturnType(str), task="Task with all protections")
```

### 4. Conditional Hook Activation

```python
def my_function(enable_logging: bool = True):
    if enable_logging:
        with PrintLogger(level=logging.INFO):
            return invoke(ReturnType(str), task="task")
    else:
        return invoke(ReturnType(str), task="silent task")  # No logging hook
```

### 5. Use in Agent (Internal)

```python
from jaz.hooks import get_dispatcher

# In Agent.do_one_repl_iteration()
dispatcher = get_dispatcher()

with dispatcher.span_repl_exec(enter_event) as span:
    # ...apply span.ctx namespace deltas to the REPL...
    span.send()  # fires REPLExecSend; suppliers compose here (an Abort raises)
    if span.supplied is not None:
        # A Send-composed SupplyExecResult short-circuits execution.
        exec_result = span.supplied
    else:
        # Execute REPL code (REPL inputs are injected once at InvokeEnter, not here —
        # AddInputs is InvokeEnter-only, #481)
        exec_result = repl.exec(code, ...)

    # Complete the span; REPLExecComplete fires as the block unwinds, then REPLExecExit
    # carries the post-transform result.
    span.complete(exec_result=exec_result)

# Apply any Complete-time ModifyExecResult transform.
exec_result = span.get_final_exec_result()
```

## Composition Rules

The dispatcher applies explicit composition rules when combining effects from multiple hooks:

### Exec-result / abort effects (`SupplyExecResult` / `ModifyExecResult` / `Abort`)
`SupplyExecResult` is valid only at `REPLExecSend` (supply, with the committed input in view),
`ModifyExecResult` only at the `*Complete` transform boundaries (`REPLExecComplete` /
`InvokeComplete`); `Abort` is valid at every control event. Where they co-occur they compose into
the REPL result types (`Continue` / `Return` / `Raise`):

- **Send (`SupplyExecResult`)**: execution is skipped and the supplied result is used. Multiple
  `SupplyExecResult`s **fold** among themselves (no `exec_result` to fold onto yet) by the same
  precedence as the transform boundary — carried `Continue`s concatenate their `output` and group
  their exceptions, so two hooks vetoing the same input (e.g. a `ValidateREPLInput` and the evals
  `RestrictReturnValue`) compose into one recoverable rejection; distinct carried `Return` *values*
  still raise `ReturnValueConflictError`. `Abort` supersedes them.
- **Complete (`ModifyExecResult`)**: the composed result supersedes the raw `exec_result`; multiple
  **fold** by carried-result kind precedence `Raise` > `Return` > `Continue`. Carried `Continue`s
  concatenate their `output` onto the original's and group their exceptions into an
  `ExceptionGroup`; carried `Return`s cannot merge (two distinct values raise
  `ReturnValueConflictError`, identical ones coalesce); carried `Raise`s group their exceptions. The original's
  exception is folded in only when the outcome is of the same final type; the original `output` is
  preserved. The subsequent `*Exit` carries the post-transform result (#906).

Both boundaries share one fold (`_fold_carried_results`); a supply passes `original=None`, a
transform passes the produced result.
- **`Abort` supersedes** the exec-result effects at either boundary (termination trumps
  supply/transform): the span CM raises the carried exception *before* the fold runs, and the
  span closes `Aborted`. Multiple `Abort`s group their exceptions into an `ExceptionGroup`.

### Prompt Modifications
- **Rule**: prompt text is added via `AddMessages` at `LLMQueryEnter`; all adds compose into the query through the message-edit fold (`apply_message_edits`)
- **Order**: deterministic and hook-order-independent — same-slot adds are ordered by `(sort_key, canonical content)`
- **Example**: Budget warning + safety note both appear in the query

## Type Safety

The dispatcher uses typed contexts to prevent invalid operations:

```python
# Type checker knows ctx is REPLExecContext (the *enter* context)!
with dispatcher.span_repl_exec(...) as span:
    span.ctx.added_variables  # ✓ Exists
    span.ctx.message_edits  # ✗ Type error - doesn't exist on REPLExecContext
    # (supply effects live on REPLExecSendContext, composed when span.send() fires)

# Type checker knows ctx is LLMQueryContext (the *enter* context)!
with dispatcher.span_llm_query(...) as span:
    span.ctx.message_edits  # ✓ Exists
    span.ctx.added_variables  # ✗ Type error - doesn't exist on LLMQueryContext
    # (a supplied response lands on span.supplied_response after span.send(...))
```

Read-only contexts prevent invalid operations:

```python
# A SupplyExecResult / ModifyExecResult emitted at an event that can't compose it
# (e.g. LLMQueryEnter) raises InvalidEffectError — but Abort is
# valid there and terminates the invoke.
```

## Hook Coordination

For complex scenarios where hooks need to coordinate, create a "meta-hook" that makes coordinated decisions internally:

```python
class AdaptiveBudgetHook(Hook):
    """Single hook that coordinates a budget warning + abort decision."""

    def __init__(self, cost_tracker: CostTracker, budget: float):
        # CostTracker is a budget-less accounting node now; the meta-hook holds its
        # own budget (mirrors how BudgetTrackingHook owns the one pool budget).
        self.cost_tracker = cost_tracker
        self.budget = budget

    def _utilization(self) -> float:
        return self.cost_tracker.total_llm_cost / self.budget

    def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Effect]:
        # Both decisions live on LLMQueryEnter (once per turn, before the query).
        # Check the hard stop first so an exhausted budget terminates rather than
        # merely warning.
        if self._utilization() >= 1.0:
            # Abort is valid at every control event; its carried error propagates.
            return [Abort(error=BudgetExhaustedError())]

        # Otherwise, warn the agent to wrap up (prompt modification, composed at the query).
        if self._utilization() > 0.8:
            return [AddMessages(
                messages=[{
                    "role": "user",
                    "content": f"Budget at {self._utilization():.1%}; finish and RETURN soon.",
                }],
                index=len(event.messages),
            )]

        return []
```

This is cleaner than having two separate hooks that might conflict.

## Advanced Patterns

### Context-Aware Hook

Hook behavior that changes based on context:

```python
class AdaptiveLogger(Hook):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def on_any(self, event: Event) -> list[Effect]:
        # Prompt additions only compose at LLMQueryEnter; add more detail when verbose.
        if isinstance(event, LLMQueryEnter):
            note = "verbose context note" if self.verbose else "brief context note"
            return [
                AddMessages(
                    messages=[{"role": "user", "content": note}],
                    index=len(event.messages),
                )
            ]
        return []

# Use different verbosity in different scopes
with AdaptiveLogger(verbose=False):
    invoke(ReturnType(str), task="production task")

with AdaptiveLogger(verbose=True):
    invoke(ReturnType(str), task="debug task")
```

### Conditional Hook Wrapper

```python
from contextlib import contextmanager

@contextmanager
def maybe_hook(hook: Hook, enabled: bool):
    """Conditionally activate a hook."""
    if enabled:
        with hook:
            yield
    else:
        yield

# Use it
with maybe_hook(PrintLogger(level=logging.DEBUG), enable_debug):
    invoke(ReturnType(str), task="task")
```

### Hook Stack Inspection

```python
from jaz.hooks.context import get_current_hooks

def print_active_hooks():
    """Debug utility to see what hooks are active."""
    hook_ctx = get_current_hooks()
    print(f"Active hooks: {[type(h).__name__ for h in hook_ctx.hooks]}")
```

## Testing Hooks

The context-based design makes hooks easy to test in isolation:

```python
from jaz.hooks import get_dispatcher

def test_my_hook():
    """Test hook in isolation."""
    # The default context is empty, so MyHook is the only active hook.
    with MyHook():
        dispatcher = get_dispatcher()

        # Emit test event. Every Event carries the invoke's effective `config`
        # (the dispatcher resolves baseline_hooks from it — see #463); in a test,
        # `get_config()` is the convenient default.
        event = LLMQueryEnter(
            config=get_config(),
            invoke_id="test",
            messages=[],
            model="test-model",
            iteration=1,
            depth=0,
            can_recurse=True,
        )
        effects = dispatcher.emit(event)

        # Assert expected effects (a prompt addition is an AddMessages at LLMQueryEnter).
        assert len(effects) == 1
        assert isinstance(effects[0], AddMessages)
```

## Extension Points

To add new capabilities to the hook system:

### 1. New Event Type

```python
# events/my_event.py
@dataclass
class MyCustomEvent(Event):
    """My custom event. Inherits the required `config` field from Event (#463),
    plus `timestamp` — the emission stamp emit() sets on every event. Do NOT
    redeclare a field named `timestamp`: emit() overwrites it."""
    data: str

# dispatcher.py - add span method. The Agent passes the invoke's effective config
# into the enter event (self.config); the dispatcher copies it onto the exit event.
@contextmanager
def span_my_custom(self, config: "Config", data: str):
    enter_event = MyCustomEvent(config=config, data=data)
    effects = self.emit(enter_event)
    ctx = self._compose_my_custom(effects)
    span = Span(ctx=ctx)
    try:
        yield span
    finally:
        if span.is_completed():
            exit_event = MyCustomEventExit(...)
            self.emit(exit_event)
```

### 2. New Effect Type

```python
# effects.py
@dataclass
class MyCustomEffect(Effect):
    """My custom effect."""
    action: str
    data: dict

# dispatcher.py - handle in composition method
def _compose_my_custom(self, effects: list[Effect]) -> MyCustomContext:
    ctx = MyCustomContext()
    for effect in effects:
        match effect:
            case MyCustomEffect(action=action, data=data):
                ctx.actions.append((action, data))
    return ctx
```

### 3. New Hook

Just implement the `Hook` base class:

```python
class MyCustomHook(Hook):
    def on_any(self, event: Event) -> list[Effect]:
        if isinstance(event, MyCustomEvent):
            return [MyCustomEffect(action="process", data={})]
        return []
```

## Migration from Old API

If you have code using the old dispatcher API:

```python
# OLD WAY (deprecated)
dispatcher = HookDispatcher()
dispatcher.add_hook(LLMQueryEnter, MyHook())
agent = Agent(repls=["python"], dispatcher=dispatcher)
result = agent.invoke(...)

# NEW WAY
with MyHook():
    result = invoke(...)  # MyHook automatically active
```

The old API methods (`add_hook`, `add_hook_for_all`, `dispatcher` parameter) have been removed.

## FAQ

**Q: Why can't I register hooks for specific event types anymore?**

A: Hooks now see all events and filter internally using `isinstance()`. This simplifies the API and makes hook behavior more explicit.

**Q: How do I control hook ordering?**

A: Hooks are called in registration order (outer `with` to inner `with`). If you need strict ordering, use a single meta-hook that coordinates internally.

**Q: What if I need different hooks for different invoke calls?**

A: Use nested `with` statements to change the active hooks:

```python
with Hook1():
    invoke(ReturnType(str), task="task 1")  # Has Hook1

with Hook2():
    invoke(ReturnType(str), task="task 2")  # Has Hook2
```

**Q: Are hooks thread-safe?**

A: Yes! `contextvars` automatically creates per-thread contexts, so each thread has its own set of active hooks.

**Q: Can I use hooks with async code?**

A: Yes! `contextvars` are async-safe, so hooks work correctly across `await` boundaries.

**Q: How do I see what hooks are currently active?**

A: Use `get_current_hooks()` from `jaz.hooks.context`:

```python
from jaz.hooks.context import get_current_hooks

hook_ctx = get_current_hooks()
print([type(h).__name__ for h in hook_ctx.hooks])
```
