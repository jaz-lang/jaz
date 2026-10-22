# Changelog

All notable user-facing changes to JAZ. Versions follow the distribution on PyPI
(`jaz-lang`); the import name is `jaz`.

## 0.2.0b1 — 2026-10-21

### Breaking changes

**Public surface**

- **Every non-public module is underscore-prefixed** (`jaz._config`, `jaz.repl._python_repl`,
  `jaz.hooks._builtin`, ...). The forwarding shim modules and the deprecated top-level aliases
  are removed, and so is `NonPublicAPIWarning`. Import from the public modules.
- **`jaz.Config` is public and complete.** It requires `llm=`, is frozen, and
  `jaz.get_config()` raises `ValueError` when no LLM is configured. Per-depth configuration
  (`configure_by_depth`, `ConfigOverrideByDepth`) is removed; nest `ConfigOverride` scopes.
- **`UNSET` and `UnsetType` are exported from `jaz`.**
- **The experimental description system is removed:** `jaz.describe`, `DescriptionOverride`,
  `__desc__`, the `desc_*` constructors and its two warnings, and `jaz.get_default_description`.
  To control what the agent sees for an input, write the docstring of a function, class, module
  or callable object, define `__str__` on any other object, and define `__repr__` on one that
  goes inside a list, dict or dataclass. `jaz.invoke`'s docstring has the full rule.
- **Removed:** `WorkflowReplay`; `parent_print`; the bundled model-price table; the shared
  `HookDispatcher` singleton; the `jaz.budget`, `jaz.llm.extras_merge` and other unused modules.

**LLM backends**

- **`BaseLLM`'s contract is `complete(messages) -> LLMResponse` plus `model_info`.** The base
  has no `model`, per-call keyword arguments, `request_defaults` or bundled pricing.
  `get_model`, `can_report_cost` and `get_model_info` are replaced by the `model_info` property
  (`ModelInfo(name, max_input_tokens, can_report_cost)`); `finalize` and `from_dict` are
  removed, and `complete_with_retry` / `acomplete_with_retry` are private. Backends take
  `model=` themselves. `consumes_internal_keys` is `extra_message_keys`.
- **Retries are one `retry_policy: RetryPolicy`.** A call is retried only on the backend's
  `retryable_exceptions` (by default `RateLimitError`, `ServerError` and `APIConnectionError`),
  plus `RetryPolicy.extra_retryable_exceptions`; `non_retryable_exceptions` is gone. `limits`
  is keyed by error class and every count is a cap.
- **`OpenAILLM` and `AnthropicLLM` are removed**; use `LiteLLM`. `CopilotLLM` is
  `GitHubCopilotLLM`. Backends no longer expose their settings as attributes or `close()`.
- **LLM error classes follow the provider SDKs.** `APIError` is `ServerError`, the generic 4xx
  class is `BadRequestError`, and `UnsupportedParamsError`, `ContextWindowExceededError` and
  `ContentPolicyViolationError` subclass `BadRequestError`. A class now names what has to
  change: a rate limit clears by retrying and a quota does not (`QuotaExceededError`), a code
  naming a cause wins over the status, an unregistered 4xx or a redirect fails at once.
  `LLMError` no longer takes `llm_provider`.
- **`LiteLLM` raises jaz's errors for every LiteLLM error**, classified by the provider's
  status, with LiteLLM's exception as `__cause__`. A bug inside LiteLLM passes through and is
  not retried. It raises `ContentPolicyViolationError` for a refusal or content filter on a
  successful response, reports usage and cost only when the provider gave them, and refuses
  `stream=True` and LiteLLM's own `num_retries`.
- **jaz requires `litellm[mcp]>=1.105.0`.**
- **`GitHubCopilotLLM`** does not retry an error the Copilot CLI already retried, and reports an
  unknown cost (`None`) when GitHub's credit figure is missing.

**REPL, protocol and prompts**

- **The `__history__` record belongs to the REPL.** `BaseREPL` gains the abstract
  `repl_history_description` and `update_repl_history`, and the protocol's history methods are
  gone.
- **`BaseREPL` is generic over its state** (`BaseREPL[StateT]`); `REPLState` is removed and
  `PythonREPLState` is public. `description` is an abstract property and
  `get_namespace(state)` a new abstract method; `add_variables` / `drop_variables` are
  `add_to_namespace` / `drop_from_namespace`; `add_inputs`, `get_finish_command_hint`,
  `from_dict` and `construction_keys` are removed. `initialize` takes `subinvoke` (was
  `invoke_tool`) and `repl_history_withheld` (in place of `allowed_builtins` and
  `initial_repl_history`), and `exec` / `aexec` no longer take `exec_timeout_override`.
- **The REPL namespace effects are `AddToNamespace` / `DropFromNamespace`.** A hook's namespace
  view is the agent's `globals()`, with `invoke` and `__history__` inside `__builtins__`;
  dropping one of those raises `MissingDropTargetError`. Withhold them with `DisableRecursion`
  and `WithholdREPLHistory`.
- **`BaseProtocol`:** `format_instructions()` is the `response_format_description` property,
  `parse` returns the code as a `str` (`ParsedCode` is removed), `render_observation` takes no
  `iteration`, and `render_initial_message_list` takes `subinvoke` and `repl_history_withheld`
  and no longer takes `depth`, `recursion_available` or `input_display_overrides`.
  `CodeOnlyProtocol` no longer takes `truncation_prefix_ratio`.
- **Prompt-template variables are renamed:** `response_format_description`, `repl_description`,
  `repl_history_description` and `subinvoke_description`, and `inputs` / `scoped_inputs` replace
  the block and name lists. `depth`, `show_depth` and `recursion_available` are gone. In the
  REPL description, `allowed_imports` and the path allow-lists are objects with `.patterns` and
  `.allow_all`, and `allows_everything` is not passed. No shipped template defines blocks. A
  description is a non-empty string or `None`.
- **Inputs render through their own `str()`.** An object that defines only `__repr__` now
  renders as its repr rather than its API, a class shows its constructor, and an input whose
  rendering raises fails the invoke with `InputRenderError`.
- **A no-argument reader is a property; one that takes an argument is `get_*`.** Overriding such
  a property with a plain method raises `TypeError` at class definition.

**Hooks and events**

- **Hook write declarations are reservations:** `reserves_llm_response`,
  `reserved_input_names`, `reserved_namespace_names`, `reserved_blackboard_keys`, and new
  `reserved_llm_response_fields`, `reserves_exec_return`, `reserves_invoke_return`. The old
  names (`supplies_llm_response`, `adds_inputs`, `adds_variables`, `writes_blackboard`) are
  silently ignored. `blackboard_consumes` is removed.
- **Event fields:** `LLMQuerySend.edits` is `message_edits`; `InvokeSend.added_inputs` /
  `dropped_inputs` are `input_edits`; `REPLExecSend.added_variables` / `dropped_variables` are
  `namespace_edits`; `model`, `can_recurse`, `can_report_cost` and `invoke_tool` are gone from
  the events; `event.blackboard` is a read-only `Mapping`. `LLMQuerySend.messages` is the
  stamped buffer view, not the wire form.
- **Effects:** `DropInputs.keys` is `names` and drops inputs only; `AddMessages.sort_key` and
  `InsertCode.sort_key` are removed, so same-slot adds apply in hook order.
- **Rollout export is `Rollout.to_spans() -> list[Sample]`**; `to_flat`, `to_pieces`,
  `to_turn_samples`, `FlatSample`, `TurnSample` and `jaz.exceptions.NonMonotoneRolloutError` are
  removed, and `Rollout` and `TurnRecord` drop their timestamp, model and iteration fields.
- **Tracing span and attribute names follow the events** (`invoke`, `repl_exec`, `llm_query`,
  `jaz.<field>`); `TrajectoryRecorder`'s `extra` block and `TrajectoryDirectoryRecorder` use the
  events' field names. `MetaData(task_name=...)` is `MetaData(label=...)`, and
  `TrajectoryReplay.replayed_count` is `replayed_response_count`.
- **`TrajectoryReplay(detect_divergence=True)` also compares each invoke's starting prompt.**

**Exceptions**

- **Renamed:** `BlackboardSeedError` → `BlackboardSeedConflictError`, `SandboxPermissionError` →
  `FileAccessDeniedError`, `SandboxKeyError` → `KeyEditDeniedError`.
- **`FileAccessDeniedError` is a `PermissionError`.** Every REPL policy refusal agent code can
  meet now raises the builtin Python raises for the ordinary failure.
- **`REPLInputConflictError` is no longer a `ValueError`.**
- **`PythonREPL(traceback_verbosity=...)`** accepts only the known values at type-check time.

### Added

- **`WithholdREPLHistory`** withholds `__history__` from an invoke, in the REPL and the prompt.
- **New LLM error classes:** `APITimeoutError` and `APIConnectionError` (also the builtin
  `TimeoutError` and `ConnectionError`), `QuotaExceededError`, `RequestAbortedError`, and the
  `ServerError` subclasses `InternalServerError`, `OverloadedError` and
  `APIResponseValidationError`.
- **`LLMError` carries the provider's answer:** `body`, `headers`, `code`, `request_id`,
  `retry_after` and `is_retryable`.
- **Retries follow the provider's `Retry-After` and `x-should-retry`, with jitter**, up to
  `RetryPolicy.retry_after_max`.
- **Event fields:** `scope` on every event; `REPLExecEnter.namespace` and
  `REPLExecSend.namespace`; `REPLExecSend.code_edits` and `namespace_edits`;
  `InvokeSend.input_edits`, `recursion_disabled` and `repl_history_withheld`.
- **Newly public:** `LLMResponse`, `MockLLM`, `GitHubCopilotLLM`, `SGLangLLM`, `ChatFormat`, `HFChatFormat`,
  `ModelInfo`, `RetryPolicy`, `TokenStamp`, `TurnRecord`, `Rollout`, `Sample`, `PythonREPLState`,
  `REPLHistoryEntry`, `TerminalExecResult`, `TracebackVerbosity`, the edit-record types in
  `jaz.hooks.events` (`ResolvedInputEdits`, `InputAddRecord`, ...), `CodeEditOffsetError` and
  `is_fatal`.
- **`LiteLLM` reports `finish_reason`** in `LLMResponse.extra`, and every shipped component's
  `repr` shows the call that built it, with secrets redacted.

### Fixed

- **Sandbox:** binding the name `__builtins__` in any statement is refused, closing an escape
  through an allowed module's `__builtins__`; an invoke input named `__builtins__`, or agent
  code replacing it through `globals()`, fails with `KeyEditDeniedError`.
- **Execution timeouts** swallowed by library code are enforced at the next blocking call, and
  an async `TimeoutError` raised by the agent's own code is no longer reported as the timeout.
- **Error classification** on `LiteLLM`, `GitHubCopilotLLM` and `SGLangLLM`: many provider
  errors that were retried in vain now fail at once, and transient ones that failed are retried,
  across Gemini, Vertex, Bedrock, watsonx, Together, OpenRouter, Azure and SGLang; `status_code`
  is the provider's own, or `None` when nothing came back.
- **Works with LiteLLM 1.106**, which needs the `mcp` package for any call with tools and for its
  Responses bridge (`gpt-5-pro`, the `-codex` models).
- `GitHubCopilotLLM` aborts a turn whose model calls a tool, cancels the message it stops waiting
  for, and no longer returns an empty reply for a lost error.
- `LiteLLM.model_info.can_report_cost` is `False` for a model LiteLLM does not price per token.
- Hook composition no longer crashes on values such as numpy arrays; a `DropFromNamespace` that
  raises removes nothing; a `KeyboardInterrupt` during teardown still runs the other hooks'
  teardowns.
- `TrajectoryReplay` replays a trace of several top-level invokes and no longer replays a
  subagent launched by a live turn.
- `max_field_length=None` disables the loggers' truncation, as documented.

### Documentation

- Exception and backend docstrings state their rule and cite the providers' documentation;
  docstrings no longer describe the previous design.

## 0.2.0a4 — 2026-09-11

### Breaking changes

- **Component registries are removed.** Pass constructed LLM, REPL, and protocol
  objects to configuration instead of registered names. Custom backends no longer
  implement `validate_model`; model IDs are passed through to the backend.
- **`ATIFTrace` / `ATIFReplay` are now `TrajectoryRecorder` / `TrajectoryReplay`.**
  Deprecated hook-name aliases are removed; import hooks by their current names.
- **`__inputs__` is removed.** Inputs are available directly under their names in
  the agent's REPL.
- **Prompt customization uses Jinja loaders.** Replace `register_template_dir` and
  `register_template_string` with `jaz.add_template_loader`. Core inheritance
  parents use the reserved `jaz/` prefix; inheritance cycles raise Jinja's
  `TemplateAssertionError`.
- **Custom REPLs must implement `repl_history`.** The `maintains_repl_history`,
  `describe_repl_history`, and REPL language-identity surfaces are removed.
  Protocol parsing no longer takes a language, and `CodeOnlyProtocol` no longer
  rescues Markdown-fenced responses.
- **Compaction and SlidingWindow are removed from core.** Compaction is maintained
  with the evaluation tooling instead.
- The input/output truncation-advice surfaces are removed.

### Added

- **GitHub Copilot backend** through `jaz-lang[copilot]`, billed to GitHub AI
  credits, with optional session reuse. `GitHubCopilotLLM` is experimental.
- **Console setup and conversation memory.** First-run provider/model setup,
  persistent settings, validated model switching, Copilot authentication, and
  conversation history for prompt turns and the `%` helper. The console requires
  Python 3.14+; the framework continues to support Python 3.12+.
- **`TrajectoryDirectoryRecorder`** writes a trajectory directory live.
- **`OTelTracing`** is now a public hook.
- **`jaz.add_template_loader`** supports custom prompt templates, named template
  blocks, and template-controlled input placement.
- **`jaz.get_default_description`** exposes the default input rendering.
- **`IterationLimit(min_depth=..., max_depth=...)`** can target selected recursion
  depths.

### Fixed

- Hook reactivation policies and lifecycle cleanup, including recorder run-state
  resets and budget activation.
- Per-depth configuration containers are isolated from mutation.
- Replay detects divergence in message edits and committed assistant turns;
  workflow replay no longer generates broken packages or deleted sentinels.
- Importing `jaz` no longer configures logging or creates a log file.
- Non-hook positional arguments to `invoke()` raise a clear `TypeError`.
- Copilot responses trim hallucinated transcript continuations.
- Async Anthropic requests preserve `max_tokens`.
- `__history__[i].repl_output` is always a string, including on terminal turns.

### Documentation

- Expanded console, Copilot authentication, and prompt-template authoring guidance.

## 0.2.0a3 — 2026-08-22

### Breaking changes

- **`ValidateREPLInput` is now `ValidateREPLCode`.** The rename disambiguates REPL *code*
  (what the agent submits) from invoke *inputs* (what the caller passes). The old name is
  gone; this is the only change to `jaz.hooks.__all__`.
- **RLM support is removed from core.** `jaz.llm_client_rlm` no longer exists, the `rlm`
  backend tag no longer resolves, and the `rlm` extra is gone. It moves to the eval repo.
- **The REPL is stateless.** `BaseREPL.initialize()` now *returns* a `REPLState`, and
  `exec`, `aexec`, `add_inputs`, `add_variables` and `drop_variables` all take that state as
  their first argument. This affects anyone driving a REPL directly or subclassing
  `BaseREPL`; agents and hooks are unaffected.

### Added

Five new hook effects, all public API via `jaz.hooks.effects` and documented in the API
reference:

- **`InsertCode` / `DeleteCode`** — edit the proposed REPL code at `REPLExecEnter`, before
  it runs.
- **`ModifyLLMResponse`** — transform the LLM-query response.
- **`SupplyInvokeResult` / `ModifyInvokeResult`** — supply or transform an invoke's result,
  completing the supply/modify grid alongside the REPL and LLM equivalents.

Reach them as `jaz.hooks.effects.InsertCode` and so on. The top-level shorthand
(`jaz.hooks.InsertCode`) warns, but that is true of every effect, old and new alike — only
the `effects` sub-namespace itself is in `jaz.hooks.__all__`.

### Changed

- `Abort` is accepted on **every** arm of the per-turn `*Exit` events, not just the normal
  one. On an abnormal arm it folds into the in-flight exception rather than replacing it.
- ATIF traces record the buffer index on message steps and edits, and the iteration on
  steps.
- REPL prompt guidance: workflow classification is stated in terms of the first user
  prompt, and the small-incremental-steps advice is more specific.

### Fixed

- `BudgetPool` coherence when an abort and a budget stop land in the same turn.
- The generated API reference builds again under `--strict`. A malformed inline-markup span
  (`**non-**` placed directly against a literal) made docutils reject the docstring, which
  blocked the whole reference from being written.

### Documentation

- Public docstrings no longer carry issue numbers, design-doc pointers, or references to
  private helpers and modules; that material moved into comments beside the code, where it
  stays for maintainers without shipping to users. One docstring pointed readers at
  `jaz._llm_client.LLMResponse` — a private module — where the reachable name is
  `jaz.llm.LLMResponse`.
- Fixed a docstring that rendered as `non-    UNSET` in the published reference, from a
  backslash line-continuation that folded the next line's indentation into the text.

## 0.2.0a2 — 2026-08-18

The first release published to PyPI. Previous versions were distributed only as a
private git repository, so this is where installation changes for everyone.

### Installing

JAZ is now on PyPI under the distribution name **`jaz-lang`**. The import name is
unchanged:

```bash
pip install jaz-lang
```

```python
import jaz
```

The split exists because `jaz` on PyPI belongs to an unrelated project. If you
previously installed from a git URL, replace that command — an upgrade will not find
this release on its own.

The package is published with [PEP 740 attestations](https://docs.pypi.org/attestations/)
via GitHub's trusted publishing, so every artifact is verifiably built from this
repository. The license is now declared as Apache-2.0 in package metadata, matching the
LICENSE file that has always shipped.

### Breaking changes

**Renamed.** Old names are gone unless noted:

| Old | New |
| --- | --- |
| `jaz.providers` (package) | `jaz.llm` |
| `LLM` | `BaseLLM` (all base classes are now `Base*`) |
| `DefaultProtocol`, protocol tag `"default"` | `CodeOnlyProtocol`, tag `"code_only"` |
| `current_scope()` | `get_scope()` — old name still works, deprecated |
| `retry_max_attempts` | `max_retries` — **and it now counts retries, not total attempts** |
| `OverrideLLMResponse` and sibling `Override*` effects | `Supply*` / `Modify*` (still experimental — these warn on access) |
| `__repl_history__` (REPL variable) | `__history__` |
| LLM selector key `tag` | `backend` |
| `InvokeSpan.enter_supply` | `abort` |

**Removed:**

- **Built-in `openai` and `anthropic` backends.** LiteLLM is now the default and only
  registered backend; reach those providers through its `openai/…` and `anthropic/…`
  routes. `litellm` is a core dependency.
- **Silent model defaults.** A backend and model must be named explicitly; there is no
  fallback.
- **`ConversationHistory` and `trace_to_atif`.** ATIF is the sole history format, and the
  older log/replay machinery is gone. `ATIFTrace` and `ATIFReplay` are now public in
  `jaz.hooks`.
- **Hook serialization.** Observability renders `repr()` instead.
- **`BudgetPool.show_status` and `IterationLimit.show_status`.**
- **`BudgetPool` cost tracking.** It is now a pure scalar enforcer; derive costs from an
  ATIF trace.
- **`allow_config_hooks_in_subinvoke`.** Sub-invokes always get the full signature.

**Behavior changes:**

- REPL iterations are numbered from **0** on every surface.
- Budget and iteration exhaustion now raise distinct types —
  `BudgetPoolExhaustedError` and `IterationLimitExhaustedError`.
- `configure()` and `ConfigOverride()` take explicit keyword arguments.
- The recursive sub-invoke primitive is bound as bare `invoke`, not `jaz.invoke`.
- The REPL sandbox denies frame-bearing interpreter surfaces by default.
- Default input/output truncation raised to 50,000 characters.
- Limit warnings fire on absolute remaining budget rather than a fraction;
  `ContextWindowWarning` defaults to 0.8.
- `Library` is experimental and no longer part of the public API — every public route to
  it emits `NonPublicAPIWarning`. Pass tools as plain functions instead: a function given
  as a keyword argument binds into the REPL, and its signature and docstring become the
  description the agent reads.

### Added

- **LiteLLM backend**, routing to every provider LiteLLM supports.
- **Token-native training support**: a token sidecar (`TokenStamp`, `TurnRecord`,
  `LLMResponse.tokens`), `RolloutRecorder` for exportable rollouts, caller-owned chat
  templating via `ChatFormat` / `HFChatFormat`, and an `SGLangLLM` token-in/token-out
  backend. Worked examples cover GSM8K self-training (STaR + GRPO) and distributed GRPO
  through SLIME.
- **ATIF-driven replay** with per-invoke divergence detection, plus message provenance
  and compaction edits captured in traces.
- **Persistent credentials** in `~/.jaz/credentials.json`, with `set_credential()` in the
  console; console history moved to `~/.jaz/history`.
- **`ReturnType(max_failures=…)`** to fail terminally when a return type cannot be
  constructed.
- **Misspelled LLM request parameters now warn** instead of passing silently.
- Console diagnostics for unbalanced braces in a prompt, and `.jaz` script paths named in
  tracebacks.
- `jaz.credentials` is public API for backends resolving ambient credentials.

### Fixed

- A content-less completion normalizes to `""` at the agent boundary.
- Invalid ATIF trace shapes raise `ValueError` rather than failing obscurely.
- `FatalError` stays fatal inside eval harnesses.
- Removed a bogus `service_tier=batch` pricing discount from cost accounting.
