# Changelog

All notable user-facing changes to JAZ. Versions follow the distribution on PyPI
(`jaz-lang`); the import name is `jaz`.

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
