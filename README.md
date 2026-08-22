# JAZ

A Python framework for building and optimizing LLM-agents through intelligent, iterative execution.

## Overview

JAZ provides a unified framework for creating agents that combine deterministic Python code with LLM reasoning. Agents execute in a REPL loop where the LLM generates code, observes results, and iterates until the task is complete.

### Prerequisites

- **Python ≥ 3.12** — 3.14+ is recommended for the full feature set. Everything works on
  3.12/3.13 except the t-string prompt syntax (`invoke(task=t"Analyze the {data}")`), which
  is PEP 750 and requires Python ≥ 3.14; the equivalent
  ``invoke(task="Analyze the `data`", data=data)`` works on every supported version.

## Installation

```bash
pip install jaz-lang
```

The distribution is named `jaz-lang`; the import name is `jaz`:

```python
import jaz
```

Optional extras:

```bash
pip install "jaz-lang[tracing]"
```

To pin a specific release, or to install one that has not reached PyPI, use the
distribution repo and a release tag:

```bash
pip install "jaz-lang @ git+https://github.com/jaz-lang/jaz.git@v0.2.0a3"
```

### API Keys

JAZ reads provider API keys from the environment, under each provider's standard variable name. Set them via any of:

- **Shell profile** — add `export OPENAI_API_KEY=sk-...` to `~/.zshrc` or `~/.bashrc` (simplest, available globally)
- **`.env` file** — create a `.env` at the project root with `OPENAI_API_KEY=sk-...` per line, then load it with:
  - **VS Code** — set `"python.terminal.useEnvFile": true` in settings
  - **direnv** — add `dotenv` to an `.envrc` file to auto-load on `cd` (`brew install direnv`)

### VS Code Setup

After installation, select the Python interpreter so VS Code uses the correct environment:

1. Open the command palette (`Cmd+Shift+P` / `Ctrl+Shift+P`)
2. Search **"Python: Select Interpreter"**
3. Choose the one marked **(Recommended)** — this is the virtual environment created by uv or conda

## Quick Start

```python
from jaz import invoke
from jaz.hooks import ReturnType

# Simple task
result = invoke(
    ReturnType(int),
    task="Calculate the factorial of 10",
)

# With custom inputs
result = invoke(
    ReturnType(list),
    task="Double each number in the list",
    numbers=[1, 2, 3, 4, 5]
)
```

Later snippets assume the Quick Start imports (`from jaz import invoke`,
`from jaz.hooks import ReturnType`, `import jaz`) and show only what each adds.

## Key Features

### Recursive Agent Composition

Agents can invoke nested agents with separate budgets and automatic cost tracking:

```python
from jaz.hooks import RecursionLimit, ReturnType

with RecursionLimit(max_depth=3):  # optional cap; recursion is unbounded by default
    result = invoke(
        ReturnType(list),
        task="""
        For each item, use jaz.invoke to process it.
        Combine and return the results.
        """,
        items=data,
    )
```

### Custom Tools

A tool is just a function you pass in. Give it as a keyword argument and it binds to
that name in the agent's REPL; its signature and docstring render in the prompt, so
the docstring *is* the description the agent reads:

```python
def my_tool(x: int) -> int:
    """Double a number."""
    return x * 2

result = invoke(ReturnType(int), task="Use my_tool on 5", my_tool=my_tool)
```

The agent sees `` `my_tool(x: int) -> int`: Double a number. `` — type hints and a
one-line docstring are what make a tool usable. Nothing is registered up front, so tools
sit alongside data in the same call:

```python
def web_search(query: str) -> list[str]:
    """Search the web and return result snippets."""
    ...

result = invoke(
    ReturnType(dict),
    task="Look up each name and summarize what you find",
    names=["ada", "grace"],
    web_search=web_search,
)
```

To group related tools under one name, pass an object instead: its public methods render
as a catalog under that name, and the agent calls them as `utils.my_tool(...)`.

```python
class Utils:
    """Utility helpers."""

    def my_tool(self, x: int) -> int:
        """Double a number."""
        return x * 2

result = invoke(ReturnType(int), task="Use utils.my_tool on 5", utils=Utils())
```

To make a tool propagate automatically to nested `invoke` calls, bind it with
`jaz.scope` instead of passing it per call:

```python
from jaz import scope

with scope(my_tool=my_tool):
    invoke(ReturnType(int), task="Use my_tool on 5")  # nested invokes inherit `my_tool`
```

### Hooks for Extensibility

Use hooks to add logging, tracing, workflow strategies, and more:

```python
from jaz.hooks import BudgetPool, FileLogger

with FileLogger("agent.log"), BudgetPool(cost_budget=1.0):
    result = invoke(ReturnType(str), task="Build a web scraper")
```

### Budget Control

Pool budgets (LLM cost / call count, shared across the whole invoke/recursion
tree) are enforced by the opt-in `BudgetPool`; per-level execution limits
are configuration:

```python
from jaz.hooks import BudgetPool, IterationLimit, RecursionLimit

# Pool budgets: tracked + enforced only while the hook is active.
with BudgetPool(cost_budget=0.50, calls_budget=100):  # USD / call-count
    result = invoke(ReturnType(str), task="Complex task")

# Per-level execution limits are hooks too:
with IterationLimit(max_iterations=20), RecursionLimit(max_depth=2):
    result = invoke(ReturnType(str), task="Complex task")
```

### Configuration

Override defaults globally or per-invoke:

Each group takes the **configured component**, whose constructor is that group's settings.
Setting a group *replaces* it — a component states itself completely, so there is no partial
update of one setting.

```python
from jaz.llm import LiteLLM
from jaz.repl.python_repl import PythonREPL

# Per-invoke override
with jaz.ConfigOverride(llm=LiteLLM(model="openai/gpt-5-mini", temperature=0.7, max_tokens=2000)):
    result = invoke(ReturnType(str), task="Creative task")

# Global configuration
jaz.configure(llm=LiteLLM(model="openai/gpt-5-mini"), repl=PythonREPL(exec_timeout=60))
```


### Custom & Local LLM Backends

JAZ talks to LLMs through a small `BaseLLM` layer. Currently, **LiteLLM is the only backend covered by the stable API, and the default** — one backend that routes to every provider LiteLLM supports (OpenAI, Anthropic, Gemini, Bedrock, Vertex, …). Other backends ship in the package but can change or be removed. You can add your own backend without forking, mirroring the REPL extension pattern.

A config selects **one** backend: the `BaseLLM` you pass to `llm=`, defaulting to `LiteLLM`. Its constructor takes everything it needs — the `model` id (a LiteLLM route like `openai/gpt-5-mini`), the backend's own settings, and any per-request defaults. `@register_llm` also gives a backend a `tag`, which is how JAZ's own config-file loaders (the eval harness, the console's `--model`) name it as data; because litellm is the default, a config need only name the `model`.

**Local OpenAI-compatible server** (Ollama, vLLM, LM Studio, llama.cpp, …) — no code, just config:

```python
# route the litellm backend to a local OpenAI-compatible server
# (api_key is required by the openai/ route even for a local server — a dummy value is fine;
#  it rides request_defaults into litellm.completion)
jaz.configure(
    llm=LiteLLM(model="openai/llama3", api_base="http://localhost:11434/v1", api_key="dummy"),
)
```

**A custom backend** — subclass `BaseLLM` and pass an instance (registering it is only needed if a config *file* must name it):

```python
import os
from jaz.llm import BaseLLM, register_llm, CompletionResponse, Usage, Choice, Message

@register_llm("mybackend")                    # the tag a config file can name
class MyLLM(BaseLLM):
    def __init__(self, base_url: str | None = None, **retry):
        super().__init__(**retry)             # forwards the retry_* settings
        self.base_url = base_url or os.environ["MYBACKEND_API_BASE"]

    def complete(self, model, messages, **kwargs):
        # ... call self.base_url; raise jaz.llm.RateLimitError / AuthenticationError / ...
        # on API errors so retry classification works.
        response = CompletionResponse(
            choices=[Choice(message=Message("assistant", "..."))],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            model=model,
        )
        return self.finalize(response, model)  # token accounting + cost, for free

jaz.configure(llm=MyLLM(model="my-model", base_url="http://localhost:8000"))
```

One class owns the whole job: the API call, retry, cost accounting and model metadata. `finalize()` turns your wire response into the normalized `jaz.llm.LLMResponse` and prices it from the bundled table (models absent from it simply report `cost=None`); the retry wrappers and the non-retryable-error classification come from the base.

When a config *file* names your backend by tag, the split between "settings for the backend object" and "params for the request" is taken from your `__init__` signature, so `base_url` reaches `MyLLM(...)` while `model` rides each call — a backend declares its construction keys just by declaring `__init__`. Constructing it yourself needs no such rule: `ConfigOverride(llm=my_llm)`.

`register_llm` deliberately refuses to clobber an existing tag — registering `"litellm"` again raises. `OpenAILLM` and `AnthropicLLM` ship in the package but are not registered, so `backend: openai` does not resolve — reach those providers through litellm's `openai/…` / `anthropic/…` routes. Registration and stability are separate axes: a tag can resolve without being covered by the stable API. To add or replace a backend, subclass `BaseLLM` and register your own tag.

## Architecture

```
jaz/
├── invoke.py       # Public invoke() API
├── config.py       # Configuration system
├── budget.py       # Cost tracking
├── repl/           # Python REPL implementations
├── hooks/          # Hook system, event orchestration, and built-in hooks
├── llm/            # LLM backends (LiteLLM by default)
└── protocol/       # Wire-format codec between the LLM and the REPL
```

### Core Concepts

- **Agent loop**: Orchestrates LLM queries and REPL execution
- **REPL**: Executes agent-generated code with safety controls
- **Hooks**: Event-based system for logging, budget control, workflow capture, and extensibility
- **Tools**: Plain functions passed as inputs; their signature and docstring become the description the agent reads

## Built-in Hooks

| Hook | Purpose |
|------|---------|
| `ReturnType` | Declare + enforce the invoke's return type |
| `ValidateReturn` / `ValidateREPLCode` | Validate the return value / veto REPL code before it runs |
| `BudgetPool` | Shared LLM cost / call-count budget with hard-stop enforcement |
| `IterationLimit` / `RecursionLimit` | Per-level turn cap / invoke-nesting cap |
| `BudgetForcing` | Refuse early finishes so the agent keeps working |
| `Compaction` | Summarize old turns to stay inside the context window |
| `ContextWindowWarning` | Warn the agent as its prompt nears the model's window |
| `PrintLogger` / `FileLogger` | Log events to console / file |
| `ATIFTrace` | Write the run as an ATIF trajectory (replay/cost source) |
| `ATIFReplay` | Resume a run from an ATIF trace: saved responses replay, then live calls take over |
| `RolloutRecorder` | Record token-native rollouts for training |
| `JaegerTracing` / `LangfuseTracing` | OpenTelemetry tracing presets |


### Writing Custom Hooks

Hooks observe agent execution via **events** (from `jaz.hooks.events`) and influence it by
returning **effects** (from `jaz.hooks.effects`):

```python
from jaz import invoke
from jaz.hooks import Hook
from jaz.hooks.effects import AddMessages, Effect
from jaz.hooks.events import LLMQueryEnter

class ConciseHook(Hook):
    """Instruct the agent to be concise after iteration 3."""

    # Override the typed per-event handler for the event you care about — no
    # isinstance/dispatch boilerplate. (A cross-cutting observer that wants *every*
    # event overrides `on_any` instead.)
    def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Effect]:
        if event.iteration >= 3:  # iterations are 0-based
            return [AddMessages([{"role": "user", "content": "Be concise."}])]
        return []

# Hooks activate via context managers and propagate to nested invoke() calls
with ConciseHook():
    result = invoke(task="Solve this step by step")
```

Hooks compose naturally as context managers:

```python
# Stack multiple hooks
with ConciseHook(), FileLogger("agent.log"):
    result = invoke(...)
```

### Event Types

Events fire around three spans — the whole **Invoke**, each turn's **LLMQuery**, and each
turn's **REPLExec** — and every span walks the same four stages. All are importable from
`jaz.hooks.events`.

| Stage | Observes | Accepts | Fires |
|-------|----------|---------|-------|
| `*Enter` | the proposal | edit effects + `Abort` | always |
| `*Send` | the committed input | supply effects + `Abort` | iff the input committed |
| `*Complete` | the raw result | modify effects + `Abort` | iff a raw result exists |
| `*Exit` | the outcome union (`Completed \| Aborted \| Failed`) | nothing (observation-only) | whenever the span opened |

Plus `LLMQueryRetry`, fired per retry attempt of an LLM call. Every event carries
contextual data (e.g. `LLMQueryEnter.model`, `LLMQueryEnter.iteration`) and a
`timestamp` (its emission time — durations are arithmetic over two events' timestamps).

### Effect Types

Effects are returned by hooks to influence execution. All are importable from
`jaz.hooks.effects`; each composes at a specific stage (an effect returned anywhere else
raises `InvalidEffectError`).

| Effect | Stage | Influence |
|--------|-------|-----------|
| `AddInputs` / `DropInputs` | `InvokeEnter` | Add/un-pass invoke inputs (prompt + REPL) |
| `DisableRecursion` | `InvokeEnter` | Withhold the recursive `jaz.invoke` tool |
| `AddMessages` / `DropMessages` | `LLMQueryEnter` | Edit the messages sent to the model (transient or persistent) |
| `AddVariables` / `DropVariables` | `REPLExecEnter` | Bind/unbind REPL namespace names for the turn |
| `SupplyLLMResponse` | `LLMQuerySend` | Supply the response, skipping the API call |
| `SupplyExecResult` | `REPLExecSend` | Supply a result, skipping execution |
| `ModifyExecResult` | `REPLExecComplete` / `InvokeComplete` | Replace the raw result |
| `Abort` | any `Enter`/`Send`/`Complete` | Abort the invoke; its error propagates out of `invoke()` |
| `BlackboardWrite` | any event | Write to the per-invoke cross-hook blackboard |

### Langfuse Tracing Quick Start

```python
from jaz import invoke
from jaz.hooks import LangfuseTracing, ReturnType

with LangfuseTracing():
    result = invoke(ReturnType(int), task="Calculate 2+2")
```

Environment variables:
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_HOST` (optional, default: `https://cloud.langfuse.com`)

## Contributing

This repository is generated: each release publishes a cleaned copy of the package, so
changes are not made here. Contributor setup and code conventions live with the source.

## License

See LICENSE file.
