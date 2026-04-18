# jaz

A Python framework for building and optimizing LLM-agents through intelligent, iterative execution.

## Overview

jaz provides a unified framework for creating agents that combine deterministic Python code with LLM reasoning. Agents execute in a REPL loop where the LLM generates code, observes results, and iterates until the task is complete.

### Prerequisites

- **Python ≥ 3.12**
- **GitHub SSH access** — your SSH key must be authorized for the `jaz-lang` GitHub org

## Installation

```bash
pip install git+ssh://git@github.com/jaz-lang/jaz-dist.git@v0.1.0
```

Or with a specific version tag:

```bash
pip install "jaz @ git+ssh://git@github.com/jaz-lang/jaz-dist.git@v0.1.0"
```

Optional extras:

```bash
pip install "jaz[tracing] @ git+ssh://git@github.com/jaz-lang/jaz-dist.git@v0.1.0"
```

### API Keys

JAZ uses native OpenAI and Anthropic HTTP clients. Set your API keys via any of:

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
import jaz

# Simple task
result = jaz.invoke(
    "Calculate the factorial of 10",
    return_type=int
)

# With custom inputs
result = jaz.invoke(
    "Double each number in the list",
    return_type=list,
    numbers=[1, 2, 3, 4, 5]
)
```

## Key Features

### Recursive Agent Composition

Agents can invoke nested agents with separate budgets and automatic cost tracking:

```python
result = jaz.invoke(
    """
    For each item, use jaz.invoke to process it.
    Combine and return the results.
    """,
    return_type=list,
    max_recursion_depth=3,
    items=data
)
```

### Tool Libraries

Provide custom tools to agents:

```python
from jaz import Library, invoke

def my_tool(x: int) -> int:
    """Double a number."""
    return x * 2

my_tools = Library("tools", "Custom tools",
                   modules=[("utils", "Utility helpers")],
                   tools=[("utils.my_tool", my_tool)])
result = invoke("Use tools.utils.my_tool on 5", return_type=int, libraries=[my_tools])
```

### Hooks for Extensibility

Use hooks to add logging, tracing, workflow strategies, and more:

```python
from jaz.hooks import FileLogger, WorkflowStrategyHook

with FileLogger("agent.log"):
    with WorkflowStrategyHook(enable_invoke_start=True):
        result = jaz.invoke("Build a web scraper", return_type=str)
```

### Budget Control

Control costs and execution limits:

```python
result = jaz.invoke(
    "Complex task",
    return_type=str,
    max_cost_budget=0.50,      # USD limit
    max_recursion_depth=2,
    max_invoke_calls=10,
    max_iterations=20
)
```

### Configuration

Override defaults globally or per-invoke:

```python
# Per-invoke override
with jaz.config_override(model_config={"temperature": 0.7, "max_tokens": 2000}):
    result = jaz.invoke("Creative task", return_type=str)

# Global configuration
jaz.configure(model_config={"model": "openai/gpt-4"}, max_repl_iterations=15)
```

## Architecture

```
jaz/
├── agent.py        # Core Agent class
├── invoke.py       # Public invoke() API
├── config.py       # Configuration system
├── budget.py       # Cost tracking
├── repl/           # Python and Bash REPL implementations
├── hooks/          # Hook system, event orchestration, and built-in hooks
├── providers/      # LLM provider clients (OpenAI, Anthropic)
└── library/        # Tool library system
```

### Core Concepts

- **Agent**: Orchestrates LLM queries and REPL execution
- **REPL**: Executes agent-generated code with safety controls
- **Hooks**: Event-based system for logging, budget control, workflow capture, and extensibility
- **Libraries**: Hierarchical tool namespaces for agent use

## Built-in Hooks

| Hook | Purpose |
|------|---------|
| `PrintLogger` | Log events to console |
| `FileLogger` | Log events to file |
| `WorkflowReplayHook` | Materialize agent trajectories as Python code |
| `WorkflowStrategyHook` | Multi-select workflow strategy at decision points |
| `MemoryStoreHook` | Inject per-episode in-memory code memory into the REPL |
| `JaegerTracingHook` | OpenTelemetry distributed tracing |
| `LangfuseTracingHook` | OpenTelemetry tracing to Langfuse Cloud |


### Writing Custom Hooks

Hooks observe agent execution via **events** and influence it by returning **effects**. All imports come from `jaz.hooks`:

```python
from jaz.hooks import Hook, Event, Effect, ReplIterationEnter, AddInstructionPrompt

class ConciseHook(Hook):
    """Instruct the agent to be concise after iteration 3."""

    def on_event(self, event: Event) -> list[Effect]:
        if isinstance(event, ReplIterationEnter) and event.iteration > 3:
            return [AddInstructionPrompt("Be concise — you're running low on iterations.")]
        return []

# Hooks activate via context managers and propagate to nested invoke() calls
with ConciseHook():
    result = jaz.invoke("Solve this step by step", return_type=str)
```

Hooks compose naturally as context managers:

```python
from jaz.hooks import disable_hook, isolated_hooks

# Stack multiple hooks
with ConciseHook(), FileLogger("agent.log"):
    result = jaz.invoke(...)

# Temporarily disable a specific hook type
with disable_hook(ConciseHook):
    result = jaz.invoke(...)  # ConciseHook is inactive here

# Clear all parent hooks for a clean slate
with isolated_hooks():
    result = jaz.invoke(...)  # No hooks active
```

### Event Types

Events are fired at discrete points in the agent's execution lifecycle. All are importable from `jaz.hooks`.

| Event | Fired when... |
|-------|---------------|
| `InvokeEnter` | `invoke()` is called, before any execution begins |
| `InvokeExit` | `invoke()` completes (success or failure) |
| `ReplIterationEnter` | Before a REPL iteration (LLM query + code execution) |
| `ReplIterationExit` | After a REPL iteration completes |
| `ReplExecutionEnter` | After LLM generates code, before it's executed |
| `ReplExecutionExit` | After code execution completes |
| `LLMQueryEnter` | Before making an LLM API call |
| `LLMQueryExit` | After receiving the LLM response |
| `Message` | The agent emits a message (e.g., status update) |
| `BudgetUpdate` | Cost or iteration budget changes |
| `LLMRetry` | An LLM API call is being retried after failure |

Each event carries contextual data (e.g., `ReplIterationEnter.iteration`, `LLMQueryEnter.model`). Span variants (`InvokeSpan`, `ReplIterationSpan`, etc.) are context managers that fire the corresponding enter/exit events.

### Effect Types

Effects are returned by hooks to influence execution. All are importable from `jaz.hooks`.

| Effect | Influence |
|--------|-----------|
| `HaltExecution` | Stop execution immediately with a reason and error |
| `AddInstructionPrompt` | Append text to the next user prompt sent to the agent |
| `AddSystemPrompt` | Append text to the system prompt |
| `AddReplInput` | Inject a Python object into the REPL environment |
| `ModifyIterationLimit` | Change the maximum number of REPL iterations |
| `OverrideLLMClient` | Switch the LLM client for this query |
| `OverrideModelConfig` | Override model parameters (temperature, max_tokens, etc.) |
| `AddLibrary` | Add a tool library to the agent's environment |

### Episode Memory Store (Eval Harness)

For Oolong evals, you can enable an episode-local memory store that is exposed
directly in the agent REPL:

```yaml
jaz:
  memory_store: true
```

When enabled, each episode gets a fresh in-memory store (no cross-episode/task persistence),
and the REPL receives:

- `store`: memory store object
- `__store_catalog__`: `{name: description}` for all stored items

REPL API:

- `store.get(name)` -> live object
- `store.get_code(name)` -> raw code string
- `store.insert(name, description, code)` -> add item (code must define symbol `name`)
- `store.delete(name)` -> remove item
- `store.catalog` -> `{name: description}` mapping

### Langfuse Tracing Quick Start

```python
import jaz
from jaz.hooks import LangfuseTracingHook

with LangfuseTracingHook():
    result = jaz.invoke("Calculate 2+2", return_type=int)
```

Environment variables:
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_HOST` (optional, default: `https://cloud.langfuse.com`)

## Contributing

Contributor setup, eval install matrix, and pre-commit instructions live in [DEV_SETUP.md](DEV_SETUP.md). PR workflow and code conventions are documented in [CLAUDE.md](CLAUDE.md).

## License

See LICENSE file.
