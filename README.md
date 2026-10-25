# JAZ

A Python framework for building and optimizing LLM-agents through intelligent, iterative execution.

If you're using JAZ, please cite [our paper](https://arxiv.org/abs/2609.26891) using the following BibTeX entry:
```bibtex
@article{li2026jaz,
  title={Harness as a Language: A Minimalist Agent Framework With Maximal Expressivity},
  author={Li, Zhening and Liu, Joshua and Vukelic, Mateja and Shen, Nicole and Lall, Supriya and Thakur, Amitayush and Zhang, Alex and Khattab, Omar and Light, Jonathan and Solar-Lezama, Armando},
  journal={arXiv preprint arXiv:2609.26891},
  year={2026}
}
```

## Overview

JAZ provides a unified framework for creating agents that combine deterministic Python code with LLM reasoning. Agents execute in a REPL loop where the LLM generates code, observes results, and iterates until the task is complete.

### Prerequisites

- **Python ≥ 3.12** — 3.14+ is recommended for the full feature set. The t-string prompt
  syntax (`invoke(task=t"Analyze the {data}")`) and the interactive
  [`jaz` console](#the-jaz-console) need PEP 750 t-strings, hence Python ≥ 3.14.
  Everything else works on 3.12/3.13; the equivalent
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
pip install "jaz-lang[tracing]"   # OpenTelemetry tracing
pip install "jaz-lang[copilot]"   # GitHub Copilot backend — no API key, billed to AI credits
```

To pin a specific release, or to install one that has not reached PyPI, use the
distribution repo and a release tag:

```bash
pip install "jaz-lang @ git+https://github.com/jaz-lang/jaz.git@v0.2.0b1"
```

### API Keys

JAZ reads provider API keys from the environment, under each provider's standard variable name. Set them via any of:

- **Shell profile** — add `export OPENAI_API_KEY=sk-...` to `~/.zshrc` or `~/.bashrc` (simplest, available globally)
- **`.env` file** — create a `.env` at the project root with `OPENAI_API_KEY=sk-...` per line, then load it with:
  - **direnv** — add `dotenv` to an `.envrc` file to auto-load on `cd` (`brew install direnv`)

No API key at hand? The `copilot` backend (the `copilot` extra) bills a Copilot
subscription's GitHub AI credits instead. Authentication checks an explicit `github_token`
first, then `COPILOT_GITHUB_TOKEN` / `GH_TOKEN` / `GITHUB_TOKEN` in the environment
(in that order), then your signed-in `gh`/`copilot` CLI. An exported token therefore
takes precedence over your CLI login:

```python
import jaz
from jaz.llm import GitHubCopilotLLM

jaz.configure(llm=GitHubCopilotLLM(model="gpt-5-mini"))   # ids are Copilot's own, unprefixed
```

For CLI authentication, run `gh auth login`, or start `copilot` and enter `/login`.

Like every backend other than LiteLLM it sits outside the stable API surface — see
[Custom & Local LLM Backends](#custom--local-llm-backends).

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

Every input is a **keyword** argument — the prompt included. `invoke()`'s positional
arguments are hooks (`ReturnType(int)`, a budget, …) and an optional `ConfigOverride`, so
`invoke("Calculate the factorial of 10")` is an error, not a shorthand.

## The `jaz` console

Installing puts a `jaz` command on `PATH` (it needs Python ≥ 3.14 to run): a real Python
REPL — variables persist, `_` holds the last result — with conversational sugar layered
on top. Run `jaz`, then enter these lines at its prompt:

```
> what is 2 + 2?
n: int <- how many days in a leap year?
?n
% switch to gpt-5-mini
import math
```

`>` asks the agent; `n: int <-` captures a typed result; `?n` shows what the agent sees
for `n`; `%` asks the jaz helper to change settings, with confirmation. Plain Python
still runs. Enter the sigil lines without trailing `#` annotations: they become prompt
text, or can leave a `?` expression waiting for more input.

A session with no model configured opens a setup wizard first: pick a provider and paste
its API key, or pick **github copilot** — no provider key, just the `copilot` extra and
[GitHub authentication](#api-keys). The choice is validated with one live request before
anything is kept. Backend and model defaults are saved to `~/.jaz/settings.json`;
provider API keys entered in the wizard go in `~/.jaz/credentials.json`.

Flags configure one session without touching that default — `jaz --backend copilot --model
gpt-5-mini`, `jaz --max-cost 0.50`, `jaz --log run.log` — and `jaz -c "…"` runs a single
prompt and exits. In-session, `setup_llm()` re-runs the wizard and
`switch_model("openai/gpt-5")` changes model (keeping the backend) after validating it.

## Key Features

### Recursive Agent Composition

Agents can invoke nested agents with separate budgets and automatic cost tracking:

```python
from jaz.hooks import RecursionLimit, ReturnType

with RecursionLimit(max_depth=3):  # optional cap; recursion is unbounded by default
    result = invoke(
        ReturnType(list),
        task="""
        For each item, use invoke to process it.
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

To group related tools under one name, pass an object instead: it renders as its class
docstring with one bullet per public method, and the agent calls them as
`utils.my_tool(...)`.

```python
class Utils:
    """Utility helpers."""

    def my_tool(self, x: int) -> int:
        """Double a number."""
        return x * 2

result = invoke(ReturnType(int), task="Use utils.my_tool on 5", utils=Utils())
```

The agent sees:

```
`utils` [Utils]: Utility helpers.
  - `my_tool(x: int) -> int`: Double a number.
```

Data and other values render as their `str()`, so define `__str__` on a type to control what
the agent sees for it, or `__repr__` for a value that sits inside a list or dict.

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
are opt-in hooks as well:

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
from jaz.repl import PythonREPL

# Per-invoke override
with jaz.ConfigOverride(llm=LiteLLM(model="openai/gpt-5-mini", temperature=0.7, max_tokens=2000)):
    result = invoke(ReturnType(str), task="Creative task")

# Global configuration
jaz.configure(llm=LiteLLM(model="openai/gpt-5-mini"), repl=PythonREPL(exec_timeout=60))
```


### Custom & Local LLM Backends

JAZ talks to LLMs through a small `BaseLLM` layer. **`LiteLLM` is the default** — one backend that routes to every provider LiteLLM supports (OpenAI, Anthropic, Gemini, Bedrock, Vertex, …) — and `GitHubCopilotLLM` and `SGLangLLM` cover the two cases it does not serve well: Copilot billing through GitHub AI credits, and a self-hosted SGLang server addressed token-in/token-out. All three are part of the stable API. You can add your own backend without forking.

A config selects **one** backend: the `BaseLLM` you pass to `llm=`, defaulting to `LiteLLM`. Its constructor takes everything it needs — the `model` id (a LiteLLM route like `openai/gpt-5-mini`), the backend's own settings, and any per-request params. A config *file* can name a backend instead of constructing one, but the name lives in whatever reads the file: JAZ's own loaders (the eval harness, the console) each keep their own tag → class mapping, and core carries neither the mapping nor a name on the class. Because litellm is the default, a config need only name the `model`.

**Local OpenAI-compatible server** (Ollama, vLLM, LM Studio, llama.cpp, …) — no code, just config:

```python
# route the litellm backend to a local OpenAI-compatible server
# (api_key is required by the openai/ route even for a local server — a dummy value is fine;
#  it rides the query params into litellm.completion)
jaz.configure(
    llm=LiteLLM(model="openai/llama3", api_base="http://localhost:11434/v1", api_key="dummy"),
)
```

**A custom backend** — subclass `BaseLLM` and pass an instance (a name is only needed if a config *file* must select it, and it is that file's reader that holds the name):

```python
import os
from functools import cached_property
from jaz.llm import BaseLLM, LLMResponse, ModelInfo, RetryPolicy

class MyLLM(BaseLLM):
    def __init__(self, model: str, base_url: str | None = None, retry_policy=RetryPolicy(), **query_params):
        super().__init__(retry_policy=retry_policy)   # the one thing the base holds
        self._model = model                          # yours: the base has no `model`
        self._base_url = base_url or os.environ["MYBACKEND_API_BASE"]
        self._query_params = query_params            # yours too; you put them on the call

    def complete(self, messages):
        # ... call self._base_url; raise jaz.exceptions.RateLimitError / AuthenticationError / ...
        # on API errors so retry classification works.
        wire = my_http_post(self._base_url, model=self._model, messages=messages, **self._query_params)
        return LLMResponse(
            content=wire["text"],
            prompt_tokens=wire["usage"]["input"],
            completion_tokens=wire["usage"]["output"],
            cost_usd=None,                    # or price it yourself; see below
            extra={},
        )

    @cached_property
    def model_info(self):
        return ModelInfo(name=self._model)    # what you know; every field defaults to "unknown"

jaz.configure(llm=MyLLM(model="my-model", base_url="http://localhost:8000"))
```

One class owns the whole job: the API call, its parameters, and whatever accounting you choose to report. `complete()` receives the messages and returns an `LLMResponse` — that is the entire contract, and `BaseLLM`, `LLMResponse`, `ModelInfo` and `RetryPolicy` are the only names you need from `jaz.llm`. Nothing passes through the base: the model, the sampling parameters and the credentials are your own state, set in your `__init__` and put on the request by your `complete`. The base holds retry alone, so a backend with a `**query_params` tail names `retry_policy` explicitly and forwards only that. The base has no `model`: a backend that serves a nameable model takes `model=` as its own setting (all three built-ins do) and reports it through `model_info.name`, which is what tracing hooks label spans with; one that serves whatever its server loaded leaves the name `None`. The retry wrappers come from the base, which retries a failed call only on its `retryable_exceptions` (`RateLimitError`, `ServerError` and `APIConnectionError`, timeouts included, from `jaz.exceptions` by default): raise a transient failure as one of those, or add your own class to that tuple, and anything else fails at once. A user can retry more errors with `RetryPolicy(extra_retryable_exceptions=...)`.

**Cost is yours to report.** Set `cost_usd` if you can price the call — [LiteLLM's pricing helpers](https://docs.litellm.ai/docs/completion/token_usage) (`litellm.completion_cost`, `litellm.model_cost`) are what the built-in `LiteLLM` backend prices with, and litellm is a JAZ dependency, so it is already installed.

If your backend *can* price calls, say so with `ModelInfo(can_report_cost=True)` — the default is `False`, because JAZ itself prices nothing. Leaving it alone is safe: a `cost_budget` is then rejected before any call with an error naming your backend, and `calls_budget` still works. `max_input_tokens` is the same shape — it defaults to `None` ("unknown", not "unlimited"), and the `ContextWindowWarning` hook sizes its window from it, so a backend that does not report one just leaves that hook disabled rather than erroring. Retry is one value too: `MyLLM(model=..., retry_policy=RetryPolicy(max_retries=2))`.

When a config *file* names your backend, every key in its block is a constructor argument: `model` and `base_url` reach the named parameters, and anything else (`temperature`) lands in your `**query_params` tail — a backend declares its settings just by declaring `__init__`. Whoever loads that file supplies the name → class mapping (the eval harness does it with the non-public `jaz._instantiate.build_config(..., resolvers={"llm": my_lookup})`); JAZ core holds no registry, and a backend class carries no name of its own. Constructing it yourself needs no name at all: `ConfigOverride(llm=my_llm)`.

**Testing without a provider.** `MockLLM` is a backend that answers from a callable, so an agent can be exercised with no API key, no network and no nondeterminism — through the same `configure`/`ConfigOverride` as any backend, with retry and cost accounting intact:

```python
from jaz.llm import MockLLM

def scripted(messages):
    return "return 42"                    # a str is the assistant's reply; or return an LLMResponse

with jaz.ConfigOverride(llm=MockLLM(scripted)):
    assert jaz.invoke(task="add 40 and 2") == 42
```

`LiteLLM` is the default; `GitHubCopilotLLM` and `SGLangLLM` ship alongside it for the cases it does not cover, and all three are covered by the stable API. Reach OpenAI and Anthropic through litellm's `openai/…` / `anthropic/…` routes — the hand-written vendor-direct backends were removed before the first release. To add or replace a backend, subclass `BaseLLM`; add it to your loader's map only if a config file needs to name it.

The configured model id is sent verbatim: JAZ neither strips nor rewrites a `provider/` prefix, so write the id the backend's own documentation uses. A backend is chosen by configuring it, not by prefixing the model — the prefix tells the configured backend which provider to route to, and a backend that does not route (`SGLangLLM`) sends the id to its own server verbatim.

## Architecture

```
jaz/
├── _invoke.py      # invoke() / ainvoke(), re-exported from `jaz`
├── _config.py      # configure() / ConfigOverride / get_config()
├── _agent.py       # The REPL loop that drives one invoke
├── repl/           # REPL implementations (PythonREPL)
├── hooks/          # Hook system: events, effects, dispatcher, built-in hooks (BudgetPool, ...)
├── llm/            # LLM backends (LiteLLM by default)
└── protocol/       # Wire-format codec between the LLM and the REPL
```

Modules with a leading underscore are non-public; everything supported is exported from `jaz`
or one of its public modules (`jaz.hooks`, `jaz.llm`, `jaz.repl`, `jaz.protocol`,
`jaz.exceptions`, `jaz.credentials`).

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
| `ContextWindowWarning` | Warn the agent as its prompt nears the model's window |
| `PrintLogger` / `FileLogger` | Log events to console / file |
| `TrajectoryRecorder` | Write the run as an ATIF trajectory (replay/cost source) |
| `TrajectoryDirectoryRecorder` | Write the run as a browsable Markdown directory, live (`tail -f` a running agent) |
| `TrajectoryReplay` | Resume a run from an ATIF trace: saved responses replay, then live calls take over |
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
contextual data (e.g. `LLMQueryEnter.iteration`, `event.config`) and a
`timestamp` (its emission time — durations are arithmetic over two events' timestamps).

### Effect Types

Effects are returned by hooks to influence execution. All are importable from
`jaz.hooks.effects`; each composes at a specific stage (an effect returned anywhere else
raises `InvalidEffectError`).

| Effect | Stage | Influence |
|--------|-------|-----------|
| `AddInputs` / `DropInputs` | `InvokeEnter` | Add/un-pass invoke inputs (prompt + REPL) |
| `DisableRecursion` | `InvokeEnter` | Withhold the recursive `invoke` tool |
| `SupplyInvokeResult` | `InvokeSend` | Supply the invoke's result, skipping the agent loop |
| `ModifyInvokeResult` | `InvokeComplete` | Replace the invoke's result |
| `AddMessages` / `DropMessages` | `LLMQueryEnter` | Edit the messages sent to the model (transient or persistent) |
| `SupplyLLMResponse` | `LLMQuerySend` | Supply the response, skipping the API call |
| `ModifyLLMResponse` | `LLMQueryComplete` | Replace the model's response |
| `AddToNamespace` / `DropFromNamespace` | `REPLExecEnter` | Bind/unbind REPL namespace names for the turn |
| `InsertCode` / `DeleteCode` | `REPLExecEnter` | Edit the code about to run |
| `SupplyExecResult` | `REPLExecSend` | Supply a result, skipping execution |
| `ModifyExecResult` | `REPLExecComplete` | Replace the raw execution result |
| `Abort` | any `Enter`/`Send`/`Complete`, and `LLMQueryExit` / `REPLExecExit` | Abort the invoke; its error propagates out of `invoke()` |
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
