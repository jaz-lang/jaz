"""The GitHub Copilot backend — completions billed to GitHub AI credits.

:class:`CopilotLLM` reaches every model GitHub Copilot serves (``gpt-5-mini``, ``claude-opus-5``,
``gemini-3.1-pro-preview``, …) through the official Copilot SDK, so usage is charged to a Copilot
subscription's **GitHub AI credits** rather than to a per-vendor API key::

    jaz.configure(llm=CopilotLLM(model="gpt-5-mini"))

Model ids are the ones ``copilot`` itself lists (``CopilotLLM.list_models()`` returns them); they
carry no ``provider/`` prefix.

Authentication is the SDK's, in its own order: an explicit ``github_token``, then
``COPILOT_GITHUB_TOKEN`` / ``GH_TOKEN`` / ``GITHUB_TOKEN``, then a signed-in ``gh``/``copilot``
CLI. jaz's credential store is not consulted — the token is GitHub's, not a model provider's.

Requires the ``jaz[copilot]`` extra (``github-copilot-sdk``) and a GitHub Copilot subscription.

Limitations:
    * **Sampling parameters are not available.** The Copilot runtime owns the request body, so
      ``temperature``, ``top_p``, ``max_tokens`` and friends are rejected rather than silently
      dropped. ``reasoning_effort`` is the one knob that maps through.
    * **The conversation is flattened into a single prompt.** The SDK's unit of input is one
      prompt string per turn, so every message after the system message is rendered into a
      ``[[role]]``-delimited transcript. Content that itself contains a line reading
      ``[[assistant]]`` is indistinguishable from a role marker — on the way in it is read as
      one, and on the way out the completion is cut short there, because a model that has been
      shown a transcript sometimes carries on writing it past the end of its own turn.
    * **No streaming and no token sidecar** — one prompt in, one final message out.
    * Cost is exact (GitHub reports the credits it billed), but arrives *after* the call, so a
      ``cost_budget`` is enforced on completed calls like any other backend's.
"""

# Why the SDK rather than an HTTP endpoint (the shape of this backend follows from it): GitHub
# publishes no OpenAI-compatible chat endpoint that spends AI credits. The credit-billed surface is
# the Copilot CLI runtime, which the SDK drives over JSON-RPC:
#
#     jaz Agent -> CopilotLLM -> copilot SDK -> (JSON-RPC) -> Copilot CLI -> model
#
# The CLI is an *agent* runtime, not an inference API, so the work here is subtracting the agent:
# `mode="empty"` + `available_tools=[]` + a `replace` system message leave a bare completion
# primitive. jaz owns the loop, the tools (the REPL) and the prompt; Copilot supplies only the
# model call. Leaving the CLI's own tools on would run a second agent loop inside every jaz turn.
#
# LiteLLM's `github_copilot/` provider was the alternative and was rejected: it authenticates by
# device flow against the Copilot *editor* token and reports no cost, where the SDK reuses an
# existing `gh`/`copilot` login and returns GitHub's own per-call credit charge.

from __future__ import annotations

import asyncio
import importlib
import re
import threading
import weakref
from pathlib import Path
from typing import Any

from ..paths import ensure_config_dir
from .exceptions import (
    APIError,
    AuthenticationError,
    ContextWindowExceededError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnsupportedParamsError,
)
from .llm import BaseLLM, LLMResponse

#: GitHub prices every model in AI credits and fixes one credit at a cent, so the credit charge
#: the runtime reports converts to dollars exactly — no pricing table, no drift.
#: https://docs.github.com/en/copilot/reference/copilot-billing/models-and-pricing
USD_PER_AI_CREDIT = 0.01

#: Credits are reported in nano-credits (``totalNanoAiu``).
_NANO = 1e9

_DEFAULT_TIMEOUT_SECONDS = 600.0

#: The only per-request parameter the SDK exposes; everything else jaz might forward from
#: ``request_defaults`` would be silently discarded, so it is rejected instead.
_SUPPORTED_REQUEST_PARAMS = frozenset({"reasoning_effort"})

_CONTEXT_LENGTH_MARKERS = ("context length", "context_length", "too long", "max_tokens")


def _import_sdk() -> tuple[Any, Any]:
    """Import the Copilot SDK, returning ``(copilot, copilot.session_events)``."""
    # Deferred and via importlib on purpose: the SDK is an optional extra, so neither the import
    # nor its types may be reachable at module-import or type-check time.
    try:
        return (
            importlib.import_module("copilot"),
            importlib.import_module("copilot.session_events"),
        )
    except ImportError as exc:  # pragma: no cover - exercised by the extra being absent
        raise ImportError(
            "The copilot backend needs the GitHub Copilot SDK. Install it with "
            "`pip install jaz-lang[copilot]` (or `pip install github-copilot-sdk`)."
        ) from exc


def _text_of(content: Any) -> str:
    """Flatten one message's ``content`` to text, tolerating the multimodal list form."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # OpenAI's content-parts form: keep the text parts, drop what the SDK cannot carry.
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


#: A line that is *exactly* a ``[[role]]`` transcript marker. Used both to build the flattened
#: prompt and to cut the completion short at the first marker the model emits itself.
_ROLE_MARKER_LINE = re.compile(
    r"^\[\[(?:user|assistant|system|developer|tool)\]\][ \t]*$", re.M
)

#: The same leak wearing a different costume, for reused sessions.
_SESSION_MARKER_LINE = re.compile(
    r"^(?:(?:user|assistant)<current_datetime>|(?:\[exit=-?\d+\]|\*\(\d+\.\d+s\)\*)[ \t]*$)",
    re.M,
)
# Under `reuse_session` the model is never shown jaz's `[[role]]` transcript — each send is one
# bare turn, and the conversation it reads is the session's *own* history. So when it runs past
# the end of its turn it imitates that history's format instead: a user turn opens with the
# caller's `<current_datetime>` preamble, and REPL output arrives as an `[exit=N]` line with a
# `*(1.2s)*` latency footer. `_ROLE_MARKER_LINE` never matches any of those, so without a second
# pattern the reuse path silently loses the trim that the fresh-session path has.
#
# The alternatives are anchored differently on purpose. `[exit=N]` and `*(1.2s)*` stand alone on
# their line in the observed history, so they are matched as whole lines like `_ROLE_MARKER_LINE`
# is: a bare `^` prefix would cut any completion containing a line that merely *starts* with
# `[exit=0]` — an agent echoing shell output in a heredoc or docstring, which on this benchmark
# family it does constantly. The `<current_datetime>` preamble cannot be anchored, since its line
# continues with the timestamp and closing tag, but it is a long enough literal that a line
# starting with it is not plausibly anything else. The residual cost of anchoring is a missed trim
# if a real `[exit=N]` line ever carries trailing text; that fails safe, leaving the pre-existing
# behaviour, whereas a false cut silently destroys good output.


def _conversation_turns(messages: list[Any]) -> tuple[str, list[tuple[str, str]]]:
    """Split an OpenAI-format conversation into ``(system_text, [(role, text), ...])``."""
    system_parts: list[str] = []
    turns: list[tuple[str, str]] = []
    for message in messages:
        role = message.get("role", "user")
        text = _text_of(message.get("content"))
        if role == "system" or role == "developer":
            system_parts.append(text)
        else:
            turns.append((role, text))
    return "\n\n".join(p for p in system_parts if p), turns


def _render_turns(turns: list[tuple[str, str]]) -> tuple[str, bool]:
    """Render turns into one prompt, returning ``(prompt, flattened)``."""
    # A single user turn is sent verbatim: the transcript markers below are pure overhead for
    # the one-shot case, and every token of framing is a token the model has to read past.
    flattened = not (len(turns) == 1 and turns[0][0] == "user")
    if not flattened:
        return turns[0][1], False
    return "\n\n".join(f"[[{role}]]\n{text}" for role, text in turns), True


def _split_conversation(messages: list[Any]) -> tuple[str, str, bool]:
    """Split an OpenAI-format conversation into ``(system_text, prompt_text, flattened)``.

    System messages become the session's system message; the rest are rendered into one prompt.
    ``flattened`` is whether that rendering used ``[[role]]`` markers.
    """
    system_text, turns = _conversation_turns(messages)
    prompt, flattened = _render_turns(turns)
    return system_text, prompt, flattened


def _trim_transcript_continuation(
    content: str | None, pattern: re.Pattern[str] = _ROLE_MARKER_LINE
) -> tuple[str | None, int]:
    """Cut ``content`` at the first line matching ``pattern``, a transcript-format marker.

    Returns the trimmed text and the number of characters dropped.
    """
    # Why this exists: the flattened prompt IS a transcript, so a long conversation teaches the
    # model the transcript's own format, and it sometimes keeps writing past the end of its turn —
    # emitting `[[user]]`, a *hallucinated* next user message, and then a reply to it. Observed on
    # claude-opus-4.8 in a Terminal-Bench run: a turn whose real content was fine continued with
    # `[[user]]\n[Note: this is an automatic nudge ...]` and two invented turns. Under a code-only
    # protocol the invented prose is not valid Python, so the whole turn dies as a SyntaxError and
    # is spent for nothing.
    #
    # This is the output-side counterpart of the input-side ambiguity in the module docstring, and
    # it inherits that ambiguity: a completion that legitimately contains a line reading
    # `[[assistant]]` is truncated there. That is accepted — the marker vocabulary is fixed and
    # narrow, a whole line matching one exactly is overwhelmingly the leak, and the alternative
    # (leave it in) corrupts the turn outright. The fresh-session path applies this only when the
    # prompt actually used markers, so a single-turn call — sent verbatim, with nothing to
    # imitate — is never cut there; the reuse path calls it unconditionally, for the reason given
    # at that call site.
    #
    # A stop sequence would be the right mechanism; the Copilot SDK exposes none, so the cut is
    # made here on the returned text.
    if not content:
        return content, 0
    match = pattern.search(content)
    if match is None:
        return content, 0
    # Counted against the trimmed text, not `match.start()`: the `rstrip` drops whitespace before
    # the marker too, and that is text the caller no longer receives.
    trimmed = content[: match.start()].rstrip()
    return trimmed, len(content) - len(trimmed)


def _mapped_error(message: str, status_code: int | None) -> Exception:
    """Translate a runtime error into jaz's taxonomy so retry knows what is hopeless."""
    # Fed only from `SessionErrorData`, the sole channel that carries an HTTP status. The setup
    # calls (`start`, `get_auth_status`, `create_session`) instead fail with the SDK's
    # `JsonRpcError`, whose `code` is a JSON-RPC code and not a status, so there is nothing to
    # map: those stay generic and retryable, which is the right default for a transport that
    # never came up.
    # TODO(#1361): a bad model id also lands there (it is handed to `create_session`), and
    # retrying cannot fix it — it is retried anyway, or reaches here as a 400 and is labelled
    # `UnsupportedParamsError` rather than `NotFoundError`.
    lowered = message.lower()
    if status_code == 401:
        return AuthenticationError(message, llm_provider="copilot")
    if status_code == 403:
        return PermissionDeniedError(message, llm_provider="copilot")
    if status_code == 404:
        return NotFoundError(message, llm_provider="copilot")
    if status_code == 429:
        return RateLimitError(message, status_code=429, llm_provider="copilot")
    if status_code == 400:
        if any(marker in lowered for marker in _CONTEXT_LENGTH_MARKERS):
            return ContextWindowExceededError(message, llm_provider="copilot")
        return UnsupportedParamsError(message, llm_provider="copilot")
    return APIError(message, status_code=status_code, llm_provider="copilot")


def _terminate(client: Any) -> None:
    """Kill the CLI subprocess directly, for when the RPC channel is no longer usable."""
    # Reaches for the SDK's private Popen handle: it exposes no "stop without talking to it",
    # and leaving the child for garbage collection to reap is how orphans happen.
    process = getattr(client, "_process", None)
    if process is not None and process.poll() is None:
        process.terminate()


class _Runtime:
    """A background event loop thread plus the one started ``CopilotClient`` living on it.

    Held apart from :class:`CopilotLLM` so a ``weakref.finalize`` can shut it down without
    keeping the backend itself alive.
    """

    # The loop is dedicated rather than the caller's, for two reasons that both bite in practice:
    # `complete()` is a *sync* method on the ABC (so it needs a loop of its own to drive), and the
    # SDK's JSON-RPC transport binds its connection to the loop it was opened on — reusing an
    # instance across separate `asyncio.run` calls would fail with "event loop is closed". One
    # long-lived loop serves the sync path, the async path and any number of concurrent invokes.

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.client: Any = None
        self._thread = threading.Thread(
            target=self._run, name="jaz-copilot-runtime", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coro: Any) -> Any:
        """Schedule ``coro`` on the runtime loop, returning a ``concurrent.futures.Future``."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def shutdown(self, *, graceful: bool = True) -> None:
        """Stop the CLI subprocess and the loop thread. Idempotent.

        ``graceful=False`` kills the subprocess instead of asking it to stop — see the comment
        on why the finalizer cannot afford the polite version.
        """
        if self.loop.is_closed():
            return
        client, self.client = self.client, None
        if client is not None:
            if graceful:
                try:
                    self.submit(client.stop()).result(timeout=30)
                except Exception:
                    # Best effort: a lingering subprocess beats an exception nobody can act on.
                    _terminate(client)
            else:
                _terminate(client)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=10)
        if not self.loop.is_closed():
            self.loop.close()

    def shutdown_at_exit(self) -> None:
        """Shut down without the graceful stop — the finalizer's entry point."""
        # A graceful stop is an RPC, and the SDK writes RPCs through asyncio's DEFAULT executor,
        # which CPython has already torn down by the time a `weakref.finalize` atexit callback
        # runs (threads are joined before atexit). The call cannot succeed there; it can only
        # log a traceback out of the SDK's internals. So the finalizer kills the child instead,
        # and `close()` stays the way to end a session politely.
        self.shutdown(graceful=False)


class CopilotLLM(BaseLLM):
    """LLM backend that routes completions through GitHub Copilot. See the module docstring
    for authentication, model ids and limitations.

    Args:
        base_directory: Where the Copilot runtime keeps its per-session state.
            Defaults to ``<jaz config dir>/copilot``.
        github_token: GitHub token to authenticate with. Defaults to the SDK's own
            resolution (environment variables, then a signed-in CLI).
        timeout: Seconds to wait for one completion.
        **kwargs: ``model``, the retry settings, and ``reasoning_effort``.
    """

    def __init__(
        self,
        *,
        base_directory: str | None = None,
        github_token: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        reuse_session: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)  # model + retry settings + **request_defaults
        unsupported = sorted(set(self.request_defaults) - _SUPPORTED_REQUEST_PARAMS)
        if unsupported:
            raise UnsupportedParamsError(
                f"CopilotLLM cannot forward {', '.join(unsupported)}: the Copilot runtime builds "
                f"the request body itself. Supported: {', '.join(sorted(_SUPPORTED_REQUEST_PARAMS))}.",
                llm_provider="copilot",
            )
        # Rejected at construction, not per call, so a config that can never be honoured fails
        # once at `jaz.configure` rather than mid-run on the first query.
        self.base_directory = base_directory
        self.github_token = github_token
        self.timeout = timeout
        # Keep one Copilot session alive across `complete()` calls and send only the turns that
        # are new, instead of re-sending the whole conversation to a fresh session every time.
        #
        # This exists because a fresh session per call throws away the provider's prompt cache.
        # Measured on the Terminal-Bench timeout group: `cache_read_tokens` was pinned at exactly
        # 2145 on every call however long the conversation got — the fixed preamble and nothing
        # else — so by turn 15 about 20,000 tokens of history were re-prefilled uncached on every
        # turn, and median call latency rose from 3.8s at turn 0 to 20.6s past turn 15. A direct
        # probe confirmed the cause rather than inferring it: replaying one recorded conversation
        # turn by turn both ways, fresh-per-turn reads 0 cached tokens on every turn while writing
        # thousands, and a reused session reaches 75% cache reads by turn 3.
        #
        # Off by default because it trades away the strict statelessness that `complete()`
        # promises: the session, not the caller, then holds the history. The reuse path below
        # only takes effect when the incoming conversation is exactly the previous one plus new
        # turns, and falls back to a fresh session otherwise, so a hook that rewrites history
        # still gets what it asked for.
        self.reuse_session = reuse_session
        self._runtime: _Runtime | None = None
        self._runtime_lock = threading.Lock()
        self._client_lock: asyncio.Lock | None = None
        self._session_lock: asyncio.Lock | None = None
        #: The live reusable session, and the state needed to decide whether it still applies.
        self._session: Any = None
        self._session_key: tuple[str, str, str] | None = None
        #: Turns the live session has already seen, including its own replies as jaz recorded
        #: them. The next call must extend this exactly, or the session is discarded.
        self._session_turns: list[tuple[str, str]] = []
        self._usages: list[Any] = []
        self._errors: list[Any] = []

    # ------------------------------------------------------------------
    # Runtime lifecycle
    # ------------------------------------------------------------------

    def _ensure_runtime(self) -> _Runtime:
        with self._runtime_lock:
            if self._runtime is None:
                runtime = _Runtime()
                self._client_lock = asyncio.Lock()
                self._session_lock = asyncio.Lock()
                self._runtime = runtime
                # Tied to this backend's lifetime rather than to `atexit`: an eval that builds a
                # backend per config would otherwise pin every CLI subprocess it ever started
                # until the process exits.
                weakref.finalize(self, runtime.shutdown_at_exit)
            return self._runtime

    async def _ensure_client(self, runtime: _Runtime) -> Any:
        assert self._client_lock is not None
        async with self._client_lock:
            if runtime.client is not None:
                return runtime.client
            copilot, _ = _import_sdk()
            base = Path(self.base_directory or ensure_config_dir() / "copilot")
            # The runtime chdirs into it and stores session state there, so it must exist.
            base.mkdir(parents=True, exist_ok=True)
            client = copilot.CopilotClient(
                mode="empty",
                base_directory=str(base),
                working_directory=str(base),
                github_token=self.github_token,
                log_level="error",
            )
            await client.start()
            status = await client.get_auth_status()
            if not status.isAuthenticated:
                await client.stop()
                raise AuthenticationError(
                    "Not signed in to GitHub Copilot. Run `gh auth login` (or `copilot` and "
                    "`/login`), or set COPILOT_GITHUB_TOKEN / GH_TOKEN. "
                    f"Runtime reported: {status.statusMessage!r}.",
                    llm_provider="copilot",
                )
            runtime.client = client
            return client

    def close(self) -> None:
        """Stop the Copilot CLI subprocess and the background loop. Idempotent."""
        with self._runtime_lock:
            runtime, self._runtime = self._runtime, None
        # Dropped without disconnecting: the session lives on the runtime's loop, which is about
        # to be torn down along with the CLI process that hosts it, so there is nothing left to
        # close politely and no loop to await the close on.
        self._session = None
        self._session_key = None
        self._session_turns = []
        if runtime is not None:
            runtime.shutdown()

    # ------------------------------------------------------------------
    # Completions
    # ------------------------------------------------------------------

    def complete(self, model: str, messages: list[Any], **kwargs: Any) -> LLMResponse:
        runtime = self._ensure_runtime()
        return runtime.submit(
            self._complete(runtime, model, messages, **kwargs)
        ).result()

    async def acomplete(
        self, model: str, messages: list[Any], **kwargs: Any
    ) -> LLMResponse:
        runtime = self._ensure_runtime()
        # Bridged onto the caller's loop rather than run in a thread (the base's default), so a
        # concurrent async invoke does not consume a thread-pool slot per in-flight query.
        return await asyncio.wrap_future(
            runtime.submit(self._complete(runtime, model, messages, **kwargs))
        )

    async def _complete(
        self, runtime: _Runtime, model: str, messages: list[Any], **kwargs: Any
    ) -> LLMResponse:
        _, events = _import_sdk()

        timeout = self._resolve_request_timeout(kwargs, self.timeout)
        unsupported = sorted(set(kwargs) - _SUPPORTED_REQUEST_PARAMS)
        if unsupported:
            raise UnsupportedParamsError(
                f"CopilotLLM cannot forward {', '.join(unsupported)}: the Copilot runtime builds "
                f"the request body itself.",
                llm_provider="copilot",
            )
        client = await self._ensure_client(runtime)

        if self.reuse_session and self._session_lock is not None:
            # Skipped when another call already holds the session: two conversations sharing one
            # session would interleave their histories. The loser takes the fresh path below and
            # is merely uncached, not wrong.
            if not self._session_lock.locked():
                async with self._session_lock:
                    return await self._complete_reusing(
                        client, events, model, messages, timeout, **kwargs
                    )

        system_text, prompt, flattened = _split_conversation(messages)

        usages: list[Any] = []
        errors: list[Any] = []

        def collect(event: Any) -> None:
            if isinstance(event.data, events.AssistantUsageData):
                usages.append(event.data)
            elif isinstance(event.data, events.SessionErrorData):
                errors.append(event.data)

        def collected_error() -> Exception | None:
            """The turn's first ``SessionErrorData``, in jaz's taxonomy."""
            if not errors:
                return None
            error = errors[0]
            return _mapped_error(
                f"Copilot runtime error ({error.error_type}): {error.message}",
                error.status_code,
            )

        # A fresh session per call by default, which is what makes this backend obey `complete()`'s
        # stateless contract: jaz hands the full conversation every turn and hooks may rewrite
        # history arbitrarily, so a session that remembered its own turns would answer from a
        # history the caller had already edited. `reuse_session` opts out of that above, and pays
        # for it with the prefix checks in `_complete_reusing` that detect exactly this case.
        # Session creation is local to the CLI runtime — no model call, so no credits.
        session = await client.create_session(
            model=model,
            available_tools=[],
            system_message={"mode": "replace", "content": system_text},
            streaming=False,
            enable_session_telemetry=False,
            on_event=collect,
            **{k: v for k, v in kwargs.items() if v is not None},
        )
        try:
            try:
                response = await session.send_and_wait(prompt, timeout=timeout)
            except TimeoutError as exc:
                raise APIError(
                    f"Copilot did not answer within {timeout}s.", llm_provider="copilot"
                ) from exc
            except Exception as exc:
                # A model/HTTP failure reaches us twice: as the `SessionErrorData` event
                # `collect` saw, and again as the bare `Exception("Session error: ...")` that
                # `send_and_wait` re-raises from it — status code discarded. Re-deriving the
                # taxonomy from the event is what keeps a 401/403/400 failing fast instead of
                # being retried `max_retries` times as a generic (retryable) error.
                mapped = collected_error()
                if mapped is None:
                    raise
                raise mapped from exc
            # Reachable only for an error event that arrived outside `send_and_wait`'s own
            # subscription window (it registers its handler after `create_session` installed
            # `collect`, and raises for every error it does see).
            mapped = collected_error()
            if mapped is not None:
                raise mapped
            content = (
                response.data.content
                if response is not None
                and isinstance(response.data, events.AssistantMessageData)
                else None
            )
        finally:
            await session.disconnect()

        dropped = 0
        if flattened:
            content, dropped = _trim_transcript_continuation(content)
        return self._to_llm_response(content, usages, response, dropped)

    async def _complete_reusing(
        self,
        client: Any,
        events: Any,
        model: str,
        messages: list[Any],
        timeout: float,
        **kwargs: Any,
    ) -> LLMResponse:
        """Answer from the live session when the conversation merely extends what it has seen."""
        # Callers must hold `self._session_lock`: the usage buffers below are per-instance, not
        # per-call, because a reused session's event handler is registered once at creation and
        # cannot be re-bound per turn.
        system_text, turns = _conversation_turns(messages)
        key = (model, system_text, repr(sorted(kwargs.items())))
        # `repr` is a safe key only because `kwargs` is validated against `_SUPPORTED_REQUEST_PARAMS`
        # before this runs, so the values are the small set of scalars the Copilot runtime accepts.
        # An object falling back to `object.__repr__` would embed its address, making the key differ
        # every call and silently degrading reuse to a fresh session per turn — which is why any new
        # supported param must be a scalar, and why `copilot_session_restarted` below exists.

        # The reuse test. `_session_turns` holds what this session was actually told, including
        # the replies exactly as they were handed back to jaz, so an edited or truncated history
        # fails the prefix check and gets a fresh session — which is how `complete()`'s stateless
        # contract survives being answered from session-held state.
        extends = (
            self._session is not None
            and self._session_key == key
            and len(turns) > len(self._session_turns)
            and turns[: len(self._session_turns)] == self._session_turns
        )
        # Reported on the response because a reuse path that rebuilds every turn is otherwise
        # invisible: it returns the same answers and only shows up in the bill. Recorded from the
        # second call on — the first has no session to discard, so it is not a degradation.
        restarted = self._session is not None and not extends
        if not extends:
            await self._drop_session()
            self._session_key = key

        if self._session is None:

            def collect(event: Any) -> None:
                if isinstance(event.data, events.AssistantUsageData):
                    self._usages.append(event.data)
                elif isinstance(event.data, events.SessionErrorData):
                    self._errors.append(event.data)

            self._session = await client.create_session(
                model=model,
                available_tools=[],
                system_message={"mode": "replace", "content": system_text},
                streaming=False,
                enable_session_telemetry=False,
                on_event=collect,
                **{k: v for k, v in kwargs.items() if v is not None},
            )
            self._session_turns = []

        new_turns = turns[len(self._session_turns) :]
        prompt, flattened = _render_turns(new_turns)
        self._usages.clear()
        self._errors.clear()

        def collected_error() -> Exception | None:
            """The turn's first ``SessionErrorData``, in jaz's taxonomy."""
            if not self._errors:
                return None
            error = self._errors[0]
            return _mapped_error(
                f"Copilot runtime error ({error.error_type}): {error.message}",
                error.status_code,
            )

        try:
            try:
                response = await self._session.send_and_wait(prompt, timeout=timeout)
            except TimeoutError as exc:
                raise APIError(
                    f"Copilot did not answer within {timeout}s.", llm_provider="copilot"
                ) from exc
            except Exception as exc:
                # The same double delivery the fresh-session path handles: `send_and_wait`
                # re-raises a model/HTTP failure as a bare `Exception` with the status discarded,
                # so the taxonomy has to be re-derived from the event `collect` saw. Without this
                # a 401 is retried `max_retries` times instead of failing fast.
                mapped = collected_error()
                if mapped is None:
                    raise
                raise mapped from exc
            mapped = collected_error()
            if mapped is not None:
                raise mapped
            content = (
                response.data.content
                if response is not None
                and isinstance(response.data, events.AssistantMessageData)
                else None
            )
        except BaseException:
            # The session's history is now unknown — the send may have landed, may not have — so
            # it cannot be trusted to prefix-match anything. Drop it and let the next call rebuild.
            await self._drop_session()
            raise

        dropped = 0
        # Session markers are checked unconditionally, unlike the fresh-session path's
        # `flattened` gate: a reused send is usually a single bare turn, but the model can read
        # the whole conversation in the session's own history, so there is always a format for it
        # to continue. `[[role]]` is still checked as a fallback for the first send, which does
        # carry the flattened transcript.
        content, dropped = _trim_transcript_continuation(content, _SESSION_MARKER_LINE)
        if not dropped and flattened:
            content, dropped = _trim_transcript_continuation(content)

        if dropped:
            # The session remembers the untrimmed reply while jaz records the trimmed one, so the
            # two histories have diverged and no later prefix check could detect it. Rebuilding is
            # the only safe move; it costs one uncached turn.
            await self._drop_session()
        else:
            self._session_turns = [*turns, ("assistant", content or "")]

        return self._to_llm_response(
            content, list(self._usages), response, dropped, session_restarted=restarted
        )

    async def _drop_session(self) -> None:
        """Disconnect the reusable session, if any, and forget its history."""
        session, self._session = self._session, None
        self._session_turns = []
        if session is not None:
            try:
                await session.disconnect()
            except Exception:
                # Best-effort: the session is being abandoned either way, and a failure to close
                # a dying session must not mask the error that led here.
                pass

    @staticmethod
    def _to_llm_response(
        content: str | None,
        usages: list[Any],
        raw: Any = None,
        dropped_chars: int = 0,
        session_restarted: bool = False,
    ) -> LLMResponse:
        """Fold the turn's ``assistant.usage`` events into one :class:`LLMResponse`.

        Summed rather than taken from the last event because one prompt can span several model
        calls (a server-side retry, an auto model switch), and every one of them is billed.
        """
        prompt_tokens = sum(u.input_tokens or 0 for u in usages)
        completion_tokens = sum(u.output_tokens or 0 for u in usages)
        cached_tokens = sum(u.cache_read_tokens or 0 for u in usages)
        cache_write_tokens = sum(u.cache_write_tokens or 0 for u in usages)
        reasoning_tokens = sum(u.reasoning_tokens or 0 for u in usages)
        nano_credits = sum(
            u.copilot_usage.total_nano_aiu
            for u in usages
            if u.copilot_usage is not None
        )
        credits = nano_credits / _NANO

        extra: dict[str, Any] = {"copilot_ai_credits": credits}
        if dropped_chars:
            # Recorded rather than logged: it is the one thing a trajectory cannot show after the
            # fact, since the dropped text never reaches the protocol or the history.
            extra["copilot_dropped_continuation_chars"] = dropped_chars
        if session_restarted:
            # Only recorded when it happened, so a healthy reuse run adds nothing to every step
            # while a run that rebuilds on every turn is visible as a flag on every turn.
            extra["copilot_session_restarted"] = True
        if cache_write_tokens:
            extra["cache_creation_input_tokens"] = cache_write_tokens
        if reasoning_tokens:
            extra["reasoning_tokens"] = reasoning_tokens
        if len(usages) > 1:
            # How many billed model calls one `complete()` actually cost. The sums above already
            # fold them together, which makes a retry storm invisible: a Terminal-Bench turn
            # reported 32,295 completion tokens against 377 characters of returned content and
            # took 274s, and without this count there is no way to tell an enormous single
            # response from a dozen silently retried ones. Only recorded when it exceeds 1, so
            # the common case adds nothing to every trajectory step.
            extra["copilot_model_calls"] = len(usages)
        if usages:
            # The id actually served, which `model="auto"` and auto-switch make differ from the
            # one requested.
            extra["copilot_model"] = usages[-1].model
            if usages[-1].finish_reason:
                extra["finish_reason"] = usages[-1].finish_reason

        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            cost_usd=credits * USD_PER_AI_CREDIT,
            extra=extra,
            # The assistant-message event, not the usage event: `raw_response` is defined as the
            # payload a protocol reads for refusal / finish_reason / tool_calls, so it has to be
            # the one carrying the message. The event SDK has no single "raw response" object,
            # and nothing is lost by the choice — the usage events' billing and served-model
            # fields are already folded into `extra` and the token counts above.
            raw_response=raw,
        )

    def can_report_cost(self) -> bool:
        """Always ``True`` — GitHub reports the credits it billed for every call."""
        # Overridden away from the base's pricing-table lookup: that table is keyed by vendor
        # model ids and prices at vendor rates, neither of which describes what Copilot charged.
        return True

    @property
    def non_retryable_exceptions(self) -> tuple[type[BaseException], ...]:
        """The base set plus ``ImportError`` — a missing SDK is not a transient failure."""
        # The SDK import is deferred to the first call (it is an optional extra), so "extra not
        # installed" surfaces as an exception from `complete` like any API error — and the base
        # retries anything it does not recognise. That made an install mistake, the single most
        # likely failure for a new user of this backend, present as ~5 minutes of silence
        # (10 retries backing off to 60 s) before the message naming the fix finally appeared.
        return (*super().non_retryable_exceptions, ImportError)

    def get_model_info(self) -> dict[str, Any]:
        """Empty — Copilot's models are not the bundled pricing table's models."""
        # A bare `gpt-5-mini` *would* hit the table, and would then describe this backend's calls
        # at OpenAI's rates and OpenAI's context window. Reporting nothing is the honest answer.
        return {}

    def list_models(self) -> list[str]:
        """The model ids this GitHub account can use, as the Copilot runtime reports them."""
        runtime = self._ensure_runtime()

        async def _list() -> list[str]:
            client = await self._ensure_client(runtime)
            return [m.id for m in await client.list_models()]

        return runtime.submit(_list()).result()
