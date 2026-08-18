"""The SGLang backend — token-in/token-out completions over SGLang's native ``/generate``.

Where the text backends send OpenAI-format messages and let the server own the chat
template, :class:`SGLangLLM` renders the conversation to token ids itself (via a
:class:`~jaz.llm.chat_format.ChatFormat`) and posts ``input_ids`` to ``/generate``,
receiving the sampled ids and per-token logprobs back. Every response carries a
:class:`~jaz.tokens.TurnRecord` on ``LLMResponse.tokens`` — the exact tokens sent and
sampled — which is what makes agent runs recordable as training-grade rollouts
(see ``jaz.hooks.builtin.rollout.RolloutRecorder``).

Usage::

    jaz.configure(llm=SGLangLLM(model="Qwen/Qwen3-8B",
                                base_url="http://localhost:30000"))

``chat_format`` defaults to ``HFChatFormat(model)``, which needs the ``jaz[tokens]``
extra (HuggingFace ``transformers``); pass a constructed :class:`ChatFormat` to avoid the
dependency.

Reasoning models (hidden thinking traces):
    For models that emit an inline reasoning trace (Qwen3's ``<think>…</think>``), the
    trace is treated the way reasoning-model APIs treat it — it is **hidden from the
    visible message**: ``LLMResponse.content`` carries only the text after the trace, so
    the REPL never executes reasoning prose and later turns' context does not replay it.
    The token sidecar (``LLMResponse.tokens``) is unaffected: the sampled ids, logprobs,
    and ``content_text`` keep the full generation, trace included, so training still
    optimizes the reasoning tokens. Configure with ``reasoning_tags=(open, close)``
    (default ``("<think>", "</think>")``); pass ``reasoning_tags=None`` to disable and
    expose raw generations. An *unterminated* trace (generation truncated mid-thought)
    strips to an empty visible message rather than leaking a partial trace.

    Only a *leading* trace is stripped — one opening at the start of the generation,
    matching how reasoning models emit them. Known limitations of textual matching: a
    generation that literally *starts* with the open-tag string is indistinguishable from
    a trace (a mid-text literal, e.g. agent code printing ``"<think>"``, is safe and
    stays visible); a hypothetical mid-text trace is left visible; a nested trace
    (``<think>a<think>b</think>c</think>``) leaks its tail past the first close tag; and
    a template that pre-appends the open tag to the *prompt* (DeepSeek-R1 style), so the
    generation begins mid-trace with no open tag of its own, is not detected — this class
    supports open-tag-emitting families (Qwen3). If any of these bite, pass
    ``reasoning_tags=None`` and strip caller-side.

Limitations:
    * No cost reporting: local/self-hosted serving has no meaningful per-token price, so
      ``cost_usd`` is ``None`` and :meth:`can_report_cost` returns ``False`` (a
      ``cost_budget`` on this backend is rejected pre-flight).
    * Non-streaming only: ``/generate`` is called in one shot — fine for rollout
      production; revisit if interactive use matters.
    * The weight-update/memory helpers are SGLang-specific transport and live on this
      class, not the ``BaseLLM`` ABC.
    * One event loop per instance: the async client's pooled connections bind to the loop
      they were opened on, so reusing an instance across separate ``asyncio.run`` calls can
      raise "event loop is closed" — construct a fresh instance per loop (training's single
      long-lived loop is the design point). :meth:`close`/:meth:`aclose` release the pools.
"""

# Design rationale (maintainers; token-native rollouts design "Option C — messages own
# their tokens, turns own their sends", PR #1099; executive call, user, 2026-08-13). Key
# properties this backend is built around:
#
# * Fully stateless and mode-less: one instance serves all invokes concurrently, holds no
#   per-invoke state, reads no ambient context, and behaves identically whether or not a
#   RolloutRecorder is listening. ("Stateless" = no per-invoke state; the pooled HTTP
#   clients below are connection reuse, not state.)
# * Retry safety is structural: the agent calls through complete_with_retry, so
#   complete() may run several times for one logical turn. Each attempt builds its stamps
#   and TurnRecord from its own server response and mutates nothing shared; a failed
#   attempt's objects are dropped with it. (The rejected ledger design could double-append
#   a sampled segment on retry; that hazard does not exist here.)
# * Incremental append aligns with SGLang's radix prefix cache, so the stamped-reuse path
#   is also the server's fast path. First calls and un-stamped runs full-render every
#   message — correct, just cache-cold.

from __future__ import annotations

import re
import uuid
from typing import Any

import httpx

from jaz.tokens import TOKENS_KEY, TokenStamp, TurnRecord, tokens_of

from .chat_format import ChatFormat, HFChatFormat
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
from .registry import register_llm

# Body markers of SGLang's over-length 400 ("the input (N tokens) is longer than the
# model's context length (M tokens)" family). Matched lowercased. Prose matching is
# drift-prone; a missed marker falls through to UnsupportedParamsError (non-retryable) —
# mislabeled, but no retry loop.
_CONTEXT_LENGTH_MARKERS = ("context length", "context_len", "longer than the maximum")

_DEFAULT_TIMEOUT_SECONDS = 600.0  # generation can legitimately take minutes


def _mapped_exception(exc: BaseException) -> Exception | None:
    """Translate an httpx failure into jaz's error taxonomy, or ``None`` to re-raise.

    Without this the base retryer's non-retryable check never matches, so exactly the
    failures the taxonomy fails fast on (bad request, auth, unknown route) would be
    retried with backoff. 429/5xx/transport errors map to retryable types.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        body = exc.response.text
        message = f"SGLang server returned {status}: {body[:500]}"
        if status == 400:
            if any(m in body.lower() for m in _CONTEXT_LENGTH_MARKERS):
                jaz_type: type = ContextWindowExceededError
            else:
                jaz_type = UnsupportedParamsError
        elif status == 401:
            jaz_type = AuthenticationError
        elif status == 403:
            jaz_type = PermissionDeniedError
        elif status == 404:
            jaz_type = NotFoundError
        elif status == 429:
            jaz_type = RateLimitError
        else:  # 5xx and anything else: treat as retryable server trouble
            jaz_type = APIError
        return jaz_type(message, status_code=status, llm_provider="sglang")
    if isinstance(exc, httpx.TimeoutException | httpx.TransportError):
        # Connect/read failures are the routine retryable case for a local server that is
        # (re)starting or briefly saturated.
        return APIError(
            f"SGLang transport error: {exc}", status_code=None, llm_provider="sglang"
        )
    return None


@register_llm("sglang")
class SGLangLLM(BaseLLM):
    """Token-native LLM backend for SGLang's ``/generate`` API. See the module docstring
    for usage and limitations."""

    # Declares the token sidecar so to_wire_messages leaves TokenStamps on this backend's
    # wire list — the stamped-reuse path below reads them. Provenance stays stripped.
    consumes_internal_keys = frozenset({TOKENS_KEY})

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:30000",
        chat_format: ChatFormat | str | None = None,
        reasoning_tags: tuple[str, str] | list[str] | None = ("<think>", "</think>"),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)  # model + retry settings + **request_defaults
        self.base_url = base_url.rstrip("/")
        # Hidden-reasoning interface (executive call, user, 2026-08-15): the standard
        # contract for thinking models is that the trace is hidden from the output —
        # reasoning stays ON and gets gradient, but only the post-trace text is the
        # message. Three distinct surfaces, split deliberately:
        #   * execution + cross-turn context: LLMResponse.content is stripped, so the
        #     agent's REPL parse and the history it appends never see the trace (the
        #     stale-stamp guard in _prepare_stamps then re-renders the stripped text
        #     fresh, keeping trace ids out of later input_ids);
        #   * loss: the sampled TokenStamp keeps the FULL ids/logprobs/content_text, so
        #     recorded rollouts still train the reasoning tokens (mask 1).
        # Consequence: multi-turn hidden-reasoning rollouts are non-monotone BY DESIGN
        # (the next turn's context is not an extension of prev sent+sampled), so
        # Rollout.to_flat() rejects them — export via to_turn_samples() instead.
        # Default ON: shipping the raw trace as .content fed `<think>` prose to the REPL
        # (SyntaxError every thinking turn) and tanked Qwen3-0.6B GSM8K greedy pass@1
        # 0.27→0.147 with 69% aborts (measured 2026-08-15, N=100×3); stamped-full/
        # visible-stripped restored it. Alternatives rejected: enable_thinking=False at
        # the template (kills the reasoning that helps: 0.220 but degenerate loops) and
        # /no_think (Qwen3-0.6B ignores the soft switch).
        if reasoning_tags is not None:
            open_tag, close_tag = reasoning_tags  # 2-tuple contract; raises on misuse
            if not open_tag or not close_tag:
                raise ValueError("reasoning_tags must be two non-empty strings or None")
            # LEADING trace only (executive call, user, 2026-08-15, on review of #1177):
            # matches how reasoning models actually emit traces (Qwen3: start of the
            # generation, nowhere else — mid-text traces were never observed). Bounds the
            # literal-tag false positive to generations that *start* with the open tag; a
            # mid-text occurrence (agent code printing "<think>") stays visible. A closed
            # leading trace strips through the close tag (+ whitespace); an unterminated
            # leading open tag (truncation mid-thought) strips to end-of-text so a
            # partial trace never leaks into the visible message.
            #
            # Precedent: hiding the trace from the message is the serving-stack standard —
            # SGLang --reasoning-parser and vLLM reasoning outputs both split
            # reasoning_content out of content
            # (https://docs.sglang.ai/advanced_features/separate_reasoning.html,
            # https://docs.vllm.ai/en/latest/features/reasoning_outputs/), and the Qwen3
            # model card mandates history without thinking content
            # (https://huggingface.co/Qwen/Qwen3-0.6B). Those parsers are strictly MORE
            # permissive than this leading-tag match: they also detect a trace whose open
            # tag lives in the *prompt* (DeepSeek-R1-style templates pre-append <think>,
            # so the generation starts mid-trace) — documented here as an unsupported
            # family (module docstring) rather than guessed at.
            self._reasoning_closed: re.Pattern[str] | None = re.compile(
                r"\A\s*" + re.escape(open_tag) + r".*?" + re.escape(close_tag) + r"\s*",
                re.DOTALL,
            )
            self._reasoning_open: re.Pattern[str] | None = re.compile(
                r"\A\s*" + re.escape(open_tag) + r".*\Z", re.DOTALL
            )
        else:
            self._reasoning_closed = None
            self._reasoning_open = None
        # Public introspection surface mirroring the constructor param (deliberately
        # unread internally — the compiled patterns above are the working form).
        self.reasoning_tags = tuple(reasoning_tags) if reasoning_tags else None
        # `chat_format` may be a ChatFormat, a tokenizer name, or None (= the model id).
        # String/None resolve lazily on first call so constructing the backend (registry,
        # config files, `import jaz`) never pays the transformers import or tokenizer
        # download — and never *requires* jaz[tokens] unless the default is actually used.
        self._chat_format: ChatFormat | None = (
            chat_format if isinstance(chat_format, ChatFormat) else None
        )
        self._chat_format_spec: str | None = (
            chat_format if isinstance(chat_format, str) else None
        )
        # Instance-held pooled clients (connection reuse at hundreds of concurrent
        # rollouts). Construction is cheap and opens no sockets. Tests replace these with
        # httpx.MockTransport-backed clients. `_ahttp`'s pooled connections bind to the
        # event loop they are first used on, so one instance assumes one long-lived loop
        # (training's design point); a per-loop client keyed by asyncio.get_running_loop()
        # would lift that but adds loop-bookkeeping state to a deliberately stateless
        # backend, so the single-loop assumption is documented instead (see close/aclose).
        self._http = httpx.Client()
        self._ahttp = httpx.AsyncClient()

    def _format(self) -> ChatFormat:
        if self._chat_format is None:
            self._chat_format = HFChatFormat(self._chat_format_spec or self.get_model())
        return self._chat_format

    # Explicit cleanup rather than context managers: the stateless design shares one
    # instance across concurrent invokes, so pool lifetime is the constructing driver's
    # call, not a per-request `with`. Without these the pools live until GC — leaked
    # sockets for long-lived drivers that build several backends, and ResourceWarnings
    # under `-W error`.
    def close(self) -> None:
        """Release the pooled HTTP connections held by this instance."""
        self._http.close()

    async def aclose(self) -> None:
        """Release the pooled async HTTP connections held by this instance."""
        await self._ahttp.aclose()

    # ------------------------------------------------------------------
    # complete / acomplete
    # ------------------------------------------------------------------

    def complete(self, model: str, messages: list[Any], **kwargs: Any) -> LLMResponse:
        fmt = self._format()
        stamps = self._prepare_stamps(messages, fmt)
        timeout = self._resolve_request_timeout(
            kwargs, default=_DEFAULT_TIMEOUT_SECONDS
        )
        payload = self._build_payload(stamps, fmt, kwargs)
        try:
            response = self._http.post(
                f"{self.base_url}/generate", json=payload, timeout=timeout
            )
            response.raise_for_status()
        except Exception as exc:
            mapped = _mapped_exception(exc)
            if mapped is not None:
                raise mapped from exc
            raise
        return self._finish(response.json(), model, stamps, fmt)

    async def acomplete(
        self, model: str, messages: list[Any], **kwargs: Any
    ) -> LLMResponse:
        # Native async, not the base's asyncio.to_thread(complete) fallback: a blocking
        # thread per in-flight generation is untenable at training-batch scale (hundreds
        # of concurrent rollouts). Shares every non-transport helper with complete().
        fmt = self._format()
        stamps = self._prepare_stamps(messages, fmt)
        timeout = self._resolve_request_timeout(
            kwargs, default=_DEFAULT_TIMEOUT_SECONDS
        )
        payload = self._build_payload(stamps, fmt, kwargs)
        try:
            response = await self._ahttp.post(
                f"{self.base_url}/generate", json=payload, timeout=timeout
            )
            response.raise_for_status()
        except Exception as exc:
            mapped = _mapped_exception(exc)
            if mapped is not None:
                raise mapped from exc
            raise
        return self._finish(response.json(), model, stamps, fmt)

    # ------------------------------------------------------------------
    # Shared request/response helpers (sync and async cannot drift)
    # ------------------------------------------------------------------

    def _prepare_stamps(self, messages: list[Any], fmt: ChatFormat) -> list[TokenStamp]:
        """One TokenStamp per wire message, in order: reuse or fresh-render."""
        stamps: list[TokenStamp] = []
        for message in messages:
            content = message.get("content")
            stamp = tokens_of(message)
            # Stamped-reuse iff the content still matches the stamp — the stale-stamp
            # guard (runtime guard 2): a hook that edited a stamped message's content
            # after stamping must not have the old ids silently shadow the edit. The
            # mismatch re-renders under a fresh message_id, which surfaces at export as a
            # monotonicity break — the same representable event as any other history edit
            # — instead of a silent divergence between hook intent and what the model saw.
            if stamp is not None and stamp.content_text == content:
                stamps.append(stamp)
            else:
                stamps.append(
                    TokenStamp(
                        role=str(message.get("role", "")),
                        ids=tuple(fmt.render_message(message)),
                        content_text="" if content is None else str(content),
                        message_id=uuid.uuid4().hex,
                    )
                )
        return stamps

    def _build_payload(
        self, stamps: list[TokenStamp], fmt: ChatFormat, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        input_ids = list(fmt.initial_ids())
        for stamp in stamps:
            input_ids.extend(stamp.ids)
        input_ids.extend(fmt.generation_prompt_ids())
        return {
            "input_ids": input_ids,
            "sampling_params": self._sampling_params(kwargs),
            "return_logprob": True,
        }

    def _sampling_params(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Map jaz/OpenAI-style request kwargs onto SGLang ``sampling_params``."""
        params = dict(kwargs)
        # OpenAI spelling → SGLang spelling; an explicit max_new_tokens wins. Everything
        # else (temperature, top_p, top_k, min_p, stop, stop_token_ids, penalties, seed,
        # and any key we don't know) passes through verbatim — an unknown param fails on
        # the server as a mapped 400, fail-loud rather than silently dropped.
        if "max_tokens" in params:
            params.setdefault("max_new_tokens", params.pop("max_tokens"))
        # Forced after the merge so no config can undo them (LiteLLM's num_retries=0
        # precedent): the RL sampling-correctness convention. no_stop_trim keeps the stop
        # token in output_ids; skip_special_tokens=False keeps output_ids exactly what was
        # sampled. The *text* the protocol parses is decoded separately with specials
        # stripped (ChatFormat.decode) — recorded ids and parsed text serve different
        # masters.
        params["no_stop_trim"] = True
        params["skip_special_tokens"] = False
        return params

    def _finish(
        self,
        data: dict[str, Any],
        model: str,
        stamps: list[TokenStamp],
        fmt: ChatFormat,
    ) -> LLMResponse:
        meta = data.get("meta_info") or {}
        # The output_token_logprobs triplets [logprob, token_id, text|None] are the
        # *portable* source of sampled ids + logprobs — top-level `output_ids` is newer
        # and absent in older servers; verl/slime/AReaL all parse the triplets.
        triplets = meta.get("output_token_logprobs") or []
        output_ids = [int(t[1]) for t in triplets]
        logprobs = tuple(float(t[0]) for t in triplets)
        finish_reason = meta.get("finish_reason")
        # Structured ({"type": "stop", ...}) in current servers, a bare string in older
        # ones — normalize to the string.
        if isinstance(finish_reason, dict):
            finish_reason = finish_reason.get("type")
        weight_version = meta.get("weight_version")
        if weight_version is not None:
            weight_version = str(weight_version)

        gen_prompt = fmt.generation_prompt_ids()
        content = fmt.decode(output_ids)
        # Visible message vs sidecar (see the reasoning_tags rationale in __init__):
        # .content hides the reasoning trace; the sampled stamp below keeps the full
        # decode as content_text, both because the stamp documents what its ids decode to
        # and because a stripped content_text would make the stale-stamp guard *match*
        # the stripped history text and replay the trace ids into the next context.
        visible = content
        if self._reasoning_closed is not None and self._reasoning_open is not None:
            visible = self._reasoning_closed.sub("", visible)
            visible = self._reasoning_open.sub("", visible)
        # The sampled stamp's ids are the turn's *full* contribution to a future context:
        # the generation-prompt tokens that opened it, the verbatim sampled ids (including
        # the stop token — no_stop_trim), and any trailing template separator the format
        # appends after the stop token (the "\n" after <|im_end|> in ChatML-family
        # templates; #1134). Capturing the framing here, from the call that produced it, is
        # what lets later contexts replay the turn byte-exactly, matching a full-template
        # render. The separator sits *outside* sampled_span, so the loss mask still covers
        # only the sampled ids.
        #
        # This "separator in the sequence but *outside* the loss span" split is the convention of
        # the trainers this backend exports to, so a JAZ rollout replayed there is byte-exact with
        # correct masking. verl inserts it as a non-loss boundary token ("boundary tokens ... must
        # not carry loss") in verl/utils/tokenizer/continuous_token.py
        # (https://github.com/volcengine/verl); slime and AReaL reinsert it from a full-template
        # re-render and mask it out, in slime/agent/trajectory.py (https://github.com/THUDM/slime)
        # and areal/experimental/openai/client.py (https://github.com/inclusionAI/AReaL). HF/TRL's
        # {% generation %} assistant mask has the same shape. (Only the string-diff maskers —
        # OpenRLHF, NeMo-RL — train the separator, as a side effect of the diff, not by design.)
        #
        # Only a natural stop leaves the end-of-turn token in output_ids; a length / abort /
        # content-filter finish truncates mid-turn with no stop token, so the *post-stop*
        # separator does not belong — appending it would inject a phantom token matching
        # neither what was sampled nor a real full-template render of the (unfinished) turn.
        suffix = fmt.sampled_suffix_ids() if finish_reason == "stop" else ()
        sampled = TokenStamp(
            role="assistant",
            ids=tuple(gen_prompt) + tuple(output_ids) + tuple(suffix),
            content_text=content,
            sampled_span=(len(gen_prompt), len(gen_prompt) + len(output_ids)),
            logprobs=logprobs,
            weight_version=weight_version,
            finish_reason=finish_reason,
        )
        record = TurnRecord(
            sent=tuple(stamps),
            sampled=sampled,
            iteration=-1,  # unknowable here; RolloutRecorder rewrites from LLMQueryExit
            model=model,
            initial=tuple(fmt.initial_ids()),
        )
        extra: dict[str, Any] = {}
        if weight_version is not None:
            extra["weight_version"] = weight_version
        # Built directly rather than via finalize(): there is no wire CompletionResponse
        # (SGLang's envelope is not OpenAI-shaped) and pricing does not apply.
        return LLMResponse(
            content=visible,
            prompt_tokens=meta.get("prompt_tokens"),
            completion_tokens=meta.get("completion_tokens"),
            cached_tokens=meta.get("cached_tokens"),
            cost_usd=None,
            extra=extra,
            raw_response=data,
            tokens=record,
        )

    def can_report_cost(self) -> bool:
        # Local/self-hosted serving has no meaningful per-token price; report "cannot"
        # so a cost_budget is rejected pre-flight rather than found unenforceable mid-run.
        return False

    # ------------------------------------------------------------------
    # Weight-update / memory orchestration (SGLang-specific transport)
    # ------------------------------------------------------------------

    # Thin typed wrappers over the server's training-loop endpoints — how a trainer swaps
    # fresh policy weights into the rollout server between batches and parks/restores its
    # GPU memory around its own training steps. On the instance, not the BaseLLM ABC: this
    # is transport-specific surface (a vLLM backend would ship its own equivalents), and the
    # rollout record stays backend-neutral. `weight_version` on every sampled stamp is
    # what lets a trainer attribute each segment to the policy that produced it when
    # weights update mid-batch.

    def _post(self, endpoint: str, payload: dict[str, Any] | None = None) -> Any:
        try:
            response = self._http.post(
                f"{self.base_url}/{endpoint}",
                json=payload or {},
                timeout=_DEFAULT_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except Exception as exc:
            mapped = _mapped_exception(exc)
            if mapped is not None:
                raise mapped from exc
            raise
        try:
            return response.json()
        except ValueError:
            return None  # some endpoints return an empty/plain body

    def update_weights_from_disk(self, model_path: str, **options: Any) -> Any:
        """Load fresh weights from ``model_path`` into the serving engine."""
        return self._post(
            "update_weights_from_disk", {"model_path": model_path, **options}
        )

    def init_weights_update_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
    ) -> Any:
        """Set up the distributed group for :meth:`update_weights_from_distributed`."""
        return self._post(
            "init_weights_update_group",
            {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
                "group_name": group_name,
                "backend": backend,
            },
        )

    def update_weights_from_distributed(self, **payload: Any) -> Any:
        """Receive weights over the distributed group set up by
        :meth:`init_weights_update_group`."""
        return self._post("update_weights_from_distributed", payload)

    def update_weights_from_tensor(self, **payload: Any) -> Any:
        """Load weights from serialized tensors (colocated trainer/server setups)."""
        return self._post("update_weights_from_tensor", payload)

    def release_memory_occupation(self) -> Any:
        """Release the engine's GPU memory (weights + KV cache) to the trainer."""
        return self._post("release_memory_occupation")

    def resume_memory_occupation(self) -> Any:
        """Re-acquire GPU memory released by :meth:`release_memory_occupation`."""
        return self._post("resume_memory_occupation")

    def pause_generation(self) -> Any:
        """Pause in-flight generation (e.g. around a weight update)."""
        return self._post("pause_generation")

    def continue_generation(self) -> Any:
        """Resume generation paused by :meth:`pause_generation`."""
        return self._post("continue_generation")
