"""Chat formats: caller-owned, per-message chat templating for token-native backends.

With token-in serving (``SGLangLLM``), the caller — not the server — owns the chat
template: it must render each message to token ids itself and send ``input_ids``. A
:class:`ChatFormat` is that renderer, factored per message so a conversation's ids can be
built *incrementally* — each message rendered exactly once, on first send — instead of
re-templating the whole conversation every call (which is both slow and, for some
templates, not even reproducible: ``apply_chat_template`` over a full conversation is not
the concatenation of per-turn renders).

Pass one to a token-native backend's constructor::

    llm = SGLangLLM(model="Qwen/Qwen3-8B", base_url="http://localhost:30000",
                    chat_format=HFChatFormat("Qwen/Qwen3-8B"))

``HFChatFormat`` (a HuggingFace ``transformers`` tokenizer wrapper) requires the optional
``jaz[tokens]`` extra.
"""

# A constructor argument of token-native backends, not a `Config` component: only they have
# any use for it, and binding it to the backend keeps "swap the model" a one-object change.
# The extra is named `tokens`, not the design draft's `tito`, per review — existing extras
# are descriptive.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .base import MessageDict

if TYPE_CHECKING:
    # `transformers` is an optional dependency (the jaz[tokens] extra), so it cannot be
    # imported for typing either — annotate the tokenizer loosely instead of forcing the
    # dependency on every type-check run.
    PreTrainedTokenizerBase = Any


def _transformers() -> Any:
    """Import ``transformers`` lazily, only on the string-tokenizer path.

    Deferred so that importing :mod:`jaz.llm` (and constructing an ``HFChatFormat`` around
    an already-built tokenizer object) never pays for — or requires — the dependency.
    """
    try:
        import transformers  # pyright: ignore[reportMissingImports]
    except ImportError as e:
        raise ImportError(
            "HFChatFormat with a tokenizer name requires the 'transformers' package: "
            "pip install 'jaz[tokens]'"
        ) from e
    return transformers


class ChatFormat(ABC):
    """Renders chat messages to token ids, one message at a time.

    A conversation's generation input is assembled as::

        initial_ids() + render_message(m_1) + ... + render_message(m_n) + generation_prompt_ids()

    The invariant implementations aim for: that sum equals
    ``apply_chat_template(messages, add_generation_prompt=True)`` wherever the template is
    compositional across turns. Where a template is *not* compositional (e.g. templates
    that strip earlier-turn ``<think>`` blocks on full re-render), the incremental form is
    by definition what the model actually saw turn by turn, and is the recorded truth.
    """

    @abstractmethod
    def initial_ids(self) -> list[int]:
        """Ids that open every conversation (BOS / template preamble); may be empty."""

    @abstractmethod
    def render_message(self, message: MessageDict) -> list[int]:
        """The ids for one fully framed turn (role header + content + turn separator)."""

    @abstractmethod
    def generation_prompt_ids(self) -> list[int]:
        """The ids that cue the model to answer (e.g. ``<|im_start|>assistant\\n``)."""

    @abstractmethod
    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode sampled ids to text with special tokens stripped — the text the
        protocol parses."""

    def sampled_suffix_ids(self) -> list[int]:
        r"""Template ids that follow the stop token at the end of a sampled assistant turn.

        A sampled turn's ids run up to *and including* the stop token; a full-template
        render of that turn may append a separator after it (the ``\n`` after ``<|im_end|>``
        in ChatML-family templates). Returning those ids lets a recorded turn replay as an
        *exact* prefix of a later context, matching what ``apply_chat_template`` produces.

        Concrete with an empty default rather than abstract: an empty suffix is the correct
        answer for any frame that has none (most of them — Llama-3's ``<|eot_id|>`` is already
        a turn's last token), so a format only overrides when its template adds a separator.
        """
        return []


class HFChatFormat(ChatFormat):
    """A :class:`ChatFormat` backed by a HuggingFace ``transformers`` tokenizer.

    ``tokenizer`` is a model name/path (loaded via ``AutoTokenizer.from_pretrained`` —
    requires the ``jaz[tokens]`` extra) or an already-constructed tokenizer object (no
    ``transformers`` import needed). Construction fails (``ValueError``) if the tokenizer
    has no ``chat_template``, or if the template is not compositional per message (see
    below).

    Limitations:

    - Templates that inject a *default system prompt* per turn (e.g. the Qwen2.5 and
      SmolLM2 families) are **not compositional**: rendering each message alone folds the
      default system block into every turn, so the incremental ids diverge from the real
      conversation. Such tokenizers are **rejected at construction** rather than silently
      recording ids the model never saw — there is no per-message workaround (an explicit
      system message does not suppress the per-turn re-injection). Use a template without
      per-turn default injection (e.g. Qwen3) or supply a custom :class:`ChatFormat`.
    - Templates whose turn framing is plain text rather than special tokens (Vicuna-style
      ``USER:``) may re-encode differently at the preamble seam; the compositionality is
      only guaranteed when the frame opens with a special token (ChatML, Llama-3, ...).
    - Reasoning templates that drop earlier-turn ``<think>`` blocks on full re-render
      diverge from the incremental form by design; the incremental form is what the model
      actually saw (see :class:`ChatFormat`).
    """

    # All template calls use tokenize=False and diff/strip in *text* space, then
    # encode(..., add_special_tokens=False). This is "encodes with add_special_tokens=False
    # (BOS never doubled)" made concrete, and it sidesteps transformers' historical
    # double-BOS ambiguity inside tokenizing apply_chat_template variants.

    def __init__(self, tokenizer: str | PreTrainedTokenizerBase) -> None:
        if isinstance(tokenizer, str):
            tokenizer = _transformers().AutoTokenizer.from_pretrained(tokenizer)
        if getattr(tokenizer, "chat_template", None) is None:
            # Fail at construction, not first render: a chat-template-less tokenizer would
            # otherwise surface as an opaque jinja error mid-run.
            raise ValueError(
                "HFChatFormat requires a tokenizer with a chat_template; "
                f"{getattr(tokenizer, 'name_or_path', tokenizer)!r} has none"
            )
        # Any, not a transformers type: the dependency is optional, and the fake-tokenizer
        # duck type used in tests is equally valid here.
        self.tokenizer: Any = tokenizer
        self._initial_ids: list[int] | None = None
        self._generation_prompt_ids: list[int] | None = None
        self._sampled_suffix_ids: list[int] | None = None
        self._check_compositional()

    def _check_compositional(self) -> None:
        # Third construction-time guard (alongside the missing-chat_template check above and
        # the non-suffix generation-prompt check in generation_prompt_ids): reject templates
        # whose per-message render does not compose into the whole-conversation render. The
        # motivating case is default-system-prompt injection — Qwen2.5 / SmolLM2 re-emit the
        # full default system block on every single-message render, so `initial + Σ render`
        # carries ~3x the real tokens (measured: 87 vs 24 ids). Token-in serving would then
        # POST ids the model never saw and RolloutRecorder would freeze them as training
        # truth, corrupting every rollout silently. Failing loud here is the same call the
        # class already makes for the other two non-compositional shapes.
        #
        # Checked in TEXT space (tokenize=False + our own encode), not via
        # apply_chat_template(tokenize=True): the class deliberately never tokenizes through
        # the template (the double-BOS ambiguity its comments cite), and the fake tokenizer
        # in tests asserts tokenize is False. The generation prompt is excluded so this can't
        # pre-empt the separate non-suffix guard (a template may have a compositional body
        # but a body-rewriting generation prompt — that must still surface from
        # generation_prompt_ids(), not here).
        #
        # This probe is a spot check, not a proof: divergences that depend on message
        # *content* (the seam-re-encoding limitation in the class docstring) may not show
        # on these 1-char contents. It also coexists with the base-class note blessing
        # think-stripping templates only because the probe contents are plain — no
        # <think> block ever appears for such a template to strip, so those templates
        # pass here and their stripping remains the documented per-message limitation.
        probe = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "v"},
        ]
        # Some templates cannot render the probe at all — Gemma raise_exception()s on a
        # system role, Llama-2/Mistral enforce strict user/assistant alternation that the
        # lone-assistant per-message render violates. Those templates genuinely cannot be
        # rendered per message, so failing at construction is correct — but it must be
        # this class's informative error, not a raw jinja2 TemplateError from inside
        # transformers.
        try:
            incremental = self.initial_ids()
            for message in probe:
                incremental += self.render_message(message)
            body_text = self.tokenizer.apply_chat_template(
                probe, tokenize=False, add_generation_prompt=False
            )
        except Exception as exc:
            raise ValueError(
                f"{getattr(self.tokenizer, 'name_or_path', self.tokenizer)!r} has a chat "
                "template that cannot render messages individually (it raised while "
                "rendering a per-message probe — commonly a template that rejects a "
                "system role or enforces strict role alternation). Token-in serving "
                "renders incrementally, so this template cannot be used. Use a "
                "per-message-renderable template, or supply a custom ChatFormat."
            ) from exc
        # Reconstruct the whole-conversation ids the same way the class builds them
        # (initial specials + the rest encoded without specials), so any mismatch is a
        # genuine per-message/whole divergence rather than a BOS-handling artifact.
        initial_text = self._initial_text()
        if initial_text and body_text.startswith(initial_text):
            body_text = body_text[len(initial_text) :]
        whole = self.initial_ids() + list(
            self.tokenizer.encode(body_text, add_special_tokens=False)
        )
        if incremental != whole:
            raise ValueError(
                f"{getattr(self.tokenizer, 'name_or_path', self.tokenizer)!r} has a chat "
                "template that is not compositional per message: rendering each message "
                "alone diverges from rendering the whole conversation (commonly a default "
                "system prompt injected on every turn, as in the Qwen2.5 / SmolLM2 "
                "families). Token-in serving would record ids the model never saw, so this "
                "is rejected rather than silently corrupting rollouts. Use a template "
                "without per-turn default injection (e.g. Qwen3), or supply a custom "
                "ChatFormat."
            )

    def initial_ids(self) -> list[int]:
        # Defined as the tokenizer's empty-encode specials — [BOS] for Llama/Gemma/Mistral
        # families, [] for ChatML/Qwen — because apply_chat_template([]) is not viable:
        # most templates iterate messages and several raise_exception on an empty list. A
        # template whose true preamble exceeds BOS (default-system-prompt injection) is a
        # documented limitation (class docstring).
        if self._initial_ids is None:
            self._initial_ids = list(self.tokenizer.encode("", add_special_tokens=True))
        return list(self._initial_ids)

    def _initial_text(self) -> str:
        # The preamble as text (the BOS token string, or "") — the prefix stripped off
        # every per-message render below.
        return self.tokenizer.decode(self.initial_ids(), skip_special_tokens=False)

    def render_message(self, message: MessageDict) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            [message], tokenize=False, add_generation_prompt=False
        )
        # Slicing at the preamble boundary is safe when the frame opens with a special
        # token (ChatML <|im_start|>, Llama-3 <|start_header_id|> — specials are atomic, so
        # re-encoding cannot shift across the cut); textual framing may not survive the
        # seam and is a documented limitation. The compositionality tests are the arbiter.
        pre = self._initial_text()
        if pre and text.startswith(pre):
            text = text[len(pre) :]
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def generation_prompt_ids(self) -> list[int]:
        if self._generation_prompt_ids is None:
            # Suffix-diff on a fixed dummy message: the generation prompt is whatever
            # add_generation_prompt=True appends beyond the plain render. A template whose
            # generation prompt is not a plain suffix (it rewrites the body) cannot be
            # rendered incrementally at all — fail loudly rather than record wrong ids.
            probe = [{"role": "user", "content": "x"}]
            with_gp = self.tokenizer.apply_chat_template(
                probe, tokenize=False, add_generation_prompt=True
            )
            without_gp = self.tokenizer.apply_chat_template(
                probe, tokenize=False, add_generation_prompt=False
            )
            if not with_gp.startswith(without_gp):
                raise ValueError(
                    "this tokenizer's generation prompt is not a plain suffix of the "
                    "rendered conversation; it cannot be rendered incrementally"
                )
            self._generation_prompt_ids = list(
                self.tokenizer.encode(
                    with_gp[len(without_gp) :], add_special_tokens=False
                )
            )
        return list(self._generation_prompt_ids)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)

    def sampled_suffix_ids(self) -> list[int]:
        # Derived, not hard-coded: render two assistant turns whose contents differ, take
        # the common id *suffix* (the shared framing after the content, i.e. the stop token
        # plus any trailing separator), and drop its first token — the stop token, which the
        # sampled ids already carry (SGLang records output_ids with no_stop_trim). What's
        # left is the separator alone. Robust for special-token frames (ChatML, Llama-3),
        # the same case render_message's preamble slicing relies on; textual frames are a
        # documented limitation.
        #
        # Assumes the end-of-turn marker is a SINGLE token (true for special-token frames:
        # <|im_end|>, <|eot_id|>). A multi-token stop marker would leak its earlier tokens
        # into the returned suffix; such a frame is not incrementally reproducible here
        # anyway (render_message's slicing has the same requirement).
        if self._sampled_suffix_ids is None:
            a = self.render_message({"role": "assistant", "content": "a"})
            b = self.render_message({"role": "assistant", "content": "b"})
            if a == b:
                # A template that renders assistant turns independently of their content
                # cannot be diffed for the separator — the whole render would read as
                # "common" and yield a bogus content-inclusive suffix. Fail loud, matching
                # generation_prompt_ids's refusal of non-suffix templates.
                raise ValueError(
                    "this chat template renders assistant turns independently of their "
                    "content; its sampled suffix cannot be derived"
                )
            n = 0
            while n < len(a) and n < len(b) and a[-1 - n] == b[-1 - n]:
                n += 1
            common = a[
                len(a) - n :
            ]  # == [stop_token, *separator]; [] if the tails diverge
            self._sampled_suffix_ids = list(common[1:])
        return list(self._sampled_suffix_ids)
