"""Prompt assembly: turn an invoke's inputs and REPL into the system and user messages.

Non-public. Note the reachability asymmetry this module's location creates: ``jaz``'s
``warn_on_nonpublic_submodules`` guard is installed on the top-level package only, so
``import jaz.protocol.prompts`` is silent where ``import jaz.prompts`` would have warned. The
``protocol`` package name is the signpost instead — see the enumeration in ``jaz/__init__.py``.

Out-of-tree consumers exist despite the warning: ``jaz-evals-internal`` imports ``get_system_prompt``
/ ``get_user_prompt`` from the old path in its own tests. TODO(#1383) tracks updating them.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..descriptions import (
    HIDDEN,
    get_default_description,
    get_description,
    resolve_jaz_get,
)
from ..repl.python_repl import PythonREPL
from ..string_utils import abbreviate_string
from ..template_loader import _jinja_env

# TODO: move `prompt_additions_str` out of the task block?


@dataclass(frozen=True)
class RenderedInput:
    """One input, resolved and abbreviated, ready for a template to lay out.

    ``name`` is the identifier the agent binds in its REPL, ``type`` the runtime type it believes
    it holds, and ``value`` the already-described, already-truncated text.
    """

    name: str
    type: str
    value: str

    # Why the shape is what it is (#538): `task` was demoted to an ordinary input, so there is no
    # privileged `<task>` block and the prompt is a flat concatenation of one block per input. The
    # tag name IS the bound name so each section's closing sentence — written in the same template
    # that lays these out — can refer to every input by the exact identifier the agent binds in its
    # REPL. That coupling is why the layout moved out of Python: the blocks and the sentence about
    # them are one surface, and it now has one owner.


def _render_input_block(
    key: str,
    value: object,
    *,
    max_invoke_input_length: int,
    prefix_ratio: float,
) -> RenderedInput | None:
    """Resolve one input into a :class:`RenderedInput`.

    Returns ``None`` when the input is hidden for this call via ``jaz.Display(value, None)`` (it
    stays bound as a REPL variable, just not shown). Both input sections reach this through
    :func:`render_input_section`, so an input resolves identically wherever it lands.

    Description lookup goes through the single :func:`jaz.get_description` path: a
    ``jaz.Display`` directive, a ``jaz.describe``'d value, and a ``Library`` are all
    resolved the same way (the directive reports ``HIDDEN`` to hide). ``__jaz_get__``
    is resolved for the type label and default rendering so they reflect the value
    the agent actually binds (e.g. a ``Library``'s root module), not the wrapper.
    """
    # `key` is threaded as the description's bound name so callable/Catalog
    # descriptions render against the kwarg the agent sees (e.g. a Library passed as
    # `tools=lib` renders its catalog rooted at `tools.*`, not its module name).
    desc = get_description(value, bound_name=key)
    if desc is HIDDEN:
        return None  # jaz.Display(value, None): hidden for this call
    real_value = resolve_jaz_get(value)
    # The per-input `max_invoke_input_length` cap is a property of the invoke, applied
    # uniformly to whatever text renders for this input — author-authored
    # (Display / describe / Library catalog) and auto-stringified defaults alike.
    # Provenance does not earn an exemption: a huge `describe` string or catalog
    # would otherwise escape the invoke's ceiling entirely. `abbreviate_string`
    # splices the cut in place ("[...N characters omitted...]"), so a truncated value announces
    # itself in the text rather than silently appearing whole. The companion flag is discarded:
    # nothing downstream reports truncation any more (see `get_user_prompt`).
    # TODO(#553): give the agent a REPL tool to read the omitted portion on demand,
    # so uniform truncation stays ergonomic instead of exempting text from the cap.
    # `isinstance(desc, str)` rather than `desc is not None`: HIDDEN is already
    # returned above, so the only non-str cases left are None and (unreached) HIDDEN —
    # the str check both selects the description and narrows the type for the cap.
    # `value`, NOT `real_value`: `get_default_description` resolves `__jaz_get__` itself.
    # Passing the already-resolved payload would resolve a second time, so a value whose
    # payload is itself a `__jaz_get__` carrier would render one level deeper than
    # `inputs.resolve_inputs` binds — the block's type label and its text would disagree.
    value_str = desc if isinstance(desc, str) else get_default_description(value, key)
    abbreviated_value, _ = abbreviate_string(
        value_str, max_invoke_input_length, prefix_ratio=prefix_ratio
    )
    # `__class__.__name__`, not `type(...).__name__`: identical for ordinary values, but a
    # transparent proxy (wrapt.ObjectProxy and friends) spoofs `__class__` to the wrapped type
    # while `type()` still reports the proxy — so this labels the input with the type the agent
    # believes it holds. JAZ itself no longer hands one out (`describe` tags built-ins in a side
    # table rather than proxying them), so this is now purely defensive against caller code.
    return RenderedInput(
        name=key, type=real_value.__class__.__name__, value=abbreviated_value
    )


def render_input_section(
    inputs: Mapping[str, object],
    *,
    max_invoke_input_length: int,
    # Mirrors ``Config.protocol.truncation_prefix_ratio``'s default; the production caller
    # (``CodeOnlyProtocol``) always passes the configured value, so this only covers direct
    # callers. Kept in step with the config so those two never disagree about the split.
    prefix_ratio: float = 0.5,
) -> tuple[list[RenderedInput], list[str]]:
    """Resolve every entry of ``inputs`` into a list of :class:`RenderedInput`.

    Returns ``(blocks, shown_names)``. Both input sections go through this: the explicit
    per-invoke inputs that land in the USER prompt (:func:`get_user_prompt`), and the scoped
    inputs that land in the SYSTEM prompt (ambient/global, like tool libraries; see
    :func:`get_system_prompt`). Inputs and scope arrive as separate mappings, so each caller
    passes its own whole mapping rather than filtering a merged namespace by key.
    ``shown_names`` omits inputs hidden via ``jaz.Display(value, None)`` and feeds each
    section's closing sentence.

    Per-call ``jaz.Display`` directives need no separate channel: they arrive in
    ``inputs`` and render through :func:`_render_input_block`'s ``get_description``
    lookup like any other metadata wrapper.
    """
    # Scope arrives already separated from inputs (#727), so no filtering is needed here.
    # Being the single resolver is the point: `get_user_prompt` used to carry a copy of this
    # loop, so "the two sections render an input the same way" was a claim two bodies had to
    # keep true rather than something the code guaranteed.
    blocks: list[RenderedInput] = []
    shown: list[str] = []
    for key, value in inputs.items():
        block = _render_input_block(
            key,
            value,
            max_invoke_input_length=max_invoke_input_length,
            prefix_ratio=prefix_ratio,
        )
        if block is None:
            continue
        blocks.append(block)
        shown.append(key)
    return blocks, shown


def get_system_prompt(
    *,
    system_prompt_template: str = "jaz/system_prompt.jinja2",
    subinvoke_description: str | None,
    depth: int,
    recursion_available: bool,
    repl: object | None = None,
    scoped_input_blocks: Sequence[RenderedInput] = (),
    scoped_names: Sequence[str] = (),
    # The wire-format instruction block, owned and supplied by the BaseProtocol
    # (#639) — this helper only places it via `{{ format_instructions }}`, it does not
    # author it. Defaults to "" so direct callers that don't drive a protocol (tests,
    # bare prompt renders) still satisfy the template's StrictUndefined without inventing
    # the text here; the production caller (CodeOnlyProtocol) always passes the real block.
    format_instructions: str = "",
    # The `__history__` block, likewise owned and supplied by the protocol (its record seam —
    # BaseProtocol.get_history_description) and merely placed here. Defaults to "" on the same
    # reasoning as `format_instructions`: a direct caller with no protocol renders a prompt that
    # simply does not mention history, rather than this helper inventing a record shape it has no
    # way to know. The production caller passes the seam's output through unconditionally, so the
    # block is dropped either by "" here (a protocol with no agent-readable record) or by the
    # template — `system_prompt.jinja2` wraps the section in a `repl_history` block precisely so a
    # vendored template can empty it, which is how a REPL that cannot bind the name opts out.
    history_description: str = "",
) -> str:
    # Scoped inputs (ambient via `jaz.scope`) render here in the SYSTEM prompt,
    # not the user prompt: they are global/ambient to the whole invoke — like the
    # tool libraries above — rather than part of the per-task request. Callers
    # pass the resolved `RenderedInput` sequence (see `render_input_section`); the
    # template lays it out. Trade-off: scoped *values* now sit in the system
    # prompt varies with scope content rather than being purely instructional
    # (acceptable — it's identical across iterations of a given invoke).
    #
    # The section's SHAPE mirrors the user prompt's inputs section exactly, and both
    # shapes now live in their templates rather than here: each renders its own
    # `<name type="...">value</name>` blocks from the same `RenderedInput` sequence,
    # followed by one closing sentence naming them, with no enclosing wrapper tag. The
    # values were already identical (both go through `_render_input_block`); the old
    # `<available_objects>` wrapper plus its "The following objects are available…"
    # preamble was the only thing making the two sections read as different formats.
    # `scoped_names` is the raw list of shown names; the template writes the sentence
    # around it exactly as `user_prompt.jinja2` does for explicit inputs.
    # Generate REPL description header (an agent has exactly one REPL).
    #
    # `repl_summary` used to be passed here — "You have access to the `python` REPL." — and no
    # shipped template has rendered it since the description below started opening with
    # "## Python REPL specification". It was kept only so an out-of-tree template still naming
    # `{{ repl_summary }}` would not raise under `StrictUndefined`. The sentence's only content
    # was the REPL's language, which no longer exists, so there is nothing left to keep: such a
    # template now raises and its author writes their own wording. Passing `repl_summary=""`
    # would have kept those templates rendering, and was rejected — it renders a sentence-shaped
    # hole where the author expects a sentence, and silently, which is worse than a named error
    # pointing at the variable.

    # Collect the REPL-specific instructions for the agent's REPL.
    # Ask the configured REPL for its own description rather than re-deriving it from a config
    # bag: the REPL holds its settings, and its `description` property is what keeps the rendered
    # text agreeing with what is actually enforced (`allow_raise` especially — see PythonREPL).
    # A caller with no instance (a bare prompt render) gets the default REPL's class default:
    # rendering a prompt for a REPL nobody passed is a preview, not a run.
    description = (
        repl.description  # type: ignore[union-attr]
        if repl is not None
        else PythonREPL.get_description()
    )
    # Passed through unwrapped: the template owns the tag (`<repl_spec>`). This used to add a
    # second `<repl_description lang="...">` wrapper of its own, which nested one tag inside
    # another around a single block once the template grew its own.
    #
    # The `lang` attribute went with it, which leaves the *body* of whatever description
    # template is configured as the only place the language is named — the shipped
    # `python_repl_description.jinja2` opens "## Python REPL specification" and the evals
    # variants open "PYTHON REPL INPUT SPECIFICATION", so every template in this repo still
    # says it. Not a guarantee though: `repl_description_template` is caller-configurable, so
    # an out-of-tree description that never names its language now yields a prompt that names
    # it nowhere, where the dropped `lang=` and `{{ repl_summary }}` were two independent
    # statements of it. Accepted — the agent's own code makes the language obvious.
    repl_specific_instructions = description

    # Tell the agent the propagation rule (scoped values ride along into every `invoke()`)
    # exactly when it can recurse. Keyed on `recursion_available` rather than on
    # `subinvoke_description is not None` even though the two now always agree: `recursion_available`
    # is the *reason* (a DisableRecursion effect), and reading the reason keeps this correct if a
    # future caller ever withholds the tool for some other cause.
    #
    # The template variable is still spelled `show_depth` because out-of-tree system-prompt
    # templates use that name; it no longer gates any depth line (there hasn't been one for a
    # while — `depth` itself is passed below and read by nothing in the shipped template).
    # `user_prompt.jinja2` is new enough to have no out-of-tree users, so its half of the same
    # rule gets the honest name, `recursion_available`.
    show_depth = recursion_available

    # One catalog entry — `- `invoke(**inputs)`: <docstring>` — rather than the library card
    # (name / description / top-level modules) this used to render via
    # `Library.render_prompt_description`. The description is produced by the PROTOCOL
    # (`BaseProtocol.get_subinvoke_description`, whose default introspects the bound `invoke`
    # closure) and arrives here pre-rendered as `subinvoke_description`; this helper only places
    # it, staying codec-agnostic. It is still exposed to the template under the name `invoke_tool`.
    #
    # RENAMED (`jaz_library` -> `invoke_tool`) even though `show_depth` above was deliberately
    # NOT renamed, for the same out-of-tree-template audience. The two point opposite ways on
    # purpose: what differs is whether the old name still tells the truth. `show_depth`'s VALUE
    # is unchanged, so a stale `{% if show_depth %}` keeps working AND keeps meaning what it
    # meant — renaming it would break out-of-tree templates to buy nothing but a tidier word.
    # This variable's value changed shape — a library card became one function's catalog entry —
    # so keeping the name would hand out-of-tree templates a `jaz_library` that is not a library,
    # and their `<jaz_library>` wrapper tag would end up labelling an `invoke` entry.
    #
    # Renaming is safe to do loudly here: the Jinja env uses `StrictUndefined`, which raises on
    # *any* operation on an undefined — including a boolean test — so a stale
    # `{% if jaz_library %}` fails at render with "'jaz_library' is undefined" rather than
    # quietly testing falsy and dropping the block. That is the outcome a rename wants (a named,
    # immediate error), and it is why no compatibility alias is passed alongside. In-tree
    # exposure is nil regardless: every `system_prompt_template:` in `evals/configs/` points at
    # the shipped `system_prompt.jinja2`.
    template = _jinja_env.get_template(system_prompt_template)
    prompt = template.render(
        repl_specific_instructions=repl_specific_instructions,
        format_instructions=format_instructions,
        history_description=history_description,
        # `subinvoke_description` (protocol-rendered) is exposed to the template under the name
        # `invoke_tool` — see the block comment above on why the template var keeps that name.
        invoke_tool=subinvoke_description,
        show_depth=show_depth,
        depth=depth,
        scoped_input_blocks=scoped_input_blocks,
        scoped_names=scoped_names,
    )

    return prompt


def get_user_prompt(
    *,
    user_prompt_template: str = "jaz/user_prompt.jinja2",
    inputs: Mapping[str, object],
    max_invoke_input_length: int,
    prefix_ratio: float = 0.5,  # mirrors Config.protocol.truncation_prefix_ratio
    # Whether this agent can spawn sub-invokes, keyed on the same flag `get_system_prompt`
    # uses: the closing sentence tells the agent its inputs do NOT propagate to a
    # `jaz.invoke()` and must be passed explicitly, which is only meaningful when it can make
    # one. Same flag on purpose -- the two sections state opposite halves of one rule (scoped
    # values propagate, inputs do not), so an agent that sees one must see the other or it is
    # left to generalize from the half it got. Defaulted rather than required (unlike
    # `get_system_prompt`) so the direct callers that never recurse render exactly as before.
    recursion_available: bool = False,
) -> str:
    """Render the user prompt: the explicit per-invoke inputs, laid out by the template.

    Each entry of ``inputs`` is resolved to a :class:`RenderedInput` and handed to
    ``user_prompt_template`` as ``input_blocks``; the template decides the markup. An input
    hidden with ``jaz.Display(value, None)`` stays bound in the REPL but is absent from both
    ``input_blocks`` and ``input_names``.
    """
    # There is no distinguished task string (#538): the prompt is one block per explicit input
    # followed by a single sentence naming them, and what used to be the `task` argument is an
    # ordinary input (conventionally named `task`) rendered like any other. Both the block markup
    # and the sentence are the template's now — core hands over resolved values and raw names.

    # Per-call `jaz.Display` directives arrive inline in `inputs` and render through
    # `_render_input_block`'s `get_description` lookup (the directive reports HIDDEN
    # to hide, or its resolved text to relabel) — no separate override channel.

    # `inputs` is the explicit, per-invoke namespace only — scoped inputs (ambient
    # via `jaz.scope`) arrive as a separate mapping and render into the SYSTEM prompt
    # via `render_input_section` (#727), so there is nothing to filter out here.
    # The SAME resolver the system prompt's scoped-input section uses. The two sections must
    # resolve inputs identically — same description lookup, same ceiling, same hidden-input rule —
    # or an input would render one way as an argument and another as ambient scope. Sharing the
    # function is what makes that structural instead of a convention two loops have to keep.
    input_blocks, shown_names = render_input_section(
        inputs,
        max_invoke_input_length=max_invoke_input_length,
        prefix_ratio=prefix_ratio,
    )

    # Core says nothing about truncated inputs and gives the template nothing to say it with.
    #
    # There used to be an `input_truncation_advice_template` knob whose output core wrapped in a
    # `<truncation_advice>` tag and spliced in here — one rendered surface across three owners,
    # of which a host could replace only the middle. Folding it into this template was the obvious
    # repair, but the list it would need has properties nobody can guess: it is not a subset of
    # `input_names` (scoped inputs render in the SYSTEM prompt yet truncate against the same
    # ceiling), those scoped names would lead, and an input hidden by `jaz.Display(v, None)` is
    # absent even when truncated. Rather than ship a surface that needs three caveats for a
    # feature no in-tree template uses, it is gone; the flag `abbreviate_string` still returns is
    # simply dropped. Reintroduce it when something asks — with the caveats designed away.
    #
    # The caveats are the a-priori argument; the observed one is stronger. Out of tree every use was
    # accidental: `jaz-evals-internal` set the knob in 10 configs, added across four consecutive
    # env-introduction commits inside 33 minutes and carried forward each time, 7 of them naming
    # the `_subagents` variant whether or not the env had sub-agents. The only configs that engaged
    # with it deliberately were the AppWorld CodeAct arms, which set aside ~60 lines of comment to
    # say DON'T: the tag lands after the trailing input sentence their hook anchors on, aborting
    # every task into a plausible 0.0. Offering the setting was what taught readers to set it.
    template = _jinja_env.get_template(user_prompt_template)
    return template.render(
        # `inputs_str` was REPLACED by this, not kept alongside it. Keeping it would mean
        # maintaining a second renderer of the very surface this moved into the template — the
        # layout would then live in two places and be free to disagree, which is the thing this
        # change exists to stop. An out-of-tree template still naming `inputs_str` raises under
        # StrictUndefined rather than rendering a silently empty section.
        input_blocks=input_blocks,
        # #568: the closing sentence — wording, backtick-quoting of the names, and
        # singular/plural agreement alike — belongs to the (config-selectable) template,
        # which is why core hands over the raw name list and nothing else. It used to hand
        # over pre-composed `inputs_names`/`inputs_noun`/`inputs_verb` strings, which split
        # one sentence across two layers: a template variant could reword the frame but not
        # the noun inside it, and the pluralization rule lived in Python where no reader of
        # the prompt would look for it. A variant now writes its whole sentence (see the
        # evals' `user_prompt_inputs_withheld.jinja2`, which says the inputs are described
        # but NOT bound as REPL variables), the same way "history off" pairs a hook with a
        # config-set description.
        #
        # `shown_names` is empty when every input is hidden via `jaz.Display(v, None)` or
        # none were passed, so the template's `{% if input_names %}` guard drops the
        # sentence entirely. The rename from `inputs_names` is deliberate: an out-of-tree
        # template still expecting the old pre-joined string fails loudly under
        # StrictUndefined rather than silently rendering `['a', 'b']`.
        input_names=shown_names,
        recursion_available=recursion_available,
    )
