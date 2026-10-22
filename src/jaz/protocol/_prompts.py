"""Prompt assembly: turn an invoke's inputs and REPL into the system and user messages.

Non-public: the leading underscore is the boundary (see ``jaz/__init__.py``).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .._string_utils import abbreviate_string
from .._template_loader import _jinja_env
from .._value_rendering import render_value
from ..exceptions import InputRenderError
from ..repl._base import BaseREPL


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
) -> RenderedInput:
    """Resolve one input into a :class:`RenderedInput`.

    Every input renders — there is no way to bind a value and withhold its block. Both input
    sections reach this through :func:`render_input_section`, so an input resolves identically
    wherever it lands.

    Rendering is one call to :func:`~jaz._value_rendering.render_value`. The type label reads
    the value directly, so what renders and what the REPL binds are the same object.
    """
    # `key` is threaded as the bound name so the rendering names the value the way the agent
    # does — the kwarg it is passed under, not any internal name the value carries.
    # Every `Exception` becomes `InputRenderError` naming the input, never fatal; why is on the
    # class. `Exception`, not `BaseException`: a Ctrl-C during a slow `__repr__` still ends the run.
    try:
        description = render_value(value, key)
    except Exception as exc:
        raise InputRenderError(
            f"Rendering input {key!r} failed: {type(exc).__name__} raised while rendering "
            f"a {type(value).__name__}"
        ) from exc
    # The per-input `max_invoke_input_length` cap is a property of the invoke, applied
    # uniformly to whatever text renders for this input. `abbreviate_string`
    # splices the cut in place ("[...N characters omitted...]"), so a truncated value announces
    # itself in the text rather than silently appearing whole. The companion flag is discarded:
    # nothing downstream reports truncation any more (see `get_user_prompt`).
    # TODO(#553): give the agent a REPL tool to read the omitted portion on demand,
    # so uniform truncation stays ergonomic instead of exempting text from the cap.
    abbreviated_value, _ = abbreviate_string(description, max_invoke_input_length)
    # `__class__.__name__`, not `type(...).__name__`: identical for ordinary values, but a
    # transparent proxy (wrapt.ObjectProxy and friends) spoofs `__class__` to the wrapped type
    # while `type()` still reports the proxy — so this labels the input with the type the agent
    # believes it holds. JAZ ships no wrapper types, so this is purely defensive against caller
    # code.
    return RenderedInput(
        name=key, type=value.__class__.__name__, value=abbreviated_value
    )


def render_input_section(
    inputs: Mapping[str, object],
    *,
    max_invoke_input_length: int,
) -> list[RenderedInput]:
    """Resolve every entry of ``inputs`` into a list of :class:`RenderedInput`, in order.

    Both input sections go through this: the explicit per-invoke inputs that land in the USER
    prompt (:func:`get_user_prompt`), and the scoped inputs that land in the SYSTEM prompt
    (ambient/global, like tool libraries; see :func:`get_system_prompt`). Inputs and scope
    arrive as separate mappings, so each caller passes its own whole mapping rather than
    filtering a merged namespace by key.

    Every entry renders: nothing can bind without appearing in the prompt.
    """
    # One list, not a `(blocks, names)` pair. The names used to be able to differ from the
    # blocks — a value could bind while its block was withheld — so the closing sentence needed
    # the *shown* subset. That withholding is gone, which left the second list as exactly
    # `[b.name for b in blocks]`, handed to the template beside the thing it was derived from.
    # The templates derive it themselves now (EXECUTIVE CALL, user, 2026-09-18: one list per
    # section).
    # Scope arrives already separated from inputs (#727), so no filtering is needed here.
    # Being the single resolver is the point: `get_user_prompt` used to carry a copy of this
    # loop, so "the two sections render an input the same way" was a claim two bodies had to
    # keep true rather than something the code guaranteed.
    blocks = [
        _render_input_block(key, value, max_invoke_input_length=max_invoke_input_length)
        for key, value in inputs.items()
    ]
    return blocks


def require_description(value: object, *, seam: str) -> str:
    """Return ``value`` if it is a non-empty string, else raise ``ValueError`` naming ``seam``."""
    # One rule for every description the templates are handed — the wire-format block, the REPL
    # description, the `__history__` description and the `invoke` entry: a present description
    # is a non-empty string, and absence, where a seam has one, is spelled `None`. "" is refused
    # rather than read as absence so a template can gate on `is not none` and a section can
    # never render as an empty element (`<response_format></response_format>`, a bare `<invoke>`
    # block) beside prose that assumes it has content. EXECUTIVE CALL (user, 2026-09-18): the
    # refusal `get_subinvoke_description` got, applied to every sibling seam rather than to one.
    #
    # Applied twice on the production path, deliberately: `CodeOnlyProtocol` checks each of its
    # own seams so the error names the class and method an author overrode, and `get_system_prompt`
    # checks everything it is handed so the rule holds at the template boundary for every caller
    # — another `BaseProtocol`, a test, a bare render — not only for the one protocol that
    # happens to check. The second check is what makes "never an empty element" a property of
    # the template's inputs rather than of one caller's discipline.
    if not isinstance(value, str) or not value:
        raise ValueError(f"{seam} must return a non-empty string, got {value!r}")
    return value


def get_system_prompt(
    *,
    system_prompt_template: str = "jaz/system_prompt.jinja2",
    subinvoke_description: str | None,
    repl: BaseREPL[Any],
    scoped_inputs: Sequence[RenderedInput] = (),
    # The wire-format instruction block, owned and supplied by the BaseProtocol
    # (#639) — this helper only places it via `{{ response_format_description }}`, it does not
    # author it. Required, like `repl`: the shipped template renders it unconditionally, so the
    # "" default this once had for direct callers with no protocol rendered an empty
    # `<response_format>` element — a sentence-shaped hole, silently. A direct caller passes a
    # stand-in; the production caller (CodeOnlyProtocol) passes the real, checked block.
    response_format_description: str,
    # The `__history__` block, owned by the REPL (`BaseREPL.repl_history_description`) and
    # merely placed here. `None` is the REPL's own spelling for "I bind no such name", and the
    # template gates the section on presence (`is not none`), the same test the `invoke` block
    # uses; "" is refused at the seam (see `require_description`), never read as absence. Taken
    # as a parameter rather than read off `repl` here because the invoke may have withheld the
    # name (`WithholdREPLHistory`), which only the caller knows; `CodeOnlyProtocol` resolves the
    # two and passes the result. Defaults to `None` so a direct caller renders a prompt that
    # simply does not mention history.
    repl_history_description: str | None = None,
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
    # The template derives the names for its closing sentence from the same sequence, exactly
    # as `user_prompt.jinja2` does for explicit inputs.
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

    # Ask the configured REPL for its own description rather than re-deriving it from a config
    # bag: the REPL holds its settings, and its `description` property is what keeps the rendered
    # text agreeing with what is actually enforced (`allow_raise` especially — see PythonREPL).
    # Required, not optional: a prompt describes the REPL the agent will run in, and there is
    # no run without one — the old `None` arm rendered a default `PythonREPL`'s description,
    # which every caller that reached it was a test.
    # Checked where it is read, under the rule every protocol seam gets in
    # `render_initial_message_list`: an empty description would render an empty `<repl_spec>`.
    description = require_description(
        repl.description, seam=f"{type(repl).__name__}.description"
    )
    # Passed through unwrapped: the template owns the tag (`<repl_spec>`). This used to add a
    # second `<repl_description lang="...">` wrapper of its own, which nested one tag inside
    # another around a single block once the template grew its own.
    #
    # The `lang` attribute went with it, which leaves the *body* of whatever description
    # template is configured as the only place the language is named — the shipped
    # `python_repl_description.jinja2` opens "## Python REPL specification", so the shipped
    # template still says it. Not a guarantee though: `repl_description_template` is caller-configurable, so
    # an out-of-tree description that never names its language now yields a prompt that names
    # it nowhere, where the dropped `lang=` and `{{ repl_summary }}` were two independent
    # statements of it. Accepted — the agent's own code makes the language obvious.
    repl_description = description

    # The propagation sentence (scoped values ride along into every `invoke()`) is gated in the
    # template on `subinvoke_description` itself, not on a separate flag. There used to be one —
    # `show_depth`, fed from a `recursion_available` parameter — kept apart from "the tool is
    # present" on the argument that the flag was the *reason* and reading the reason would stay
    # correct if a caller ever withheld the tool for some other cause. That argument runs the
    # wrong way: if the tool is withheld for ANY cause, a sentence telling the agent what every
    # `invoke()` it makes will carry is false, because it cannot make one. Presence of the
    # description — `is not none`, the same test the `<invoke>` block uses — is exactly the
    # condition under which the sentence is true; `CodeOnlyProtocol` refuses an empty
    # description, so presence and "the agent is shown an `invoke`" coincide. With two inputs
    # the helper could also render the contradiction (propagation sentence, no `<invoke>`
    # block); with one, it cannot be expressed. EXECUTIVE CALL (user, 2026-09-18): one signal,
    # gated on presence, as part of making the prompt surface minimal.

    # One catalog entry — `- `invoke(**inputs)`: <docstring>` — rather than the multi-section
    # card (name / description / top-level modules) this used to render. The description is
    # produced by the PROTOCOL
    # (`BaseProtocol.get_subinvoke_description`, whose default introspects the bound `invoke`
    # closure) and arrives here pre-rendered as `subinvoke_description`; this helper only places
    # it, staying codec-agnostic.
    #
    # EXECUTIVE CALL (user, 2026-09-18): one name per concept across the prompt surface, from the
    # module down to the template variable. So the template sees this as `subinvoke_description`,
    # the name the protocol method and this helper's own parameter already used.
    #
    # That reverses an earlier, deliberate call rather than overlooking it. The variable was
    # `invoke_tool`, and it was KEPT under that name when its value became a rendered string, on
    # the argument that renaming a template variable whose value has not changed breaks
    # out-of-tree templates to buy a tidier word. Rejected now because the premise has no
    # referent: the surface is pre-1.0 and no out-of-tree template exists (jaz-evals-internal
    # ships none and customises by hook), while a name saying "tool" for a string costs every
    # reader of the contract.
    #
    # The break is loud for the ordinary spellings: the Jinja env uses `StrictUndefined`, which
    # raises on any operation on an undefined — including a boolean test — so `{% if invoke_tool
    # %}` fails at render with "'invoke_tool' is undefined" rather than testing falsy and
    # dropping the block, which is why no compatibility alias is passed alongside. A template
    # that guards the name (`is defined`, `| default(...)`) is the exception: it renders cleanly
    # and drops the block silently.
    template = _jinja_env.get_template(system_prompt_template)
    prompt = template.render(
        repl_description=repl_description,
        # The boundary check (see `require_description`), by template variable since this helper
        # cannot know which seam produced the value; `CodeOnlyProtocol` already named it.
        response_format_description=require_description(
            response_format_description, seam="response_format_description"
        ),
        repl_history_description=(
            None
            if repl_history_description is None
            else require_description(
                repl_history_description, seam="repl_history_description"
            )
        ),
        subinvoke_description=subinvoke_description,
        scoped_inputs=scoped_inputs,
    )

    return prompt


def get_user_prompt(
    *,
    user_prompt_template: str = "jaz/user_prompt.jinja2",
    inputs: Mapping[str, object],
    max_invoke_input_length: int,
    # The same value the system template gets, under the same name, gated the same way (`is
    # not none`). The closing sentence tells the agent its inputs do NOT propagate to an
    # `invoke()` and must be passed explicitly, which is only meaningful when it can make one;
    # it is the other half of the rule the system prompt states (scoped values propagate,
    # inputs do not), so the two templates read one value rather than a value and a flag
    # derived from it. The shipped template only tests presence; a variant may also render the
    # description. Defaulted so direct callers
    # that never recurse render without the sentence.
    subinvoke_description: str | None = None,
) -> str:
    """Render the user prompt: the explicit per-invoke inputs, laid out by the template.

    Each entry of ``inputs`` is resolved to a :class:`RenderedInput` and handed to
    ``user_prompt_template`` as ``inputs``; the template decides the markup. Every
    input renders — binding a value and withholding its block is not expressible.
    """
    # There is no distinguished task string (#538): the prompt is one block per explicit input
    # followed by a single sentence naming them, and what used to be the `task` argument is an
    # ordinary input (conventionally named `task`) rendered like any other. Both the block markup
    # and the sentence are the template's now — core hands over the resolved inputs and nothing
    # else; the sentence derives its names from them.

    # `inputs` is the explicit, per-invoke namespace only — scoped inputs (ambient
    # via `jaz.scope`) arrive as a separate mapping and render into the SYSTEM prompt
    # via `render_input_section` (#727), so there is nothing to filter out here.
    # The SAME resolver the system prompt's scoped-input section uses. The two sections must
    # resolve inputs identically — same description lookup, same ceiling — or an input would
    # render one way as an argument and another as ambient scope. Sharing the
    # function is what makes that structural instead of a convention two loops have to keep.
    rendered_inputs = render_input_section(
        inputs, max_invoke_input_length=max_invoke_input_length
    )

    # Core says nothing about truncated inputs and gives the template nothing to say it with.
    #
    # There used to be an `input_truncation_advice_template` knob whose output core wrapped in a
    # `<truncation_advice>` tag and spliced in here — one rendered surface across three owners,
    # of which a host could replace only the middle. Folding it into this template was the obvious
    # repair, but the list it would need has properties nobody can guess: it is not a subset of
    # the input names (scoped inputs render in the SYSTEM prompt yet truncate against the same
    # ceiling), and those scoped names would lead. Rather than ship a surface that needs caveats
    # for a
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
        #
        # The template variable is `inputs`, the same spelling as this function's mapping
        # parameter, though it holds the rendered sequence rather than the raw values. Weighed and
        # kept: the template author's `inputs` are the things that render, one per block, and the
        # system side uses `scoped_inputs` for the same shape — so the two templates match each
        # other, which matters more than this local variable matching the Python argument.
        inputs=rendered_inputs,
        # #568: the closing sentence — wording, backtick-quoting of the names, and
        # singular/plural agreement alike — belongs to the (config-selectable) template,
        # which derives the names from `inputs` and writes the sentence itself. Core used to hand
        # over pre-composed `inputs_names`/`inputs_noun`/`inputs_verb` strings, which split
        # one sentence across two layers: a template variant could reword the frame but not
        # the noun inside it, and the pluralization rule lived in Python where no reader of
        # the prompt would look for it. A variant now writes its whole sentence, so a host that
        # withholds inputs from the REPL can say they are described but NOT bound as REPL
        # variables.
        #
        # `inputs` is empty only when no inputs were passed, so the template's `{% if inputs %}`
        # guard drops the sentence entirely.
        subinvoke_description=subinvoke_description,
    )
