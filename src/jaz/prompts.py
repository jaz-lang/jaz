import inspect
from collections.abc import Callable, Collection, Mapping, Sequence

from .descriptions import HIDDEN, get_description, resolve_jaz_get
from .repl.registry import REPL_LANGUAGE_MAP
from .string_utils import abbreviate_string
from .template_loader import _jinja_env

# TODO: move `prompt_additions_str` out of the task block?


def _format_input_value_default(name: str, value: object) -> str:
    """Default rendering chain, used when no description is attached."""
    if isinstance(value, type):
        return _format_class(name, value)
    if callable(value):
        return _format_callable(name, value)
    if type(value).__str__ is object.__str__ and hasattr(value, "__dict__"):
        return _format_object(name, value)
    return str(value)


def _format_callable(name: str, value: Callable[..., object]) -> str:
    sig = _get_signature(value)
    doc = _indent_continuation(_get_doc(value), "  ")
    return f"`{name}{sig}`: {doc}"


def _format_class(name: str, cls: type) -> str:
    lines: list[str] = [
        f"`{name}` [class]: {_indent_continuation(_get_doc(cls), '  ')}"
    ]
    attr_names: set[str] = set()
    for base in cls.__mro__:
        if base is object:
            continue
        attr_names.update(vars(base))
    for attr_name in sorted(attr_names):
        if attr_name.startswith("_"):
            continue
        attr = getattr(cls, attr_name)
        if callable(attr):
            sig = _get_signature(attr)
            doc = _indent_continuation(_get_doc(attr), "    ")
            lines.append(f"  - `{attr_name}{sig}`: {doc}")
        else:
            type_name = type(attr).__name__
            lines.append(f"  - `{attr_name}` [{type_name}]")
    return "\n".join(lines)


def _format_object(name: str, obj: object) -> str:
    cls = type(obj)
    lines: list[str] = [
        f"`{name}` [{cls.__name__}]: {_indent_continuation(_get_doc(cls), '  ')}"
    ]
    attr_names: set[str] = set(vars(obj))
    for base in cls.__mro__:
        if base is object:
            continue
        attr_names.update(vars(base))
    for attr_name in sorted(attr_names):
        if attr_name.startswith("_"):
            continue
        attr = getattr(obj, attr_name)
        if callable(attr):
            sig = _get_signature(attr)
            doc = _indent_continuation(_get_doc(attr), "    ")
            lines.append(f"  - `{attr_name}{sig}`: {doc}")
        else:
            type_name = type(attr).__name__
            lines.append(f"  - `{attr_name}` [{type_name}]")
    return "\n".join(lines)


def _get_signature(obj: Callable[..., object]) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "()"


def _get_doc(obj: type | Callable[..., object]) -> str:
    doc = inspect.getdoc(obj)
    if not doc:
        return "(no description available)"
    return doc.strip()


def _indent_continuation(text: str, indent: str) -> str:
    """Indent all lines of `text` after the first by `indent`."""
    lines = text.splitlines()
    if len(lines) <= 1:
        return text
    return lines[0] + "\n" + "\n".join(indent + line for line in lines[1:])


def _render_input_block(
    key: str,
    value: object,
    *,
    max_invoke_input_length: int,
    prefix_ratio: float,
) -> tuple[str | None, bool]:
    """Render one input as a prompt block.

    Returns ``(block, was_truncated)``; ``block`` is ``None`` when the input is
    hidden for this call via ``jaz.Display(value, None)`` (it stays bound as a
    REPL variable, just not shown). Shared by the explicit-inputs section (user
    prompt) and the scoped-inputs section (system prompt) so both render
    identically — only which prompt they land in differs.

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
        return None, False  # jaz.Display(value, None): hidden for this call
    real_value = resolve_jaz_get(value)
    # The per-input `max_invoke_input_length` cap is a property of the invoke, applied
    # uniformly to whatever text renders for this input — author-authored
    # (Display / describe / Library catalog) and auto-stringified defaults alike.
    # Provenance does not earn an exemption: a huge `describe` string or catalog
    # would otherwise escape the invoke's ceiling entirely. `abbreviate_string`
    # splices the cut in place ("[...N characters omitted...]") and flags `was_truncated`, so a
    # truncated value always announces itself rather than silently appearing whole.
    # TODO(#553): give the agent a REPL tool to read the omitted portion on demand,
    # so uniform truncation stays ergonomic instead of exempting text from the cap.
    # `isinstance(desc, str)` rather than `desc is not None`: HIDDEN is already
    # returned above, so the only non-str cases left are None and (unreached) HIDDEN —
    # the str check both selects the description and narrows the type for the cap.
    value_str = (
        desc if isinstance(desc, str) else _format_input_value_default(key, real_value)
    )
    abbreviated_value, was_truncated = abbreviate_string(
        value_str, max_invoke_input_length, prefix_ratio=prefix_ratio
    )
    # Each input renders as a single XML block whose tag name IS the input's bound name
    # and whose `type=` attribute is the runtime type — the unified surface after `task`
    # was demoted to an ordinary input (#538): there is no privileged `<task>` block, so
    # the prompt is a flat concatenation of one `<name type="...">value</name>` block per
    # input. The name-as-tag choice lets each section's closing sentence (written in the
    # prompt template) refer to each input by the exact identifier the agent binds in its REPL.
    # `__class__.__name__`, not `type(...).__name__`: identical for ordinary values, but a
    # transparent proxy (wrapt.ObjectProxy and friends) spoofs `__class__` to the wrapped
    # type while `type()` still reports the proxy — so this labels the block with the type
    # the agent believes it holds. JAZ itself no longer hands one out (`describe` tags
    # built-ins in a side table rather than proxying them), so this is now purely defensive
    # against a proxy arriving from caller code.
    block = (
        f'<{key} type="{real_value.__class__.__name__}">'
        f"\n{abbreviated_value}\n</{key}>\n"
    )
    return block, was_truncated


def render_input_section(
    inputs: Mapping[str, object],
    *,
    max_invoke_input_length: int,
    # Mirrors ``Config.protocol.truncation_prefix_ratio``'s default; the production caller
    # (``CodeOnlyProtocol``) always passes the configured value, so this only covers direct
    # callers. Kept in step with the config so those two never disagree about the split.
    prefix_ratio: float = 0.5,
) -> tuple[str, list[str], list[str]]:
    """Render every entry of ``inputs`` into a single block string.

    Returns ``(blocks_str, shown_names, truncated_names)``. Used to build the
    scoped-input section that lands in the SYSTEM prompt (scoped inputs are
    ambient/global, like tool libraries; see :func:`get_system_prompt`) — the caller
    passes the already-separated ``scope`` mapping (#727), so this renders all of it
    rather than filtering a merged namespace by key. ``shown_names`` omits inputs
    hidden via ``jaz.Display(value, None)`` and feeds the section's closing sentence;
    ``truncated_names`` are handed to :func:`get_user_prompt` via
    ``extra_truncated_names`` so the single truncation-advice block stays unified
    across scoped and explicit inputs even though they render in different messages.

    Per-call ``jaz.Display`` directives need no separate channel: they arrive in
    ``inputs`` and render through :func:`_render_input_block`'s ``get_description``
    lookup like any other metadata wrapper.
    """
    blocks = ""
    shown: list[str] = []
    truncated: list[str] = []
    for key, value in inputs.items():
        block, was_truncated = _render_input_block(
            key,
            value,
            max_invoke_input_length=max_invoke_input_length,
            prefix_ratio=prefix_ratio,
        )
        if block is None:
            continue
        blocks += block
        shown.append(key)
        if was_truncated:
            truncated.append(key)
    return blocks, shown, truncated


def get_system_prompt(
    *,
    system_prompt_template: str = "system_prompt.jinja2",
    subinvoke_description: str | None,
    depth: int,
    recursion_available: bool,
    repl_language: str,
    repl: object | None = None,
    scoped_inputs_str: str = "",
    scoped_names: Sequence[str] = (),
    # The wire-format instruction block, owned and supplied by the BaseProtocol
    # (#639) — this helper only places it via `{{ format_instructions }}`, it does not
    # author it. Defaults to "" so direct callers that don't drive a protocol (tests,
    # bare prompt renders) still satisfy the template's StrictUndefined without inventing
    # the text here; the production caller (CodeOnlyProtocol) always passes the real block.
    format_instructions: str = "",
    # The `__history__` block, likewise owned and supplied by the protocol (its record seam —
    # BaseProtocol.describe_history_entry) and merely placed here. Defaults to "" on the same
    # reasoning as `format_instructions`: a direct caller with no protocol renders a prompt that
    # simply does not mention history, rather than this helper inventing a record shape it has no
    # way to know. The production caller gates it on `repl.maintains_repl_history` before passing.
    history_description: str = "",
) -> str:
    # Scoped inputs (ambient via `jaz.scope`) render here in the SYSTEM prompt,
    # not the user prompt: they are global/ambient to the whole invoke — like the
    # tool libraries above — rather than part of the per-task request. Callers
    # pass the already-rendered block (see `render_input_section`); we only place
    # it. Trade-off: scoped *values* now sit in the system message, so the system
    # prompt varies with scope content rather than being purely instructional
    # (acceptable — it's identical across iterations of a given invoke).
    #
    # The section's SHAPE mirrors the user prompt's inputs section exactly: bare
    # `<name type="...">value</name>` blocks followed by one closing sentence naming
    # them, with no enclosing wrapper tag. The blocks were already identical (both
    # go through `_render_input_block`); the old `<available_objects>` wrapper plus
    # its "The following objects are available…" preamble was the only thing making
    # the two sections read as different formats. `scoped_names` is the raw list of
    # shown names; the template writes the sentence around it exactly as
    # `user_prompt.jinja2` does for explicit inputs.
    # Generate REPL description header (an agent has exactly one REPL).
    #
    # NO LONGER RENDERED BY THE SHIPPED TEMPLATE, and still computed and passed on purpose —
    # do not delete it as dead code. The shipped `system_prompt.jinja2` dropped its
    # `{{ repl_summary }}` line because the description below already opens by naming the
    # language ("## Python REPL specification"), so the sentence was a second, wordier
    # statement of it. But the render env is `StrictUndefined`: an out-of-tree
    # system-prompt template that still references `{{ repl_summary }}` would *raise* the
    # moment we stopped passing it, rather than quietly rendering nothing. Keeping the kwarg
    # is therefore the non-breaking half of the change — the same reasoning that keeps
    # `show_depth` under its stale name below.
    repl_summary = f"You have access to the `{repl_language}` REPL. "

    # Collect the REPL-specific instructions for the agent's REPL.
    if repl_language not in REPL_LANGUAGE_MAP:
        raise ValueError(
            f"REPL language `{repl_language}` is not available. "
            "Did you misspell or forget to register it?"
        )
    # Ask the configured REPL for its own description rather than re-deriving it from a config
    # bag: the REPL holds its settings, and its `description` property is what keeps the rendered
    # text agreeing with what is actually enforced (`allow_raise` especially — see PythonREPL).
    # Falls back to the class default for a caller that has no instance (a bare prompt render).
    description = (
        repl.description  # type: ignore[union-attr]
        if repl is not None
        else REPL_LANGUAGE_MAP[repl_language].get_description()
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
    # statements of it. Accepted — the agent's own code makes the language obvious — but it is
    # why removing BOTH in one change is a bigger step than either alone.
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
        repl_summary=repl_summary,
        repl_specific_instructions=repl_specific_instructions,
        format_instructions=format_instructions,
        history_description=history_description,
        # `subinvoke_description` (protocol-rendered) is exposed to the template under the name
        # `invoke_tool` — see the block comment above on why the template var keeps that name.
        invoke_tool=subinvoke_description,
        show_depth=show_depth,
        depth=depth,
        scoped_inputs_str=scoped_inputs_str,
        scoped_names=scoped_names,
    )

    return prompt


def get_user_prompt(
    *,
    user_prompt_template: str = "user_prompt.jinja2",
    inputs: Mapping[str, object],
    max_invoke_input_length: int,
    input_truncation_advice_template: str | None = None,
    extra_truncated_names: Collection[str] | None = None,
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
    # The user prompt no longer has a distinguished task string (#538): it is a flat
    # concatenation of one `<name type="...">` block per explicit input, followed by a
    # single sentence naming those inputs. What used to be the `task` argument is now just
    # an ordinary input (conventionally named `task`) rendered as any other block.

    # Seed with truncations reported for inputs rendered elsewhere (scoped inputs
    # render in the system prompt; see `render_input_section`) so the single
    # truncation-advice block below covers every truncated input, not just the
    # explicit ones shown here.
    truncated_variable_names: list[str] = list(extra_truncated_names or [])
    # Per-call `jaz.Display` directives arrive inline in `inputs` and render through
    # `_render_input_block`'s `get_description` lookup (the directive reports HIDDEN
    # to hide, or its resolved text to relabel) — no separate override channel.

    # `inputs` is the explicit, per-invoke namespace only — scoped inputs (ambient
    # via `jaz.scope`) arrive as a separate mapping and render into the SYSTEM prompt
    # via `render_input_section` (#727), so there is nothing to filter out here.
    inputs_str = ""
    shown_names: list[str] = []
    for key, value in inputs.items():
        block, was_truncated = _render_input_block(
            key,
            value,
            max_invoke_input_length=max_invoke_input_length,
            prefix_ratio=prefix_ratio,
        )
        if block is None:
            continue  # hidden via jaz.Display(value, None): bound but not shown/named
        inputs_str += block
        shown_names.append(key)
        if was_truncated:
            truncated_variable_names.append(key)

    truncation_advice_str = ""
    if input_truncation_advice_template and truncated_variable_names:
        template = _jinja_env.get_template(input_truncation_advice_template)
        rendered_advice = template.render(
            truncated_variable_names=truncated_variable_names,
        )
        truncation_advice_str = (
            "\n<truncation_advice>\n" + rendered_advice + "\n</truncation_advice>"
        )

    template = _jinja_env.get_template(user_prompt_template)
    return template.render(
        inputs_str=inputs_str,
        truncation_advice_str=truncation_advice_str,
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
