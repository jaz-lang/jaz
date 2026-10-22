from jinja2 import (
    BaseLoader,
    ChoiceLoader,
    Environment,
    PackageLoader,
    PrefixLoader,
    StrictUndefined,
    TemplateAssertionError,
    nodes,
)

#: The prefix core's own shipped templates are addressable under, and the **only** spelling that
#: reaches them: ``jaz/system_prompt.jinja2``. A bare name never resolves to a core template.
CORE_TEMPLATE_PREFIX = "jaz"

# Prefix-only, deliberately — core's package loader is NOT also mounted under the bare names.
#
# Why: `{% extends "..." %}` and every configured template name are lookups through the *rendering*
# env, so they see whatever a host registered. While core answered bare names too, a host file
# named `system_prompt.jinja2` was reachable from inside core's own templates, and a host file that
# extended that bare name became its own parent — the render died with `RecursionError` naming
# neither file. Reserving one spelling for core and leaving bare names entirely to hosts makes the
# two namespaces disjoint, so neither can shadow the other and the failure mode cannot arise.
#
# The cost is that a config naming a core template bare stops resolving. That is affordable
# because no live config does: jaz-evals-internal ships no templates and registers no
# directory, so nothing names a core template bare. Core's own defaults are prefixed below.
_CORE_TEMPLATE_LOADER = PackageLoader("jaz", "templates")

#: The single loader object that answers ``jaz/<name>``. One instance, not one per call: the
#: chain is rebuilt on every registration, and identity against this is what tells core's entry
#: apart from a host's when filtering the old chain.
_CORE_PREFIX_LOADER = PrefixLoader({CORE_TEMPLATE_PREFIX: _CORE_TEMPLATE_LOADER})

_jinja_env = Environment(
    loader=_CORE_PREFIX_LOADER,
    autoescape=False,
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
    # A referenced-but-unpassed variable raises at render instead of silently rendering as
    # falsy. Templates therefore never need `{% if x is not defined %}` guards — every var a
    # template uses must be supplied by its render site (matches the baking env, _meta_env).
    undefined=StrictUndefined,
)


def _static_extends_targets(ast: nodes.Template) -> list[str]:
    """The ``{% extends %}`` parents a template names *unconditionally*, as literals.

    Two things are skipped, and taking either would be a false positive — this check must never
    reject a template that renders fine.

    A dynamic parent (``{% extends parent_var %}``) is a ``nodes.Name``, not a ``nodes.Const``, so
    its target is unknowable without rendering.

    A *conditional* one — ``{% if x %}{% extends "a" %}{% else %}{% extends "b" %}{% endif %}`` —
    is legal Jinja that picks its parent at render, so only one of those edges is ever real.
    Reading the whole tree with ``find_all`` takes both and can close a loop no render can reach,
    which is why this walks direct children of ``ast.body`` only: the same "is the extends known
    statically" question Jinja's own compiler asks.
    """
    return [
        node.template.value
        for node in ast.body
        if isinstance(node, nodes.Extends)
        and isinstance(node.template, nodes.Const)
        and isinstance(node.template.value, str)
    ]


def _find_cycle(graph: dict[str, list[str]]) -> list[str] | None:
    """A cycle in the inheritance graph as the path that closes it, or ``None``.

    Iterative DFS with an explicit stack: a pathological chain should report a cycle, not raise
    ``RecursionError`` while looking for one.
    """
    WHITE, GREY, BLACK = 0, 1, 2
    colour: dict[str, int] = dict.fromkeys(graph, WHITE)
    for root in graph:
        if colour[root] != WHITE:
            continue
        path: list[str] = []
        stack: list[tuple[str, bool]] = [(root, False)]
        while stack:
            name, leaving = stack.pop()
            if leaving:
                colour[name] = BLACK
                path.pop()
                continue
            if colour[name] != WHITE:
                # A stale duplicate push. Purely a performance guard: no subtree is re-walked
                # without it, but the loop below still iterates the node's parents to discover
                # that, so each stale pop costs O(out-degree). See
                # `test_a_shared_high_degree_node_is_scanned_once` for the measurement.
                #
                # DEAD in production: `_syntax_check` is the only production caller and it caps
                # out-degree at 1 by rejecting two top-level `{% extends %}`, so it cannot build a
                # graph that reaches this. Kept anyway — it is one `continue`, and the walk's
                # complexity should not depend on a property of one caller's input. Do not justify
                # it as a general routine over a graph it does not own: `_find_cycle` is
                # module-private and its production caller builds the graph.
                continue
            colour[name] = GREY
            path.append(name)
            stack.append((name, True))
            for parent in graph.get(name, ()):
                if colour.get(parent) == GREY:
                    return [*path[path.index(parent) :], parent]
                if colour.get(parent, BLACK) == WHITE:
                    stack.append((parent, False))
    return None


def _syntax_check(loader: BaseLoader) -> None:
    """Compile everything ``loader`` can enumerate and reject an inheritance loop among them.

    Scoped to the loader just added rather than the whole chain, which is what the deleted
    import-time ``_precompile`` did. Compiling at *import* put every later mutation past Jinja's
    "first template was loaded" line, and re-walking the chain made each registration
    O(everything registered so far).

    Why compile at all, when Jinja caches lazily: ``{% extends %}``, ``{% include %}`` and
    ``{% import %}`` all resolve at **render**, not at compile, so ``get_template(child)`` does
    *not* compile the parent — a broken parent survives the child's compilation and fails on the
    first turn that renders it. Enumerating the loader reaches it directly.

    A *loop* is a different failure and needs the AST, not the compiler: nothing is wrong with any
    individual template, so ``{% extends "m.jinja2" %}`` inside ``m.jinja2`` compiles clean and
    dies with ``RecursionError`` at the first render (#1351). The parse feeding that check is the
    same one the compile step needs, so it costs one pass rather than two — but the compile step
    has to stay: ``TemplateAssertionError`` (a block defined twice, say) comes from the code
    generator and ``parse`` alone never sees it.

    Limits, all deliberate. An unenumerable loader (``FunctionLoader``) is not checked at all. A
    file that is enumerated but unreadable is skipped. A loader implementing ``load`` but not
    ``get_source`` is compiled but yields no source, so it gets the syntax check and no loop
    check. Only ``{% extends %}`` edges are walked —
    ``{% include %}`` and ``{% import %}`` can form the identical loop with the identical
    ``RecursionError``, and are not detected. Only edges *within this loader* are walked,
    so a loop closed through a template in core or in another loader is not found — and the edge
    is assumed to resolve to this loader's copy. ``add_template_loader`` makes that true against
    every loader registered *so far* by inserting ahead of them — but not against a later one,
    which is inserted ahead in turn, so a loop closed by a subsequent registration still renders
    as ``RecursionError``.
    """
    try:
        names = list(loader.list_templates())
    except TypeError:
        return
    graph: dict[str, list[str]] = {}
    edge_site: dict[str, tuple[int, str | None]] = {}
    for name in names:
        # `loader.get_source`, not `_jinja_env.get_template`: the latter resolves through the
        # whole chain, so a name this loader supplies but another one shadows would be checked
        # against the WRONG loader — silently passing on someone else's template.
        try:
            source, filename, _ = loader.get_source(_jinja_env, name)
        except (UnicodeDecodeError, OSError):
            # Enumerated but not readable as a template — a `FileSystemLoader` lists every file in
            # its directory, including binaries. Skipped rather than fatal: registering a
            # directory must not fail because something in it was never a template.
            #
            # `TemplateNotFound` lands here too (it subclasses `OSError`), and that catch is what
            # makes the retry below necessary: a loader may override `load` WITHOUT implementing
            # `get_source`, a shape Jinja's own `BaseLoader` docs sanction, and `BaseLoader.
            # get_source` then raises `TemplateNotFound` for every one of its templates. Without
            # the retry the whole check switches off for such a loader — silently, and the syntax
            # half of it worked before this function stopped compiling through `load`.
            try:
                loader.load(_jinja_env, name)
            except (UnicodeDecodeError, OSError):
                pass
            # No source means no AST, so these templates contribute no `{% extends %}` edge and
            # are invisible to the loop walk. Syntax is checked; a loop among them is not.
            continue
        ast = _jinja_env.parse(source, name, filename)
        extends = [node for node in ast.body if isinstance(node, nodes.Extends)]
        if len(extends) > 1:
            # Two top-level `{% extends %}` compile fine and then raise `TemplateRuntimeError:
            # extended multiple times` on *every* render — Jinja emits that unconditionally on
            # reaching the second one and never looks up its parent. So this template can never
            # work, and saying so here beats letting it through to fail on the turn that needs it
            # (or, if its parents happen to close a loop, be mis-reported as a cycle).
            #
            # Counted over the `Extends` nodes rather than over `_static_extends_targets`: the
            # defect is how MANY top-level extends there are, which stays knowable when the
            # parents are not. Counting resolved targets instead would accept one literal plus one
            # dynamic parent, which is exactly as un-renderable — and is why the offending line
            # is reported rather than the parent names, which a dynamic target does not have.
            raise TemplateAssertionError(
                f"{name!r} has {len(extends)} top-level {{% extends %}} tags. Jinja raises "
                "'extended multiple times' whenever such a template renders, so it can never be "
                "used. Keep one parent, or make the choice conditional.",
                extends[1].lineno,
                name,
                filename,
            )
        targets = _static_extends_targets(ast)
        graph[name] = targets
        if extends:
            # Where the closing `{% extends %}` of a loop sits, so the error can carry a real
            # location. At most one per template — two were rejected above.
            edge_site[name] = (extends[0].lineno, filename)
        # From the AST rather than the source, so the parse above is not repeated. This is what
        # catches the compiler-only errors `parse` does not.
        _jinja_env.compile(ast, name, filename)

    # Edges to templates this loader does not supply need no filtering out: `_find_cycle`
    # treats an unknown target as already-finished through its `.get` defaults, so it is
    # simply not followed. An explicit pre-filter was redundant with that and is gone.
    cycle = _find_cycle(graph)
    if cycle:
        # Abbreviated: the path is as long as the loop, and a 2000-deep one produced a
        # 31,000-character message. The two ends identify it; the middle is noise.
        shown = (
            " -> ".join(cycle)
            if len(cycle) <= 8
            else " -> ".join(
                [*cycle[:4], f"...({len(cycle) - 6} more)...", *cycle[-2:]]
            )
        )
        # `_find_cycle` returns the path that closes the loop, so its last edge runs from
        # `cycle[-2]` to `cycle[-1]`: the `{% extends %}` in `cycle[-2]` is what makes the graph
        # circular, and is the one line a reader can act on. A self-loop degenerates to the same
        # thing (`[m, m]`).
        # Indexed, not `.get` with a default: `cycle[-2]` is on the cycle, so `graph[cycle[-2]]`
        # is non-empty, so it had an `{% extends %}` and is in `edge_site`. A default would be a
        # fabricated line 1 in an unnamed file — indistinguishable from a real location, which is
        # the wrong way for this to fail.
        closing = cycle[-2]
        lineno, origin = edge_site[closing]
        raise TemplateAssertionError(
            "template inheritance loop in this loader: "
            + shown
            + ". A template that inherits from itself, directly or through a chain, renders as "
            "a RecursionError naming no template, so it is rejected here instead.",
            lineno,
            closing,
            origin,
        )


def _is_core_loader(loader: BaseLoader) -> bool:
    """Whether ``loader`` is the one JAZ installed for its own templates.

    A plain identity check, which is only possible because :data:`_CORE_PREFIX_LOADER` is a single
    module-private object rather than one built per call. A caller cannot impersonate it:
    constructing an equivalent ``PrefixLoader({"jaz": ...})`` fails this *and* is rejected
    upstream for claiming the reserved prefix.
    """
    return loader is _CORE_PREFIX_LOADER


def add_template_loader(loader: BaseLoader) -> None:
    """Make a Jinja loader's templates addressable by name in JAZ's prompt templates.

    This is how you supply your own prompt: add a loader, then name one of its templates
    in the setting you want to replace — ``CodeOnlyProtocol(system_prompt_template=...)``,
    ``PythonREPL(repl_description_template=...)``, and their siblings all resolve names through
    the environment this adds to.

    Any loader works, so templates may live wherever suits you::

        jaz.add_template_loader(FileSystemLoader("/home/me/prompts"))
        jaz.add_template_loader(PackageLoader("myapp", "prompts"))
        jaz.add_template_loader(DictLoader({"mine.jinja2": source}))   # no files at all

    Your templates are rendered by JAZ's environment, so they inherit its settings — notably
    ``StrictUndefined``, under which referencing a variable JAZ does not pass raises rather than
    rendering empty. A shipped template is always addressable under the reserved ``jaz/`` prefix
    (``{% include "jaz/user_prompt.jinja2" %}``), whatever a loader of yours is called.

    Args:
        loader: Any :class:`jinja2.BaseLoader`.

    Your names are yours: JAZ answers only ``jaz/<name>``, so a template of your own called
    ``system_prompt.jinja2`` collides with nothing and is what that name resolves to.

    Every template the loader can enumerate is compiled here, so a syntax error is reported at
    this call rather than at the first render that needs it — including in a template of *this
    loader's* that another one ``{% extends %}``, which compiling the child alone would not
    catch. A loader that cannot enumerate (``FunctionLoader``) is added unchecked, and a
    file it lists that cannot be read as text is skipped. Note a ``FileSystemLoader`` lists
    **every** file in its directory, so keep non-templates out of one you add.

    Adding a loader that is already in the chain does nothing, so a caller holding one loader
    may add it defensively without disturbing precedence. The check runs when it is added, so
    templates a *mutable* loader gains afterwards are never checked — validate those yourself.

    Raises:
        ValueError: if the loader offers names under the reserved ``jaz/`` prefix, which JAZ
            answers first — those entries could never resolve.
        jinja2.TemplateSyntaxError: if one of the loader's templates fails to parse or to
            compile. Failures the compiler reports rather than the parser raise the
            ``TemplateAssertionError`` subclass, among them a duplicate ``{% block %}`` name and
            an unknown filter. Two more raise it that Jinja would not catch until render: more
            than one top-level ``{% extends %}``, and templates that inherit from each other in a
            loop — the latter otherwise a bare ``RecursionError`` naming no template. Loop
            detection covers this loader's own templates and only
            ``{% extends %}`` written as a literal: a loop closed through another loader, through
            a conditional or computed parent, or through ``{% include %}`` / ``{% import %}`` is
            not found.

    The chain is process-wide, so two configurations cannot hold different templates under one
    name — give them distinct names instead — and it does not cross a process boundary: a worker
    started by ``multiprocessing`` or ``ProcessPoolExecutor`` must add it again.

    The prompt-template guide lists the variables each template is passed and the blocks each
    shipped template exposes.
    """
    # Core's loader stays FIRST, and that is what makes the reserved prefix a guarantee rather
    # than a promise: `jaz/<name>` cannot be answered by anyone else, whether or not the offending
    # loader could have been detected. Detection is not always possible — `list_templates` is
    # optional in Jinja's protocol, so a `FunctionLoader` resolving `jaz/system_prompt.jinja2` on
    # demand enumerates nothing and no check could see it. Ordering closes that by construction.
    #
    # Bare names are the host's, in newest-first order, since core answers none of them
    # (`CORE_TEMPLATE_PREFIX`). There is nothing left for core to shadow, which is why the
    # shipped-name collision check this used to run is gone: it compared against core's bare
    # names, and core has none.
    reserved = f"{CORE_TEMPLATE_PREFIX}/"
    offered: frozenset[str] = frozenset()
    try:
        offered = frozenset(
            n for n in loader.list_templates() if n.startswith(reserved)
        )
    except TypeError:
        # Unenumerable, so nothing to inspect. The escape hatch is wider than the obvious
        # `FunctionLoader`: `ChoiceLoader.list_templates` PROPAGATES a member's `TypeError`, so
        # one unenumerable member also skips the check for its enumerable siblings. Accepted in
        # every such case, because core's position in the chain already makes its names
        # unreachable — the cost is only that the offender's `jaz/…` entries are silently dead
        # instead of reported.
        pass
    if not offered and isinstance(loader, PrefixLoader):
        # A `PrefixLoader` enumerates as `<prefix><delimiter><name>`, so the check above already
        # sees a `jaz`-prefixed one under the default `/`. This branch adds the case it cannot:
        # a `jaz` prefix over an inner loader that raises on `list_templates`. Delimiter-aware,
        # because `PrefixLoader({"jaz": …}, delimiter=".")` emits `jaz.x` and collides with
        # nothing — rejecting that would be a false positive.
        if (
            loader.mapping.get(CORE_TEMPLATE_PREFIX) is not None
            and loader.delimiter == "/"
        ):
            raise ValueError(
                f"template loader maps the reserved {CORE_TEMPLATE_PREFIX!r} prefix, which "
                "addresses JAZ's own templates and is answered by JAZ first, so nothing behind "
                "it could resolve. Map some other prefix."
            )
    if offered:
        raise ValueError(
            f"template loader offers name(s) under the reserved {reserved!r} prefix: "
            f"{', '.join(sorted(offered))}. That prefix addresses JAZ's own templates and is "
            "answered by JAZ first, so these would never resolve. Use any other name."
        )

    current = _jinja_env.loader
    # Re-registering a loader already in the chain returns without touching anything. Rebuilding
    # instead would be far from a no-op: the chain is assembled as `[core, loader, *rest]`, so it
    # would MOVE the loader to the head and flip precedence against every host loader registered
    # since — an effect the caller re-registering cannot see, and never one they asked for. It
    # would also re-run `_syntax_check` over the whole loader and flush the cache each time, which
    # a caller holding one loader and registering it on a hot path pays for repeatedly.
    #
    # Identity, not equality: two distinct loaders over the same directory both land, which is
    # what a caller who built them separately meant.
    if isinstance(current, ChoiceLoader):
        if any(ldr is loader for ldr in current.loaders):
            return
        rest = [ldr for ldr in current.loaders if not _is_core_loader(ldr)]
    elif current is loader:
        return
    elif current is not None and not _is_core_loader(current):
        rest = [current]
    else:
        rest = []
    # Mutating a live `Environment` — swapping `.loader`, clearing `.cache` — is what Jinja's
    # docs warn about: "Modifications on environments after the first template was loaded will
    # lead to surprising effects and undefined behavior." Registering BEFORE the first render is
    # therefore inside the supported window, and it is a window that exists at all only because
    # this module no longer pre-compiles at import. It used to, which put every mutation a caller
    # could make permanently after first load.
    #
    # After the first render the swap is verified rather than documented: a `Template` already
    # compiled under the old chain keeps rendering (it holds its own code object, not the
    # loader), a newly reachable name resolves, and an `{% extends %}` re-resolves through the new
    # chain — the loader is consulted per lookup, unlike the compile-time settings (`undefined`,
    # the delimiters, filters and globals) that get baked into compiled code. The cache clear is
    # what makes it take effect; without it an already-compiled name keeps its old file.
    # Checked BEFORE installing, so a `TemplateSyntaxError` leaves the chain untouched — the same
    # contract the `ValueError` paths above have, and what a caller who handles the error expects.
    _syntax_check(loader)
    _jinja_env.loader = ChoiceLoader([_CORE_PREFIX_LOADER, loader, *rest])
    if _jinja_env.cache is not None:
        # Memory hygiene rather than correctness: Jinja keys its cache on `(weakref(loader), name)`
        # and this assigns a fresh `ChoiceLoader` every time, so a stale entry could not be hit
        # anyway. Dropped here so the old chain's compiled templates are not pinned forever.
        _jinja_env.cache.clear()
