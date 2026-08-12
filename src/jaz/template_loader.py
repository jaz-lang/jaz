from jinja2 import (
    ChoiceLoader,
    DictLoader,
    Environment,
    FileSystemLoader,
    PackageLoader,
    StrictUndefined,
)

_jinja_env = Environment(
    loader=PackageLoader("jaz", "templates"),
    autoescape=False,
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
    # A referenced-but-unpassed variable raises at render instead of silently rendering as
    # falsy. Templates therefore never need `{% if x is not defined %}` guards — every var a
    # template uses must be supplied by its render site (matches the baking env, _meta_env).
    undefined=StrictUndefined,
)

# Backing store for in-memory string templates registered via ``register_template_string``.
# A single shared ``DictLoader`` (added to the loader chain once) holds them all, keyed by name;
# it aliases this dict, so mutating it in place updates what the loader can resolve. Module-level
# so the loader instance stays stable across calls.
_string_templates: dict[str, str] = {}
_string_loader = DictLoader(_string_templates)


def _precompile() -> None:
    """Pre-compile every template the current loader can list (import-time warmup)."""
    loader = _jinja_env.loader
    if loader is not None:
        for template_name in loader.list_templates():
            _jinja_env.get_template(template_name)


def register_template_dir(path: str) -> None:
    """Add a filesystem directory to the Jinja search path (used by out-of-core callers).

    Core ships only *minimal, feature-flag-free* prompt templates (REPL description,
    truncation advice, must-exit warning). The rich, boolean-gated variants live outside
    the ``jaz`` package — under ``evals/templates/`` — because whether the prompt mentions a
    given feature (REPL history, exposed inputs, …) is an eval-configuration concern, not a
    framework one (#568). An out-of-core layer (the eval harness) calls this once to make its
    template directory loadable by name, so a config that sets e.g.
    ``repl_description_template: python_repl_description_simplified.jinja2`` resolves against
    it. Core stays ignorant of what those templates say; it only renders the configured name,
    forwarding config-supplied variables generically.

    Idempotent: appends ``path`` behind the package loader (package templates win on a
    name clash) and resets the template cache so newly reachable templates are picked up.
    Re-registering an already-present ``path`` is a no-op — the test suite (and any caller)
    can call it repeatedly without the loader chain growing an unbounded run of duplicate
    ``FileSystemLoader``s for the same directory.
    """
    global _jinja_env
    current = _jinja_env.loader
    # Keep the package loader first so core templates take precedence over same-named files
    # in a registered dir; add the new dir after any already-registered ones.
    if isinstance(current, ChoiceLoader):
        # Dedupe: skip if a FileSystemLoader already searches this dir. FileSystemLoader
        # normalizes ``path`` into its ``searchpath`` list, so compare against that.
        if any(
            isinstance(loader, FileSystemLoader) and path in loader.searchpath
            for loader in current.loaders
        ):
            return
        loaders = [*current.loaders, FileSystemLoader(path)]
    else:
        loaders = (
            [current, FileSystemLoader(path)] if current else [FileSystemLoader(path)]
        )
    _jinja_env.loader = ChoiceLoader(loaders)
    # Drop the compiled-template cache so subsequent get_template() sees the new dir.
    if _jinja_env.cache is not None:
        _jinja_env.cache.clear()
    _precompile()


def register_template_string(name: str, source: str) -> None:
    """Register an in-memory Jinja template under *name* (used by out-of-core callers).

    The eval harness renders a *meta-template* into a finished inner-template string and needs core
    to load it **by name** at the usual render sites — ``PythonREPL.get_description`` /
    ``get_truncation_advice`` and the must-exit hook all call ``_jinja_env.get_template(name)`` — so
    registering the string here lets those call sites stay unchanged rather than growing a
    render-from-string path (#898). This is the in-memory sibling of :func:`register_template_dir`:
    that one makes a *directory* of template files loadable by name; this one makes a single
    *string* loadable by name, for templates the harness synthesizes at runtime and never writes to
    disk.

    The backing ``DictLoader`` sits **behind** the package loader (package templates win on a name
    clash), matching ``register_template_dir``'s precedence. Harness inner templates use distinct
    generated names, so clashes don't arise in practice. Re-registering a name overwrites its
    source; the loader's ``uptodate`` check forces a recompile on the next ``get_template`` without
    a global cache flush, so unrelated compiled templates are left intact.
    """
    global _jinja_env
    _string_templates[name] = source
    current = _jinja_env.loader
    if isinstance(current, ChoiceLoader):
        if _string_loader not in current.loaders:
            # Append behind existing loaders (package stays first, so it wins name clashes).
            _jinja_env.loader = ChoiceLoader([*current.loaders, _string_loader])
    else:
        _jinja_env.loader = ChoiceLoader(
            [current, _string_loader] if current else [_string_loader]
        )
    # Warm + syntax-check just this template (parse errors surface now, not at first render). No
    # global cache clear: the DictLoader's uptodate callback already recompiles an overwritten name.
    _jinja_env.get_template(name)


# Pre-compile all core templates at import time.
_precompile()
