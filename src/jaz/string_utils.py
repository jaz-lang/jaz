def abbrev_repr(obj: object, max_length: int) -> str:
    """Return an abbreviated string representation of an object.

    If the string representation of the object exceeds `max_length`,
    it is truncated and an ellipsis is added.
    """
    repr_str = repr(obj)
    if len(repr_str) > max_length:
        return repr_str[:max_length] + "...[truncated]"
    return repr_str


def summarize_exception(exc: BaseException) -> str:
    """One-line-per-error summary of a (possibly grouped) exception.

    ``str(group)`` only prints a group's own message plus a sub-exception *count*
    (e.g. ``"Multiple errors occurred (2 sub-exceptions)"``) — it never stringifies
    its children. So expand groups explicitly (recursing for nested groups) and
    indent each child, so every composed error is rendered rather than collapsed
    into a count. For a plain exception this is ``"Type: message"``.

    Used to render an exception into agent-facing feedback. Called directly by each
    consumer that needs it — :class:`CodeOnlyProtocol` (both its
    ``render_observation`` and its ``build_history_entry`` record seam) and OTel tracing
    (:mod:`jaz.hooks.builtin.otel_tracing`) — rather than through a derived property on
    :class:`Continue`, so the result type stays a plain fact carrier
    and serialization is each consumer's own call.
    """
    if isinstance(exc, BaseExceptionGroup):
        children = "\n".join(
            f"  {line}"
            for e in exc.exceptions
            for line in summarize_exception(e).splitlines()
        )
        return f"{type(exc).__name__}: {exc.message}\n{children}"
    return f"{type(exc).__name__}: {exc}"


def backtickify(text: str) -> str:
    """
    Return a string with backticks around it.

    Args:
        text: The string to backtickify.

    Returns:
        A string with backticks around it.
    """
    return f"`{text}`"


def abbreviate_string(
    string: str, max_length: int, prefix_ratio: float = 0.5
) -> tuple[str, bool]:
    """Abbreviate string when it exceeds max_length.

    Truncation is expressed as a single **in-place substitution**: the removed span becomes
    ``[...N characters omitted...]``, spliced exactly where the characters were, with nothing
    added around it — no surrounding newlines and no header line.

    This is deliberately one code path for every ratio, replacing three branches that each
    rendered differently and, at ``prefix_ratio`` 0 or 1, announced the truncation *without*
    ever stating how much was dropped. The marker is now the sole, uniform signal, so:

    - the omitted count is always present (previously only in the both-ends case);
    - a consumer that splices output into a larger document gets no injected blank lines, so
      the surrounding text keeps its own shape;
    - the ratio only decides *where* the cut falls, never how it is described.

    Note the result can exceed ``max_length`` by the marker's own width — ``max_length`` bounds
    the retained *content*, not the rendered string. Callers size it as a content budget.

    A corollary worth knowing before treating this as a size limiter: on a cut that only just
    crosses the cap, the marker is wider than the span it replaces, so the result comes out
    *longer than the input* — a 101-character string at ``max_length=100`` renders 128
    characters, to elide one.

    Returns:
        A tuple of (abbreviated_string, was_truncated).
    """
    if len(string) <= max_length:
        return string, False
    prefix_length = int(prefix_ratio * max_length)
    suffix_length = max_length - prefix_length
    omitted = len(string) - prefix_length - suffix_length
    # Index from the front, not ``string[-suffix_length:]``: a zero suffix_length (prefix_ratio
    # == 1.0) would make that negative slice return the WHOLE string rather than nothing.
    suffix = string[len(string) - suffix_length :]
    return (
        f"{string[:prefix_length]}[...{omitted} characters omitted...]{suffix}",
        True,
    )
