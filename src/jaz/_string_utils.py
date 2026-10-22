def abbrev_repr(obj: object, max_length: int | None) -> str:
    """Return an abbreviated string representation of an object.

    If the string representation of the object exceeds `max_length`,
    it is truncated and an ellipsis is added. ``None`` disables truncation.
    """
    # Guarded: the loggers render returned values and namespaces with this, and a raising
    # ``__repr__`` there would abort the handler and drop the line it was writing.
    repr_str = safe_repr(obj)
    if max_length is not None and len(repr_str) > max_length:
        return repr_str[:max_length] + "...[truncated]"
    return repr_str


def safe_repr(value: object, cap: int | None = None) -> str:
    """``repr(value)``, cut to ``cap`` characters if given, or a placeholder naming the error if it
    raises."""
    # For observers rendering arbitrary user objects into a trace: a ``__repr__`` is user code, and
    # a raising one must cost the observer one field, not the record. Raising out of a hook's
    # ``teardown`` is worse than losing the record — it replaces whatever exception the run itself
    # was propagating. The cap is applied after the fact because there is no way to ask ``repr``
    # for a shorter answer.
    try:
        return repr(value)[:cap]
    except Exception as exc:  # noqa: BLE001 - reporting the failure IS the fallback
        return f"<repr failed: {type(exc).__name__}>"


def summarize_exception(exc: BaseException) -> str:
    """One-line-per-error summary of a (possibly grouped) exception.

    ``str(group)`` only prints a group's own message plus a sub-exception *count*
    (e.g. ``"Multiple errors occurred (2 sub-exceptions)"``) — it never stringifies
    its children. So expand groups explicitly (recursing for nested groups) and
    indent each child, so every composed error is rendered rather than collapsed
    into a count. For a plain exception this is ``"Type: message"``.
    """
    # The one rendering of an exception that reaches the agent (CodeOnlyProtocol's
    # observation and history entry) and the traces (OTelTracing, TrajectoryRecorder).
    # Each consumer calls it directly rather than reading a derived property on
    # ``Continue``, so the result type stays a plain fact carrier and serialization is
    # each consumer's own call.
    if isinstance(exc, BaseExceptionGroup):
        children = "\n".join(
            f"  {line}"
            for e in exc.exceptions
            for line in summarize_exception(e).splitlines()
        )
        return f"{type(exc).__name__}: {exc.message}\n{children}"
    # ``str(exc)`` runs the exception's ``__str__``, which is user code, and this runs inside
    # hooks' handlers and teardown -- where the dispatcher swallows a raise -- so it must not
    # raise. Unguarded, a trace loses the outcome it is recording, and a hook relaying a failure
    # to the agent aborts, so the agent is never sent back: ``ValidateREPLCode``'s rejected code
    # runs anyway, and ``ValidateReturn``'s rejected value reaches the invoke-complete backstop,
    # which raises the validator's exception out of the invoke if it rejects again there and
    # accepts the value if it does not. Guarded the way ``safe_repr`` guards repr.
    try:
        message = str(exc)
    except Exception as str_exc:  # noqa: BLE001 - reporting the failure IS the fallback
        message = f"<str failed: {type(str_exc).__name__}>"
    return f"{type(exc).__name__}: {message}"


def backtickify(text: str) -> str:
    """
    Return a string with backticks around it.

    Args:
        text: The string to backtickify.

    Returns:
        A string with backticks around it.
    """
    return f"`{text}`"


# `prefix_ratio` is the head/tail split of the retained text. 0.5 (even), not head-weighted:
# weighting the head assumes the informative part of an over-long value is at the front — true
# for prose, false for REPL output, where the result, and a Python traceback's exception line,
# land at the end.
# No production caller passes anything else; the parameter stays because the split is this
# function's to define and its tests exercise the ends.
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
