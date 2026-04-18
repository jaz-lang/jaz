def abbrev_repr(obj: object, max_length: int) -> str:
    """Return an abbreviated string representation of an object.

    If the string representation of the object exceeds `max_length`,
    it is truncated and an ellipsis is added.
    """
    repr_str = repr(obj)
    if len(repr_str) > max_length:
        return repr_str[:max_length] + "...[truncated]"
    return repr_str


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
    string: str, max_length: int, prefix_ratio: float = 0.8
) -> tuple[str, bool]:
    """Abbreviate string when it exceeds max_length.

    Returns:
        A tuple of (abbreviated_string, was_truncated).
    """
    if len(string) <= max_length:
        return string, False
    prefix_length = int(prefix_ratio * max_length)
    suffix_length = max_length - prefix_length
    if prefix_length == 0:
        abbreviated_string = (
            f"[Truncated. Only the last {suffix_length} characters are shown]\n"
            + string[-suffix_length:]
        )
    elif suffix_length == 0:
        abbreviated_string = (
            f"[Truncated. Only the first {prefix_length} characters are shown]\n"
            + string[:prefix_length]
        )
    else:
        omitted = len(string) - prefix_length - suffix_length
        abbreviated_string = (
            f"[Truncated. Only the first {prefix_length} and "
            f"last {suffix_length} characters are shown]\n"
            + string[:prefix_length]
            + f"\n\n[...{omitted} characters omitted...]\n\n"
            + string[-suffix_length:]
        )
    return abbreviated_string, True
