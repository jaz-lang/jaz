"""Tools for inspecting Python objects in a REPL for LLM interaction.

Simplified from IPython's oinspect.py - removed colors, HTML, and terminal formatting.
Removed functionality related to source files.
Removed dependencies on IPython internals.
"""

import ast
import inspect
import linecache
import types
from inspect import signature
from textwrap import dedent, indent
from typing import Any

# Builtin docstrings to ignore
_func_call_docstring = types.FunctionType.__call__.__doc__
_object_init_docstring = object.__init__.__doc__


def getdoc(obj) -> str | None:
    """Stable wrapper around inspect.getdoc.

    Allows objects to provide their docstrings via a getdoc() method.
    """
    try:
        ds = obj.getdoc()
        if isinstance(ds, str):
            return inspect.cleandoc(ds)
    except Exception:
        pass

    return inspect.getdoc(obj)


def getsource(obj, oname: str = "") -> str | None:
    """Get source code for an object.

    Parameters
    ----------
    obj : object
        Object whose source code we will attempt to extract
    oname : str
        (optional) name under which the object is known

    Returns
    -------
    src : str or None
    """
    if isinstance(obj, property):
        sources = []
        for attrname in ["fget", "fset", "fdel"]:
            fn = getattr(obj, attrname)
            if fn is not None:
                oname_prefix = f"{oname}." if oname else ""
                sources.append(f"# {oname_prefix}{attrname}")
                if inspect.isfunction(fn):
                    _src = getsource(fn)
                    if _src:
                        sources.append(dedent(_src))
                else:
                    sources.append(f"{oname_prefix}{attrname} = {fn!r}\n")
        return "\n".join(sources) if sources else None

    obj = _get_wrapped(obj)

    try:
        src = inspect.getsource(obj)
    except TypeError:
        # Try looking for class definition
        try:
            src = inspect.getsource(obj.__class__)
        except (OSError, TypeError):
            return None
    except OSError:
        return None

    return src


def _get_wrapped(obj):
    """Get the original object if wrapped in decorators.

    Arbitrarily cuts off after 100 levels to avoid infinite loops.
    """
    orig_obj = obj
    i = 0
    while hasattr(obj, "__wrapped__"):
        obj = obj.__wrapped__
        i += 1
        if i > 100:
            return orig_obj
    return obj


def _source_contains_docstring(src: str | None, doc: str) -> bool:
    """Check whether the source contains the docstring.

    Helper to skip displaying docstring if already in source.
    """
    if not src:
        return False
    try:
        (def_node,) = ast.parse(dedent(src)).body
        return ast.get_docstring(def_node) == doc  # type: ignore
    except Exception:
        return False


def get_signature(obj, oname: str = "") -> str | None:
    """Return the call signature for any callable object.

    Returns None if the object is not callable or signature cannot be determined.
    """
    if not callable(obj):
        return None
    try:
        return f"{oname}{signature(obj)}"
    except Exception:
        return None


def get_info(obj, oname: str = "", detail_level: int = 0) -> dict[str, Any]:
    """Get detailed information about an object.

    Parameters
    ----------
    obj : any
        Object to inspect
    oname : str
        Name of the variable pointing to obj
    detail_level : int
        0 for basic info, 1 for more detailed info including source

    Returns
    -------
    dict with keys:
        - name: variable name
        - type_name: type of the object
        - base_class: base class if applicable
        - string_form: str(obj)
        - length: len(obj) if applicable
        - signature: call signature
        - docstring: documentation
        - source: source code (if detail_level > 0)
        - init_signature: __init__ signature for classes
        - init_docstring: __init__ docstring for classes
        - call_signature: __call__ signature for callable instances
        - call_docstring: __call__ docstring for callable instances
        - subclasses: list of subclasses for classes
    """
    out: dict[str, Any] = {
        "name": oname,
        "type_name": None,
        "base_class": None,
        "string_form": None,
        "length": None,
        "signature": None,
        "docstring": None,
        "source": None,
        "init_signature": None,
        "init_docstring": None,
        "call_signature": None,
        "call_docstring": None,
        "subclasses": None,
    }

    # Type
    out["type_name"] = type(obj).__name__

    # Base class
    try:
        out["base_class"] = str(obj.__class__)
    except Exception:
        pass

    # String form (truncate if too long for basic detail level)
    try:
        ostr = str(obj)
        if detail_level == 0 and len(ostr) > 200:
            # Truncate long strings at basic detail level
            ostr = ostr[:100] + " <...> " + ostr[-100:]
        out["string_form"] = ostr
    except Exception:
        pass

    # Length
    try:
        out["length"] = str(len(obj))
    except Exception:
        pass

    # Docstring
    ds = getdoc(obj)
    if ds is None:
        ds = "<no docstring>"

    # For classes
    if inspect.isclass(obj):
        # Get __init__ signature and docstring
        try:
            init_sig = get_signature(obj, oname)
            if init_sig:
                out["init_signature"] = init_sig
        except Exception:
            init_sig = None

        try:
            init_ds = getdoc(obj.__init__)
            if init_ds and init_ds != _object_init_docstring:
                out["init_docstring"] = init_ds
        except Exception:
            pass
        else:
            if init_sig is None:
                # Get signature from init if top-level sig failed.
                # Can happen for built-in types (list, etc.).
                try:
                    init_sig = get_signature(obj.__init__, oname)
                except Exception:
                    pass

        # Subclasses
        try:
            names = [sub.__name__ for sub in type.__subclasses__(obj)]
            if names:
                if len(names) < 10:
                    out["subclasses"] = ", ".join(names)
                else:
                    out["subclasses"] = ", ".join(names[:10] + ["..."])
        except Exception:
            pass

    # For instances
    else:
        # Signature
        sig = get_signature(obj, oname)
        if sig:
            out["signature"] = sig

        # Get __init__ docstring
        try:
            init_doc = getdoc(obj.__init__)
            if init_doc and init_doc != _object_init_docstring:
                out["init_docstring"] = init_doc
        except Exception:
            pass

        # For callable instances, get __call__ info
        if callable(obj) and not (inspect.isfunction(obj) or inspect.ismethod(obj)):
            try:
                call_sig = get_signature(obj.__call__, oname)  # type: ignore
                if call_sig and call_sig != out.get("signature"):
                    out["call_signature"] = call_sig
            except Exception:
                pass

            try:
                call_ds = getdoc(obj.__call__)  # type: ignore
                # Skip Python's auto-generated docstrings
                if call_ds == _func_call_docstring:
                    call_ds = None
                if call_ds:
                    out["call_docstring"] = call_ds
            except Exception:
                pass

    # Source code (only if detail_level > 0)
    if detail_level > 0:
        linecache.checkcache()
        try:
            src = getsource(obj, oname)
            if src:
                out["source"] = src.rstrip()
        except Exception:
            pass

    # Only include docstring if source doesn't contain it (avoid duplication)
    if ds and not _source_contains_docstring(out.get("source"), ds):
        out["docstring"] = ds

    return out


def format_info(info: dict[str, Any]) -> str:
    """Format object info dict as a readable string.

    Parameters
    ----------
    info : dict
        Info dict returned by get_info()

    Returns
    -------
    str : formatted information
    """
    lines = []

    if info.get("type_name"):
        lines.append(f"Type: {info['type_name']}")

    if info.get("signature"):
        lines.append(f"Signature: {info['signature']}")

    if info.get("init_signature"):
        lines.append(f"Init signature: {info['init_signature']}")

    if info.get("call_signature"):
        lines.append(f"Call signature: {info['call_signature']}")

    if info.get("string_form"):
        lines.append(f"String form: {info['string_form']}")

    if info.get("length"):
        lines.append(f"Length: {info['length']}")

    if info.get("docstring"):
        lines.append(f"Docstring:\n{indent(info['docstring'], '  ')}")

    if info.get("init_docstring"):
        lines.append(f"Init docstring:\n{indent(info['init_docstring'], '  ')}")

    if info.get("call_docstring"):
        lines.append(f"Call docstring:\n{indent(info['call_docstring'], '  ')}")

    if info.get("source"):
        lines.append(f"Source:\n{info['source']}")

    if info.get("subclasses"):
        lines.append(f"Subclasses: {info['subclasses']}")

    return "\n".join(lines)


def inspect_object(obj, oname: str = "", detail_level: int = 0) -> str:
    """Inspect an object and return formatted information.

    Parameters
    ----------
    obj : any
        Object to inspect
    oname : str
        Name of the variable
    detail_level : int
        0 for basic, 1 for detailed (includes source)

    Returns
    -------
    str : formatted information about the object
    """
    info = get_info(obj, oname, detail_level)
    return format_info(info)
