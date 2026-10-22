"""A ContextVar-routed ``sys.stdout`` proxy for isolated REPL output capture.

Capture is complete for the main context and asyncio-propagated execs; stream writes in a
*raw* agent-spawned thread are a documented boundary (see **Out of scope**).

The Python REPL captures the agent's ``print()`` by swapping ``print`` in the exec
namespace's ``__builtins__`` (see ``python_repl._make_print`` / ``_capture_print``). That
lexical swap is fast and works even in a raw worker thread, but it only catches the *name*
``print`` — it misses anything that reaches the stdout *stream* another way: ``pprint``,
``sys.stdout.write(...)``, and ``print`` inside an *imported* function (whose ``print``
resolves to the real builtin, not the swapped one). This module closes that gap by also
routing the stdout stream itself into the active capture buffer.

**Why a proxy and not ``contextlib.redirect_stdout``.** ``redirect_stdout`` *reassigns*
``sys.stdout`` to a specific buffer for a scope. Under JAZ's concurrency — the async exec
path runs each exec via ``asyncio.to_thread``, and ``asyncio.gather`` can run several nested
invokes at once — overlapping redirect scopes clobber the single global ``sys.stdout`` slot
and cross-contaminate agents. Instead we install a **stable proxy object once** and route
each write through a **``_current_buffer`` ContextVar**: the object in the ``sys.stdout``
slot never changes, and ``to_thread`` / task creation copy the context inward, so each
concurrent exec resolves its own buffer with no clobber. That is the concurrency-safe
version of the same idea.

**Why install per invoke-tree (refcounted), not at import and not per-exec.**
Installing at import would replace ``sys.stdout`` process-wide and interfere with a host
that manages stdout (pytest's ``capsys``/``capfd``, an ipykernel ``OutStream``) whenever no
invoke is even running. Reassigning per-exec would put us back in the ``redirect_stdout``
race. Installing once at the outermost invoke (via ``capturing()``, refcounted so nested and
concurrent trees share one install) is both concurrency-safe *and* transparent outside an
invoke — when no capture buffer is active the proxy delegates to the host sink, so a plain
``print()`` at the REPL prompt or a pytest test behaves exactly as before.

**``_host_stdout`` is the sink captured at install, not ``sys.__stdout__``.** It is whatever
the host had in ``sys.stdout`` when we installed — an ipykernel ``OutStream`` in a notebook,
pytest's capture object under a test, the tty in a shell. ``sys.__stdout__`` is only "the
console" in a bare terminal; empirically it is invisible in Jupyter under
``capture_fd_output=False`` and escapes pytest's ``capsys``. Framework diagnostics
(``PrintLogger``) bind that host sink directly, so they render where the human looks and are
never captured.

**Out of scope:**
- fd-level / C-level / subprocess writes (anything bypassing the Python ``sys.stdout``
  object). Catching those needs ``dup2`` on fd 1, which is process-global and reintroduces
  exactly the cross-agent clobber this design rejects.
- Stream writes in a *raw* thread the agent spawns (``threading.Thread`` /
  ``ThreadPoolExecutor``). ContextVars are **not** inherited by raw threads — only
  ``asyncio.to_thread`` / task creation copy the context inward — so ``_current_buffer.get()``
  is ``None`` there and the proxy routes to the host sink, i.e. ``pprint`` /
  ``sys.stdout.write`` in a raw agent thread escape capture. Note the asymmetry with the
  lexical ``print`` swap: that is a process-global ``builtins`` mutation, so it *does* survive
  into raw threads (which is why ``python_repl._make_print`` locks the sink for the raw-thread
  case). There is no clean fix — raw threads fundamentally can't be made to copy the context
  — so raw-thread stream writes are best-effort by design.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from io import StringIO
from typing import IO, Any, TextIO

# The buffer the *current* execution's stdout should route to, or None → the host sink.
# Set per-exec by PythonREPL._capture_print. contextvars are copied into asyncio.to_thread
# workers and gathered tasks, so concurrent execs each resolve their own buffer — but NOT
# into raw threading.Thread / ThreadPoolExecutor workers (see the module's Out of scope).
_current_buffer: ContextVar[StringIO | None] = ContextVar(
    "jaz_current_stdout_buffer", default=None
)

_install_lock = threading.Lock()
_refcount = 0
_host_stdout: TextIO | None = None
_proxy: _CapturingStdout | None = None


class _CapturingStdout:
    """Stable stdout proxy: routes writes to the active capture buffer, else the host sink.

    Installed once (refcounted) in the ``sys.stdout`` slot; the routing decision is
    per-context (``_current_buffer``), never by swapping the slot — see the module docstring.
    Delegates the rest of the text-stream interface to the host sink so libraries that probe
    ``sys.stdout`` (``isatty``/``fileno``/``encoding``/…) keep working.
    """

    def __init__(self, host: TextIO) -> None:
        self._host = host

    def _target(self) -> IO[str]:
        buf = _current_buffer.get()
        return buf if buf is not None else self._host

    def write(self, s: str) -> int:
        return self._target().write(s)

    def writelines(self, lines: Any) -> None:
        self._target().writelines(lines)

    def flush(self) -> None:
        self._target().flush()

    def isatty(self) -> bool:
        return self._host.isatty()

    def fileno(self) -> int:
        return self._host.fileno()

    def __getattr__(self, name: str) -> Any:
        # Anything not overridden (encoding, errors, buffer, …) comes from the host.
        return getattr(self._host, name)


def set_current_buffer(buffer: StringIO) -> Token[StringIO | None]:
    """Route this context's stdout writes to ``buffer``; returns a reset token."""
    return _current_buffer.set(buffer)


def reset_current_buffer(token: Token[StringIO | None]) -> None:
    _current_buffer.reset(token)


def install() -> None:
    """Install the proxy as ``sys.stdout`` (idempotent, refcounted).

    On the first (outermost) install we snapshot the host sink; later installs (nested or
    concurrent invoke trees) only bump the refcount.
    """
    global _refcount, _host_stdout, _proxy
    with _install_lock:
        if _refcount == 0:
            host = sys.stdout
            _host_stdout = host
            _proxy = _CapturingStdout(host)
            sys.stdout = _proxy
        _refcount += 1


def uninstall() -> None:
    """Balance an ``install()``; restore the host sink when the last holder leaves."""
    global _refcount, _host_stdout, _proxy
    with _install_lock:
        if _refcount == 0:
            return
        _refcount -= 1
        if _refcount == 0:
            # Only restore if nothing else replaced sys.stdout underneath us.
            if _proxy is not None and sys.stdout is _proxy:
                sys.stdout = _host_stdout  # type: ignore[assignment]
            _host_stdout = None
            _proxy = None


@contextmanager
def capturing() -> Generator[None, None, None]:
    """Keep the stdout proxy installed for the duration of an invoke (refcounted).

    Wrap the invoke body with this so ``sys.stdout`` is the proxy while any exec runs;
    outside all invokes ``sys.stdout`` is left untouched.
    """
    install()
    try:
        yield
    finally:
        uninstall()
