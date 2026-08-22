"""
Per-exec resource guards for exec and eval operations: a **timeout** guard
(prevents infinite loops) and a **memory** guard (prevents runaway allocations
from OOMing the process). Both are per-thread and per-exec, checked in the
*same* ``sys.monitoring`` LINE callback, and owner-tagged so a nested invoke can
tell its own breach from an outer exec's.

The timeout guard has two enforcement layers (see
design/design_features/repl_exec_timeout.md):

Layer 1 -- ``sys.monitoring`` (PEP 669) deadline registry. A single LINE
callback checks a per-thread stack of absolute deadlines and raises an
owner-tagged ``REPLTimeoutError`` when the nearest one has expired. LINE events
are scoped to agent-compiled code objects via ``set_local_events`` (registered
by ``secure_compile``), so jaz internals, library functions, and third-party
code incur zero overhead. Because the deadline stays expired, the callback
re-raises on every subsequent line -- a bare ``except:`` in agent code cannot
swallow enforcement. Nested deadlines compose: the stack always holds every
outstanding deadline on the thread, so an outer (parent) deadline fires inside
an inner exec instead of being suspended by it.

Layer 2 -- SIGALRM nearest-deadline heap (POSIX main thread only). Bridges
Layer 1's blind spot: blocking C calls (``time.sleep``, blocking IO,
signal-polling C code such as ``sre``). A process-global min-heap of
outstanding main-thread deadlines keeps ``setitimer`` armed for the nearest
one; the handler raises the same owner-tagged ``REPLTimeoutError``. One-shot
swallowing is not fatal because Layer 1 re-raises on the next agent line.

The ``owner_id`` tag lets REPL catch sites distinguish *their own* expired
deadline (converted to a ``Continue`` for the agent, as before) from an
*outer* exec's deadline that fired inside a nested invoke (surfaced as a
``Raise`` so the sub-invoke terminates gracefully and the exception
propagates to the parent).

Spawn-time deadline inheritance -- deadline stacks are keyed by thread ID, so
a thread the agent spawns would otherwise have no stack and escape Layer 1
(LINE events do fire there via code-object identity, but would find no
deadlines). To close this, ``threading.Thread.start`` is patched (lazily, on
the first deadline push) so a thread spawned while deadlines are outstanding
on the spawning thread *adopts* references to those same deadline entries for
its lifetime; the existing LINE callback then enforces them in the worker.
Threads spawned with no deadlines outstanding (jaz internals, litellm, test
machinery) are untouched. End-of-exec semantics: an entry whose exec
completed cleanly is *released* on pop (workers that legitimately outlive a
successful exec run free), while an entry whose own ``REPLTimeoutError``
propagated stays *latched* -- still enforceable in adopting workers, so
threads spawned by a timed-out exec are killed at their next agent line. The
next deadline pushed on the origin thread releases its latched entries
(amnesty), bounding how far a latched kill reaches into reused threads (e.g.
``ThreadPoolExecutor`` workers serving a later exec).

Memory guard -- a per-thread RSS cap checked in the same Layer 1 LINE callback
(there is no SIGALRM analogue: a single blocking C-level allocation can outrun
it, so it is a growth guard, not a hard ceiling -- see jaz-lang/jaz#812). The
check reads current *process* RSS from ``/proc/self/statm`` (throttled to one
read per ``_MEMORY_CHECK_INTERVAL``) and raises an owner-tagged
``REPLMemoryError`` when that total exceeds the nearest cap on the thread's
stack. It rides the same ``register_code`` scoping as the timeout, so it is
enforced on all agent code independently of whether a timeout is set. Unlike
deadlines, memory caps are *not* adopted by spawned threads (only the deadline
snapshot is), so a cap bounds the spawning thread, not its children.

Known limitations: blocking C calls in worker threads (SIGALRM is main-thread
only), pool threads reused across execs (the snapshot happens at thread spawn,
not task submission), and raw ``_thread.start_new_thread`` (bypasses
``Thread.start``). See the design doc's Known limitations. The memory guard adds
its own: a single large C-level allocation can outrun the throttle before the
next Python line, and spawned threads escape the per-thread cap.
"""

import asyncio
import contextlib
import heapq
import inspect
import itertools
import os
import signal
import sys
import threading
import time
import weakref
from collections.abc import Iterator
from types import CodeType, FrameType

__all__ = [
    "REPLTimeoutError",
    "REPLMemoryError",
    "guarded_exec",
    "guarded_eval",
    "aexec",
    "aeval",
    "guarded_aexec",
    "guarded_aeval",
    "is_our_timeout",
    "is_our_memory_error",
    "new_owner",
    "register_code",
    "current_rss_bytes",
]


# ---------------------------------------------------------------------------
# Freezegun-immune monotonic clock
# ---------------------------------------------------------------------------
# All deadline/expiry math below MUST use a clock that cannot be frozen at the
# Python level, or the timeout silently never fires. ``freezegun.freeze_time``
# (used by AppWorld during task execution) replaces ``time.monotonic`` with a
# frozen fake. Under that, a deadline set as ``monotonic() + timeout`` and an
# expiry check ``deadline <= monotonic()`` can never become true -- the SIGALRM
# signal still arrives in real wall-clock time (``setitimer`` is a kernel
# timer), but the handler's monotonic-based check says "not expired" and
# re-arms forever, so the loop hangs unbounded. We read ``CLOCK_MONOTONIC``
# straight from libc via ``ctypes``, bypassing any Python-level patch.
#
# Scope / assumptions (be precise -- the immunity is NOT universal):
#   * The libc read is the only path that is *actually* freeze-immune.
#   * Fallback: on platforms where the libc probe is unavailable we fall back to
#     ``time.monotonic`` captured at *import*. That fallback is freeze-immune
#     ONLY under the import-before-freeze assumption -- if this module is first
#     imported while a ``freeze_time`` is already active, the captured reference
#     is itself the frozen fake and the fallback is NOT immune. In practice we
#     import at process start (before AppWorld's per-task freeze), so this holds;
#     it is an assumption, not a guarantee.
#   * Layer 2 (SIGALRM) is POSIX-only, so non-POSIX platforms lean entirely on
#     the Layer-1 line callback -- which uses this same clock, hence the same
#     import-before-freeze caveat there.
_real_monotonic_fallback = time.monotonic


def _make_real_monotonic():
    try:
        import ctypes
        import ctypes.util

        class _timespec(ctypes.Structure):
            _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]

        _libc = ctypes.CDLL(
            ctypes.util.find_library("c") or "libc.so.6", use_errno=True
        )
        _clock_gettime = _libc.clock_gettime
        _clock_gettime.argtypes = [ctypes.c_int, ctypes.POINTER(_timespec)]
        _clock_gettime.restype = ctypes.c_int
        _CLOCK_MONOTONIC = getattr(time, "CLOCK_MONOTONIC", 1)
        # Probe once; if it fails, fall back rather than ship a broken clock.
        _probe = _timespec()
        if _clock_gettime(_CLOCK_MONOTONIC, ctypes.byref(_probe)) != 0:
            raise OSError("clock_gettime(CLOCK_MONOTONIC) probe failed")

        def real_monotonic() -> float:
            ts = _timespec()
            if _clock_gettime(_CLOCK_MONOTONIC, ctypes.byref(ts)) != 0:
                return _real_monotonic_fallback()
            return ts.tv_sec + ts.tv_nsec * 1e-9

        return real_monotonic
    except Exception:
        return _real_monotonic_fallback


_real_monotonic = _make_real_monotonic()


class REPLTimeoutError(Exception):
    """Raised when execution exceeds the specified timeout.

    Carries an ``owner_id`` identity sentinel identifying which
    ``guarded_exec`` / ``guarded_eval`` call's deadline expired,
    so nested execs can distinguish their own timeout from an outer one.
    """

    owner_id: object | None

    def __init__(self, *args: object, owner_id: object | None = None) -> None:
        super().__init__(*args)
        self.owner_id = owner_id


def new_owner() -> object:
    """Mint a fresh owner-identity sentinel for a timeout scope."""
    return object()


def is_our_timeout(e: REPLTimeoutError, owner_id: object) -> bool:
    """Whether ``e`` was raised for the deadline owned by ``owner_id``."""
    return e.owner_id is owner_id


class REPLMemoryError(Exception):
    """Raised when an exec's process RSS exceeds its configured memory cap.

    The memory guard (Layer 1) is the byte-budget analogue of the timeout: a
    per-thread cap checked in the *same* ``sys.monitoring`` LINE callback that
    enforces deadlines, so it needs no separate instrumentation and — crucially
    — fires even when ``repl_exec_timeout`` is ``None`` (the LINE events are
    enabled unconditionally by ``register_code``). It bounds runaway *memory*
    (e.g. mutate-during-iteration list growth) that a wall-clock timeout only
    catches incidentally, and can accumulate GBs before it does.

    Carries an ``owner_id`` (like ``REPLTimeoutError``) so a nested exec can
    tell its own cap breach from an enclosing exec's via ``is_our_memory_error``.
    """

    owner_id: object | None

    def __init__(self, *args: object, owner_id: object | None = None) -> None:
        super().__init__(*args)
        self.owner_id = owner_id


def is_our_memory_error(e: REPLMemoryError, owner_id: object) -> bool:
    """Whether ``e`` was raised for the memory cap owned by ``owner_id``."""
    return e.owner_id is owner_id


# Reading current RSS is Linux-specific (``/proc/self/statm``, field 1 = resident
# pages). We use *current* RSS deliberately, not ``getrusage().ru_maxrss`` — the
# latter is a monotonic *peak* (and KB on Linux / bytes on macOS), so it can't
# tell us the process has *shrunk* back under the cap and its unit is
# platform-specific. On platforms without ``/proc`` the guard is a no-op (the
# reader returns ``None`` and the check simply never fires); documented as a
# known limitation rather than emulated with a slower portable probe.
try:
    _PAGE_SIZE: int | None = os.sysconf("SC_PAGE_SIZE")
except (AttributeError, ValueError, OSError):  # pragma: no cover - non-POSIX
    _PAGE_SIZE = None

# Re-reading /proc on *every* agent line would dominate loop cost, so each cap
# reads RSS at most once per this interval. Measured (this machine): a
# /proc/self/statm read is ~16 us and a tight pure-Python loop runs ~12M lines/s,
# so reading on every line would cost ~190x the loop's own runtime — unusable.
# At 0.1s (~10 reads/s) the read overhead is ~0.02% of a core; 0.01s (~0.16%)
# would tighten the overshoot bound 10x but that is irrelevant at the realistic
# ~10-60 MB/s growth under instrumentation (0.1s overshoot is 1-6 MB, negligible
# against any headroom'd cap) — and it would NOT help the fast-alloc case below.
# So 0.1s: negligible cost, few-MB overshoot for the growth patterns we target.
#
# KNOWN LIMITATION: a tight *fast-allocating* loop (e.g.
# ``for _ in range(n): buf.append(bytearray(1 << 20))``) can add hundreds of MB
# to GBs between two checks and OOM before the next /proc read — Layer 1 can be
# outrun here, and a smaller interval doesn't fix it (the same loop overshoots at
# 0.01s too). The intended catch-all is the coarse ``RLIMIT_AS`` hard backstop
# (kernel-enforced, no per-line sampling), deferred to the #812 follow-up.
_MEMORY_CHECK_INTERVAL = 0.1  # seconds


def current_rss_bytes() -> int | None:
    """Current resident set size in bytes, or ``None`` if unavailable."""
    if _PAGE_SIZE is None:
        return None
    try:
        with open("/proc/self/statm") as f:
            resident_pages = int(f.readline().split()[1])
    except (OSError, ValueError, IndexError):  # pragma: no cover - defensive
        return None
    return resident_pages * _PAGE_SIZE


def _memory_message(limit_bytes: int, rss_bytes: int) -> str:
    return (
        f"Memory limit exceeded: total process RSS {rss_bytes / 1e6:.0f} MB exceeded "
        f"the {limit_bytes / 1e6:.0f} MB ceiling in effect for this exec. This is a "
        f"*process-wide* ceiling — it counts all memory the process holds (including "
        f"any outer or concurrent scopes), not just this exec's own allocations — so "
        f"the exec (and everything running alongside it) must fit within it. It usually "
        f"means unbounded growth — e.g. appending to a list while iterating it, or "
        f"accumulating results without a bound."
    )


def _timeout_message(timeout: float) -> str:
    return f"Execution exceeded timeout of {timeout} seconds."


def _current_task_or_none() -> "asyncio.Task[object] | None":
    """The running ``asyncio.Task``, or ``None`` outside a running loop.

    ``asyncio.current_task()`` raises ``RuntimeError`` when no loop is running (the
    ordinary sync-exec case), so it can't be called bare on the guard paths.
    """
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


# Sentinel for "the current task has not been resolved yet" in the deadline scan, so a
# pure-sync stack (every entry ``task_ref is None``) never pays for the lookup at all.
_UNRESOLVED_TASK = object()


class _DeadlineEntry:
    __slots__ = (
        "cancelled",
        "deadline",
        "origin_tid",
        "owner_id",
        "released",
        "task_ref",
        "timed_out",
        "timeout",
    )

    def __init__(
        self,
        deadline: float,
        owner_id: object,
        timeout: float,
        origin_tid: int,
        task_ref: "weakref.ref[asyncio.Task[object]] | None" = None,
    ) -> None:
        self.deadline = deadline
        self.owner_id = owner_id
        self.timeout = timeout  # original duration, for the error message
        self.origin_tid = origin_tid  # thread that pushed this entry
        # The asyncio Task that pushed this entry, or None if pushed outside a running
        # loop (the sync path). Used ONLY to tell same-thread *peer* tasks apart from
        # genuine dynamic nesting -- see `check_deadline_current_thread`. A weakref, not
        # the Task: a latched entry outlives its exec, and a strong ref would pin the
        # Task (and everything its frames hold) for as long as the entry is retained.
        self.task_ref = task_ref
        # Lazy heap deletion: popped entries are flagged and pruned when they
        # reach the heap top. A flag (not an id-keyed set) because CPython
        # reuses object ids -- a set of ids would silently cancel future
        # entries allocated at a recycled address.
        self.cancelled = False
        # released: the exec completed cleanly and popped this entry --
        # adopting worker threads stop enforcing it.
        self.released = False
        # timed_out: this entry's own REPLTimeoutError propagated out of its
        # exec -- pop latches the entry (keeps it enforceable in adopting
        # workers) instead of releasing it.
        self.timed_out = False


def _acquire_tool_id() -> int:
    """Claim a free sys.monitoring tool id.

    Prefer the unnamed ids (3, 4) to avoid colliding with debuggers (0),
    coverage (1), profilers (2), or optimizers (5) that may also be active.
    """
    for tool_id in (3, 4, sys.monitoring.OPTIMIZER_ID, sys.monitoring.PROFILER_ID):
        if sys.monitoring.get_tool(tool_id) is None:
            sys.monitoring.use_tool_id(tool_id, "jaz_timeout")
            return tool_id
    raise RuntimeError("No free sys.monitoring tool id available for jaz timeouts")


class _MemoryCap:
    """A per-thread RSS ceiling enforced while an exec runs.

    Byte-budget analogue of ``_DeadlineEntry``. Simpler because memory has no
    Layer-2 (RSS can't be sampled from a signal cheaply/portably) and no
    latching/adoption: RSS is process-wide, so the LINE callback on any thread
    with an active cap already observes growth caused by every thread. ``owner_id``
    tags the ``REPLMemoryError`` so nested execs distinguish their own breach.
    """

    __slots__ = ("limit_bytes", "owner_id", "origin_tid", "_last_check")

    def __init__(self, limit_bytes: int, owner_id: object, origin_tid: int) -> None:
        self.limit_bytes = limit_bytes
        self.owner_id = owner_id
        self.origin_tid = origin_tid
        # Monotonic time of this cap's last /proc read (per-cap throttle). Only
        # touched by the owning thread inside its own LINE callback -> no lock.
        self._last_check = 0.0


class _GuardRegistry:
    """Per-thread deadline stacks (Layer 1) + main-thread SIGALRM heap (Layer 2).

    Also holds per-thread memory caps (``_thread_memory_caps``), enforced in the
    same LINE callback — see ``_MemoryCap`` and ``check_memory_current_thread``.
    """

    def __init__(self) -> None:
        self._thread_stacks: dict[int, list[_DeadlineEntry]] = {}
        self._thread_memory_caps: dict[int, list[_MemoryCap]] = {}
        self._lock = threading.Lock()
        # Latched entries: timed out and popped, but still enforceable in
        # adopting worker threads until the origin thread's next push.
        self._latched: list[_DeadlineEntry] = []
        self._thread_start_patched = False
        # Layer 2 state -- only ever touched on the main thread.
        self._sigalrm_heap: list[tuple[float, int, _DeadlineEntry]] = []
        self._heap_counter = itertools.count()  # heap tiebreaker
        self._sigalrm_installed = False
        self._in_critical_section = False  # defer handler raise during push/pop
        self._sigalrm_supported = hasattr(signal, "setitimer") and hasattr(
            signal, "SIGALRM"
        )

    # -- Layer 1: per-thread stacks ------------------------------------------

    def push(
        self, timeout: float, owner_id: object, use_sigalrm: bool = True
    ) -> _DeadlineEntry:
        # use_sigalrm=False arms Layer 1 only. The async path (guarded_aexec) passes
        # it: Layer 2 / SIGALRM fired while the event loop is parked in epoll (a deadline
        # held across an `await`) raises into the loop internals and tears the loop down
        # (#567). Layer 1 only fires while agent Python lines execute, so it is safe.
        tid = threading.get_ident()
        # Tag the entry with the pushing Task so a concurrent sibling exec on this same
        # thread is not mistaken for an enclosing one (#489). None on the sync path.
        task = _current_task_or_none()
        entry = _DeadlineEntry(
            _real_monotonic() + timeout,
            owner_id,
            timeout,
            tid,
            weakref.ref(task) if task is not None else None,
        )
        with self._lock:
            # Amnesty: a new exec on this thread releases latched (timed-out)
            # entries it originated, bounding how far a latched kill reaches
            # into threads that survived the timed-out exec (e.g. idle
            # ThreadPoolExecutor workers serving a later exec).
            if self._latched:
                kept: list[_DeadlineEntry] = []
                for latched in self._latched:
                    if latched.origin_tid == tid:
                        latched.released = True
                    else:
                        kept.append(latched)
                self._latched = kept
            self._thread_stacks.setdefault(tid, []).append(entry)
        self._ensure_thread_start_patched()
        if use_sigalrm and self._on_main_thread():
            self._sigalrm_push(entry)
        return entry

    def pop(self, entry: _DeadlineEntry) -> None:
        tid = threading.get_ident()
        with self._lock:
            stack = self._thread_stacks.get(tid)
            if stack is not None:
                try:
                    stack.remove(entry)
                except ValueError:
                    pass
                if not stack:
                    del self._thread_stacks[tid]
            if entry.timed_out:
                # Latch: keep the expired entry enforceable in adopting
                # workers' stacks so threads spawned by the timed-out exec
                # are killed at their next agent line (released at the
                # origin thread's next push -- see amnesty above).
                #
                # Known leak (accepted): a latched entry is only ever pruned by
                # a *subsequent push on the same origin_tid*. If the origin
                # thread times out and then never runs another exec -- e.g. a
                # short-lived worker that ran one sub-agent exec, timed out, and
                # is discarded rather than reused -- its entry stays in
                # ``self._latched`` for the process lifetime. In the steady
                # main-thread case (the common one) the next exec self-cleans,
                # so it is bounded; only a churn of many distinct short-lived
                # exec-running threads accumulates unboundedly. We document the
                # leak rather than add a periodic sweep: the bound holds for
                # realistic usage and a sweep would add lock traffic to every
                # exec for a case that does not arise in practice.
                self._latched.append(entry)
            else:
                # Clean completion: workers that legitimately outlive this
                # exec stop being enforced against its deadline.
                entry.released = True
        if self._on_main_thread():
            self._sigalrm_pop(entry)

    def check_deadline_current_thread(self) -> None:
        """Raise if the nearest deadline *enforceable against the running code* expired.

        "Enforceable" is not simply "outstanding on this thread" (#489). The stack model
        assumes **dynamic nesting**: any other entry on your thread is an *enclosing*
        exec, so raising the thread's nearest deadline — tagged with that entry's owner —
        is right, and ``is_our_timeout`` sorts own-vs-foreign afterwards.

        ``asyncio`` breaks that assumption. ``asyncio.gather`` interleaves many execs as
        **siblings on one thread**, so another entry here can be a *peer* rather than an
        ancestor. Enforcing it thread-wide produced two distinct bugs: the wrong task was
        interrupted (whichever hit the next LINE event), and because the error carried the
        *peer's* ``owner_id`` the exec seam saw a foreign timeout and escalated it to a
        fatal ``Raise`` instead of a recoverable ``Continue``. A sub-invoke with a 30s
        budget could be killed by an unrelated sibling's 10ms one.

        So an entry pushed **on this thread** by a *different* Task is skipped. Same-Task
        entries are kept, which preserves genuine nesting: an inner ``aexec`` awaited
        directly from an outer one runs in the same Task, so the outer deadline still
        bounds it and still classifies as foreign.

        Entries **adopted from another thread** (``origin_tid != tid``, seeded by the
        ``Thread.start`` instrumentation) bypass the Task filter entirely: a spawned
        worker is deliberately bound by the deadlines outstanding when it was spawned, and
        it has no Task of its own to match against.

        Known limitation (accepted): an outer exec's deadline no longer line-enforces
        against a child exec the outer spawned as a *separate* Task (e.g. via
        ``gather``/``create_task``) — that child is a peer by this rule and
        indistinguishable from an unrelated one, since asyncio exposes no Task parentage.
        The outer remains bounded by its own ``asyncio.timeout`` (which cancels the awaited
        children), and the child by its own deadline; what is lost is only line-level
        interruption of a child that is CPU-bound *and* never awaits. Recovering it needs
        real Task ancestry, which would mean hooking Task creation.
        """
        tid = threading.get_ident()
        stack = self._thread_stacks.get(tid)
        if not stack:
            return
        now = _real_monotonic()
        nearest: _DeadlineEntry | None = None
        # Resolved lazily and at most once: a stack with no Task-tagged entries (every
        # sync exec) must not pay for an `asyncio.current_task()` probe per LINE event.
        current_task: object = _UNRESOLVED_TASK
        for entry in stack:
            if entry.released:
                continue
            if entry.task_ref is not None and entry.origin_tid == tid:
                if current_task is _UNRESOLVED_TASK:
                    current_task = _current_task_or_none()
                # Dead weakref => the pushing Task is gone, so it cannot be the running
                # one; `is not` handles that and the None (no running Task) case alike.
                if entry.task_ref() is not current_task:
                    continue
            if nearest is None or entry.deadline < nearest.deadline:
                nearest = entry
        if nearest is not None and now > nearest.deadline:
            raise REPLTimeoutError(
                _timeout_message(nearest.timeout), owner_id=nearest.owner_id
            )

    # -- Layer 1: per-thread memory caps -------------------------------------

    def push_memory(self, limit_bytes: int, owner_id: object) -> _MemoryCap:
        tid = threading.get_ident()
        cap = _MemoryCap(limit_bytes, owner_id, tid)
        with self._lock:
            self._thread_memory_caps.setdefault(tid, []).append(cap)
        return cap

    def pop_memory(self, cap: _MemoryCap) -> None:
        tid = threading.get_ident()
        with self._lock:
            stack = self._thread_memory_caps.get(tid)
            if stack is not None:
                try:
                    stack.remove(cap)
                except ValueError:
                    pass
                if not stack:
                    del self._thread_memory_caps[tid]

    def check_memory_current_thread(self) -> None:
        """Raise if *total* process RSS exceeds the tightest cap on this thread.

        The cap is compared against process-wide RSS, not this exec's own footprint,
        so the invariant is "each active scope must fit its declared budget within
        total process memory": with nested/concurrent execs, ``min(stack)`` (the
        tightest cap) is what a shared process RSS is held to, and owner-tagging lets
        an inner scope's own breach resolve differently from a foreign outer one.

        Throttled per cap (``_MEMORY_CHECK_INTERVAL``) so the /proc read doesn't
        dominate agent-line dispatch. Lock-free reads (like ``check_deadline_current_thread``):
        the caller is the owning thread, and a benign race only delays a check by
        one line.
        """
        stack = self._thread_memory_caps.get(threading.get_ident())
        if not stack:
            return
        cap = min(stack, key=lambda c: c.limit_bytes)
        now = time.monotonic()
        if now - cap._last_check < _MEMORY_CHECK_INTERVAL:
            return
        cap._last_check = now
        rss = current_rss_bytes()
        if rss is not None and rss > cap.limit_bytes:
            raise REPLMemoryError(
                _memory_message(cap.limit_bytes, rss), owner_id=cap.owner_id
            )

    # -- Spawn-time deadline inheritance --------------------------------------

    def snapshot_current_thread(self) -> list[_DeadlineEntry]:
        """Live (un-released) deadline entries on the calling thread.

        Captured by the ``Thread.start`` instrumentation so a spawned worker
        adopts the deadlines outstanding at spawn time. Entries are shared
        (not copied): release/latch decisions made at pop propagate to every
        adopting worker for free.
        """
        stack = self._thread_stacks.get(threading.get_ident())
        if not stack:
            return []
        return [entry for entry in stack if not entry.released]

    def adopt(self, tid: int, snapshot: list[_DeadlineEntry]) -> None:
        """Seed ``tid``'s deadline stack with a spawn-time snapshot."""
        with self._lock:
            self._thread_stacks[tid] = list(snapshot)

    def abandon(self, tid: int) -> None:
        """Drop ``tid``'s deadline stack (worker thread exiting)."""
        with self._lock:
            self._thread_stacks.pop(tid, None)

    def _ensure_thread_start_patched(self) -> None:
        """Lazily install the ``Thread.start`` instrumentation (first push)."""
        if self._thread_start_patched:
            return
        with self._lock:
            if self._thread_start_patched:
                return
            threading.Thread.start = _instrumented_thread_start  # pyright: ignore[reportAttributeAccessIssue]
            self._thread_start_patched = True

    # -- Layer 2: SIGALRM nearest-deadline heap ------------------------------

    def _on_main_thread(self) -> bool:
        return (
            self._sigalrm_supported
            and threading.current_thread() is threading.main_thread()
        )

    def _sigalrm_push(self, entry: _DeadlineEntry) -> None:
        self._in_critical_section = True
        try:
            if not self._sigalrm_installed:
                try:
                    signal.signal(signal.SIGALRM, _sigalrm_handler)
                except ValueError:
                    # Not actually on the main thread of the main interpreter
                    # (e.g. embedded); disable Layer 2 for this process.
                    self._sigalrm_supported = False
                    return
                self._sigalrm_installed = True
            heapq.heappush(
                self._sigalrm_heap, (entry.deadline, next(self._heap_counter), entry)
            )
            self._rearm()
        finally:
            self._in_critical_section = False

    def _sigalrm_pop(self, entry: _DeadlineEntry) -> None:
        if not self._sigalrm_installed:
            return
        self._in_critical_section = True
        try:
            entry.cancelled = True
            self._rearm()
        finally:
            self._in_critical_section = False

    def _prune(self) -> None:
        """Drop cancelled entries from the heap top."""
        while self._sigalrm_heap and self._sigalrm_heap[0][2].cancelled:
            heapq.heappop(self._sigalrm_heap)

    def _rearm(self) -> None:
        """(Re)arm the itimer for the nearest live deadline, or disarm."""
        self._prune()
        if self._sigalrm_heap:
            delay = max(1e-4, self._sigalrm_heap[0][0] - _real_monotonic())
            signal.setitimer(signal.ITIMER_REAL, delay)
        else:
            signal.setitimer(signal.ITIMER_REAL, 0)

    def expire_nearest(self) -> _DeadlineEntry | None:
        """Pop and return the nearest expired heap entry (handler helper)."""
        self._prune()
        if self._sigalrm_heap and self._sigalrm_heap[0][0] <= _real_monotonic():
            _, _, entry = heapq.heappop(self._sigalrm_heap)
            return entry
        return None


_registry = _GuardRegistry()


_original_thread_start = threading.Thread.start


def _instrumented_thread_start(self: threading.Thread) -> None:
    """``Thread.start`` wrapper: spawn-time deadline snapshot.

    If the spawning thread has deadlines outstanding, wrap the new thread's
    ``run`` so the worker adopts (shared references to) those entries for its
    lifetime, making the LINE callback enforce them there too. Threads spawned
    with no deadlines outstanding -- jaz internals, litellm workers, test
    machinery -- take the early exit and are untouched. This is the standard
    instrumentation point (Sentry, OpenTelemetry, gevent wrap it the same way).
    """
    snapshot = _registry.snapshot_current_thread()
    if snapshot:
        original_run = self.run

        def _run_with_adopted_deadlines() -> None:
            tid = threading.get_ident()
            _registry.adopt(tid, snapshot)
            try:
                original_run()
            finally:
                _registry.abandon(tid)

        self.run = _run_with_adopted_deadlines  # pyright: ignore[reportAttributeAccessIssue]
    _original_thread_start(self)


def _sigalrm_handler(signum: int, frame: FrameType | None) -> None:  # noqa: ARG001
    if _registry._in_critical_section:
        # Deadline expired while the registry is mid-mutation; retry shortly
        # rather than raising out of registry bookkeeping code.
        signal.setitimer(signal.ITIMER_REAL, 1e-3)
        return
    entry = _registry.expire_nearest()
    # Re-arm for any remaining deadlines *before* raising, so an exception
    # swallowed by agent code does not also lose the other pending deadlines.
    _registry._rearm()
    if entry is not None:
        raise REPLTimeoutError(_timeout_message(entry.timeout), owner_id=entry.owner_id)


def _line_callback(code: CodeType, line_number: int) -> None:  # noqa: ARG001
    _registry.check_deadline_current_thread()
    # Memory guard rides the same LINE callback (already enabled on all agent
    # code by register_code), so it fires independently of whether a timeout is
    # set — the ``repl_exec_timeout=None`` case is exactly where OOMs happen.
    _registry.check_memory_current_thread()


# sys.monitoring tool id + LINE callback, installed lazily on the first
# register_code rather than at import. Deferring the claim keeps jaz from taking
# a global sys.monitoring tool id away from a host app that merely imports jaz
# but never runs a deadline-instrumented exec, and makes this consistent with
# the other two installs (Thread.start patch, SIGALRM handler), which are also
# lazy. register_code is the only trigger needed: LINE events fire exclusively
# on code objects that register_code instrumented via set_local_events, and any
# exec that reaches push() was compiled (hence register_code'd) first; a raw
# string exec'd without register_code has no instrumented frames to enforce.
_TOOL_ID: int | None = None
_monitoring_lock = threading.Lock()


def _ensure_monitoring_installed() -> int:
    """Claim our sys.monitoring tool id and register the LINE callback (once).

    Idempotent and thread-safe (double-checked under ``_monitoring_lock``):
    the first caller installs, the rest get the cached id. Called from
    register_code so the callback is live before any code object has LINE
    events enabled on it.
    """
    global _TOOL_ID
    if _TOOL_ID is not None:
        return _TOOL_ID
    with _monitoring_lock:
        if _TOOL_ID is not None:
            return _TOOL_ID
        tool_id = _acquire_tool_id()
        # Global events stay OFF; coverage comes exclusively from
        # set_local_events on agent-compiled code objects (see register_code),
        # so unregistered frames -- jaz internals, library functions, stdlib --
        # incur zero overhead.
        sys.monitoring.register_callback(
            tool_id, sys.monitoring.events.LINE, _line_callback
        )
        _TOOL_ID = tool_id
        return _TOOL_ID


def register_code(code: CodeType) -> None:
    """Enable LINE events on ``code`` and every code object nested inside it.

    Called by ``secure_compile`` so that all agent-compiled code -- including
    functions and comprehensions the agent defines -- is covered by the
    deadline registry's LINE callback, on whatever thread it eventually runs.
    Stale registrations are inert: the callback is gated on the running
    thread's deadline stack being non-empty.

    Registration is unconditional -- we knowingly instrument even code first
    compiled while no deadline is active. Gating it on "does this exec have a
    timeout?" would be *wrong*, not just awkward, because a code object's
    deadline exposure is not a property of its compile -- it is a property of
    whichever exec (and thread) later runs it, which need not be the exec that
    compiled it:

    - Agent-defined functions and classes persist in the REPL namespace
      (``repl_state_locals``) across turns, so one compiled under no deadline
      is routinely called by a *later* turn's code that has one -- the same
      registered code object then runs under that later exec's deadline.
    - A registered code object can run on a worker thread that *adopts* an
      outstanding deadline at spawn time (see Layer 1 above).

    So threading the timeout into ``secure_compile`` to skip registration would
    not help: it would gate on the compiling exec's timeout, the wrong one. (A
    ``sys.monitoring`` ``DISABLE`` return from the callback is unusable for the
    same reason -- it disables events for that code location *permanently*,
    killing the later enforcement.) Registration is therefore decoupled from
    timeouts entirely; the deadline is consulted per line at run time.

    Accepted cost on the no-deadline path: enabling LINE events makes the
    interpreter dispatch to the callback on *every executed agent line*, even
    when the empty thread stack means nothing can be enforced (the callback
    itself just does one dict lookup and returns via ``check_deadline_current_thread``).
    That per-line dispatch -- not the lookup -- dominates: ~4x on a tight
    pure-Python loop (vs ~7x with a deadline actually armed). But it is a
    multiplier on *agent-bytecode* time only: C calls (e.g. ``json.loads``),
    Library/stdlib/jaz frames, and any unregistered code stay scoped-out and
    pay nothing. So for typical glue code -- a handful of lines between LLM and
    tool calls -- the absolute cost is negligible; it is only megaloops of pure
    Python under ``timeout=None`` that visibly pay. This is the price of
    decoupling registration from deadlines, and is accepted.
    """
    tool_id = _ensure_monitoring_installed()
    stack = [code]
    while stack:
        c = stack.pop()
        sys.monitoring.set_local_events(tool_id, c, sys.monitoring.events.LINE)
        for const in c.co_consts:
            if isinstance(const, CodeType):
                stack.append(const)


@contextlib.contextmanager
def _active_guards(
    timeout: float | None,
    memory_limit_bytes: int | None,
    owner: object,
    use_sigalrm: bool = True,
) -> "Iterator[_DeadlineEntry | None]":
    """Arm the deadline (if ``timeout``) and memory-cap (if ``memory_limit_bytes``)
    guards on the current thread for the duration of the ``with`` body, then
    release both. Yields the deadline entry (or ``None``) so the caller can latch
    it when its *own* deadline fires. The two guards are independent — a memory
    limit works with ``timeout=None`` and vice versa.
    """
    deadline = (
        _registry.push(timeout, owner, use_sigalrm=use_sigalrm)
        if timeout is not None
        else None
    )
    mem_cap = (
        _registry.push_memory(memory_limit_bytes, owner)
        if memory_limit_bytes is not None
        else None
    )
    try:
        yield deadline
    finally:
        # Pop in reverse order of push. Memory has no latching/adoption, so its
        # pop is a plain removal.
        if mem_cap is not None:
            _registry.pop_memory(mem_cap)
        if deadline is not None:
            _registry.pop(deadline)


def guarded_exec(
    code: str | CodeType,
    globals_dict: dict[str, object],
    timeout: float | None,
    owner_id: object | None = None,
    memory_limit_bytes: int | None = None,
) -> None:
    """
    Execute code, enforcing ``timeout`` and/or ``memory_limit_bytes`` via the
    registry.

    This implementation does not create any new threads, making it safe to use
    with reentrant locks and recursive calls.

    Args:
        code: Compiled code object to execute
        globals_dict: Global namespace for execution
        timeout: Timeout in seconds (None = no timeout)
        owner_id: Identity sentinel tagging this scope's ``REPLTimeoutError`` /
            ``REPLMemoryError`` (minted internally if not provided). Pass one from
            ``new_owner()`` to distinguish this scope from an outer exec's via
            ``is_our_timeout`` / ``is_our_memory_error``.
        memory_limit_bytes: Process-RSS ceiling in bytes (None = no cap). Enforced
            in the LINE callback independently of ``timeout``.

    Raises:
        REPLTimeoutError: If execution exceeds timeout (or an enclosing exec's
            deadline expires first -- check ``is_our_timeout``).
        REPLMemoryError: If process RSS exceeds the cap (check
            ``is_our_memory_error``).
    """
    if timeout is None and memory_limit_bytes is None:
        # No guards of our own; any *outer* deadlines/caps remain enforced.
        exec(code, globals_dict)
        return

    owner = owner_id if owner_id is not None else new_owner()
    with _active_guards(timeout, memory_limit_bytes, owner) as deadline:
        try:
            exec(code, globals_dict)
        except REPLTimeoutError as e:
            if deadline is not None and e.owner_id is deadline.owner_id:
                # Our own deadline fired: pop will latch the entry so threads
                # this exec spawned are killed too (instead of releasing it).
                deadline.timed_out = True
            raise


def guarded_eval(
    code: str | CodeType,
    globals_dict: dict[str, object],
    timeout: float | None,
    owner_id: object | None = None,
    memory_limit_bytes: int | None = None,
) -> object:
    """
    Evaluate code, enforcing ``timeout`` and/or ``memory_limit_bytes`` via the
    registry. Mirror of :func:`guarded_exec` for the eval/``return`` path.

    This implementation does not create any new threads, making it safe to use
    with reentrant locks and recursive calls.

    Args:
        code: Compiled code object to evaluate
        globals_dict: Global namespace for evaluation
        timeout: Timeout in seconds (None = no timeout)
        owner_id: Identity sentinel tagging this scope's ``REPLTimeoutError`` /
            ``REPLMemoryError`` (minted internally if not provided).
        memory_limit_bytes: Process-RSS ceiling in bytes (None = no cap).

    Returns:
        Result of evaluation

    Raises:
        REPLTimeoutError: If evaluation exceeds timeout (or an enclosing exec's
            deadline expires first -- check ``is_our_timeout``).
        REPLMemoryError: If process RSS exceeds the cap (check
            ``is_our_memory_error``).
    """
    if timeout is None and memory_limit_bytes is None:
        return eval(code, globals_dict)

    owner = owner_id if owner_id is not None else new_owner()
    with _active_guards(timeout, memory_limit_bytes, owner) as deadline:
        try:
            return eval(code, globals_dict)
        except REPLTimeoutError as e:
            if deadline is not None and e.owner_id is deadline.owner_id:
                # Our own deadline fired: pop will latch the entry so threads
                # this eval spawned are killed too (instead of releasing it).
                deadline.timed_out = True
            raise


async def aexec(code: CodeType, globals_dict: dict[str, object]) -> None:
    """Async ``exec``: run a code object compiled with ``PyCF_ALLOW_TOP_LEVEL_AWAIT``,
    awaiting it when the agent's code contains top-level ``await`` (``CO_COROUTINE``).

    ``eval`` (not ``exec``) is used so the coroutine an await-bearing module code object
    returns is *captured* and awaited — ``exec`` would discard it. Code without top-level
    ``await`` has ``CO_COROUTINE`` unset and runs synchronously here (free fallback).
    ``globals_dict`` is reused as both globals and locals (``eval(code, g)``) so
    top-level-await name binding writes back to the REPL namespace. No timeout — see
    ``guarded_aexec``. (#567)
    """
    result = eval(code, globals_dict)
    if code.co_flags & inspect.CO_COROUTINE:
        await result


async def aeval(code: CodeType, globals_dict: dict[str, object]) -> object:
    """Async ``eval``: evaluate a code object (compiled with
    ``PyCF_ALLOW_TOP_LEVEL_AWAIT``), awaiting the result when it is a coroutine
    (top-level ``await`` present). No timeout — see ``guarded_aeval``. (#567)"""
    result = eval(code, globals_dict)
    if code.co_flags & inspect.CO_COROUTINE:
        result = await result
    return result


async def guarded_aexec(
    code: CodeType,
    globals_dict: dict[str, object],
    timeout: float | None,
    owner_id: object | None = None,
    memory_limit_bytes: int | None = None,
) -> None:
    """Async ``exec`` under a ``timeout`` (and optional ``memory_limit_bytes``),
    for the ``ainvoke`` REPL path (#567). The memory cap is enforced independently
    of the timeout, mirroring the sync path.

    Two complementary bounds, **no SIGALRM** (Layer 2): a Layer-1 (``sys.monitoring``)
    deadline interrupts CPU-bound stretches in agent code — and carries any *outer*
    (foreign) deadlines on the thread's stack — while ``asyncio.timeout`` bounds time
    *suspended* at an ``await``, cancelling cleanly into the coroutine. Layer 2 is
    dropped because a SIGALRM firing while the loop is parked in epoll raises into the
    loop internals and tears it down; the cost is that blocking C calls *between* awaits
    aren't interruptible under async (see the aexec design doc / #567).

    Raises this module's owner-tagged ``REPLTimeoutError``: an own deadline (Layer 1 *or*
    ``asyncio.timeout``) is tagged with ``owner``; an outer exec's Layer-1 deadline keeps
    its own owner, so ``is_our_timeout`` still distinguishes foreign timeouts.
    """
    owner = owner_id if owner_id is not None else new_owner()
    # Memory cap wraps the whole body so it is active even when timeout is None
    # (no Layer-2/SIGALRM for memory; the LINE callback observes RSS on this and
    # any adopting thread). It composes with the async deadline below.
    mem_cap = (
        _registry.push_memory(memory_limit_bytes, owner)
        if memory_limit_bytes is not None
        else None
    )
    try:
        if timeout is None:
            # No own deadline; outer deadlines on the thread's stack stay enforced.
            await aexec(code, globals_dict)
            return
        entry = _registry.push(timeout, owner, use_sigalrm=False)
        try:
            async with asyncio.timeout(timeout):
                await aexec(code, globals_dict)
        except TimeoutError:
            # asyncio.timeout fired (an await overran our own deadline). asyncio raises
            # the real builtin TimeoutError; normalize it to our owner-tagged
            # REPLTimeoutError so the REPL exec seam treats it as an own timeout.
            entry.timed_out = True
            raise REPLTimeoutError(_timeout_message(timeout), owner_id=owner) from None
        except REPLTimeoutError as e:
            # Layer-1 deadline fired in agent Python code — ours, or a foreign outer one.
            if e.owner_id is entry.owner_id:
                entry.timed_out = True
            raise
        finally:
            _registry.pop(entry)
    finally:
        if mem_cap is not None:
            _registry.pop_memory(mem_cap)


async def guarded_aeval(
    code: CodeType,
    globals_dict: dict[str, object],
    timeout: float | None,
    owner_id: object | None = None,
    memory_limit_bytes: int | None = None,
) -> object:
    """Async ``eval`` under a ``timeout`` (and optional ``memory_limit_bytes``) —
    the eval-mode counterpart of ``guarded_aexec`` (same Layer-1 +
    ``asyncio.timeout``, no SIGALRM; memory cap independent of timeout). (#567)"""
    owner = owner_id if owner_id is not None else new_owner()
    mem_cap = (
        _registry.push_memory(memory_limit_bytes, owner)
        if memory_limit_bytes is not None
        else None
    )
    try:
        if timeout is None:
            return await aeval(code, globals_dict)
        entry = _registry.push(timeout, owner, use_sigalrm=False)
        try:
            async with asyncio.timeout(timeout):
                return await aeval(code, globals_dict)
        except TimeoutError:
            entry.timed_out = True
            raise REPLTimeoutError(_timeout_message(timeout), owner_id=owner) from None
        except REPLTimeoutError as e:
            if e.owner_id is entry.owner_id:
                entry.timed_out = True
            raise
        finally:
            _registry.pop(entry)
    finally:
        if mem_cap is not None:
            _registry.pop_memory(mem_cap)
