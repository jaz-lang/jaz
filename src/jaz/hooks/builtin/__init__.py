"""Built-in hook implementations."""

from .atif_trace import ATIFTrace
from .budget_forcing import BudgetForcing
from .budget_pool import BudgetPool
from .compaction import Compaction
from .context_window import ContextWindowWarning
from .iteration_limit import IterationLimit
from .loggers import FileLogger, PrintLogger
from .metadata import MetaData
from .recursion_limit import RecursionLimit
from .repl_code_hooks import ValidateREPLCode
from .replay import ATIFReplay
from .return_hooks import ReturnType, ValidateReturn
from .rollout import FlatSample, Rollout, RolloutRecorder, TurnSample
from .workflow_replay import WorkflowReplay

# Conditionally import tracing hooks if opentelemetry is installed
try:
    from .jaeger_tracing import JaegerTracing  # noqa: F401
    from .langfuse_tracing import LangfuseTracing  # noqa: F401

    _TRACING_AVAILABLE = True
except ImportError:
    # OpenTelemetry not installed, tracing hooks unavailable.
    # Do NOT assign None — let the ImportError propagate so that
    # downstream re-exports (jaz.hooks.__init__) detect the absence
    # and users get a clear ImportError instead of a silent None.
    _TRACING_AVAILABLE = False

__all__ = [
    "ATIFReplay",
    "ATIFTrace",
    "BudgetForcing",
    "BudgetPool",
    "Compaction",
    "ContextWindowWarning",
    "IterationLimit",
    "RecursionLimit",
    "MetaData",
    "PrintLogger",
    "FileLogger",
    "WorkflowReplay",
    "ReturnType",
    "ValidateReturn",
    "ValidateREPLCode",
    # Rollout recording (token-native backends)
    "RolloutRecorder",
    "Rollout",
    "FlatSample",
    "TurnSample",
]

# Only export tracing hooks if available
if _TRACING_AVAILABLE:
    __all__.append("JaegerTracing")
    __all__.append("LangfuseTracing")


# Deprecated aliases for the pre-rename spellings — see the rationale block in
# ``jaz/hooks/__init__.py``. Repeated here because ``jaz.hooks.builtin`` has its own
# ``__all__``: the old names were importable from *this* namespace too, so aliasing only
# the parent would still break `from jaz.hooks.builtin import ReplayHook`.
BudgetForcingHook = BudgetForcing
CompactionHook = Compaction
# ContextWindow: the pre-``warning_text``-required spelling, kept importable after the rename to
# ContextWindowWarning. ContextWindowHook: the original ``*Hook`` alias.
ContextWindow = ContextWindowWarning
ContextWindowHook = ContextWindowWarning
IterationLimitHook = IterationLimit
# ValidateREPLInput: the pre-"REPL code" spelling — not a ``Hook``-suffix drop, but kept here
# for the same reason (it was in ``__all__``).
ValidateREPLInput = ValidateREPLCode
# Two pre-rename generations of ATIFReplay (ReplayHook → Replay → ATIFReplay); both were
# importable from this namespace, so both stay bound here.
Replay = ATIFReplay
ReplayHook = ATIFReplay
WorkflowReplayHook = WorkflowReplay

if _TRACING_AVAILABLE:
    JaegerTracingHook = JaegerTracing
    LangfuseTracingHook = LangfuseTracing
