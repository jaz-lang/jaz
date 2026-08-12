"""Built-in hook implementations."""

from ._must_exit import TemplatedMustExitWarning
from .budget_forcing import BudgetForcing
from .budget_pool import BudgetPool
from .compaction import Compaction
from .context_window import ContextWindow
from .conversation_history import ConversationHistory
from .iteration_limit import IterationLimit
from .loggers import FileLogger, PrintLogger
from .metadata import MetaData
from .recursion_limit import RecursionLimit
from .repl_input_hooks import ValidateREPLInput
from .replay import Replay
from .return_hooks import ReturnType, ValidateReturn
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
    "BudgetForcing",
    "BudgetPool",
    "Compaction",
    "ConversationHistory",
    "ContextWindow",
    "IterationLimit",
    "RecursionLimit",
    "TemplatedMustExitWarning",
    "MetaData",
    "PrintLogger",
    "FileLogger",
    "WorkflowReplay",
    "Replay",
    "ReturnType",
    "ValidateReturn",
    "ValidateREPLInput",
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
ContextWindowHook = ContextWindow
ConversationHistoryHook = ConversationHistory
IterationLimitHook = IterationLimit
ReplayHook = Replay
WorkflowReplayHook = WorkflowReplay

if _TRACING_AVAILABLE:
    JaegerTracingHook = JaegerTracing
    LangfuseTracingHook = LangfuseTracing
