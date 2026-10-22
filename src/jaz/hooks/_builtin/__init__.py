"""Built-in hook implementations."""

from .atif_trace import TrajectoryRecorder
from .budget_forcing import BudgetForcing
from .budget_pool import BudgetPool
from .context_window import ContextWindowWarning
from .iteration_limit import IterationLimit
from .loggers import FileLogger, PrintLogger
from .metadata import MetaData
from .recursion_limit import RecursionLimit
from .repl_code_hooks import ValidateREPLCode
from .replay import TrajectoryReplay
from .return_hooks import ReturnType, ValidateReturn
from .rollout import Rollout, RolloutRecorder, Sample
from .trajectory_directory import TrajectoryDirectoryRecorder

# Conditionally import tracing hooks if opentelemetry is installed
try:
    from .jaeger_tracing import JaegerTracing  # noqa: F401
    from .langfuse_tracing import LangfuseTracing  # noqa: F401
    from .otel_tracing import OTelTracing  # noqa: F401

    _TRACING_AVAILABLE = True
except ImportError:
    # OpenTelemetry not installed, tracing hooks unavailable.
    # Do NOT assign None — let the ImportError propagate so that
    # downstream re-exports (jaz.hooks.__init__) detect the absence
    # and users get a clear ImportError instead of a silent None.
    _TRACING_AVAILABLE = False

__all__ = [
    "TrajectoryRecorder",
    "TrajectoryDirectoryRecorder",
    "TrajectoryReplay",
    "BudgetForcing",
    "BudgetPool",
    "ContextWindowWarning",
    "IterationLimit",
    "RecursionLimit",
    "MetaData",
    "PrintLogger",
    "FileLogger",
    "ReturnType",
    "ValidateReturn",
    "ValidateREPLCode",
    # Rollout recording (token-native backends)
    "RolloutRecorder",
    "Rollout",
    "Sample",
]

# Only export tracing hooks if available
if _TRACING_AVAILABLE:
    # The base class as well as its two presets — a caller exporting to some other OTLP
    # collector subclasses `OTelTracing`, so it is the seam, not an implementation detail.
    __all__.append("OTelTracing")
    __all__.append("JaegerTracing")
    __all__.append("LangfuseTracing")
