"""Built-in hook implementations."""

from .conversation_history import ConversationHistoryHook
from .identify_workflow import IdentifyWorkflowHook, WorkflowStrategyHook
from .loggers import FileLogger, PrintLogger
from .memory_store import MemoryStoreHook
from .multiple_choice_hook import MultipleChoiceHook
from .parent_updates import ParentUpdatesHook
from .workflow_replay import WorkflowReplayHook

# Conditionally import tracing hooks if opentelemetry is installed
try:
    from .jaeger_tracing import JaegerTracingHook  # noqa: F401
    from .langfuse_tracing import LangfuseTracingHook  # noqa: F401

    _TRACING_AVAILABLE = True
except ImportError:
    # OpenTelemetry not installed, tracing hooks unavailable.
    # Do NOT assign None — let the ImportError propagate so that
    # downstream re-exports (jaz.hooks.__init__) detect the absence
    # and users get a clear ImportError instead of a silent None.
    _TRACING_AVAILABLE = False

__all__ = [
    "ConversationHistoryHook",
    "MemoryStoreHook",
    "PrintLogger",
    "FileLogger",
    "WorkflowReplayHook",
    "MultipleChoiceHook",
    "IdentifyWorkflowHook",
    "WorkflowStrategyHook",
    "ParentUpdatesHook",
]

# Only export tracing hooks if available
if _TRACING_AVAILABLE:
    __all__.append("JaegerTracingHook")
    __all__.append("LangfuseTracingHook")
