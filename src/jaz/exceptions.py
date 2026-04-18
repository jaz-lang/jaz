class JazException(Exception):
    """Base exception class for Jaz-related errors."""


class LLMOutputParseError(JazException):
    """Exception raised when parsing LLM output fails."""


class CommandSyntaxError(JazException):
    """Exception raised for syntax errors in REPL commands."""


class CommandTypeError(JazException):
    """Exception raised for type errors in REPL commands."""


class JazPermissionError(JazException):
    """Exception raised for permission errors in Jaz."""


class BudgetExhaustedError(JazException):
    """Exception raised when a budget is exhausted in Jaz."""


class _JazInternalError(JazException):
    """Exception raised for internal errors in Jaz."""


class BashExitException(JazException):
    """Exception raised when RAISE command is used in Bash REPL."""


class BudgetForcingException(JazException):
    """Exception raised when budget forcing blocks an early RETURN/RAISE."""


class DuplicateReplInputError(JazException):
    """Exception raised when attempting to add a REPL input that already exists."""
