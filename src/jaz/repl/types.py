from typing import NamedTuple


class ExecuteResult(NamedTuple):
    """Result from code execution that completed without returning a value.

    Attributes:
        output: Captured stdout output from successful execution
    """

    output: str


class ErrorResult(NamedTuple):
    """Result from code execution that encountered an error.

    Attributes:
        output: Formatted output including stdout and traceback
        error_summary: Short description of the error for the LLM
        exception: The exception that caused the error (optional for bash environment errors)
    """

    output: str
    error_summary: str
    exception: BaseException | None = None


class RaiseResult(NamedTuple):
    """Result from code execution that raised an exception.

    Attributes:
        output: Captured stdout/stderr output before exception
        exception: The exception object that was raised
    """

    output: str
    exception: BaseException


class ReturnResult[ReturnT](NamedTuple):
    """Result from code execution that returned a value.

    Attributes:
        output: Captured stdout output from successful execution
        return_value: The value returned by the code
    """

    output: str
    return_value: ReturnT


type ExecResult[ReturnT] = (
    ExecuteResult | ErrorResult | RaiseResult | ReturnResult[ReturnT]
)
