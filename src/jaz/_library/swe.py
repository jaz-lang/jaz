import subprocess

from .core import Library

SWE_LIBRARY = Library(
    "SWE tool library",
    "Tool library with tools for software engineering. Contains a bash tool and tools "
    "for reading and writing files.",
    [
        (
            "swe",
            "Module with all SWE tools. Contains `swe.bash`, `swe.read`, and "
            "`swe.replace`.",
        )
    ],
    [],
)


@SWE_LIBRARY.register("swe.bash")
def bash(cmd: str) -> subprocess.CompletedProcess:
    """Execute a bash command and return the completed process.

    All file reads/writes should be done using this tool.
    All interactions with the file system (e.g., creating directories/files)
    should also be done using this tool.

    Args:
        cmd (str): The bash command to execute.
    Returns:
        subprocess.CompletedProcess: The result of the bash command execution.
    """
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    return result


@SWE_LIBRARY.register("swe.read")
def read(file_path: str) -> str:
    """Read the contents of a file.

    Args:
        file_path (str): The path to the file to read.

    Returns:
        str: The contents of the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there is an error reading the file.
    """
    with open(file_path) as f:
        return f.read()


@SWE_LIBRARY.register("swe.replace")
def replace(
    file_path: str, to_replace: str, replace_with: str, replace_all: bool = False
) -> None:
    """Replace text in a file.

    Args:
        file_path (str): The path to the file to modify.
        to_replace (str): The text to replace.
        replace_with (str): The text to replace with.
        replace_all (bool): If True, replace all occurrences. If False, only replace if
        there's exactly one occurrence.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If to_replace is not found in the file.
        ValueError: If multiple instances of to_replace are found and replace_all is
        False.
        IOError: If there is an error reading or writing the file.
    """
    with open(file_path) as f:
        content = f.read()

    count = content.count(to_replace)

    if count == 0:
        raise ValueError(f"String '{to_replace}' not found in {file_path}")

    if count > 1 and not replace_all:
        raise ValueError(
            f"Multiple instances ({count}) of '{to_replace}' found in {file_path}. "
            f"Set replace_all=True to replace all occurrences."
        )

    new_content = content.replace(to_replace, replace_with)

    with open(file_path, "w") as f:
        f.write(new_content)
