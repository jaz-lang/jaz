"""``python -m jaz`` -> the interactive console.

Delegates to :func:`jaz.console.main` so ``python -m jaz`` and the installed ``jaz``
console script (``[project.scripts]``) share one implementation.
"""

from jaz.console import main

if __name__ == "__main__":
    raise SystemExit(main())
