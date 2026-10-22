"""``python -m jaz`` -> the interactive console.

Delegates to :func:`jaz._console.main` so ``python -m jaz`` and the installed ``jaz``
console script (``[project.scripts]``) share one implementation.
"""

from jaz._console import main

if __name__ == "__main__":
    raise SystemExit(main())
