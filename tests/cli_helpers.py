"""What a CLI command printed, without the terminal styling.

Typer forces Rich's terminal output when GITHUB_ACTIONS, FORCE_COLOR or PY_COLORS is
set, so on GitHub Actions a usage error puts escape codes between the two dashes of
`--version`. Tests match fragments against the plain text, so they pass the same way on
CI and on a laptop.
"""

from __future__ import annotations

import re

from typer.testing import Result

_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def plain_output(result: Result) -> str:
    return _ESCAPE.sub("", result.output)
