"""Skips for a machine a test cannot run on, which CI is never meant to be.

CI runs as an ordinary user under a C locale that decodes as ASCII, so the root and
locale tests run there. A skip on CI would quietly leave them untested; it fails instead.
GitHub Actions sets CI=true.
"""

from __future__ import annotations

import os
from typing import NoReturn

import pytest


def skip_off_ci(reason: str) -> NoReturn:
    __tracebackhide__ = True  # report the skip at the test that asked for it
    if os.environ.get("CI"):
        pytest.fail(f"skipped on CI, where it must run: {reason}")
    pytest.skip(reason)
