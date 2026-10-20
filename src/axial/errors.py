"""Exception types. Invariant violations are their own class so callers can
distinguish 'you asked for something the methodology forbids' from a bug."""

from pydantic import ValidationError


class AxialError(Exception):
    """Base class for every error axial raises deliberately."""


class InvariantViolation(AxialError):
    """A refusal. The operation is well-formed but would produce an untrustworthy
    artifact, so axial declines to perform it."""


def validation_message(exc: ValidationError) -> str:
    """Pydantic's first complaint in one line: the field, what is wrong, the type it got.

    Pydantic's own message runs to four lines, names no file or row and ends in a URL.
    A missing field names no type, because the input pydantic reports for it is the
    whole row.
    """
    error = exc.errors()[0]
    field = ".".join(str(part) for part in error["loc"])
    detail = error["msg"]
    if error["type"] != "missing":
        detail = f"{detail}, got {type(error['input']).__name__}"
    return f"{field}: {detail}" if field else detail
