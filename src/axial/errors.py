"""Exception types. Invariant violations are their own class so callers can
distinguish 'you asked for something the methodology forbids' from a bug."""

from pathlib import Path

from pydantic import ValidationError


class AxialError(Exception):
    """Base class for every error axial raises deliberately."""


class InvariantViolation(AxialError):
    """A refusal. The operation is well-formed but would produce an untrustworthy
    artifact, so axial declines to perform it."""


class MalformedFile(ValueError):
    """A file axial reads holds something it cannot use.

    The message is one line, `<path>: <detail>`. `detail` is the reason alone, for a
    caller that names the file in its own words. A ValueError, so callers that catch
    bad file content as ValueError catch this too.
    """

    def __init__(self, path: Path, detail: str) -> None:
        super().__init__(f"{path}: {detail}")
        self.path = path
        self.detail = detail


def validation_message(exc: ValidationError) -> str:
    """Pydantic's first complaint in one line: the field, what is wrong, the type it got.

    Pydantic's own message runs to four lines, names no file or row and ends in a URL.
    A missing field names no type, because the input pydantic reports for it is the
    whole row. A key the model does not know is called that, the way config.toml
    words it; pydantic's "Extra inputs are not permitted, got str" reads as a wrong type.
    """
    error = exc.errors()[0]
    field = ".".join(str(part) for part in error["loc"])
    if error["type"] == "extra_forbidden":
        return f"{field}: unknown key"
    detail = error["msg"]
    if error["type"] != "missing":
        detail = f"{detail}, got {type(error['input']).__name__}"
    return f"{field}: {detail}" if field else detail
