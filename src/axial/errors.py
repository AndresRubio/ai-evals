"""Exception types. Invariant violations are their own class so callers can
distinguish 'you asked for something the methodology forbids' from a bug."""


class AxialError(Exception):
    """Base class for every error axial raises deliberately."""


class InvariantViolation(AxialError):
    """A refusal. The operation is well-formed but would produce an untrustworthy
    artifact, so axial declines to perform it."""
