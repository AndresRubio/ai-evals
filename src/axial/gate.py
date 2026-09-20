"""The CI gate.

Gates on a paired bootstrap confidence interval rather than a point estimate: two
runs of the same system differ by a few points routinely, and a gate set inside
that noise floor fails builds at random until someone disables it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from axial.align import paired_bootstrap_ci
from axial.errors import InvariantViolation
from axial.report import JUDGE_YAML_HASH_KEY, VERDICTS_HASH_KEY, parse_promoted_marker
from axial.store import Store

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass
class GateResult:
    passed: bool
    messages: list[str] = field(default_factory=list)


def compare(
    baseline: list[bool],
    candidate: list[bool],
    *,
    seed: int = 0,
) -> tuple[float, tuple[float, float]]:
    delta = sum(candidate) / len(candidate) - sum(baseline) / len(baseline)
    return delta, paired_bootstrap_ci(baseline, candidate, seed=seed)


def _require_bound_hash(
    recorded: dict[str, str], key: str, mode: str, version: int
) -> str:
    """Pull `key` out of a parsed `PROMOTED` marker and demand it look like a
    real sha256 hex digest.

    A marker with no hash lines at all -- or one with a value that isn't a
    64-character hex digest -- carries no binding to check against, which is
    exactly the hole an earlier version of `gate` left open: it only compared
    the hashes `if recorded_... is not None`, so a marker with the two hash
    lines stripped (or one written before this feature existed) skipped both
    checks and gated on an unverified judge.yaml/verdicts.jsonl. Missing or
    unparseable is now a violation, not a skip, because the actor who can
    edit judge.yaml can just as easily edit the marker beside it.
    """
    value = recorded.get(key)
    if value is None or not _SHA256_HEX_RE.fullmatch(value):
        raise InvariantViolation(
            f"judge {mode} v{version}'s PROMOTED marker is missing or has an unparseable "
            f"'{key}' value. The marker predates hash binding or was tampered with, and "
            "gate cannot verify it against an unbound or malformed hash. Re-run `axial "
            "promote` on the current judge, or `axial judge --new` to score it as a fresh "
            "version."
        )
    return value


def _load(store: Store, mode: str, version: int) -> list[dict[str, Any]]:
    directory = store.judge_dir(mode, version)
    marker_path = directory / "PROMOTED"
    if not marker_path.exists():
        raise InvariantViolation(
            f"judge {mode} v{version} has not been promoted. Run `axial promote` — a judge "
            "whose agreement with human labels was never measured on test cannot gate a build."
        )

    # The marker on its own only proves *some* run was promoted once. Recompute
    # both hashes it recorded and refuse if either has drifted: a hand-edited
    # judge.yaml or a verdicts.jsonl overwritten by a later `run_judge` means the
    # evidence behind PROMOTED no longer describes what's on disk, and this is
    # the only thing standing between that and a silently unmeasured judge
    # gating a build.
    recorded = parse_promoted_marker(marker_path.read_text())
    recorded_judge_hash = _require_bound_hash(recorded, JUDGE_YAML_HASH_KEY, mode, version)
    current_judge_hash = store.hash_file(directory / "judge.yaml")
    if current_judge_hash != recorded_judge_hash:
        raise InvariantViolation(
            f"judge {mode} v{version}'s judge.yaml has changed since it was promoted — "
            "the judge that earned PROMOTED is not the judge on disk. Re-run `axial "
            "promote` on the current judge, or run `axial judge --new` to score the "
            "changed prompt as a fresh version."
        )
    recorded_verdicts_hash = _require_bound_hash(recorded, VERDICTS_HASH_KEY, mode, version)
    current_verdicts_hash = store.hash_file(directory / "verdicts.jsonl")
    if current_verdicts_hash != recorded_verdicts_hash:
        raise InvariantViolation(
            f"judge {mode} v{version}'s verdicts.jsonl has changed since it was promoted — "
            "the verdicts that earned PROMOTED are not the verdicts on disk. Re-run `axial "
            "promote` on the current run, or run `axial judge --new` to score it as a "
            "fresh version."
        )

    rows = store.read_jsonl(directory / "verdicts.jsonl")
    if not rows:
        raise InvariantViolation(
            f"judge {mode} v{version} has no verdict rows. A gate cannot pass on no evidence. "
            "Check errors.jsonl for quarantined calls."
        )
    return rows


def gate(
    store: Store,
    mode: str,
    version: int,
    *,
    baseline_version: int | None = None,
    seed: int = 0,
) -> GateResult:
    rows = _load(store, mode, version)
    candidate = [bool(r["verdict"]) for r in rows]
    messages: list[str] = []
    passed = True

    failure_rate = sum(candidate) / len(candidate) if candidate else 0.0
    messages.append(f"{mode} v{version}: failure rate {failure_rate:.3f} over {len(candidate)} traces")

    if candidate and failure_rate == 0.0:
        messages.append(
            "saturation warning: no trace failed. A suite nothing fails has stopped carrying "
            "signal while still costing maintenance — consider harder cases or retiring it."
        )

    if baseline_version is not None:
        base_rows = _load(store, mode, baseline_version)
        if [r["trace_id"] for r in base_rows] != [r["trace_id"] for r in rows]:
            raise InvariantViolation(
                "baseline and candidate must cover the same cases in the same order; "
                "comparing different case sets measures the case set, not the change"
            )
        baseline = [bool(r["verdict"]) for r in base_rows]
        delta, (low, high) = compare(baseline, candidate, seed=seed)
        messages.append(
            f"delta vs v{baseline_version}: {delta:+.3f} failure rate "
            f"(95% CI [{low:+.3f}, {high:+.3f}])"
        )
        if low > 0:
            passed = False
            messages.append(
                "REGRESSION: the failure rate rose and the confidence interval excludes zero"
            )
        elif high < 0:
            messages.append("improvement: the failure rate fell and the interval excludes zero")
        else:
            messages.append("no change detectable above the noise floor")

    return GateResult(passed=passed, messages=messages)
