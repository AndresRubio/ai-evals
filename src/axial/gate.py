"""The CI gate.

Gates on a paired bootstrap confidence interval rather than a point estimate: two
runs of the same system differ by a few points routinely, and a gate set inside
that noise floor fails builds at random until someone disables it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from axial.align import paired_bootstrap_ci
from axial.errors import InvariantViolation
from axial.store import Store


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


def _load(store: Store, mode: str, version: int) -> list[dict[str, Any]]:
    directory = store.judge_dir(mode, version)
    if not (directory / "PROMOTED").exists():
        raise InvariantViolation(
            f"judge {mode} v{version} has not been promoted. Run `axial promote` — a judge "
            "whose agreement with human labels was never measured on test cannot gate a build."
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
