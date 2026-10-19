"""The CI gate.

Gates on a confidence interval rather than a point estimate: two runs of the same
system differ by a few points routinely, and a gate set inside that noise floor
fails builds at random until someone disables it.

Two modes. `gate` compares a promoted judge's own verdicts across judge versions.
`gate_batches` gates on batches of new traces that `axial measure` graded with a
promoted judge: on one batch's corrected failure rate against a ceiling, or on the
corrected difference between two batches, paired by input where they share inputs.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from axial.align import (
    corrected_difference_paired,
    corrected_difference_unpaired,
    corrected_rate,
    paired_bootstrap_ci,
)
from axial.errors import AxialError, InvariantViolation
from axial.measure import (
    Measurement,
    judge_test_matrix,
    load_measurement,
    pair_batches,
    summary_lines,
)
from axial.report import SATURATION_WARNING, require_promoted
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
    require_promoted(store, mode, version)
    rows = store.read_jsonl(store.judge_dir(mode, version) / "verdicts.jsonl")
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
        messages.append(SATURATION_WARNING)

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


def _load_fresh(
    store: Store, mode: str, batch: str
) -> tuple[Measurement, list[dict[str, Any]]]:
    """A measurement, refused unless the judge that took it is still promoted unchanged,
    it graded at least one trace, and its numbers are the ones its verdicts give.

    `measurement.json` is a plain file anyone can edit, and its interval decides
    `--max-rate`. So the test matrix is rebuilt from the judge's hash-checked verdicts,
    the rate is recomputed from the batch's own verdict rows with the recorded draws
    and seed, and any difference is refused. The gate then uses the recomputed values,
    which are equal to the stored ones once the check passes.
    """
    measurement, rows = load_measurement(store, mode, batch)
    version = measurement.judge_version
    remeasure = (
        f"`axial measure {mode} --version {version} --traces <file> --name {batch} --replace`"
    )
    try:
        hashes = require_promoted(store, mode, version)
    except InvariantViolation as exc:
        raise InvariantViolation(
            f"batch {batch!r} was measured by {mode} v{version}, which is no longer "
            f"promoted as it was: {exc} Then re-measure the batch with {remeasure}."
        ) from exc
    if (hashes.judge_yaml_sha256, hashes.verdicts_sha256) != (
        measurement.judge_yaml_sha256,
        measurement.verdicts_sha256,
    ):
        raise InvariantViolation(
            f"batch {batch!r} was measured by an earlier {mode} v{version}: its judge.yaml "
            "or verdicts changed and it was promoted again since. Re-measure the batch "
            f"with {remeasure}."
        )
    if measurement.rate is None:
        raise InvariantViolation(
            f"batch {batch!r} graded no traces (all {measurement.n_traces} were set aside). "
            "A gate cannot pass on no evidence; see the batch's errors.jsonl."
        )
    test = judge_test_matrix(store, mode, version)
    flagged = sum(1 for r in rows if r["verdict"])
    mismatch = (
        f"measurement {batch!r} for {mode} does not match its own verdicts: its rate, "
        f"interval or test matrix is not what they give. Re-measure the batch with {remeasure}."
    )
    try:
        rate = corrected_rate(
            test, flagged, len(rows), draws=measurement.draws, seed=measurement.seed
        )
    except ValueError as exc:  # a negative draws or seed
        raise AxialError(mismatch) from exc
    recomputed = dataclasses.replace(measurement, test_matrix=test, rate=rate)
    # Compared as written to disk: a nan point is null there, and a NaN read back into
    # an interval never equals a recomputed bound.
    if recomputed.to_json() != measurement.to_json():
        raise AxialError(mismatch)
    return recomputed, rows


def gate_batches(
    store: Store,
    mode: str,
    batch: str,
    *,
    baseline_batch: str | None = None,
    max_rate: float | None = None,
    draws: int = 20_000,
    seed: int = 0,
) -> GateResult:
    if baseline_batch == batch:
        raise AxialError(
            f"batch {batch!r} is both the candidate and the baseline; a batch compared with "
            "itself can never regress, so name the batch to compare against"
        )
    candidate, candidate_rows = _load_fresh(store, mode, batch)
    assert candidate.rate is not None  # _load_fresh refuses a batch with no rate
    messages = summary_lines(candidate)
    passed = True

    if candidate.flagged == 0:
        messages.append(SATURATION_WARNING)

    if max_rate is not None:
        interval = candidate.rate.interval
        if interval is None:
            passed = False
            messages.append(
                "FAIL: the corrected rate has no interval (the judge is too close to chance "
                "on its test split), so --max-rate cannot be checked"
            )
        elif interval[1] > max_rate:
            passed = False
            messages.append(
                f"ABOVE CEILING: the 95% interval's upper end, {interval[1]:.3f}, is above "
                f"--max-rate {max_rate:g}"
            )
        else:
            messages.append(
                f"within the ceiling: the 95% interval's upper end, {interval[1]:.3f}, is at "
                f"most --max-rate {max_rate:g}"
            )

    if baseline_batch is not None:
        baseline, baseline_rows = _load_fresh(store, mode, baseline_batch)
        assert baseline.rate is not None
        if baseline.judge_version != candidate.judge_version:
            raise InvariantViolation(
                f"batch {baseline_batch!r} was measured by {mode} v{baseline.judge_version} "
                f"and {batch!r} by v{candidate.judge_version}; their difference would mix a "
                "judge change with a system change. Measure both with the same judge."
            )
        messages += [f"baseline {line.strip()}" for line in summary_lines(baseline)]
        pairing = pair_batches(baseline_rows, candidate_rows)
        messages.append(pairing.describe())
        if pairing.paired:
            difference = corrected_difference_paired(
                candidate.test_matrix, pairing.cells, draws=draws, seed=seed
            )
        else:
            difference = corrected_difference_unpaired(
                candidate.test_matrix,
                (baseline.flagged, baseline.n_graded),
                (candidate.flagged, candidate.n_graded),
                draws=draws,
                seed=seed,
            )
        if difference.interval is None:
            passed = False
            messages.append(
                f"FAIL: the difference vs {baseline_batch} has no interval (the judge is too "
                "close to chance on its test split), so no regression check is possible"
            )
        else:
            low, high = difference.interval
            messages.append(
                f"difference vs {baseline_batch}: {difference.point:+.3f} corrected failure "
                f"rate (95% CI [{low:+.3f}, {high:+.3f}])"
            )
            if low > 0:
                passed = False
                messages.append(
                    "REGRESSION: the failure rate rose and the confidence interval excludes zero"
                )
            elif high < 0:
                messages.append(
                    "improvement: the failure rate fell and the interval excludes zero"
                )
            else:
                messages.append("no change detectable above the noise floor")
        clipped = {candidate.rate.clipped, baseline.rate.clipped} - {None}
        if clipped == {"low", "high"}:
            messages.append(
                "note: one rate above is clipped to 0 and the other to 1 for display; the "
                "difference was computed before clipping, so it can differ from the gap "
                "between the rates shown"
            )
        elif clipped:
            bound = 0 if clipped == {"low"} else 1
            messages.append(
                f"note: a rate above is clipped to {bound} for display; the difference was "
                f"computed before clipping, so it can be non-zero while both rates show {bound:.3f}"
            )

    return GateResult(passed=passed, messages=messages)
