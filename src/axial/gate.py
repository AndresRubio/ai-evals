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
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from axial.align import (
    corrected_difference_paired,
    corrected_difference_unpaired,
    corrected_rate,
    paired_bootstrap_ci,
)
from axial.errors import AxialError, InvariantViolation
from axial.judge import read_verdicts
from axial.measure import (
    SATURATION_WARNING,
    Measurement,
    judge_test_matrix,
    load_measurement,
    pair_batches,
    set_aside_note,
    summary_lines,
)
from axial.promotion import require_promoted
from axial.store import Store

# How far a stored interval bound may sit from the recomputed one. The bounds are
# percentiles of Beta draws, and numpy does not promise the same stream across versions,
# so an upgrade between `measure` and `gate` can move them in the third decimal. Anything
# else in a measurement is deterministic and compared exactly.
INTERVAL_TOLERANCE = 0.01
_INTERVAL_KEYS = ("interval", "interval_if_judge_exact")

# The draws a measurement may record for the gate to recompute it with. A handful gives
# percentiles that are noise (one draw gives an interval of zero width), and a huge count
# exhausts memory. `measure` uses 20,000.
MIN_DRAWS = 1_000
MAX_DRAWS = 1_000_000


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
    rows = read_verdicts(store, store.judge_dir(mode, version) / "verdicts.jsonl")
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
        # Paired by trace id, in the candidate's order. Row order in verdicts.jsonl says
        # nothing about which case is which: re-running a split appends it after the
        # others. An id repeated on either side cannot be paired, so it is refused.
        for checked_version, checked_rows in ((baseline_version, base_rows), (version, rows)):
            repeated = _repeated_ids(checked_rows)
            if repeated:
                shown = ", ".join(repr(trace_id) for trace_id in repeated[:5])
                more = f" and {len(repeated) - 5} more" if len(repeated) > 5 else ""
                raise InvariantViolation(
                    f"{mode} v{checked_version} has more than one verdict row for "
                    f"{shown}{more}; the gate pairs cases by trace id, so each must appear once"
                )
        ids = [r["trace_id"] for r in rows]
        if sorted(ids) != sorted(r["trace_id"] for r in base_rows):
            raise InvariantViolation(
                "baseline and candidate must cover the same cases; "
                "comparing different case sets measures the case set, not the change"
            )
        base_by_id = {r["trace_id"]: r for r in base_rows}
        baseline = [bool(base_by_id[trace_id]["verdict"]) for trace_id in ids]
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


def _repeated_ids(rows: list[dict[str, Any]]) -> list[str]:
    """The trace ids that more than one row carries, sorted."""
    counts = Counter(r["trace_id"] for r in rows)
    return sorted(trace_id for trace_id, n in counts.items() if n > 1)


def _bounds_agree(stored: list[float] | None, recomputed: list[float] | None) -> bool:
    """Both absent, or both present with each bound within the tolerance. A NaN bound
    never is: `abs(nan - x) <= tolerance` is False."""
    if stored is None or recomputed is None:
        return stored is None and recomputed is None
    return all(abs(a - b) <= INTERVAL_TOLERANCE for a, b in zip(stored, recomputed, strict=True))


def _same_numbers(stored: Measurement, recomputed: Measurement) -> bool:
    """Whether a stored measurement is what its verdicts give, compared as written to disk.

    Every field is exact except the two intervals' bounds, which have a tolerance. A nan
    point is null on both sides, and a NaN read back into a bound is never close to a
    recomputed one.
    """
    a, b = stored.to_json(), recomputed.to_json()
    bounds = all(_bounds_agree(a.pop(key), b.pop(key)) for key in _INTERVAL_KEYS)
    return bounds and a == b


def _load_fresh(
    store: Store, mode: str, batch: str
) -> tuple[Measurement, list[dict[str, Any]]]:
    """A measurement, refused unless the judge that took it is still promoted unchanged,
    it graded at least one trace, and its numbers are the ones its verdicts give.

    `measurement.json` is a plain file anyone can edit, and its interval decides
    `--max-rate`. So the test matrix is rebuilt from the judge's hash-checked verdicts,
    the rate is recomputed from the batch's own verdict rows with the recorded draws
    and seed, and any difference is refused. The gate then uses the recomputed values:
    an interval bound may differ from the stored one by `INTERVAL_TOLERANCE`, and an edit
    inside that tolerance cannot change a decision because the stored bound is never read.
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
    if (hashes.judge_yaml_sha256, hashes.verdicts_sha256, hashes.prompt_sha256) != (
        measurement.judge_yaml_sha256,
        measurement.verdicts_sha256,
        measurement.prompt_sha256,
    ):
        raise InvariantViolation(
            f"batch {batch!r} was measured by an earlier {mode} v{version}: its judge.yaml, "
            "verdicts or prompt changed and it was promoted again since. Re-measure the "
            f"batch with {remeasure}."
        )
    if measurement.rate is None:
        raise InvariantViolation(
            f"batch {batch!r} graded no traces (all {measurement.n_traces} were set aside). "
            "A gate cannot pass on no evidence; see the batch's errors.jsonl."
        )
    if not MIN_DRAWS <= measurement.draws <= MAX_DRAWS:
        raise AxialError(
            f"measurement {batch!r} for {mode} records {measurement.draws:,} Monte Carlo "
            f"draws; the gate recomputes its interval with them and accepts {MIN_DRAWS:,} "
            f"to {MAX_DRAWS:,}. Re-measure the batch with {remeasure}."
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
    except ValueError as exc:  # a negative seed
        raise AxialError(mismatch) from exc
    recomputed = dataclasses.replace(measurement, test_matrix=test, rate=rate)
    if not _same_numbers(measurement, recomputed):
        raise AxialError(mismatch)
    # The recomputed measurement, not the stored one: the gate decides on these bounds.
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
    # As the CLI checks it. `upper > nan` is False, so a NaN ceiling would pass any batch.
    if max_rate is not None and not 0 < max_rate < 1:
        raise AxialError(f"max_rate must lie strictly between 0 and 1, got {max_rate}")
    if baseline_batch == batch:
        raise AxialError(
            f"batch {batch!r} is both the candidate and the baseline; a batch compared with "
            "itself can never regress, so name the batch to compare against"
        )
    candidate, candidate_rows = _load_fresh(store, mode, batch)
    assert candidate.rate is not None  # _load_fresh refuses a batch with no rate
    messages = summary_lines(candidate)
    # A warning only: set-aside traces never change the decision below. They can hide
    # a regression (the pairs shrink to the traces that were graded), so the reader sees
    # the note `measure` printed.
    candidate_note = set_aside_note(candidate.n_set_aside, candidate.n_traces)
    if candidate_note:
        messages.append(f"note: {candidate_note}")
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
        baseline_note = set_aside_note(baseline.n_set_aside, baseline.n_traces)
        if baseline_note:
            messages.append(f"baseline note: {baseline_note}")
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
