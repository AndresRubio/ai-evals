# Corrected Rates and Dev Disagreements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A promoted judge grades new traces (`axial measure`) and reports a failure rate corrected for its own errors; `axial gate --batch` gates on that rate or on the corrected difference between two measured batches; `alignment.md` lists every dev case where judge and human disagree.

**Architecture:** The estimators live in `align.py`: Rogan-Gladen, with a Monte Carlo interval over Jeffreys posteriors. A new `measure.py` grades a batch through `grade_trace`, which is extracted from `run_judge` so both paths grade alike, and writes `.axial/measurements/<mode>/<batch>/` in one atomic step. `gate.py` gains `gate_batches`. It refuses a measurement whose judge has changed since, through `report.require_promoted`, and pairs two batches by input hash. The disagreement listing is pure reporting: it reads the judge's stored raw answers and leaves `verdicts.jsonl` alone.

**Tech Stack:** Python ≥ 3.13, numpy, pydantic, typer, pytest, ruff, mypy `--strict`, uv.

**Spec:** [`docs/superpowers/specs/2026-10-07-corrected-rates-and-disagreements-design.md`](../specs/2026-10-07-corrected-rates-and-disagreements-design.md). Where this plan and the spec differ, the plan wins only on the deltas listed below.

## Global Constraints

- The failure is the positive class: `True` means the failure mode is present. TPR is the share of real failures the judge flags; TNR is the share of real passes it leaves alone.
- TPR and TNR come from the promoted judge's **test** split only.
- Interval method:
  - 20,000 draws from `numpy.random.default_rng(seed)`, with `seed = 0` by default and recorded.
  - Each proportion is drawn from its Jeffreys posterior, `Beta(k + ½, n − k + ½)`.
  - A draw with `TPR* + TNR* − 1 ≤ 0.05` is invalid. An interval is reported only when at least 90% of draws are valid, otherwise it is `None`.
- Differences:
  - computed before clipping;
  - one judge draw per draw, shared by both batches;
  - paired cells drawn from `Dirichlet(n11 + ½, n10 + ½, n01 + ½, n00 + ½)`.
- Pairing:
  - the key is `input_sha256`, the SHA-256 of the input text in UTF-8;
  - batches are paired when the pairs cover at least 80% of **each** batch's graded traces;
  - an input graded twice in one batch never pairs.
- Thresholds:
  - a measurement carries the set-aside note when more than 5% of the batch was set aside;
  - disagreement excerpts are the first 300 characters.
- Error handling:
  - every refusal raises `AxialError` or a subclass, and the CLI prints one line and exits non-zero, never with a traceback;
  - mistakes in `gate`'s options are usage errors with exit code 2.
- Tooling:
  - no new dependencies;
  - CI (`.github/workflows/ci.yml`) runs `ruff check .`, `mypy src/axial` (strict), pytest on Python 3.14 and 3.13, and `scripts/knowledge_index.py --check`.
- Style: match the surrounding code. Docstrings and comments say why, not what. Commit subjects are sentence-case imperatives with no prefix, like the existing history.
- Git:
  - commit as AndresRubio with `-c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com`, and end every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`;
  - work in a separate worktree on branch `corrected-rates`, because the main checkout is shared with another session and stays on `main`; nothing is pushed.

## Spec deltas

Decided while writing this plan. Each is small, and each is pinned by a test:

1. **Batch names must start with a letter or digit:** `[A-Za-z0-9][A-Za-z0-9._-]*`. The spec's `[A-Za-z0-9._-]+` admits `.` and `..`, and hidden names would collide with the staging directories of the atomic write.
2. **`measurement.json` gains a `clipped` field** (`"low"`, `"high"` or `null`). `gate` needs it to print the note that the difference was computed before clipping.
3. **A judge whose test-split `TPR + TNR − 1` is at most 0.05 gets no point estimate.** `corrected_rate` returns `point = nan`, which is written as `null`, and a note says why. Promotion's default thresholds make this unreachable, but they are configurable.
4. **`gate --batch` with a baseline also prints the baseline's summary line,** so a clipped-rate note always refers to a rate on screen.
5. **`load_measurement` refuses a measurement whose `verdicts.jsonl` disagrees with its `measurement.json`.** That means a different row count or flag count, or a mode or batch that doesn't match the directory. Pairing reads the rows and the single rate reads the JSON, so a hand edit could otherwise make them disagree silently.
6. **`find_disagreements(..., split="test")` raises `InvariantViolation`.** The spec says test disagreements are never listed; this makes it impossible rather than merely conventional.
7. **The listing's guidance names `--new`:** `` `axial judge <mode> --split dev` again (with `--new` once this version has been scored on test) ``. This matches the existing `--new` help text.
8. **Re-judging to byte-identical verdicts and re-promoting keeps a measurement valid.** The marker binds bytes, so it is the same evidence. This is pinned by a test so nobody "fixes" it.
9. **Worked numbers:** tests assert the spec's interval endpoints to within ±0.002. The third decimal moves with draw order: this implementation gives 0.029 for the spec's 0.028 upper bound on 20/500, 0.098 for 0.099, and similar.

## Status of the code in this plan

Every code block below was run on 2026-10-14 in a scratch worktree off `3673962`. Results:
- the full suite: 817 passed, 48 xfailed, on Python 3.14 and 3.13;
- `ruff check .`, `mypy src/axial` and `scripts/knowledge_index.py --check`: all clean;
- the simulations: coverage 0.931–0.955; for the paired gate, false alarms 2.7–2.8%, coverage of a four-point rise 0.948–0.955, and power 0.835.

Copy the blocks as written. If a test fails, find out why before changing it. In particular, if a simulation band fails, the interval method goes back for review; the band does not move.

## Running things

- Set up a worktree for the branch, not the shared checkout: `git worktree add ../evals-corrected-rates corrected-rates`, then run `uv sync` in it.
- Commands below run from the worktree root.
- Tests import the shared helper module `tests/measure_helpers.py` as `from measure_helpers import ...`. This works because pytest's default `prepend` import mode puts `tests/` on `sys.path`. Do not add `tests/__init__.py`.
- Ruff sorts `measure_helpers` with the third-party imports. The blocks below are already in that order.

## File map

| File | Change | Responsibility |
|---|---|---|
| `src/axial/align.py` | modify | `CorrectedRate`, `corrected_rate`, `CorrectedDifference`, `corrected_difference_unpaired`, `corrected_difference_paired` |
| `src/axial/report.py` | modify | `require_promoted` and `SATURATION_WARNING` move here from `gate.py`; the disagreement listing; the `write_alignment(..., disagreements=)` parameter |
| `src/axial/judge.py` | modify | `resolve_few_shot`, `grade_trace`, `Graded` and `SetAside` extracted from `run_judge`, whose behaviour is unchanged |
| `src/axial/store.py` | modify | `jsonl_text`, `json_text`, `measurement_dir`, `write_dir_atomically` |
| `src/axial/measure.py` | create | `measure_batch`, `Measurement`, `load_measurement`, `summary_lines`, `pair_batches`, `Pairing` |
| `src/axial/gate.py` | modify | `_load` uses `require_promoted`; `gate_batches` |
| `src/axial/cli.py` | modify | the `measure` command; `gate` options; `align` prints the disagreement counts |
| `tests/measure_helpers.py` | create | a promoted judge with test matrix tp 28, fn 2, tn 66, fp 4, plus batch builders |
| `tests/test_corrected_rates.py`, `tests/test_corrected_differences.py`, `tests/test_corrected_rates_simulation.py` | create | estimator unit tests and the two simulations |
| `tests/test_measure.py`, `tests/test_pairing.py`, `tests/test_gate_batches.py`, `tests/test_cli_measure.py`, `tests/test_disagreements.py` | create | one file per feature |
| `tests/test_report.py`, `tests/test_judge.py`, `tests/test_store.py` | modify | tests for the extracted and added pieces |
| `tests/test_capabilities_cli_tour.py`, `tests/test_capabilities_cli_failures.py` | modify | the two changes the spec makes on purpose, plus the README command count |
| `README.md`, `knowledge/procedures/evals-in-ci.md`, `knowledge/procedures/production-monitoring.md`, `llms.txt` | modify | documentation |

## Suggested subagent models

Per `~/.claude/CLAUDE.md`:
- **opus:** Tasks 1, 2 and 13 (numeric code and the adversarial review).
- **sonnet:** all the others, whose code is given in full.

---

### Task 1: Corrected failure rate for one batch

**Files:**
- Modify: `src/axial/align.py` (imports, a constant below `_MIN_SURVIVING_FRACTION`, and new code appended at the end)
- Create: `tests/test_corrected_rates.py`
- Create: `tests/test_corrected_rates_simulation.py`

**Interfaces:**
- Consumes: `ConfusionMatrix(tp, fp, tn, fn)` and `_MIN_SURVIVING_FRACTION = 0.9`, both already in `align.py`.
- Produces:
  - `CorrectedRate(observed: float, point: float, interval: tuple[float, float] | None, interval_if_judge_exact: tuple[float, float] | None, clipped: Literal["low", "high"] | None, draws: int, seed: int)`, frozen;
  - `corrected_rate(test: ConfusionMatrix, flagged: int, n: int, *, draws: int = 20_000, seed: int = 0) -> CorrectedRate`, which raises `ValueError` on `n <= 0`, on `flagged` outside `[0, n]`, and on a test matrix with no real failures or no real passes;
  - private helpers reused by Task 2: `_judge_rates`, `_check_count`, `_jeffreys`, `_youden_draws`, `_percentile_interval` and `_MIN_YOUDEN_J`.

- [ ] **Step 1: Write the failing unit tests**

Create `tests/test_corrected_rates.py`:

```python
"""The corrected failure rate of one batch.

Every expected number here was derived by hand from the counts, or is the design's
worked number (docs/superpowers/specs/2026-10-07-corrected-rates-and-disagreements-design.md,
section 2) to within Monte Carlo error. None is the implementation's own output pasted back.
"""

import math

import pytest

from axial.align import ConfusionMatrix, corrected_rate

# The design's worked judge: TPR 28/30 = 0.9333, TNR 66/70 = 0.9429, J = 0.8762.
JUDGE = ConfusionMatrix(tp=28, fn=2, tn=66, fp=4)
PERFECT = ConfusionMatrix(tp=30, fn=0, tn=70, fp=0)
CHANCE = ConfusionMatrix(tp=15, fn=15, tn=35, fp=35)
# Monte Carlo percentiles over 20,000 draws move in the third decimal with draw order.
MC = 0.002


def width(interval: tuple[float, float] | None) -> float:
    assert interval is not None
    return interval[1] - interval[0]


# --- single rate ----------------------------------------------------------------


def test_the_course_example_converted_to_the_failure_side():
    # TPR 0.88, TNR 0.92, observed 0.20: (0.20 + 0.92 - 1) / (0.88 + 0.92 - 1) = 0.12 / 0.80.
    result = corrected_rate(ConfusionMatrix(tp=88, fn=12, tn=92, fp=8), 20, 100)
    assert result.point == pytest.approx(0.150)
    assert result.observed == pytest.approx(0.20)
    assert result.clipped is None


def test_the_worked_example_point_and_intervals():
    # (0.12 + 0.942857 - 1) / 0.876190 = 0.071739
    result = corrected_rate(JUDGE, 60, 500)
    assert result.point == pytest.approx(0.071739, abs=1e-6)
    assert result.interval == pytest.approx((0.000, 0.128), abs=MC)
    assert result.interval_if_judge_exact == pytest.approx((0.042, 0.107), abs=MC)
    assert (result.draws, result.seed) == (20_000, 0)


def test_a_perfect_judge_returns_the_observed_rate():
    assert corrected_rate(PERFECT, 60, 500).point == pytest.approx(0.12)


def test_an_observed_rate_below_the_false_alarm_rate_clips_to_zero():
    # 20/500 = 0.04 is below 1 - TNR = 0.057.
    result = corrected_rate(JUDGE, 20, 500)
    assert result.point == 0.0
    assert result.clipped == "low"
    assert result.interval == pytest.approx((0.000, 0.028), abs=MC)


def test_an_observed_rate_above_tpr_clips_to_one():
    # 490/500 = 0.98 is above TPR = 0.933.
    result = corrected_rate(JUDGE, 490, 500)
    assert result.point == 1.0
    assert result.clipped == "high"


def test_the_interval_contains_the_point_and_is_reproducible():
    first = corrected_rate(JUDGE, 60, 500, seed=7)
    again = corrected_rate(JUDGE, 60, 500, seed=7)
    assert first == again
    assert first.interval is not None
    assert first.interval[0] <= first.point <= first.interval[1]


def test_the_interval_narrows_as_the_test_split_grows():
    tenfold = ConfusionMatrix(tp=280, fn=20, tn=660, fp=40)
    assert width(corrected_rate(tenfold, 60, 500).interval) < width(
        corrected_rate(JUDGE, 60, 500).interval
    )


def test_the_interval_narrows_as_the_batch_grows():
    assert width(corrected_rate(JUDGE, 600, 5000).interval) < width(
        corrected_rate(JUDGE, 60, 500).interval
    )


def test_a_perfect_test_score_still_leaves_the_judge_uncertain():
    # A bootstrap would resample 30/30 to TPR = 1 every time; Jeffreys draws stay below 1.
    result = corrected_rate(PERFECT, 60, 500)
    assert result.interval is not None and result.interval_if_judge_exact is not None
    assert result.interval[0] < result.interval_if_judge_exact[0]
    assert result.interval == pytest.approx((0.080, 0.151), abs=MC)


def test_a_judge_near_chance_gets_no_interval_and_no_point():
    result = corrected_rate(CHANCE, 60, 500)
    assert result.interval is None
    assert result.interval_if_judge_exact is None
    assert math.isnan(result.point)


@pytest.mark.parametrize(
    ("test", "flagged", "n", "message"),
    [
        (JUDGE, 0, 0, "0 graded traces"),
        (JUDGE, 6, 5, "outside"),
        (ConfusionMatrix(tp=0, fn=0, tn=66, fp=4), 1, 10, "no real failures"),
        (ConfusionMatrix(tp=28, fn=2, tn=0, fp=0), 1, 10, "no real passes"),
    ],
)
def test_corrected_rate_refuses_what_it_cannot_correct(test, flagged, n, message):
    with pytest.raises(ValueError, match=message):
        corrected_rate(test, flagged, n)
```

- [ ] **Step 2: Write the coverage simulation**

Create `tests/test_corrected_rates_simulation.py`:

```python
"""Does the 95% interval cover the truth 95% of the time, and does the gate catch a rise?

Each trial draws a test split from a judge with known error rates and a batch from a
known true failure rate, then asks whether the interval covered the truth. The bands
come from the design (section 5). If one of these fails, the interval method goes back
for review; the band does not move.
"""

import numpy as np
import pytest

from axial.align import ConfusionMatrix, corrected_rate

# (TPR, TNR, real failures on test, real passes on test)
STRONG = (0.93, 0.94, 30, 70)
BORDERLINE = (0.90, 0.90, 30, 30)


def draw_test_split(
    rng: np.random.Generator, judge: tuple[float, float, int, int]
) -> ConfusionMatrix:
    tpr, tnr, positives, negatives = judge
    tp = int(rng.binomial(positives, tpr))
    tn = int(rng.binomial(negatives, tnr))
    return ConfusionMatrix(tp=tp, fn=positives - tp, tn=tn, fp=negatives - tn)


@pytest.mark.parametrize("judge", [STRONG, BORDERLINE], ids=["strong", "borderline"])
@pytest.mark.parametrize("true_rate", [0.10, 0.01])
def test_the_single_rate_interval_covers_the_true_rate(judge, true_rate):
    tpr, tnr, _, _ = judge
    rng = np.random.default_rng(1)
    flag_probability = tpr * true_rate + (1 - tnr) * (1 - true_rate)
    trials, covered = 1000, 0
    for _ in range(trials):
        test = draw_test_split(rng, judge)
        flagged = int(rng.binomial(500, flag_probability))
        result = corrected_rate(test, flagged, 500, draws=4000, seed=int(rng.integers(2**31)))
        if result.interval is not None and result.interval[0] <= true_rate <= result.interval[1]:
            covered += 1
    assert 0.92 <= covered / trials <= 0.99
```

- [ ] **Step 3: Run both to verify they fail**

Run: `uv run pytest tests/test_corrected_rates.py tests/test_corrected_rates_simulation.py -q`
Expected: collection errors, `ImportError: cannot import name 'corrected_rate' from 'axial.align'`.

- [ ] **Step 4: Implement**

In `src/axial/align.py`, replace the imports

```python
from dataclasses import dataclass, field

import numpy as np
```

with

```python
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
```

Directly below `_MIN_SURVIVING_FRACTION = 0.9`, add:

```python

# A judge whose TPR + TNR - 1 (Youden's J) is at or below this is no better than chance
# for correcting a rate: dividing by a J near zero sends the corrected rate anywhere.
# Monte Carlo draws below it are invalid, and a point estimate below it is undefined.
_MIN_YOUDEN_J = 0.05
```

Append to the end of the file:

```python
@dataclass(frozen=True)
class CorrectedRate:
    """A batch's flag rate corrected for the judge's own errors (Rogan-Gladen).

    `point` is clipped to [0, 1] and `clipped` says which way, if at all; it is nan
    when the judge's test-split Youden's J is at most `_MIN_YOUDEN_J`. `interval` lets
    the judge's error rates vary with their test-split counts; `interval_if_judge_exact`
    holds them at their point estimates, so comparing the two shows whether more test
    labels or a bigger batch would tighten the result. Either is `None` when it cannot
    be computed honestly.
    """

    observed: float
    point: float
    interval: tuple[float, float] | None
    interval_if_judge_exact: tuple[float, float] | None
    clipped: Literal["low", "high"] | None
    draws: int
    seed: int


def _judge_rates(test: ConfusionMatrix) -> tuple[float, float]:
    positives = test.tp + test.fn
    negatives = test.tn + test.fp
    if positives == 0 or negatives == 0:
        missing = "real failures" if positives == 0 else "real passes"
        raise ValueError(
            f"the test matrix has no {missing}, so the judge's error rates are undefined"
        )
    return test.tp / positives, test.tn / negatives


def _check_count(flagged: int, n: int, what: str) -> None:
    if n <= 0:
        raise ValueError(f"{what}: cannot correct a rate over {n} graded traces")
    if not 0 <= flagged <= n:
        raise ValueError(f"{what}: flagged={flagged} is outside [0, {n}]")


def _jeffreys(rng: np.random.Generator, k: int, n: int, size: int) -> NDArray[np.float64]:
    """Draws from the Jeffreys posterior Beta(k + 1/2, n - k + 1/2) of a proportion.

    Unlike resampling, this keeps a 30-of-30 count uncertain: its draws sit below 1.
    """
    return rng.beta(k + 0.5, n - k + 0.5, size)


def _youden_draws(
    rng: np.random.Generator, test: ConfusionMatrix, draws: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """TNR* and J* = TPR* + TNR* - 1, one pair per draw, TPR drawn first."""
    tpr = _jeffreys(rng, test.tp, test.tp + test.fn, draws)
    tnr = _jeffreys(rng, test.tn, test.tn + test.fp, draws)
    return tnr, tpr + tnr - 1.0


def _percentile_interval(
    values: NDArray[np.float64], draws: int
) -> tuple[float, float] | None:
    """The central 95% of `values`, if they are at least 90% of the requested draws."""
    if draws <= 0 or len(values) < _MIN_SURVIVING_FRACTION * draws:
        return None
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def corrected_rate(
    test: ConfusionMatrix,
    flagged: int,
    n: int,
    *,
    draws: int = 20_000,
    seed: int = 0,
) -> CorrectedRate:
    """Correct a batch's flag rate with the judge's test-split TPR and TNR.

    theta = (p + TNR - 1) / (TPR + TNR - 1), from p = TPR * theta + (1 - TNR) * (1 - theta).
    Raises `ValueError` on an empty batch or a test matrix missing a class.
    """
    tpr, tnr = _judge_rates(test)
    _check_count(flagged, n, "batch")
    observed = flagged / n
    youden = tpr + tnr - 1.0

    point = math.nan
    clipped: Literal["low", "high"] | None = None
    if youden > _MIN_YOUDEN_J:
        raw = (observed + tnr - 1.0) / youden
        if raw < 0.0:
            clipped = "low"
        elif raw > 1.0:
            clipped = "high"
        point = min(1.0, max(0.0, raw))

    rng = np.random.default_rng(seed)
    tnr_s, youden_s = _youden_draws(rng, test, draws)
    p_s = _jeffreys(rng, flagged, n, draws)
    valid = youden_s > _MIN_YOUDEN_J
    theta_s = np.clip((p_s[valid] + tnr_s[valid] - 1.0) / youden_s[valid], 0.0, 1.0)
    interval = _percentile_interval(theta_s, draws)

    exact = None
    if youden > _MIN_YOUDEN_J:
        exact = _percentile_interval(np.clip((p_s + tnr - 1.0) / youden, 0.0, 1.0), draws)

    return CorrectedRate(
        observed=observed,
        point=point,
        interval=interval,
        interval_if_judge_exact=exact,
        clipped=clipped,
        draws=draws,
        seed=seed,
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_corrected_rates.py tests/test_corrected_rates_simulation.py -q`
Expected: 14 + 4 passed, in a few seconds.

- [ ] **Step 6: Lint and type-check**

Run: `uv run ruff check . && uv run mypy src/axial`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 7: Commit**

```bash
git add src/axial/align.py tests/test_corrected_rates.py tests/test_corrected_rates_simulation.py
git -c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com commit -m "Correct a batch's flag rate for the judge's test-split errors" \
  -m "Rogan-Gladen with a Monte Carlo interval over Jeffreys posteriors, so a judge that scored 30/30 on test keeps its own uncertainty. A second interval holds the judge exact, to show which side more data would tighten. Checked against the course example and by a coverage simulation." \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 2: Corrected difference between two batches

**Files:**
- Modify: `src/axial/align.py` (one import, and new code appended)
- Create: `tests/test_corrected_differences.py`
- Modify: `tests/test_corrected_rates_simulation.py` (its import line, and the difference simulations appended)

**Interfaces:**
- Consumes: Task 1's `_judge_rates`, `_check_count`, `_jeffreys`, `_youden_draws`, `_percentile_interval` and `_MIN_YOUDEN_J`.
- Produces:
  - `CorrectedDifference(point: float, interval: tuple[float, float] | None, paired: bool, n: int, draws: int, seed: int)`, frozen;
  - `corrected_difference_unpaired(test, baseline: tuple[int, int], candidate: tuple[int, int], *, draws=20_000, seed=0)`, where each pair is `(flagged, n_graded)` and `n = n_a + n_b`;
  - `corrected_difference_paired(test, cells: tuple[int, int, int, int], *, draws=20_000, seed=0)`, where `cells = (n11, n10, n01, n00)` and `n` is the number of pairs.
  - Both raise `ValueError` on empty or negative input.

- [ ] **Step 1: Write the failing unit tests**

Create `tests/test_corrected_differences.py`:

```python
"""The corrected difference between two batches' failure rates.

Every expected number here was derived by hand from the counts, or is the design's
worked number (docs/superpowers/specs/2026-10-07-corrected-rates-and-disagreements-design.md,
section 2) to within Monte Carlo error. None is the implementation's own output pasted back.
"""

import numpy as np
import pytest

from axial.align import (
    ConfusionMatrix,
    corrected_difference_paired,
    corrected_difference_unpaired,
    corrected_rate,
)

# The design's worked judge: TPR 28/30 = 0.9333, TNR 66/70 = 0.9429, J = 0.8762.
JUDGE = ConfusionMatrix(tp=28, fn=2, tn=66, fp=4)
CHANCE = ConfusionMatrix(tp=15, fn=15, tn=35, fp=35)
# Monte Carlo percentiles over 20,000 draws move in the third decimal with draw order.
MC = 0.002


def width(interval: tuple[float, float] | None) -> float:
    assert interval is not None
    return interval[1] - interval[0]


def test_identical_batches_straddle_zero():
    unpaired = corrected_difference_unpaired(JUDGE, (60, 500), (60, 500))
    paired = corrected_difference_paired(JUDGE, (45, 15, 15, 425))
    for result in (unpaired, paired):
        assert result.point == pytest.approx(0.0)
        assert result.interval is not None
        assert result.interval[0] < 0 < result.interval[1]
    assert paired.interval == pytest.approx((-0.026, 0.026), abs=MC)


def test_the_difference_has_the_sign_of_the_observed_difference():
    # (80 - 60) / 500 / 0.876190 = +0.045652
    up = corrected_difference_unpaired(JUDGE, (60, 500), (80, 500))
    down = corrected_difference_unpaired(JUDGE, (80, 500), (60, 500))
    assert up.point == pytest.approx(0.045652, abs=1e-6)
    assert down.point == pytest.approx(-0.045652, abs=1e-6)
    assert (up.paired, up.n) == (False, 1000)


def test_the_worked_unpaired_and_paired_comparisons():
    unpaired = corrected_difference_unpaired(JUDGE, (60, 500), (80, 500))
    assert unpaired.interval == pytest.approx((-0.003, 0.099), abs=MC)

    one_way = corrected_difference_paired(JUDGE, (60, 0, 20, 420))
    assert one_way.point == pytest.approx(0.045652, abs=1e-6)
    assert one_way.interval == pytest.approx((0.028, 0.071), abs=MC)
    assert (one_way.paired, one_way.n) == (True, 500)

    both_ways = corrected_difference_paired(JUDGE, (50, 10, 30, 410))
    assert both_ways.interval == pytest.approx((0.018, 0.078), abs=MC)


def test_sharing_the_judge_draws_narrows_the_difference():
    """Drawing the judge independently per batch, as if two judges graded them, is wider."""
    shared = corrected_difference_unpaired(JUDGE, (60, 500), (80, 500))
    rng = np.random.default_rng(0)
    d = 20_000
    j_a = rng.beta(28.5, 2.5, d) + rng.beta(66.5, 4.5, d) - 1
    tnr_a = rng.beta(66.5, 4.5, d)
    j_b = rng.beta(28.5, 2.5, d) + rng.beta(66.5, 4.5, d) - 1
    tnr_b = rng.beta(66.5, 4.5, d)
    p_a = rng.beta(60.5, 440.5, d)
    p_b = rng.beta(80.5, 420.5, d)
    independent = (p_b + tnr_b - 1) / j_b - (p_a + tnr_a - 1) / j_a
    low, high = np.percentile(independent, [2.5, 97.5])
    assert width(shared.interval) < high - low


def test_pairing_narrows_the_difference_when_changes_are_concentrated():
    paired = corrected_difference_paired(JUDGE, (60, 0, 20, 420))
    unpaired = corrected_difference_unpaired(JUDGE, (60, 500), (80, 500))
    assert width(paired.interval) < width(unpaired.interval)


def test_a_rise_hidden_by_clipping_both_rates_is_still_a_difference():
    """The regression the first design hid: both rates clip to 0, the flags still rose."""
    assert corrected_rate(JUDGE, 20, 500).point == 0.0
    assert corrected_rate(JUDGE, 26, 500).point == 0.0
    # (26 - 20) / 500 / 0.876190 = +0.013696
    assert corrected_difference_unpaired(JUDGE, (20, 500), (26, 500)).point == pytest.approx(
        0.013696, abs=1e-6
    )
    paired = corrected_difference_paired(JUDGE, (20, 0, 6, 474))
    assert paired.interval is not None and paired.interval[0] > 0


def test_a_judge_near_chance_gets_no_difference_interval():
    assert corrected_difference_unpaired(CHANCE, (60, 500), (80, 500)).interval is None
    assert corrected_difference_paired(CHANCE, (60, 0, 20, 420)).interval is None


def test_the_differences_refuse_empty_or_negative_input():
    with pytest.raises(ValueError, match="zero pairs"):
        corrected_difference_paired(JUDGE, (0, 0, 0, 0))
    with pytest.raises(ValueError, match="non-negative"):
        corrected_difference_paired(JUDGE, (1, -1, 0, 5))
    with pytest.raises(ValueError, match="baseline"):
        corrected_difference_unpaired(JUDGE, (0, 0), (5, 10))
    with pytest.raises(ValueError, match="no real passes"):
        corrected_difference_paired(ConfusionMatrix(tp=3, fn=0, tn=0, fp=0), (1, 0, 0, 1))
```

- [ ] **Step 2: Add the difference simulations**

In `tests/test_corrected_rates_simulation.py`, change the import

```python
from axial.align import ConfusionMatrix, corrected_rate
```

to

```python
from axial.align import ConfusionMatrix, corrected_difference_paired, corrected_rate
```

and append:

```python
def simulate_paired_gate(
    judge: tuple[float, float, int, int], rise: float, *, seed: int = 3
) -> tuple[float, float]:
    """Share of trials that call a regression, and share whose interval covers `rise`.

    1,000 shared inputs, 10% failing at baseline. A rise turns that share of inputs from
    passing to failing. Every output is graded afresh, so the judge can err on an input
    whose behaviour did not change.
    """
    tpr, tnr, _, _ = judge
    rng = np.random.default_rng(seed)
    inputs, trials, called, covered = 1000, 600, 0, 0
    for _ in range(trials):
        test = draw_test_split(rng, judge)
        fails_a = rng.random(inputs) < 0.10
        fails_b = fails_a | (~fails_a & (rng.random(inputs) < rise / 0.90))
        flag_a = np.where(fails_a, rng.random(inputs) < tpr, rng.random(inputs) < 1 - tnr)
        flag_b = np.where(fails_b, rng.random(inputs) < tpr, rng.random(inputs) < 1 - tnr)
        cells = (
            int((flag_a & flag_b).sum()),
            int((flag_a & ~flag_b).sum()),
            int((~flag_a & flag_b).sum()),
            int((~flag_a & ~flag_b).sum()),
        )
        result = corrected_difference_paired(
            test, cells, draws=3000, seed=int(rng.integers(2**31))
        )
        if result.interval is None:
            continue
        called += result.interval[0] > 0
        covered += result.interval[0] <= rise <= result.interval[1]
    return called / trials, covered / trials


@pytest.mark.parametrize("judge", [STRONG, BORDERLINE], ids=["strong", "borderline"])
def test_the_paired_gate_rarely_calls_a_regression_that_did_not_happen(judge):
    false_alarms, coverage = simulate_paired_gate(judge, 0.0)
    assert false_alarms <= 0.04
    assert 0.92 <= coverage <= 0.99


@pytest.mark.parametrize("judge", [STRONG, BORDERLINE], ids=["strong", "borderline"])
def test_the_paired_interval_covers_a_four_point_rise(judge):
    _, coverage = simulate_paired_gate(judge, 0.04)
    assert 0.92 <= coverage <= 0.99


def test_the_strong_judge_catches_a_four_point_rise_at_a_thousand_inputs():
    caught, _ = simulate_paired_gate(STRONG, 0.04)
    assert caught >= 0.70
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/test_corrected_differences.py tests/test_corrected_rates_simulation.py -q`
Expected: collection errors, `ImportError: cannot import name 'corrected_difference_paired'`.

- [ ] **Step 4: Implement**

In `src/axial/align.py`, add `from collections.abc import Callable` directly below `import math`. Append to the end of the file:

```python
@dataclass(frozen=True)
class CorrectedDifference:
    """Candidate minus baseline corrected failure rate, computed before clipping.

    `n` is the number of pairs when `paired`, otherwise both batches' graded counts
    added together.
    """

    point: float
    interval: tuple[float, float] | None
    paired: bool
    n: int
    draws: int
    seed: int


def _corrected_difference(
    test: ConfusionMatrix,
    observed_difference: float,
    draw_differences: Callable[[np.random.Generator], NDArray[np.float64]],
    *,
    paired: bool,
    n: int,
    draws: int,
    seed: int,
) -> CorrectedDifference:
    """Shared tail of both difference estimators.

    The judge is drawn once per draw and applied to both batches, because one judge
    graded both. The difference is divided by J* before any clipping: the correction
    is linear in p with the same slope for both batches, and clipping each rate first
    turns real rises below the judge's false-alarm rate into a difference of exactly 0.
    """
    tpr, tnr = _judge_rates(test)
    youden = tpr + tnr - 1.0
    point = observed_difference / youden if youden > _MIN_YOUDEN_J else math.nan

    rng = np.random.default_rng(seed)
    _, youden_s = _youden_draws(rng, test, draws)
    flag_difference_s = draw_differences(rng)
    valid = youden_s > _MIN_YOUDEN_J
    interval = _percentile_interval(flag_difference_s[valid] / youden_s[valid], draws)
    return CorrectedDifference(
        point=point, interval=interval, paired=paired, n=n, draws=draws, seed=seed
    )


def corrected_difference_unpaired(
    test: ConfusionMatrix,
    baseline: tuple[int, int],
    candidate: tuple[int, int],
    *,
    draws: int = 20_000,
    seed: int = 0,
) -> CorrectedDifference:
    """Corrected (candidate - baseline) failure rate for two unrelated batches.

    `baseline` and `candidate` are `(flagged, n_graded)`.
    """
    flagged_a, n_a = baseline
    flagged_b, n_b = candidate
    _check_count(flagged_a, n_a, "baseline")
    _check_count(flagged_b, n_b, "candidate")

    def draw(rng: np.random.Generator) -> NDArray[np.float64]:
        p_a = _jeffreys(rng, flagged_a, n_a, draws)
        p_b = _jeffreys(rng, flagged_b, n_b, draws)
        return p_b - p_a

    return _corrected_difference(
        test, flagged_b / n_b - flagged_a / n_a, draw,
        paired=False, n=n_a + n_b, draws=draws, seed=seed,
    )


def corrected_difference_paired(
    test: ConfusionMatrix,
    cells: tuple[int, int, int, int],
    *,
    draws: int = 20_000,
    seed: int = 0,
) -> CorrectedDifference:
    """Corrected (candidate - baseline) failure rate over matched pairs.

    `cells` is `(n11, n10, n01, n00)`: flagged in both, baseline only, candidate only,
    neither. The four cell shares are drawn from their Jeffreys posterior,
    Dirichlet(n + 1/2), and the flag-rate difference is pi01 - pi10.
    """
    if any(c < 0 for c in cells):
        raise ValueError(f"paired cells must be non-negative, got {cells}")
    n11, n10, n01, n00 = cells
    n = n11 + n10 + n01 + n00
    if n == 0:
        raise ValueError("cannot compare two batches over zero pairs")

    def draw(rng: np.random.Generator) -> NDArray[np.float64]:
        shares = rng.dirichlet([n11 + 0.5, n10 + 0.5, n01 + 0.5, n00 + 0.5], draws)
        return shares[:, 2] - shares[:, 1]

    return _corrected_difference(
        test, (n01 - n10) / n, draw, paired=True, n=n, draws=draws, seed=seed
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_corrected_rates.py tests/test_corrected_differences.py tests/test_corrected_rates_simulation.py -q`
Expected: 14 + 8 + 9 passed.

- [ ] **Step 6: Lint and type-check**

Run: `uv run ruff check . && uv run mypy src/axial`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/axial/align.py tests/test_corrected_differences.py tests/test_corrected_rates_simulation.py
git -c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com commit -m "Correct the difference between two batches, paired or not, before clipping" \
  -m "One judge draw serves both batches, and the difference is divided by Youden's J before any clipping, so a rise below the judge's false-alarm rate is still a rise. Paired comparisons draw the four cells from their Jeffreys Dirichlet. A simulation pins false alarms at most 4%, coverage in [0.92, 0.99] and power of at least 0.70 at 1,000 inputs." \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 3: One definition of "promoted", shared by `gate` and `measure`

**Files:**
- Modify: `src/axial/report.py` (imports, and a block inserted before `_fmt_rate`)
- Modify: `src/axial/gate.py` (remove `_SHA256_HEX_RE` and `_require_bound_hash`, and rewrite `_load`)
- Modify: `src/axial/store.py` (one docstring reference)
- Test: `tests/test_report.py` (append)

**Interfaces:**
- Produces:
  - `report.PromotedHashes(judge_yaml_sha256: str, verdicts_sha256: str)`, a `NamedTuple`;
  - `report.require_promoted(store: Store, mode: str, version: int) -> PromotedHashes`, which raises `InvariantViolation` with the same messages `gate._load` raises today;
  - `report.SATURATION_WARNING: str`, the exact text `gate` prints today.
- `gate.gate` keeps its signature and output byte for byte.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_report.py`:

```python
def test_require_promoted_returns_the_hashes_the_marker_recorded(tmp_path: Path):
    from axial.report import require_promoted

    store = Store(tmp_path)
    directory = store.judge_dir("merged", 1)
    directory.mkdir(parents=True)
    (directory / "judge.yaml").write_text("mode: merged\n")
    (directory / "verdicts.jsonl").write_text('{"trace_id": "t1"}\n')
    promote(store, "merged", 1, perfect(), perfect(), Thresholds())

    hashes = require_promoted(store, "merged", 1)

    marker = parse_promoted_marker((directory / "PROMOTED").read_text())
    assert hashes.judge_yaml_sha256 == marker[JUDGE_YAML_HASH_KEY]
    assert hashes.verdicts_sha256 == marker[VERDICTS_HASH_KEY]
    assert hashes.judge_yaml_sha256 == store.hash_file(directory / "judge.yaml")


def test_require_promoted_refuses_an_unpromoted_or_edited_judge(tmp_path: Path):
    from axial.report import require_promoted

    store = Store(tmp_path)
    directory = store.judge_dir("merged", 1)
    directory.mkdir(parents=True)
    (directory / "judge.yaml").write_text("mode: merged\n")
    with pytest.raises(InvariantViolation, match="has not been promoted"):
        require_promoted(store, "merged", 1)

    promote(store, "merged", 1, perfect(), perfect(), Thresholds())
    (directory / "judge.yaml").write_text("mode: merged\nprompt_template: edited\n")
    with pytest.raises(InvariantViolation, match="judge.yaml has changed"):
        require_promoted(store, "merged", 1)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_report.py -q -k require_promoted`
Expected: 2 failed, `ImportError: cannot import name 'require_promoted'`.

- [ ] **Step 3: Move the check into `report.py`**

In `src/axial/report.py`, change the imports to:

```python
import math
import re
from pathlib import Path
from typing import Any, NamedTuple
```

Insert this block directly above `def _fmt_rate`. `_require_bound_hash` and its docstring move verbatim from `gate.py`; `require_promoted` is `gate._load`'s body up to the hash checks.

```python
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

SATURATION_WARNING = (
    "saturation warning: no trace failed. A suite nothing fails has stopped carrying "
    "signal while still costing maintenance — consider harder cases or retiring it."
)


class PromotedHashes(NamedTuple):
    judge_yaml_sha256: str
    verdicts_sha256: str


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


def require_promoted(store: Store, mode: str, version: int) -> PromotedHashes:
    """Refuse unless judge `mode` v`version` is promoted and unchanged since; return
    the two hashes its `PROMOTED` marker recorded.

    `gate` and `measure` both call this, so they cannot disagree about what
    "promoted" means. The marker on its own only proves *some* run was promoted
    once: both hashes it recorded are recomputed, because a hand-edited
    judge.yaml or a verdicts.jsonl overwritten by a later `run_judge` means the
    evidence behind PROMOTED no longer describes what is on disk.
    """
    directory = store.judge_dir(mode, version)
    marker_path = directory / "PROMOTED"
    if not marker_path.exists():
        raise InvariantViolation(
            f"judge {mode} v{version} has not been promoted. Run `axial promote` — a judge "
            "whose agreement with human labels was never measured on test cannot gate a build."
        )
    recorded = parse_promoted_marker(marker_path.read_text())
    recorded_judge_hash = _require_bound_hash(recorded, JUDGE_YAML_HASH_KEY, mode, version)
    if store.hash_file(directory / "judge.yaml") != recorded_judge_hash:
        raise InvariantViolation(
            f"judge {mode} v{version}'s judge.yaml has changed since it was promoted — "
            "the judge that earned PROMOTED is not the judge on disk. Re-run `axial "
            "promote` on the current judge, or run `axial judge --new` to score the "
            "changed prompt as a fresh version."
        )
    recorded_verdicts_hash = _require_bound_hash(recorded, VERDICTS_HASH_KEY, mode, version)
    if store.hash_file(directory / "verdicts.jsonl") != recorded_verdicts_hash:
        raise InvariantViolation(
            f"judge {mode} v{version}'s verdicts.jsonl has changed since it was promoted — "
            "the verdicts that earned PROMOTED are not the verdicts on disk. Re-run `axial "
            "promote` on the current run, or run `axial judge --new` to score it as a "
            "fresh version."
        )
    return PromotedHashes(recorded_judge_hash, recorded_verdicts_hash)
```

In `format_promoted_marker`'s docstring, change ``its own marker. `gate._load` recomputes both hashes`` to ``its own marker. `require_promoted` recomputes both hashes``.

- [ ] **Step 4: Make `gate.py` use it**

In `src/axial/gate.py`:
- delete `import re`, `_SHA256_HEX_RE` and the whole `_require_bound_hash` function;
- replace the `axial.report` import with `from axial.report import SATURATION_WARNING, require_promoted`;
- replace `_load` with:

```python
def _load(store: Store, mode: str, version: int) -> list[dict[str, Any]]:
    require_promoted(store, mode, version)
    rows = store.read_jsonl(store.judge_dir(mode, version) / "verdicts.jsonl")
    if not rows:
        raise InvariantViolation(
            f"judge {mode} v{version} has no verdict rows. A gate cannot pass on no evidence. "
            "Check errors.jsonl for quarantined calls."
        )
    return rows
```

In `gate()`, replace the inline saturation message with `messages.append(SATURATION_WARNING)`.

In `src/axial/store.py`, in `hash_file`'s docstring, change `` `gate._load` `` to `` `report.require_promoted` ``.

- [ ] **Step 5: Run the gate and report tests to verify nothing moved**

Run: `uv run pytest tests/test_report.py tests/test_gate.py tests/test_capabilities_promote_gate.py -q`
Expected: all pass; the two xfails in `test_capabilities_promote_gate.py` are unchanged.

- [ ] **Step 6: Lint, type-check and commit**

Run: `uv run ruff check . && uv run mypy src/axial`
Expected: clean. If ruff reports `I001` in `gate.py`, the deletions left one blank line between the imports and `@dataclass` where two belong; `uv run ruff check --fix .` restores them.

```bash
git add src/axial/report.py src/axial/gate.py src/axial/store.py tests/test_report.py
git -c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com commit -m "Move the promotion check into report so gate and measure share it" \
  -m "require_promoted returns the two hashes the marker recorded, so a measurement can bind itself to the judge that took it. gate's checks and messages are unchanged." \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 4: Extract `grade_trace` and `resolve_few_shot` from `run_judge`

**Files:**
- Modify: `src/axial/judge.py`
- Test: `tests/test_judge.py` (imports, and tests appended)

**Interfaces:**
- Produces:
  - `Graded(verdict: bool, raw: str, model: str, usage: dict[str, int], attempts: int, latency_s: float)`, frozen;
  - `SetAside(failure_class: str, detail: str)`, frozen;
  - `resolve_few_shot(spec: JudgeSpec, corpus: list[Trace], labels: list[Label]) -> list[tuple[Trace, Label]]`;
  - `grade_trace(spec: JudgeSpec, trace: Trace, few_shot: list[tuple[Trace, Label]], splits: Splits | None, model: Model, *, max_tokens: int) -> Graded | SetAside`. Its failure classes are `bad_template`, `harness`, `truncated` and `unparseable`; an `InvariantViolation` propagates.
- `run_judge` keeps its signature, rows and on-disk behaviour.

- [ ] **Step 1: Write the failing tests**

In `tests/test_judge.py`, replace the `axial.judge` import with:

```python
from axial.judge import (
    Graded,
    JudgeSpec,
    SetAside,
    Verdict,
    assemble_prompt,
    grade_trace,
    parse_verdict,
    resolve_few_shot,
    run_judge,
)
```

and append:

```python
def test_grade_trace_returns_the_verdict_and_what_the_model_said():
    model = FakeModel(lambda _p: '{"verdict": "fail", "reason": "merged"}', model="m-1")
    outcome = grade_trace(spec(), trace(1), [], None, model, max_tokens=64)
    assert isinstance(outcome, Graded)
    assert (outcome.verdict, outcome.model, outcome.raw) == (
        True, "m-1", '{"verdict": "fail", "reason": "merged"}',
    )


class _Broken:
    def complete(self, prompt, *, max_tokens=1024, schema=None):
        raise TimeoutError("provider timed out")


@pytest.mark.parametrize(
    ("template", "model", "failure_class"),
    [
        ("{nonsense}", FakeModel(lambda _p: '{"verdict": "pass"}'), "bad_template"),
        (TEMPLATE, _Broken(), "harness"),
        (TEMPLATE, FakeModel(lambda _p: '{"verdict": "pass"}', truncated=True), "truncated"),
        (TEMPLATE, FakeModel(lambda _p: "maybe?"), "unparseable"),
    ],
)
def test_grade_trace_sets_aside_what_it_cannot_score(template, model, failure_class):
    outcome = grade_trace(spec(prompt_template=template), trace(1), [], None, model, max_tokens=64)
    assert isinstance(outcome, SetAside)
    assert outcome.failure_class == failure_class


def test_grade_trace_lets_a_refusal_propagate():
    splits = Splits(train=[], dev=["t2"], test=[])
    with pytest.raises(InvariantViolation, match="not in the train split"):
        grade_trace(
            spec(), trace(1), [(trace(2), label(2, True))], splits,
            FakeModel(lambda _p: '{"verdict": "pass"}'), max_tokens=64,
        )


def test_resolve_few_shot_refuses_an_id_it_cannot_find():
    gold = label(1, True)
    with pytest.raises(InvariantViolation, match="'t9' could not be resolved"):
        resolve_few_shot(spec(few_shot_ids=["t9"]), [trace(1)], [gold])
    assert resolve_few_shot(spec(few_shot_ids=["t1"]), [trace(1)], [gold]) == [(trace(1), gold)]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_judge.py -q`
Expected: collection error, `ImportError: cannot import name 'Graded'`.

- [ ] **Step 3: Implement**

In `src/axial/judge.py`, replace everything from `@dataclass` / `class RunResult:` down to, but not including, the line `    # Merge by split rather than overwrite, so running dev and then test leaves both` with:

```python
@dataclass
class RunResult:
    verdicts: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class Graded:
    """A model answer that was read as a verdict."""

    verdict: bool
    raw: str
    model: str
    usage: dict[str, int]
    attempts: int
    latency_s: float


@dataclass(frozen=True)
class SetAside:
    """A trace that could not be scored, and why. Never a verdict."""

    failure_class: str
    detail: str


def resolve_few_shot(
    spec: JudgeSpec, corpus: list[Trace], labels: list[Label]
) -> list[tuple[Trace, Label]]:
    """Look up each of `spec.few_shot_ids` in `corpus` and `labels`, refusing a miss."""
    corpus_by_id = {t.trace_id: t for t in corpus}
    label_by_id = {lbl.trace_id: lbl for lbl in labels}
    few_shot: list[tuple[Trace, Label]] = []
    for few_shot_id in spec.few_shot_ids:
        example = corpus_by_id.get(few_shot_id)
        gold = label_by_id.get(few_shot_id)
        if example is None or gold is None:
            missing = "trace" if example is None else "label"
            raise InvariantViolation(
                f"few-shot id {few_shot_id!r} could not be resolved: its {missing} is "
                "missing from the corpus/labels given to run_judge. Silently dropping "
                "an unresolved few-shot id is what let this go unnoticed before."
            )
        few_shot.append((example, gold))
    return few_shot


def grade_trace(
    spec: JudgeSpec,
    trace: Trace,
    few_shot: list[tuple[Trace, Label]],
    splits: Splits | None,
    model: Model,
    *,
    max_tokens: int,
) -> Graded | SetAside:
    """Grade one trace, or say why it was set aside.

    Shared by `run_judge` and `measure`, so a labelled trace and a new one are graded
    the same way. An `InvariantViolation` from prompt assembly is a refusal, not a
    per-trace failure, so it propagates.
    """
    try:
        prompt = assemble_prompt(spec, trace, few_shot=few_shot, splits=splits)
    except InvariantViolation:
        raise
    except (KeyError, IndexError, ValueError) as exc:
        return SetAside("bad_template", f"{type(exc).__name__}: {exc}")

    try:
        completion = model.complete(prompt, max_tokens=max_tokens, schema=Verdict)
    except Exception as exc:  # noqa: BLE001 - quarantined, never scored
        return SetAside("harness", f"{type(exc).__name__}: {exc}")

    if completion.truncated:
        return SetAside("truncated", f"stop_reason={completion.stop_reason}")

    try:
        verdict = parse_verdict(completion.text)
    except ValueError as exc:
        return SetAside("unparseable", str(exc))

    return Graded(
        verdict=verdict,
        raw=completion.text,
        model=completion.model,
        usage=completion.usage,
        attempts=completion.attempts,
        latency_s=completion.latency_s,
    )


def run_judge(
    store: Store,
    spec: JudgeSpec,
    traces: list[Trace],
    labels: list[Label],
    splits: Splits,
    model: Model,
    *,
    split: str,
    max_tokens: int = 512,
    corpus: list[Trace] | None = None,
) -> RunResult:
    """Run `spec` over `traces`, writing verdict and error rows to `store`.

    `corpus` is the full trace set to resolve few-shot examples against — it
    defaults to `traces`, but a real run judges one split at a time, and a
    few-shot id almost always names a trace from train, not from the split
    under judgment. Passing only `traces` here would silently starve
    `few_shot` (and, with it, the train-only guarantee in `assemble_prompt`,
    which has nothing to check when its input list is empty) — pass the whole
    corpus explicitly.
    """
    few_shot = resolve_few_shot(spec, traces if corpus is None else corpus, labels)
    label_by_id = {lbl.trace_id: lbl for lbl in labels}

    result = RunResult()
    for trace in traces:
        outcome = grade_trace(spec, trace, few_shot, splits, model, max_tokens=max_tokens)
        if isinstance(outcome, SetAside):
            result.errors.append(
                {
                    "trace_id": trace.trace_id,
                    "split": split,
                    "failure_class": outcome.failure_class,
                    "detail": outcome.detail,
                }
            )
            continue

        gold = label_by_id.get(trace.trace_id)
        if gold is None:
            result.errors.append(
                {
                    "trace_id": trace.trace_id,
                    "split": split,
                    "failure_class": "unlabelled",
                    "detail": "no human label for this trace; a verdict with no ground "
                    "truth cannot be scored and must not carry truth=None downstream",
                }
            )
            continue

        result.verdicts.append(
            {
                "trace_id": trace.trace_id,
                "split": split,
                "verdict": outcome.verdict,
                "truth": gold.label,
                "model": outcome.model,
                "usage": outcome.usage,
                "attempts": outcome.attempts,
                "latency_s": outcome.latency_s,
                "raw": outcome.raw,
            }
        )
```

The rest of `run_judge`, the merge-by-split writing and the PROMOTED deletion, stays as it is.

- [ ] **Step 4: Run every judge-path test to verify the extraction changed nothing**

Run: `uv run pytest -q`
Expected: everything passes. The xfail count is still 49, because Tasks 1–3 added none.

- [ ] **Step 5: Lint, type-check and commit**

Run: `uv run ruff check . && uv run mypy src/axial`
Expected: clean.

```bash
git add src/axial/judge.py tests/test_judge.py
git -c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com commit -m "Extract grade_trace and resolve_few_shot from run_judge" \
  -m "measure grades unlabelled traces the way judge grades labelled ones, so the per-trace grading and the few-shot lookup become functions both can call. run_judge's rows and files are unchanged." \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 5: Write a directory all at once

**Files:**
- Modify: `src/axial/store.py`
- Test: `tests/test_store.py` (append)

**Interfaces:**
- Produces:
  - `store.jsonl_text(rows) -> str` and `store.json_text(obj) -> str`, the exact bytes `write_jsonl` and `write_json` write;
  - `Store.measurement_dir(mode: str, batch: str) -> Path`, which is `.axial/measurements/<mode>/<batch>`;
  - `Store.write_dir_atomically(path: Path, files: dict[str, str]) -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_store.py`, which already imports `os`, `stat`, `Path`, `pytest` and `Store`:

```python
def test_write_dir_atomically_writes_every_file(tmp_path: Path):
    store = Store(tmp_path)
    target = store.root / "measurements" / "m" / "b1"
    store.write_dir_atomically(target, {"a.txt": "one\n", "b.txt": "two\n"})
    assert {p.name: p.read_text() for p in target.iterdir()} == {"a.txt": "one\n", "b.txt": "two\n"}
    assert sorted(p.name for p in target.parent.iterdir()) == ["b1"]


def test_write_dir_atomically_replaces_the_whole_directory(tmp_path: Path):
    store = Store(tmp_path)
    target = tmp_path / "out"
    store.write_dir_atomically(target, {"old.txt": "x", "kept-name.txt": "old"})
    store.write_dir_atomically(target, {"kept-name.txt": "new"})
    assert {p.name: p.read_text() for p in target.iterdir()} == {"kept-name.txt": "new"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out"]


def test_write_dir_atomically_leaves_the_old_directory_when_a_write_fails(tmp_path: Path):
    store = Store(tmp_path)
    target = tmp_path / "out"
    store.write_dir_atomically(target, {"a.txt": "old"})
    with pytest.raises(FileNotFoundError):
        # A name inside a subdirectory that was never created fails mid-write.
        store.write_dir_atomically(target, {"a.txt": "new", "missing/b.txt": "x"})
    assert {p.name: p.read_text() for p in target.iterdir()} == {"a.txt": "old"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out"]


def test_write_dir_atomically_gives_the_directory_a_normal_mode(tmp_path: Path):
    store = Store(tmp_path)
    target = tmp_path / "out"
    store.write_dir_atomically(target, {"a.txt": "x"})
    mask = os.umask(0)
    os.umask(mask)
    assert stat.S_IMODE(target.stat().st_mode) == 0o777 & ~mask
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_store.py -q -k write_dir_atomically`
Expected: 4 failed, `AttributeError: 'Store' object has no attribute 'write_dir_atomically'`.

- [ ] **Step 3: Implement**

In `src/axial/store.py`, add `import shutil` and `import uuid` to the standard-library imports, in sorted order. Insert directly above `class Store:`:

```python
def jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    """The exact bytes `Store.write_jsonl` writes for `rows`."""
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def json_text(obj: Any) -> str:
    """The exact bytes `Store.write_json` writes for `obj`."""
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"
```

Below `judge_dir`, add:

```python
    def measurement_dir(self, mode: str, batch: str) -> Path:
        return self.root / "measurements" / mode / batch
```

Replace `write_jsonl` and `write_json` with the following, which also adds `write_dir_atomically` after them:

```python
    def write_jsonl(self, path: Path, rows: Iterable[dict[str, Any]]) -> None:
        self._atomic_write(path, jsonl_text(rows))

    def write_json(self, path: Path, obj: Any) -> None:
        self._atomic_write(path, json_text(obj))

    def write_dir_atomically(self, path: Path, files: dict[str, str]) -> None:
        """Make `path` a directory holding exactly `files` (name -> text), all or nothing.

        The files go into a hidden sibling directory that is then renamed onto
        `path`; an existing `path` is renamed aside first and removed only once the
        new one is in place, and put back if that rename fails. A run interrupted
        part-way leaves `path` as it was, never half-written -- the same guarantee
        `_atomic_write` gives a single file, for a set of files that only mean
        something together. `mkdtemp` creates its directory mode 0700, so the mode
        is fixed up to what a plain `mkdir` would give.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"))
        try:
            os.chmod(staging, 0o777 & ~_current_umask())
            for name, data in files.items():
                (staging / name).write_text(data)
            if not path.exists():
                os.replace(staging, path)
                return
            retired = path.with_name(f".{path.name}.{uuid.uuid4().hex}.old")
            os.replace(path, retired)
            try:
                os.replace(staging, path)
            except BaseException:
                os.replace(retired, path)
                raise
            shutil.rmtree(retired)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_store.py -q`
Expected: 16 passed.

- [ ] **Step 5: Lint, type-check and commit**

Run: `uv run ruff check . && uv run mypy src/axial`
Expected: clean.

```bash
git add src/axial/store.py tests/test_store.py
git -c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com commit -m "Write a set of files as one directory, all or nothing" \
  -m "A measurement's four files only mean something together. They are staged in a hidden sibling and renamed into place; a replaced directory is moved aside first and restored if the swap fails." \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 6: `measure_batch`, which grades a named batch with a promoted judge

**Files:**
- Create: `src/axial/measure.py`
- Create: `tests/measure_helpers.py`
- Create: `tests/test_measure.py`

**Interfaces:**
- Consumes:
  - `require_promoted`, `PromotedHashes` and `SATURATION_WARNING` (Task 3);
  - `resolve_few_shot`, `grade_trace` and `Graded` (Task 4);
  - `Store.measurement_dir`, `Store.write_dir_atomically`, `jsonl_text` and `json_text` (Task 5);
  - `corrected_rate` and `CorrectedRate` (Task 1).
- Produces:
  - `measure_batch(store, mode, version, batch_name, batch: list[Trace], build_model: Callable[[JudgeSpec], Model], *, replace=False, max_tokens=512, draws=20_000, seed=0) -> Measurement`;
  - `Measurement`, a frozen dataclass with `to_json()` and `from_json()`. Its `rate` is a `CorrectedRate`, or `None` when no trace was graded.
  - `load_measurement(store, mode, batch) -> tuple[Measurement, list[dict]]`, which returns the measurement and its graded rows;
  - `summary_lines(m: Measurement) -> list[str]`;
  - `input_sha256(text: str) -> str`;
  - `validate_batch_name(name: str) -> str`;
  - `ASSUMPTION_NOTE`.
- Row and file shapes are exactly the spec's section 1, plus the `clipped` field (delta 2).

`build_model` is a parameter rather than a call to `build_judge_model`, so the CLI can pass its own module-level `build_judge_model`. The existing CLI tests monkeypatch that name to a `FakeModel`.

- [ ] **Step 1: Create the shared test helpers**

Create `tests/measure_helpers.py`:

```python
"""Shared helpers for the measurement tests: a promoted judge with a known test matrix.

The judge is a FakeModel that reads the graded OUTPUT. "BAD" and "GOOD-ALARM" are
flagged, everything else passes, so its test split comes out at exactly tp 28, fn 2,
tn 66, fp 4 -- the worked judge of the corrected-rates design -- and a batch's flag
count is set by how many of its outputs are "BAD".
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from axial.config import Thresholds
from axial.dataset import Label, Splits, add_labels
from axial.judge import JudgeSpec, run_judge
from axial.model import FakeModel
from axial.report import alignment_from_verdicts, promote
from axial.store import Store
from axial.traces import Trace

MEASURE_MODE = "merged"
MEASURE_TEMPLATE = "{few_shot}\nINPUT: {input}\nOUTPUT: {output}"


def judge_reply(prompt: str) -> str:
    output = prompt.rsplit("OUTPUT: ", 1)[1]
    if output == "GARBLED":
        return "maybe?"
    verdict = "fail" if output in ("BAD", "GOOD-ALARM") else "pass"
    return json.dumps({"verdict": verdict, "reason": f"the output was {output}"})


def build_promoted_judge(
    store: Store, *, version: int = 1, few_shot_ids: list[str] | None = None
) -> None:
    corpus: list[Trace] = []
    labels: list[Label] = []

    def add(trace_id: str, output: str, failed: bool) -> None:
        corpus.append(Trace(trace_id=trace_id, input=f"question {trace_id}", output=output))
        labels.append(
            Label(trace_id=trace_id, label=failed, critique=f"{output} answer", annotator="andres")
        )

    train = [f"train-{i}" for i in range(4)]
    dev = [f"dev-{i:02d}" for i in range(26)]
    test = [f"test-{i:03d}" for i in range(100)]
    for i, trace_id in enumerate(train + dev):
        add(trace_id, "BAD" if i % 2 == 0 else "GOOD", i % 2 == 0)
    outputs = ["BAD"] * 28 + ["BAD-MISSED"] * 2 + ["GOOD"] * 66 + ["GOOD-ALARM"] * 4
    for trace_id, output in zip(test, outputs, strict=True):
        add(trace_id, output, output.startswith("BAD"))

    store.write_traces(corpus)
    add_labels(store, MEASURE_MODE, labels)
    splits = Splits(train=train, dev=dev, test=test)
    splits.save(store, MEASURE_MODE)
    spec = JudgeSpec(
        mode=MEASURE_MODE,
        prompt_template=MEASURE_TEMPLATE,
        model="fake-judge",
        few_shot_ids=few_shot_ids or [],
        version=version,
    )
    spec.save(store)
    by_id = {t.trace_id: t for t in corpus}
    model = FakeModel(judge_reply, model="fake-judge")
    for split in ("dev", "test"):
        run_judge(
            store, spec, [by_id[t] for t in getattr(splits, split)], labels, splits, model,
            split=split, corpus=corpus,
        )
    verdicts = store.read_jsonl(store.judge_dir(MEASURE_MODE, version) / "verdicts.jsonl")
    promote(
        store, MEASURE_MODE, version,
        alignment_from_verdicts(verdicts, "dev"),
        alignment_from_verdicts(verdicts, "test"),
        Thresholds(),
    )


def batch_traces(outputs: list[str], *, prefix: str = "new", start: int = 0) -> list[Trace]:
    """Traces whose input depends only on the position, so two batches built with the
    same `start` share their inputs whatever their prefix."""
    return [
        Trace(trace_id=f"{prefix}-{start + i}", input=f"case {start + i}", output=output)
        for i, output in enumerate(outputs)
    ]


def write_batch(
    path: Path, outputs: list[str], *, prefix: str = "new", start: int = 0
) -> Path:
    """Write a batch to a JSONL file and return its path."""
    path.write_text(
        "".join(
            json.dumps(t.model_dump()) + "\n"
            for t in batch_traces(outputs, prefix=prefix, start=start)
        )
    )
    return path


def fake_judge_builder(
    specs: list[JudgeSpec] | None = None, prompts: list[str] | None = None
) -> Callable[[JudgeSpec], FakeModel]:
    """Build the judge's model as a FakeModel, recording the spec it was built from and,
    if asked, every prompt it is sent."""

    def build(spec: JudgeSpec) -> FakeModel:
        if specs is not None:
            specs.append(spec)
        model = FakeModel(judge_reply, model=spec.model)
        if prompts is not None:
            model.calls = prompts
        return model

    return build
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_measure.py`:

```python
"""`measure_batch`: grading new traces with a promoted judge, and pairing two batches."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest
from measure_helpers import (
    MEASURE_MODE,
    batch_traces,
    build_promoted_judge,
    fake_judge_builder,
)

from axial.align import ConfusionMatrix
from axial.errors import AxialError, InvariantViolation
from axial.judge import JudgeSpec
from axial.measure import (
    ASSUMPTION_NOTE,
    Measurement,
    input_sha256,
    load_measurement,
    measure_batch,
)
from axial.store import Store
from axial.traces import Trace

MEASUREMENT_FIELDS = {
    "mode", "batch", "judge_version", "judge_yaml_sha256", "verdicts_sha256", "test_matrix",
    "n_traces", "n_graded", "n_set_aside", "flagged", "observed_rate", "corrected_rate",
    "clipped", "interval", "interval_if_judge_exact", "draws", "seed", "notes", "created",
}


@pytest.fixture
def store(tmp_path: Path) -> Store:
    store = Store(tmp_path)
    build_promoted_judge(store)
    return store


def measure(store: Store, traces: list[Trace], name: str = "rel-1", **kw):
    return measure_batch(
        store, MEASURE_MODE, 1, name, traces, fake_judge_builder(), draws=2000, **kw
    )


def snapshot(directory: Path) -> dict[str, bytes]:
    return {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob("*") if p.is_file()}


# --- refusals --------------------------------------------------------------------


def test_an_unpromoted_judge_cannot_measure(store: Store):
    (store.judge_dir(MEASURE_MODE, 1) / "PROMOTED").unlink()
    with pytest.raises(InvariantViolation, match="has not been promoted"):
        measure(store, batch_traces(["GOOD"]))


def test_a_judge_edited_after_promotion_cannot_measure(store: Store):
    path = store.judge_dir(MEASURE_MODE, 1) / "judge.yaml"
    path.write_text(path.read_text() + "\n# tweaked\n")
    with pytest.raises(InvariantViolation, match="judge.yaml has changed"):
        measure(store, batch_traces(["GOOD"]))


def test_an_empty_batch_is_refused(store: Store):
    with pytest.raises(AxialError, match="empty"):
        measure(store, [])


def test_duplicate_trace_ids_are_refused(store: Store):
    twice = [Trace(trace_id="x", input="a", output="GOOD"), Trace(trace_id="x", input="b", output="BAD")]
    with pytest.raises(AxialError, match="repeats 1 trace id"):
        measure(store, twice)


@pytest.mark.parametrize("name", ["", ".", "..", "../escape", "a/b", ".hidden", "sp ace"])
def test_a_name_that_is_not_a_plain_directory_name_is_refused(store: Store, name: str):
    with pytest.raises(AxialError, match="not allowed"):
        measure(store, batch_traces(["GOOD"]), name=name)


def test_an_existing_name_is_refused_unless_replaced(store: Store):
    measure(store, batch_traces(["GOOD"] * 3))
    with pytest.raises(AxialError, match="already exists"):
        measure(store, batch_traces(["BAD"] * 5))

    replaced = measure(store, batch_traces(["BAD"] * 5), replace=True)
    assert replaced.n_traces == 5
    directory = store.measurement_dir(MEASURE_MODE, "rel-1")
    assert len(store.read_jsonl(directory / "traces.jsonl")) == 5
    assert sorted(p.name for p in directory.parent.iterdir()) == ["rel-1"]


def test_a_test_split_missing_a_class_is_refused(store: Store):
    path = store.judge_dir(MEASURE_MODE, 1) / "verdicts.jsonl"
    rows = [r for r in store.read_jsonl(path) if not (r["split"] == "test" and r["truth"])]
    store.write_jsonl(path, rows)
    marker = store.judge_dir(MEASURE_MODE, 1) / "PROMOTED"
    text = marker.read_text().splitlines()
    marker.write_text(
        "\n".join(
            line if not line.startswith("verdicts_sha256=") else f"verdicts_sha256={store.hash_file(path)}"
            for line in text
        ) + "\n"
    )
    with pytest.raises(AxialError, match="no real failures"):
        measure(store, batch_traces(["GOOD"]))


# --- what it writes --------------------------------------------------------------


def test_the_judge_and_the_labelled_corpus_are_untouched(store: Store):
    judges, traces = store.root / "judges", store.root / "traces"
    before = (snapshot(judges), snapshot(traces))
    measure(store, batch_traces(["BAD", "GOOD", "GOOD"]))
    assert (snapshot(judges), snapshot(traces)) == before


def test_it_writes_four_files_and_every_measurement_field(store: Store):
    traces = batch_traces(["BAD"] * 60 + ["GOOD"] * 440)
    result = measure(store, traces)

    directory = store.measurement_dir(MEASURE_MODE, "rel-1")
    assert sorted(p.name for p in directory.iterdir()) == [
        "errors.jsonl", "measurement.json", "traces.jsonl", "verdicts.jsonl",
    ]
    assert [Trace.model_validate(r) for r in store.read_jsonl(directory / "traces.jsonl")] == traces
    data = json.loads((directory / "measurement.json").read_text())
    assert set(data) == MEASUREMENT_FIELDS
    assert data["test_matrix"] == {"tp": 28, "fn": 2, "tn": 66, "fp": 4}
    assert (data["n_traces"], data["n_graded"], data["n_set_aside"], data["flagged"]) == (500, 500, 0, 60)
    # (0.12 + 66/70 - 1) / (28/30 + 66/70 - 1) = 0.071739
    assert data["corrected_rate"] == pytest.approx(0.071739, abs=1e-6)
    assert data["observed_rate"] == pytest.approx(0.12)
    assert data["notes"] == [ASSUMPTION_NOTE]
    assert result.test_matrix == ConfusionMatrix(tp=28, fn=2, tn=66, fp=4)


def test_verdict_rows_carry_the_input_hash_and_no_truth(store: Store):
    traces = batch_traces(["BAD", "GOOD"])
    measure(store, traces)
    rows = store.read_jsonl(store.measurement_dir(MEASURE_MODE, "rel-1") / "verdicts.jsonl")
    assert [r["input_sha256"] for r in rows] == [input_sha256(t.input) for t in traces]
    assert [r["verdict"] for r in rows] == [True, False]
    assert all("truth" not in r for r in rows)
    assert set(rows[0]) == {
        "trace_id", "input_sha256", "verdict", "model", "usage", "attempts", "latency_s", "raw",
    }


def test_prompts_use_the_promoted_judge_yaml_and_its_train_examples(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    built: list[JudgeSpec] = []
    prompts: list[str] = []
    measure_batch(
        store, MEASURE_MODE, 1, "rel-1", batch_traces(["GOOD"]),
        fake_judge_builder(built, prompts), draws=2000,
    )
    assert [(s.model, s.few_shot_ids) for s in built] == [("fake-judge", ["train-0"])]
    assert len(prompts) == 1
    assert "EXAMPLE\nINPUT:\nquestion train-0\nOUTPUT:\nBAD\n" in prompts[0]
    assert prompts[0].endswith("INPUT: case 0\nOUTPUT: GOOD")
    raw = store.read_jsonl(store.measurement_dir(MEASURE_MODE, "rel-1") / "verdicts.jsonl")
    assert raw[0]["model"] == "fake-judge"


def test_a_few_shot_example_in_the_batch_is_set_aside(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    traces = [Trace(trace_id="train-0", input="question train-0", output="BAD")]
    traces += batch_traces(["GOOD"] * 9)
    result = measure_batch(
        store, MEASURE_MODE, 1, "rel-1", traces, fake_judge_builder(), draws=2000
    )
    errors = store.read_jsonl(store.measurement_dir(MEASURE_MODE, "rel-1") / "errors.jsonl")
    assert [(e["trace_id"], e["failure_class"]) for e in errors] == [("train-0", "few_shot_example")]
    assert (result.n_graded, result.n_set_aside, result.flagged) == (9, 1, 0)


def test_set_aside_traces_stay_out_of_the_rate_and_are_noted_above_five_percent(store: Store):
    result = measure(store, batch_traces(["GARBLED"] * 2 + ["BAD"] * 6 + ["GOOD"] * 12))
    assert (result.n_traces, result.n_graded, result.n_set_aside, result.flagged) == (20, 18, 2, 6)
    assert result.rate is not None and result.rate.observed == pytest.approx(6 / 18)
    errors = store.read_jsonl(store.measurement_dir(MEASURE_MODE, "rel-1") / "errors.jsonl")
    assert {e["failure_class"] for e in errors} == {"unparseable"}
    assert all("input_sha256" in e for e in errors)
    assert any("2 of 20 traces (10%) were set aside" in n for n in result.notes)


def test_five_percent_set_aside_is_not_noted(store: Store):
    result = measure(store, batch_traces(["GARBLED"] + ["GOOD"] * 19))
    assert not any("set aside" in n for n in result.notes)


def test_a_rate_below_the_false_alarm_rate_is_noted_as_clipped(store: Store):
    result = measure(store, batch_traces(["BAD"] * 20 + ["GOOD"] * 480))
    assert result.rate is not None and result.rate.clipped == "low"
    assert any("clipped to 0" in n and "1 - TNR = 0.057" in n for n in result.notes)


def test_a_batch_with_nothing_flagged_gets_the_saturation_warning(store: Store):
    result = measure(store, batch_traces(["GOOD"] * 10))
    assert any(n.startswith("saturation warning") for n in result.notes)


def test_a_batch_that_was_all_set_aside_is_written_without_a_rate(store: Store):
    result = measure(store, batch_traces(["GARBLED"] * 4))
    assert result.rate is None
    data = json.loads(
        (store.measurement_dir(MEASURE_MODE, "rel-1") / "measurement.json").read_text()
    )
    assert (data["n_graded"], data["corrected_rate"], data["interval"]) == (0, None, None)
    assert data["observed_rate"] is None
    assert "every trace was set aside, so no rate was estimated" in data["notes"]


# --- reading a measurement back ----------------------------------------------------


def test_a_measurement_reads_back_as_written(store: Store):
    written = measure(store, batch_traces(["BAD"] * 60 + ["GOOD"] * 440))
    read, rows = load_measurement(store, MEASURE_MODE, "rel-1")
    assert read == written
    assert len(rows) == 500


def test_reading_a_missing_measurement_names_the_command_to_run(store: Store):
    with pytest.raises(AxialError, match="axial measure merged"):
        load_measurement(store, MEASURE_MODE, "nope")


def test_a_hand_edited_measurement_is_refused(store: Store):
    measure(store, batch_traces(["BAD", "GOOD"]))
    path = store.measurement_dir(MEASURE_MODE, "rel-1") / "verdicts.jsonl"
    rows = store.read_jsonl(path)
    rows[1]["verdict"] = True
    store.write_jsonl(path, rows)
    with pytest.raises(AxialError, match="edited by hand"):
        load_measurement(store, MEASURE_MODE, "rel-1")


def test_an_undefined_point_is_written_as_null_and_read_back_as_nan(store: Store):
    written = measure(store, batch_traces(["BAD", "GOOD"]))
    assert written.rate is not None
    undefined = dataclasses.replace(
        written, rate=dataclasses.replace(written.rate, point=math.nan, interval=None)
    )
    data = json.loads(json.dumps(undefined.to_json()))
    assert (data["corrected_rate"], data["interval"]) == (None, None)
    back = Measurement.from_json(data)
    assert back.rate is not None and math.isnan(back.rate.point)
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/test_measure.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'axial.measure'`.

- [ ] **Step 4: Implement**

Create `src/axial/measure.py`:

```python
"""Grading new traces with a promoted judge, and pairing two measured batches.

A promoted judge's own verdicts say how well it agrees with the human labels; they
say nothing about what the system has produced since. `measure_batch` grades a named
batch of unlabelled traces with that judge and corrects the batch's flag rate for the
judge's test-split errors. Measurements live in their own tree,
`.axial/measurements/<mode>/<batch>/`, because writing into the judge's
`verdicts.jsonl` would change the hash its PROMOTED marker binds and withdraw it.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from axial.align import ConfusionMatrix, CorrectedRate, confusion_matrix, corrected_rate
from axial.dataset import Splits, read_labels
from axial.errors import AxialError
from axial.judge import Graded, JudgeSpec, grade_trace, resolve_few_shot
from axial.model import Model
from axial.report import SATURATION_WARNING, require_promoted
from axial.store import Store, json_text, jsonl_text
from axial.traces import Trace

# A batch name becomes a directory. It must start with a letter or digit, which also
# keeps "." and ".." out and leaves dot-names free for `write_dir_atomically`'s staging.
BATCH_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

# Above this share of set-aside traces, a measurement says the rate may be biased.
SET_ASIDE_NOTE_SHARE = 0.05


ASSUMPTION_NOTE = (
    "the correction assumes the judge errs on this batch at the rates it showed on its "
    "test split; only labels on this batch can confirm that for outputs unlike the "
    "labelled corpus"
)


def input_sha256(text: str) -> str:
    """The pairing key: SHA-256 of a trace's input text, UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_batch_name(name: str) -> str:
    if not BATCH_NAME_RE.fullmatch(name):
        raise AxialError(
            f"batch name {name!r} is not allowed: it becomes a directory, so use letters, "
            "digits, '.', '_' and '-', starting with a letter or digit"
        )
    return name


@dataclass(frozen=True)
class Measurement:
    """What `measurement.json` holds. `rate` is `None` when no trace was graded."""

    mode: str
    batch: str
    judge_version: int
    judge_yaml_sha256: str
    verdicts_sha256: str
    test_matrix: ConfusionMatrix
    n_traces: int
    n_graded: int
    n_set_aside: int
    flagged: int
    rate: CorrectedRate | None
    draws: int
    seed: int
    notes: tuple[str, ...]
    created: str

    def to_json(self) -> dict[str, Any]:
        rate = self.rate
        return {
            "mode": self.mode,
            "batch": self.batch,
            "judge_version": self.judge_version,
            "judge_yaml_sha256": self.judge_yaml_sha256,
            "verdicts_sha256": self.verdicts_sha256,
            "test_matrix": {
                "tp": self.test_matrix.tp,
                "fn": self.test_matrix.fn,
                "tn": self.test_matrix.tn,
                "fp": self.test_matrix.fp,
            },
            "n_traces": self.n_traces,
            "n_graded": self.n_graded,
            "n_set_aside": self.n_set_aside,
            "flagged": self.flagged,
            "observed_rate": rate.observed if rate else None,
            "corrected_rate": None if rate is None or math.isnan(rate.point) else rate.point,
            "clipped": rate.clipped if rate else None,
            "interval": list(rate.interval) if rate and rate.interval else None,
            "interval_if_judge_exact": (
                list(rate.interval_if_judge_exact)
                if rate and rate.interval_if_judge_exact
                else None
            ),
            "draws": self.draws,
            "seed": self.seed,
            "notes": list(self.notes),
            "created": self.created,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Measurement:
        matrix = data["test_matrix"]
        rate = None
        if data["n_graded"] > 0:
            point = data["corrected_rate"]
            rate = CorrectedRate(
                observed=float(data["observed_rate"]),
                point=math.nan if point is None else float(point),
                interval=_pair(data["interval"]),
                interval_if_judge_exact=_pair(data["interval_if_judge_exact"]),
                clipped=data["clipped"],
                draws=int(data["draws"]),
                seed=int(data["seed"]),
            )
        return cls(
            mode=str(data["mode"]),
            batch=str(data["batch"]),
            judge_version=int(data["judge_version"]),
            judge_yaml_sha256=str(data["judge_yaml_sha256"]),
            verdicts_sha256=str(data["verdicts_sha256"]),
            test_matrix=ConfusionMatrix(
                tp=int(matrix["tp"]), fn=int(matrix["fn"]),
                tn=int(matrix["tn"]), fp=int(matrix["fp"]),
            ),
            n_traces=int(data["n_traces"]),
            n_graded=int(data["n_graded"]),
            n_set_aside=int(data["n_set_aside"]),
            flagged=int(data["flagged"]),
            rate=rate,
            draws=int(data["draws"]),
            seed=int(data["seed"]),
            notes=tuple(str(n) for n in data["notes"]),
            created=str(data["created"]),
        )


def _pair(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    low, high = value
    return float(low), float(high)


def _fmt(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.3f}"


def _fmt_interval(interval: tuple[float, float] | None) -> str:
    return "[n/a]" if interval is None else f"[{interval[0]:.3f}, {interval[1]:.3f}]"


def summary_lines(m: Measurement) -> list[str]:
    """The two lines `measure` prints and `gate --batch` repeats."""
    head = f"{m.mode} v{m.judge_version} on {m.batch}"
    if m.rate is None:
        return [f"{head}: no rate, all {m.n_traces} traces set aside (see errors.jsonl)"]
    rate = m.rate
    return [
        (
            f"{head}: observed {rate.observed:.3f} -> corrected {_fmt(rate.point)} "
            f"{_fmt_interval(rate.interval)}"
        ),
        (
            f"  (judge exact: {_fmt_interval(rate.interval_if_judge_exact)}), "
            f"n={m.n_graded} graded, {m.n_set_aside} set aside"
        ),
    ]


def _check_batch(batch: list[Trace]) -> None:
    if not batch:
        raise AxialError("the batch is empty; there is nothing to measure")
    repeated = sorted(tid for tid, n in Counter(t.trace_id for t in batch).items() if n > 1)
    if repeated:
        raise AxialError(
            f"the batch repeats {len(repeated)} trace id(s) ({', '.join(repeated[:3])}); "
            "each would be counted more than once"
        )


def _test_matrix(store: Store, mode: str, version: int) -> ConfusionMatrix:
    rows = [
        r for r in store.read_jsonl(store.judge_dir(mode, version) / "verdicts.jsonl")
        if r.get("split") == "test"
    ]
    matrix = confusion_matrix([bool(r["truth"]) for r in rows], [bool(r["verdict"]) for r in rows])
    for count, name in ((matrix.tp + matrix.fn, "failures"), (matrix.tn + matrix.fp, "passes")):
        if count == 0:
            raise AxialError(
                f"judge {mode} v{version}'s test split has no real {name}, so its error rates "
                "are undefined and no flag rate can be corrected with them"
            )
    return matrix


def _notes(
    test: ConfusionMatrix, rate: CorrectedRate | None, n_traces: int, n_set_aside: int
) -> list[str]:
    notes = [ASSUMPTION_NOTE]
    if n_set_aside > SET_ASIDE_NOTE_SHARE * n_traces:
        notes.append(
            f"{n_set_aside} of {n_traces} traces ({n_set_aside / n_traces:.0%}) were set "
            "aside (see errors.jsonl); if what set them aside correlates with failure, as "
            "truncation on long outputs can, the rate is biased"
        )
    if rate is None:
        notes.append("every trace was set aside, so no rate was estimated")
        return notes
    if math.isnan(rate.point):
        notes.append(
            "the judge's test-split TPR + TNR - 1 is at most 0.05, no better than chance, "
            "so its flags cannot be corrected into a failure rate"
        )
    elif rate.clipped == "low":
        false_alarm_rate = test.fp / (test.tn + test.fp)
        notes.append(
            f"the corrected rate is clipped to 0: the observed rate {rate.observed:.3f} is "
            f"below the judge's false-alarm rate (1 - TNR = {false_alarm_rate:.3f}), so the "
            "batch is consistent with no failures at all"
        )
    elif rate.clipped == "high":
        tpr = test.tp / (test.tp + test.fn)
        notes.append(
            f"the corrected rate is clipped to 1: the observed rate {rate.observed:.3f} is "
            f"above the judge's TPR ({tpr:.3f})"
        )
    if rate.observed == 0.0:
        notes.append(SATURATION_WARNING)
    return notes


def measure_batch(
    store: Store,
    mode: str,
    version: int,
    batch_name: str,
    batch: list[Trace],
    build_model: Callable[[JudgeSpec], Model],
    *,
    replace: bool = False,
    max_tokens: int = 512,
    draws: int = 20_000,
    seed: int = 0,
) -> Measurement:
    """Grade `batch` with promoted judge `mode` v`version` and write the measurement.

    Nothing is written until every trace has been graded, and then all four files
    at once, so an interrupted run leaves no partial measurement behind.
    """
    hashes = require_promoted(store, mode, version)
    directory = store.measurement_dir(mode, validate_batch_name(batch_name))
    if directory.exists() and not replace:
        raise AxialError(
            f"a measurement named {batch_name!r} already exists for {mode} at {directory}. "
            "Re-running costs money and replaces evidence: pass --replace if you mean "
            "to, or choose another --name."
        )
    _check_batch(batch)
    test = _test_matrix(store, mode, version)
    spec = JudgeSpec.load(store, mode, version)
    few_shot = resolve_few_shot(spec, store.read_traces(), read_labels(store, mode))
    splits = Splits.load(store, mode) if few_shot else None
    model = build_model(spec)

    few_shot_ids = set(spec.few_shot_ids)
    verdicts: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for trace in batch:
        key = input_sha256(trace.input)
        if trace.trace_id in few_shot_ids:
            errors.append(
                {
                    "trace_id": trace.trace_id,
                    "input_sha256": key,
                    "failure_class": "few_shot_example",
                    "detail": "this trace is one of the judge's few-shot examples, so its "
                    "answer is printed in its own prompt",
                }
            )
            continue
        outcome = grade_trace(spec, trace, few_shot, splits, model, max_tokens=max_tokens)
        if isinstance(outcome, Graded):
            verdicts.append(
                {
                    "trace_id": trace.trace_id,
                    "input_sha256": key,
                    "verdict": outcome.verdict,
                    "model": outcome.model,
                    "usage": outcome.usage,
                    "attempts": outcome.attempts,
                    "latency_s": outcome.latency_s,
                    "raw": outcome.raw,
                }
            )
        else:
            errors.append(
                {
                    "trace_id": trace.trace_id,
                    "input_sha256": key,
                    "failure_class": outcome.failure_class,
                    "detail": outcome.detail,
                }
            )

    flagged = sum(1 for v in verdicts if v["verdict"])
    rate = (
        corrected_rate(test, flagged, len(verdicts), draws=draws, seed=seed) if verdicts else None
    )
    measurement = Measurement(
        mode=mode,
        batch=batch_name,
        judge_version=version,
        judge_yaml_sha256=hashes.judge_yaml_sha256,
        verdicts_sha256=hashes.verdicts_sha256,
        test_matrix=test,
        n_traces=len(batch),
        n_graded=len(verdicts),
        n_set_aside=len(errors),
        flagged=flagged,
        rate=rate,
        draws=draws,
        seed=seed,
        notes=tuple(_notes(test, rate, len(batch), len(errors))),
        created=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    store.write_dir_atomically(
        directory,
        {
            "traces.jsonl": jsonl_text(t.model_dump() for t in batch),
            "verdicts.jsonl": jsonl_text(verdicts),
            "errors.jsonl": jsonl_text(errors),
            "measurement.json": json_text(measurement.to_json()),
        },
    )
    return measurement


def load_measurement(
    store: Store, mode: str, batch: str
) -> tuple[Measurement, list[dict[str, Any]]]:
    """Read a measurement and its graded rows, refusing anything that does not add up."""
    directory = store.measurement_dir(mode, validate_batch_name(batch))
    path = directory / "measurement.json"
    if not path.exists():
        raise AxialError(
            f"no measurement named {batch!r} for {mode}: run `axial measure {mode} "
            f"--version N --traces <file> --name {batch}` first"
        )
    try:
        measurement = Measurement.from_json(store.read_json(path))
        rows = store.read_jsonl(directory / "verdicts.jsonl")
        flagged = sum(1 for r in rows if r["verdict"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AxialError(
            f"{path} is not a readable measurement ({type(exc).__name__}: {exc}); re-run "
            "`axial measure` with --replace"
        ) from exc
    if (measurement.mode, measurement.batch) != (mode, batch) or (
        len(rows), flagged
    ) != (measurement.n_graded, measurement.flagged):
        raise AxialError(
            f"measurement {batch!r} for {mode} does not match its own verdicts.jsonl or its "
            "directory; it was edited by hand. Re-run `axial measure` with --replace"
        )
    return measurement, rows
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_measure.py -q`
Expected: 27 passed.

- [ ] **Step 6: Lint, type-check and commit**

Run: `uv run ruff check . && uv run mypy src/axial`
Expected: clean.

```bash
git add src/axial/measure.py tests/measure_helpers.py tests/test_measure.py
git -c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com commit -m "Measure a batch of new traces with a promoted judge" \
  -m "measure_batch grades unlabelled traces with the promoted judge.yaml and its train examples. It sets aside what cannot be scored, including a batch trace that is one of the judge's own few-shot examples, and writes traces, verdicts, errors and measurement.json under .axial/measurements in one atomic step. The judge directory and the labelled corpus are never touched." \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 7: Pair two measured batches by input

**Files:**
- Modify: `src/axial/measure.py` (a constant, and new code appended)
- Create: `tests/test_pairing.py`

**Interfaces:**
- Consumes: graded rows as `load_measurement` returns them. Each row has `input_sha256` and `verdict`.
- Produces:
  - `MIN_PAIRED_SHARE = 0.8`;
  - `Pairing(cells: tuple[int, int, int, int], pairs: int, unmatched_baseline: int, unmatched_candidate: int, share: float, paired: bool)`, frozen, with a `describe() -> str` method;
  - `pair_batches(baseline_rows, candidate_rows) -> Pairing`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pairing.py`:

```python
"""Pairing two measured batches by input."""

from __future__ import annotations

from pathlib import Path

import pytest
from measure_helpers import MEASURE_MODE, batch_traces, build_promoted_judge, fake_judge_builder

from axial.measure import input_sha256, load_measurement, measure_batch, pair_batches
from axial.store import Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    store = Store(tmp_path)
    build_promoted_judge(store)
    return store


def rows(verdicts: list[bool], *, start: int = 0) -> list[dict[str, object]]:
    return [
        {"input_sha256": input_sha256(f"case {start + i}"), "verdict": v}
        for i, v in enumerate(verdicts)
    ]


def test_batches_with_the_same_inputs_pair_and_count_each_cell():
    baseline = rows([True, True, False, False, False])
    candidate = rows([True, False, True, True, False])
    pairing = pair_batches(baseline, candidate)
    assert pairing.cells == (1, 1, 2, 1)
    assert (pairing.pairs, pairing.unmatched_baseline, pairing.unmatched_candidate) == (5, 0, 0)
    assert pairing.paired
    assert pairing.describe() == "paired on 5 shared inputs (0 baseline and 0 candidate traces unmatched)"


def test_an_input_graded_twice_in_one_batch_is_left_unpaired():
    baseline = rows([False] * 10)
    candidate = rows([False] * 10) + rows([True], start=0)
    pairing = pair_batches(baseline, candidate)
    assert pairing.pairs == 9
    assert (pairing.unmatched_baseline, pairing.unmatched_candidate) == (1, 2)
    assert pairing.paired  # 9 of 10 and 9 of 11 are both at least 80%


def test_below_eighty_percent_shared_the_comparison_is_unpaired():
    baseline = rows([False] * 10)
    candidate = rows([False] * 5) + rows([False] * 5, start=100)
    pairing = pair_batches(baseline, candidate)
    assert not pairing.paired
    assert pairing.describe() == "unpaired: only 50% of inputs shared"


def test_eighty_percent_of_each_batch_is_enough():
    baseline = rows([False] * 10)
    candidate = rows([False] * 8) + rows([False] * 2, start=100)
    assert pair_batches(baseline, candidate).paired


def test_set_aside_traces_never_pair(store: Store):
    for name, outputs in (("base", ["GOOD"] * 10), ("cand", ["GARBLED"] * 3 + ["GOOD"] * 7)):
        measure_batch(
            store, MEASURE_MODE, 1, name, batch_traces(outputs), fake_judge_builder(), draws=2000
        )
    _, base_rows = load_measurement(store, MEASURE_MODE, "base")
    _, cand_rows = load_measurement(store, MEASURE_MODE, "cand")
    pairing = pair_batches(base_rows, cand_rows)
    assert (pairing.pairs, pairing.unmatched_baseline, pairing.unmatched_candidate) == (7, 3, 0)
    assert not pairing.paired  # 7 of the baseline's 10 is below 80%
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_pairing.py -q`
Expected: collection error, `ImportError: cannot import name 'pair_batches'`.

- [ ] **Step 3: Implement**

In `src/axial/measure.py`, directly below `SET_ASIDE_NOTE_SHARE = 0.05`, add:

```python

# Two batches are compared pair by pair only if the pairs cover this share of each.
MIN_PAIRED_SHARE = 0.8
```

Append to the end of the file:

```python
@dataclass(frozen=True)
class Pairing:
    """How two measured batches line up by input.

    `cells` is `(n11, n10, n01, n00)` over the pairs: flagged in both, baseline only,
    candidate only, neither. `share` is the smaller of the two batches' paired shares.
    """

    cells: tuple[int, int, int, int]
    pairs: int
    unmatched_baseline: int
    unmatched_candidate: int
    share: float
    paired: bool

    def describe(self) -> str:
        if self.paired:
            return (
                f"paired on {self.pairs} shared inputs ({self.unmatched_baseline} baseline "
                f"and {self.unmatched_candidate} candidate traces unmatched)"
            )
        return f"unpaired: only {self.share:.0%} of inputs shared"


def _graded_once(rows: list[dict[str, Any]]) -> dict[str, bool]:
    """Each input graded exactly once, with its verdict. An input graded twice has no
    single partner to compare with, so it is left out."""
    counts = Counter(str(r["input_sha256"]) for r in rows)
    return {
        str(r["input_sha256"]): bool(r["verdict"])
        for r in rows
        if counts[str(r["input_sha256"])] == 1
    }


def pair_batches(
    baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> Pairing:
    """Match two batches' graded rows by `input_sha256`.

    Only graded rows are passed in, so a set-aside trace never pairs.
    """
    baseline = _graded_once(baseline_rows)
    candidate = _graded_once(candidate_rows)
    shared = baseline.keys() & candidate.keys()
    n11 = sum(1 for k in shared if baseline[k] and candidate[k])
    n10 = sum(1 for k in shared if baseline[k] and not candidate[k])
    n01 = sum(1 for k in shared if not baseline[k] and candidate[k])
    n00 = len(shared) - n11 - n10 - n01
    pairs = len(shared)
    share = (
        min(pairs / len(baseline_rows), pairs / len(candidate_rows))
        if baseline_rows and candidate_rows
        else 0.0
    )
    return Pairing(
        cells=(n11, n10, n01, n00),
        pairs=pairs,
        unmatched_baseline=len(baseline_rows) - pairs,
        unmatched_candidate=len(candidate_rows) - pairs,
        share=share,
        paired=pairs > 0 and share >= MIN_PAIRED_SHARE,
    )
```

- [ ] **Step 4: Run the tests, lint, type-check and commit**

Run: `uv run pytest tests/test_pairing.py tests/test_measure.py -q && uv run ruff check . && uv run mypy src/axial`
Expected: 5 + 27 passed, clean.

```bash
git add src/axial/measure.py tests/test_pairing.py
git -c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com commit -m "Pair two measured batches by input hash" \
  -m "An input graded exactly once in each batch is a pair. The comparison is paired when the pairs cover at least 80% of each batch's graded traces; set-aside traces have no verdict row, so they never pair." \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 8: `gate_batches`

**Files:**
- Modify: `src/axial/gate.py` (module docstring, imports, and new code appended)
- Create: `tests/test_gate_batches.py`

**Interfaces:**
- Consumes:
  - `load_measurement`, `summary_lines`, `pair_batches` and `Measurement` (Tasks 6–7);
  - `require_promoted` and `SATURATION_WARNING` (Task 3);
  - `corrected_difference_paired` and `corrected_difference_unpaired` (Task 2).
- Produces: `gate_batches(store, mode, batch, *, baseline_batch: str | None = None, max_rate: float | None = None, draws=20_000, seed=0) -> GateResult`.
  - It raises `InvariantViolation` on a stale measurement, a measurement with `n_graded = 0`, or two batches measured by different judge versions.
  - `GateResult.passed` is `False` on `REGRESSION`, on `ABOVE CEILING`, and on a needed interval that is `None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gate_batches.py`:

```python
"""`gate_batches`: gating on measured batches of new traces."""

from __future__ import annotations

from pathlib import Path

import pytest
from measure_helpers import (
    MEASURE_MODE,
    batch_traces,
    build_promoted_judge,
    fake_judge_builder,
    judge_reply,
)

from axial.config import Thresholds
from axial.dataset import Splits, read_labels
from axial.errors import InvariantViolation
from axial.gate import gate_batches
from axial.judge import JudgeSpec, run_judge
from axial.measure import measure_batch
from axial.model import FakeModel
from axial.report import alignment_from_verdicts, promote
from axial.store import Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    store = Store(tmp_path)
    build_promoted_judge(store)
    return store


def measure(store: Store, name: str, outputs: list[str], *, version: int = 1, start: int = 0) -> None:
    measure_batch(
        store, MEASURE_MODE, version, name, batch_traces(outputs, prefix=name, start=start),
        fake_judge_builder(), draws=4000,
    )


def run(store: Store, batch: str, **kw):
    return gate_batches(store, MEASURE_MODE, batch, draws=4000, **kw)


def flags(n_bad: int, n: int = 500) -> list[str]:
    return ["BAD"] * n_bad + ["GOOD"] * (n - n_bad)


# --- one batch ---------------------------------------------------------------------


def test_one_batch_without_options_prints_and_passes(store: Store):
    measure(store, "rel-1", flags(60))
    result = run(store, "rel-1")
    assert result.passed
    assert result.messages[0].startswith("merged v1 on rel-1: observed 0.120 -> corrected 0.072")


def test_max_rate_fails_above_the_ceiling_and_passes_below_it(store: Store):
    measure(store, "rel-1", flags(60))  # interval about [0.000, 0.128]
    above = run(store, "rel-1", max_rate=0.10)
    assert not above.passed
    assert any(m.startswith("ABOVE CEILING") for m in above.messages)
    below = run(store, "rel-1", max_rate=0.20)
    assert below.passed
    assert any(m.startswith("within the ceiling") for m in below.messages)


def test_a_batch_that_flagged_nothing_gets_the_saturation_warning(store: Store):
    measure(store, "rel-1", flags(0, 50))
    assert any(m.startswith("saturation warning") for m in run(store, "rel-1").messages)


def test_a_batch_with_nothing_graded_fails_the_gate(store: Store):
    measure(store, "rel-1", ["GARBLED"] * 5)
    with pytest.raises(InvariantViolation, match="graded no traces"):
        run(store, "rel-1")


# --- two batches -------------------------------------------------------------------


def test_a_paired_regression_fails_the_gate(store: Store):
    measure(store, "base", flags(40))
    measure(store, "cand", flags(60))  # the same 500 inputs, 20 newly flagged
    result = run(store, "cand", baseline_batch="base")
    assert not result.passed
    assert "paired on 500 shared inputs (0 baseline and 0 candidate traces unmatched)" in result.messages
    assert any(m.startswith("difference vs base: +0.046") for m in result.messages)
    assert any(m.startswith("REGRESSION") for m in result.messages)


def test_an_improvement_and_no_change_pass(store: Store):
    measure(store, "base", flags(40))
    measure(store, "cand", flags(60))
    measure(store, "same", flags(40))
    improvement = run(store, "base", baseline_batch="cand")
    assert improvement.passed and any(m.startswith("improvement") for m in improvement.messages)
    unchanged = run(store, "same", baseline_batch="base")
    assert unchanged.passed
    assert "no change detectable above the noise floor" in unchanged.messages


def test_batches_without_shared_inputs_are_compared_unpaired(store: Store):
    measure(store, "base", flags(60))
    measure(store, "cand", flags(80), start=10_000)
    result = run(store, "cand", baseline_batch="base")
    assert "unpaired: only 0% of inputs shared" in result.messages
    assert result.passed  # the unpaired interval for 60 vs 80 of 500 straddles zero


def test_batches_measured_by_different_judge_versions_are_refused(store: Store):
    build_promoted_judge(store, version=2)
    measure(store, "base", flags(40), version=1)
    measure(store, "cand", flags(40), version=2)
    with pytest.raises(InvariantViolation, match="mix a judge change with a system change"):
        run(store, "cand", baseline_batch="base")


def test_a_rise_both_rates_clip_away_is_still_called_and_explained(store: Store):
    measure(store, "base", flags(20))  # 0.040 and 0.052, both below 1 - TNR = 0.057
    measure(store, "cand", flags(26))
    result = run(store, "cand", baseline_batch="base")
    assert "corrected 0.000" in result.messages[0]
    assert any(m.startswith("REGRESSION") for m in result.messages)
    assert any("computed before clipping" in m for m in result.messages)


# --- stale measurements ------------------------------------------------------------


def rejudge(store: Store) -> None:
    """Re-run the judge on test, which deletes PROMOTED, as `axial judge` does."""
    spec = JudgeSpec.load(store, MEASURE_MODE, 1)
    splits = Splits.load(store, MEASURE_MODE)
    corpus = store.read_traces()
    run_judge(
        store, spec, [t for t in corpus if t.trace_id in set(splits.test)],
        read_labels(store, MEASURE_MODE), splits, FakeModel(judge_reply), split="test",
        corpus=corpus,
    )


def test_a_measurement_taken_before_the_judge_was_rejudged_fails(store: Store):
    measure(store, "rel-1", flags(60))
    rejudge(store)
    with pytest.raises(InvariantViolation) as caught:
        run(store, "rel-1")
    message = str(caught.value)
    assert "batch 'rel-1' was measured by merged v1" in message
    assert "has not been promoted" in message
    assert "--name rel-1 --replace" in message


def reword_and_repromote(store: Store) -> None:
    path = store.judge_dir(MEASURE_MODE, 1) / "judge.yaml"
    path.write_text(path.read_text() + "\n# reworded\n")
    rejudge(store)
    verdicts = store.read_jsonl(store.judge_dir(MEASURE_MODE, 1) / "verdicts.jsonl")
    promote(
        store, MEASURE_MODE, 1, alignment_from_verdicts(verdicts, "dev"),
        alignment_from_verdicts(verdicts, "test"), Thresholds(),
    )


def test_a_measurement_taken_before_the_judge_changed_and_was_repromoted_fails(store: Store):
    measure(store, "rel-1", flags(60))
    reword_and_repromote(store)
    with pytest.raises(InvariantViolation, match="measured by an earlier merged v1"):
        run(store, "rel-1")


def test_rejudging_to_identical_verdicts_and_repromoting_keeps_a_measurement_valid(store: Store):
    """The marker binds bytes: the same judge.yaml and verdicts promoted again is the same
    evidence, so the measurement still describes the judge on disk."""
    measure(store, "rel-1", flags(60))
    rejudge(store)
    build_promoted_judge(store)
    assert run(store, "rel-1").passed


def test_a_stale_baseline_fails_too(store: Store):
    measure(store, "base", flags(40))
    reword_and_repromote(store)
    measure(store, "cand", flags(40))
    with pytest.raises(InvariantViolation, match="batch 'base'"):
        run(store, "cand", baseline_batch="base")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_gate_batches.py -q`
Expected: collection error, `ImportError: cannot import name 'gate_batches'`.

- [ ] **Step 3: Implement**

In `src/axial/gate.py`, replace everything above `@dataclass` / `class GateResult:` (the module docstring and the imports) with:

```python
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

from dataclasses import dataclass, field
from typing import Any

from axial.align import (
    corrected_difference_paired,
    corrected_difference_unpaired,
    paired_bootstrap_ci,
)
from axial.errors import InvariantViolation
from axial.measure import Measurement, load_measurement, pair_batches, summary_lines
from axial.report import SATURATION_WARNING, require_promoted
from axial.store import Store
```

Append to the end of the file:

```python
def _load_fresh(
    store: Store, mode: str, batch: str
) -> tuple[Measurement, list[dict[str, Any]]]:
    """A measurement, refused unless the judge that took it is still promoted unchanged
    and it graded at least one trace."""
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
    return measurement, rows


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
        if candidate.rate.clipped or baseline.rate.clipped:
            messages.append(
                "note: a rate above is clipped to [0, 1] for display; the difference was "
                "computed before clipping, so it can be non-zero while both rates show 0.000"
            )

    return GateResult(passed=passed, messages=messages)
```

- [ ] **Step 4: Run the tests, lint, type-check and commit**

Run: `uv run pytest tests/test_gate_batches.py tests/test_gate.py -q && uv run ruff check . && uv run mypy src/axial`
Expected: all pass, clean.

```bash
git add src/axial/gate.py tests/test_gate_batches.py
git -c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com commit -m "Gate on measured batches: a ceiling on one, a regression check between two" \
  -m "gate_batches refuses a measurement taken by a judge that has since been re-judged or edited, refuses batches measured by different judge versions, and fails on any interval it needs but cannot compute. The difference is paired by input when the batches share their inputs, and computed before clipping." \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 9: The `measure` command and `gate --batch`

**Files:**
- Modify: `src/axial/cli.py`
- Create: `tests/test_cli_measure.py`
- Modify: `tests/test_capabilities_cli_failures.py` (the `test_gate_without_version_is_a_usage_error` test)
- Modify: `tests/test_capabilities_cli_tour.py`: rewrite the strict xfail `test_a_promoted_judge_can_grade_new_traces_for_the_gate`

**Interfaces:**
- Consumes: `measure_batch`, `summary_lines` (Task 6) and `gate_batches` (Task 8). It also uses the CLI's own module-level `build_judge_model`, which tests monkeypatch.
- CLI:
  - `axial measure <mode> --version N --traces <file> --name <batch> [--adapter A] [--replace] [--repo R]`. It exits 1 on any refusal, and also after writing a measurement with `n_graded = 0`.
  - `axial gate <mode> (--version N [--baseline M] | --batch B [--baseline-batch A] [--max-rate R]) [--repo R]`. Mixing the two modes, or a `--max-rate` outside (0, 1), is a `typer.BadParameter` with exit code 2.

- [ ] **Step 1: Write the failing CLI tests**

Create `tests/test_cli_measure.py`:

```python
"""`axial measure` and `axial gate --batch` through the CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from measure_helpers import MEASURE_MODE, build_promoted_judge, fake_judge_builder, write_batch
from typer.testing import CliRunner, Result

from axial import cli
from axial.store import Store

runner = CliRunner()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    build_promoted_judge(Store(tmp_path))
    monkeypatch.setattr(cli, "build_judge_model", fake_judge_builder())
    monkeypatch.chdir(tmp_path)
    return tmp_path


def axial(*args: str) -> Result:
    return runner.invoke(cli.app, list(args))


def assert_clean(result: Result, code: int, *fragments: str) -> None:
    assert result.exit_code == code, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit), result.exception
    for fragment in fragments:
        assert fragment in result.output, (fragment, result.output)


def measure(repo: Path, name: str, n_bad: int, n: int = 500, *more: str) -> Result:
    path = write_batch(repo / f"{name}.jsonl", ["BAD"] * n_bad + ["GOOD"] * (n - n_bad))
    return axial("measure", MEASURE_MODE, "--version", "1", "--traces", str(path), "--name", name, *more)


# --- measure ------------------------------------------------------------------------


def test_measure_prints_the_corrected_rate_and_its_notes(repo: Path):
    result = measure(repo, "rel-1", 60)
    assert_clean(
        result, 0,
        "merged v1 on rel-1: observed 0.120 -> corrected 0.072 [0.000, 0.128]",
        "(judge exact: [0.042, 0.107]), n=500 graded, 0 set aside",
        "note: the correction assumes the judge errs on this batch",
    )
    assert (repo / ".axial" / "measurements" / MEASURE_MODE / "rel-1" / "measurement.json").exists()


def test_measure_refuses_an_existing_name_and_replace_overrides(repo: Path):
    assert measure(repo, "rel-1", 60).exit_code == 0
    assert_clean(measure(repo, "rel-1", 60), 1, "already exists", "--replace")
    assert_clean(measure(repo, "rel-1", 80, 500, "--replace"), 0, "observed 0.160")


def test_measure_refuses_an_unpromoted_judge(repo: Path):
    (repo / ".axial" / "judges" / MEASURE_MODE / "v1" / "PROMOTED").unlink()
    assert_clean(measure(repo, "rel-1", 60), 1, "has not been promoted")


def test_measure_reports_an_unreadable_batch_file(repo: Path):
    result = axial("measure", MEASURE_MODE, "--version", "1", "--traces", "nope.jsonl", "--name", "x")
    assert_clean(result, 1, "could not read nope.jsonl")


def test_measure_exits_non_zero_after_writing_a_batch_nothing_could_grade(repo: Path):
    path = write_batch(repo / "garbled.jsonl", ["GARBLED"] * 3)
    result = axial("measure", MEASURE_MODE, "--version", "1", "--traces", str(path), "--name", "g")
    assert_clean(result, 1, "no rate, all 3 traces set aside", "every trace was set aside")
    assert (repo / ".axial" / "measurements" / MEASURE_MODE / "g" / "errors.jsonl").exists()


# --- gate --batch -------------------------------------------------------------------


def test_gate_batch_exit_codes(repo: Path):
    measure(repo, "base", 40)
    measure(repo, "cand", 60)
    assert_clean(axial("gate", MEASURE_MODE, "--batch", "cand"), 0, "corrected 0.072")
    assert_clean(
        axial("gate", MEASURE_MODE, "--batch", "cand", "--baseline-batch", "base"), 1,
        "paired on 500 shared inputs", "REGRESSION",
    )
    assert_clean(
        axial("gate", MEASURE_MODE, "--batch", "base", "--baseline-batch", "cand"), 0, "improvement"
    )
    assert_clean(axial("gate", MEASURE_MODE, "--batch", "cand", "--max-rate", "0.1"), 1, "ABOVE CEILING")
    assert_clean(axial("gate", MEASURE_MODE, "--batch", "cand", "--max-rate", "0.2"), 0, "within the ceiling")


def test_gate_batch_names_a_missing_measurement(repo: Path):
    assert_clean(axial("gate", MEASURE_MODE, "--batch", "nope"), 1, "no measurement named 'nope'")


@pytest.mark.parametrize(
    ("args", "fragments"),
    [
        ((), ("--version", "--batch")),
        (("--version", "1", "--batch", "x"), ("--version", "--batch")),
        (("--version", "1", "--max-rate", "0.1"), ("--max-rate", "need --batch")),
        (("--version", "1", "--baseline-batch", "x"), ("--baseline-batch", "need --batch")),
        (("--batch", "x", "--baseline", "1"), ("--baseline", "--baseline-batch")),
        (("--batch", "x", "--max-rate", "1.5"), ("strictly between 0 and 1",)),
        (("--batch", "x", "--max-rate", "0"), ("strictly between 0 and 1",)),
    ],
)
def test_gate_option_mistakes_are_usage_errors(repo: Path, args: tuple[str, ...], fragments: tuple[str, ...]):
    assert_clean(axial("gate", MEASURE_MODE, *args), 2, *fragments)
```

- [ ] **Step 2: Update the two capability tests the spec changes on purpose**

In `tests/test_capabilities_cli_failures.py`, replace

```python
def test_gate_without_version_is_a_usage_error() -> None:
    """`gate` requires `--version`; omitting it is a usage error that names the option."""
    result = run("gate", "m")
    assert result.exit_code == 2
    assert_clean_failure(result, "Missing option '--version'")
```

with

```python
def test_gate_without_version_or_batch_is_a_usage_error() -> None:
    """`gate` needs exactly one of `--version` and `--batch`; giving neither is a usage
    error that names both."""
    result = run("gate", "m")
    assert result.exit_code == 2
    assert_clean_failure(result, "--version", "--batch")
```

In `tests/test_capabilities_cli_tour.py`, replace the whole strict-xfail test. That is everything from its `@pytest.mark.xfail(` decorator down to, but not including, `def test_split_is_stratified_seeded_and_frozen(`:

```python
@pytest.mark.xfail(
    strict=True,
    reason="no path grades new, unlabelled traces with a promoted judge: run_judge "
    "quarantines every unlabelled trace as an error and deletes PROMOTED, and `judge "
    "--split` only accepts train/dev/test. `gate` can therefore only ever re-read the "
    "judge's own alignment verdicts, so it never gates new system output and its "
    "saturation warning is unreachable for a judge that cleared the TPR bar.",
)
def test_a_promoted_judge_can_grade_new_traces_for_the_gate(promoted_repo: Path) -> None:
    """A promoted judge can score fresh production traces and `gate` can pass on them.

    README: "`axial promote` | alignment on test | a judge cleared for CI" /
    "`axial gate` | promoted judges | an exit code"
    """
    from axial.dataset import Splits, read_labels
    from axial.judge import JudgeSpec, run_judge
    from axial.store import Store
    from axial.traces import Trace

    store = Store(promoted_repo)
    fresh = [
        Trace(trace_id=f"new-{i}", input=f"THE WIRE — issue {1000 + i}\n\nA story.",
              output='{"stories": [{"headline": "a"}, {"headline": "b"}]}')
        for i in range(40)
    ]
    result = run_judge(
        store, JudgeSpec.load(store, MODE, 1), fresh, read_labels(store, MODE),
        Splits.load(store, MODE), FakeModel(perfect_judge), split="gate",
        corpus=store.read_traces() + fresh,
    )
    assert len(result.verdicts) == 40
    assert (judge_dir(promoted_repo) / "PROMOTED").exists()
    gate = axial_raw("gate", MODE, "--version", "1")
    assert gate.exit_code == 0 and "saturation warning" in gate.output
```

with the same test without its marker, now going through `measure` and `gate --batch`:

```python
def test_a_promoted_judge_can_grade_new_traces_for_the_gate(
    promoted_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A promoted judge scores fresh traces with `measure`, and `gate --batch` passes on them.

    README: "`axial measure` | a promoted judge + a batch of new traces | a corrected
    failure rate" / "`axial gate` | promoted judges, or measured batches | an exit code"
    """
    use_fake_judge(monkeypatch, perfect_judge)
    fresh = promoted_repo / "fresh.jsonl"
    fresh.write_text(
        "".join(
            json.dumps({
                "trace_id": f"new-{i}",
                "input": f"THE WIRE — issue {1000 + i}\n\nA story.",
                "output": '{"stories": [{"headline": "a"}, {"headline": "b"}]}',
            }) + "\n"
            for i in range(40)
        )
    )
    promoted_before = (judge_dir(promoted_repo) / "PROMOTED").read_bytes()

    measured = axial_raw("measure", MODE, "--version", "1", "--traces", str(fresh), "--name", "fresh")
    assert measured.exit_code == 0, measured.output
    assert "n=40 graded, 0 set aside" in measured.output
    assert (judge_dir(promoted_repo) / "PROMOTED").read_bytes() == promoted_before

    gate = axial_raw("gate", MODE, "--batch", "fresh")
    assert gate.exit_code == 0 and "saturation warning" in gate.output
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/test_cli_measure.py tests/test_capabilities_cli_failures.py tests/test_capabilities_cli_tour.py -q`
Expected: `test_cli_measure.py` fails with `No such command 'measure'` and usage errors; the tour test fails the same way; the failures test fails on the old message.

- [ ] **Step 4: Implement**

In `src/axial/cli.py`, add these imports in sorted position:

```python
from axial.gate import gate_batches
from axial.measure import measure_batch, summary_lines
```

`from axial.gate import gate_batches` goes on its own line below `from axial.gate import gate as run_gate`.

Below `SessionsArgument`, add:

```python
# Same B008 false positive, for `measure`'s batch file.
BatchTracesOption = typer.Option(
    ..., "--traces", help="JSONL or CSV of new traces to measure; never written to .axial/traces"
)
```

Replace the whole `gate` command, from `@app.command()` / `def gate(` to the end of the file, with:

```python
@app.command()
def measure(
    mode: str,
    version: int = typer.Option(..., "--version", help="Promoted judge version to grade with"),
    traces: Path = BatchTracesOption,
    name: str = typer.Option(
        ..., "--name", help="Batch name; the measurement goes to .axial/measurements/<mode>/<name>/"
    ),
    adapter: str | None = typer.Option(None, "--adapter", help="Name of a registered adapter"),
    replace: bool = typer.Option(
        False, "--replace", help="Replace an existing measurement with the same name"
    ),
    repo: str = RepoOption,
) -> None:
    """Grade new traces with a promoted judge and report a corrected failure rate."""
    store = Store(repo)
    load_env(repo)
    try:
        batch = do_ingest(traces, adapter=adapter)
    except (OSError, ValueError) as exc:
        _fail(f"could not read {traces}: {exc}")
    try:
        result = measure_batch(
            store, mode, version, name, batch, build_judge_model, replace=replace
        )
    except AxialError as exc:
        _fail(str(exc))
    for line in summary_lines(result):
        typer.echo(line)
    for note in result.notes:
        typer.echo(f"note: {note}")
    if result.rate is None:
        raise typer.Exit(code=1)


@app.command()
def gate(
    mode: str,
    version: int | None = typer.Option(
        None, "--version", help="Promoted judge version to gate on (compares judge versions)"
    ),
    baseline: int | None = typer.Option(
        None, "--baseline", help="With --version: judge version to compare against"
    ),
    batch: str | None = typer.Option(
        None, "--batch", help="Measured batch to gate on (see `axial measure`)"
    ),
    baseline_batch: str | None = typer.Option(
        None, "--baseline-batch", help="With --batch: measured batch to compare against"
    ),
    max_rate: float | None = typer.Option(
        None,
        "--max-rate",
        help="With --batch: fail if the upper end of the corrected rate's 95% interval "
        "is above this",
    ),
    repo: str = RepoOption,
) -> None:
    """Fail the build on a regression outside the noise floor."""
    if (version is None) == (batch is None):
        raise typer.BadParameter(
            "give exactly one of --version (compare judge versions) or --batch (gate "
            "measured batches)"
        )
    if version is not None and (baseline_batch is not None or max_rate is not None):
        raise typer.BadParameter("--baseline-batch and --max-rate need --batch")
    if batch is not None and baseline is not None:
        raise typer.BadParameter(
            "--baseline compares judge versions and needs --version; to compare measured "
            "batches use --baseline-batch"
        )
    if max_rate is not None and not 0 < max_rate < 1:
        raise typer.BadParameter("--max-rate must lie strictly between 0 and 1")
    store = Store(repo)
    try:
        if version is not None:
            result = run_gate(store, mode, version, baseline_version=baseline)
        else:
            assert batch is not None  # exactly one of the two was checked above
            result = gate_batches(
                store, mode, batch, baseline_batch=baseline_batch, max_rate=max_rate
            )
    except AxialError as exc:
        _fail(str(exc))
    for message in result.messages:
        typer.echo(message)
    if not result.passed:
        raise typer.Exit(code=1)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q`
Expected: everything passes, with 48 xfailed: one fewer, because the tour xfail is gone.

- [ ] **Step 6: Lint, type-check and commit**

Run: `uv run ruff check . && uv run mypy src/axial`
Expected: clean.

```bash
git add src/axial/cli.py tests/test_cli_measure.py tests/test_capabilities_cli_failures.py tests/test_capabilities_cli_tour.py
git -c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com commit -m "Add axial measure and gate --batch to the CLI" \
  -m "gate takes exactly one of --version and --batch. --baseline-batch and --max-rate belong to batch mode and --baseline to version mode, and mixing them is a usage error. The capability tour's strict xfail on grading new traces now passes through measure and gate --batch, and loses its marker." \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 10: The dev disagreement listing

**Files:**
- Modify: `src/axial/report.py`
- Create: `tests/test_disagreements.py`

**Interfaces:**
- Produces:
  - `Disagreement(trace_id, kind: Literal["missed_failure", "false_alarm"], critique, annotator, judge_reason, input_excerpt: str | None, input_length: int | None, output_excerpt: str | None, output_length: int | None)`, frozen;
  - `parse_reason(raw: str) -> str`;
  - `find_disagreements(verdicts, labels: list[Label], traces: list[Trace], split="dev") -> list[Disagreement]`, which raises `InvariantViolation` for `split="test"`;
  - `disagreement_counts(disagreements) -> str`;
  - `write_alignment(..., disagreements: list[Disagreement] | None = None)`. With `None`, both files are byte-identical to today's.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_disagreements.py`. The CLI test is added in Task 11.

``````python
"""The dev disagreement listing in alignment.md and alignment.json."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from axial.align import align
from axial.dataset import Label
from axial.errors import InvariantViolation
from axial.report import (
    disagreement_counts,
    find_disagreements,
    parse_reason,
    write_alignment,
)
from axial.store import Store
from axial.traces import Trace


def row(trace_id: str, truth: bool, verdict: bool, *, split: str = "dev", raw: str | None = None):
    if raw is None:
        raw = json.dumps({"verdict": "fail" if verdict else "pass", "reason": f"why {trace_id}"})
    return {"trace_id": trace_id, "split": split, "truth": truth, "verdict": verdict, "raw": raw}


def label(trace_id: str, value: bool, critique: str = "a critique") -> Label:
    return Label(trace_id=trace_id, label=value, critique=critique, annotator="andres")


def trace(trace_id: str, text: str = "short") -> Trace:
    return Trace(trace_id=trace_id, input=f"in {text}", output=f"out {text}")


VERDICTS = [
    row("d4", truth=False, verdict=True),   # false alarm
    row("d2", truth=True, verdict=False),   # missed failure
    row("d1", truth=True, verdict=True),    # agreement
    row("d3", truth=False, verdict=False),  # agreement
    row("d0", truth=True, verdict=False),   # missed failure
    row("t1", truth=True, verdict=False, split="test"),
]
LABELS = [label(t, t in ("d0", "d1", "d2", "t1")) for t in ("d0", "d1", "d2", "d3", "d4", "t1")]
TRACES = [trace(t) for t in ("d0", "d1", "d2", "d3", "d4", "t1")]


# --- finding them ------------------------------------------------------------------


def test_only_disagreements_are_listed_missed_failures_first_then_by_id():
    found = find_disagreements(VERDICTS, LABELS, TRACES)
    assert [(d.kind, d.trace_id) for d in found] == [
        ("missed_failure", "d0"),
        ("missed_failure", "d2"),
        ("false_alarm", "d4"),
    ]
    assert found[0].critique == "a critique" and found[0].annotator == "andres"
    assert found[0].judge_reason == "why d0"


def test_test_split_disagreements_are_never_listed():
    assert all(d.trace_id != "t1" for d in find_disagreements(VERDICTS, LABELS, TRACES))
    with pytest.raises(InvariantViolation, match="counted, never listed"):
        find_disagreements(VERDICTS, LABELS, TRACES, split="test")


def test_a_missing_critique_shows_a_placeholder():
    found = find_disagreements([row("d0", True, False)], [label("d0", True, "  ")], [trace("d0")])
    assert found[0].critique == "(no critique recorded)"


def test_the_judge_reason_comes_from_reason_then_critique_then_the_raw_text():
    assert parse_reason('{"verdict": "fail", "reason": "two headlines"}') == "two headlines"
    assert parse_reason('Sure: {"verdict": "pass", "critique": "one story"}') == "one story"
    assert parse_reason('{"verdict": "pass", "reason": ""}') == '{"verdict": "pass", "reason": ""}'
    assert parse_reason("no json here") == "no json here"
    assert parse_reason("x" * 400) == "x" * 300 + "…"
    assert parse_reason("") == "(no reason recorded)"


def test_excerpts_are_the_first_300_characters_with_the_full_length():
    long_trace = Trace(trace_id="d0", input="i" * 2184, output="o" * 64)
    found = find_disagreements([row("d0", True, False)], [label("d0", True)], [long_trace])
    d = found[0]
    assert (d.input_excerpt, d.input_length) == ("i" * 300, 2184)
    assert (d.output_excerpt, d.output_length) == ("o" * 64, 64)


def test_a_trace_missing_from_the_corpus_has_no_excerpt():
    d = find_disagreements([row("d0", True, False)], [label("d0", True)], [])[0]
    assert (d.input_excerpt, d.input_length, d.output_excerpt) == (None, None, None)


def test_the_counts_line():
    assert disagreement_counts([]) == "dev: no disagreements"
    found = find_disagreements(VERDICTS, LABELS, TRACES)
    assert disagreement_counts(found) == "dev: 2 missed failures, 1 false alarm (listed in alignment.md)"


# --- rendering them ----------------------------------------------------------------


def perfect():
    y = [True] * 30 + [False] * 30
    return align(y, list(y), n_boot=200, seed=1)


def render(tmp_path: Path, disagreements) -> tuple[str, dict]:
    store = Store(tmp_path)
    path = write_alignment(store, "merged", 1, perfect(), None, n_errors=0, disagreements=disagreements)
    return path.read_text(), json.loads((path.parent / "alignment.json").read_text())


def test_the_markdown_lists_each_case_with_its_excerpts(tmp_path: Path):
    traces = [Trace(trace_id="d0", input="i" * 2184, output="o" * 64), trace("d2"), trace("d4")]
    md, _ = render(tmp_path, find_disagreements(VERDICTS, LABELS, traces))
    assert "## Dev disagreements" in md
    assert "### Missed failures (2)" in md and "### False alarms (1)" in md
    assert md.index("missed failure — `d2`") < md.index("false alarm — `d4`")
    assert "**Human** (annotator `andres`): a critique" in md
    assert "**Judge:** why d0" in md
    assert "Input (first 300 of 2,184 characters):\n\n```\n" + "i" * 300 + "\n```" in md
    assert "Output (all 64 characters):" in md
    assert "Test disagreements are counted in the table above and never listed here" in md


def test_a_backtick_run_in_an_excerpt_gets_a_longer_fence(tmp_path: Path):
    tricky = Trace(trace_id="d0", input="see ````code````", output="fine")
    md, _ = render(tmp_path, find_disagreements([row("d0", True, False)], [label("d0", True)], [tricky]))
    assert "\n`````\nsee ````code````\n`````\n" in md


def test_a_missing_trace_is_named_in_the_markdown(tmp_path: Path):
    md, _ = render(tmp_path, find_disagreements([row("d0", True, False)], [label("d0", True)], []))
    assert "Input: (trace not found in .axial/traces)" in md


def test_no_disagreements_says_so(tmp_path: Path):
    md, data = render(tmp_path, [])
    assert "## Dev disagreements\n\nNo dev disagreements.\n" in md
    assert data["disagreements"] == {"dev": []}


def test_alignment_json_lists_each_case_without_excerpts(tmp_path: Path):
    _, data = render(tmp_path, find_disagreements(VERDICTS, LABELS, TRACES))
    assert data["disagreements"]["dev"][0] == {
        "trace_id": "d0", "kind": "missed_failure", "critique": "a critique", "judge_reason": "why d0",
    }


def test_without_disagreements_the_report_is_byte_identical_to_before(tmp_path: Path):
    """The hashes are of the files `write_alignment` wrote for these exact inputs at commit
    3673962, before the listing existed. A change here changes every existing report."""
    y = [True] * 30 + [False] * 30
    test = align(y, [True] * 27 + [False] * 3 + [False] * 28 + [True] * 2, n_boot=200, seed=1)
    store = Store(tmp_path)
    path = write_alignment(store, "merged", 1, perfect(), test, n_errors=2, notes=["a note"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "548fb47a2ed4ffc3fb4a3139e3a128e29fbb12af6a9af321cdab2fd5a0096362"
    )
    assert hashlib.sha256((path.parent / "alignment.json").read_bytes()).hexdigest() == (
        "d256897b394a2afc023aa2c5d59ce7ff16a400a723d08849553df955bdb978be"
    )
``````

The two hashes in `test_without_disagreements_the_report_is_byte_identical_to_before` were computed from `write_alignment` at `3673962` for exactly those inputs. **Run this test once before Step 3:** it must already pass against the unchanged `write_alignment`. That proves the hashes describe today's output. Then keep it passing.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_disagreements.py -q`
Expected: collection error, `ImportError: cannot import name 'disagreement_counts'`. Temporarily comment out the new imports to check the byte-identity test on its own, as Step 1 says.

- [ ] **Step 3: Implement**

In `src/axial/report.py`, change the imports to:

```python
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple

from axial.align import Alignment, align
from axial.config import Thresholds
from axial.dataset import Label
from axial.errors import InvariantViolation
from axial.store import Store
from axial.traces import Trace
```

Insert directly above `def write_alignment(`:

```python
EXCERPT_CHARS = 300

DisagreementKind = Literal["missed_failure", "false_alarm"]


@dataclass(frozen=True)
class Disagreement:
    """One dev trace where the judge and the human label disagree.

    A missed failure (human fail, judge pass) lowers TPR; a false alarm (human pass,
    judge fail) lowers TNR. "False pass" and "false fail" are avoided because they flip
    meaning under the AI Evals course's Pass-is-positive convention.
    """

    trace_id: str
    kind: DisagreementKind
    critique: str
    annotator: str
    judge_reason: str
    input_excerpt: str | None
    input_length: int | None
    output_excerpt: str | None
    output_length: int | None


def parse_reason(raw: str) -> str:
    """The judge's stated reason, read from its stored raw answer.

    The JSON `reason` field, else a `critique` field, else the raw text cut to
    `EXCERPT_CHARS`. Reading it from `raw` at report time leaves `run_judge` and the
    hash-bound `verdicts.jsonl` alone, and gives runs made before this a listing too.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            for key in ("reason", "critique"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    text = raw.strip()
    if not text:
        return "(no reason recorded)"
    return text if len(text) <= EXCERPT_CHARS else text[:EXCERPT_CHARS] + "…"


def find_disagreements(
    verdicts: list[dict[str, Any]],
    labels: list[Label],
    traces: list[Trace],
    split: str = "dev",
) -> list[Disagreement]:
    """Every verdict on `split` that disagrees with its label, missed failures first,
    then false alarms, each sorted by trace id so the listing diffs cleanly."""
    if split == "test":
        raise InvariantViolation(
            "test disagreements are counted, never listed: reading them to revise the "
            "judge turns test into a second dev set and inflates the number measured on it"
        )
    label_by_id = {lbl.trace_id: lbl for lbl in labels}
    trace_by_id = {t.trace_id: t for t in traces}
    found: list[Disagreement] = []
    for row in verdicts:
        if row.get("split") != split or row.get("truth") is None:
            continue
        truth, verdict = bool(row["truth"]), bool(row["verdict"])
        if truth == verdict:
            continue
        trace_id = str(row["trace_id"])
        label = label_by_id.get(trace_id)
        trace = trace_by_id.get(trace_id)
        found.append(
            Disagreement(
                trace_id=trace_id,
                kind="missed_failure" if truth else "false_alarm",
                critique=(label.critique.strip() if label else "") or "(no critique recorded)",
                annotator=label.annotator if label else "(unknown)",
                judge_reason=parse_reason(str(row.get("raw", ""))),
                input_excerpt=trace.input[:EXCERPT_CHARS] if trace else None,
                input_length=len(trace.input) if trace else None,
                output_excerpt=trace.output[:EXCERPT_CHARS] if trace else None,
                output_length=len(trace.output) if trace else None,
            )
        )
    order = {"missed_failure": 0, "false_alarm": 1}
    return sorted(found, key=lambda d: (order[d.kind], d.trace_id))


def disagreement_counts(disagreements: list[Disagreement]) -> str:
    """The one line `axial align` prints."""
    if not disagreements:
        return "dev: no disagreements"
    missed = sum(1 for d in disagreements if d.kind == "missed_failure")
    alarms = len(disagreements) - missed
    return (
        f"dev: {missed} missed failure{'' if missed == 1 else 's'}, "
        f"{alarms} false alarm{'' if alarms == 1 else 's'} (listed in alignment.md)"
    )


def _fence(text: str) -> str:
    """A code fence one backtick longer than the longest backtick run in `text`, so
    trace text can never close it early."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _excerpt_lines(name: str, excerpt: str | None, length: int | None) -> list[str]:
    if excerpt is None or length is None:
        return [f"{name}: (trace not found in .axial/traces)", ""]
    shown = f"first {len(excerpt):,} of {length:,}" if length > len(excerpt) else f"all {length:,}"
    fence = _fence(excerpt)
    return [f"{name} ({shown} characters):", "", fence, excerpt, fence, ""]


def _disagreement_section(mode: str, disagreements: list[Disagreement]) -> list[str]:
    lines = ["## Dev disagreements", ""]
    if not disagreements:
        lines += ["No dev disagreements.", ""]
    else:
        lines += [
            "Each case is a dev trace where the judge and the human label disagree. A missed",
            "failure usually means the failure mode's definition or examples are too narrow;",
            "a false alarm, that they are too broad. Revise the judge, then run",
            f"`axial judge {mode} --split dev` again (with `--new` once this version has",
            "been scored on test).",
            "",
        ]
        for kind, heading in (("missed_failure", "Missed failures"), ("false_alarm", "False alarms")):
            group = [d for d in disagreements if d.kind == kind]
            if not group:
                continue
            lines += [f"### {heading} ({len(group)})", ""]
            for d in group:
                title = "missed failure" if d.kind == "missed_failure" else "false alarm"
                lines += [
                    f"#### {title} — `{d.trace_id}`",
                    "",
                    f"**Human** (annotator `{d.annotator}`): {_one_line(d.critique)}",
                    "",
                    f"**Judge:** {_one_line(d.judge_reason)}",
                    "",
                ]
                lines += _excerpt_lines("Input", d.input_excerpt, d.input_length)
                lines += _excerpt_lines("Output", d.output_excerpt, d.output_length)
    lines.append(
        "Test disagreements are counted in the table above and never listed here, so the "
        "test split stays a held-out measurement."
    )
    return lines
```

Replace `write_alignment`'s signature and its `store.write_json(...)` call, from `def write_alignment(` through the closing parenthesis of that `store.write_json` call, with:

```python
def write_alignment(
    store: Store,
    mode: str,
    version: int,
    dev: Alignment,
    test: Alignment | None,
    n_errors: int,
    notes: list[str] | None = None,
    disagreements: list[Disagreement] | None = None,
) -> Path:
    """Write alignment.json and alignment.md. With `disagreements=None` both files are
    exactly what they were before the dev listing existed."""
    directory = store.judge_dir(mode, version)
    directory.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "mode": mode,
        "version": version,
        "dev": dev.to_dict(),
        "test": test.to_dict() if test else None,
        "quarantined_errors": n_errors,
    }
    if disagreements is not None:
        report["disagreements"] = {
            "dev": [
                {
                    "trace_id": d.trace_id,
                    "kind": d.kind,
                    "critique": d.critique,
                    "judge_reason": d.judge_reason,
                }
                for d in disagreements
            ]
        }
    store.write_json(directory / "alignment.json", _json_safe(report))
```

In the body below, replace

```python
    if notes:
        lines += ["", "## Notes", ""]
        lines += [f"- {note}" for note in notes]
    lines += [""]
```

with

```python
    if notes:
        lines += ["", "## Notes", ""]
        lines += [f"- {note}" for note in notes]
    if disagreements is not None:
        lines += [""] + _disagreement_section(mode, disagreements)
    lines += [""]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_disagreements.py tests/test_report.py -q`
Expected: 13 + 23 passed.

- [ ] **Step 5: Lint, type-check and commit**

Run: `uv run ruff check . && uv run mypy src/axial`
Expected: clean.

```bash
git add src/axial/report.py tests/test_disagreements.py
git -c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com commit -m "List dev disagreements in the alignment report" \
  -m "Missed failures first, then false alarms, each with the human's critique, the judge's stated reason read from its stored raw answer, and fenced 300-character excerpts. Test disagreements are counted, never listed. Without the new argument the report is byte-identical to before." \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 11: `axial align` writes the listing and prints the counts

**Files:**
- Modify: `src/axial/cli.py` (imports and the `align` command)
- Modify: `tests/test_disagreements.py` (imports, and one test appended)

**Interfaces:**
- Consumes: `find_disagreements`, `disagreement_counts` and `write_alignment(..., disagreements=)` (Task 10).

- [ ] **Step 1: Write the failing test**

In `tests/test_disagreements.py`, add `from measure_helpers import MEASURE_MODE, build_promoted_judge` and `from typer.testing import CliRunner` after `import pytest`, and `from axial import cli` before `from axial.align import align`. Then append:

```python
# --- through the CLI -----------------------------------------------------------------


def test_axial_align_prints_the_counts_and_writes_the_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = Store(tmp_path)
    build_promoted_judge(store)
    path = store.judge_dir(MEASURE_MODE, 1) / "verdicts.jsonl"
    rows = store.read_jsonl(path)
    dev = [r for r in rows if r["split"] == "dev"]
    dev[0]["verdict"] = not dev[0]["verdict"]  # dev-00 is a real failure: now missed
    dev[1]["verdict"] = not dev[1]["verdict"]  # dev-01 is a real pass: now a false alarm
    store.write_jsonl(path, rows)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli.app, ["align", MEASURE_MODE, "1"])

    assert result.exit_code == 0, result.output
    assert "dev: 1 missed failure, 1 false alarm (listed in alignment.md)" in result.output
    md = (store.judge_dir(MEASURE_MODE, 1) / "alignment.md").read_text()
    assert "missed failure — `dev-00`" in md and "false alarm — `dev-01`" in md
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_disagreements.py -q -k axial_align`
Expected: 1 failed: the counts line is missing from the output.

- [ ] **Step 3: Implement**

In `src/axial/cli.py`, replace the `axial.report` import that is not `promote` with:

```python
from axial.report import (
    alignment_from_verdicts,
    check_promotion,
    disagreement_counts,
    find_disagreements,
    write_alignment,
)
```

Replace the `align` command with:

```python
@app.command()
def align(mode: str, version: int, repo: str = RepoOption) -> None:
    """Report agreement between a judge and your labels."""
    store = Store(repo)
    config = load_config(store)
    directory = store.judge_dir(mode, version)
    verdicts = store.read_jsonl(directory / "verdicts.jsonl")
    errors = store.read_jsonl(directory / "errors.jsonl")
    dev = alignment_from_verdicts(verdicts, "dev")
    test = None
    notes: list[str] = []
    if any(v["split"] == "test" for v in verdicts):
        test = alignment_from_verdicts(verdicts, "test")
        # `check_promotion` is the only place that decides a gap check was
        # skipped (e.g. because a rate is nan on dev or test); reuse its notes
        # here so that a skipped check is recorded in alignment.md rather than
        # only surfacing later, at `promote` time.
        notes = check_promotion(dev, test, config.thresholds).notes
    disagreements = find_disagreements(verdicts, read_labels(store, mode), store.read_traces())
    path = write_alignment(
        store, mode, version, dev, test,
        n_errors=len(errors), notes=notes, disagreements=disagreements,
    )
    typer.echo(f"dev  {dev.summary()}")
    if test:
        typer.echo(f"test {test.summary()}")
    typer.echo(disagreement_counts(disagreements))
    typer.echo(f"written to {path}")
```

- [ ] **Step 4: Run everything, lint, type-check and commit**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy src/axial`
Expected: everything passes, 48 xfailed, clean.

```bash
git add src/axial/cli.py tests/test_disagreements.py
git -c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com commit -m "Print the dev disagreement counts from axial align" \
  -m "align now passes the dev disagreements to write_alignment and prints how many of each kind it listed in alignment.md." \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 12: Documentation

**Files:**
- Modify: `README.md`
- Modify: `knowledge/procedures/evals-in-ci.md`
- Modify: `knowledge/procedures/production-monitoring.md`
- Modify: `llms.txt` (regenerated, never hand-edited)
- Modify: `tests/test_capabilities_cli_tour.py` (the README command count)

- [ ] **Step 1: Make the README test expect the new pipeline row**

In `tests/test_capabilities_cli_tour.py`, in `test_every_readme_command_is_on_the_installed_console_script`, change `assert len(commands) == 12` to `assert len(commands) == 13`.

Run: `uv run pytest tests/test_capabilities_cli_tour.py -q -k readme_command`
Expected: 1 failed, `assert 12 == 13`.

- [ ] **Step 2: README**

Replace

```markdown
read from the repo root by `axial taxonomy` and `axial judge`,
```

with

```markdown
read from the repo root by `axial taxonomy`, `axial judge` and `axial measure`,
```

Replace

```markdown
| `axial promote` | alignment on test | a judge cleared for CI |
| `axial gate` | promoted judges | an exit code |
```

with

```markdown
| `axial promote` | alignment on test | a judge cleared for CI |
| `axial measure` | a promoted judge + a batch of new traces | the batch's failure rate, corrected for the judge's errors, with a 95% interval |
| `axial gate` | promoted judges, or measured batches | an exit code |
```

Replace the gate invariant bullet

```markdown
- **Gates compare intervals, not point estimates — when a `--baseline` is given.**
  Without one, `gate` only prints the failure rate and always passes; pass a
  baseline version to get a paired confidence interval and an actual regression
  check. Either way, it warns when a suite stops failing anything — a saturated
  eval has stopped carrying signal.
```

with

```markdown
- **Gates compare intervals, not point estimates.** `gate --batch` gates on batches
  that `measure` graded: `--max-rate` fails when the upper end of a batch's corrected
  failure rate interval is above the ceiling, and `--baseline-batch` fails when the
  interval on the difference from a baseline batch excludes zero on the side of more
  failures. The difference is paired by input when the two batches share at least
  80% of their inputs. `gate --version` compares a judge's own verdicts across judge
  versions with `--baseline`; without a baseline or a ceiling, either mode only
  prints and passes. Both warn when a suite stops failing anything — a saturated
  eval has stopped carrying signal.
- **A corrected rate comes only from a promoted judge's test-split error rates,
  and every measurement states the assumption it rests on:** that the judge errs on
  the new batch as it did on test. A measurement taken by a judge that has since
  been re-judged or edited fails the gate until it is re-measured.
```

In the Quickstart, directly after the code block that ends `uv run axial gate merged_stories --version 1`, insert:

````markdown
Once promoted, the judge grades new output. Put a batch of the system's traces in a
JSONL or CSV file (it is never merged into the labelled corpus), measure it, and gate
on the result:

```bash
uv run axial measure merged_stories --version 1 --traces release-1.jsonl --name release-1
uv run axial gate merged_stories --batch release-1 --max-rate 0.10
# after the next change, run the same inputs through it and compare:
uv run axial measure merged_stories --version 1 --traces release-2.jsonl --name release-2
uv run axial gate merged_stories --batch release-2 --baseline-batch release-1
```

`measure` reports the flag rate corrected for the judge's test-split TPR and TNR,
with an interval that includes the judge's own uncertainty, and writes everything to
`.axial/measurements/<mode>/<name>/`. Sizing a comparison: with a judge at TPR 0.93
and TNR 0.94 on 30 real failures and 70 real passes, catching a rise from 10% to 14%
failing reliably takes about 1,000 inputs run through both versions, or about 500 if
most inputs whose behaviour did not change produce identical output. A weaker judge
needs more. Unrelated batches, compared unpaired, need about twice as many.
````

The sizing numbers come from the spec's section 2 simulation. Keep the positive-class sentence in the invariants as it is.

- [ ] **Step 3: Knowledge procedures**

In `knowledge/procedures/evals-in-ci.md`, replace

```markdown
   someone disables it. axial's `gate` compares a paired bootstrap interval against a
   baseline. A batch mode, `gate --batch`, comparing a corrected failure rate against a
   maximum or against a baseline batch, is proposed in axial's corrected-rates design
   and not yet implemented.
```

with

```markdown
   someone disables it. axial's `measure` grades the CI inputs with a promoted judge
   and corrects the flag rate for the judge's test-split errors; `gate --batch` then
   fails on a corrected rate whose interval reaches above a ceiling (`--max-rate`), or
   on a rise against a baseline batch (`--baseline-batch`). Running the same inputs
   through both versions lets the comparison pair them, which needs about half as many
   inputs as comparing two unrelated batches.
```

In `knowledge/procedures/production-monitoring.md`, replace

```markdown
   neither the failure rate nor an interval; the correction and interval are what
   axial's `measure` would add, which is proposed in axial's corrected-rates design and
   not yet implemented. Evidently's scheduled evals and
   batch monitoring jobs run the equivalent for its own descriptors.
```

with

```markdown
   neither the failure rate nor an interval. axial's `measure` adds both: it corrects
   the batch's flag rate with the judge's test-split error rates and reports a 95%
   interval that includes the judge's own uncertainty, plus the interval the batch
   alone would give if the judge's rates were exact. Evidently's scheduled evals and
   batch monitoring jobs run the equivalent for its own descriptors.
```

Set `updated: ` to the day of the change in both files' frontmatter, then regenerate the index:

Run: `uv run python scripts/knowledge_index.py && uv run python scripts/knowledge_index.py --check`
Expected: the second command exits 0; `llms.txt` shows the two new dates.

- [ ] **Step 4: Run everything and commit**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy src/axial`
Expected: everything passes.

```bash
git add README.md knowledge/procedures/evals-in-ci.md knowledge/procedures/production-monitoring.md llms.txt tests/test_capabilities_cli_tour.py
git -c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com commit -m "Document measure and gate --batch, with sizing for a paired comparison" \
  -m "The README gains a measure row, a gate invariant that covers both modes, the corrected-rate invariant, and a Quickstart step that measures a batch. The two procedures stop calling measure and gate --batch proposed." \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 13: Verification and adversarial review

**Files:**
- Modify: `docs/superpowers/specs/2026-10-07-corrected-rates-and-disagreements-design.md` (status line)

- [ ] **Step 1: The full CI set, on both Pythons**

```bash
uv run ruff check .
```

```bash
uv run mypy src/axial
```

```bash
uv run pytest -q
```

```bash
uv sync --python 3.13 && uv run --python 3.13 pytest -q; uv sync
```

```bash
uv run python scripts/knowledge_index.py --check
```

Expected: all clean. pytest reports 48 xfailed: the original 49, minus the tour xfail.

- [ ] **Step 2: Adversarial review (opus subagent)**

Dispatch one reviewer with no stake in the code. Its prompt must include the spec path, this plan's path, and `git diff main...corrected-rates`. It must ask the reviewer to try to break, with a failing test for each claim:
- the estimators: off-by-one in the Beta parameters, a draw-order dependence beyond the third decimal, `clip` applied before a difference, and the 90% survival rule;
- the staleness checks: a measurement that `gate --batch` accepts after `judge.yaml`, `verdicts.jsonl` or `PROMOTED` changed in any way, or after the measurement's own files were edited;
- pairing: duplicate inputs, set-aside traces, and the 80% boundary;
- every CLI path ending in a traceback rather than a one-line message;
- `write_alignment` without disagreements differing from before in any byte.

Fix confirmed findings test-first, and commit each fix separately.

- [ ] **Step 3: Mark the spec implemented**

In the spec, change `**Status:** approved design, not yet implemented` to `**Status:** implemented on branch corrected-rates (plan: docs/superpowers/plans/2026-10-14-corrected-rates-and-disagreements.md)`.

```bash
git add docs/superpowers/specs/2026-10-07-corrected-rates-and-disagreements-design.md
git -c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com commit -m "Mark the corrected-rates spec implemented" \
  -m "Points the spec at the plan it was built from." \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
