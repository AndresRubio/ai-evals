# Corrected failure rates on new traces, and a dev disagreement listing

**Date:** 2026-10-07, revised 2026-10-08 (pairing by input; difference before clipping)
**Status:** implemented (plan: docs/superpowers/plans/2026-10-14-corrected-rates-and-disagreements.md)
**Builds on:** [`2026-09-17-axial-evals-toolkit-design.md`](2026-09-17-axial-evals-toolkit-design.md)

## Why

A promoted judge cannot grade anything new. `run_judge` quarantines every unlabelled trace
(`judge.py:270`) and `judge --split` accepts only train, dev and test, so `gate` only ever
re-reads the judge's own alignment verdicts. A baseline comparison compares two versions of
the judge, not two versions of the system. The README's "a judge cleared for CI" promises
more than that, and a capability test records the gap as a strict xfail
(`tests/test_capabilities_cli_tour.py:794`).

Grading new traces is not enough on its own. A judge with TPR 0.93 and TNR 0.94 that flags
12% of a batch is not reporting a 12% failure rate: some flags are false alarms, and some
failures went unflagged. The AI Evals course
([`validate-evaluator`](https://github.com/ai-evals-course/evals-skills/blob/main/skills/validate-evaluator/SKILL.md))
corrects this with the Rogan-Gladen estimator and a bootstrap interval. `axial` has the
inputs, a promoted judge's test-split confusion matrix, but no estimator.

The correction earns its place in two places: the absolute rate a team reports, and a gate
on an absolute ceiling. It does not decide whether a *change* is a regression. That call
depends only on the two batches' flag rates (section 2), and the correction only sets the
reported size of the difference. What decides whether the regression gate catches anything
is pairing: CI usually runs the same eval inputs through both versions of a system, and
comparing those inputs pair by pair needs far fewer of them than comparing two unrelated
samples.

The second gap is in the loop that produces a promotable judge. Revising a judge means
reading the cases where it disagrees with the human and asking why. `alignment.md` reports
the confusion matrix but never names those cases, so the loop runs through hand-written
`jq` over `verdicts.jsonl`.

## Decisions taken

| Decision | Choice | Why |
|---|---|---|
| What `gate` compares | A second mode, comparing measured batches; the judge-vs-judge mode stays | Asked 2026-10-07. Judge-vs-judge comparison is still useful while developing a judge. |
| Absolute ceiling | `--max-rate` on a single batch | Asked 2026-10-07. Gives CI a check before any baseline batch exists. |
| Where measurements live | Their own tree, `.axial/measurements/` | Writing into the judge's `verdicts.jsonl` changes its hash and deletes PROMOTED. |
| Interval method | Monte Carlo over Jeffreys Beta posteriors | A percentile bootstrap gives a judge that scored 30/30 on test zero uncertainty. |
| Which split supplies TPR and TNR | Test only | Dev was used to revise the judge, so its rates are optimistic. |
| Test disagreements | Counted, never listed | Listing them turns test into a second dev set. |
| Comparing two batches | Paired by input when at least 80% of each batch's graded inputs are shared, otherwise unpaired | Pairing approved 2026-10-08; the 80% threshold is a default chosen in this revision, not asked. Simulated, pairing catches a 4-point regression at 500 inputs about twice as often (section 2). |
| Difference and clipping | The difference is computed before clipping | Clipping each rate first hid real regressions whenever the test split understated TNR (section 2). |

## Scope

In:

1. `axial measure`: grade a named batch of unlabelled traces with a promoted judge and
   report a corrected failure rate with a 95% interval.
2. `axial gate --batch`: gate on one measured batch (`--max-rate`) or on the difference
   between two (`--baseline-batch`), paired by input when the batches share their inputs.
3. A dev disagreement listing in `alignment.md` and `alignment.json`.

Out, each a possible later change:

- Labelling a random slice of a batch to check the judge's error rates still hold on it.
  This is the next change after this one: the correction's one untested assumption is that
  the judge errs on new output as it did on test, and a changed system can produce subtler
  failures it misses more often.
- Calling `verify_test_manifest` from `measure` (it has no caller anywhere today).
- Fixing the judge-vs-judge gate's positional matching (`gate.py:133`, a strict xfail).
- Critique-before-verdict judge output, and the few-shot JSON-escaping bug
  (`judge.py:135`). The disagreement listing already reads a `critique` field if a later
  template produces one.

## Conventions

As everywhere in `axial`, the failure is the positive class: `True` means the failure mode
is present. TPR is the share of real failures the judge flags. TNR is the share of real
passes it leaves alone. The course's skills count Pass as positive, so their TPR is this
TNR. The README states this beside the invariants.

## Changes made during implementation

An adversarial review at the end of the build changed the code in these ways. Where a later
section says otherwise, this section describes what the code does.

- Mode names and batch names must be one plain directory name: letters, digits, `.`, `_`
  and `-`, starting with a letter or digit, at most 100 characters. Every path built from a
  mode goes through one check, so `../judges/merged` or an absolute path is refused in one
  line before anything is read or written.
- A batch name that differs from an existing one only in case is refused by `measure` and
  `gate` on every filesystem. Mode names that differ only in case are not checked.
- A batch gated against itself is a usage error (exit 2) in the CLI and an `AxialError` in
  the library, because it can never regress. So is a `--max-rate` outside (0, 1), NaN
  included, since a NaN ceiling would pass any batch.
- A batch with an input or output that has no UTF-8 form (a lone surrogate, as a cut emoji
  leaves) is refused before any trace is graded.
- `measure` reads the traces, labels and splits only for a judge with few-shot ids. When it
  does and cannot read them, it refuses in one line before the model is built.
- `load_measurement` checks the four files against each other. All four must be files, the
  mode and batch must match the directory, the counts must match the rows, every trace must
  be graded or set aside exactly once, every verdict must be true or false, and each
  verdict row's `input_sha256` must be the hash of its trace's input.
- `gate --batch` recomputes the corrected rate from the measurement's own verdicts and the
  promoted judge's test split, and refuses a measurement whose stored numbers differ. Counts,
  observed rate, point estimate, clip flag, test matrix and whether each interval exists
  are compared exactly. Interval bounds are compared with an absolute tolerance of 0.01,
  because a numpy upgrade between `measure` and `gate` can move the Monte Carlo percentiles
  and would otherwise force a paid re-measure. The recomputation uses the draws and seed the
  measurement records, so an edit to either is caught only when it moves a bound by more
  than the tolerance. The gate decides on the recomputed values, so an edit inside the
  tolerance cannot change a decision. A measurement that records fewer than 1,000 or more
  than 1,000,000 draws is refused: one draw gives an interval of zero width, and a huge
  count exhausts memory.
- The chance cut-off, J of 0.05 or less, that withholds the point estimate and the
  judge-exact interval is decided in exact arithmetic on the test counts. A judge exactly at
  the cut-off gets no point. The per-draw rule is unchanged.
- Both intervals follow a Jeffreys boundary rule: the lower bound is 0 when nothing was
  flagged and the upper bound is 1 when everything was. At those two counts the point
  estimate stays inside the interval even when the test split dwarfs the batch.
- `alignment.md` escapes HTML in critiques and judge reasons, and shows trace ids and
  annotators as code spans that nothing inside them can close. When a label changed since
  the run, the case shows "(label changed since this run; re-run `axial judge`)" in place
  of the new label's critique. Both reports are rendered before either is written, and
  `alignment.md` is written as UTF-8, with a lone surrogate written as its escape. Before
  the judge has been run on test, the table has no test row, so the closing line says only
  that test disagreements are never listed.
- A `PROMOTED` marker that is not readable text is refused in one line, and
  `write_dir_atomically` refuses a target that exists and is not a directory.
- The gate's clipping note names the side the rates were clipped to, and says so when one
  rate was clipped to 0 and the other to 1.
- `gate --batch` prints a batch's set-aside note, for the candidate and the baseline, when
  more than 5% of that batch was set aside. It does not fail on it (owner's ruling): the
  decision is made as before.
- Known limitation, deferred to a follow-up: the promotion binds `judge.yaml` and the
  judge's verdicts, and the few-shot examples are read from the current corpus and labels
  each time the judge grades. `axial promote` itself does not check which `judge.yaml` or
  examples produced the test rows, so editing `judge.yaml` and then promoting binds the
  edited prompt to verdicts from the old one. After relabelling or re-ingesting a few-shot
  example, or after editing `judge.yaml`, the judge has to be re-run on test and promoted
  again before measuring. The README says so.

## 1. Commands and data flow

### `axial measure`

```
axial measure <mode> --version N --traces <file> --name <batch> [--adapter A] [--replace]
```

1. **Require a promoted judge.** The marker and hash checks now in `gate._load` move to
   `report.require_promoted(store, mode, version)`, which returns the two recorded hashes.
   `gate` and `measure` both call it, so they cannot disagree about what "promoted" means.
   `--version` is required: the judge doing the measuring is always named.
2. **Read the batch** with `traces.ingest(path, adapter=...)`, the same loader as
   `axial ingest`. It is never written to `.axial/traces/`, because `ingest` replaces the
   labelled corpus. Refused: an empty batch, and a batch with duplicate trace ids, which
   would be counted twice. Each trace's `input_sha256` (SHA-256 of its `input` text, UTF-8)
   is recorded on its row; it is the pairing key in `gate`.
3. **Validate the name.** `<batch>` must match `[A-Za-z0-9._-]+`; it becomes a directory.
   If `.axial/measurements/<mode>/<batch>/` exists, refuse unless `--replace`, because a
   re-run costs money and replaces evidence.
4. **Grade each trace** with the promoted `judge.yaml`: its prompt template, its model and
   its few-shot ids, resolved against the labelled corpus and labels exactly as `judge`
   resolves them. Two functions come out of `run_judge` so both paths share them:
   `resolve_few_shot(spec, corpus, labels)` and `grade_trace(spec, trace, few_shot, splits,
   model, *, max_tokens)`. `grade_trace` returns either a verdict (bool, raw text, model,
   usage, attempts, latency) or an error row with a failure class. `run_judge` keeps its
   signature and on-disk behaviour.
5. **Set aside** what cannot be scored, in `errors.jsonl` with the existing failure
   classes (`bad_template`, `harness`, `truncated`, `unparseable`) plus one new one:
   `few_shot_example`, for a batch trace whose id is one of the judge's few-shot ids. That
   trace's answer is printed in its own prompt.
6. **Estimate** the corrected rate (section 2) from the judge's test-split confusion matrix
   and the batch's flagged count. If every trace was set aside (`n_graded = 0`), nothing is
   estimated: the rate fields are `null`, a note says so, and the command exits non-zero
   after writing the measurement, so the cost already spent is still on record.
7. **Write** `.axial/measurements/<mode>/<batch>/`, then print one line.

Output files:

| File | Contents |
|---|---|
| `traces.jsonl` | The batch as read, so the measurement is reviewable on its own |
| `verdicts.jsonl` | Per graded trace: `trace_id`, `input_sha256`, `verdict`, `model`, `usage`, `attempts`, `latency_s`, `raw`. No `truth`. |
| `errors.jsonl` | Set-aside traces, with `input_sha256`, `failure_class` and `detail` |
| `measurement.json` | See below |

`measurement.json` fields: `mode`, `batch`, `judge_version`, `judge_yaml_sha256`,
`verdicts_sha256` (the two hashes the PROMOTED marker recorded), `test_matrix`
(`tp`, `fn`, `tn`, `fp`), `n_traces`, `n_graded`, `n_set_aside`, `flagged`,
`observed_rate`, `corrected_rate`, `interval` (`[low, high]` or `null`),
`interval_if_judge_exact` (`[low, high]` or `null`), `draws`, `seed`, `notes` (list of
strings), `created` (UTC ISO timestamp).

Printed line, for example:

```
merged_stories v3 on rel-10-07: observed 0.120 -> corrected 0.072 [0.000, 0.128]
  (judge exact: [0.042, 0.107]), n=498 graded, 2 set aside
```

followed by each note on its own line.

Notes, each added only when it applies:

- **Always:** the correction assumes the judge errs on this batch at the rates it showed on
  test, and only labels can confirm that for outputs unlike the labelled corpus.
- More than 5% of the batch set aside: errors that correlate with failure (for example,
  truncation on long outputs) bias the rate.
- The point estimate clipped to 0: the observed rate is below the judge's false-alarm rate
  (1 − TNR), so the batch is consistent with no failures at all.
- The point estimate clipped to 1.
- No trace flagged: the saturation warning `gate` already gives.

### `axial gate --batch`

```
axial gate <mode> --batch <name> [--baseline-batch <name>] [--max-rate R]
axial gate <mode> --version N [--baseline M]          # unchanged
```

`--version` stops being a required option. Exactly one of `--version` and `--batch` must
be given, otherwise exit non-zero with a message naming both. `--baseline-batch` and
`--max-rate` are accepted only with `--batch`; `--baseline` only with `--version`.
`--max-rate` must lie strictly between 0 and 1.

In batch mode, `gate`:

1. Loads `measurement.json` for each named batch, failing cleanly if one is missing.
2. **Refuses a stale measurement.** It calls `require_promoted` for the measurement's
   `judge_version` and compares the returned hashes with the measurement's. A judge that
   was re-judged (PROMOTED deleted) or edited (hash changed) since the measurement fails
   the gate, naming the batch and the command to re-measure it.
3. With a baseline, refuses batches measured by different judge versions: the difference
   would mix a judge change with a system change.
4. With a baseline, **pairs the batches by input** (`measure.pair_batches`):
   - A pair is an `input_sha256` that appears exactly once among each batch's graded
     traces. An input graded more than once in either batch is left out of pairing, since
     there is no single partner to compare it with.
   - If the pairs cover at least 80% of each batch's graded traces, the comparison is
     paired and uses only the pairs. Otherwise it is unpaired over both full batches.
   - Either way the gate prints which, with the counts, for example
     `paired on 987 shared inputs (13 baseline and 9 candidate traces unmatched)` or
     `unpaired: only 41.0% of inputs shared`.
5. Prints the candidate's corrected rate and interval, and with a baseline the corrected
   difference and its interval.
6. Fails (exit 1) when:
   - with `--baseline-batch`: the lower end of the difference interval is above 0
     (`REGRESSION`, the same rule as the judge-vs-judge mode);
   - with `--max-rate`: the upper end of the candidate's interval is above `R`
     (`ABOVE CEILING`);
   - any interval the active checks need is `null`: a gate cannot pass on an interval
     that could not be computed;
   - a named batch has `n_graded = 0`, whatever other options were given.
7. Prints the saturation warning when the candidate batch flagged nothing, and, when
   either batch's rate is shown clipped, a note that the difference was computed before
   clipping (so it can be non-zero while both rates print as 0.000).

With neither `--baseline-batch` nor `--max-rate`, batch mode prints and passes, as the
judge-vs-judge mode does without `--baseline`.

## 2. Statistics

All of it lives in `align.py`, whose docstring makes it the one module that computes
rates.

### Estimator

From the promoted judge's **test** verdicts: `TPR = tp / (tp + fn)`, `TNR = tn / (tn + fp)`.
From the batch: `p = flagged / n_graded`.

```
θ = (p + TNR − 1) / (TPR + TNR − 1),   clipped to [0, 1]
```

Derivation: the judge flags a trace with probability `p = TPR·θ + (1 − TNR)(1 − θ)`, which
rearranges to the line above. The denominator `J = TPR + TNR − 1` is Youden's J.

Cross-check, converted from the course's worked example: TPR 0.88, TNR 0.92, observed 0.20
gives θ = 0.12 / 0.80 = 0.150, which is the course's 0.85 pass rate from the failure side.

### Interval

`D = 20,000` draws from `numpy.random.default_rng(seed)`, `seed = 0` by default and recorded:

```
TPR* ~ Beta(tp + ½, fn + ½)
TNR* ~ Beta(tn + ½, fp + ½)
p*   ~ Beta(flagged + ½, n_graded − flagged + ½)
θ*   = clip((p* + TNR* − 1) / (TPR* + TNR* − 1), 0, 1)
```

The interval is the 2.5th and 97.5th percentiles of `θ*`.

- A draw with `TPR* + TNR* − 1 ≤ 0.05`, a judge no better than chance in that draw, is
  invalid. The interval is reported only if at least 90% of draws are valid, matching
  `align.py`'s existing `_MIN_SURVIVING_FRACTION`; otherwise it is `null`.
- `interval_if_judge_exact` repeats the computation with TPR and TNR fixed at their point
  estimates, so only `p*` varies. Comparing the two intervals tells the reader whether more
  test labels or a bigger batch would tighten the result.
- Why not the course's percentile bootstrap: a judge that scored 30/30 and 70/70 on test
  resamples to TPR = TNR = 1 every time, so its own uncertainty vanishes. For that judge and
  a batch flagging 60 of 500, the Jeffreys interval is [0.080, 0.151] and the bootstrap's
  [0.094, 0.151].

Worked numbers this design was checked against (seed 0, computed 2026-10-07):

| Test matrix | Batch | Point | Interval | Judge exact |
|---|---|---|---|---|
| tp 28, fn 2, tn 66, fp 4 | 60 / 500 | 0.072 | [0.000, 0.128] | [0.042, 0.107] |
| tp 28, fn 2, tn 66, fp 4 | 20 / 500 | 0.000 (clipped) | [0.000, 0.028] | [0.000, 0.003] |

With a test split near the 30-per-class floor, the judge's uncertainty dominates a
single-batch interval, so `--max-rate` is a loose gate until the test split grows.

### Difference between two batches

Both batches are graded by the same judge, so each draw takes `TPR*` and `TNR*` once and
applies them to both. The difference is computed **before clipping**:

```
Δ* = (p_B* − p_A*) / (TPR* + TNR* − 1)
```

This is exact under the model, because the correction is linear in `p` with the same slope
for both batches. Clipping keeps a single rate inside [0, 1] for display; it has no place
in a difference. The point estimate is `(p̂_B − p̂_A) / Ĵ` from the same counts.

**Unpaired**, over both full batches: `p_A* ~ Beta(flagged_A + ½, n_A − flagged_A + ½)`,
likewise `p_B*`, drawn independently.

**Paired**, over the matched pairs only: count the four cells, `n11` (flagged in both),
`n10` (baseline only), `n01` (candidate only) and `n00` (neither), and draw

```
(π11*, π10*, π01*, π00*) ~ Dirichlet(n11 + ½, n10 + ½, n01 + ½, n00 + ½)
p_B* − p_A* = π01* − π10*
```

`Dirichlet(½, ½, ½, ½)` is the Jeffreys prior for a four-cell multinomial, the same family
as the single-rate interval. The invalid-draw rule and the 90% survival rule apply as for a
single rate.

Because `J* > 0.05` in every valid draw, `Δ*` always has the sign of `p_B* − p_A*`. The
judge's error rates rescale a difference; they can neither create nor hide one, and whether
a regression is called depends only on the flag rates.

**Why not clip first.** The first version of this design clipped each `θ*` and subtracted.
When the test split happens to understate TNR, a large share of draws put both batches
below the judge's false-alarm rate, both clip to 0, and `Δ* = 0` exactly, so the lower
bound sits at 0 and a real rise from 10% to 14% failures goes uncalled. In simulation that
capped the paired gate's power at about 0.82 even with 2,000 inputs.

**Worked numbers** (test matrix tp 28, fn 2, tn 66, fp 4; 20,000 draws, seed 0; computed
2026-10-08):

| Comparison | Point | Interval | Regression called |
|---|---|---|---|
| Unpaired, 60/500 against 80/500 | +0.046 | [−0.003, +0.099] | no |
| Paired, 500 inputs: 20 newly flagged, none newly clean | +0.046 | [+0.028, +0.071] | yes |
| Paired, 500 inputs: 30 newly flagged, 10 newly clean | +0.046 | [+0.018, +0.078] | yes |
| Paired, 500 inputs: 15 each way, no net change | +0.000 | [−0.026, +0.026] | no |

Drawing the judge independently for each batch, instead of sharing its draws, widened the
unpaired interval in the first row to about [−0.056, +0.142] when it was computed with
clipping on 2026-10-07.

**What the gate can detect.** Simulated, 600 trials per setting, with a judge at TPR 0.93
and TNR 0.94 on a 30/70 test split, and failures rising from 10% to 14%. Real use lies
between the two kinds of row: an input whose behaviour did not change either produces the
identical output, which a deterministic judge grades identically, or produces new wording,
which the judge grades afresh and may err on.

| Inputs shared | Unchanged inputs with identical output | Paired: regression caught | Unpaired: caught |
|---|---|---|---|
| 500 | none | 0.51 | 0.27 |
| 1,000 | none | 0.81 | 0.58 |
| 2,000 | none | 0.96 | 0.91 |
| 500 | 90% | 0.91 | not simulated |
| 1,000 | 90% | 1.00 | not simulated |

With nothing changed, the paired gate called a regression in 2.7–2.8% of trials against a
nominal 2.5%, and the difference interval covered the true difference in 94–96% of trials,
for this judge and for a borderline one (TPR = TNR = 0.90 on 30/30). The borderline judge
caught the 4-point rise at 1,000 inputs only about 0.55 of the time: a weaker judge needs
more inputs. The README states the sizing: to catch a 4-point rise about four times in five
with a judge like the first, run about 1,000 shared inputs through both versions, or about
500 if most unchanged inputs produce identical output.

### Code

```python
@dataclass(frozen=True)
class CorrectedRate:
    observed: float
    point: float
    interval: tuple[float, float] | None
    interval_if_judge_exact: tuple[float, float] | None
    clipped: Literal["low", "high"] | None
    draws: int
    seed: int

def corrected_rate(test: ConfusionMatrix, flagged: int, n: int, *,
                   draws: int = 20_000, seed: int = 0) -> CorrectedRate: ...

@dataclass(frozen=True)
class CorrectedDifference:
    point: float
    interval: tuple[float, float] | None
    paired: bool
    n: int                      # pairs if paired, else n_A + n_B
    draws: int
    seed: int

def corrected_difference_unpaired(test: ConfusionMatrix,
                                  baseline: tuple[int, int], candidate: tuple[int, int], *,
                                  draws: int = 20_000, seed: int = 0) -> CorrectedDifference: ...

def corrected_difference_paired(test: ConfusionMatrix,
                                cells: tuple[int, int, int, int], *,
                                draws: int = 20_000, seed: int = 0) -> CorrectedDifference: ...
```

`baseline` and `candidate` are `(flagged, n_graded)`; `cells` is `(n11, n10, n01, n00)`.
Matching rows into pairs is not statistics, so it lives in `measure.py`:
`pair_batches(baseline_rows, candidate_rows) -> Pairing`, returning the four cells, the
pair count, the unmatched counts and whether the 80% rule was met.

All three statistics functions raise `ValueError` on `n = 0` or a test matrix with no real
failures or no real passes. Promotion requires 30 of
each by default, but the floor is configurable, so `measure` checks the test matrix first
and refuses with an `AxialError` naming the empty class rather than letting the
`ValueError` escape.

## 3. Disagreement listing

### What is listed

Every **dev** verdict where judge and human disagree, in two groups:

- **Missed failures** (`missed_failure`): human said fail, judge said pass. These lower TPR
  and are listed first, as usually the costlier mistake.
- **False alarms** (`false_alarm`): human said pass, judge said fail.

Within a group, sorted by trace id, so the file diffs cleanly between runs. The names
"false pass" and "false fail" are avoided because they flip meaning under the course's
convention.

Test disagreements are never listed. The section says so in one line; their counts are in
the confusion matrix already.

### Each case

````markdown
#### missed failure — `trace-0412`

**Human** (annotator `andres`): Two separate stories were merged under one headline.

**Judge:** The output lists two headlines, so stories are not merged.

Input (first 300 of 2,184 characters):

```
THE WIRE — issue 1187 ...
```

Output (first 300 of 640 characters):

```
{"stories": [{"headline": ...
```
````

- The human's critique and annotator come from the mode's labels. A missing or empty
  critique shows `(no critique recorded)`.
- The judge's reason is read from the verdict row's stored `raw` text when the report is
  built: the JSON `reason` field, else `critique`, else the raw text truncated to 300
  characters. `run_judge` and the hash-bound `verdicts.jsonl` do not change, and runs made
  before this feature get a listing too.
- Excerpts are the first 300 characters, with the full length. A trace id missing from
  `.axial/traces/` shows `(trace not found in .axial/traces)`.
- Excerpts sit in fenced code blocks so trace text cannot break the Markdown. The fence is
  one backtick longer than the longest backtick run in the excerpt, minimum three.

The section opens with one line of guidance from the course's table: a missed failure
usually means the failure's definition or examples are too narrow, a false alarm that they
are too broad; revise `judge.yaml`, then run `axial judge --split dev` again.

### Outputs

- `alignment.md`: a `## Dev disagreements` section after the summary lines, or the line
  `No dev disagreements.`
- `alignment.json`: `disagreements.dev`, a list of
  `{trace_id, kind, critique, judge_reason}`, without excerpts.
- `axial align` prints `dev: 3 missed failures, 2 false alarms (listed in alignment.md)`,
  or `dev: no disagreements`.

### Code

In `report.py`, since this is reporting rather than statistics:

```python
@dataclass(frozen=True)
class Disagreement:
    trace_id: str
    kind: Literal["missed_failure", "false_alarm"]
    critique: str
    annotator: str
    judge_reason: str
    input_excerpt: str | None
    input_length: int | None
    output_excerpt: str | None
    output_length: int | None

def find_disagreements(verdicts: list[dict[str, Any]], labels: list[Label],
                       traces: list[Trace], split: str = "dev") -> list[Disagreement]: ...
def parse_reason(raw: str) -> str: ...
```

`write_alignment` gains `disagreements: list[Disagreement] | None = None`. When it is
`None` the section is left out and the output is byte-identical to today's, so existing
callers and tests are unaffected. The `align` command passes it.

## 4. Error handling

- Every refusal in `measure` and batch-mode `gate` raises `AxialError` (or a subclass) and
  the CLI turns it into a one-line message and a non-zero exit, never a traceback.
- A model call that fails during `measure` is set aside, never scored and never fatal, as
  in `run_judge`. An `InvariantViolation` from prompt assembly still propagates.
- `measure` writes its four files only after grading finishes, through the store's atomic
  writes, so an interrupted run leaves no partial measurement directory behind.

## 5. Testing

Test-first. Statistical tests assert numbers derived by hand, not the implementation's own
formula.

**Estimator**

- The course cross-check: θ = 0.150.
- A perfect judge returns the observed rate.
- Below `1 − TNR` clips to 0 with `clipped="low"`; above TPR clips to 1 with `"high"`.
- The interval contains the point estimate, is identical for a fixed seed, and narrows as
  the test split or the batch grows.
- A perfect test score (30/30, 70/70): the interval's lower bound is below
  `interval_if_judge_exact`'s.
- A judge near chance (15/30, 35/70): interval `null`.
- Differences, paired and unpaired: identical batches straddle 0; Δ has the sign of the
  observed difference; shared judge draws give a narrower interval than independent ones;
  on the same flags with concentrated changes, the paired interval is narrower than the
  unpaired one; two batches whose single rates both clip to 0 still give a non-zero Δ when
  their flag rates differ, the regression the first design hid.
- **Coverage simulation.** Simulate a known true rate and judge, draw test labels and a
  500-trace batch, build the interval, repeat 1,000 times with 4,000 draws each. Require
  coverage in [0.92, 0.99] at nominal 95% for four settings: true rate 0.10 and 0.01, each
  with a judge at TPR 0.93 / TNR 0.94 on a 30/70 test split and at TPR = TNR = 0.90 on
  30/30. A scratch run on 2026-10-07 (seed 1) gave 0.952, 0.949, 0.937 and 0.926; across
  nine settings coverage ranged 0.926–0.969, lowest for the borderline 0.90 judge. If the
  test fails, the interval method goes back for review, not the test's band.
- **Difference simulation.** 1,000 shared inputs, judge outputs graded independently,
  600 trials with 3,000 draws each, for both judges above. With no change in the true
  failure rate, the paired gate calls a regression in at most 4% of trials. With a 4-point
  rise, the paired interval covers the true difference in [0.92, 0.99] of trials, and the
  0.93/0.94 judge calls the regression in at least 70%. A scratch run on 2026-10-08 (seed 3)
  gave 2.7%, 0.960 and 0.81.

**`measure`**

- Refuses an unpromoted judge, and a judge whose `judge.yaml` changed after promotion.
- The judge directory, PROMOTED and its `verdicts.jsonl` are byte-identical afterwards;
  `.axial/traces/` is untouched.
- Writes the four files; `traces.jsonl` matches the input; `measurement.json` has every
  field.
- Prompts use `judge.yaml`'s model and train few-shot examples (a recording FakeModel).
- Set-aside classes, including `few_shot_example`, stay out of the rate; the note appears
  above 5%.
- Refuses an empty batch, duplicate trace ids, a bad name and an existing name;
  `--replace` replaces.
- Every existing `run_judge` test passes unchanged after the extraction.

**`gate --batch`**

- `--max-rate` fails above the ceiling and passes below it.
- With a baseline: regression exits 1, improvement and no change exit 0.
- Different judge versions across batches are refused.
- Pairing: batches with the same inputs are compared paired, with the counts printed; an
  input graded twice in one batch is left unpaired; below 80% shared the comparison is
  unpaired and says so; set-aside traces never pair.
- A measurement taken before the judge was re-judged, or before `judge.yaml` was edited,
  fails the gate.
- Both `--version` and `--batch`, neither, or a mode-mismatched option: non-zero exit with
  a message.
- The saturation warning when the batch flagged nothing.
- The note when a rate is shown clipped and the difference was computed before clipping.
- A measurement with `n_graded = 0` fails the gate.

**Disagreements**

- Handmade verdict rows: one missed failure, one false alarm, agreements not listed; the
  order; the placeholder critique; the reason from `reason`, `critique` and raw fallback;
  300-character excerpts with lengths; a missing trace; a backtick run in an excerpt.
- Test-split disagreements are never listed.
- `write_alignment` without `disagreements` is byte-identical to today's output.
- `alignment.json` carries `disagreements.dev`; `axial align` prints the counts.

**Existing tests this changes on purpose**

- `tests/test_capabilities_cli_tour.py:794`, a strict xfail, calls `run_judge(split="gate")`,
  a path that stays refused by design. It is rewritten to run `measure` and then
  `gate --batch` on the promoted repo, and its marker is removed.
- `tests/test_capabilities_cli_failures.py:90` asserts `Missing option '--version'`; it is
  updated to the new message.
- The judge-vs-judge positional-matching xfail (`test_capabilities_promote_gate.py:262`)
  is untouched; batch comparison pairs by input hash, not by row position.

**Before calling it done:** the full suite, ruff and mypy on Python 3.14 and 3.13; both
simulations; and an adversarial review of the statistics and the stale-measurement
checks by a separate reviewer.

## 6. Documentation

- README pipeline table: a `measure` row (a promoted judge + a batch file → a corrected
  failure rate), and the `gate` row updated to name both modes.
- README invariants: "A corrected rate comes only from a promoted judge's test-split error
  rates, and every measurement states the assumption it rests on." The gate bullet gains
  the batch mode and `--max-rate`.
- README Quickstart: a final step measuring a batch.
- README sizing guidance: the shared-input numbers from section 2.
- `knowledge/procedures/evals-in-ci.md` and `knowledge/procedures/production-monitoring.md`
  (on `main` since 2026-10-07) describe `measure` and `gate --batch` as proposed. Once this
  branch sits on a base that has them, they are updated to describe what shipped,
  including pairing.
- The positive-class sentence added on 2026-10-07 stays.
