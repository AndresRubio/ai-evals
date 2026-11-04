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
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from axial.align import ConfusionMatrix, CorrectedRate, confusion_matrix, corrected_rate
from axial.errors import AxialError, InvariantViolation, MalformedFile
from axial.judge import (
    Graded,
    JudgeSpec,
    grade_trace,
    load_few_shot,
    prompt_sha256,
    read_verdicts,
)
from axial.model import Model
from axial.promotion import require_promoted
from axial.store import Store, json_text, jsonl_text, refuse_non_directory
from axial.traces import Trace, lone_surrogate

# Above this share of set-aside traces, a measurement says the rate may be biased.
SET_ASIDE_NOTE_SHARE = 0.05

# Two batches are compared pair by pair only if the pairs cover this share of each.
MIN_PAIRED_SHARE = 0.8


ASSUMPTION_NOTE = (
    "the correction assumes the judge errs on this batch at the rates it showed on its "
    "test split; only labels on this batch can confirm that for outputs unlike the "
    "labelled corpus"
)

SATURATION_WARNING = (
    "saturation warning: no trace failed. A suite nothing fails has stopped carrying "
    "signal while still costing maintenance — consider harder cases or retiring it."
)


def input_sha256(text: str) -> str:
    """The pairing key: SHA-256 of a trace's input text, UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Measurement:
    """What `measurement.json` holds. `rate` is `None` when no trace was graded."""

    mode: str
    batch: str
    judge_version: int
    judge_yaml_sha256: str
    verdicts_sha256: str
    prompt_sha256: str
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
            "prompt_sha256": self.prompt_sha256,
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
            prompt_sha256=str(data["prompt_sha256"]),
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
    # A lone surrogate has no UTF-8 form to hash or send. A batch file's rows were checked
    # as they were read, but an adapter's traces or a caller's own were not, and meeting
    # one mid-run would come after the traces before it were graded and paid for.
    for trace in batch:
        problem = lone_surrogate(trace.input, trace.output)
        if problem:
            raise AxialError(f"trace {trace.trace_id!r} has {problem}; fix it in the batch file")


def _refuse_case_variant(directory: Path, mode: str, batch: str) -> None:
    """Refuse `batch` if a batch whose name differs from it only in case is on disk.

    On a case-insensitive filesystem (the macOS default) both names open one directory,
    so `measure` would replace the other batch and `gate` would read it. Refusing on
    every filesystem keeps a repository that works on Linux working on a Mac.
    """
    if not directory.parent.is_dir():
        return
    for entry in directory.parent.iterdir():
        if entry.name != batch and entry.name.casefold() == batch.casefold():
            raise AxialError(
                f"a measurement named {entry.name!r} already exists for {mode}, and "
                f"{batch!r} differs from it only in case, which many filesystems treat as "
                f"the same directory; use {entry.name!r} or a name that differs in more "
                "than case"
            )


def judge_test_matrix(store: Store, mode: str, version: int) -> ConfusionMatrix:
    """The judge's test-split confusion matrix, which every corrected rate it gives uses."""
    rows = [
        r for r in read_verdicts(store, store.judge_dir(mode, version) / "verdicts.jsonl")
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


def _percent(part: int, whole: int, *, round_up: bool) -> str:
    """`part / whole` as a percentage to one decimal, rounded in the direction given.

    A share is printed beside the threshold it was compared with, so it is rounded away
    from that threshold: rounded to nearest, 1,001 of 20,000 (5.005%) would print as the
    5.0% it exceeds. Integer arithmetic, so no float error moves it across a tenth.
    """
    tenths, rest = divmod(part * 1000, whole)
    if round_up and rest:
        tenths += 1
    return f"{tenths // 10}.{tenths % 10}%"


def set_aside_note(n_set_aside: int, n_traces: int) -> str | None:
    """The note for a batch with more than `SET_ASIDE_NOTE_SHARE` of its traces set aside.

    `measure` stores it and `gate --batch` prints it from the batch's counts, which
    `load_measurement` has checked against the batch's files.
    """
    if n_set_aside <= SET_ASIDE_NOTE_SHARE * n_traces:
        return None
    share = _percent(n_set_aside, n_traces, round_up=True)
    return (
        f"{n_set_aside} of {n_traces} traces ({share}) were set "
        "aside (see errors.jsonl); if what set them aside correlates with failure, as "
        "truncation on long outputs can, the rate is biased"
    )


def _notes(
    test: ConfusionMatrix, rate: CorrectedRate | None, n_traces: int, n_set_aside: int
) -> list[str]:
    notes = [ASSUMPTION_NOTE]
    set_aside = set_aside_note(n_set_aside, n_traces)
    if set_aside:
        notes.append(set_aside)
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
    draws: int = 20_000,
    seed: int = 0,
) -> Measurement:
    """Grade `batch` with promoted judge `mode` v`version` and write the measurement.

    Nothing is written until every trace has been graded, and then all four files
    at once, so an interrupted run leaves no partial measurement behind.
    """
    hashes = require_promoted(store, mode, version)
    directory = store.measurement_dir(mode, batch_name)
    _refuse_case_variant(directory, mode, batch_name)
    # `write_dir_atomically` refuses this too, but only after every trace was graded.
    refuse_non_directory(directory)
    if directory.exists() and not replace:
        raise AxialError(
            f"a measurement named {batch_name!r} already exists for {mode} at {directory}. "
            "Re-running costs money and replaces evidence: pass --replace if you mean "
            "to, or choose another --name."
        )
    _check_batch(batch)
    test = judge_test_matrix(store, mode, version)
    spec = JudgeSpec.load(store, mode, version)
    few_shot, splits = load_few_shot(store, spec)
    # require_promoted resolved the examples too; resolve-then-compare here catches
    # labels or traces that changed in between, before anything is paid for.
    if prompt_sha256(spec, few_shot) != hashes.prompt_sha256:
        raise InvariantViolation(
            f"judge {mode} v{version}'s few-shot examples changed while this measurement "
            "was starting. Nothing was graded; run `axial measure` again."
        )
    model = build_model(spec)

    # By input, whatever the id: an example's answer is in the prompt under any id, and
    # a batch id that happens to equal an example's says nothing about its content.
    example_inputs = {input_sha256(example.input) for example, _ in few_shot}
    verdicts: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for trace in batch:
        key = input_sha256(trace.input)
        if key in example_inputs:
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
        outcome = grade_trace(spec, trace, few_shot, splits, model, max_tokens=spec.max_tokens)
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
        prompt_sha256=hashes.prompt_sha256,
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


MEASUREMENT_FILES = ("traces.jsonl", "verdicts.jsonl", "errors.jsonl", "measurement.json")


class _MeasuredRow(BaseModel):
    """The fields of a measurement's error row that `load_measurement` reads by name.
    Values are kept as written, so `_inconsistency` still sees a bad one."""

    model_config = ConfigDict(extra="allow")

    trace_id: str


class _MeasuredVerdict(_MeasuredRow):
    verdict: Any
    input_sha256: Any


def _measured_rows(store: Store, path: Path, model: type[_MeasuredRow]) -> list[dict[str, Any]]:
    return [row.model_dump(exclude_unset=True) for row in store.read_models(path, model)]


def _inconsistency(
    measurement: Measurement,
    mode: str,
    batch: str,
    traces: list[Trace],
    rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> str | None:
    """What makes a measurement's four files disagree with each other, if anything.

    `pair_batches` trusts each verdict row's `input_sha256`, and the gate trusts the
    counts, so both are checked against `traces.jsonl`, the batch as it was read.
    """
    if (measurement.mode, measurement.batch) != (mode, batch):
        return "names another mode or batch than its directory"
    if (len(traces), len(rows), len(errors), sum(1 for r in rows if r["verdict"])) != (
        measurement.n_traces, measurement.n_graded, measurement.n_set_aside, measurement.flagged
    ):
        return "does not match its own traces.jsonl, verdicts.jsonl or errors.jsonl in its counts"
    by_id = {t.trace_id: t for t in traces}
    accounted = Counter(r["trace_id"] for r in rows) + Counter(e["trace_id"] for e in errors)
    if len(by_id) != len(traces) or accounted != Counter(by_id.keys()):
        return "does not grade or set aside each trace in its traces.jsonl exactly once"
    for row in rows:
        if not isinstance(row["verdict"], bool):
            return f"has a verdict for trace {row['trace_id']!r} that is not true or false"
        if row["input_sha256"] != input_sha256(by_id[row["trace_id"]].input):
            return (
                f"records an input_sha256 for trace {row['trace_id']!r} that is not the hash "
                "of its input"
            )
    return None


def load_measurement(
    store: Store, mode: str, batch: str
) -> tuple[Measurement, list[dict[str, Any]]]:
    """Read a measurement and its graded rows, refusing anything that does not add up."""
    directory = store.measurement_dir(mode, batch)
    _refuse_case_variant(directory, mode, batch)
    path = directory / "measurement.json"
    if not path.exists():
        raise AxialError(
            f"no measurement named {batch!r} for {mode}: run `axial measure {mode} "
            f"--version N --traces <file> --name {batch}` first"
        )
    redo = "Re-run `axial measure` with --replace"
    for name in MEASUREMENT_FILES:
        if not (directory / name).is_file():
            raise AxialError(
                f"measurement {batch!r} for {mode} has no readable {name}; it was deleted "
                f"or edited by hand. {redo}"
            )
    try:
        document = store.read_json(path)
        if not isinstance(document, dict):
            raise MalformedFile(path, f"expected a JSON object, got {type(document).__name__}")
        try:
            measurement = Measurement.from_json(document)
        except (KeyError, TypeError, ValueError) as exc:
            missing = isinstance(exc, KeyError)
            reason = f"no {exc} key" if missing else f"{type(exc).__name__}: {exc}"
            raise MalformedFile(path, reason) from exc
        traces = store.read_models(directory / "traces.jsonl", Trace)
        rows = _measured_rows(store, directory / "verdicts.jsonl", _MeasuredVerdict)
        errors = _measured_rows(store, directory / "errors.jsonl", _MeasuredRow)
        problem = _inconsistency(measurement, mode, batch, traces, rows, errors)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        # The first line only: a pydantic error runs to several. A MalformedFile already
        # names its file and needs no exception name in front.
        detail = (str(exc).splitlines() or [""])[0]
        if not isinstance(exc, MalformedFile):
            detail = f"{type(exc).__name__}: {detail}"
        raise AxialError(f"{directory} is not a readable measurement ({detail}). {redo}") from exc
    if problem:
        raise AxialError(
            f"measurement {batch!r} for {mode} {problem}; it was edited by hand. {redo}"
        )
    return measurement, rows


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
        # `share` from the counts: the pairs over the larger batch, rounded down.
        larger = self.pairs + max(self.unmatched_baseline, self.unmatched_candidate)
        shown = _percent(self.pairs, larger, round_up=False) if larger else "0.0%"
        return f"unpaired: only {shown} of inputs shared"


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
