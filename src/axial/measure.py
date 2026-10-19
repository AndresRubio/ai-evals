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
