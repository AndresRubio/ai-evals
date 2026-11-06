"""The harness that runs a judge, and the verdict and error rows it writes.

A call that failed, for any reason, produces an error row and never a verdict row: an
infra failure scored as a model failure contaminates the headline in a way nothing
downstream can undo.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from axial.analysis.dataset import Label, Splits
from axial.core.model import Model, is_probability
from axial.core.store import Store
from axial.core.traces import Trace
from axial.errors import InvariantViolation
from axial.judge.spec import (
    JudgeSpec,
    Verdict,
    assemble_prompt,
    prompt_sha256,
    refuse_a_verdict_first_template,
    resolve_few_shot,
)


def parse_verdict(text: str) -> bool:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = {}
        verdict = str(payload.get("verdict", "")).strip().lower()
        if verdict in ("fail", "true", "yes"):
            return True
        if verdict in ("pass", "false", "no"):
            return False
    raise ValueError(f"could not read a verdict from {text[:120]!r}")


@dataclass
class RunResult:
    verdicts: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)


class VerdictRow(BaseModel):
    """The fields of a verdicts.jsonl row that axial reads by name. Every other field is
    kept as it is."""

    model_config = ConfigDict(extra="allow")

    trace_id: str
    verdict: bool
    split: str | None = None
    truth: bool | None = None
    # A decision model's score, read by `align` and `tune-threshold`.
    probability: float | None = None

    @field_validator("probability", mode="before")
    @classmethod
    def _is_a_probability(cls, value: object) -> object:
        if value is not None and not is_probability(value):
            raise ValueError("not a number between 0 and 1")
        return value


class ErrorRow(BaseModel):
    """The fields of an errors.jsonl row that axial reads by name."""

    model_config = ConfigDict(extra="allow")

    trace_id: str
    failure_class: str
    split: str | None = None


def read_verdicts(store: Store, path: Path) -> list[dict[str, Any]]:
    """The rows of a judge's verdicts.jsonl, each checked for the fields read by name.

    A hand-edited row without `verdict` used to end in a KeyError far from the file; it
    is now a MalformedFile naming the file, the line and the field.
    """
    return [row.model_dump(exclude_unset=True) for row in store.read_models(path, VerdictRow)]


def read_errors(store: Store, path: Path) -> list[dict[str, Any]]:
    """The rows of a judge's errors.jsonl, checked the same way."""
    return [row.model_dump(exclude_unset=True) for row in store.read_models(path, ErrorRow)]


def require_human_labels(rows: list[dict[str, Any]]) -> None:
    """Refuse verdict rows that carry no human label."""
    missing = [v.get("trace_id") for v in rows if v.get("truth") is None]
    if missing:
        raise ValueError(
            f"{len(missing)} verdict row(s) have no human label ({missing[:3]}); a judge can "
            "only be aligned against labels a human made"
        )


def row_probability(row: dict[str, Any]) -> float | None:
    """A verdict row's recorded probability, or `None` for a row without one.

    `read_verdicts` already refuses such a row in a file, naming its line. This is for
    rows a caller built in memory: a nan, a bool or a value outside [0, 1] would make
    every number computed from it wrong, so it is refused by trace id.
    """
    value = row.get("probability")
    if value is None:
        return None
    if not is_probability(value):
        raise ValueError(
            f"verdict row {row.get('trace_id')!r} records probability {value!r}, which is "
            "not a number between 0 and 1"
        )
    return float(value)


def kept_rows(store: Store, mode: str, version: int, split: str) -> dict[str, list[dict[str, Any]]]:
    """The verdict and error rows a run of `split` leaves in place: every other split's."""
    directory = store.judge_dir(mode, version)
    verdicts = read_verdicts(store, directory / "verdicts.jsonl")
    errors = read_errors(store, directory / "errors.jsonl")
    return {
        "verdicts.jsonl": [r for r in verdicts if r.get("split") != split],
        "errors.jsonl": [r for r in errors if r.get("split") != split],
    }


@dataclass(frozen=True)
class Graded:
    """A model answer that was read as a verdict."""

    verdict: bool
    raw: str
    model: str
    usage: dict[str, int]
    attempts: int
    latency_s: float
    # A decision model's probability of the failure; `None` for a text model.
    probability: float | None = None


@dataclass(frozen=True)
class SetAside:
    """A trace that could not be scored, and why. Never a verdict."""

    failure_class: str
    detail: str


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
        probability=completion.probability,
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
    # Merge by split rather than overwrite, so running dev and then test leaves both
    # on disk. Re-running one split replaces only that split's rows. The rows kept are
    # read first, so a damaged file is refused before a call is paid for or PROMOTED
    # is deleted. So is a template that asks for the verdict first.
    refuse_a_verdict_first_template(store, spec)
    directory = store.judge_dir(spec.mode, spec.version)
    kept = kept_rows(store, spec.mode, spec.version, split)
    few_shot = resolve_few_shot(spec, traces if corpus is None else corpus, labels)
    fingerprint = prompt_sha256(spec, few_shot)
    label_by_id = {lbl.trace_id: lbl for lbl in labels}

    result = RunResult()
    for trace in traces:
        outcome = grade_trace(spec, trace, few_shot, splits, model, max_tokens=spec.max_tokens)
        if isinstance(outcome, SetAside):
            result.errors.append(
                {
                    "trace_id": trace.trace_id,
                    "split": split,
                    "failure_class": outcome.failure_class,
                    "detail": outcome.detail,
                    "prompt_sha256": fingerprint,
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
                    "prompt_sha256": fingerprint,
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
                # A decision model's score, unrounded, so `tune-threshold` and the
                # Brier score can be computed from what is on disk without calling
                # the model again. `None` for a text model, which has no score.
                "probability": outcome.probability,
                "raw": outcome.raw,
                "prompt_sha256": fingerprint,
            }
        )

    directory.mkdir(parents=True, exist_ok=True)

    # A PROMOTED marker binds specific hashes of judge.yaml and verdicts.jsonl
    # (see promotion.promote) to the alignment that earned it. We are about to
    # rewrite verdicts.jsonl for this split, so whatever evidence backed that
    # marker no longer describes what will be on disk -- delete it rather than
    # let a stale marker keep gating a build on a run nobody re-measured.
    marker = directory / "PROMOTED"
    if marker.exists():
        marker.unlink()

    for filename, fresh in (("verdicts.jsonl", result.verdicts), ("errors.jsonl", result.errors)):
        store.write_jsonl(directory / filename, kept[filename] + fresh)
    return result
