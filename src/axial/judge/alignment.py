"""Alignment reports: alignment.md, alignment.json and the dev disagreement listing.

Everything written here leads with TPR and TNR. Whether the numbers clear the bar is
`promotion`'s call, not this module's.
"""

from __future__ import annotations

import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from axial.analysis.dataset import Label
from axial.core.store import Store
from axial.core.traces import Trace
from axial.errors import InvariantViolation
from axial.judge.run import require_human_labels, row_probability
from axial.stats.align import Alignment, align


def alignment_from_verdicts(
    verdicts: list[dict[str, Any]], split: str, *, seed: int = 0
) -> Alignment:
    rows = [v for v in verdicts if v.get("split") == split]
    require_human_labels(rows)
    y_true = [bool(v["truth"]) for v in rows]
    y_pred = [bool(v["verdict"]) for v in rows]
    # A Brier score over part of a split would be a different number from the one
    # the table implies, so it is all rows or none.
    recorded = [row_probability(v) for v in rows]
    probabilities = [p for p in recorded if p is not None]
    return align(
        y_true,
        y_pred,
        probabilities=probabilities if rows and len(probabilities) == len(rows) else None,
        seed=seed,
    )


def _fmt_rate(value: float) -> str:
    """Point estimate only, `n/a` when undefined.

    The same fallback `Alignment.fmt`/`summary()` use, so a markdown table
    cell and the one-line summary printed by the CLI never disagree about
    whether a rate is reportable -- without this, an undefined rate rendered
    as a bare `nan` in the table while `summary()` said `n/a` for the same
    number.
    """
    return "n/a" if math.isnan(value) else f"{value:.3f}"


def _json_safe(obj: object) -> object:
    """Recursively replace non-finite floats (nan, inf, -inf) with `None`.

    `json.dumps` happily emits bare `NaN`/`Infinity` literals for these, which
    Python reads back but which are not valid JSON — `jq` and JavaScript's
    `JSON.parse` both reject them. This only affects how the report is
    serialised; `Alignment.to_dict()`'s in-memory contract (real floats,
    including nan) is unchanged.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


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
        critique = (label.critique.strip() if label else "") or "(no critique recorded)"
        if label is not None and label.label != truth:
            # The kind, like the alignment table, follows the truth this run was scored
            # on; a critique written for the new label would argue against its heading.
            critique = "(label changed since this run; re-run `axial judge`)"
        found.append(
            Disagreement(
                trace_id=trace_id,
                kind="missed_failure" if truth else "false_alarm",
                critique=critique,
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


def _longest_backtick_run(text: str) -> int:
    return max((len(run) for run in re.findall(r"`+", text)), default=0)


def _fence(text: str) -> str:
    """A code fence one backtick longer than the longest backtick run in `text`, so
    trace text can never close it early."""
    return "`" * max(3, _longest_backtick_run(text) + 1)


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _inline_text(text: str) -> str:
    """Human or model text on one line with &, < and > escaped, so a stray tag stays text:
    an unclosed <details> would hide the rest of the file on GitHub."""
    return html.escape(_one_line(text), quote=False)


def _code_span(text: str) -> str:
    """`text` on one line as an inline code span nothing in it can close early.

    One line keeps a newline from starting a heading. The fence is one backtick longer
    than any run inside, as `_fence` does for blocks, and since CommonMark strips one
    space from each end of a span, padding lets the text start or end with a backtick.
    """
    text = _one_line(text)
    fence = "`" * (_longest_backtick_run(text) + 1)
    pad = " " if not text or text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def _excerpt_lines(name: str, excerpt: str | None, length: int | None) -> list[str]:
    if excerpt is None or length is None:
        return [f"{name}: (trace not found in .axial/traces)", ""]
    shown = f"first {len(excerpt):,} of {length:,}" if length > len(excerpt) else f"all {length:,}"
    fence = _fence(excerpt)
    return [f"{name} ({shown} characters):", "", fence, excerpt, fence, ""]


def _disagreement_section(
    mode: str, disagreements: list[Disagreement], *, has_test: bool
) -> list[str]:
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
                    f"#### {title} — {_code_span(d.trace_id)}",
                    "",
                    f"**Human** (annotator {_code_span(d.annotator)}): {_inline_text(d.critique)}",
                    "",
                    f"**Judge:** {_inline_text(d.judge_reason)}",
                    "",
                ]
                lines += _excerpt_lines("Input", d.input_excerpt, d.input_length)
                lines += _excerpt_lines("Output", d.output_excerpt, d.output_length)
    # The table has a test row only once the judge has been run on test.
    counted = "are counted in the table above and " if has_test else "are "
    lines.append(
        f"Test disagreements {counted}never listed here, so the test split stays a held-out "
        "measurement."
    )
    return lines


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

    lines = [
        f"# Alignment — {mode} v{version}",
        "",
        "Agreement between this judge and the human labels. TPR is the share of real",
        "failures the judge caught; TNR is the share of clean traces it left alone.",
        "Accuracy is never reported alone: it is shown last, beside TPR and TNR, precisely",
        "because under class imbalance a high accuracy on its own is the number that gets a",
        "judge which detects nothing shipped.",
        "",
        "| split | n | TPR | TNR | kappa | accuracy |",
        "|---|---|---|---|---|---|",
    ]
    for name, result in (("dev", dev), ("test", test)):
        if result is None:
            continue
        lines.append(
            f"| {name} | {result.n} | {_fmt_rate(result.tpr)} | {_fmt_rate(result.tnr)} | "
            f"{_fmt_rate(result.kappa)} | {_fmt_rate(result.accuracy)} |"
        )
    lines += [
        "",
        f"Dev: {dev.summary()}",
    ]
    if test is not None:
        lines.append(f"Test: {test.summary()}")
    scored = [(name, r) for name, r in (("dev", dev), ("test", test)) if r and r.brier is not None]
    if scored:
        lines += [
            "",
            "## Calibration",
            "",
            "The Brier score is the mean squared gap between the judge's probability of a",
            "failure and the human label (1 for a failure, 0 for a pass); lower is better.",
            "The reference is what always predicting the split's own failure rate scores. A",
            "Brier score at or above it means the probabilities forecast no better than that",
            "failure rate, so do not read them as failure rates. They can still rank traces",
            "well, which is all a threshold needs: TPR and TNR above say how well this one",
            "separates them.",
            "",
            "| split | Brier | reference |",
            "|---|---|---|",
        ]
        for name, result in scored:
            lines.append(f"| {name} | {result.brier:.3f} | {result.brier_reference:.3f} |")
    lines += [
        "",
        f"Quarantined calls (never scored): {n_errors}. See `errors.jsonl`.",
    ]
    if notes:
        lines += ["", "## Notes", ""]
        lines += [f"- {note}" for note in notes]
    if disagreements is not None:
        lines += [""] + _disagreement_section(mode, disagreements, has_test=test is not None)
    lines += [""]
    # Both reports are rendered before either is written, so a rendering error leaves
    # the pair as it was. UTF-8 whatever the locale; a lone surrogate, which JSON lets a
    # label carry and UTF-8 cannot encode, is written as its escape.
    store.write_json(directory / "alignment.json", _json_safe(report))
    path = directory / "alignment.md"
    path.write_text("\n".join(lines), encoding="utf-8", errors="backslashreplace")
    return path
