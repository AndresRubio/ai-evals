"""Alignment reports and the promotion gate.

A judge becomes usable in CI only by passing `promote`, and `promote` reads the
test split. Everything it prints leads with TPR and TNR.
"""

from __future__ import annotations

import html
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


def alignment_from_verdicts(
    verdicts: list[dict[str, Any]], split: str, *, seed: int = 0
) -> Alignment:
    rows = [v for v in verdicts if v.get("split") == split]
    missing = [v["trace_id"] for v in rows if v.get("truth") is None]
    if missing:
        raise ValueError(
            f"{len(missing)} verdict row(s) have no human label ({missing[:3]}); a judge can "
            "only be aligned against labels a human made"
        )
    y_true = [bool(v["truth"]) for v in rows]
    y_pred = [bool(v["verdict"]) for v in rows]
    return align(y_true, y_pred, seed=seed)


class PromotionCheck:
    def __init__(
        self, ok: bool, reasons: list[str], notes: list[str] | None = None
    ) -> None:
        self.ok = ok
        self.reasons = reasons
        self.notes = notes if notes is not None else []


def check_promotion(dev: Alignment, test: Alignment, thresholds: Thresholds) -> PromotionCheck:
    reasons: list[str] = []
    notes: list[str] = []

    # A rate measured on a handful of cases is not evidence, however clean the
    # point estimate looks: a test split of 4 real failures / 4 real passes, all
    # judged correctly, gives TPR 1.00 and TNR 1.00 with a Wilson lower bound
    # around 0.51. Refuse the split itself before trusting anything computed
    # from it.
    real_positives = test.matrix.tp + test.matrix.fn
    real_negatives = test.matrix.tn + test.matrix.fp
    if real_positives < thresholds.min_test_positives:
        reasons.append(
            f"test split has only {real_positives} real failure(s), below the minimum "
            f"of {thresholds.min_test_positives} — a rate measured on a handful of "
            "cases is not evidence, however clean the point estimate looks"
        )
    if real_negatives < thresholds.min_test_negatives:
        reasons.append(
            f"test split has only {real_negatives} real pass(es), below the minimum "
            f"of {thresholds.min_test_negatives} — a rate measured on a handful of "
            "cases is not evidence, however clean the point estimate looks"
        )

    if math.isnan(test.tpr) or test.tpr < thresholds.min_tpr:
        reasons.append(
            f"test TPR {test.fmt('tpr')} is below the bar of {thresholds.min_tpr:.2f} — "
            "the judge misses real failures"
        )
    if math.isnan(test.tnr) or test.tnr < thresholds.min_tnr:
        reasons.append(
            f"test TNR {test.fmt('tnr')} is below the bar of {thresholds.min_tnr:.2f} — "
            "the judge cries wolf"
        )
    for name in ("tpr", "tnr"):
        dev_value = getattr(dev, name)
        test_value = getattr(test, name)
        if math.isnan(dev_value) or math.isnan(test_value):
            notes.append(
                f"dev->test gap check skipped for {name.upper()}: the rate is undefined "
                "(nan) on the dev or the test split, so no gap could be computed"
            )
            continue
        gap = dev_value - test_value
        if gap > thresholds.max_gap:
            reasons.append(
                f"dev->test gap on {name.upper()} is {gap:.3f} (dev {dev.fmt(name)}, "
                f"test {test.fmt(name)}), above the tolerance of {thresholds.max_gap:.2f} "
                "— the judge prompt is tuned to dev"
            )
    return PromotionCheck(ok=not reasons, reasons=reasons, notes=notes)


JUDGE_YAML_HASH_KEY = "judge_yaml_sha256"
VERDICTS_HASH_KEY = "verdicts_sha256"


def format_promoted_marker(
    dev_summary: str, test_summary: str, judge_yaml_hash: str, verdicts_hash: str
) -> str:
    """Render the `PROMOTED` marker: a human-readable summary plus two
    machine-parseable `key=value` lines binding the marker to the exact
    `judge.yaml` and `verdicts.jsonl` it was earned on.

    Without this binding, `gate` could only check that *some* `PROMOTED` file
    existed, not that the judge or its verdicts still matched what earned it
    -- which is exactly what let a promoted judge be silently replaced under
    its own marker. `require_promoted` recomputes both hashes with `store.hash_file`
    and calls `parse_promoted_marker` to compare them against these.
    """
    return (
        f"test {test_summary}\n"
        f"dev {dev_summary}\n"
        f"{JUDGE_YAML_HASH_KEY}={judge_yaml_hash}\n"
        f"{VERDICTS_HASH_KEY}={verdicts_hash}\n"
    )


def parse_promoted_marker(text: str) -> dict[str, str]:
    """Parse the `key=value` lines out of a `PROMOTED` marker.

    Shared with `gate`, the only other reader of this file, so the marker's
    format is defined in exactly one place.
    """
    values: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


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
    try:
        text = marker_path.read_text()
    except (OSError, UnicodeDecodeError) as exc:
        raise InvariantViolation(
            f"judge {mode} v{version}'s PROMOTED marker is not readable text "
            f"({type(exc).__name__}), so its hashes cannot be checked. Re-run `axial "
            "promote` on the current judge, or `axial judge --new` to score it as a fresh "
            "version."
        ) from exc
    recorded = parse_promoted_marker(text)
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
                    f"#### {title} — {_code_span(d.trace_id)}",
                    "",
                    f"**Human** (annotator {_code_span(d.annotator)}): {_inline_text(d.critique)}",
                    "",
                    f"**Judge:** {_inline_text(d.judge_reason)}",
                    "",
                ]
                lines += _excerpt_lines("Input", d.input_excerpt, d.input_length)
                lines += _excerpt_lines("Output", d.output_excerpt, d.output_length)
    lines.append(
        "Test disagreements are counted in the table above and never listed here, so the "
        "test split stays a held-out measurement."
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
    store.write_json(directory / "alignment.json", _json_safe(report))

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
    lines += [
        "",
        f"Quarantined calls (never scored): {n_errors}. See `errors.jsonl`.",
    ]
    if notes:
        lines += ["", "## Notes", ""]
        lines += [f"- {note}" for note in notes]
    if disagreements is not None:
        lines += [""] + _disagreement_section(mode, disagreements)
    lines += [""]
    path = directory / "alignment.md"
    path.write_text("\n".join(lines))
    return path


def promote(
    store: Store,
    mode: str,
    version: int,
    dev: Alignment,
    test: Alignment,
    thresholds: Thresholds,
) -> None:
    check = check_promotion(dev, test, thresholds)
    if not check.ok:
        raise InvariantViolation(
            f"judge {mode} v{version} cannot be promoted:\n  - " + "\n  - ".join(check.reasons)
        )
    directory = store.judge_dir(mode, version)
    directory.mkdir(parents=True, exist_ok=True)
    # Bind the marker to the exact judge.yaml and verdicts.jsonl it was earned
    # on. Without this, `gate` could only check that the marker existed, and
    # nothing stopped a judge from being scored on a fresh run (or a hand-edited
    # prompt) while the stale, unrelated PROMOTED marker stayed in place.
    judge_yaml_hash = store.hash_file(directory / "judge.yaml")
    verdicts_hash = store.hash_file(directory / "verdicts.jsonl")
    (directory / "PROMOTED").write_text(
        format_promoted_marker(dev.summary(), test.summary(), judge_yaml_hash, verdicts_hash)
    )
