"""The promotion gate: what makes a judge usable in CI, and what is checked before
one is trusted.

A judge becomes usable in CI only by passing `promote`, which reads the test split and
binds a PROMOTED marker to the exact judge.yaml, verdicts.jsonl and prompt it was
earned on. `gate` and `measure` call `require_promoted` before trusting a judge, so
they cannot disagree about what "promoted" means: a marker whose files or prompt have
changed since is refused here. `run_judge` deletes the marker when it rewrites the
verdicts.

A verdict row also records the trace it judged and the label it was scored against, and
`promote` refuses rows that no longer match the corpus and labels on disk. `gate` and
`measure` do not read the corpus for a judge without few-shot examples, so the data is
guarded where it changes instead: `ingest` and the label commands ask
`promotions_hit_by_traces` and `promotions_hit_by_labels` which promoted judges a change
would leave stale, and `guard_promotions` refuses the change or withdraws those
promotions.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

from axial.analysis.dataset import Label, read_labels
from axial.core.config import Thresholds
from axial.core.store import Store
from axial.core.traces import Trace, trace_sha256
from axial.errors import InvariantViolation, MalformedFile
from axial.judge.run import TRACE_HASH_KEY, read_errors, read_verdicts
from axial.judge.spec import JudgeSpec, current_prompt_sha256
from axial.stats.align import Alignment


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
PROMPT_HASH_KEY = "prompt_sha256"


def format_promoted_marker(
    dev_summary: str,
    test_summary: str,
    judge_yaml_hash: str,
    verdicts_hash: str,
    prompt_hash: str,
) -> str:
    """Render the `PROMOTED` marker: a human-readable summary plus three
    machine-parseable `key=value` lines binding the marker to the exact
    `judge.yaml`, `verdicts.jsonl` and prompt it was earned on.

    Without this binding, `gate` could only check that *some* `PROMOTED` file
    existed, not that the judge or its verdicts still matched what earned it
    -- which is exactly what let a promoted judge be silently replaced under
    its own marker. `require_promoted` recomputes the two file hashes with
    `store.hash_file`, recomputes the prompt fingerprint from the current
    judge.yaml, traces and labels, and calls `parse_promoted_marker` to compare
    all three against these. The prompt fingerprint covers the few-shot examples,
    so a relabelled example withdraws the promotion even though neither file changed.
    """
    return (
        f"test {test_summary}\n"
        f"dev {dev_summary}\n"
        f"{JUDGE_YAML_HASH_KEY}={judge_yaml_hash}\n"
        f"{VERDICTS_HASH_KEY}={verdicts_hash}\n"
        f"{PROMPT_HASH_KEY}={prompt_hash}\n"
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


class PromotedHashes(NamedTuple):
    judge_yaml_sha256: str
    verdicts_sha256: str
    prompt_sha256: str


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


def _hash_bound_file(store: Store, path: Path, mode: str, version: int) -> str:
    """`store.hash_file`, with a file it cannot read (a directory in its place, say)
    refused like every other broken binding."""
    try:
        return store.hash_file(path)
    except OSError as exc:
        raise InvariantViolation(
            f"judge {mode} v{version}'s {path.name} cannot be read ({exc.strerror}), so it "
            "cannot be checked against its PROMOTED marker. Restore the file, or run `axial "
            "judge --new` to score a fresh version."
        ) from exc


def _load_spec(store: Store, mode: str, version: int) -> JudgeSpec:
    if not (store.judge_dir(mode, version) / "judge.yaml").exists():
        raise InvariantViolation(
            f"judge {mode} v{version} has no judge.yaml, so the prompt its verdicts came "
            f"from cannot be checked. Run `axial judge {mode} --version {version} --split "
            "dev` and `--split test` first."
        )
    path = store.judge_dir(mode, version) / "judge.yaml"
    try:
        return JudgeSpec.load(store, mode, version)
    except (OSError, ValueError) as exc:
        # `JudgeSpec.load` reports malformed YAML, and a value the spec rejects such as
        # `decision_threshold: high`, as a MalformedFile whose detail is one line. Both
        # would otherwise reach the CLI as a traceback.
        # Both a MalformedFile and an OSError name the path this message names already.
        if isinstance(exc, MalformedFile):
            detail = exc.detail
        elif isinstance(exc, OSError) and exc.strerror:
            detail = exc.strerror
        else:
            lines = str(exc).strip().splitlines()
            detail = lines[0] if lines else type(exc).__name__
        raise InvariantViolation(
            f"judge {mode} v{version} ({path}) cannot be loaded: {detail}. "
            "Fix the file, or run `axial judge --new` to score a fresh version."
        ) from exc


def _splits_from_another_prompt(
    store: Store, directory: Path, fingerprint: str
) -> list[str | None]:
    """The splits holding a row not stamped with `fingerprint`, whatever the split.

    `axial judge --split train` writes into the same verdicts.jsonl as dev and test,
    and the judge-vs-judge gate counts every row, so a train row from another prompt
    is bound at promotion too. A row with no `split` key is reported as `None`.
    """
    stale: set[str | None] = set()
    rows = read_verdicts(store, directory / "verdicts.jsonl")
    for row in rows + read_errors(store, directory / "errors.jsonl"):
        if row.get(PROMPT_HASH_KEY) != fingerprint:
            split = row.get("split")
            stale.add(split if split is None else str(split))
    return sorted(s for s in stale if s is not None) + ([None] if None in stale else [])


def require_promoted(store: Store, mode: str, version: int) -> PromotedHashes:
    """Refuse unless judge `mode` v`version` is promoted and unchanged since; return
    the hashes its `PROMOTED` marker recorded.

    `gate` and `measure` both call this, so they cannot disagree about what
    "promoted" means. The marker on its own only proves *some* run was promoted
    once: both hashes it recorded are recomputed, because a hand-edited
    judge.yaml or a verdicts.jsonl overwritten by a later `run_judge` means the
    evidence behind PROMOTED no longer describes what is on disk. The prompt is
    recomputed too, from the current traces and labels, so a relabelled or re-ingested
    few-shot example withdraws the promotion.
    """
    directory = store.judge_dir(mode, version)
    marker_path = directory / "PROMOTED"
    if not marker_path.exists():
        raise InvariantViolation(
            f"judge {mode} v{version} has not been promoted. Run `axial promote` — a judge "
            "whose agreement with human labels was never measured on test cannot gate a build."
        )
    try:
        text = marker_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise InvariantViolation(
            f"judge {mode} v{version}'s PROMOTED marker is not readable text "
            f"({type(exc).__name__}), so its hashes cannot be checked. Re-run `axial "
            "promote` on the current judge, or `axial judge --new` to score it as a fresh "
            "version."
        ) from exc
    recorded = parse_promoted_marker(text)
    recorded_judge_hash = _require_bound_hash(recorded, JUDGE_YAML_HASH_KEY, mode, version)
    if _hash_bound_file(store, directory / "judge.yaml", mode, version) != recorded_judge_hash:
        raise InvariantViolation(
            f"judge {mode} v{version}'s judge.yaml has changed since it was promoted — "
            "the judge that earned PROMOTED is not the judge on disk. Re-run `axial "
            "promote` on the current judge, or run `axial judge --new` to score the "
            "changed prompt as a fresh version."
        )
    recorded_verdicts_hash = _require_bound_hash(recorded, VERDICTS_HASH_KEY, mode, version)
    verdicts_hash = _hash_bound_file(store, directory / "verdicts.jsonl", mode, version)
    if verdicts_hash != recorded_verdicts_hash:
        raise InvariantViolation(
            f"judge {mode} v{version}'s verdicts.jsonl has changed since it was promoted — "
            "the verdicts that earned PROMOTED are not the verdicts on disk. Re-run `axial "
            "promote` on the current run, or run `axial judge --new` to score it as a "
            "fresh version."
        )
    recorded_prompt_hash = _require_bound_hash(recorded, PROMPT_HASH_KEY, mode, version)
    if current_prompt_sha256(store, _load_spec(store, mode, version)) != recorded_prompt_hash:
        raise InvariantViolation(
            f"judge {mode} v{version} would send a prompt that is not the prompt it was "
            "promoted with. judge.yaml is unchanged, so either a few-shot example, resolved "
            "from the current traces and labels, was relabelled or re-ingested, or an axial "
            "upgrade changed how the prompt is built (a JudgeSpec field or default, the "
            "few-shot rendering, the answer schema, or the question a decision model is "
            "asked). Re-run "
            f"`axial judge {mode} --version {version} --split "
            "dev` and `--split test`, then `axial promote` again."
        )
    return PromotedHashes(recorded_judge_hash, recorded_verdicts_hash, recorded_prompt_hash)


# --- the evidence a promotion rests on -------------------------------------------------


class StaleVerdict(NamedTuple):
    """A verdict row that no longer describes the data on disk, and why."""

    trace_id: str
    split: str | None
    # One of the keys of `_STALE_REASONS`.
    reason: str


class PromotionHit(NamedTuple):
    """A promoted judge, and the verdicts of it a change would leave stale."""

    mode: str
    version: int
    stale: list[StaleVerdict]
    # Why its verdicts.jsonl is missing, unreadable or empty, naming the file, when it is:
    # then nothing says what the judge was scored on, so any change counts as hitting it.
    unreadable: str | None = None


# How each reason reads after "N trace(s)".
_STALE_REASONS = {
    "changed": "whose text changed after judging",
    "missing": "no longer in the corpus",
    "unhashed": "judged before axial recorded the text it judged",
    "relabelled": "relabelled after judging",
    "unlabelled": "with no label now",
}


def _split_of(row: Mapping[str, Any]) -> str | None:
    split = row.get("split")
    return split if split is None else str(split)


def _hashes_by_id(traces: Iterable[Trace]) -> dict[str, set[str]]:
    """Each id's `trace_sha256`s. Nothing refuses a repeated id, and `axial judge` grades
    every copy, so an id can hold several traces and a verdict judged one of them."""
    hashes: dict[str, set[str]] = {}
    for trace in traces:
        hashes.setdefault(trace.trace_id, set()).add(trace_sha256(trace))
    return hashes


def _trace_problem(
    row: Mapping[str, Any], corpus: Mapping[str, set[str]], judged: set[str]
) -> str | None:
    """Why the row's id in `corpus` (hashes by id) no longer holds every trace the row
    may have judged, whose hashes are `judged` (empty when nothing recorded them), or
    `None` when it does. A row's own hash is one trace; a row with none may have judged
    any copy its id held, so every one of them has to survive."""
    now = corpus.get(str(row["trace_id"]))
    if now is None:
        return "missing"
    if not judged:
        return "unhashed"
    if not judged <= now:
        return "changed"
    return None


def _judged(row: Mapping[str, Any]) -> set[str]:
    """The hash the row recorded of the trace it judged, as a set, empty if none."""
    judged = row.get(TRACE_HASH_KEY)
    return {str(judged)} if judged is not None else set()


def _label_problem(row: Mapping[str, Any], labels: Mapping[str, bool]) -> str | None:
    """Why the row's label in `labels` is not the one it was scored against, or `None`.

    A row that recorded no label is left alone: alignment refuses it on its own.
    """
    truth = row.get("truth")
    if truth is None:
        return None
    label = labels.get(str(row["trace_id"]))
    if label is None:
        return "unlabelled"
    if label != bool(truth):
        return "relabelled"
    return None


def _verdict_rows(store: Store, mode: str, version: int) -> list[dict[str, Any]]:
    return read_verdicts(store, store.judge_dir(mode, version) / "verdicts.jsonl")


def _guarded_rows(
    store: Store, mode: str, version: int
) -> tuple[list[dict[str, Any]], PromotionHit | None]:
    """A promoted judge's verdict rows, or, when its verdicts.jsonl is missing, cannot be
    read or holds none, the hit that says so. Its gate already fails then, as the file
    no longer matches its marker, so a change is refused only until
    `--withdraw-promotions` withdraws it."""
    path = store.judge_dir(mode, version) / "verdicts.jsonl"
    if not path.exists():
        return [], PromotionHit(mode, version, [], unreadable=f"{path}: no such file")
    try:
        rows = read_verdicts(store, path)
    except MalformedFile as exc:
        return [], PromotionHit(mode, version, [], unreadable=str(exc))
    except OSError as exc:
        return [], PromotionHit(mode, version, [], unreadable=f"{path}: {exc.strerror}")
    if not rows:
        # Promotion was earned on verdicts, so a file with none has lost them.
        return [], PromotionHit(mode, version, [], unreadable=f"{path}: holds no verdicts")
    return rows, None


def stale_verdicts(store: Store, mode: str, version: int) -> list[StaleVerdict]:
    """The verdicts of judge `mode` v`version` that no longer describe the corpus and
    labels on disk: their trace is gone or its text changed, their label changed, or
    they recorded no hash of the trace they judged, so whether it changed is unknown."""
    rows = _verdict_rows(store, mode, version)
    if not rows:
        return []
    corpus = _hashes_by_id(store.read_traces())
    labels = {lbl.trace_id: lbl.label for lbl in read_labels(store, mode)}
    stale: list[StaleVerdict] = []
    for row in rows:
        reason = _trace_problem(row, corpus, _judged(row)) or _label_problem(row, labels)
        if reason:
            stale.append(StaleVerdict(str(row["trace_id"]), _split_of(row), reason))
    return stale


def _promoted_judges(store: Store, mode: str | None = None) -> list[tuple[str, int]]:
    return [
        (m, v)
        for m, v in store.judge_versions()
        if (mode is None or m == mode) and (store.judge_dir(m, v) / "PROMOTED").exists()
    ]


def promotions_hit_by_traces(store: Store, traces: list[Trace]) -> list[PromotionHit]:
    """The promoted judges whose verdicts `traces`, written as the new corpus, would
    leave stale: a trace they judged would be gone or have other text.

    A row compares the new text with the hash it recorded. A row written before axial
    recorded one is compared with the corpus being replaced, the nearest thing to what
    it judged that is left, and counts as stale when that corpus cannot be read or does
    not hold its trace, since then nothing says what it judged.
    """
    new = _hashes_by_id(traces)
    replaced: dict[str, set[str]] | None = None
    hits: list[PromotionHit] = []
    for mode, version in _promoted_judges(store):
        rows, broken = _guarded_rows(store, mode, version)
        if broken:
            hits.append(broken)
            continue
        stale: list[StaleVerdict] = []
        for row in rows:
            judged = _judged(row)
            if not judged:
                if replaced is None:
                    try:
                        replaced = _hashes_by_id(store.read_traces())
                    except (OSError, ValueError):  # unreadable, so nothing can be compared
                        replaced = {}
                judged = replaced.get(str(row["trace_id"]), set())
            reason = _trace_problem(row, new, judged)
            if reason:
                stale.append(StaleVerdict(str(row["trace_id"]), _split_of(row), reason))
        if stale:
            hits.append(PromotionHit(mode, version, stale))
    return hits


def promotions_hit_by_labels(store: Store, mode: str, labels: list[Label]) -> list[PromotionHit]:
    """The promoted judges of `mode` whose verdicts `labels`, added to the mode's labels,
    would leave stale: a trace they judged would get the other label. A new critique
    under the same label changes no verdict's score, so it hits nothing."""
    incoming = {lbl.trace_id: lbl.label for lbl in labels}
    hits: list[PromotionHit] = []
    for judge_mode, version in _promoted_judges(store, mode):
        rows, broken = _guarded_rows(store, judge_mode, version)
        if broken:
            hits.append(broken)
            continue
        stale = [
            StaleVerdict(str(row["trace_id"]), _split_of(row), reason)
            for row in rows
            if str(row["trace_id"]) in incoming
            and (reason := _label_problem(row, incoming)) is not None
        ]
        if stale:
            hits.append(PromotionHit(judge_mode, version, stale))
    return hits


def _stale_lines(stale: list[StaleVerdict], prefix: str = "") -> list[str]:
    """One line per split and reason, naming up to three traces."""
    groups: dict[tuple[str | None, str], list[str]] = {}
    for verdict in stale:
        groups.setdefault((verdict.split, verdict.reason), []).append(verdict.trace_id)

    def order(key: tuple[str | None, str]) -> tuple[bool, str, str]:
        split, reason = key
        return split is None, split or "", reason

    lines = []
    for (split, reason), ids in sorted(groups.items(), key=lambda item: order(item[0])):
        named = ", ".join(ids[:3]) + (", ..." if len(ids) > 3 else "")
        where = split if split is not None else "rows with no split"
        lines.append(
            f"  - {prefix}{where}: {len(ids)} trace(s) {_STALE_REASONS[reason]} ({named})"
        )
    return lines


def _missing_note(stale: list[StaleVerdict]) -> str:
    """For verdicts whose trace left the corpus: judging their split again without the
    trace drops the verdict, so say how to bring the trace back too."""
    if not any(verdict.reason == "missing" for verdict in stale):
        return ""
    return (
        " A trace no longer in the corpus is judged again only once it is ingested "
        "again; judging its split without it drops its verdict."
    )


def _rerun_advice(mode: str, version: int, splits: Iterable[str | None]) -> str:
    """What to run so the rows of `splits` are judged again, then promoted."""
    found = set(splits)
    named = sorted(s for s in found if s is not None)
    advice: list[str] = []
    if named:
        rerun = " and ".join(
            f"`axial judge {mode} --version {version} --split {split}`" for split in named
        )
        advice.append(f"Run {rerun}, then promote.")
    if None in found:
        advice.append(
            "Rows with no split cannot be re-run by split; run `axial judge --new` to "
            "score a fresh version."
        )
    return " ".join(advice)


def guard_promotions(store: Store, hits: list[PromotionHit], *, withdraw: bool) -> list[str]:
    """Refuse a change that `hits` says would leave promoted judges' verdicts stale, or,
    with `withdraw`, withdraw those promotions so the change can be written, and say so.

    Called before the change is written, as `run_judge` deletes PROMOTED before it
    rewrites the verdicts: a write that then fails leaves a judge unpromoted, and `axial
    promote` restores it while its verdicts still match the data. The notes it returns
    describe a written change, so they are shown once the write succeeds; a failed write
    says what was withdrawn with `unwritten_note` instead.
    """
    if not hits:
        return []
    if not withdraw:
        for hit in hits:
            if hit.unreadable:
                # One line, so a damaged file reads as one refusal naming it once.
                others = [
                    f"{h.mode} v{h.version}, "
                    + (
                        "whose verdicts cannot be read either"
                        if h.unreadable
                        else "which this change makes stale"
                    )
                    for h in hits
                    if h is not hit
                ]
                also = f"; that also withdraws {'; '.join(others)}" if others else ""
                raise InvariantViolation(
                    f"the verdicts of promoted judge {hit.mode} v{hit.version} cannot be "
                    f"read ({hit.unreadable}), so whether this change alters what it was "
                    "scored on cannot be checked. Nothing was written. Pass "
                    "--withdraw-promotions to withdraw its promotion and write the change"
                    f"{also}."
                )
        lines = [
            line
            for hit in hits
            for line in _stale_lines(hit.stale, prefix=f"{hit.mode} v{hit.version}, ")
        ]
        stale = [verdict for hit in hits for verdict in hit.stale]
        raise InvariantViolation(
            "this change would alter what promoted judges were scored on, so their "
            "promotions would no longer describe the data:\n"
            + "\n".join(lines)
            + "\nNothing was written. Pass --withdraw-promotions to write it and withdraw "
            "these promotions; `axial promote` then refuses each judge until the splits "
            "named above are judged again."
            + _missing_note(stale)
        )
    notes = []
    for hit in hits:
        (store.judge_dir(hit.mode, hit.version) / "PROMOTED").unlink(missing_ok=True)
        if hit.unreadable:
            notes.append(
                f"withdrew the promotion of {hit.mode} v{hit.version}: its verdicts could "
                f"not be read ({hit.unreadable}). Restore the file, or run `axial judge "
                "--new` to score a fresh version."
            )
            continue
        notes.append(
            f"withdrew the promotion of {hit.mode} v{hit.version}: {len(hit.stale)} of its "
            "verdicts no longer describe the data. "
            + _rerun_advice(hit.mode, hit.version, (s.split for s in hit.stale))
            + _missing_note(hit.stale)
        )
    return notes


def unwritten_note(hits: list[PromotionHit]) -> str:
    """What to add to a failed write's message once `guard_promotions` withdrew `hits`:
    the notes it returned say the data changed, and it did not."""
    if not hits:
        return ""
    judges = ", ".join(f"{hit.mode} v{hit.version}" for hit in hits)
    return (
        f". The promotion of {judges} was withdrawn first and nothing was written; "
        "`axial promote` restores a judge whose verdicts can be read and still match the "
        "data"
    )


def promote(
    store: Store,
    mode: str,
    version: int,
    dev: Alignment,
    test: Alignment,
    thresholds: Thresholds,
) -> None:
    directory = store.judge_dir(mode, version)
    fingerprint = current_prompt_sha256(store, _load_spec(store, mode, version))
    stale = _splits_from_another_prompt(store, directory, fingerprint)
    if stale:
        named = [split for split in stale if split is not None]
        subject = f"its {' and '.join(named)} rows" if named else ""
        if None in stale:
            subject += " and its rows with no split" if named else "its rows with no split"
        raise InvariantViolation(
            f"judge {mode} v{version} cannot be promoted: {subject} "
            "were not produced by the prompt its judge.yaml, traces and labels give now. "
            "judge.yaml was edited, a few-shot example was relabelled or re-ingested, or an "
            "axial upgrade changed how the prompt is built, after they ran. "
            f"{_rerun_advice(mode, version, stale)}"
        )
    stale_rows = stale_verdicts(store, mode, version)
    if stale_rows:
        raise InvariantViolation(
            f"judge {mode} v{version} cannot be promoted: some of its verdicts no longer "
            "describe the traces and labels on disk.\n"
            + "\n".join(_stale_lines(stale_rows))
            + "\n"
            + _rerun_advice(mode, version, (s.split for s in stale_rows))
            + _missing_note(stale_rows)
        )
    check = check_promotion(dev, test, thresholds)
    if not check.ok:
        raise InvariantViolation(
            f"judge {mode} v{version} cannot be promoted:\n  - " + "\n  - ".join(check.reasons)
        )
    directory.mkdir(parents=True, exist_ok=True)
    # Bind the marker to the exact judge.yaml and verdicts.jsonl it was earned
    # on. Without this, `gate` could only check that the marker existed, and
    # nothing stopped a judge from being scored on a fresh run (or a hand-edited
    # prompt) while the stale, unrelated PROMOTED marker stayed in place.
    judge_yaml_hash = store.hash_file(directory / "judge.yaml")
    verdicts_hash = store.hash_file(directory / "verdicts.jsonl")
    (directory / "PROMOTED").write_text(
        format_promoted_marker(
            dev.summary(), test.summary(), judge_yaml_hash, verdicts_hash, fingerprint
        ),
        encoding="utf-8",
    )
