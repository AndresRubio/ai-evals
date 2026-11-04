"""The promotion gate: what makes a judge usable in CI, and what is checked before
one is trusted.

A judge becomes usable in CI only by passing `promote`, which reads the test split and
binds a PROMOTED marker to the exact judge.yaml, verdicts.jsonl and prompt it was
earned on. `gate` and `measure` call `require_promoted` before trusting a judge, so
they cannot disagree about what "promoted" means: a marker whose files or prompt have
changed since is refused here. `run_judge` deletes the marker when it rewrites the
verdicts.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import NamedTuple

from axial.align import Alignment
from axial.config import Thresholds
from axial.errors import InvariantViolation, MalformedFile
from axial.judge import JudgeSpec, current_prompt_sha256, read_errors, read_verdicts
from axial.store import Store


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
        rerun = " and ".join(
            f"`axial judge {mode} --version {version} --split {split}`" for split in named
        )
        advice: list[str] = []
        if named:
            advice.append(f"Run {rerun}, then promote.")
        if None in stale:
            advice.append(
                "Rows with no split cannot be re-run by split; run `axial judge --new` to "
                "score a fresh version."
            )
        raise InvariantViolation(
            f"judge {mode} v{version} cannot be promoted: {subject} "
            "were not produced by the prompt its judge.yaml, traces and labels give now. "
            "judge.yaml was edited, a few-shot example was relabelled or re-ingested, or an "
            "axial upgrade changed how the prompt is built, after they ran. "
            f"{' '.join(advice)}"
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
