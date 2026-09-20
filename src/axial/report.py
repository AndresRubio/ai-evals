"""Alignment reports and the promotion gate.

A judge becomes usable in CI only by passing `promote`, and `promote` reads the
test split. Everything it prints leads with TPR and TNR.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from axial.align import Alignment, align
from axial.config import Thresholds
from axial.errors import InvariantViolation
from axial.store import Store


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


def write_alignment(
    store: Store,
    mode: str,
    version: int,
    dev: Alignment,
    test: Alignment | None,
    n_errors: int,
    notes: list[str] | None = None,
) -> Path:
    directory = store.judge_dir(mode, version)
    directory.mkdir(parents=True, exist_ok=True)

    store.write_json(
        directory / "alignment.json",
        _json_safe(
            {
                "mode": mode,
                "version": version,
                "dev": dev.to_dict(),
                "test": test.to_dict() if test else None,
                "quarantined_errors": n_errors,
            }
        ),
    )

    lines = [
        f"# Alignment — {mode} v{version}",
        "",
        "Agreement between this judge and the human labels. TPR is the share of real",
        "failures the judge caught; TNR is the share of clean traces it left alone.",
        "Accuracy alone is not reported, because under class imbalance it is the number",
        "that gets a judge which detects nothing shipped.",
        "",
        "| split | n | TPR | TNR | kappa | accuracy |",
        "|---|---|---|---|---|---|",
    ]
    for name, result in (("dev", dev), ("test", test)):
        if result is None:
            continue
        lines.append(
            f"| {name} | {result.n} | {result.tpr:.3f} | {result.tnr:.3f} | "
            f"{result.kappa:.3f} | {result.accuracy:.3f} |"
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
    (directory / "PROMOTED").write_text(f"test {test.summary()}\ndev {dev.summary()}\n")
