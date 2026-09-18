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
    def __init__(self, ok: bool, reasons: list[str]) -> None:
        self.ok = ok
        self.reasons = reasons


def check_promotion(dev: Alignment, test: Alignment, thresholds: Thresholds) -> PromotionCheck:
    reasons: list[str] = []
    if math.isnan(test.tpr) or test.tpr < thresholds.min_tpr:
        reasons.append(
            f"test TPR {test.tpr:.3f} is below the bar of {thresholds.min_tpr:.2f} — "
            "the judge misses real failures"
        )
    if math.isnan(test.tnr) or test.tnr < thresholds.min_tnr:
        reasons.append(
            f"test TNR {test.tnr:.3f} is below the bar of {thresholds.min_tnr:.2f} — "
            "the judge cries wolf"
        )
    for name in ("tpr", "tnr"):
        dev_value = getattr(dev, name)
        test_value = getattr(test, name)
        if math.isnan(dev_value) or math.isnan(test_value):
            continue
        gap = dev_value - test_value
        if gap > thresholds.max_gap:
            reasons.append(
                f"dev->test gap on {name.upper()} is {gap:.3f}, above the tolerance of "
                f"{thresholds.max_gap:.2f} — the judge prompt is tuned to dev"
            )
    return PromotionCheck(ok=not reasons, reasons=reasons)


def write_alignment(
    store: Store,
    mode: str,
    version: int,
    dev: Alignment,
    test: Alignment | None,
    n_errors: int,
) -> Path:
    directory = store.judge_dir(mode, version)
    directory.mkdir(parents=True, exist_ok=True)

    store.write_json(
        directory / "alignment.json",
        {
            "mode": mode,
            "version": version,
            "dev": dev.to_dict(),
            "test": test.to_dict() if test else None,
            "quarantined_errors": n_errors,
        },
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
        "",
    ]
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
