"""Ground-truth labels and the train/dev/test split.

Two refusals live here. Machine-generated labels may not enter a dataset, because a
judge aligned to machine labels is aligned to itself. And the test split is
write-once: once saved, it can only be re-saved identically, so a number reported
against test cannot quietly become a number reported against a split redrawn until
it flattered the judge.
"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime

import numpy as np
from pydantic import BaseModel, Field

from axial.errors import InvariantViolation
from axial.store import Store

MACHINE_ANNOTATORS = frozenset(
    {"gpt", "gpt-4", "gpt-4o", "claude", "llm", "model", "machine", "auto", "axial"}
)

MIN_USABLE_LABELS = 60


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Label(BaseModel):
    trace_id: str
    label: bool
    critique: str
    annotator: str
    ts: str = Field(default_factory=_now)


def _check_human(label: Label) -> None:
    annotator = (label.annotator or "").strip()
    if not annotator:
        raise InvariantViolation(
            f"label for {label.trace_id!r} has no annotator; every label must name the "
            "human who made it"
        )
    stem = annotator.lower().split("-")[0].split("/")[-1]
    if stem in MACHINE_ANNOTATORS or annotator.lower() in MACHINE_ANNOTATORS:
        raise InvariantViolation(
            f"annotator {annotator!r} looks like a model, not a human. Ground-truth labels "
            "must be human. A judge aligned to machine labels is aligned to itself."
        )
    if not label.critique.strip():
        raise InvariantViolation(
            f"label for {label.trace_id!r} has an empty critique. The critique carries the "
            "reasoning that becomes few-shot material and is how disagreements get settled."
        )


def _labels_path(store: Store, mode: str):
    return store.dataset_dir(mode) / "labels.jsonl"


def add_labels(store: Store, mode: str, labels: list[Label]) -> None:
    for label in labels:
        _check_human(label)
    existing = {lbl.trace_id: lbl for lbl in read_labels(store, mode)}
    for label in labels:
        existing[label.trace_id] = label
    store.write_jsonl(
        _labels_path(store, mode),
        [existing[k].model_dump() for k in sorted(existing)],
    )


def read_labels(store: Store, mode: str) -> list[Label]:
    return [Label.model_validate(r) for r in store.read_jsonl(_labels_path(store, mode))]


class Splits(BaseModel):
    train: list[str] = Field(default_factory=list)
    dev: list[str] = Field(default_factory=list)
    test: list[str] = Field(default_factory=list)
    seed: int = 0
    test_hash: str = ""

    def save(self, store: Store, mode: str) -> None:
        path = store.dataset_dir(mode) / "splits.json"
        if path.exists():
            previous = Splits.model_validate(store.read_json(path))
            if sorted(previous.test) != sorted(self.test):
                raise InvariantViolation(
                    f"the test split for {mode!r} is write-once and already exists. "
                    "Redrawing it invalidates every number previously reported against it. "
                    "Delete it deliberately if you really mean to start over."
                )
        store.write_json(path, self.model_dump())

    @classmethod
    def load(cls, store: Store, mode: str) -> Splits:
        return cls.model_validate(store.read_json(store.dataset_dir(mode) / "splits.json"))


def make_splits(
    store: Store,
    labels: list[Label],
    *,
    seed: int = 0,
    fractions: tuple[float, float, float] = (0.15, 0.42, 0.43),
) -> Splits:
    if len(labels) < MIN_USABLE_LABELS:
        warnings.warn(
            f"{len(labels)} labels is below the usable floor of {MIN_USABLE_LABELS}; "
            "confidence intervals on alignment will be too wide to act on",
            UserWarning,
            stacklevel=2,
        )

    rng = np.random.default_rng(seed)
    buckets: dict[bool, list[str]] = {True: [], False: []}
    for label in sorted(labels, key=lambda x: x.trace_id):
        buckets[label.label].append(label.trace_id)

    train: list[str] = []
    dev: list[str] = []
    test: list[str] = []
    for klass in (True, False):
        ids = buckets[klass]
        order = rng.permutation(len(ids))
        shuffled = [ids[i] for i in order]
        n_train = round(len(shuffled) * fractions[0])
        n_dev = round(len(shuffled) * fractions[1])
        train.extend(shuffled[:n_train])
        dev.extend(shuffled[n_train : n_train + n_dev])
        test.extend(shuffled[n_train + n_dev :])

    return Splits(
        train=sorted(train),
        dev=sorted(dev),
        test=sorted(test),
        seed=seed,
        test_hash=store.hash_ids(test),
    )


def verify_test_manifest(store: Store, mode: str) -> None:
    splits = Splits.load(store, mode)
    if store.hash_ids(splits.test) != splits.test_hash:
        raise InvariantViolation(
            f"the test split for {mode!r} does not match its recorded hash — it has been "
            "edited since it was frozen. Every number reported against it is suspect."
        )


def class_balance_report(labels: list[Label], splits: Splits) -> dict[str, dict[str, int]]:
    by_id = {lbl.trace_id: lbl.label for lbl in labels}
    report: dict[str, dict[str, int]] = {}
    for name, ids in (("train", splits.train), ("dev", splits.dev), ("test", splits.test)):
        report[name] = {
            "fail": sum(1 for i in ids if by_id.get(i)),
            "pass": sum(1 for i in ids if by_id.get(i) is False),
        }
    return report
