"""Ground-truth labels and the train/dev/test split.

Two refusals live here. Machine-generated labels may not enter a dataset, because a
judge aligned to machine labels is aligned to itself. And the test split is
write-once: once saved, it can only be re-saved identically, so a number reported
against test cannot quietly become a number reported against a split redrawn until
it flattered the judge.

The machine-annotator check is a speed bump, not a security boundary. It is a
name-pattern match against a family list of known model/vendor terms, and it errs
broad on purpose: a false negative here silently poisons ground truth, so every
downstream alignment number becomes self-referential. A false positive is loud and
a human can work around it by picking a different annotator name. But name-pattern
matching cannot stop someone determined to record machine output under a human
name — if that is a real risk for your team, label the data yourself, or record the
machine's output as a judge run to be validated, not as ground truth.
"""

from __future__ import annotations

import re
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field, field_validator

from axial.errors import InvariantViolation
from axial.sample import Review
from axial.store import Store

# Distinctive vendor/model terms. These are matched as a substring of the
# annotator name (after separators are normalised to "-" and digits are
# stripped), because they are unlikely to appear inside a real human name and
# stripping digits catches version suffixes and mild obfuscation ("gpt-4o",
# "cla8ude", "llama-3-70b") without having to enumerate every version number.
# An empirical probe of real human names found no false positive from any of
# these (notably "cohere" does not match "Cohen") — the ones that did collide
# with real names live in `_MACHINE_VENDOR_SEGMENT_TERMS` below instead.
_MACHINE_SUBSTRING_TERMS = frozenset(
    {
        "gpt",
        "claude",
        "opus",
        "sonnet",
        "haiku",
        "gemini",
        "llama",
        "mistral",
        "mixtral",
        "deepseek",
        "qwen",
        "cohere",
        "vicuna",
    }
)

# OpenAI's "o" series is a family of its own. It used to be matched as a raw,
# digit-included substring, but that refused any name that merely *contained*
# "o1"/"o3"/"o4" as a run of characters, with no regard for word boundaries —
# e.g. "o1iver" or "Theo3dore". Matched as a whole "-"-delimited segment
# instead, so the digits have to stand on their own. Digit-stripping isn't an
# option here: stripping the digit from "o1"/"o3"/"o4" would leave the single
# letter "o", far too common to trust, so these are compared against the
# *un-stripped* segment and (unlike the vendor-family terms below) match
# wherever they appear, single segment or not — a bare "o3" is exactly as
# suspicious as "o3" inside "o3-mini".
_MACHINE_RAW_SEGMENT_TERMS = frozenset({"o1", "o3", "o4"})

# Vendor/family terms that collide with real given and family names when
# matched as a plain substring or a plain segment: an empirical probe found
# "phi" refusing the entire Soph-/Phil-/-phine family, and "nova", "gemma",
# "palm", "bard", "titan", "falcon", "command", and "grok" refusing Nova,
# Novak, Gemma, Palmer, Bardia, Titania, Falconer, Grokowski, and usernames
# like "p.nova" and "d.palmer". Real model identifiers put the family term
# anywhere but last: it leads a bare model name ("nova-pro", "falcon-40b",
# "command-r-plus", ...) and it leads or sits mid-identifier once a provider
# prefix is involved, as litellm-style "provider/model" ids normalise to
# ("google-gemma-2-9b", "openrouter-google-gemma-2", ...). A human name that
# happens to collide, by contrast, either stands alone ("Nova", "Gemma") or
# has the colliding word fall at the end, trailing an unrelated lead segment
# ("p-nova", "d-palmer") — never leading or mid-word. So a match here
# requires *all* of:
#   - the term equals some "-"-delimited segment OTHER THAN THE LAST,
#     digit-stripped (so "grok-4", "nova-pro", and a provider-prefixed
#     "google-gemma-2-9b" all match, but a trailing occurrence — "p-nova",
#     "d-palmer" — does not, because real vendor names never trail an
#     unrelated segment); and
#   - either there is more than one segment (so "nova-pro" matches), or this
#     lone segment needed digit-stripping to reach the term (so "grok4"
#     matches, since with only one segment "not the last" is vacuous and this
#     second condition is what still lets a bare digit-suffixed form through).
#     A clean, undecorated single-word match ("Nova", "Gemma") is just as
#     likely to be a genuine given name as a machine label, so it is not
#     trusted on its own — see the module docstring's false-positive/
#     false-negative tradeoff.
# The accepted new false positive from widening "first" to "any but last" is
# a medial human form like "Anna Nova Smith"; that's squarely inside the
# "speed bump, not a security boundary" framing above, same as the
# unseparated run-together forms ("phi3mini") that still escape this.
_MACHINE_VENDOR_SEGMENT_TERMS = frozenset(
    {
        "phi",
        "nova",
        "gemma",
        "palm",
        "bard",
        "titan",
        "falcon",
        "command",
        "grok",
    }
)

# Generic markers, short or common enough that a substring match would catch
# plain human names mid-word (e.g. "bot" inside "Abbott", "auto" inside
# "Autumn"). These are only refused when they appear as a whole "-"-delimited
# segment of the normalised annotator name, never as a raw substring.
#
# "ai" used to live in this set too, matched only when it was the *entire*
# annotator name — refusing bare "Ai" (a real Japanese given name) while
# letting "Ai Nakamura" through unrefused. That's backwards: the single-
# segment case is exactly the one most likely to be a genuine short given
# name, not less likely. Rather than flip the asymmetry the other way (which
# would start refusing "Ai Nakamura" instead), "ai" is no longer trusted as
# machine-name evidence in either shape. It is too short and too common a
# name/initial to carry this signal on its own, and a quiet false negative
# here is exactly the tradeoff this module's docstring says to accept.
_MACHINE_SEGMENT_TERMS = frozenset(
    {
        "llm",
        "model",
        "machine",
        "auto",
        "bot",
        "agent",
        "judge",
        "synthetic",
        "generated",
        "axial",
    }
)

# Kept for backward compatibility / introspection: the full family of terms
# this module refuses, across all matching strategies above.
MACHINE_ANNOTATORS = frozenset(
    _MACHINE_SUBSTRING_TERMS
    | _MACHINE_RAW_SEGMENT_TERMS
    | _MACHINE_VENDOR_SEGMENT_TERMS
    | _MACHINE_SEGMENT_TERMS
)

MIN_USABLE_LABELS = 60

_SEPARATOR_RE = re.compile(r"[_/.:@\s]+")


def _looks_machine(annotator: str) -> bool:
    """Return True if `annotator` matches the machine-annotator family list.

    This is a name-pattern heuristic, not proof of authorship: it exists to
    catch the obvious, common cases (a model's own name or an obvious
    "bot"/"synthetic"/"judge" label left in the annotator field), not to
    withstand someone determined to record machine output under a human name.
    """
    normalized = _SEPARATOR_RE.sub("-", annotator.strip().lower())
    segments = [s for s in normalized.split("-") if s]
    digit_stripped = re.sub(r"\d+", "", normalized)

    if any(term in digit_stripped for term in _MACHINE_SUBSTRING_TERMS):
        return True
    if any(term in segments for term in _MACHINE_RAW_SEGMENT_TERMS):
        return True
    if any(term in segments for term in _MACHINE_SEGMENT_TERMS):
        return True
    for i, raw in enumerate(segments):
        stripped = re.sub(r"\d+", "", raw)
        if stripped in _MACHINE_VENDOR_SEGMENT_TERMS and (
            i < len(segments) - 1 or raw != stripped
        ):
            return True
    return False


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Label(BaseModel):
    trace_id: str
    label: bool
    critique: str
    annotator: str
    ts: str = Field(default_factory=_now)

    @field_validator("trace_id", mode="before")
    @classmethod
    def _integer_id_as_text(cls, value: object) -> object:
        # `ingest` stores an integer trace id as its decimal string, so a labels file
        # exported alongside the traces must name trace 123 the same way to match it.
        # A float or a bool has no one spelling and is refused, as it is for traces.
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return value


def _check_human(label: Label) -> None:
    annotator = (label.annotator or "").strip()
    if not annotator:
        raise InvariantViolation(
            f"label for {label.trace_id!r} has no annotator; every label must name the "
            "human who made it"
        )
    if _looks_machine(annotator):
        raise InvariantViolation(
            f"annotator {annotator!r} looks like a model, not a human. Ground-truth labels "
            "must be human, because a judge aligned to machine labels is aligned to itself. "
            "This is a name-pattern speed bump, not a security boundary — it catches the "
            "obvious cases and nothing more; someone determined to record machine labels "
            "under a human name can. If that's a real concern here, label the data yourself, "
            "or record the machine's output as a judge run to be validated, not as ground "
            "truth."
        )
    if not label.critique.strip():
        raise InvariantViolation(
            f"label for {label.trace_id!r} has an empty critique. The critique carries the "
            "reasoning that becomes few-shot material and is how disagreements get settled."
        )


def _labels_path(store: Store, mode: str) -> Path:
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
    return store.read_models(_labels_path(store, mode), Label)


def labels_from_reviews(reviews: list[Review]) -> tuple[list[Label], int]:
    """Convert completed annotate-app reviews into labels for one failure mode.

    The annotate app already records `verdict`, `critique` and `annotator` per
    review -- that is a `Label` in all but name. `verdict == "fail"` means the
    failure mode graded in this review session IS present, so it becomes
    `label=True`; `"pass"` becomes `label=False`. A `pending` review carries no
    verdict yet and is skipped rather than defaulted to a pass, and the skip
    count is returned so a caller can report it instead of silently losing
    rows.

    This does not itself enforce the human-annotator or non-blank-critique
    rules -- the labels it returns still go through `add_labels`, so every
    existing refusal there (a blank critique, a machine-looking annotator)
    still applies to labels that arrive this way.
    """
    labels: list[Label] = []
    skipped = 0
    for review in reviews:
        if review.status != "done":
            skipped += 1
            continue
        labels.append(
            Label(
                trace_id=review.trace_id,
                label=review.verdict == "fail",
                critique=review.critique or "",
                annotator=review.annotator or "",
            )
        )
    return labels, skipped


class Splits(BaseModel):
    train: list[str] = Field(default_factory=list)
    dev: list[str] = Field(default_factory=list)
    test: list[str] = Field(default_factory=list)
    seed: int = 0
    test_hash: str = ""

    def save(self, store: Store, mode: str) -> None:
        recomputed_hash = store.hash_ids(self.test)
        if self.test_hash and self.test_hash != recomputed_hash:
            raise InvariantViolation(
                f"the supplied test_hash {self.test_hash!r} does not match the hash "
                f"recomputed from this split's own test ids ({recomputed_hash!r}). A "
                "caller who supplies a test_hash means something by it, so this is refused "
                "rather than silently repaired — pass an empty test_hash to have it "
                "computed for you, or fix the mismatch."
            )
        path = store.dataset_dir(mode) / "splits.json"
        if path.exists():
            previous = store.read_json_model(path, Splits)
            if sorted(previous.test) != sorted(self.test):
                raise InvariantViolation(
                    f"the test split for {mode!r} is write-once and already exists. "
                    "Redrawing it invalidates every number previously reported against it. "
                    "Delete it deliberately if you really mean to start over."
                )
        to_store = self.model_copy(update={"test_hash": recomputed_hash})
        store.write_json(path, to_store.model_dump())

    @classmethod
    def load(cls, store: Store, mode: str) -> Splits:
        return store.read_json_model(store.dataset_dir(mode) / "splits.json", cls)


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
    """Check that the saved test split's ids still match its recorded hash.

    Both the ids and the hash live in the same file, and the hash is an unkeyed,
    truncated sha256 of the ids — so this can only catch a *partial* edit, one
    that adds or removes an id without recomputing test_hash to match. It
    cannot catch a full rewrite that recomputes the hash correctly, deletion
    and regeneration of the whole split, or drift from a number published
    elsewhere. It is a tripwire for accidental or careless edits, not
    cryptographic tamper evidence.
    """
    splits = Splits.load(store, mode)
    if store.hash_ids(splits.test) != splits.test_hash:
        raise InvariantViolation(
            f"the test split for {mode!r} does not match its recorded hash — some ids were "
            "added or removed without updating test_hash to match. This only catches that "
            "kind of partial edit, not a full rewrite with a recomputed hash, so treat this "
            "as a sign of accidental tampering, not proof there was none."
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
