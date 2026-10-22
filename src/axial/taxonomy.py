"""Open codes to a failure taxonomy.

The LLM may propose groupings; a human accepts them. The refusal below is the
load-bearing part of this module: no taxonomy may be proposed until enough traces
carry human open codes, because open coding is where failure modes nobody
anticipated are discovered, and a model cannot discover what it was not shown.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from axial.errors import InvariantViolation, validation_message
from axial.sample import Review
from axial.store import MAX_NAME_LENGTH, Store

PROMPT = """\
You are helping with axial coding of qualitative annotations on LLM traces.

Below are free-text open codes an engineer wrote while reading traces of their
system, one per trace. Group them into distinct failure modes.

Rules:
- Derive the modes from the codes given. Do not invent failure modes that no code
  describes.
- Prefer few, well-separated modes over many overlapping ones.
- Each mode needs an id (snake_case), a short name, and a one-sentence definition
  that someone could apply to a new trace without seeing these codes.

Open codes:
{codes}

Return JSON only, in this shape:
{{"modes": [{{"id": "...", "name": "...", "definition": "...", "examples": ["trace_id"]}}]}}
"""


class FailureMode(BaseModel):
    id: str
    name: str
    definition: str
    examples: list[str] = Field(default_factory=list)
    prevalence: int = 0


class Taxonomy(BaseModel):
    modes: list[FailureMode] = Field(default_factory=list)

    def save(self, store: Store) -> None:
        store.taxonomy_path.parent.mkdir(parents=True, exist_ok=True)
        store.taxonomy_path.write_text(
            yaml.safe_dump(self.model_dump(), sort_keys=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, store: Store) -> Taxonomy:
        if not store.taxonomy_path.exists():
            return cls()
        document = store.read_yaml_model(store.taxonomy_path, _TaxonomyFile, empty={})
        return cls.model_validate(document.model_dump())

    def recount(self, reviews: list[Review]) -> None:
        """Set each mode's prevalence from how many open codes mention its name."""
        for mode in self.modes:
            pattern = re.compile(rf"\b{re.escape(mode.name)}\b", re.IGNORECASE)
            mode.prevalence = sum(
                1 for r in reviews if r.open_code and pattern.search(r.open_code)
            )


class _TaxonomyFile(Taxonomy):
    """taxonomy.yaml as read back. `modes` has a default, so a misspelt key would read as
    an empty taxonomy; the file is refused instead. A model's reply is validated against
    the lenient `Taxonomy`, so the schema sent to the model is unchanged."""

    model_config = ConfigDict(extra="forbid")


def count_open_codes(reviews: list[Review]) -> int:
    """Distinct traces carrying an open code, not review rows.

    `sample` defaults to `--seed 0`, so two review sessions can draw the exact
    same batch of traces; annotating both then produces two rows per trace.
    Summing rows would double that batch's evidence for free -- counting
    distinct trace ids is what keeps the 30-trace floor meaning 30 traces
    actually looked at, not 30 rows over as few as 15.
    """
    return len({r.trace_id for r in reviews if r.open_code and r.open_code.strip()})


def saturation_curve(reviews: list[Review]) -> list[int]:
    """Cumulative distinct open codes, one point per coded trace.

    Theoretical saturation is the stopping rule every source gives and none
    quantifies. A flattening curve is what it looks like.
    """
    seen: set[str] = set()
    curve: list[int] = []
    for review in reviews:
        if review.open_code and review.open_code.strip():
            seen.add(review.open_code.strip().lower())
            curve.append(len(seen))
    return curve


def _extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"could not parse a taxonomy from the model output: {text[:200]!r}")
    try:
        payload: dict[str, Any] = json.loads(match.group(0))
        return payload
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not parse a taxonomy from the model output: {exc}") from exc


_OUTSIDE_SLUG = re.compile(r"[^a-z0-9._-]+")
_SEPARATORS = "._-"


def _slug(text: str, limit: int = MAX_NAME_LENGTH) -> str:
    """`text` lowercased, each run of other characters made one "_", trimmed so it starts
    and ends with a letter or digit, and cut to `limit` characters."""
    slug = _OUTSIDE_SLUG.sub("_", text.lower()).strip(_SEPARATORS)
    return slug[:limit].rstrip(_SEPARATORS)


def _mode_ids(modes: list[FailureMode]) -> None:
    """Give each proposed mode an id every mode command accepts.

    A mode becomes a directory name, so `validate_name` refuses anything else, and a
    model's "Wrong Topic" or "naïve_mode" would be saved and then refused by `split`,
    `judge` and the rest. An id with nothing usable falls back to the mode's name, then
    to "mode"; a repeated id gets "_2", "_3" and so on.
    """
    taken: set[str] = set()
    for mode in modes:
        base = _slug(mode.id) or _slug(mode.name) or "mode"
        candidate, n = base, 1
        while candidate in taken:
            n += 1
            suffix = f"_{n}"
            candidate = _slug(base, MAX_NAME_LENGTH - len(suffix)) + suffix
        taken.add(candidate)
        mode.id = candidate


def propose_taxonomy(
    reviews: list[Review],
    complete: Callable[[str], str],
    *,
    min_open_codes: int = 30,
) -> Taxonomy:
    coded = count_open_codes(reviews)
    if coded < min_open_codes:
        raise InvariantViolation(
            f"{coded} traces carry an open code; {min_open_codes} are required before a "
            "taxonomy can be proposed. Open coding is where you discover failure modes "
            "you did not know existed — an LLM cannot discover what it was not shown. "
            f"Annotate {min_open_codes - coded} more trace(s) with `axial annotate`."
        )
    codes = "\n".join(
        f"- [{r.trace_id}] {r.open_code.strip()}"
        for r in reviews
        if r.open_code and r.open_code.strip()
    )
    payload = _extract_json(complete(PROMPT.format(codes=codes)))
    try:
        taxonomy = Taxonomy.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(
            f"the model's taxonomy does not have the expected shape: {validation_message(exc)}"
        ) from exc
    _mode_ids(taxonomy.modes)
    taxonomy.recount(reviews)
    return taxonomy
