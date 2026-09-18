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

import yaml
from pydantic import BaseModel, Field

from axial.errors import InvariantViolation
from axial.sample import Review
from axial.store import Store

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
        store.taxonomy_path.write_text(yaml.safe_dump(self.model_dump(), sort_keys=False))

    @classmethod
    def load(cls, store: Store) -> Taxonomy:
        if not store.taxonomy_path.exists():
            return cls()
        return cls.model_validate(yaml.safe_load(store.taxonomy_path.read_text()) or {})

    def recount(self, reviews: list[Review]) -> None:
        """Set each mode's prevalence from how many open codes mention its name."""
        for mode in self.modes:
            pattern = re.compile(rf"\b{re.escape(mode.name)}\b", re.IGNORECASE)
            mode.prevalence = sum(
                1 for r in reviews if r.open_code and pattern.search(r.open_code)
            )


def count_open_codes(reviews: list[Review]) -> int:
    return sum(1 for r in reviews if r.open_code and r.open_code.strip())


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


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"could not parse a taxonomy from the model output: {text[:200]!r}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not parse a taxonomy from the model output: {exc}") from exc


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
    taxonomy = Taxonomy.model_validate(payload)
    taxonomy.recount(reviews)
    return taxonomy
