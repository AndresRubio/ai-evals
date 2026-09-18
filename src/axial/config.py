"""Configuration, with the spec's defaults baked in.

Every number here is a methodology default, not a preference: they come from the
practitioner sources cited in the design document. They are configurable because
a team's bar is theirs to set, not because the defaults are arbitrary.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass

from axial.store import Store

DEFAULT_CONFIG = """\
# axial configuration. See docs for where each default comes from.
judge_model = "claude-opus-5"
# annotator = "your-name"      # recorded on every label you create
min_open_codes = 30            # before a taxonomy may be proposed
split_fractions = [0.15, 0.42, 0.43]

[thresholds]
min_tpr = 0.85                 # a judge below this misses real failures
min_tnr = 0.85                 # a judge below this cries wolf
max_gap = 0.10                 # dev -> test generalisation gap
"""


@dataclass(frozen=True)
class Thresholds:
    min_tpr: float = 0.85
    min_tnr: float = 0.85
    max_gap: float = 0.10


@dataclass(frozen=True)
class Config:
    judge_model: str = "claude-opus-5"
    annotator: str | None = None
    min_open_codes: int = 30
    split_fractions: tuple[float, float, float] = (0.15, 0.42, 0.43)
    thresholds: Thresholds = Thresholds()


def write_default_config(store: Store) -> None:
    store.config_path.parent.mkdir(parents=True, exist_ok=True)
    store.config_path.write_text(DEFAULT_CONFIG)


def load_config(store: Store) -> Config:
    if not store.config_path.exists():
        return Config()
    raw = tomllib.loads(store.config_path.read_text())
    thresholds_raw = raw.get("thresholds", {})
    thresholds = Thresholds(
        min_tpr=float(thresholds_raw.get("min_tpr", 0.85)),
        min_tnr=float(thresholds_raw.get("min_tnr", 0.85)),
        max_gap=float(thresholds_raw.get("max_gap", 0.10)),
    )
    fractions = raw.get("split_fractions", [0.15, 0.42, 0.43])
    return Config(
        judge_model=raw.get("judge_model", "claude-opus-5"),
        annotator=raw.get("annotator"),
        min_open_codes=int(raw.get("min_open_codes", 30)),
        split_fractions=(float(fractions[0]), float(fractions[1]), float(fractions[2])),
        thresholds=thresholds,
    )
