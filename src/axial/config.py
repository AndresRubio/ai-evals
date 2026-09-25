"""Configuration, with the spec's defaults baked in.

Every number here is a methodology default, not a preference: they come from the
practitioner sources cited in the design document. They are configurable because
a team's bar is theirs to set, not because the defaults are arbitrary.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from axial.errors import AxialError
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
min_test_positives = 30        # real failures the test split must contain before a rate means anything
min_test_negatives = 30        # real passes the test split must contain before a rate means anything
"""


def load_env(repo: str | Path = ".") -> Path | None:
    """Load `<repo>/.env` into the environment, and report whether there was one.

    Model credentials are read from the environment by litellm and by `JevModel`.
    This puts them in one gitignored file so a run does not depend on what happens
    to be exported in the shell that started it.

    Two deliberate rules. An already-set variable is never overwritten: exporting a
    key is a deliberate act and must beat a stale file. And an empty value is
    skipped rather than set, so a freshly copied `.env.example` cannot mask a real
    export with an empty string.

    The parser handles `KEY=value`, `export KEY=value`, `#` comment lines and
    surrounding quotes. It is not a shell: no interpolation, no multi-line values.
    """
    path = Path(repo) / ".env"
    if not path.is_file():
        return None

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip("\"'")
        if not key or not value or key in os.environ:
            continue
        os.environ[key] = value
    return path


@dataclass(frozen=True)
class Thresholds:
    min_tpr: float = 0.85
    min_tnr: float = 0.85
    max_gap: float = 0.10
    min_test_positives: int = 30
    min_test_negatives: int = 30


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
        min_test_positives=int(thresholds_raw.get("min_test_positives", 30)),
        min_test_negatives=int(thresholds_raw.get("min_test_negatives", 30)),
    )
    fractions = raw.get("split_fractions", [0.15, 0.42, 0.43])

    # Validate split_fractions
    if len(fractions) != 3:
        raise AxialError(
            f"split_fractions must have exactly 3 entries (train, dev, test), "
            f"but found {len(fractions)}"
        )

    fractions_float = []
    for i, frac in enumerate(fractions):
        try:
            frac_val = float(frac)
        except (TypeError, ValueError):
            raise AxialError(f"split_fractions[{i}] must be a number, got {frac!r}")
        if frac_val <= 0:
            raise AxialError(
                f"split_fractions[{i}] must be positive, got {frac_val}"
            )
        fractions_float.append(frac_val)

    total = sum(fractions_float)
    if abs(total - 1.0) > 1e-6:
        raise AxialError(
            f"split_fractions must sum to 1.0, but got {total}"
        )

    return Config(
        judge_model=raw.get("judge_model", "claude-opus-5"),
        annotator=raw.get("annotator"),
        min_open_codes=int(raw.get("min_open_codes", 30)),
        split_fractions=(fractions_float[0], fractions_float[1], fractions_float[2]),
        thresholds=thresholds,
    )
