"""The artifact tree.

Everything axial knows lives under `.axial/` as plain text, so eval state is
diffable and reviewable in a pull request like any other source.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from axial.traces import Trace


class Store:
    def __init__(self, base: Path | str = ".") -> None:
        self.base = Path(base)
        self.root = self.base / ".axial"

    # --- paths ---------------------------------------------------------
    @property
    def traces_path(self) -> Path:
        return self.root / "traces" / "traces.jsonl"

    @property
    def reviews_dir(self) -> Path:
        return self.root / "reviews"

    @property
    def taxonomy_path(self) -> Path:
        return self.root / "taxonomy.yaml"

    @property
    def config_path(self) -> Path:
        return self.root / "config.toml"

    @property
    def gates_path(self) -> Path:
        return self.root / "gates.yaml"

    def dataset_dir(self, mode: str) -> Path:
        return self.root / "datasets" / mode

    def judge_dir(self, mode: str, version: int) -> Path:
        return self.root / "judges" / mode / f"v{version}"

    def latest_judge_version(self, mode: str) -> int | None:
        """The highest existing judge version for `mode`, or `None` if none exists yet.

        Sibling to `next_judge_version`: that one is for a deliberate "start a new
        judge" action, this one is for "resume whatever is already there" — the
        default a caller should reach for when no version was named.
        """
        parent = self.root / "judges" / mode
        if not parent.exists():
            return None
        versions = [int(p.name[1:]) for p in parent.iterdir() if p.name.startswith("v")]
        return max(versions) if versions else None

    def next_judge_version(self, mode: str) -> int:
        return (self.latest_judge_version(mode) or 0) + 1

    # --- io ------------------------------------------------------------
    def append_jsonl(self, path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    def read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        with path.open() as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def write_jsonl(self, path: Path, rows: Iterable[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

    def write_json(self, path: Path, obj: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")

    def read_json(self, path: Path) -> Any:
        return json.loads(path.read_text())

    # --- traces --------------------------------------------------------
    def write_traces(self, traces: Iterable[Trace]) -> None:
        self.write_jsonl(self.traces_path, [t.model_dump() for t in traces])

    def read_traces(self) -> list[Trace]:
        return [Trace.model_validate(row) for row in self.read_jsonl(self.traces_path)]

    # --- hashing -------------------------------------------------------
    def hash_ids(self, ids: Iterable[str]) -> str:
        joined = "\x00".join(sorted(ids))
        return hashlib.sha256(joined.encode()).hexdigest()[:16]
