"""The artifact tree.

Everything axial knows lives under `.axial/` as plain text, so eval state is
diffable and reviewable in a pull request like any other source.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
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
        rows: list[dict[str, Any]] = []
        with path.open() as fh:
            for lineno, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    # Name the file and the line so a corrupt session or
                    # labels file produces a diagnosis, not an anonymous
                    # JSONDecodeError that aborts every session that happens
                    # to load it.
                    raise ValueError(f"{path}: invalid JSON on line {lineno}: {exc}") from exc
        return rows

    def _atomic_write(self, path: Path, data: str) -> None:
        """Write `data` to `path` via a same-directory temp file plus `os.replace`.

        A plain truncate-then-write leaves a truncated file behind if the
        process dies or the disk fills mid-write. That is tolerable for most
        of what axial writes, but not for `labels.jsonl`: hand-made ground
        truth that cannot be regenerated. Writing to a temp file in the same
        directory first, then renaming it over the target, means the target
        is either the old complete file or the new complete file, never a
        partial one -- `os.replace` is atomic on the same filesystem, which a
        same-directory temp file guarantees.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(data)
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise

    def write_jsonl(self, path: Path, rows: Iterable[dict[str, Any]]) -> None:
        data = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        self._atomic_write(path, data)

    def write_json(self, path: Path, obj: Any) -> None:
        data = json.dumps(obj, indent=2, sort_keys=True) + "\n"
        self._atomic_write(path, data)

    def read_json(self, path: Path) -> Any:
        return json.loads(path.read_text())

    def hash_file(self, path: Path) -> str:
        """SHA-256 of a file's raw bytes, or of empty bytes if it doesn't exist.

        Used to bind a `PROMOTED` marker to the exact `judge.yaml` and
        `verdicts.jsonl` it was earned on (see `report.promote` and
        `gate._load`) -- the marker records this hash, and a later mismatch
        means the judge or its verdicts changed since promotion.
        """
        data = path.read_bytes() if path.exists() else b""
        return hashlib.sha256(data).hexdigest()

    # --- traces --------------------------------------------------------
    def write_traces(self, traces: Iterable[Trace]) -> None:
        self.write_jsonl(self.traces_path, [t.model_dump() for t in traces])

    def read_traces(self) -> list[Trace]:
        return [Trace.model_validate(row) for row in self.read_jsonl(self.traces_path)]

    # --- hashing -------------------------------------------------------
    def hash_ids(self, ids: Iterable[str]) -> str:
        joined = "\x00".join(sorted(ids))
        return hashlib.sha256(joined.encode()).hexdigest()[:16]
