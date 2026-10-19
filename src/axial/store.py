"""The artifact tree.

Everything axial knows lives under `.axial/` as plain text, so eval state is
diffable and reviewable in a pull request like any other source.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from axial.errors import AxialError
from axial.traces import Trace

# A mode or a batch name becomes one directory name under `.axial/`. Starting with a
# letter or digit keeps "." and ".." out and leaves dot-names free for the hidden
# siblings `_atomic_write` and `write_dir_atomically` stage in.
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

# `write_dir_atomically` moves an old directory to `.<name>.<32 hex>.old`, 38 bytes
# longer than the name, and file names stop at 255 bytes. Capping well below that
# also leaves room for `mkdtemp`'s and `mkstemp`'s own hidden names.
MAX_NAME_LENGTH = 100


def validate_name(name: str, kind: str) -> str:
    """Refuse a `kind` ("mode" or "batch") name that is not one plain directory name.

    Unchecked, `../judges/x` or an absolute path resolves outside its own tree, and
    `x/y` lands inside another mode's directory, where `write_dir_atomically` would
    replace whatever it found. A name too long for the hidden siblings fails only
    when they are created, which for `measure` is after every trace was graded.
    """
    if len(name) > MAX_NAME_LENGTH:
        raise AxialError(
            f"{kind} name {name[:20]!r}... is {len(name)} characters long; it becomes a "
            f"directory, so keep it to {MAX_NAME_LENGTH}"
        )
    if not NAME_RE.fullmatch(name):
        raise AxialError(
            f"{kind} name {name!r} is not allowed: it becomes a directory, so use letters, "
            "digits, '.', '_' and '-', starting with a letter or digit"
        )
    return name


def _current_umask() -> int:
    """Read the process umask without permanently changing it.

    `os.umask` is the only way the standard library exposes to *read* the
    umask: setting it returns the previous value, so reading it means setting
    it back immediately after. This mirrors what the OS itself would apply to
    a newly created file's default `0o666` mode.
    """
    mask = os.umask(0)
    os.umask(mask)
    return mask


def jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    """The exact bytes `Store.write_jsonl` writes for `rows`."""
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def json_text(obj: Any) -> str:
    """The exact bytes `Store.write_json` writes for `obj`."""
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


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

    def _mode_dir(self, tree: str, mode: str) -> Path:
        """`.axial/<tree>/<mode>`: every mode-keyed path is built here, so none can
        skip the check on `mode`."""
        return self.root / tree / validate_name(mode, "mode")

    def dataset_dir(self, mode: str) -> Path:
        return self._mode_dir("datasets", mode)

    def judge_dir(self, mode: str, version: int) -> Path:
        return self._mode_dir("judges", mode) / f"v{version}"

    def measurement_dir(self, mode: str, batch: str) -> Path:
        return self._mode_dir("measurements", mode) / validate_name(batch, "batch")

    def latest_judge_version(self, mode: str) -> int | None:
        """The highest existing judge version for `mode`, or `None` if none exists yet.

        Sibling to `next_judge_version`: that one is for a deliberate "start a new
        judge" action, this one is for "resume whatever is already there" — the
        default a caller should reach for when no version was named.
        """
        parent = self._mode_dir("judges", mode)
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

        `mkstemp` always creates its temp file mode `0600`, and `os.replace`
        carries that mode onto the target -- so a naive version of this would
        silently drop every rewritten file from, say, `0644` to `0600`. That
        is exactly wrong for `.axial/`, which is meant to be committed and
        reviewed in a pull request: the mode is fixed up before the replace,
        to the existing file's own mode when overwriting one, or to the
        umask-respecting default a normal file create would get otherwise.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            mode = stat.S_IMODE(path.stat().st_mode)
        else:
            mode = 0o666 & ~_current_umask()
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(data)
            os.chmod(tmp_name, mode)
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise

    def write_jsonl(self, path: Path, rows: Iterable[dict[str, Any]]) -> None:
        self._atomic_write(path, jsonl_text(rows))

    def write_json(self, path: Path, obj: Any) -> None:
        self._atomic_write(path, json_text(obj))

    def write_dir_atomically(self, path: Path, files: dict[str, str]) -> None:
        """Make `path` a directory holding exactly `files` (name -> text), all or nothing.

        The files only mean something together, so a reader must never find some
        of them new and some old. They are written into a hidden sibling directory,
        an existing `path` is renamed aside, and the sibling is renamed onto `path`.
        If anything raises along the way, `KeyboardInterrupt` included, the old
        directory is renamed back. After an exception `path` is therefore either
        exactly as it was before the call or holds all of the new files.

        One window stays open. The standard library cannot swap two directories in
        a single step (that needs `RENAME_EXCHANGE` on Linux or `renamex_np` on
        macOS), so a hard kill such as SIGKILL or power loss between the two
        renames leaves nothing at `path` and the previous contents in a hidden
        `.<name>.<hex>.old` sibling. The same happens if putting the old directory
        back itself fails or is interrupted. Nothing restores that sibling
        automatically.

        Deleting the old copy comes last and only tidies up. The new directory is
        in place by then, so a failure to delete the old one is ignored. That
        failure, or an interrupt during the deletion, can leave the hidden `.old`
        sibling behind.

        `mkdtemp` creates its directory mode 0700, so the mode is fixed up to what
        a plain `mkdir` would give.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        retired = path.with_name(f".{path.name}.{uuid.uuid4().hex}.old")
        staging = Path(tempfile.mkdtemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"))
        try:
            os.chmod(staging, 0o777 & ~_current_umask())
            for name, data in files.items():
                (staging / name).write_text(data)
            if path.exists():
                os.replace(path, retired)
            os.replace(staging, path)
        except BaseException:
            # An interrupt can arrive just after a rename has returned, before any
            # variable could record it, so the disk decides whether to put the old
            # directory back.
            if retired.exists() and not path.exists():
                os.replace(retired, path)
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            # Both existing means the new directory went in, which leaves the old
            # copy as clutter. When `path` is missing the old copy is the only one
            # left and has to stay.
            if retired.exists() and path.exists():
                try:
                    shutil.rmtree(retired)
                except OSError:
                    pass

    def read_json(self, path: Path) -> Any:
        return json.loads(path.read_text())

    def hash_file(self, path: Path) -> str:
        """SHA-256 of a file's raw bytes, or of empty bytes if it doesn't exist.

        Used to bind a `PROMOTED` marker to the exact `judge.yaml` and
        `verdicts.jsonl` it was earned on (see `report.promote` and
        `report.require_promoted`) -- the marker records this hash, and a later mismatch
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
