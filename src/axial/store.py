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
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from axial.errors import AxialError, MalformedFile, validation_message
from axial.traces import Trace

# A mode or a batch name becomes one directory name under `.axial/`, and a review
# session's name one file name under `.axial/reviews/`. Starting with a
# letter or digit keeps "." and ".." out and leaves dot-names free for the hidden
# siblings `_atomic_write` and `write_dir_atomically` stage in.
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

# `write_dir_atomically` moves an old directory to `.<name>.<32 hex>.old`, 38 bytes
# longer than the name, and file names stop at 255 bytes. Capping well below that
# also leaves room for `mkdtemp`'s and `mkstemp`'s own hidden names.
MAX_NAME_LENGTH = 100

# A judge version's directory under `.axial/judges/<mode>/`.
VERSION_RE = re.compile(r"v(\d+)")


def validate_name(name: str, kind: str) -> str:
    """Refuse a `kind` ("mode", "batch" or "session") name that is not one plain file or
    directory name.

    Unchecked, `../judges/x` or an absolute path resolves outside its own tree, and
    `x/y` lands inside another mode's directory, where `write_dir_atomically` would
    replace whatever it found. A name too long for the hidden siblings fails only
    when they are created, which for `measure` is after every trace was graded.
    """
    if len(name) > MAX_NAME_LENGTH:
        raise AxialError(
            f"{kind} name {name[:20]!r}... is {len(name)} characters long; it becomes a "
            f"file or directory name, so keep it to {MAX_NAME_LENGTH}"
        )
    if not NAME_RE.fullmatch(name):
        raise AxialError(
            f"{kind} name {name!r} is not allowed: it becomes a file or directory name, so "
            "use letters, digits, '.', '_' and '-', starting with a letter or digit"
        )
    return name


def refuse_non_directory(path: Path) -> None:
    """Refuse when something other than a directory is at `path`.

    `write_dir_atomically` can move a file or a symlink to one aside, but `rmtree`
    cannot remove it afterwards, which left a hidden `.old` sibling behind.
    """
    if os.path.lexists(path) and not path.is_dir():
        raise AxialError(
            f"{path} exists and is not a directory, so it was left alone; move it aside "
            "and run again"
        )


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
        # Only `v<digits>` is a version; a copy such as `v1-backup` is left alone.
        matches = (VERSION_RE.fullmatch(p.name) for p in parent.iterdir())
        versions = [int(m.group(1)) for m in matches if m]
        return max(versions) if versions else None

    def next_judge_version(self, mode: str) -> int:
        return (self.latest_judge_version(mode) or 0) + 1

    # --- io ------------------------------------------------------------
    # Files are read and written as UTF-8 whatever the locale. A hand-edited file is
    # almost always UTF-8, which the locale's own encoding (ASCII under a bare C locale)
    # would refuse.
    def append_jsonl(self, path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    def _numbered_rows(self, path: Path) -> Iterator[tuple[int, Any]]:
        with path.open(encoding="utf-8") as fh:
            try:
                for lineno, line in enumerate(fh, start=1):
                    if not line.strip():
                        continue
                    try:
                        yield lineno, json.loads(line)
                    except json.JSONDecodeError as exc:
                        # Name the file and the line so a corrupt session or
                        # labels file produces a diagnosis, not an anonymous
                        # JSONDecodeError that aborts every session that happens
                        # to load it. json's own "line 1 column 5" counts within
                        # this one line, so only its column is kept.
                        raise MalformedFile(
                            path, f"invalid JSON on line {lineno}, column {exc.colno}: {exc.msg}"
                        ) from exc
            except UnicodeDecodeError as exc:
                # The file is decoded a block at a time, so the bad byte's line is unknown.
                raise MalformedFile(path, f"not {exc.encoding} text: {exc.reason}") from exc

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        """Every row of `path`, each a JSON object whose fields nothing has checked. A line
        holding any other JSON value is a MalformedFile naming the line. Other modules read
        rows through `read_models`, which checks them."""
        if not path.exists():
            return []
        return [row for _, row in self._object_rows(path)]

    def _object_rows(self, path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
        for lineno, row in self._numbered_rows(path):
            if not isinstance(row, dict):
                raise MalformedFile(
                    path, f"line {lineno}: expected a JSON object, got {type(row).__name__}"
                )
            yield lineno, row

    def read_models[M: BaseModel](
        self, path: Path, model: type[M], *, missing_ok: bool = True
    ) -> list[M]:
        """Every row of `path`, each a JSON object validated as `model`.

        A row that does not validate is a MalformedFile naming the file and the line,
        as corrupt JSON already is. A missing file reads as no rows unless
        `missing_ok` is False, for a file the user named rather than one of the store's.
        """
        if missing_ok and not path.exists():
            return []
        models: list[M] = []
        for lineno, row in self._object_rows(path):
            try:
                models.append(model.model_validate(row))
            except ValidationError as exc:
                raise MalformedFile(path, f"line {lineno}: {validation_message(exc)}") from exc
        return models

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
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
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

        Anything at `path` other than a directory is refused before staging (see
        `refuse_non_directory`). A symlink to a directory is replaced, never followed,
        and the link itself is what gets removed.
        """
        refuse_non_directory(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        retired = path.with_name(f".{path.name}.{uuid.uuid4().hex}.old")
        staging = Path(tempfile.mkdtemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"))
        try:
            os.chmod(staging, 0o777 & ~_current_umask())
            for name, data in files.items():
                (staging / name).write_text(data, encoding="utf-8")
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
                    if retired.is_symlink():
                        retired.unlink()
                    else:
                        shutil.rmtree(retired)
                except OSError:
                    pass

    def read_json(self, path: Path) -> Any:
        """The JSON document at `path`. Bad JSON is a MalformedFile naming the line and
        column in one line, as bad YAML is."""
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError as exc:
            raise MalformedFile(path, f"not {exc.encoding} text: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise MalformedFile(
                path, f"invalid JSON on line {exc.lineno}, column {exc.colno}: {exc.msg}"
            ) from exc

    def read_json_model[M: BaseModel](self, path: Path, model: type[M]) -> M:
        """`read_json` validated as `model`. A document that does not validate is a
        MalformedFile naming the field."""
        data = self.read_json(path)
        try:
            return model.model_validate(data)
        except ValidationError as exc:
            raise MalformedFile(path, validation_message(exc)) from exc

    def read_yaml(self, path: Path) -> Any:
        """The YAML document at `path`. A syntax error is a MalformedFile naming the line
        and column in one line; PyYAML's own message runs to several."""
        try:
            return yaml.safe_load(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError as exc:
            raise MalformedFile(path, f"not {exc.encoding} text: {exc.reason}") from exc
        except yaml.MarkedYAMLError as exc:
            mark = exc.problem_mark
            where = f" on line {mark.line + 1}, column {mark.column + 1}" if mark else ""
            raise MalformedFile(path, f"invalid YAML{where}: {exc.problem}") from exc
        except yaml.YAMLError as exc:
            raise MalformedFile(path, f"invalid YAML: {(str(exc).splitlines() or [''])[0]}") from exc

    def read_yaml_model[M: BaseModel](self, path: Path, model: type[M], empty: Any = None) -> M:
        """`read_yaml` validated as `model`, reading an empty document as `empty`. A
        document that does not validate is a MalformedFile naming the field."""
        data = self.read_yaml(path)
        try:
            return model.model_validate(empty if data is None else data)
        except ValidationError as exc:
            raise MalformedFile(path, validation_message(exc)) from exc

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
        return self.read_models(self.traces_path, Trace)

    # --- hashing -------------------------------------------------------
    def hash_ids(self, ids: Iterable[str]) -> str:
        joined = "\x00".join(sorted(ids))
        return hashlib.sha256(joined.encode()).hexdigest()[:16]
