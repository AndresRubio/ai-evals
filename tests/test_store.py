import os
import stat
from pathlib import Path

import pytest

from axial.store import Store
from axial.traces import Trace


def test_store_creates_the_artifact_tree(tmp_path: Path):
    s = Store(tmp_path)
    assert s.root == tmp_path / ".axial"
    assert s.traces_path == tmp_path / ".axial" / "traces" / "traces.jsonl"


def test_write_and_read_traces_round_trip(tmp_path: Path):
    s = Store(tmp_path)
    traces = [Trace(trace_id="a", input="i", output="o")]
    s.write_traces(traces)
    assert s.read_traces() == traces


def test_read_traces_is_empty_before_anything_is_written(tmp_path: Path):
    assert Store(tmp_path).read_traces() == []


def test_append_jsonl_accumulates_rows(tmp_path: Path):
    s = Store(tmp_path)
    p = s.root / "x.jsonl"
    s.append_jsonl(p, {"a": 1})
    s.append_jsonl(p, {"a": 2})
    assert s.read_jsonl(p) == [{"a": 1}, {"a": 2}]


def test_hash_ids_is_order_independent(tmp_path: Path):
    s = Store(tmp_path)
    assert s.hash_ids(["b", "a"]) == s.hash_ids(["a", "b"])
    assert s.hash_ids(["a"]) != s.hash_ids(["a", "b"])


def test_next_judge_version_starts_at_one_and_increments(tmp_path: Path):
    s = Store(tmp_path)
    assert s.next_judge_version("mode") == 1
    s.judge_dir("mode", 1).mkdir(parents=True)
    assert s.next_judge_version("mode") == 2


# --- Finding 5: atomic writes and diagnosable parse errors ---


def test_write_jsonl_replaces_the_target_leaving_no_temp_file_behind(tmp_path: Path):
    s = Store(tmp_path)
    path = s.root / "x.jsonl"
    s.write_jsonl(path, [{"a": 1}])
    s.write_jsonl(path, [{"a": 2}])
    assert s.read_jsonl(path) == [{"a": 2}]
    assert list(path.parent.iterdir()) == [path]


def test_write_json_replaces_the_target_leaving_no_temp_file_behind(tmp_path: Path):
    s = Store(tmp_path)
    path = s.root / "x.json"
    s.write_json(path, {"a": 1})
    s.write_json(path, {"a": 2})
    assert s.read_json(path) == {"a": 2}
    assert list(path.parent.iterdir()) == [path]


def test_write_jsonl_writes_via_a_temp_file_and_os_replace(tmp_path: Path, monkeypatch):
    """`labels.jsonl` is hand-made ground truth that cannot be regenerated -- a
    crash or full disk mid-write must never leave a truncated file in its
    place. Writing to a temp file in the same directory and renaming over the
    target (rather than truncating it in place) is what guarantees that."""
    s = Store(tmp_path)
    path = s.root / "x.jsonl"
    calls: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy_replace(src: object, dst: object) -> None:
        calls.append((str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy_replace)
    s.write_jsonl(path, [{"a": 1}])

    assert len(calls) == 1
    tmp_src, dst = calls[0]
    assert dst == str(path)
    assert tmp_src != str(path)
    assert Path(tmp_src).parent == path.parent  # same filesystem as the target


def test_write_json_survives_a_failed_write_without_touching_the_original(tmp_path: Path, monkeypatch):
    """If writing the temp file itself fails, the original file must be left
    exactly as it was -- never truncated."""
    s = Store(tmp_path)
    path = s.root / "x.json"
    s.write_json(path, {"a": "original"})

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "fdopen", boom)
    with pytest.raises(OSError, match="disk full"):
        s.write_json(path, {"a": "new"})

    assert s.read_json(path) == {"a": "original"}
    assert list(path.parent.iterdir()) == [path]  # the failed temp file was cleaned up


def test_write_jsonl_preserves_the_existing_files_mode_across_a_rewrite(tmp_path: Path):
    """`mkstemp` creates its temp file `0600`, and a naive `os.replace` would
    carry that mode onto the target -- silently dropping a `0644` artifact
    meant to be committed and reviewed in a PR down to owner-only. The
    existing file's mode must survive a rewrite untouched."""
    s = Store(tmp_path)
    path = s.root / "x.jsonl"
    s.write_jsonl(path, [{"a": 1}])
    os.chmod(path, 0o644)

    s.write_jsonl(path, [{"a": 2}])

    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    assert s.read_jsonl(path) == [{"a": 2}]


def test_read_jsonl_reports_the_file_and_line_number_on_corrupt_json(tmp_path: Path):
    s = Store(tmp_path)
    path = s.root / "bad.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"a": 1}\nnot json\n{"a": 3}\n')
    with pytest.raises(ValueError, match=r"bad\.jsonl.*line 2"):
        s.read_jsonl(path)


def test_write_dir_atomically_writes_every_file(tmp_path: Path):
    store = Store(tmp_path)
    target = store.root / "measurements" / "m" / "b1"
    store.write_dir_atomically(target, {"a.txt": "one\n", "b.txt": "two\n"})
    assert {p.name: p.read_text() for p in target.iterdir()} == {"a.txt": "one\n", "b.txt": "two\n"}
    assert sorted(p.name for p in target.parent.iterdir()) == ["b1"]


def test_write_dir_atomically_replaces_the_whole_directory(tmp_path: Path):
    store = Store(tmp_path)
    target = tmp_path / "out"
    store.write_dir_atomically(target, {"old.txt": "x", "kept-name.txt": "old"})
    store.write_dir_atomically(target, {"kept-name.txt": "new"})
    assert {p.name: p.read_text() for p in target.iterdir()} == {"kept-name.txt": "new"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out"]


def test_write_dir_atomically_leaves_the_old_directory_when_a_write_fails(tmp_path: Path):
    store = Store(tmp_path)
    target = tmp_path / "out"
    store.write_dir_atomically(target, {"a.txt": "old"})
    with pytest.raises(FileNotFoundError):
        # A name inside a subdirectory that was never created fails mid-write.
        store.write_dir_atomically(target, {"a.txt": "new", "missing/b.txt": "x"})
    assert {p.name: p.read_text() for p in target.iterdir()} == {"a.txt": "old"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out"]


def test_write_dir_atomically_gives_the_directory_a_normal_mode(tmp_path: Path):
    store = Store(tmp_path)
    target = tmp_path / "out"
    store.write_dir_atomically(target, {"a.txt": "x"})
    mask = os.umask(0)
    os.umask(mask)
    assert stat.S_IMODE(target.stat().st_mode) == 0o777 & ~mask
