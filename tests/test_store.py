import os
import shutil
import stat
from pathlib import Path

import pytest

from axial.errors import AxialError
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


# A mode becomes a directory under judges/, datasets/ and measurements/. Each of these
# once resolved to a path outside its own tree, or inside another mode's.
NOT_A_PLAIN_NAME = [
    "../judges/merged", "/elsewhere/.axial/judges/merged", "merged/x", "merged/",
    "", ".", "..", ".hidden", "sp ace",
]


@pytest.mark.parametrize("mode", NOT_A_PLAIN_NAME)
def test_every_mode_keyed_path_refuses_a_mode_that_is_not_a_plain_name(tmp_path: Path, mode: str):
    s = Store(tmp_path)
    for path_for in (
        s.dataset_dir,
        lambda m: s.judge_dir(m, 1),
        lambda m: s.measurement_dir(m, "rel-1"),
        s.latest_judge_version,
        s.next_judge_version,
    ):
        with pytest.raises(AxialError, match="mode name .* is not allowed"):
            path_for(mode)
    assert not s.root.exists()


def test_a_name_over_the_length_cap_is_refused_as_mode_and_as_batch(tmp_path: Path):
    s = Store(tmp_path)
    with pytest.raises(AxialError, match="101 characters long"):
        s.judge_dir("m" * 101, 1)
    with pytest.raises(AxialError, match="101 characters long"):
        s.measurement_dir("merged", "b" * 101)
    assert s.measurement_dir("m" * 100, "b" * 100).name == "b" * 100


def test_a_name_at_the_length_cap_leaves_room_for_the_hidden_siblings(tmp_path: Path):
    """`write_dir_atomically` stages in `.<name>.<random>.tmp` and moves the old copy to
    `.<name>.<32 hex>.old`; both have to fit in a 255-byte file name."""
    s = Store(tmp_path)
    target = s.measurement_dir("m" * 100, "b" * 100)
    s.write_dir_atomically(target, {"a.txt": "old"})
    s.write_dir_atomically(target, {"a.txt": "new"})
    assert (target / "a.txt").read_text() == "new"
    assert sorted(p.name for p in target.parent.iterdir()) == ["b" * 100]


@pytest.mark.parametrize("mode", ["merged", "merged_stories", "m", "wrong_topic", "v1.2-rc_3"])
def test_a_plain_mode_name_is_one_directory_under_each_tree(tmp_path: Path, mode: str):
    s = Store(tmp_path)
    assert s.judge_dir(mode, 1) == s.root / "judges" / mode / "v1"
    assert s.dataset_dir(mode) == s.root / "datasets" / mode
    assert s.measurement_dir(mode, "rel-1") == s.root / "measurements" / mode / "rel-1"


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
    # A fixed umask makes the expected mode differ from mkdtemp's 0700, so the
    # test fails if the mode fix-up is dropped.
    previous = os.umask(0o022)
    try:
        store.write_dir_atomically(target, {"a.txt": "x"})
    finally:
        os.umask(previous)
    assert stat.S_IMODE(target.stat().st_mode) == 0o755


def test_write_dir_atomically_restores_the_old_directory_when_interrupted_between_renames(
    tmp_path: Path, monkeypatch
):
    """Ctrl-C can land just after the old directory has been moved aside and
    before the new one is in place. The old directory must go back to `path`."""
    store = Store(tmp_path)
    target = tmp_path / "out"
    store.write_dir_atomically(target, {"a.txt": "old"})
    real_replace = os.replace

    def replace_then_interrupt(src: object, dst: object) -> None:
        real_replace(src, dst)
        if Path(str(src)) == target:
            raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", replace_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        store.write_dir_atomically(target, {"a.txt": "new"})
    assert {p.name: p.read_text() for p in target.iterdir()} == {"a.txt": "old"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out"]


def test_write_dir_atomically_restores_the_old_directory_when_installing_the_new_one_fails(
    tmp_path: Path, monkeypatch
):
    store = Store(tmp_path)
    target = tmp_path / "out"
    store.write_dir_atomically(target, {"a.txt": "old"})
    real_replace = os.replace

    def refuse_staging(src: object, dst: object) -> None:
        if Path(str(src)).suffix == ".tmp":
            raise OSError("rename refused")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", refuse_staging)
    with pytest.raises(OSError, match="rename refused"):
        store.write_dir_atomically(target, {"a.txt": "new"})
    assert {p.name: p.read_text() for p in target.iterdir()} == {"a.txt": "old"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out"]


def test_write_dir_atomically_succeeds_when_removing_the_old_directory_fails(
    tmp_path: Path, monkeypatch
):
    """The new directory is already in place by then, so the write must not be
    reported as failed. The old copy is left in a hidden `.old` sibling."""
    store = Store(tmp_path)
    target = tmp_path / "out"
    store.write_dir_atomically(target, {"a.txt": "old"})
    real_rmtree = shutil.rmtree

    def refuse_retired(path: object, *args: object, **kwargs: object) -> None:
        if Path(str(path)).suffix == ".old":
            raise PermissionError("cannot remove")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", refuse_retired)
    store.write_dir_atomically(target, {"a.txt": "new"})
    assert {p.name: p.read_text() for p in target.iterdir()} == {"a.txt": "new"}
    leftover = sorted(p.name for p in tmp_path.iterdir() if p.name != "out")
    assert len(leftover) == 1
    assert leftover[0].startswith(".out.") and leftover[0].endswith(".old")
    assert {p.name: p.read_text() for p in (tmp_path / leftover[0]).iterdir()} == {"a.txt": "old"}


@pytest.mark.parametrize("kind", ["file", "symlink-to-file", "dangling-symlink"])
def test_write_dir_atomically_refuses_a_target_that_is_not_a_directory(tmp_path: Path, kind: str):
    """Moving such a target aside worked, but `rmtree` cannot remove it, which left a
    hidden `.old` sibling behind; it is refused before anything is staged."""
    store = Store(tmp_path)
    target = tmp_path / "out"
    if kind == "file":
        target.write_text("a file, not a directory")
    elif kind == "symlink-to-file":
        (tmp_path / "real.txt").write_text("real")
        target.symlink_to(tmp_path / "real.txt")
    else:
        target.symlink_to(tmp_path / "nowhere")
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(AxialError, match="is not a directory"):
        store.write_dir_atomically(target, {"a.txt": "x"})
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert target.is_symlink() or target.read_text() == "a file, not a directory"


def test_write_dir_atomically_replaces_a_symlinked_directory_without_following_it(tmp_path: Path):
    store = Store(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    target = tmp_path / "out"
    target.symlink_to(outside)
    store.write_dir_atomically(target, {"a.txt": "x"})
    assert not target.is_symlink()
    assert {p.name: p.read_text() for p in target.iterdir()} == {"a.txt": "x"}
    assert (outside / "keep.txt").read_text() == "keep"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out", "outside"]
