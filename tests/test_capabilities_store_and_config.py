"""What the artifact store (store.py) and configuration (config.py) can and cannot do.

Each test pins one capability or limit. Tests marked xfail(strict) document a real
defect: they state the behaviour a user would expect, and will start failing
(loudly) the day the defect is fixed, at which point the marker should be removed.
"""

import hashlib
import os
import stat
import tomllib
from pathlib import Path

import pytest

from axial.config import Config, load_config, load_env, write_default_config
from axial.errors import AxialError
from axial.store import Store
from axial.traces import Trace

# --- Store: layout ---------------------------------------------------------------


def test_every_artifact_lives_at_a_fixed_path_under_dot_axial(tmp_path: Path):
    """The on-disk layout is part of the interface: reviewers look for these exact paths."""
    s = Store(tmp_path)
    root = tmp_path / ".axial"
    assert s.traces_path == root / "traces" / "traces.jsonl"
    assert s.reviews_dir == root / "reviews"
    assert s.taxonomy_path == root / "taxonomy.yaml"
    assert s.config_path == root / "config.toml"
    assert s.gates_path == root / "gates.yaml"
    assert s.dataset_dir("merged") == root / "datasets" / "merged"
    assert s.judge_dir("merged", 3) == root / "judges" / "merged" / "v3"


def test_a_store_accepts_a_string_or_a_path_and_touches_nothing_on_creation(tmp_path: Path):
    """Constructing a Store is free of side effects; `.axial/` appears only on first write."""
    assert Store(str(tmp_path)).root == Store(tmp_path).root
    assert not (tmp_path / ".axial").exists()


def test_writing_creates_missing_parent_directories(tmp_path: Path):
    """No `init` is needed before a write: nested directories are created on demand."""
    s = Store(tmp_path)
    s.write_json(s.dataset_dir("deep") / "nested" / "x.json", {"a": 1})
    assert s.read_json(s.dataset_dir("deep") / "nested" / "x.json") == {"a": 1}


# --- Store: byte-level format ------------------------------------------------------


def test_json_artifacts_are_sorted_indented_and_newline_terminated(tmp_path: Path):
    """The exact text format is fixed, so diffs in a pull request show only real changes."""
    s = Store(tmp_path)
    p = s.root / "x.json"
    s.write_json(p, {"b": 1, "a": {"d": 2, "c": 3}})
    assert p.read_text() == '{\n  "a": {\n    "c": 3,\n    "d": 2\n  },\n  "b": 1\n}\n'


def test_jsonl_rows_are_one_line_each_with_sorted_keys(tmp_path: Path):
    """Row key order never depends on how the dict was built."""
    s = Store(tmp_path)
    p = s.root / "x.jsonl"
    s.write_jsonl(p, [{"b": 1, "a": 2}, {"a": 2, "b": 1}])
    assert p.read_text() == '{"a": 2, "b": 1}\n{"a": 2, "b": 1}\n'


def test_stored_text_is_pure_ascii_even_for_unicode_content(tmp_path: Path):
    """Non-ASCII is escaped, so artifacts read back correctly under any locale or editor."""
    s = Store(tmp_path)
    s.write_traces([Trace(trace_id="a", input="café \U0001f600", output="日本")])
    s.traces_path.read_bytes().decode("ascii")  # would raise if any raw non-ASCII were written
    assert s.read_traces()[0].input == "café \U0001f600"


def test_write_traces_keeps_order_and_duplicates_and_replaces_rather_than_appends(tmp_path: Path):
    """The corpus file is exactly the list last written, in that order."""
    s = Store(tmp_path)
    s.write_traces([Trace(trace_id="x", input="1", output="1"), Trace(trace_id="x", input="2", output="2")])
    s.write_traces([Trace(trace_id="z", input="3", output="3"), Trace(trace_id="y", input="4", output="4")])
    assert [t.trace_id for t in s.read_traces()] == ["z", "y"]


# --- Store: reading ------------------------------------------------------------------


def test_read_jsonl_skips_blank_lines_and_returns_empty_for_a_missing_file(tmp_path: Path):
    """Hand-edited files with spare blank lines still load; an absent file is just empty."""
    s = Store(tmp_path)
    p = s.root / "x.jsonl"
    p.parent.mkdir(parents=True)
    p.write_text('\n{"a": 1}\n  \n{"a": 2}\n\n')
    assert s.read_jsonl(p) == [{"a": 1}, {"a": 2}]
    assert s.read_jsonl(s.root / "absent.jsonl") == []


def test_read_json_of_a_missing_file_raises_rather_than_returning_a_default(tmp_path: Path):
    """Unlike JSONL, a missing JSON document is an error: callers must check existence."""
    with pytest.raises(FileNotFoundError):
        Store(tmp_path).read_json(Store(tmp_path).root / "absent.json")


def test_append_jsonl_is_not_atomic_and_extends_the_file_in_place(tmp_path: Path):
    """Limit: appends open the existing file directly (no temp-file swap), keeping earlier rows."""
    s = Store(tmp_path)
    p = s.root / "x.jsonl"
    s.append_jsonl(p, {"a": 1})
    inode = p.stat().st_ino
    s.append_jsonl(p, {"a": 2})
    assert p.stat().st_ino == inode


# --- Store: atomic writes -------------------------------------------------------------


def test_a_new_file_gets_the_mode_the_process_umask_would_give_it(tmp_path: Path):
    """Fresh artifacts are 0644 under a normal umask, not mkstemp's owner-only 0600."""
    s = Store(tmp_path)
    old = os.umask(0o022)
    try:
        s.write_json(s.root / "a.json", {})
        os.umask(0o077)
        s.write_json(s.root / "b.json", {})
    finally:
        os.umask(old)
    assert stat.S_IMODE((s.root / "a.json").stat().st_mode) == 0o644
    assert stat.S_IMODE((s.root / "b.json").stat().st_mode) == 0o600


def test_write_json_also_preserves_the_existing_files_mode(tmp_path: Path):
    """The mode-preserving rewrite applies to JSON documents, not only JSONL."""
    s = Store(tmp_path)
    p = s.root / "x.json"
    s.write_json(p, {"a": 1})
    os.chmod(p, 0o664)
    s.write_json(p, {"a": 2})
    assert stat.S_IMODE(p.stat().st_mode) == 0o664


def test_an_unserialisable_row_fails_before_the_existing_file_is_touched(tmp_path: Path):
    """The whole payload is serialised first, so a bad row can never truncate good data."""
    s = Store(tmp_path)
    p = s.root / "x.jsonl"
    s.write_jsonl(p, [{"a": 1}])
    with pytest.raises(TypeError):
        s.write_jsonl(p, [{"a": 2}, {"a": object()}])
    assert s.read_jsonl(p) == [{"a": 1}]
    assert list(p.parent.iterdir()) == [p]


def test_a_row_generator_that_dies_midway_leaves_the_original_intact(tmp_path: Path):
    """Streaming rows into a write is safe: a failure partway through changes nothing."""
    s = Store(tmp_path)
    p = s.root / "x.jsonl"
    s.write_jsonl(p, [{"a": "original"}])

    def rows():
        yield {"a": "new"}
        raise RuntimeError("source died")

    with pytest.raises(RuntimeError, match="source died"):
        s.write_jsonl(p, rows())
    assert s.read_jsonl(p) == [{"a": "original"}]
    assert list(p.parent.iterdir()) == [p]


def test_a_failed_rename_cleans_up_its_temp_file_and_keeps_the_original(tmp_path: Path, monkeypatch):
    """If the final os.replace fails, nothing is left behind and the old file is unchanged."""
    s = Store(tmp_path)
    p = s.root / "x.json"
    s.write_json(p, {"a": "original"})

    def boom(src: object, dst: object) -> None:
        raise OSError("cross-device link")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="cross-device"):
        s.write_json(p, {"a": "new"})
    assert s.read_json(p) == {"a": "original"}
    assert list(p.parent.iterdir()) == [p]


# --- Store: judge versions -------------------------------------------------------------


def test_judge_versions_compare_numerically_not_alphabetically(tmp_path: Path):
    """v10 is newer than v9; the next version after that is 11."""
    s = Store(tmp_path)
    for v in (2, 9, 10):
        s.judge_dir("m", v).mkdir(parents=True)
    assert s.latest_judge_version("m") == 10
    assert s.next_judge_version("m") == 11


def test_a_gap_in_judge_versions_is_not_filled(tmp_path: Path):
    """Versions are never reused: with v1 and v3 present the next one is v4."""
    s = Store(tmp_path)
    for v in (1, 3):
        s.judge_dir("m", v).mkdir(parents=True)
    assert s.next_judge_version("m") == 4


def test_judge_versions_are_tracked_per_failure_mode(tmp_path: Path):
    """One mode's history never advances another's numbering."""
    s = Store(tmp_path)
    s.judge_dir("a", 5).mkdir(parents=True)
    assert s.latest_judge_version("b") is None
    assert s.next_judge_version("b") == 1


def test_directory_entries_not_named_like_versions_are_ignored(tmp_path: Path):
    """Stray files such as notes.md next to the version folders do not affect numbering."""
    s = Store(tmp_path)
    s.judge_dir("m", 2).mkdir(parents=True)
    (s.root / "judges" / "m" / "notes.md").write_text("scratch")
    assert s.latest_judge_version("m") == 2


def test_a_folder_that_only_starts_with_v_does_not_break_version_lookup(tmp_path: Path):
    """Copying a judge to `v1-backup` should not make `judge` unusable for that mode."""
    s = Store(tmp_path)
    s.judge_dir("m", 1).mkdir(parents=True)
    (s.root / "judges" / "m" / "v1-backup").mkdir()
    assert s.latest_judge_version("m") == 1


# --- Store: hashing ---------------------------------------------------------------------


def test_hash_file_of_a_missing_file_equals_the_hash_of_empty_bytes(tmp_path: Path):
    """An absent file hashes as empty, so 'deleted' and 'emptied' are indistinguishable."""
    s = Store(tmp_path)
    assert s.hash_file(tmp_path / "absent") == hashlib.sha256(b"").hexdigest()


def test_hash_file_changes_with_a_single_byte_of_content(tmp_path: Path):
    """Promotion markers bind to exact bytes: one changed character changes the hash."""
    s = Store(tmp_path)
    p = tmp_path / "judge.yaml"
    p.write_text("prompt: is it merged?\n")
    before = s.hash_file(p)
    p.write_text("prompt: is it merged!\n")
    assert s.hash_file(p) != before


def test_hash_ids_ignores_order_but_counts_duplicates(tmp_path: Path):
    """The id hash is over a sorted list, so a repeated id is a different set of ids."""
    s = Store(tmp_path)
    assert s.hash_ids(["b", "a", "a"]) == s.hash_ids(["a", "a", "b"])
    assert s.hash_ids(["a", "a"]) != s.hash_ids(["a"])


# --- Config: loading and validation ------------------------------------------------------


def test_the_written_default_config_loads_to_exactly_the_built_in_defaults(tmp_path: Path):
    """The template `axial init` writes cannot drift from the defaults the code assumes."""
    store = Store(tmp_path)
    write_default_config(store)
    assert load_config(store) == Config()


def test_an_empty_config_file_means_all_defaults(tmp_path: Path):
    """An empty config.toml is valid and changes nothing."""
    store = Store(tmp_path)
    store.config_path.parent.mkdir(parents=True)
    store.config_path.write_text("")
    assert load_config(store) == Config()


def test_the_annotator_is_unset_unless_the_config_names_one(tmp_path: Path):
    """There is no default annotator; it must be chosen deliberately."""
    assert load_config(Store(tmp_path)).annotator is None


def test_malformed_toml_is_a_hard_error_never_silently_defaulted(tmp_path: Path):
    """A broken config file stops the run rather than falling back to default thresholds."""
    store = Store(tmp_path)
    store.config_path.parent.mkdir(parents=True)
    store.config_path.write_text("[thresholds\nmin_tpr = 0.9\n")
    with pytest.raises(AxialError, match=r"config\.toml is not valid TOML") as caught:
        load_config(store)
    assert isinstance(caught.value.__cause__, tomllib.TOMLDecodeError)


def test_a_non_numeric_split_fraction_is_refused_by_position(tmp_path: Path):
    """The error names which entry was bad."""
    store = Store(tmp_path)
    store.config_path.parent.mkdir(parents=True)
    store.config_path.write_text('split_fractions = ["a", 0.5, 0.5]\n')
    with pytest.raises(AxialError, match=r"split_fractions\[0\] must be a number"):
        load_config(store)


def test_a_negative_split_fraction_is_refused(tmp_path: Path):
    """Negative shares are refused even when the total would still be 1.0."""
    store = Store(tmp_path)
    store.config_path.parent.mkdir(parents=True)
    store.config_path.write_text("split_fractions = [-0.1, 0.6, 0.5]\n")
    with pytest.raises(AxialError, match="must be positive"):
        load_config(store)


def test_a_misspelt_config_key_is_silently_ignored(tmp_path: Path):
    """Limit: unknown keys are not errors, so a typo'd `judge_modle` leaves the default in force."""
    store = Store(tmp_path)
    store.config_path.parent.mkdir(parents=True)
    store.config_path.write_text('judge_modle = "claude-sonnet-5"\n[thresholds]\nmin_tpp = 0.5\n')
    cfg = load_config(store)
    assert cfg.judge_model == "claude-opus-5"
    assert cfg.thresholds.min_tpr == 0.85


def test_a_threshold_written_outside_the_thresholds_table_is_silently_ignored(tmp_path: Path):
    """Limit: `min_tpr = 0.5` at the top level is a different key from `[thresholds] min_tpr`."""
    store = Store(tmp_path)
    store.config_path.parent.mkdir(parents=True)
    store.config_path.write_text("min_tpr = 0.5\n")
    assert load_config(store).thresholds.min_tpr == 0.85


def test_integer_toml_thresholds_are_read_as_floats(tmp_path: Path):
    """`min_tpr = 1` is fine: whole numbers are accepted for rate thresholds."""
    store = Store(tmp_path)
    store.config_path.parent.mkdir(parents=True)
    store.config_path.write_text("[thresholds]\nmin_tpr = 1\n")
    assert load_config(store).thresholds.min_tpr == 1.0


def test_a_scalar_split_fractions_is_refused_with_an_axial_error(tmp_path: Path):
    """A wrong-shaped value should get the same friendly AxialError as a wrong-length list."""
    store = Store(tmp_path)
    store.config_path.parent.mkdir(parents=True)
    store.config_path.write_text("split_fractions = 0.5\n")
    with pytest.raises(AxialError):
        load_config(store)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("split_fractions = [0.5, nan, 0.5]\n", r"split_fractions\[1\] must be positive"),
        ('min_open_codes = "thirty"\n', "min_open_codes must be a number, got 'thirty'"),
        ('[thresholds]\nmin_tpr = "high"\n', "min_tpr must be a number, got 'high'"),
        ("thresholds = 0.9\n", r"thresholds must be a \[thresholds\] table"),
        ("judge_model = 5\n", "judge_model must be a string, got 5"),
        ("annotator = [1]\n", r"annotator must be a string, got \[1\]"),
    ],
)
def test_a_wrongly_typed_config_value_is_refused_naming_the_file_and_key(
    tmp_path: Path, text: str, message: str
):
    """Every bad value is one AxialError naming config.toml, never a TypeError or ValueError."""
    store = Store(tmp_path)
    store.config_path.parent.mkdir(parents=True)
    store.config_path.write_text(text)
    with pytest.raises(AxialError, match=rf"config\.toml: {message}"):
        load_config(store)


@pytest.mark.xfail(strict=True, reason="config.py:90 write_text() overwrites unconditionally, so "
                   "running `axial init` again wipes a configured annotator and thresholds")
def test_writing_the_default_config_does_not_clobber_an_existing_one(tmp_path: Path):
    """Re-running init on a configured project should keep the user's settings."""
    store = Store(tmp_path)
    store.config_path.parent.mkdir(parents=True)
    store.config_path.write_text('annotator = "andres"\n')
    write_default_config(store)
    assert load_config(store).annotator == "andres"


# --- .env loading ---------------------------------------------------------------------------

KEYS = ("AXIAL_T_A", "AXIAL_T_B", "AXIAL_T_C", "AXIAL_T_D")


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Guarantee the test keys start unset and are removed again after the test."""
    for key in KEYS:
        monkeypatch.setenv(key, "")  # registers the key so teardown restores it to absent
        monkeypatch.delenv(key)
    return monkeypatch


def write_env(tmp_path: Path, text: str) -> None:
    (tmp_path / ".env").write_bytes(text.encode())


def test_export_prefix_quotes_and_spaces_around_equals_are_understood(tmp_path, clean_env):
    """`export K = 'v'` and `K="v"` both load with the wrapping quotes removed."""
    write_env(tmp_path, "export AXIAL_T_A = 'alpha'\nAXIAL_T_B=\"beta\"\n")
    load_env(tmp_path)
    assert (os.environ["AXIAL_T_A"], os.environ["AXIAL_T_B"]) == ("alpha", "beta")


def test_a_value_containing_equals_signs_is_kept_whole(tmp_path, clean_env):
    """Only the first '=' splits, so base64-style keys ending in '==' survive."""
    write_env(tmp_path, "AXIAL_T_A=abc==\n")
    load_env(tmp_path)
    assert os.environ["AXIAL_T_A"] == "abc=="


def test_windows_line_endings_do_not_leak_into_values(tmp_path, clean_env):
    """A CRLF-terminated .env yields clean values with no trailing carriage return."""
    write_env(tmp_path, "AXIAL_T_A=alpha\r\nAXIAL_T_B=beta\r\n")
    load_env(tmp_path)
    assert (os.environ["AXIAL_T_A"], os.environ["AXIAL_T_B"]) == ("alpha", "beta")


def test_an_indented_comment_is_still_a_comment(tmp_path, clean_env):
    """Leading whitespace before '#' does not turn a comment into an assignment."""
    write_env(tmp_path, "   # AXIAL_T_A=nope\nAXIAL_T_B=yes\n")
    load_env(tmp_path)
    assert "AXIAL_T_A" not in os.environ
    assert os.environ["AXIAL_T_B"] == "yes"


def test_lines_without_an_equals_sign_are_ignored(tmp_path, clean_env):
    """Stray words and bare names are skipped rather than raising or setting empty vars."""
    write_env(tmp_path, "AXIAL_T_A\njust some words\nAXIAL_T_B=ok\n")
    load_env(tmp_path)
    assert "AXIAL_T_A" not in os.environ
    assert os.environ["AXIAL_T_B"] == "ok"


def test_export_with_an_empty_value_and_empty_quotes_set_nothing(tmp_path, clean_env):
    """`export K=` and `K=""` are both placeholders, not values."""
    write_env(tmp_path, 'export AXIAL_T_A=\nAXIAL_T_B=""\nAXIAL_T_C=\'\'\n')
    load_env(tmp_path)
    assert not {"AXIAL_T_A", "AXIAL_T_B", "AXIAL_T_C"} & set(os.environ)


def test_an_exported_empty_string_still_blocks_the_file_value(tmp_path, clean_env):
    """Limit: 'already set' means present, even as ''; the file cannot fill a blanked export."""
    clean_env.setenv("AXIAL_T_A", "")
    write_env(tmp_path, "AXIAL_T_A=sk-from-file\n")
    load_env(tmp_path)
    assert os.environ["AXIAL_T_A"] == ""


def test_when_a_key_appears_twice_in_the_file_the_first_occurrence_wins(tmp_path, clean_env):
    """Limit: unlike most dotenv loaders, a repeated key is not overridden by a later line."""
    write_env(tmp_path, "AXIAL_T_A=first\nAXIAL_T_A=second\n")
    load_env(tmp_path)
    assert os.environ["AXIAL_T_A"] == "first"


def test_a_trailing_inline_comment_becomes_part_of_the_value(tmp_path, clean_env):
    """Limit: only whole-line comments are recognised; `KEY=v # note` keeps ' # note'."""
    write_env(tmp_path, "AXIAL_T_A=abc # my key\n")
    load_env(tmp_path)
    assert os.environ["AXIAL_T_A"] == "abc # my key"


def test_interpolation_is_not_performed(tmp_path, clean_env):
    """Limit: it is not a shell; `$OTHER` is stored literally."""
    write_env(tmp_path, "AXIAL_T_A=$HOME/keys\n")
    load_env(tmp_path)
    assert os.environ["AXIAL_T_A"] == "$HOME/keys"


def test_load_env_reads_only_the_repo_root_and_returns_the_path_it_used(tmp_path, clean_env):
    """A parent directory's .env is never consulted; a found file's path is returned."""
    write_env(tmp_path, "AXIAL_T_A=from-parent\n")
    child = tmp_path / "project"
    child.mkdir()
    assert load_env(child) is None
    assert "AXIAL_T_A" not in os.environ
    assert load_env(tmp_path) == tmp_path / ".env"


def test_a_directory_named_dot_env_is_treated_as_no_file(tmp_path, clean_env):
    """Only a regular file counts; a `.env/` directory does not crash the loader."""
    (tmp_path / ".env").mkdir()
    assert load_env(tmp_path) is None
