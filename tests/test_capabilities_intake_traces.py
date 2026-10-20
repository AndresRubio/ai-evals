"""What `axial ingest` (traces.py) accepts, what it rejects, and how it fails.

Each test pins one capability or limit of the trace-intake layer. Tests marked
xfail(strict) document a real defect: they state the behaviour a user would
expect, and will start failing (loudly) the day the defect is fixed.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from axial.cli import app
from axial.store import Store
from axial.traces import Trace, compute_trace_id, ingest, ingest_csv, ingest_jsonl

runner = CliRunner()


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


# --- JSONL vs CSV ----------------------------------------------------------------


def test_jsonl_and_csv_with_the_same_rows_ingest_to_identical_traces(tmp_path: Path):
    """The two formats are interchangeable for input/output/trace_id rows."""
    write_jsonl(
        tmp_path / "t.jsonl",
        [
            {"trace_id": "a", "input": "i1", "output": "o1"},
            {"trace_id": "b", "input": "i2", "output": "o2"},
        ],
    )
    (tmp_path / "t.csv").write_text("trace_id,input,output\na,i1,o1\nb,i2,o2\n")
    assert ingest_jsonl(tmp_path / "t.jsonl") == ingest_csv(tmp_path / "t.csv")


def test_a_blank_csv_trace_id_cell_falls_back_to_the_content_hash(tmp_path: Path):
    """An empty trace_id cell means 'no id', so CSV and JSONL derive the same hash id."""
    (tmp_path / "t.csv").write_text("trace_id,input,output\n,i,o\n")
    assert ingest_csv(tmp_path / "t.csv")[0].trace_id == compute_trace_id("i", "o")


def test_csv_metadata_stays_a_string_while_jsonl_metadata_keeps_its_type(tmp_path: Path):
    """Limit: CSV has no types, so a numeric column is metadata text, not a number."""
    write_jsonl(tmp_path / "t.jsonl", [{"input": "i", "output": "o", "latency_s": 1.5}])
    (tmp_path / "t.csv").write_text("input,output,latency_s\ni,o,1.5\n")
    assert ingest_jsonl(tmp_path / "t.jsonl")[0].metadata == {"latency_s": 1.5}
    assert ingest_csv(tmp_path / "t.csv")[0].metadata == {"latency_s": "1.5"}


def test_csv_quoted_fields_may_contain_commas_and_newlines(tmp_path: Path):
    """Multi-line, comma-bearing traces survive CSV quoting intact."""
    (tmp_path / "t.csv").write_text('input,output\n"line one\nline two","yes, really"\n')
    trace = ingest_csv(tmp_path / "t.csv")[0]
    assert trace.input == "line one\nline two"
    assert trace.output == "yes, really"


def test_ingest_picks_csv_only_for_a_lowercase_csv_suffix(tmp_path: Path):
    """Anything that is not `.csv` is parsed as JSONL, whatever it is called."""
    write_jsonl(tmp_path / "traces.txt", [{"input": "i", "output": "o"}])
    assert ingest(tmp_path / "traces.txt")[0].output == "o"


@pytest.mark.xfail(strict=True, reason="traces.py:78 compares path.suffix case-sensitively, so a "
                   "Windows/Excel-style TRACES.CSV is parsed as JSONL and crashes")
def test_ingest_treats_an_uppercase_csv_suffix_as_csv(tmp_path: Path):
    """A `.CSV` file should be read as CSV."""
    (tmp_path / "TRACES.CSV").write_text("input,output\ni,o\n")
    assert ingest(tmp_path / "TRACES.CSV")[0].output == "o"


# --- required fields and what counts as missing ----------------------------------


def test_an_empty_string_output_is_accepted_but_null_is_not(tmp_path: Path):
    """A model that answered with nothing is a real trace; a missing field is bad data."""
    write_jsonl(tmp_path / "ok.jsonl", [{"input": "i", "output": ""}])
    assert ingest_jsonl(tmp_path / "ok.jsonl")[0].output == ""
    write_jsonl(tmp_path / "bad.jsonl", [{"input": "i", "output": None}])
    with pytest.raises(ValueError, match="missing required field 'output'"):
        ingest_jsonl(tmp_path / "bad.jsonl")


def test_a_csv_row_with_too_few_cells_is_rejected_by_row_number(tmp_path: Path):
    """A truncated CSV row names its row instead of importing a half trace."""
    (tmp_path / "t.csv").write_text("input,output\ni1,o1\nonly-input\n")
    with pytest.raises(ValueError, match="row 2: missing required field 'output'"):
        ingest_csv(tmp_path / "t.csv")


def test_a_csv_without_an_input_column_is_rejected(tmp_path: Path):
    """Column names are exact: a file headed `prompt,response` is refused, not guessed at."""
    (tmp_path / "t.csv").write_text("prompt,response\ni,o\n")
    with pytest.raises(ValueError, match="missing required field 'input'"):
        ingest_csv(tmp_path / "t.csv")


def test_a_csv_row_with_extra_cells_is_rejected(tmp_path: Path):
    """Limit: a ragged CSV row (more cells than headers) is refused rather than truncated."""
    (tmp_path / "t.csv").write_text("input,output\ni,o,surplus\n")
    with pytest.raises(ValueError):
        ingest_csv(tmp_path / "t.csv")


@pytest.mark.xfail(strict=True, reason="traces.py:55 opens the CSV as plain utf-8, so the BOM Excel "
                   "prepends glues onto the first header and 'input' is reported missing")
def test_a_csv_saved_by_excel_with_a_utf8_bom_is_readable(tmp_path: Path):
    """A BOM-prefixed CSV should ingest like any other."""
    (tmp_path / "t.csv").write_bytes(b"\xef\xbb\xbfinput,output\ni,o\n")
    assert ingest_csv(tmp_path / "t.csv")[0].input == "i"


# --- ids -------------------------------------------------------------------------


def test_an_empty_source_trace_id_is_treated_as_no_id(tmp_path: Path):
    """`"trace_id": ""` is not kept as an id; the content hash replaces it."""
    write_jsonl(tmp_path / "t.jsonl", [{"trace_id": "", "input": "i", "output": "o"}])
    assert ingest_jsonl(tmp_path / "t.jsonl")[0].trace_id == compute_trace_id("i", "o")


def test_an_integer_jsonl_trace_id_is_kept_as_its_decimal_string(tmp_path: Path):
    """`"trace_id": 123` becomes "123", the id the CSV export of the same row carries."""
    write_jsonl(tmp_path / "t.jsonl", [{"trace_id": 123, "input": "i", "output": "o"}])
    assert ingest_jsonl(tmp_path / "t.jsonl")[0].trace_id == "123"


def test_a_float_or_boolean_jsonl_trace_id_is_rejected_not_stringified(tmp_path: Path):
    """Limit: a float or a boolean id has no one spelling to keep, so it is refused."""
    for bad in (123.0, True):
        write_jsonl(tmp_path / "t.jsonl", [{"trace_id": bad, "input": "i", "output": "o"}])
        with pytest.raises(ValueError, match="row 1: trace_id: "):
            ingest_jsonl(tmp_path / "t.jsonl")


def test_duplicate_trace_ids_are_all_kept_and_never_deduplicated(tmp_path: Path):
    """Limit: ingest does not police id uniqueness; two rows with one id are two traces."""
    write_jsonl(
        tmp_path / "t.jsonl",
        [
            {"trace_id": "same", "input": "i1", "output": "o1"},
            {"trace_id": "same", "input": "i2", "output": "o2"},
        ],
    )
    traces = ingest_jsonl(tmp_path / "t.jsonl")
    store = Store(tmp_path)
    store.write_traces(traces)
    assert [t.trace_id for t in store.read_traces()] == ["same", "same"]


def test_identical_rows_without_ids_share_one_content_id(tmp_path: Path):
    """The hash id is a pure function of content, so a repeated row repeats its id."""
    write_jsonl(tmp_path / "t.jsonl", [{"input": "i", "output": "o"}] * 2)
    a, b = ingest_jsonl(tmp_path / "t.jsonl")
    assert a.trace_id == b.trace_id


def test_the_content_id_depends_on_input_and_output_separately():
    """Swapping input and output yields a different id: the two fields are not interchangeable."""
    assert compute_trace_id("x", "y") != compute_trace_id("y", "x")


@pytest.mark.xfail(strict=True, reason="traces.py:31 joins input and output with a NUL byte but "
                   "does not escape it, so ('a\\0b','c') and ('a','b\\0c') hash identically")
def test_the_content_id_cannot_collide_across_the_input_output_boundary():
    """Moving text across the input/output boundary must change the id."""
    assert compute_trace_id("a\x00b", "c") != compute_trace_id("a", "b\x00c")


# --- malformed and awkward JSONL --------------------------------------------------


def test_blank_and_whitespace_only_jsonl_lines_are_skipped(tmp_path: Path):
    """Trailing newlines and blank separator lines never produce a phantom trace."""
    (tmp_path / "t.jsonl").write_text(
        '\n{"input": "i1", "output": "o1"}\n   \n\n{"input": "i2", "output": "o2"}\n\n'
    )
    assert [t.input for t in ingest_jsonl(tmp_path / "t.jsonl")] == ["i1", "i2"]


def test_jsonl_row_numbers_in_errors_count_blank_lines(tmp_path: Path):
    """An error's 'row N' is the physical line number, so it matches what an editor shows."""
    (tmp_path / "t.jsonl").write_text('\n\n{"input": "i"}\n')
    with pytest.raises(ValueError, match="row 3"):
        ingest_jsonl(tmp_path / "t.jsonl")


def test_one_bad_row_aborts_the_whole_ingest_and_writes_nothing(tmp_path: Path):
    """Ingest is all-or-nothing: a bad row halfway leaves the previous corpus untouched."""
    good = write_jsonl(tmp_path / "good.jsonl", [{"input": "i", "output": "o"}])
    repo = tmp_path / "repo"
    assert runner.invoke(app, ["ingest", str(good), "--repo", str(repo)]).exit_code == 0
    before = Store(repo).traces_path.read_text()

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"input": "i2", "output": "o2"}\n{"input": "no output"}\n')
    result = runner.invoke(app, ["ingest", str(bad), "--repo", str(repo)])

    assert result.exit_code != 0
    assert Store(repo).traces_path.read_text() == before


def test_reingesting_replaces_the_corpus_instead_of_appending(tmp_path: Path):
    """Limit: `ingest` is a replace, not a merge; to combine sources, concatenate them first."""
    a = write_jsonl(tmp_path / "a.jsonl", [{"input": "a", "output": "a"}])
    b = write_jsonl(tmp_path / "b.jsonl", [{"input": "b", "output": "b"}])
    repo = tmp_path / "repo"
    runner.invoke(app, ["ingest", str(a), "--repo", str(repo)])
    runner.invoke(app, ["ingest", str(b), "--repo", str(repo)])
    assert [t.input for t in Store(repo).read_traces()] == ["b"]


def test_an_empty_file_ingests_as_zero_traces(tmp_path: Path):
    """An empty export is not an error; it is a corpus of nothing."""
    (tmp_path / "t.jsonl").write_text("")
    assert ingest_jsonl(tmp_path / "t.jsonl") == []


def test_a_missing_source_file_fails_the_cli_without_creating_artifacts(tmp_path: Path):
    """A typo'd ingest path exits non-zero and does not create `.axial/`."""
    result = runner.invoke(app, ["ingest", str(tmp_path / "nope.jsonl"), "--repo", str(tmp_path)])
    assert result.exit_code != 0
    assert not (tmp_path / ".axial").exists()


def test_a_json_line_that_is_not_an_object_is_rejected(tmp_path: Path):
    """Limit: each JSONL line must be an object; a bare array or number is refused."""
    (tmp_path / "t.jsonl").write_text("[1, 2]\n")
    with pytest.raises(Exception):  # noqa: B017 - the exact type is the defect below
        ingest_jsonl(tmp_path / "t.jsonl")


def test_a_non_object_json_line_is_reported_as_a_row_error(tmp_path: Path):
    """A non-object line should fail like every other bad row: ValueError naming the row."""
    (tmp_path / "t.jsonl").write_text("[1, 2]\n")
    with pytest.raises(ValueError, match="row 1"):
        ingest_jsonl(tmp_path / "t.jsonl")


def test_a_corrupt_jsonl_line_is_reported_with_its_row_number(tmp_path: Path):
    """A syntax error in line 2 should say 'row 2', as missing-field errors do."""
    (tmp_path / "t.jsonl").write_text('{"input": "i", "output": "o"}\n{oops\n')
    with pytest.raises(ValueError, match="row 2"):
        ingest_jsonl(tmp_path / "t.jsonl")


def test_a_corrupt_jsonl_line_is_reported_in_one_line_without_jsons_own_line_count(
    tmp_path: Path,
):
    """json counts lines within the one line it was given, which would read as 'line 1'."""
    (tmp_path / "t.jsonl").write_text('{"input": "i", "output": "o"}\n{oops\n')
    with pytest.raises(ValueError) as caught:
        ingest_jsonl(tmp_path / "t.jsonl")
    assert str(caught.value) == (
        "row 2: invalid JSON at column 2: Expecting property name enclosed in double quotes"
    )


# --- unicode, size, metadata -------------------------------------------------------


def test_unicode_traces_survive_ingest_and_the_store_unchanged(tmp_path: Path):
    """Non-Latin text, emoji and combining marks round-trip byte-for-byte."""
    text = "café 日本語 \U0001f600 é"
    write_jsonl(tmp_path / "t.jsonl", [{"input": text, "output": text[::-1]}])
    store = Store(tmp_path / "repo")
    store.write_traces(ingest_jsonl(tmp_path / "t.jsonl"))
    back = store.read_traces()[0]
    assert (back.input, back.output) == (text, text[::-1])


def test_a_very_long_trace_is_kept_whole(tmp_path: Path):
    """There is no truncation limit on trace text: a 2 MB output comes back complete."""
    big = "x" * 2_000_000
    write_jsonl(tmp_path / "t.jsonl", [{"input": "i", "output": big}])
    store = Store(tmp_path / "repo")
    store.write_traces(ingest_jsonl(tmp_path / "t.jsonl"))
    assert store.read_traces()[0].output == big


def test_a_row_key_named_metadata_is_nested_not_merged(tmp_path: Path):
    """Extra keys are all filed under `metadata`, including one that is itself called that."""
    write_jsonl(tmp_path / "t.jsonl", [{"input": "i", "output": "o", "metadata": {"a": 1}}])
    assert ingest_jsonl(tmp_path / "t.jsonl")[0].metadata == {"metadata": {"a": 1}}


def test_reserved_keys_never_leak_into_metadata(tmp_path: Path):
    """trace_id, input and output are promoted to fields and not duplicated as metadata."""
    write_jsonl(tmp_path / "t.jsonl", [{"trace_id": "a", "input": "i", "output": "o", "k": 1}])
    trace = ingest_jsonl(tmp_path / "t.jsonl")[0]
    assert trace == Trace(trace_id="a", input="i", output="o", metadata={"k": 1})


def test_metadata_that_json_cannot_hold_fails_the_write_and_keeps_the_old_corpus(tmp_path: Path):
    """A trace built in code with an unserialisable metadata value cannot corrupt the store."""
    store = Store(tmp_path)
    store.write_traces([Trace(trace_id="ok", input="i", output="o")])
    before = store.traces_path.read_text()
    bad = Trace(trace_id="bad", input="i", output="o", metadata={"when": object()})
    with pytest.raises(TypeError):
        store.write_traces([bad])
    assert store.traces_path.read_text() == before
