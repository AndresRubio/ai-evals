import json
from pathlib import Path

import pytest

from axial.core.traces import Trace, compute_trace_id, ingest_csv, ingest_jsonl


def test_compute_trace_id_is_stable_and_short():
    a = compute_trace_id("in", "out")
    b = compute_trace_id("in", "out")
    assert a == b
    assert len(a) == 16
    assert a != compute_trace_id("in", "other")


def test_ingest_jsonl_uses_source_id_when_present(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"trace_id": "abc", "input": "i", "output": "o"}) + "\n")
    traces = ingest_jsonl(p)
    assert len(traces) == 1
    assert traces[0].trace_id == "abc"


def test_ingest_jsonl_falls_back_to_content_hash(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"input": "i", "output": "o"}) + "\n")
    traces = ingest_jsonl(p)
    assert traces[0].trace_id == compute_trace_id("i", "o")


def test_ingest_jsonl_keeps_unknown_keys_as_metadata(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"input": "i", "output": "o", "latency_s": 1.5}) + "\n")
    assert ingest_jsonl(p)[0].metadata == {"latency_s": 1.5}


def test_ingest_jsonl_rejects_a_row_missing_output(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"input": "i"}) + "\n")
    with pytest.raises(ValueError, match="row 1"):
        ingest_jsonl(p)


@pytest.mark.parametrize(
    ("row", "field"),
    [
        ({"trace_id": "b", "input": "emoji cut \ud83d", "output": "o"}, "input"),
        ({"trace_id": "b", "input": "i", "output": "emoji cut \ud83d"}, "output"),
        ({"input": "emoji cut \ud83d", "output": "o"}, "input"),
    ],
    ids=["input", "output", "input-without-id"],
)
def test_ingest_jsonl_refuses_a_lone_surrogate_naming_the_row_and_field(
    tmp_path: Path, row: dict, field: str
):
    """JSON allows a lone surrogate such as "\\ud83d", which JavaScript writes when it
    cuts an emoji in half. UTF-8 cannot encode one, and Jev's client on httpx 0.28 or
    later, and litellm's OpenAI route, fail on it before sending a prompt that carries
    the trace. A row without an id used to fail first in the hash that derives one,
    naming no row."""
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps({"input": "fine", "output": "fine"}) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError) as caught:
        ingest_jsonl(path)
    assert str(caught.value) == (
        f"row 2 has an {field} that is not valid UTF-8 (a lone surrogate at character 11, "
        "which a cut emoji can leave)"
    )


def test_ingest_csv_reads_input_and_output_columns(tmp_path: Path):
    p = tmp_path / "t.csv"
    p.write_text("input,output\ni,o\n")
    traces = ingest_csv(p)
    assert traces[0].input == "i"
    assert traces[0].output == "o"


def test_trace_round_trips_through_json():
    t = Trace(trace_id="x", input="i", output="o", metadata={"k": 1})
    assert Trace.model_validate_json(t.model_dump_json()) == t


def write_row(tmp_path: Path, row: dict) -> Path:
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps(row) + "\n")
    return path


@pytest.mark.parametrize("trace_id", [7, 0])
def test_an_integer_trace_id_is_kept_as_its_decimal_string(tmp_path: Path, trace_id: int):
    """The CSV export of the same rows carries "7", so both formats give one id. A 0 is an
    id like any other, never swapped for the content hash."""
    path = write_row(tmp_path, {"trace_id": trace_id, "input": "i", "output": "o"})
    assert ingest_jsonl(path)[0].trace_id == str(trace_id)


@pytest.mark.parametrize(
    ("row", "field", "got"),
    [
        ({"trace_id": 7.5, "input": "i", "output": "o"}, "trace_id", "float"),
        ({"trace_id": True, "input": "i", "output": "o"}, "trace_id", "bool"),
        ({"trace_id": [], "input": "i", "output": "o"}, "trace_id", "list"),
        ({"input": [{"role": "user", "content": "x"}], "output": "o"}, "input", "list"),
        ({"input": "i", "output": {"text": "o"}}, "output", "dict"),
    ],
    ids=["float-id", "bool-id", "empty-list-id", "chat-input", "object-output"],
)
def test_a_wrongly_typed_field_is_refused_in_one_line_naming_the_row_and_field(
    tmp_path: Path, row: dict, field: str, got: str
):
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps({"input": "fine", "output": "fine"}) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError) as caught:
        ingest_jsonl(path)
    message = str(caught.value)
    assert message.startswith(f"row 2: {field}: ")
    assert message.endswith(f"got {got}")
    assert "\n" not in message
    assert "pydantic" not in message


def test_a_csv_row_with_more_cells_than_the_header_is_refused_in_one_line(tmp_path: Path):
    path = tmp_path / "t.csv"
    path.write_text("input,output\ni,o\ni,o,surplus,more\n")
    with pytest.raises(ValueError, match=r"^row 2: 2 more cell\(s\) than the header has columns$"):
        ingest_csv(path)
