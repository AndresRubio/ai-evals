import json
from pathlib import Path

import pytest

from axial.traces import Trace, compute_trace_id, ingest_csv, ingest_jsonl


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


def test_ingest_csv_reads_input_and_output_columns(tmp_path: Path):
    p = tmp_path / "t.csv"
    p.write_text("input,output\ni,o\n")
    traces = ingest_csv(p)
    assert traces[0].input == "i"
    assert traces[0].output == "o"


def test_trace_round_trips_through_json():
    t = Trace(trace_id="x", input="i", output="o", metadata={"k": 1})
    assert Trace.model_validate_json(t.model_dump_json()) == t
