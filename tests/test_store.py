from pathlib import Path

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
