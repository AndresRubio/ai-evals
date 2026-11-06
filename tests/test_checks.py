import math
from pathlib import Path

import pytest

from axial.core.traces import Trace, ingest, register_adapter
from axial.judge.checks import (
    ablated_spec,
    ci_half_width,
    cost_usd,
    slice_alignment,
    verbosity_bias,
)
from axial.judge.spec import JudgeSpec


def verdicts(rows):
    return [
        {"trace_id": tid, "split": "test", "verdict": v, "truth": t}
        for tid, v, t in rows
    ]


def test_ci_half_width_matches_the_published_rule_of_thumb():
    # 25 cases at 2 reps is about plus or minus 14 points.
    assert ci_half_width(25, reps=2) == pytest.approx(0.1414, abs=0.002)
    assert ci_half_width(100, reps=2) == pytest.approx(0.0707, abs=0.002)


def test_ci_half_width_rejects_an_empty_set():
    with pytest.raises(ValueError):
        ci_half_width(0)


def test_slice_alignment_splits_by_a_metadata_key():
    traces = [
        Trace(trace_id="a", input="i", output="o", metadata={"topic": "x"}),
        Trace(trace_id="b", input="i", output="o", metadata={"topic": "y"}),
    ]
    rows = verdicts([("a", True, True), ("b", False, True)])
    result = slice_alignment(rows, traces, "topic", split="test")
    assert set(result) == {"x", "y"}
    assert result["x"].tpr == 1.0
    assert result["y"].tpr == 0.0


def test_slice_alignment_ignores_traces_without_the_key():
    traces = [Trace(trace_id="a", input="i", output="o", metadata={})]
    assert slice_alignment(verdicts([("a", True, True)]), traces, "topic", split="test") == {}


def test_verbosity_bias_is_positive_when_the_judge_fails_longer_outputs():
    traces = [
        Trace(trace_id="a", input="i", output="x" * 10),
        Trace(trace_id="b", input="i", output="x" * 20),
        Trace(trace_id="c", input="i", output="x" * 30),
        Trace(trace_id="d", input="i", output="x" * 40),
    ]
    rows = verdicts([("a", False, False), ("b", False, False), ("c", True, True), ("d", True, True)])
    # Pearson correlation between these four lengths and this exact fail pattern
    # is ~0.894 (verified against numpy directly), not the >0.9 the brief's own
    # draft asserted -- the within-group spread (10 vs 20, 30 vs 40) caps it
    # below 1.0. 0.85 keeps the intent ("clearly, strongly positive") without
    # asserting a bound the fixture data cannot reach.
    assert verbosity_bias(rows, traces) > 0.85


def test_verbosity_bias_is_nan_when_every_verdict_is_the_same():
    traces = [Trace(trace_id="a", input="i", output="xx"), Trace(trace_id="b", input="i", output="y")]
    rows = verdicts([("a", True, True), ("b", True, True)])
    assert math.isnan(verbosity_bias(rows, traces))


def test_cost_is_derived_from_recorded_tokens():
    # claude-opus-5: $5 per Mtok in, $25 per Mtok out.
    cost = cost_usd({"input_tokens": 1_000_000, "output_tokens": 1_000_000}, "claude-opus-5")
    assert cost == pytest.approx(30.0)


def test_cost_is_none_for_an_unpriced_model():
    """Better no number than a flat assumed rate."""
    assert cost_usd({"input_tokens": 10, "output_tokens": 10}, "some-local-model") is None


def test_cost_is_none_for_a_row_with_no_usage():
    """Invariant 9: a missing `usage` is a runner bug, not a $0.00 measurement."""
    assert cost_usd({}, "claude-opus-5") is None


def test_cost_is_none_for_a_row_with_explicitly_empty_usage():
    """An explicitly empty `{}` -- the shape the model layer now records when a
    provider sent no token counts -- must be treated the same as a `usage` key
    that was never set at all."""
    usage: dict[str, int] = {}
    assert cost_usd(usage, "claude-opus-5") is None


def test_ablated_spec_drops_a_field_from_the_prompt():
    spec = JudgeSpec(mode="m", prompt_template="IN {input}\nOUT {output}")
    ablated = ablated_spec(spec, "input")
    assert "{input}" not in ablated.prompt_template
    assert "{output}" in ablated.prompt_template
    assert ablated.mode == "m"


def test_ablated_spec_rejects_an_unknown_field():
    with pytest.raises(ValueError, match="cannot ablate"):
        ablated_spec(JudgeSpec(mode="m"), "nonsense")


def test_a_registered_adapter_is_used_by_ingest(tmp_path: Path):
    path = tmp_path / "weird.log"
    path.write_text("in|out\n")

    def parse(p: Path) -> list[Trace]:
        left, right = p.read_text().strip().split("|")
        return [Trace(trace_id="x", input=left, output=right)]

    register_adapter("pipe", parse)
    assert ingest(path, adapter="pipe")[0].output == "out"


def test_ingest_picks_the_reader_from_the_suffix(tmp_path: Path):
    path = tmp_path / "t.csv"
    path.write_text("input,output\ni,o\n")
    assert ingest(path)[0].input == "i"


def test_an_unknown_adapter_is_rejected(tmp_path: Path):
    path = tmp_path / "t.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="no adapter"):
        ingest(path, adapter="nope")
