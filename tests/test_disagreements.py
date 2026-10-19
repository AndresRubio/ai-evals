"""The dev disagreement listing in alignment.md and alignment.json."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from measure_helpers import MEASURE_MODE, build_promoted_judge
from typer.testing import CliRunner

from axial import cli
from axial.align import align
from axial.dataset import Label
from axial.errors import InvariantViolation
from axial.report import (
    disagreement_counts,
    find_disagreements,
    parse_reason,
    write_alignment,
)
from axial.store import Store
from axial.traces import Trace


def row(trace_id: str, truth: bool, verdict: bool, *, split: str = "dev", raw: str | None = None):
    if raw is None:
        raw = json.dumps({"verdict": "fail" if verdict else "pass", "reason": f"why {trace_id}"})
    return {"trace_id": trace_id, "split": split, "truth": truth, "verdict": verdict, "raw": raw}


def label(trace_id: str, value: bool, critique: str = "a critique") -> Label:
    return Label(trace_id=trace_id, label=value, critique=critique, annotator="andres")


def trace(trace_id: str, text: str = "short") -> Trace:
    return Trace(trace_id=trace_id, input=f"in {text}", output=f"out {text}")


VERDICTS = [
    row("d4", truth=False, verdict=True),   # false alarm
    row("d2", truth=True, verdict=False),   # missed failure
    row("d1", truth=True, verdict=True),    # agreement
    row("d3", truth=False, verdict=False),  # agreement
    row("d0", truth=True, verdict=False),   # missed failure
    row("t1", truth=True, verdict=False, split="test"),
]
LABELS = [label(t, t in ("d0", "d1", "d2", "t1")) for t in ("d0", "d1", "d2", "d3", "d4", "t1")]
TRACES = [trace(t) for t in ("d0", "d1", "d2", "d3", "d4", "t1")]


# --- finding them ------------------------------------------------------------------


def test_only_disagreements_are_listed_missed_failures_first_then_by_id():
    found = find_disagreements(VERDICTS, LABELS, TRACES)
    assert [(d.kind, d.trace_id) for d in found] == [
        ("missed_failure", "d0"),
        ("missed_failure", "d2"),
        ("false_alarm", "d4"),
    ]
    assert found[0].critique == "a critique" and found[0].annotator == "andres"
    assert found[0].judge_reason == "why d0"


def test_test_split_disagreements_are_never_listed():
    assert all(d.trace_id != "t1" for d in find_disagreements(VERDICTS, LABELS, TRACES))
    with pytest.raises(InvariantViolation, match="counted, never listed"):
        find_disagreements(VERDICTS, LABELS, TRACES, split="test")


def test_a_missing_critique_shows_a_placeholder():
    found = find_disagreements([row("d0", True, False)], [label("d0", True, "  ")], [trace("d0")])
    assert found[0].critique == "(no critique recorded)"


def test_the_judge_reason_comes_from_reason_then_critique_then_the_raw_text():
    assert parse_reason('{"verdict": "fail", "reason": "two headlines"}') == "two headlines"
    assert parse_reason('Sure: {"verdict": "pass", "critique": "one story"}') == "one story"
    assert parse_reason('{"verdict": "pass", "reason": ""}') == '{"verdict": "pass", "reason": ""}'
    assert parse_reason("no json here") == "no json here"
    assert parse_reason("x" * 400) == "x" * 300 + "…"
    assert parse_reason("") == "(no reason recorded)"


def test_excerpts_are_the_first_300_characters_with_the_full_length():
    long_trace = Trace(trace_id="d0", input="i" * 2184, output="o" * 64)
    found = find_disagreements([row("d0", True, False)], [label("d0", True)], [long_trace])
    d = found[0]
    assert (d.input_excerpt, d.input_length) == ("i" * 300, 2184)
    assert (d.output_excerpt, d.output_length) == ("o" * 64, 64)


def test_a_trace_missing_from_the_corpus_has_no_excerpt():
    d = find_disagreements([row("d0", True, False)], [label("d0", True)], [])[0]
    assert (d.input_excerpt, d.input_length, d.output_excerpt) == (None, None, None)


def test_the_counts_line():
    assert disagreement_counts([]) == "dev: no disagreements"
    found = find_disagreements(VERDICTS, LABELS, TRACES)
    assert disagreement_counts(found) == "dev: 2 missed failures, 1 false alarm (listed in alignment.md)"


# --- rendering them ----------------------------------------------------------------


def perfect():
    y = [True] * 30 + [False] * 30
    return align(y, list(y), n_boot=200, seed=1)


def render(tmp_path: Path, disagreements) -> tuple[str, dict]:
    store = Store(tmp_path)
    path = write_alignment(store, "merged", 1, perfect(), None, n_errors=0, disagreements=disagreements)
    return path.read_text(), json.loads((path.parent / "alignment.json").read_text())


def test_the_markdown_lists_each_case_with_its_excerpts(tmp_path: Path):
    traces = [Trace(trace_id="d0", input="i" * 2184, output="o" * 64), trace("d2"), trace("d4")]
    md, _ = render(tmp_path, find_disagreements(VERDICTS, LABELS, traces))
    assert "## Dev disagreements" in md
    assert "### Missed failures (2)" in md and "### False alarms (1)" in md
    assert md.index("missed failure — `d2`") < md.index("false alarm — `d4`")
    assert "**Human** (annotator `andres`): a critique" in md
    assert "**Judge:** why d0" in md
    assert "Input (first 300 of 2,184 characters):\n\n```\n" + "i" * 300 + "\n```" in md
    assert "Output (all 64 characters):" in md
    assert "Test disagreements are counted in the table above and never listed here" in md


def test_a_backtick_run_in_an_excerpt_gets_a_longer_fence(tmp_path: Path):
    tricky = Trace(trace_id="d0", input="see ````code````", output="fine")
    md, _ = render(tmp_path, find_disagreements([row("d0", True, False)], [label("d0", True)], [tricky]))
    assert "\n`````\nsee ````code````\n`````\n" in md


def test_a_missing_trace_is_named_in_the_markdown(tmp_path: Path):
    md, _ = render(tmp_path, find_disagreements([row("d0", True, False)], [label("d0", True)], []))
    assert "Input: (trace not found in .axial/traces)" in md


def test_html_in_the_critique_and_the_reason_is_shown_as_text(tmp_path: Path):
    # An unclosed <details> would collapse the rest of the file on GitHub.
    critique = "opened a <details> block and\nnever closed it"
    raw = json.dumps({"verdict": "pass", "reason": "it said <script>x</script> & left"})
    found = find_disagreements(
        [row("d0", True, False, raw=raw)], [label("d0", True, critique)], [trace("d0")]
    )
    md, data = render(tmp_path, found)
    assert "<details>" not in md and "<script>" not in md
    assert "**Human** (annotator `andres`): opened a &lt;details&gt; block and never closed it\n" in md
    assert "**Judge:** it said &lt;script&gt;x&lt;/script&gt; &amp; left\n" in md
    # alignment.json is data, not Markdown, so it keeps the text as written.
    assert data["disagreements"]["dev"][0]["critique"] == critique


@pytest.mark.parametrize(
    ("text", "span"),
    [
        ("d0", "`d0`"),
        ("a`b", "``a`b``"),
        # CommonMark strips one space from each side, so a span may start or end with `.
        ("`x``", "``` `x`` ```"),
        ("two\n#### lines", "`two #### lines`"),
    ],
)
def test_trace_ids_and_annotators_are_safe_code_spans(tmp_path: Path, text: str, span: str):
    found = find_disagreements(
        [row(text, True, False)], [Label(trace_id=text, label=True, critique="c", annotator=text)], []
    )
    md, _ = render(tmp_path, found)
    assert f"#### missed failure — {span}\n" in md
    assert f"**Human** (annotator {span}): c\n" in md
    assert not any(line.startswith("#### lines") for line in md.splitlines())


def test_no_disagreements_says_so(tmp_path: Path):
    md, data = render(tmp_path, [])
    assert "## Dev disagreements\n\nNo dev disagreements.\n" in md
    assert data["disagreements"] == {"dev": []}


def test_alignment_json_lists_each_case_without_excerpts(tmp_path: Path):
    _, data = render(tmp_path, find_disagreements(VERDICTS, LABELS, TRACES))
    assert data["disagreements"]["dev"][0] == {
        "trace_id": "d0", "kind": "missed_failure", "critique": "a critique", "judge_reason": "why d0",
    }


def test_without_disagreements_the_report_is_byte_identical_to_before(tmp_path: Path):
    """The hashes are of the files `write_alignment` wrote for these exact inputs at commit
    3673962, before the listing existed. A change here changes every existing report."""
    y = [True] * 30 + [False] * 30
    test = align(y, [True] * 27 + [False] * 3 + [False] * 28 + [True] * 2, n_boot=200, seed=1)
    store = Store(tmp_path)
    path = write_alignment(store, "merged", 1, perfect(), test, n_errors=2, notes=["a note"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "548fb47a2ed4ffc3fb4a3139e3a128e29fbb12af6a9af321cdab2fd5a0096362"
    )
    assert hashlib.sha256((path.parent / "alignment.json").read_bytes()).hexdigest() == (
        "d256897b394a2afc023aa2c5d59ce7ff16a400a723d08849553df955bdb978be"
    )


# --- through the CLI -----------------------------------------------------------------


def test_axial_align_prints_the_counts_and_writes_the_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = Store(tmp_path)
    build_promoted_judge(store)
    path = store.judge_dir(MEASURE_MODE, 1) / "verdicts.jsonl"
    rows = store.read_jsonl(path)
    dev = [r for r in rows if r["split"] == "dev"]
    dev[0]["verdict"] = not dev[0]["verdict"]  # dev-00 is a real failure: now missed
    dev[1]["verdict"] = not dev[1]["verdict"]  # dev-01 is a real pass: now a false alarm
    store.write_jsonl(path, rows)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli.app, ["align", MEASURE_MODE, "1"])

    assert result.exit_code == 0, result.output
    assert "dev: 1 missed failure, 1 false alarm (listed in alignment.md)" in result.output
    md = (store.judge_dir(MEASURE_MODE, 1) / "alignment.md").read_text()
    assert "missed failure — `dev-00`" in md and "false alarm — `dev-01`" in md
