"""`axial measure` and `axial gate --batch` through the CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from measure_helpers import MEASURE_MODE, build_promoted_judge, fake_judge_builder, write_batch
from typer.testing import CliRunner, Result

from axial import cli
from axial.store import Store

runner = CliRunner()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    build_promoted_judge(Store(tmp_path))
    monkeypatch.setattr(cli, "build_judge_model", fake_judge_builder())
    monkeypatch.chdir(tmp_path)
    return tmp_path


def axial(*args: str) -> Result:
    return runner.invoke(cli.app, list(args))


def assert_clean(result: Result, code: int, *fragments: str) -> None:
    assert result.exit_code == code, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit), result.exception
    for fragment in fragments:
        assert fragment in result.output, (fragment, result.output)


def measure(repo: Path, name: str, n_bad: int, n: int = 500, *more: str) -> Result:
    path = write_batch(repo / f"{name}.jsonl", ["BAD"] * n_bad + ["GOOD"] * (n - n_bad))
    return axial("measure", MEASURE_MODE, "--version", "1", "--traces", str(path), "--name", name, *more)


# --- measure ------------------------------------------------------------------------


def test_measure_prints_the_corrected_rate_and_its_notes(repo: Path):
    result = measure(repo, "rel-1", 60)
    assert_clean(
        result, 0,
        "merged v1 on rel-1: observed 0.120 -> corrected 0.072 [0.000, 0.128]",
        "(judge exact: [0.042, 0.107]), n=500 graded, 0 set aside",
        "note: the correction assumes the judge errs on this batch",
    )
    assert (repo / ".axial" / "measurements" / MEASURE_MODE / "rel-1" / "measurement.json").exists()


def test_measure_refuses_an_existing_name_and_replace_overrides(repo: Path):
    assert measure(repo, "rel-1", 60).exit_code == 0
    assert_clean(measure(repo, "rel-1", 60), 1, "already exists", "--replace")
    assert_clean(measure(repo, "rel-1", 80, 500, "--replace"), 0, "observed 0.160")


def test_measure_refuses_an_unpromoted_judge(repo: Path):
    (repo / ".axial" / "judges" / MEASURE_MODE / "v1" / "PROMOTED").unlink()
    assert_clean(measure(repo, "rel-1", 60), 1, "has not been promoted")


def test_measure_reports_an_unreadable_batch_file(repo: Path):
    result = axial("measure", MEASURE_MODE, "--version", "1", "--traces", "nope.jsonl", "--name", "x")
    assert_clean(result, 1, "could not read nope.jsonl")


def test_measure_exits_non_zero_after_writing_a_batch_nothing_could_grade(repo: Path):
    path = write_batch(repo / "garbled.jsonl", ["GARBLED"] * 3)
    result = axial("measure", MEASURE_MODE, "--version", "1", "--traces", str(path), "--name", "g")
    assert_clean(result, 1, "no rate, all 3 traces set aside", "every trace was set aside")
    assert (repo / ".axial" / "measurements" / MEASURE_MODE / "g" / "errors.jsonl").exists()


# --- gate --batch -------------------------------------------------------------------


def test_gate_batch_exit_codes(repo: Path):
    measure(repo, "base", 40)
    measure(repo, "cand", 60)
    assert_clean(axial("gate", MEASURE_MODE, "--batch", "cand"), 0, "corrected 0.072")
    assert_clean(
        axial("gate", MEASURE_MODE, "--batch", "cand", "--baseline-batch", "base"), 1,
        "paired on 500 shared inputs", "REGRESSION",
    )
    assert_clean(
        axial("gate", MEASURE_MODE, "--batch", "base", "--baseline-batch", "cand"), 0, "improvement"
    )
    assert_clean(axial("gate", MEASURE_MODE, "--batch", "cand", "--max-rate", "0.1"), 1, "ABOVE CEILING")
    assert_clean(axial("gate", MEASURE_MODE, "--batch", "cand", "--max-rate", "0.2"), 0, "within the ceiling")


def test_gate_batch_names_a_missing_measurement(repo: Path):
    assert_clean(axial("gate", MEASURE_MODE, "--batch", "nope"), 1, "no measurement named 'nope'")


@pytest.mark.parametrize(
    ("args", "fragments"),
    [
        ((), ("--version", "--batch")),
        (("--version", "1", "--batch", "x"), ("--version", "--batch")),
        (("--version", "1", "--max-rate", "0.1"), ("--max-rate", "need --batch")),
        (("--version", "1", "--baseline-batch", "x"), ("--baseline-batch", "need --batch")),
        (("--batch", "x", "--baseline", "1"), ("--baseline", "--baseline-batch")),
        (("--batch", "x", "--max-rate", "1.5"), ("strictly between 0 and 1",)),
        (("--batch", "x", "--max-rate", "0"), ("strictly between 0 and 1",)),
    ],
)
def test_gate_option_mistakes_are_usage_errors(repo: Path, args: tuple[str, ...], fragments: tuple[str, ...]):
    assert_clean(axial("gate", MEASURE_MODE, *args), 2, *fragments)
