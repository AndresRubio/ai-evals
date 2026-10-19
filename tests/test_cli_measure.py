"""`axial measure` and `axial gate --batch` through the CLI."""

from __future__ import annotations

import json
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


def tree(root: Path) -> dict[str, bytes | None]:
    """Every path under `root` with its bytes, `None` for a directory."""
    return {
        str(p.relative_to(root)): None if p.is_dir() else p.read_bytes()
        for p in sorted(root.rglob("*"))
    }


# --- modes that are not one plain directory name ------------------------------------


def bad_mode(repo: Path, kind: str) -> str:
    """Each mode resolves to a path outside .axial/measurements/<mode>/; the first two
    to the promoted judge's own directory, which `measure --replace` then replaced."""
    return {
        "dotdot": f"../judges/{MEASURE_MODE}",
        "absolute": str(repo / ".axial" / "judges" / MEASURE_MODE),
        "slash": f"{MEASURE_MODE}/x",
        "trailing-slash": f"{MEASURE_MODE}/",
    }[kind]


MODE_KINDS = ["dotdot", "absolute", "slash", "trailing-slash"]


@pytest.mark.parametrize("kind", MODE_KINDS)
def test_measure_refuses_a_mode_that_is_not_a_plain_name_and_touches_nothing(repo: Path, kind: str):
    path = write_batch(repo / "b.jsonl", ["GOOD"] * 5)
    before = tree(repo / ".axial")
    result = axial(
        "measure", bad_mode(repo, kind), "--version", "1", "--traces", str(path),
        "--name", "v1", "--replace",
    )
    assert_clean(result, 1, "mode name")
    assert len(result.output.strip().splitlines()) == 1
    assert tree(repo / ".axial") == before


@pytest.mark.parametrize("kind", MODE_KINDS)
@pytest.mark.parametrize(
    "command", [("judge", "--split", "dev"), ("align", "1"), ("promote", "1"), ("gate", "--version", "1")]
)
def test_judge_align_promote_and_gate_refuse_a_mode_that_is_not_a_plain_name(
    repo: Path, kind: str, command: tuple[str, ...]
):
    before = tree(repo / ".axial")
    result = axial(command[0], bad_mode(repo, kind), *command[1:])
    assert_clean(result, 1, "mode name")
    assert len(result.output.strip().splitlines()) == 1
    assert tree(repo / ".axial") == before


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


def test_measure_reports_a_jsonl_line_that_is_not_an_object(repo: Path):
    bad = repo / "bad.jsonl"
    bad.write_text("[1, 2]\n")
    result = axial("measure", MEASURE_MODE, "--version", "1", "--traces", str(bad), "--name", "x")
    assert_clean(result, 1, "could not read", "row 1")


def test_measure_reports_a_csv_field_over_the_parser_limit(repo: Path):
    bad = repo / "huge.csv"
    bad.write_text("input,output\n" + "x" * 131073 + ",y\n")
    result = axial("measure", MEASURE_MODE, "--version", "1", "--traces", str(bad), "--name", "x")
    assert_clean(result, 1, "could not read")


def test_measure_refuses_a_batch_name_over_the_length_cap_in_one_line(repo: Path):
    path = write_batch(repo / "long.jsonl", ["BAD"] * 5 + ["GOOD"] * 15)
    result = axial(
        "measure", MEASURE_MODE, "--version", "1", "--traces", str(path), "--name", "a" * 250
    )
    assert_clean(result, 1, "250 characters long")
    assert len(result.output.strip().splitlines()) == 1


def test_measure_reports_a_failed_write_in_one_line(repo: Path, monkeypatch: pytest.MonkeyPatch):
    def disk_full(self: Store, path: Path, files: dict[str, str]) -> None:
        raise OSError(28, "No space left on device", str(path))

    monkeypatch.setattr(Store, "write_dir_atomically", disk_full)
    result = measure(repo, "rel-1", 5, 20)
    assert_clean(result, 1, "No space left on device")
    assert len(result.output.strip().splitlines()) == 1


def test_measure_refuses_a_lone_surrogate_in_one_line(repo: Path):
    path = repo / "cut.jsonl"
    path.write_text(
        '{"trace_id": "a", "input": "fine", "output": "GOOD"}\n'
        '{"trace_id": "b", "input": "emoji cut \\ud83d", "output": "GOOD"}\n'
    )
    result = axial("measure", MEASURE_MODE, "--version", "1", "--traces", str(path), "--name", "s")
    assert_clean(result, 1, "trace 'b' has an input that is not valid UTF-8")
    assert len(result.output.strip().splitlines()) == 1


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


def test_gate_batch_refuses_a_nan_interval_in_one_line(repo: Path):
    assert measure(repo, "cand", 60).exit_code == 0
    path = repo / ".axial" / "measurements" / MEASURE_MODE / "cand" / "measurement.json"
    data = json.loads(path.read_text())
    data["interval"] = [0.0, float("nan")]
    path.write_text(json.dumps(data))
    result = axial("gate", MEASURE_MODE, "--batch", "cand", "--max-rate", "0.1")
    assert_clean(result, 1, "does not match its own verdicts")
    assert len(result.output.strip().splitlines()) == 1


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
        (("--batch", "x", "--baseline-batch", "x"), ("--baseline-batch", "the same batch")),
    ],
)
def test_gate_option_mistakes_are_usage_errors(repo: Path, args: tuple[str, ...], fragments: tuple[str, ...]):
    assert_clean(axial("gate", MEASURE_MODE, *args), 2, *fragments)
