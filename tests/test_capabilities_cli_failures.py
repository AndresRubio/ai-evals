"""CLI failure UX: missing inputs, stages run out of order, and bad flags.

Each test runs one command the way a user would get it wrong and asserts what a
usable CLI owes them: a non-zero exit, a message that says what is wrong, and no
uncaught exception (under Typer's CliRunner an uncaught exception surfaces as
`result.exception` being something other than SystemExit — in a terminal it is a
traceback). Where the CLI falls short, the test is a strict xfail naming the gap.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from axial.cli import app

runner = CliRunner()


def _refuse_network(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("network access attempted in an offline capability test")


@pytest.fixture(autouse=True)
def _offline_in_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", _refuse_network)
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def run(*args: str) -> Result:
    return runner.invoke(app, list(args))


def assert_clean_failure(result: Result, *fragments: str) -> None:
    """Non-zero exit, a message containing every fragment, and no uncaught exception."""
    assert result.exit_code != 0, f"expected a failure, got exit 0:\n{result.output}"
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"uncaught {type(result.exception).__name__}: {result.exception}"
    )
    for fragment in fragments:
        assert fragment.lower() in result.output.lower(), (fragment, result.output)


def write_traces(tmp_path: Path, n: int = 6) -> Path:
    path = tmp_path / "in.jsonl"
    path.write_text(
        "".join(
            json.dumps({"trace_id": f"t{i}", "input": f"q{i}", "output": f"a{i}"}) + "\n"
            for i in range(n)
        )
    )
    return path


def write_labels(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "labels.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def human_labels(n: int = 12) -> list[dict[str, object]]:
    return [
        {"trace_id": f"t{i}", "label": i % 2 == 0, "critique": "c", "annotator": "andres"}
        for i in range(n)
    ]


# --- bad flags and usage errors (Typer's own validation) -----------------------


def test_unknown_command_exits_2_with_usage() -> None:
    """An unknown subcommand is a usage error naming the command, not a crash."""
    assert_clean_failure(run("frobnicate"), "No such command")


def test_non_integer_flag_is_rejected_with_usage() -> None:
    """`sample --n abc` is rejected by option parsing with the offending value named."""
    result = run("sample", "--n", "abc", "--name", "s")
    assert result.exit_code == 2
    assert_clean_failure(result, "'abc' is not a valid int")


def test_gate_without_version_is_a_usage_error() -> None:
    """`gate` requires `--version`; omitting it is a usage error that names the option."""
    result = run("gate", "m")
    assert result.exit_code == 2
    assert_clean_failure(result, "Missing option '--version'")


def test_label_import_without_path_is_a_usage_error() -> None:
    """`label-import` requires `--path`; omitting it is a usage error that names the option."""
    result = run("label-import", "m")
    assert result.exit_code == 2
    assert_clean_failure(result, "Missing option '--path'")


def test_align_with_non_integer_version_is_a_usage_error() -> None:
    """A version must be an integer, and the parser says so."""
    result = run("align", "m", "one")
    assert result.exit_code == 2
    assert_clean_failure(result, "'one'")


@pytest.mark.xfail(
    strict=True,
    reason="an unknown --strategy reaches sample_batch and escapes as a ValueError "
    "traceback (cli.py:123) instead of a clean exit",
)
def test_sample_with_an_unknown_strategy_fails_cleanly(tmp_path: Path) -> None:
    """`sample --strategy bogus` names the valid strategies without a traceback."""
    run("ingest", str(write_traces(tmp_path)))
    assert_clean_failure(
        run("sample", "--n", "3", "--name", "s", "--strategy", "bogus"), "bogus", "random"
    )


@pytest.mark.xfail(
    strict=True,
    reason="an out-of-range --random-fraction escapes as a ValueError traceback "
    "(cli.py:123 -> sample.py:70)",
)
def test_sample_with_an_out_of_range_random_fraction_fails_cleanly(tmp_path: Path) -> None:
    """`--random-fraction 1.5` is refused with a message, not a traceback."""
    run("ingest", str(write_traces(tmp_path)))
    result = run(
        "sample", "--n", "3", "--name", "s", "--strategy", "longest", "--random-fraction", "1.5"
    )
    assert_clean_failure(result, "random_fraction")


@pytest.mark.xfail(
    strict=True,
    reason="an unknown --perplexity-field escapes as a ValueError traceback "
    "(cli.py:112 -> perplexity.py:211); tests/test_cli.py only checks the exit code",
)
def test_sample_with_an_unknown_perplexity_field_fails_cleanly(tmp_path: Path) -> None:
    """`--perplexity-field nonsense` names the valid fields without a traceback."""
    run("ingest", str(write_traces(tmp_path)))
    result = run(
        "sample", "--n", "3", "--name", "s", "--strategy", "perplexity",
        "--perplexity-field", "nonsense",
    )
    assert_clean_failure(result, "nonsense")


# --- missing or malformed inputs -----------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="ingest of a missing file escapes as FileNotFoundError (cli.py:68); the "
    "`path` argument is not declared exists=True",
)
def test_ingest_of_a_missing_file_fails_cleanly(tmp_path: Path) -> None:
    """Ingesting a path that does not exist says so, without a traceback."""
    assert_clean_failure(run("ingest", str(tmp_path / "nope.jsonl")), "nope.jsonl")


@pytest.mark.xfail(
    strict=True,
    reason="a row missing `output` raises ValueError('row 1: missing required field') "
    "which cli.py:68 does not catch, so the useful message arrives as a traceback",
)
def test_ingest_of_a_row_missing_a_field_fails_cleanly(tmp_path: Path) -> None:
    """A JSONL row without `output` is refused with the row number and the field."""
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"input": "x"}\n')
    assert_clean_failure(run("ingest", str(bad)), "row 1", "output")


@pytest.mark.xfail(
    strict=True,
    reason="a line that is not JSON escapes as json.JSONDecodeError (traces.py:50) "
    "with no file or line context",
)
def test_ingest_of_invalid_json_fails_cleanly(tmp_path: Path) -> None:
    """A line that is not JSON is refused with the line it is on."""
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n")
    assert_clean_failure(run("ingest", str(bad)), "line 1")


@pytest.mark.xfail(
    strict=True,
    reason="label-import of a path that does not exist reads it as empty "
    "(Store.read_jsonl returns []) and reports '0 labels recorded' with exit 0",
)
def test_label_import_of_a_missing_file_fails(tmp_path: Path) -> None:
    """Importing labels from a path that does not exist is an error, not zero labels."""
    assert_clean_failure(
        run("label-import", "m", "--path", str(tmp_path / "nope.jsonl")), "nope.jsonl"
    )


@pytest.mark.xfail(
    strict=True,
    reason="a label row missing `critique`/`annotator` escapes as a pydantic "
    "ValidationError traceback (cli.py:190 only catches AxialError)",
)
def test_label_import_of_an_incomplete_row_fails_cleanly(tmp_path: Path) -> None:
    """A label row missing required fields names the missing field, without a traceback."""
    path = write_labels(tmp_path, [{"trace_id": "t0", "label": True}])
    assert_clean_failure(run("label-import", "m", "--path", str(path)), "critique")


def test_label_import_of_a_machine_annotator_fails_cleanly(tmp_path: Path) -> None:
    """A synthetic-looking annotator is refused with exit 1 and the reason."""
    rows = [{"trace_id": "t0", "label": True, "critique": "c", "annotator": "synthetic-labeler"}]
    path = write_labels(tmp_path, rows)
    assert_clean_failure(run("label-import", "m", "--path", str(path)), "looks like a model")


@pytest.mark.xfail(
    strict=True,
    reason="label-from-reviews with a session name that does not exist reads it as "
    "empty and reports '0 labels recorded' with exit 0 (cli.py:211)",
)
def test_label_from_reviews_of_an_unknown_session_fails() -> None:
    """Converting a review session that does not exist is an error naming the session."""
    assert_clean_failure(run("label-from-reviews", "m", "no-such-session"), "no-such-session")


def test_annotate_without_an_annotator_fails_cleanly(tmp_path: Path) -> None:
    """`annotate` refuses to start until `annotator` is set in config.toml."""
    run("init")
    run("ingest", str(write_traces(tmp_path)))
    run("sample", "--n", "3", "--name", "s")
    assert_clean_failure(run("annotate", "--name", "s"), "annotator", "config.toml")


@pytest.mark.xfail(
    strict=True,
    reason="an invalid split_fractions raises AxialError from load_config, which "
    "`split` calls outside its try block (cli.py:224), so it escapes as a traceback",
)
def test_invalid_split_fractions_in_config_fail_cleanly(tmp_path: Path) -> None:
    """A bad `split_fractions` in config.toml is reported as a config error."""
    run("label-import", "m", "--path", str(write_labels(tmp_path, human_labels())))
    (tmp_path / ".axial" / "config.toml").write_text("split_fractions = [0.5, 0.5]\n")
    assert_clean_failure(run("split", "m"), "split_fractions")


@pytest.mark.xfail(
    strict=True,
    reason="a config.toml that is not valid TOML escapes as tomllib.TOMLDecodeError "
    "(config.py:96) from every command that loads config",
)
def test_malformed_config_toml_fails_cleanly(tmp_path: Path) -> None:
    """An unparseable config.toml is reported as such, naming the file."""
    run("init")
    (tmp_path / ".axial" / "config.toml").write_text("judge_model = = nope\n")
    assert_clean_failure(run("split", "m"), "config.toml")


# --- stages run out of order ---------------------------------------------------


@pytest.mark.xfail(
    strict=False,
    reason="Design question, not a confirmed defect: sample before ingest writes an empty review session and exits 0 "
    "('session s: 0 traces ready to annotate'); nothing says to run ingest first",
)
def test_sample_before_ingest_fails() -> None:
    """Sampling with no traces ingested is an error pointing at `ingest`."""
    assert_clean_failure(run("sample", "--n", "3", "--name", "s"), "ingest")
    assert not Path(".axial/reviews/s.jsonl").exists()


@pytest.mark.xfail(
    strict=True,
    reason="re-using a session name escapes as FileExistsError from create_session "
    "(cli.py:131), a traceback rather than a message",
)
def test_sample_into_an_existing_session_fails_cleanly(tmp_path: Path) -> None:
    """Re-using a review session name is refused with a message, not a traceback."""
    run("ingest", str(write_traces(tmp_path)))
    assert run("sample", "--n", "3", "--name", "s").exit_code == 0
    assert_clean_failure(run("sample", "--n", "3", "--name", "s"), "already exists")


@pytest.mark.xfail(
    strict=True,
    reason="split with no labels exits 0 and saves an empty splits.json; because the "
    "test split is write-once, the next `split` after labelling is then refused",
)
def test_split_before_any_labels_fails_and_does_not_freeze_an_empty_test_split(
    tmp_path: Path,
) -> None:
    """Splitting a mode with no labels is an error, and it leaves nothing frozen behind."""
    result = run("split", "m")
    assert not Path(".axial/datasets/m/splits.json").exists()
    assert_clean_failure(result, "label")


def test_split_run_too_early_locks_out_the_real_split(tmp_path: Path) -> None:
    """Documents the consequence: after an empty split, the real one is refused as a redraw."""
    run("split", "m")
    run("label-import", "m", "--path", str(write_labels(tmp_path, human_labels())))
    result = run("split", "m")
    assert result.exit_code == 1
    assert "write-once" in result.output


@pytest.mark.xfail(
    strict=True,
    reason="judge before split escapes as FileNotFoundError on splits.json "
    "(cli.py:313 Splits.load), after already writing judge.yaml",
)
def test_judge_before_split_fails_cleanly(tmp_path: Path) -> None:
    """Judging a mode that has no split yet points the user at `axial split`."""
    run("init")
    run("ingest", str(write_traces(tmp_path)))
    assert_clean_failure(run("judge", "m", "--split", "dev"), "split")


@pytest.mark.xfail(
    strict=True,
    reason="align with no verdicts escapes as ValueError('cannot align on an empty set') "
    "from align.align (cli.py:340), a traceback",
)
def test_align_before_judge_fails_cleanly() -> None:
    """Aligning a judge that has produced no verdicts says so and points at `judge`."""
    run("init")
    assert_clean_failure(run("align", "m", "1"), "verdict")


@pytest.mark.xfail(
    strict=True,
    reason="promote with no verdicts escapes as ValueError('cannot align on an empty "
    "set') (cli.py:423), a traceback",
)
def test_promote_before_judge_fails_cleanly() -> None:
    """Promoting a judge that has produced no verdicts is refused with a message."""
    run("init")
    assert_clean_failure(run("promote", "m", "1"), "verdict")


def test_checks_before_judge_fails_cleanly() -> None:
    """`checks` on a judge with no verdicts exits 1 naming the missing split."""
    assert_clean_failure(run("checks", "m", "1"), "no test verdicts for m v1")


def test_gate_before_promote_fails_cleanly() -> None:
    """`gate` on an unpromoted judge exits 1 and says to run `axial promote`."""
    assert_clean_failure(run("gate", "m", "--version", "1"), "has not been promoted", "promote")


def test_taxonomy_before_enough_open_codes_fails_cleanly(tmp_path: Path) -> None:
    """`taxonomy` with no annotations refuses before any model call, with the floor named."""
    run("init")
    assert_clean_failure(run("taxonomy"), "0 traces carry an open code", "30")
