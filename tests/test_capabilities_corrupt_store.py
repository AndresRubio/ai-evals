"""Every command against every .axial/ file it reads, damaged each way a file can be.

The error contract: a command that meets a damaged file exits 1, prints one line that
names that file exactly once, and writes nothing. `READS` lists the files each
command reads through the store. `test_reads_names_every_file_each_command_reads` records
the real reads on a working store, so a command that starts reading a new file fails
there until `READS` names it, and the matrix then damages that file too.

Files read outside the store's readers (config.toml, the PROMOTED marker, a file named on
the command line) are not recorded here; their own tests cover them.
"""

from __future__ import annotations

import json
import shutil
import socket
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from measure_helpers import MEASURE_MODE, build_promoted_judge, fake_judge_builder, write_batch
from typer.testing import CliRunner, Result

from axial import cli
from axial.cli import app
from axial.config import write_default_config
from axial.dataset import Label, add_labels
from axial.model import FakeModel
from axial.sample import Review, write_session
from axial.store import Store
from axial.taxonomy import FailureMode, Taxonomy

runner = CliRunner()

M = MEASURE_MODE
OTHER = "other"

COMMANDS: dict[str, list[str]] = {
    "init": ["init"],
    "ingest": ["ingest", "in.jsonl"],
    "sample": ["sample", "--n", "3", "--name", "fresh"],
    "taxonomy": ["taxonomy"],
    "label-import": ["label-import", M, "--path", "import.jsonl"],
    "label-from-reviews": ["label-from-reviews", M, "s"],
    "split": ["split", OTHER],
    "judge": ["judge", M, "--split", "dev"],
    "align": ["align", M, "1"],
    "checks": ["checks", M, "1"],
    "promote": ["promote", M, "1"],
    "measure": ["measure", M, "--version", "1", "--traces", "batch.jsonl", "--name", "h"],
    "gate": ["gate", M, "--version", "1"],
    "gate --batch": ["gate", M, "--batch", "g"],
}

TRACES = ".axial/traces/traces.jsonl"
SESSION = ".axial/reviews/s.jsonl"
TAXONOMY = ".axial/taxonomy.yaml"
LABELS = f".axial/datasets/{M}/labels.jsonl"
SPLITS = f".axial/datasets/{M}/splits.json"
JUDGE = f".axial/judges/{M}/v1/judge.yaml"
VERDICTS = f".axial/judges/{M}/v1/verdicts.jsonl"
ERRORS = f".axial/judges/{M}/v1/errors.jsonl"
# The few-shot judge's examples are resolved from these three.
FEW_SHOT = [TRACES, LABELS, SPLITS]
BATCH = [
    f".axial/measurements/{M}/g/{name}"
    for name in ("errors.jsonl", "measurement.json", "traces.jsonl", "verdicts.jsonl")
]

READS: dict[str, list[str]] = {
    "init": [],
    "ingest": [],
    "sample": [TRACES],
    "taxonomy": [SESSION],
    "label-import": [LABELS],
    "label-from-reviews": [SESSION, LABELS],
    "split": [f".axial/datasets/{OTHER}/labels.jsonl", f".axial/datasets/{OTHER}/splits.json"],
    "judge": [TAXONOMY, JUDGE, VERDICTS, ERRORS, *FEW_SHOT],
    "align": [VERDICTS, ERRORS, LABELS, TRACES],
    "checks": [VERDICTS, TRACES],
    "promote": [JUDGE, VERDICTS, ERRORS, *FEW_SHOT],
    "measure": [JUDGE, VERDICTS, *FEW_SHOT],
    "gate": [JUDGE, VERDICTS, *FEW_SHOT],
    "gate --batch": [JUDGE, VERDICTS, *FEW_SHOT, *BATCH],
}

TAXONOMY_REPLY = json.dumps(
    {"modes": [{"id": M, "name": "Merged", "definition": "the answer merges two questions"}]}
)


def run(argv: list[str]) -> Result:
    return runner.invoke(app, argv)


def _refuse_network(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("network access attempted in an offline capability test")


@pytest.fixture(scope="session")
def template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A store every command in `COMMANDS` runs on cleanly, with a few-shot judge so the
    commands that resolve its examples read the traces, labels and splits too."""
    root = tmp_path_factory.mktemp("template")
    store = Store(root)
    write_default_config(store)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    Taxonomy(
        modes=[FailureMode(id=M, name="Merged", definition="the answer merges two questions")]
    ).save(store)
    reviewed = [f"train-{i}" for i in range(4)] + [f"dev-{i:02d}" for i in range(26)]
    reviews = [
        Review(
            trace_id=trace_id, open_code=f"code {i}", verdict="fail" if i % 2 else "pass",
            critique="read the whole answer", annotator="andres", status="done",
        )
        for i, trace_id in enumerate(reviewed)
    ]
    write_session(store, "s", reviews)
    # `split` redraws the same split from the same labels, so it needs a mode whose split
    # it drew itself; the judge's hand-made split of `merged` would be refused as a redraw.
    add_labels(
        store, OTHER,
        [Label(trace_id=f"t{i}", label=i % 2 == 0, critique="c", annotator="andres") for i in range(60)],
    )
    (root / "in.jsonl").write_text(
        "".join(json.dumps({"trace_id": f"t{i}", "input": "q", "output": "a"}) + "\n" for i in range(3))
    )
    (root / "import.jsonl").write_text(
        json.dumps({"trace_id": "dev-00", "label": True, "critique": "c", "annotator": "andres"}) + "\n"
    )
    write_batch(root / "batch.jsonl", ["BAD"] * 5 + ["GOOD"] * 15)
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(root)
        mp.setattr(cli, "build_judge_model", fake_judge_builder())
        for argv in (
            ["measure", M, "--version", "1", "--traces", "batch.jsonl", "--name", "g"],
            ["split", OTHER],
        ):
            result = run(argv)
            assert result.exit_code == 0, result.output
    return root


@pytest.fixture
def case(template: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    shutil.copytree(template, tmp_path, dirs_exist_ok=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(socket.socket, "connect", _refuse_network)
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "build_judge_model", fake_judge_builder())
    monkeypatch.setattr(cli, "LiteLLMModel", lambda _model: FakeModel(lambda _prompt: TAXONOMY_REPLY))
    return tmp_path


def record_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every path the store's readers open from here on, as written."""
    seen: list[str] = []
    for name in ("_numbered_rows", "read_json", "read_yaml"):
        original = getattr(Store, name)

        def recording(
            self: Store, path: Path, *args: Any, _original: Any = original, **kwargs: Any
        ) -> Any:
            seen.append(Path(path).as_posix())
            return _original(self, path, *args, **kwargs)

        monkeypatch.setattr(Store, name, recording)
    return seen


def test_commands_names_every_command_but_annotate() -> None:
    """annotate starts a web server, so it has no single run to damage a file under."""
    registered = {
        info.name or info.callback.__name__.replace("_", "-")
        for info in app.registered_commands
        if info.callback is not None
    }
    assert registered - {"annotate"} == {argv[0] for argv in COMMANDS.values()}


@pytest.mark.parametrize("command", COMMANDS)
def test_reads_names_every_file_each_command_reads(
    case: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    seen = record_reads(monkeypatch)
    result = run(COMMANDS[command])
    assert result.exit_code == 0, result.output
    reads = sorted({p for p in seen if p.startswith(".axial/")})
    assert reads == sorted(READS[command]), f"READS[{command!r}] should be {reads}"


def appending(data: bytes) -> Callable[[Path], None]:
    return lambda path: path.write_bytes(path.read_bytes() + data)


def overwriting(data: bytes) -> Callable[[Path], None]:
    return lambda path: path.write_bytes(data)


def a_directory(path: Path) -> None:
    path.unlink()
    path.mkdir()


DAMAGE: dict[str, dict[str, Callable[[Path], None]]] = {
    ".jsonl": {
        "not-json": appending(b"{oops\n"),
        "not-an-object": appending(b"5\n"),
        "not-utf-8": appending(b"\xff\xfe\n"),
        "a-directory": a_directory,
        "wrong-fields": appending(b'{"unexpected": 1}\n'),
    },
    ".json": {
        "not-json": overwriting(b"{oops"),
        "not-an-object": overwriting(b"5"),
        "not-utf-8": overwriting(b'{"a": "\xff"}'),
        "a-directory": a_directory,
        "wrong-fields": overwriting(b'{"unexpected": 1}'),
    },
    ".yaml": {
        "not-yaml": appending(b"x: [oops\n"),
        "not-a-mapping": overwriting(b"- 1\n"),
        "not-utf-8": appending(b"\xff\n"),
        "a-directory": a_directory,
        "wrong-fields": overwriting(b"unexpected: 1\n"),
    },
}

CASES = [
    pytest.param(command, path, kind, id=f"{command}:{path.removeprefix('.axial/')}:{kind}")
    for command, paths in READS.items()
    for path in paths
    for kind in DAMAGE[Path(path).suffix]
]


def snapshot(root: Path) -> dict[str, tuple[int, int, int]]:
    return {
        str(p.relative_to(root)): (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.mark.parametrize(("command", "path", "kind"), CASES)
def test_a_damaged_file_is_refused_in_one_line_naming_it_and_nothing_is_written(
    case: Path, command: str, path: str, kind: str
) -> None:
    damaged = case / path
    DAMAGE[damaged.suffix][kind](damaged)
    before = snapshot(case / ".axial")
    result = run(COMMANDS[command])
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        trace = "".join(traceback.format_exception(result.exception))
        pytest.fail(f"uncaught {type(result.exception).__name__}:\n{trace}")
    assert result.exit_code == 1, result.output
    assert len(result.output.strip().splitlines()) == 1, result.output
    assert result.output.count(damaged.name) == 1, result.output
    assert snapshot(case / ".axial") == before
