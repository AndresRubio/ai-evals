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
from typing import Any, ForwardRef, Literal, get_args, get_origin

import pytest
from measure_helpers import MEASURE_MODE, build_promoted_judge, fake_judge_builder, write_batch
from pydantic import BaseModel, ConfigDict, create_model
from typer.testing import CliRunner, Result

from axial import cli
from axial.cli import app
from axial.config import write_default_config
from axial.dataset import Label, add_labels
from axial.judge import JudgeSpec
from axial.model import FakeModel
from axial.sample import Review, write_session
from axial.store import Store
from axial.taxonomy import FailureMode, Taxonomy

runner = CliRunner()

M = MEASURE_MODE
OTHER = "other"
# A decision-model judge with recorded probabilities, for tune-threshold.
SCORED = "scored"

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
    "tune-threshold": ["tune-threshold", SCORED, "1"],
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
    "align": [JUDGE, VERDICTS, ERRORS, LABELS, TRACES],
    "checks": [VERDICTS, TRACES],
    "promote": [JUDGE, VERDICTS, ERRORS, *FEW_SHOT],
    "measure": [JUDGE, VERDICTS, *FEW_SHOT],
    "gate": [JUDGE, VERDICTS, *FEW_SHOT],
    "gate --batch": [JUDGE, VERDICTS, *FEW_SHOT, *BATCH],
    "tune-threshold": [
        f".axial/judges/{SCORED}/v1/judge.yaml", f".axial/judges/{SCORED}/v1/verdicts.jsonl"
    ],
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
    JudgeSpec(mode=SCORED, model="typesafe/jev-1.13", version=1).save(store)
    store.write_jsonl(
        store.judge_dir(SCORED, 1) / "verdicts.jsonl",
        [
            {"trace_id": f"s{i}", "split": "dev", "verdict": failed, "truth": failed,
             "probability": 0.8 if failed else 0.2}
            for i, failed in enumerate([True, False] * 5)
        ],
    )
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


def record_document_models(monkeypatch: pytest.MonkeyPatch) -> set[type[BaseModel]]:
    """Every model the store's document readers validate against from here on."""
    seen: set[type[BaseModel]] = set()
    for name in ("read_json_model", "read_yaml_model"):
        original = getattr(Store, name)

        def recording(
            self: Store, path: Path, model: type[BaseModel], *args: Any,
            _original: Any = original, **kwargs: Any,
        ) -> Any:
            seen.add(model)
            return _original(self, path, model, *args, **kwargs)

        monkeypatch.setattr(Store, name, recording)
    return seen


def nested_models(annotation: object) -> list[type[BaseModel]]:
    """The models a field's type holds, however deep: `list[M]`, `M | None`, `dict[str, M]`.
    A type pydantic has not resolved hides the models in it, so it fails the walk."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    if get_origin(annotation) is Literal:
        return []  # its arguments are values, not types
    assert not isinstance(annotation, str | ForwardRef), f"unresolved type {annotation!r}"
    return [model for arg in get_args(annotation) for model in nested_models(arg)]


def lenient(
    model: type[BaseModel], where: str = "", inside: frozenset[type[BaseModel]] = frozenset()
) -> list[str]:
    """Where `model`, or a model one of its fields holds, would drop an unknown key. A model
    met again inside itself was checked where it was first met, so the walk stops there."""
    where = where or model.__name__
    found = [] if model.model_config.get("extra") == "forbid" else [where]
    inside = inside | {model}
    for name, field in model.model_fields.items():
        for inner in nested_models(field.annotation):
            if inner not in inside:
                found += lenient(inner, f"{where}.{name}", inside)
    return found


def test_every_model_a_document_is_read_as_refuses_unknown_keys_at_every_level(
    case: Path, template: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """judge.yaml, taxonomy.yaml and splits.json are edited by hand, and a key the model
    ignores reads as the field's default and vanishes on the next save. Pydantic does not
    pass `extra="forbid"` down to the models a field holds, so each one must set it too.
    The models come from every command's real reads, so a new one is checked unlisted."""
    models = record_document_models(monkeypatch)
    for i, (command, argv) in enumerate(COMMANDS.items()):
        root = case / f"run-{i}"
        shutil.copytree(template, root)
        monkeypatch.chdir(root)
        result = run(argv)
        assert result.exit_code == 0, f"{command}: {result.output}"
    names = {model.__name__ for model in models}
    assert {"JudgeSpec", "Splits", "_TaxonomyFile"} <= names, names
    found = [where for model in models for where in lenient(model)]
    assert not found, f"give these model_config = ConfigDict(extra='forbid'): {sorted(found)}"


class _Loose(BaseModel):
    """No `extra` setting, so an unknown key is dropped."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


STRICT = ConfigDict(extra="forbid")


@pytest.mark.parametrize(
    "shape",
    [
        lambda m: m,
        lambda m: list[m],
        lambda m: m | None,
        lambda m: dict[str, m],
        lambda m: list[dict[str, m | None]],
    ],
    ids=["bare", "list", "optional", "dict", "nested"],
)
def test_lenient_finds_a_model_however_a_field_holds_it(
    shape: Callable[[type[BaseModel]], object],
) -> None:
    """The test of every document model checks only the models this walk opens, so a shape
    it skipped would hide a lenient model and pass."""

    def holder(inner: type[BaseModel]) -> type[BaseModel]:
        return create_model("Holder", __config__=STRICT, inner=(shape(inner), None))

    assert lenient(holder(_Loose)) == ["Holder.inner"]
    assert lenient(holder(_Strict)) == []


class _Tree(BaseModel):
    model_config = ConfigDict(extra="forbid")
    children: list[_Tree] = []


class _LooseTree(BaseModel):
    children: list[_LooseTree] = []


def test_lenient_walks_a_model_that_holds_itself_once() -> None:
    assert lenient(_Tree) == []
    assert lenient(_LooseTree) == ["_LooseTree"]


def test_lenient_fails_on_a_type_pydantic_has_not_resolved() -> None:
    """Such a type holds no model the walk can see, so a lenient one inside would pass."""
    holder = create_model("Holder", __config__=STRICT, inner=("Missing", None))
    with pytest.raises(AssertionError, match="unresolved type"):
        lenient(holder)


def test_lenient_reads_a_literals_strings_as_values() -> None:
    holder = create_model("Holder", __config__=STRICT, kind=(Literal["a", "b"], "a"))
    assert lenient(holder) == []


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
        "an-unknown-key": appending(b"unexpected: 1\n"),
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
