"""Capability tour of the axial CLI, plus one test per concrete README claim.

The first test is an executable tour: every stage of the README pipeline is run
through the Typer app (the object `axial = "axial.cli:app"` installs as the console
script), from the repo directory, with no `--repo` flag, exactly as the Quickstart
types it. The only substitutions are the model seams: `build_judge_model` and
`LiteLLMModel` are swapped for the deterministic `FakeModel`, and `uvicorn.run` is
captured so the real annotate app can be driven in-process with a TestClient.

Every other test takes one sentence from README.md (quoted in its docstring) and
checks it. A claim the code does not honour is kept as a strict xfail with the reason.

Nothing here reaches the network: an autouse fixture makes any socket connect fail,
and model credentials are removed from the environment for every test.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from typer.testing import CliRunner, Result

import axial.judge as judge_module
from axial import cli
from axial.cli import app
from axial.model import FakeModel

REPO_ROOT = Path(__file__).resolve().parent.parent
README = (REPO_ROOT / "README.md").read_text()
CORPUS = REPO_ROOT / "examples" / "newsroom" / "traces.jsonl"
MODE = "merged_stories"
MODEL_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")
JEV = "typesafe/jev-1.13"

runner = CliRunner()


# --- harness -------------------------------------------------------------------


def _refuse_network(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("network access attempted in an offline capability test")


def _clear_model_env(mp: pytest.MonkeyPatch) -> None:
    # setenv-then-delenv records the original state, so undo restores it even when
    # `load_env` later writes one of these keys straight into os.environ.
    for name in MODEL_ENV_VARS:
        mp.setenv(name, "placeholder")
        mp.delenv(name)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", _refuse_network)
    monkeypatch.setattr(socket, "create_connection", _refuse_network)
    _clear_model_env(monkeypatch)


def axial(*args: str) -> Result:
    """Run one CLI command in the current directory and demand it succeed."""
    result = runner.invoke(app, list(args))
    assert result.exit_code == 0, (
        f"`axial {' '.join(args)}` exited {result.exit_code}:\n{result.output}\n"
        f"{result.exception!r}"
    )
    return result


def axial_raw(*args: str) -> Result:
    return runner.invoke(app, list(args))


def planted_modes() -> dict[str, str]:
    rows = [json.loads(line) for line in CORPUS.read_text().splitlines() if line.strip()]
    return {row["trace_id"]: row["planted_mode"] for row in rows}


def enable_annotator(repo: Path, name: str = "andres") -> None:
    """The README step: uncomment `annotator = ...` near the top of config.toml."""
    config = repo / ".axial" / "config.toml"
    text = config.read_text()
    assert '# annotator = "your-name"' in text
    config.write_text(text.replace('# annotator = "your-name"', f'annotator = "{name}"', 1))


def set_config(repo: Path, key: str, value: str) -> None:
    config = repo / ".axial" / "config.toml"
    config.write_text(
        re.sub(
            rf"^{key} = .*$", f"{key} = {value}", config.read_text(), count=1, flags=re.MULTILINE
        )
    )


def open_annotate_app(session: str) -> TestClient:
    """Run `axial annotate` and hand back the app it would have served."""
    captured: dict[str, object] = {}

    def fake_run(asgi_app: object, host: str, port: int) -> None:
        captured.update(app=asgi_app, host=host, port=port)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("uvicorn.run", fake_run)
        axial("annotate", "--name", session)
    assert captured["host"] == "127.0.0.1"
    served = captured["app"]
    assert isinstance(served, FastAPI)
    return TestClient(served)


def annotate_via_app(session: str, verdict_for: Callable[[str], str]) -> int:
    """Annotate every item in `session` through the annotate app's own HTTP API."""
    client = open_annotate_app(session)
    items = client.get("/api/session").json()["items"]
    for item in items:
        planted = item["metadata"]["planted_mode"]
        body = {
            "open_code": "looks right" if planted == "clean" else planted.replace("_", " "),
            "verdict": verdict_for(planted),
            "critique": f"planted {planted}",
        }
        client.post(f"/api/review/{item['trace_id']}", json=body).raise_for_status()
    return len(items)


def annotate_directly(repo: Path, session: str, verdict_for: Callable[[str], str]) -> None:
    """The same result as `annotate_via_app`, written straight to the session file."""
    from axial.sample import read_session, write_session
    from axial.store import Store

    store = Store(repo)
    planted = planted_modes()
    reviews = read_session(store, session)
    for review in reviews:
        mode = planted[review.trace_id]
        review.open_code = "looks right" if mode == "clean" else mode.replace("_", " ")
        review.verdict = verdict_for(mode)
        review.critique = f"planted {mode}"
        review.annotator = "andres"
        review.status = "done"
    write_session(store, session, reviews)


def merged_verdict(planted: str) -> str:
    return "fail" if planted == MODE else "pass"


def graded_output(prompt: str) -> str:
    return prompt.rsplit("OUTPUT:\n", 1)[1]


def graded_issue(prompt: str) -> int:
    match = re.search(r"issue (\d+)", prompt.rsplit("INPUT:\n", 1)[1])
    assert match is not None
    return int(match.group(1))


def perfect_judge(prompt: str) -> str:
    """Reads the planted mode out of the graded OUTPUT: a merged story has one headline."""
    merged = graded_output(prompt).count('"headline"') == 1
    return json.dumps({"verdict": "fail" if merged else "pass", "reason": "headline count"})


def always_pass_judge(_prompt: str) -> str:
    return json.dumps({"verdict": "pass", "reason": "looks fine"})


def regressed_judge(prompt: str) -> str:
    """The perfect judge, plus a false alarm on every 20th issue (all clean traces)."""
    if graded_issue(prompt) % 20 == 0:
        return json.dumps({"verdict": "fail", "reason": "false alarm"})
    return perfect_judge(prompt)


def use_fake_judge(mp: pytest.MonkeyPatch, judge_fn: Callable[[str], str]) -> list[str]:
    """Swap the judge's model for a FakeModel; return the list its prompts land in."""
    prompts: list[str] = []

    def build(spec: judge_module.JudgeSpec) -> FakeModel:
        model = FakeModel(judge_fn, model=spec.model)
        model.calls = prompts
        return model

    mp.setattr(cli, "build_judge_model", build)
    return prompts


def build_pipeline(
    repo: Path,
    mp: pytest.MonkeyPatch,
    *,
    judge_fn: Callable[[str], str] = perfect_judge,
    promote: bool = True,
) -> None:
    """init → ingest → sample → (annotate) → label → split → judge → align [→ promote]."""
    repo.mkdir(parents=True, exist_ok=True)
    mp.chdir(repo)
    axial("init")
    enable_annotator(repo)
    axial("ingest", str(CORPUS))
    axial("sample", "--n", "400", "--name", "labels")
    annotate_directly(repo, "labels", merged_verdict)
    axial("label-from-reviews", MODE, "labels")
    axial("split", MODE)
    use_fake_judge(mp, judge_fn)
    axial("judge", MODE, "--split", "dev")
    axial("judge", MODE, "--split", "test")
    axial("align", MODE, "1")
    if promote:
        axial("promote", MODE, "1")


@pytest.fixture(scope="module")
def promoted_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A repo with a perfect judge for merged_stories promoted at v1, built once."""
    repo = tmp_path_factory.mktemp("promoted") / "repo"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(socket.socket, "connect", _refuse_network)
        _clear_model_env(mp)
        build_pipeline(repo, mp)
    return repo


@pytest.fixture
def promoted_repo(promoted_template: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    shutil.copytree(promoted_template, repo)
    monkeypatch.chdir(repo)
    return repo


def judge_dir(repo: Path, version: int = 1, mode: str = MODE) -> Path:
    return repo / ".axial" / "judges" / mode / f"v{version}"


def number_after(prefix: str, text: str) -> float:
    match = re.search(re.escape(prefix) + r"([\d.]+)", text)
    assert match is not None, f"{prefix!r} not found in {text!r}"
    return float(match.group(1))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- 1. the tour ---------------------------------------------------------------


TAXONOMY_REPLY = json.dumps(
    {
        "modes": [
            {
                "id": MODE,
                "name": "merged stories",
                "definition": "Two distinct stories emitted as one.",
                "examples": [],
            }
        ]
    }
)


class FakeTextModel:
    """Stands in for `LiteLLMModel` inside `axial taxonomy`."""

    instances: ClassVar[list[FakeTextModel]] = []

    def __init__(self, model: str) -> None:
        self.model = model
        self.prompts: list[str] = []
        self.api_key_seen = os.environ.get("ANTHROPIC_API_KEY")
        FakeTextModel.instances.append(self)

    def complete(self, prompt: str, *, max_tokens: int = 1024, schema: object = None) -> object:
        self.prompts.append(prompt)
        return FakeModel(lambda _p: TAXONOMY_REPLY, model=self.model).complete(prompt)


def test_capability_tour_runs_the_whole_pipeline_through_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every README stage runs from the CLI, lands in .axial/ as plain text, and feeds the next."""
    repo = tmp_path / "repo"
    repo.mkdir()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    monkeypatch.chdir(repo)
    axial_dir = repo / ".axial"
    planted = planted_modes()

    # init: a config with the documented defaults and the commented annotator line.
    axial("init")
    config = tomllib.loads((axial_dir / "config.toml").read_text())
    assert config["judge_model"] == "claude-opus-5"
    assert "annotator" not in config
    enable_annotator(repo)

    # ingest: the committed corpus becomes normalised traces.
    out = axial("ingest", str(CORPUS)).output
    assert "ingested 400 traces" in out
    traces = read_jsonl(axial_dir / "traces" / "traces.jsonl")
    assert len(traces) == 400
    assert set(traces[0]) == {"trace_id", "input", "output", "metadata"}

    # sample: a first-pass review batch for open coding.
    out = axial("sample", "--n", "40", "--name", "first-pass").output
    assert "session first-pass: 40 traces ready to annotate" in out
    first_pass = read_jsonl(axial_dir / "reviews" / "first-pass.jsonl")
    assert len(first_pass) == 40 and {r["status"] for r in first_pass} == {"pending"}
    assert {r["trace_id"] for r in first_pass} <= set(planted)

    # annotate: open codes, verdicts and critiques through the real annotate app.
    assert annotate_via_app("first-pass", lambda m: "pass" if m == "clean" else "fail") == 40
    first_pass = read_jsonl(axial_dir / "reviews" / "first-pass.jsonl")
    assert {r["status"] for r in first_pass} == {"done"}
    assert {r["annotator"] for r in first_pass} == {"andres"}
    assert all(r["open_code"] and r["critique"] for r in first_pass)

    # taxonomy: the open codes go to the configured judge model; a taxonomy comes back.
    FakeTextModel.instances.clear()
    monkeypatch.setattr(cli, "LiteLLMModel", FakeTextModel)
    out = axial("taxonomy").output
    (proposer,) = FakeTextModel.instances
    assert proposer.model == "claude-opus-5"
    assert all(f"[{r['trace_id']}] {r['open_code']}" in proposer.prompts[0] for r in first_pass)
    taxonomy = yaml.safe_load((axial_dir / "taxonomy.yaml").read_text())
    assert [m["id"] for m in taxonomy["modes"]] == [MODE]
    assert taxonomy["modes"][0]["prevalence"] > 0
    assert f"{MODE}: merged stories" in out

    # A second session, labelled for the one chosen mode, across the whole corpus.
    axial("sample", "--n", "400", "--name", "labels")
    annotate_via_app("labels", merged_verdict)

    # label-from-reviews: completed reviews become human labels.
    out = axial("label-from-reviews", MODE, "labels").output
    assert f"400 labels recorded for {MODE}; 0 pending review(s) skipped" in out
    labels = read_jsonl(axial_dir / "datasets" / MODE / "labels.jsonl")
    assert len(labels) == 400
    assert sum(1 for lbl in labels if lbl["label"]) == 80
    assert {lbl["annotator"] for lbl in labels} == {"andres"}

    # label-import: labels made elsewhere land in the same place, in the same shape.
    elsewhere = inputs / "wrong_topic_labels.jsonl"
    elsewhere.write_text(
        "".join(
            json.dumps(
                {"trace_id": tid, "label": mode == "wrong_topic",
                 "critique": f"planted {mode}", "annotator": "maria"}
            ) + "\n"
            for tid, mode in planted.items()
        )
    )
    out = axial("label-import", "wrong_topic", "--path", str(elsewhere)).output
    assert "400 labels recorded for wrong_topic" in out
    imported = read_jsonl(axial_dir / "datasets" / "wrong_topic" / "labels.jsonl")
    assert set(imported[0]) == set(labels[0])

    # split: stratified, seeded, with a frozen test manifest.
    out = axial("split", MODE).output
    balance = json.loads(out)
    splits = json.loads((axial_dir / "datasets" / MODE / "splits.json").read_text())
    assert splits["test_hash"]
    assert sum(b["fail"] for b in balance.values()) == 80
    assert all(b["fail"] > 0 and b["pass"] > 0 for b in balance.values())
    train_failure = next(t for t in splits["train"] if planted[t] == MODE)

    # judge: dev first, with a train-only few-shot example.
    prompts = use_fake_judge(monkeypatch, perfect_judge)
    out = axial("judge", MODE, "--split", "dev", "--few-shot", train_failure).output
    assert f"{len(splits['dev'])} verdicts, 0 quarantined" in out
    judge_yaml = judge_dir(repo) / "judge.yaml"
    spec = yaml.safe_load(judge_yaml.read_text())
    assert spec["few_shot_ids"] == [train_failure]
    assert spec["mode_name"] == "merged stories"  # carried over from taxonomy.yaml
    assert spec["mode_definition"] == "Two distinct stories emitted as one."
    assert all("merged stories" in p and "EXAMPLE" in p for p in prompts)

    # Iterate on judge.yaml, as the Quickstart says, and re-run dev: the edit is used.
    spec["prompt_template"] = "HOUSE RULES v2\n" + spec["prompt_template"]
    judge_yaml.write_text(yaml.safe_dump(spec, sort_keys=False))
    prompts.clear()
    axial("judge", MODE, "--split", "dev")
    assert prompts and all(p.startswith("HOUSE RULES v2") for p in prompts)

    # judge: then test, on the same version.
    axial("judge", MODE, "--split", "test")
    assert not judge_dir(repo, 2).exists()
    verdicts = read_jsonl(judge_dir(repo) / "verdicts.jsonl")
    assert {v["split"] for v in verdicts} == {"dev", "test"}
    assert len(verdicts) == len(splits["dev"]) + len(splits["test"])
    assert read_jsonl(judge_dir(repo) / "errors.jsonl") == []

    # align: TPR/TNR with intervals, written to alignment.md and alignment.json.
    out = axial("align", MODE, "1").output
    assert re.search(r"test\s+n=\d+\s+TPR=1\.000 \[[\d.]+, 1\.000\]\s+TNR=1\.000", out)
    alignment = json.loads((judge_dir(repo) / "alignment.json").read_text())
    assert alignment["test"]["tpr"] == 1.0 and alignment["test"]["tnr"] == 1.0
    assert "| test |" in (judge_dir(repo) / "alignment.md").read_text()

    # checks: noise floor, verbosity bias, cost from recorded usage, slice breakdown.
    out = axial("checks", MODE, "1", "--slice-by", "planted_mode").output
    assert f"noise floor at n={len(splits['test'])}" in out
    assert "verbosity bias" in out
    assert re.search(r"^cost: \$\d+\.\d{4}$", out, re.MULTILINE)  # priced, no unknown rows
    assert "planted_mode=merged_stories: n=" in out
    assert "planted_mode=clean: n=" in out

    # promote: a PROMOTED marker bound to the judge and verdicts on disk.
    out = axial("promote", MODE, "1").output
    assert f"{MODE} v1 promoted" in out
    marker = (judge_dir(repo) / "PROMOTED").read_text()
    assert "judge_yaml_sha256=" in marker and "verdicts_sha256=" in marker

    # gate: an exit code.
    out = axial("gate", MODE, "--version", "1").output
    assert f"{MODE} v1: failure rate" in out

    # Everything the pipeline wrote is under .axial/, and all of it is plain text.
    assert sorted(os.listdir(repo)) == [".axial"]
    parsers: dict[str, Callable[[str], object]] = {
        ".jsonl": lambda s: [json.loads(line) for line in s.splitlines() if line.strip()],
        ".json": json.loads,
        ".yaml": yaml.safe_load,
        ".toml": tomllib.loads,
        ".md": str,
        "": str,  # PROMOTED
    }
    written = [p for p in axial_dir.rglob("*") if p.is_file()]
    assert len(written) >= 12
    for path in written:
        parsers[path.suffix](path.read_bytes().decode("utf-8"))


# --- 2. README claims ----------------------------------------------------------


def test_every_readme_command_is_on_the_installed_console_script() -> None:
    """The installed `axial` script exposes every command the README pipeline table names.

    README: "Every command below runs as `uv run axial ...` from that clone."
    """
    commands = re.findall(r"^\| `axial ([a-z-]+)`", README, re.MULTILINE)
    assert len(commands) == 13
    script = Path(sys.executable).parent / "axial"
    assert script.exists(), f"console script missing at {script}"
    env = {**os.environ, "COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"}
    help_text = subprocess.run(
        [str(script), "--help"], capture_output=True, text=True, env=env, check=True, timeout=60
    ).stdout
    for command in ["init", *commands]:
        assert re.search(rf"\b{re.escape(command)}\b", help_text), command


def test_offline_commands_call_no_model_and_need_no_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """init/ingest/sample/annotate/label-from-reviews/split/align/promote/gate run keyless.

    README: "Most of the pipeline — `init`, `ingest`, `sample`, `annotate`,
    `label-from-reviews`, `split`, `align`, `promote`, `gate` — calls no model at all
    and needs no key."
    """

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a model was constructed by a command documented as model-free")

    real_fake_init = FakeModel.__init__
    monkeypatch.setattr("axial.model.LiteLLMModel.__init__", forbidden)
    monkeypatch.setattr("axial.model.JevModel.__init__", forbidden)
    monkeypatch.setattr("axial.model.FakeModel.__init__", forbidden)
    monkeypatch.setattr(cli, "build_judge_model", forbidden)
    assert not any(name in os.environ for name in MODEL_ENV_VARS)

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    axial("init")
    enable_annotator(repo)
    axial("ingest", str(CORPUS))
    axial("sample", "--n", "400", "--name", "labels")
    annotate_via_app("labels", merged_verdict)
    axial("label-from-reviews", MODE, "labels")
    axial("split", MODE)

    # `judge` is the documented exception: it alone gets a (fake) model, briefly.
    with pytest.MonkeyPatch.context() as judge_only:
        judge_only.setattr("axial.model.FakeModel.__init__", real_fake_init)
        use_fake_judge(judge_only, perfect_judge)
        axial("judge", MODE, "--split", "dev")
        axial("judge", MODE, "--split", "test")

    axial("align", MODE, "1")
    axial("promote", MODE, "1")
    axial("gate", MODE, "--version", "1")
    assert not any(name in os.environ for name in MODEL_ENV_VARS)


def _tiny_labelled_repo(repo: Path, *, judge_model: str | None = None) -> None:
    """12 traces, 6 failing, labelled and split — enough to run `judge` on dev."""
    repo.mkdir(parents=True, exist_ok=True)
    os.chdir(repo)
    axial("init")
    if judge_model is not None:
        set_config(repo, "judge_model", json.dumps(judge_model))
    traces = repo.parent / "tiny.jsonl"
    traces.write_text(
        "".join(
            json.dumps({"trace_id": f"t{i}", "input": f"question {i}", "output": f"answer {i}"})
            + "\n"
            for i in range(12)
        )
    )
    axial("ingest", str(traces))
    labels = repo.parent / "tiny_labels.jsonl"
    labels.write_text(
        "".join(
            json.dumps({"trace_id": f"t{i}", "label": i % 2 == 0, "critique": "c",
                        "annotator": "andres"}) + "\n"
            for i in range(12)
        )
    )
    axial("label-import", "m", "--path", str(labels))
    with pytest.warns(UserWarning, match="below the usable floor"):
        axial("split", "m")


def _stub_decisions_api(
    monkeypatch: pytest.MonkeyPatch, noul: float = 0.6
) -> list[dict[str, object]]:
    """Replace the httpx module Jev imports with one that answers every decision locally."""
    calls: list[dict[str, object]] = []

    def post(url: str, *, json: dict[str, object], headers: dict[str, str], timeout: float):
        calls.append({"url": url, "body": json, "headers": headers})
        payload = {"model": JEV, "answers": {"verdict": {"type": "noul", "noul": noul}}}
        return SimpleNamespace(json=lambda: payload, raise_for_status=lambda: None)

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(post=post))
    return calls


def test_judge_reads_the_key_from_dotenv_in_the_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`axial judge` picks a key up from `<repo>/.env` when the shell exports none.

    README: "`.env` is gitignored, read from the repo root by `axial taxonomy`, `axial judge`
    and `axial measure`"
    """
    repo = tmp_path / "repo"
    monkeypatch.chdir(tmp_path)
    _tiny_labelled_repo(repo, judge_model=JEV)
    (repo / ".env").write_text("# credentials\nexport OPENROUTER_API_KEY='from-dotenv'\n")
    calls = _stub_decisions_api(monkeypatch)

    axial("judge", "m", "--split", "dev")

    assert calls, "the stubbed Decisions API was never called"
    assert {c["headers"]["Authorization"] for c in calls} == {"Bearer from-dotenv"}


def test_dotenv_never_overrides_a_variable_exported_in_the_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exported key beats the one in `.env`, all the way to the request `judge` sends.

    README: "`.env` ... never overrides a variable already exported in your shell."
    """
    repo = tmp_path / "repo"
    monkeypatch.chdir(tmp_path)
    _tiny_labelled_repo(repo, judge_model=JEV)
    (repo / ".env").write_text("OPENROUTER_API_KEY=from-dotenv\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-shell")
    calls = _stub_decisions_api(monkeypatch)

    axial("judge", "m", "--split", "dev")

    assert {c["headers"]["Authorization"] for c in calls} == {"Bearer from-shell"}
    assert os.environ["OPENROUTER_API_KEY"] == "from-shell"


def test_taxonomy_reads_dotenv_and_sends_open_codes_to_the_configured_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`axial taxonomy` loads `.env` and calls the judge model from config.toml.

    README: "it sends your open codes to the judge model configured in
    `.axial/config.toml` (`claude-opus-5` by default)"
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    axial("init")
    axial("ingest", str(CORPUS))
    axial("sample", "--n", "30", "--name", "codes")
    annotate_directly(repo, "codes", merged_verdict)
    (repo / ".env").write_text("ANTHROPIC_API_KEY=from-dotenv\n")
    FakeTextModel.instances.clear()
    monkeypatch.setattr(cli, "LiteLLMModel", FakeTextModel)

    axial("taxonomy")

    (proposer,) = FakeTextModel.instances
    assert proposer.model == "claude-opus-5"
    assert proposer.api_key_seen == "from-dotenv"
    assert len(re.findall(r"^- \[", proposer.prompts[0], re.MULTILINE)) == 30  # every coded trace


def test_targeted_sampling_mixes_in_a_30_percent_random_slice_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A targeted batch keeps 30% random by default and honours `--random-fraction`.

    README: "targeted strategies mix in a random slice (30% by default, `--random-fraction`)"
    """
    monkeypatch.chdir(tmp_path)
    corpus = tmp_path / "lengths.jsonl"
    corpus.write_text(
        "".join(
            json.dumps({"trace_id": f"t{i:02d}", "input": f"q{i}", "output": "x" * i}) + "\n"
            for i in range(1, 31)
        )
    )
    axial("ingest", str(corpus))
    longest_first = [f"t{i:02d}" for i in range(30, 0, -1)]

    def batch(name: str, *extra: str) -> list[str]:
        axial("sample", "--n", "10", "--name", name, "--strategy", "longest", *extra)
        session = tmp_path / ".axial" / "reviews" / f"{name}.jsonl"
        return [str(r["trace_id"]) for r in read_jsonl(session)]

    default = batch("default")
    assert default[:7] == longest_first[:7]  # 70% targeted, in rank order
    assert not set(default[7:]) & set(longest_first[:10])  # 30% drawn from the rest
    assert len(default) == 10

    half = batch("half", "--random-fraction", "0.5")
    assert half[:5] == longest_first[:5]
    assert not set(half[5:]) & set(longest_first[:10])


def test_judge_yaml_is_the_whole_portable_judge_definition(promoted_repo: Path) -> None:
    """judge.yaml carries the prompt template, model, few-shot ids and a 0.5 decision threshold.

    README: "`axial` produces a validated judge definition (`judge.yaml`: prompt template,
    model, few-shot trace ids, decision threshold) and the alignment report behind it."
    README: "Tune the threshold in `judge.yaml` (`decision_threshold`, default `0.5`)"
    """
    spec = yaml.safe_load((judge_dir(promoted_repo) / "judge.yaml").read_text())
    assert {"prompt_template", "model", "few_shot_ids", "decision_threshold"} <= set(spec)
    assert spec["model"] == "claude-opus-5"
    assert spec["decision_threshold"] == 0.5
    assert "{input}" in spec["prompt_template"] and "{output}" in spec["prompt_template"]
    assert (judge_dir(promoted_repo) / "alignment.md").exists()
    assert (judge_dir(promoted_repo) / "alignment.json").exists()


def test_there_is_no_exporter_to_other_eval_tools() -> None:
    """No CLI command or source module exports judge.yaml to another eval framework.

    README: "There is no exporter to any of these tools; to run the judge inside one, you
    port the prompt and threshold by hand"
    """
    names = {
        str(cmd.name or getattr(cmd.callback, "__name__", "")) for cmd in app.registered_commands
    }
    assert not {n for n in names if "export" in n}
    source = "\n".join(p.read_text() for p in (REPO_ROOT / "src" / "axial").rglob("*.py"))
    for tool in ("promptfoo", "deepeval", "ragas", "inspect_ai", "autoevals", "pydantic_evals"):
        assert tool not in source.lower(), tool


def test_decision_threshold_in_judge_yaml_decides_jev_verdicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Jev verdicts threshold the probability at judge.yaml's `decision_threshold` (0.5).

    README: "thresholds the calibrated probability into the same verdict JSON an LLM judge
    emits ... Tune the threshold in `judge.yaml` (`decision_threshold`, default `0.5`)"
    README: "the recorded reason is the probability and the threshold"
    """
    repo = tmp_path / "repo"
    monkeypatch.chdir(tmp_path)
    _tiny_labelled_repo(repo, judge_model=JEV)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-dummy")
    _stub_decisions_api(monkeypatch, noul=0.6)

    axial("judge", "m", "--split", "dev")
    judge_yaml = judge_dir(repo, mode="m") / "judge.yaml"
    spec = yaml.safe_load(judge_yaml.read_text())
    assert spec["model"] == JEV and spec["decision_threshold"] == 0.5
    rows = read_jsonl(judge_dir(repo, mode="m") / "verdicts.jsonl")
    assert rows and all(r["verdict"] is True for r in rows)
    assert all("noul=0.600 threshold=0.5" in str(r["raw"]) for r in rows)

    spec["decision_threshold"] = 0.9
    judge_yaml.write_text(yaml.safe_dump(spec, sort_keys=False))
    axial("judge", "m", "--split", "dev")
    rows = read_jsonl(judge_dir(repo, mode="m") / "verdicts.jsonl")
    assert rows and all(r["verdict"] is False for r in rows)
    assert all("threshold=0.9" in str(r["raw"]) for r in rows)


def test_retuning_the_threshold_after_promotion_blocks_the_gate(promoted_repo: Path) -> None:
    """Editing judge.yaml (e.g. decision_threshold) after promote makes `gate` exit 1.

    README: "it changes every verdict, which is why it lives in the hashed judge definition."
    """
    judge_yaml = judge_dir(promoted_repo) / "judge.yaml"
    spec = yaml.safe_load(judge_yaml.read_text())
    spec["decision_threshold"] = 0.7
    judge_yaml.write_text(yaml.safe_dump(spec, sort_keys=False))

    result = axial_raw("gate", MODE, "--version", "1")

    assert result.exit_code == 1
    assert "judge.yaml has changed since it was promoted" in result.output


def test_rejudging_a_promoted_version_withdraws_its_promotion(
    promoted_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running `judge` on a promoted version deletes PROMOTED, so `gate` refuses it.

    README: "`axial promote` | alignment on test | a judge cleared for CI"
    """
    use_fake_judge(monkeypatch, perfect_judge)
    axial("judge", MODE, "--split", "dev")

    assert not (judge_dir(promoted_repo) / "PROMOTED").exists()
    result = axial_raw("gate", MODE, "--version", "1")
    assert result.exit_code == 1
    assert "has not been promoted" in result.output


def test_gate_without_a_baseline_only_prints_the_failure_rate_and_passes(
    promoted_repo: Path,
) -> None:
    """With no `--baseline`, `gate` exits 0 whatever the failure rate, and compares nothing.

    README: "without a baseline or a ceiling, either mode only prints and passes"
    """
    result = axial("gate", MODE, "--version", "1")
    rate = number_after("failure rate ", result.output)
    assert rate > 0.15  # a fifth of the judged traces fail, and it still passes
    assert "delta" not in result.output and "REGRESSION" not in result.output


def test_gate_exit_codes_across_promoted_versions(
    promoted_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`gate` exits 1 on a CI-backed regression or an unpromoted judge, 0 otherwise.

    README: "`gate --version` compares a judge's own verdicts across judge versions with
    `--baseline`"
    """
    use_fake_judge(monkeypatch, regressed_judge)
    axial("judge", MODE, "--split", "dev", "--new")
    axial("judge", MODE, "--split", "test")
    axial("align", MODE, "2")
    axial("promote", MODE, "2")

    regression = axial_raw("gate", MODE, "--version", "2", "--baseline", "1")
    assert regression.exit_code == 1, regression.output
    assert "95% CI" in regression.output and "REGRESSION" in regression.output

    improvement = axial_raw("gate", MODE, "--version", "1", "--baseline", "2")
    assert improvement.exit_code == 0, improvement.output
    assert "improvement" in improvement.output

    unchanged = axial_raw("gate", MODE, "--version", "1", "--baseline", "1")
    assert unchanged.exit_code == 0
    assert "no change detectable" in unchanged.output

    unpromoted = axial_raw("gate", MODE, "--version", "3")
    assert unpromoted.exit_code == 1
    assert "has not been promoted" in unpromoted.output


def test_a_promoted_judge_can_grade_new_traces_for_the_gate(
    promoted_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A promoted judge scores fresh traces with `measure`, and `gate --batch` passes on them.

    README: "`axial measure` | a promoted judge + a batch of new traces | the batch's
    failure rate, corrected for the judge's errors, with a 95% interval" /
    "`axial gate` | promoted judges, or measured batches | an exit code"
    """
    use_fake_judge(monkeypatch, perfect_judge)
    fresh = promoted_repo / "fresh.jsonl"
    fresh.write_text(
        "".join(
            json.dumps({
                "trace_id": f"new-{i}",
                "input": f"THE WIRE — issue {1000 + i}\n\nA story.",
                "output": '{"stories": [{"headline": "a"}, {"headline": "b"}]}',
            }) + "\n"
            for i in range(40)
        )
    )
    promoted_before = (judge_dir(promoted_repo) / "PROMOTED").read_bytes()

    measured = axial_raw("measure", MODE, "--version", "1", "--traces", str(fresh), "--name", "fresh")
    assert measured.exit_code == 0, measured.output
    assert "n=40 graded, 0 set aside" in measured.output
    assert (judge_dir(promoted_repo) / "PROMOTED").read_bytes() == promoted_before

    gate = axial_raw("gate", MODE, "--batch", "fresh")
    assert gate.exit_code == 0 and "saturation warning" in gate.output


def test_split_is_stratified_seeded_and_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same labels and seed give the same split; each class is split in proportion.

    README: "`axial split` | labels | a stratified, seeded, frozen split"
    """
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "".join(
            json.dumps({"trace_id": f"t{i:03d}", "label": i % 5 == 0, "critique": "c",
                        "annotator": "andres"}) + "\n"
            for i in range(200)
        )
    )

    def split_in(name: str, seed: str) -> dict[str, list[str]]:
        repo = tmp_path / name
        repo.mkdir()
        monkeypatch.chdir(repo)
        axial("label-import", "m", "--path", str(labels))
        axial("split", "m", "--seed", seed)
        return json.loads((repo / ".axial" / "datasets" / "m" / "splits.json").read_text())

    a, b, c = split_in("a", "0"), split_in("b", "0"), split_in("c", "1")
    assert a == b
    assert a["test"] != c["test"]
    for name, frac in (("train", 0.15), ("dev", 0.42), ("test", 0.43)):
        fails = sum(1 for t in a[name] if int(t[1:]) % 5 == 0)
        assert abs(fails - 40 * frac) <= 1, (name, fails)
        assert abs(len(a[name]) - fails - 160 * frac) <= 1
    assert a["test_hash"]


def test_the_test_split_is_write_once_through_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-drawing the split with a different seed is refused; the same draw is idempotent.

    README: "The test split is write-once. Redrawing it invalidates every number reported
    against it, so `axial` will not do it quietly."
    """
    monkeypatch.chdir(tmp_path)
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "".join(
            json.dumps({"trace_id": f"t{i:03d}", "label": i % 2 == 0, "critique": "c",
                        "annotator": "andres"}) + "\n"
            for i in range(80)
        )
    )
    axial("label-import", "m", "--path", str(labels))
    axial("split", "m", "--seed", "0")
    splits_path = tmp_path / ".axial" / "datasets" / "m" / "splits.json"
    before = splits_path.read_text()

    redraw = axial_raw("split", "m", "--seed", "1")
    assert redraw.exit_code == 1
    assert "write-once" in redraw.output
    assert splits_path.read_text() == before
    axial("split", "m", "--seed", "0")


def test_an_always_pass_judge_scores_high_agreement_and_is_not_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`align` prints TPR/TNR with intervals beside accuracy; a do-nothing judge is refused.

    README: "Never raw agreement alone. TPR and TNR are printed beside it, with a confidence
    interval." / "No promotion below the bar."
    """
    repo = tmp_path / "repo"
    build_pipeline(repo, monkeypatch, judge_fn=always_pass_judge, promote=False)

    out = axial("align", MODE, "1").output
    test_line = next(line for line in out.splitlines() if line.startswith("test"))
    assert re.search(r"TPR=0\.000 \[[\d.]+, [\d.]+\]", test_line)
    assert re.search(r"TNR=1\.000 \[[\d.]+, 1\.000\]", test_line)
    accuracy = number_after("accuracy=", test_line)
    assert accuracy >= 0.75
    assert test_line.index("TPR") < test_line.index("accuracy")

    refused = axial_raw("promote", MODE, "1")
    assert refused.exit_code == 1
    assert "test TPR 0.000" in refused.output
    assert not (judge_dir(repo) / "PROMOTED").exists()


def test_infra_failures_are_quarantined_with_a_failure_class_never_scored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A judge call that raises lands in errors.jsonl with a class, never in verdicts.jsonl.

    README: "Infra failures never score as model failures. They are quarantined in
    `errors.jsonl` with a failure class."
    """

    def flaky(prompt: str) -> str:
        if graded_issue(prompt) % 10 == 3:
            raise TimeoutError("provider timed out")
        return perfect_judge(prompt)

    repo = tmp_path / "repo"
    build_pipeline(repo, monkeypatch, judge_fn=flaky, promote=False)
    errors = read_jsonl(judge_dir(repo) / "errors.jsonl")
    verdicts = read_jsonl(judge_dir(repo) / "verdicts.jsonl")

    assert errors and {e["failure_class"] for e in errors} == {"harness"}
    assert all("TimeoutError" in str(e["detail"]) for e in errors)
    assert not {e["trace_id"] for e in errors} & {v["trace_id"] for v in verdicts}
    out = axial("align", MODE, "1").output
    assert out
    assert f"Quarantined calls (never scored): {len(errors)}" in (
        judge_dir(repo) / "alignment.md"
    ).read_text()


def test_a_model_named_annotator_is_refused_by_label_from_reviews(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reviews recorded under a model-looking annotator never become labels.

    README: "Refuses annotators that look like model identifiers. A name-pattern check
    catches the obvious cases — a model's own name, "bot", "synthetic" — before a label
    enters dev or test"
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    axial("init")
    enable_annotator(repo, "claude-opus-5")
    axial("ingest", str(CORPUS))
    axial("sample", "--n", "5", "--name", "s")
    annotate_via_app("s", merged_verdict)  # the annotate app itself does not object

    result = axial_raw("label-from-reviews", MODE, "s")

    assert result.exit_code == 1
    assert "looks like a model" in result.output
    assert not (repo / ".axial" / "datasets" / MODE / "labels.jsonl").exists()


def _judge_with_real_model_builder(
    monkeypatch: pytest.MonkeyPatch, repo: Path, judge_model: str
) -> Result:
    """Run `judge` with the real `build_judge_model` (so its refusals apply) but a fake model."""
    real_build = judge_module.build_judge_model

    def build(spec: judge_module.JudgeSpec) -> FakeModel:
        real_build(spec)
        return FakeModel(perfect_judge, model=spec.model)

    _tiny_labelled_repo(repo, judge_model=judge_model)
    monkeypatch.setattr(cli, "build_judge_model", build)
    return axial_raw("judge", "m", "--split", "dev")


def test_a_floating_decision_model_alias_is_refused_by_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`axial judge` refuses a floating Jev alias with a message, not a traceback.

    README: "No floating model aliases. A judge model must be pinned to an exact version."
    """
    monkeypatch.chdir(tmp_path)
    result = _judge_with_real_model_builder(monkeypatch, tmp_path / "repo", "~typesafe/jev-latest")
    assert result.exit_code == 1
    assert "floating alias" in result.output
    assert isinstance(result.exception, SystemExit)  # a clean exit, not a traceback


@pytest.mark.xfail(
    strict=True,
    reason="the pinned-version check (_require_pinned) runs only in JevModel; "
    "LiteLLMModel accepts any id, so a text judge_model such as "
    "'anthropic/claude-opus-latest' is judged, aligned and promotable",
)
def test_a_floating_text_model_alias_is_refused_by_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`axial judge` refuses a floating text-model alias as it does a Jev one.

    README: "No floating model aliases. A judge model must be pinned to an exact version."
    """
    monkeypatch.chdir(tmp_path)
    result = _judge_with_real_model_builder(
        monkeypatch, tmp_path / "repo", "anthropic/claude-opus-latest"
    )
    assert result.exit_code == 1
    assert "floating alias" in result.output


def test_ingest_reads_csv_as_well_as_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`axial ingest` accepts a CSV file, keeping extra columns as metadata.

    README: "`axial ingest` | JSONL or CSV | normalised traces"
    """
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "traces.csv"
    src.write_text("trace_id,input,output,region\na,hi,hello,eu\nb,bye,goodbye,us\n")
    assert "ingested 2 traces" in axial("ingest", str(src)).output
    rows = read_jsonl(tmp_path / ".axial" / "traces" / "traces.jsonl")
    assert [(r["trace_id"], r["metadata"]) for r in rows] == [
        ("a", {"region": "eu"}), ("b", {"region": "us"})
    ]


def test_credentials_files_are_as_documented() -> None:
    """`.env` and `.axial/` are gitignored and `.env.example` names each model path's key.

    README: "`.env` is gitignored ... `.env.example` lists which key each model path needs"
    / "This repo gitignores its own `.axial/`"
    """
    ignored = (REPO_ROOT / ".gitignore").read_text().split()
    assert ".env" in ignored and ".axial/" in ignored
    example = (REPO_ROOT / ".env.example").read_text()
    for key in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
        assert re.search(rf"^{key}=$", example, re.MULTILINE), key


@pytest.mark.xfail(
    strict=True,
    reason="a 40-trace review session can never clear the default promotion bar: promote "
    "needs >=30 real failures and >=30 real passes in test, i.e. ~70 of each labelled at "
    "the 0.43 test fraction; the Quickstart's own session yields ~8 failures in total",
)
def test_the_quickstart_as_written_reaches_a_promoted_gate_ready_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Following the Quickstart verbatim (40-trace session, then label, split, judge) promotes.

    README: "then carry your own annotations the rest of the way to a promoted, gate-ready
    judge" (after `sample --n 40 --name first-pass` and "annotate 30+, then stop")
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    axial("init")
    enable_annotator(repo)
    axial("ingest", str(CORPUS))
    axial("sample", "--n", "40", "--name", "first-pass")
    annotate_via_app("first-pass", merged_verdict)  # a perfect annotator, generously
    axial("label-from-reviews", MODE, "first-pass")
    with pytest.warns(UserWarning):
        axial("split", MODE)
    use_fake_judge(monkeypatch, perfect_judge)  # and a perfect judge
    axial("judge", MODE, "--split", "dev")
    axial("judge", MODE, "--split", "test")
    axial("align", MODE, "1")
    promote = axial_raw("promote", MODE, "1")
    assert promote.exit_code == 0, promote.output
    assert axial_raw("gate", MODE, "--version", "1").exit_code == 0


@pytest.mark.xfail(
    strict=True,
    reason="the corpus plants 'hallucinated_source' and 'promo_not_filtered', not "
    "'hallucinated_sources' and 'promotional_content' as the README says",
)
def test_readme_names_the_failure_modes_the_newsroom_corpus_plants() -> None:
    """The four mode ids the Quickstart lists are the ones planted in the corpus.

    README: "the newsroom corpus plants `merged_stories`, `hallucinated_sources`,
    `wrong_topic` and `promotional_content`"
    """
    match = re.search(r"corpus\s+plants (.+?), so it", README, re.DOTALL)
    assert match is not None
    claimed = set(re.findall(r"`([a-z_]+)`", match.group(1)))
    assert claimed == set(planted_modes().values()) - {"clean"}


@pytest.mark.xfail(
    strict=True,
    reason="examples/newsroom/ holds only the corpus, its generator and a README that "
    "points at tests/test_end_to_end.py; nothing in it is FakeModel-backed, and that test "
    "drives the library rather than the CLI walkthrough",
)
def test_examples_newsroom_holds_a_fakemodel_backed_walkthrough() -> None:
    """examples/newsroom/ contains runnable code that walks the pipeline on FakeModel.

    README: "See `examples/newsroom/` for the fully offline, `FakeModel`-backed version of
    this same walkthrough."
    """
    code = [p for p in (REPO_ROOT / "examples" / "newsroom").rglob("*.py")]
    assert any("FakeModel" in p.read_text() for p in code)
