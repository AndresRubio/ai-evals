"""The CLI. A thin shell over the library — no logic lives here."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NoReturn

import typer

from axial.config import load_config, write_default_config
from axial.dataset import Label, add_labels, class_balance_report, make_splits, read_labels
from axial.errors import AxialError
from axial.gate import gate as run_gate
from axial.judge import JudgeSpec, run_judge
from axial.model import LiteLLMModel
from axial.report import alignment_from_verdicts, check_promotion, write_alignment
from axial.report import promote as do_promote
from axial.sample import all_reviews, create_session, sample_batch
from axial.store import Store
from axial.taxonomy import Taxonomy, count_open_codes, propose_taxonomy, saturation_curve
from axial.traces import ingest_csv, ingest_jsonl

app = typer.Typer(help="Error-analysis-first eval toolkit: from traces to a judge you can trust.")

RepoOption = typer.Option(".", "--repo", help="Repository root holding .axial/")
# Hoisted out of `label_import`'s signature: ruff's B008 flags any function-call
# default on a parameter annotated `Path`, on the (here mistaken) assumption it
# might be `fastapi.Path` — a module-level singleton sidesteps that false positive,
# exactly as its own message suggests.
LabelPathOption = typer.Option(..., help="JSONL of {trace_id, label, critique, annotator}")


def _fail(message: str) -> NoReturn:
    typer.echo(message)
    raise typer.Exit(code=1)


@app.command()
def init(repo: str = RepoOption) -> None:
    """Create .axial/ and a default config."""
    store = Store(repo)
    write_default_config(store)
    typer.echo(f"initialised {store.root}")


@app.command()
def ingest(path: Path, repo: str = RepoOption) -> None:
    """Load traces from a JSONL or CSV file."""
    store = Store(repo)
    traces = ingest_csv(path) if path.suffix == ".csv" else ingest_jsonl(path)
    store.write_traces(traces)
    typer.echo(f"ingested {len(traces)} traces into {store.traces_path}")


@app.command()
def sample(
    n: int = typer.Option(20, help="How many traces to put in the batch"),
    name: str = typer.Option(..., help="Name for this review session"),
    strategy: str = typer.Option("random", help="random | outlier | longest"),
    seed: int = typer.Option(0),
    repo: str = RepoOption,
) -> None:
    """Draw a review batch. Targeted strategies always keep a random slice."""
    store = Store(repo)
    batch = sample_batch(store.read_traces(), n, strategy=strategy, seed=seed)
    create_session(store, batch, name=name)
    typer.echo(f"session {name}: {len(batch)} traces ready to annotate")


@app.command()
def annotate(
    name: str = typer.Option(..., help="Review session to open"),
    port: int = typer.Option(8000),
    repo: str = RepoOption,
) -> None:
    """Open the annotation app in a local server."""
    import uvicorn

    from axial.annotate.app import create_app

    store = Store(repo)
    config = load_config(store)
    if not config.annotator:
        _fail("set `annotator` in .axial/config.toml first — every label records who made it")
    typer.echo(f"annotating {name} at http://127.0.0.1:{port}")
    uvicorn.run(create_app(store, name, config.annotator), host="127.0.0.1", port=port)


@app.command()
def taxonomy(repo: str = RepoOption) -> None:
    """Propose a failure taxonomy from your open codes."""
    store = Store(repo)
    config = load_config(store)
    reviews = all_reviews(store)
    typer.echo(f"{count_open_codes(reviews)} open codes; saturation {saturation_curve(reviews)[-20:]}")
    model = LiteLLMModel(config.judge_model)
    try:
        result = propose_taxonomy(
            reviews,
            lambda p: model.complete(p, max_tokens=2048, schema=Taxonomy).text,
            min_open_codes=config.min_open_codes,
        )
    except AxialError as exc:
        _fail(str(exc))
    result.save(store)
    for mode in result.modes:
        typer.echo(f"  {mode.id}: {mode.name} (prevalence {mode.prevalence})")


@app.command("label-import")
def label_import(
    mode: str,
    path: Path = LabelPathOption,
    repo: str = RepoOption,
) -> None:
    """Import human labels you made elsewhere."""
    store = Store(repo)
    rows = store.read_jsonl(path)
    try:
        add_labels(store, mode, [Label.model_validate(r) for r in rows])
    except AxialError as exc:
        _fail(str(exc))
    typer.echo(f"{len(rows)} labels recorded for {mode}")


@app.command()
def split(mode: str, seed: int = typer.Option(0), repo: str = RepoOption) -> None:
    """Draw a stratified train/dev/test split and freeze the test manifest."""
    store = Store(repo)
    config = load_config(store)
    labels = read_labels(store, mode)
    splits = make_splits(store, labels, seed=seed, fractions=config.split_fractions)
    try:
        splits.save(store, mode)
    except AxialError as exc:
        _fail(str(exc))
    typer.echo(json.dumps(class_balance_report(labels, splits), indent=2))


@app.command()
def judge(
    mode: str,
    split_name: str = typer.Option("dev", "--split", help="dev | test"),
    version: int = typer.Option(0, help="0 means next version"),
    repo: str = RepoOption,
) -> None:
    """Run the judge over one split."""
    from axial.dataset import Splits

    store = Store(repo)
    config = load_config(store)
    version = version or store.next_judge_version(mode)
    taxonomy_modes = {m.id: m for m in Taxonomy.load(store).modes}
    entry = taxonomy_modes.get(mode)
    spec = JudgeSpec(
        mode=mode,
        model=config.judge_model,
        version=version,
        mode_name=entry.name if entry else mode,
        mode_definition=entry.definition if entry else "",
    )
    spec.save(store)
    splits = Splits.load(store, mode)
    wanted = set(getattr(splits, split_name))
    # `corpus` is read once, in full, and passed separately from `traces`: the
    # split under judgment is almost never where its own few-shot examples live
    # (those come from train), so `run_judge` needs the whole trace set to
    # resolve `few_shot_ids` against — passing only the split's traces here
    # would silently starve few-shot resolution.
    corpus = store.read_traces()
    traces = [t for t in corpus if t.trace_id in wanted]
    result = run_judge(
        store, spec, traces, read_labels(store, mode), splits,
        LiteLLMModel(config.judge_model), split=split_name, corpus=corpus,
    )
    typer.echo(f"{len(result.verdicts)} verdicts, {len(result.errors)} quarantined")


@app.command()
def align(mode: str, version: int, repo: str = RepoOption) -> None:
    """Report agreement between a judge and your labels."""
    store = Store(repo)
    config = load_config(store)
    directory = store.judge_dir(mode, version)
    verdicts = store.read_jsonl(directory / "verdicts.jsonl")
    errors = store.read_jsonl(directory / "errors.jsonl")
    dev = alignment_from_verdicts(verdicts, "dev")
    test = None
    notes: list[str] = []
    if any(v["split"] == "test" for v in verdicts):
        test = alignment_from_verdicts(verdicts, "test")
        # `check_promotion` is the only place that decides a gap check was
        # skipped (e.g. because a rate is nan on dev or test); reuse its notes
        # here so that a skipped check is recorded in alignment.md rather than
        # only surfacing later, at `promote` time.
        notes = check_promotion(dev, test, config.thresholds).notes
    path = write_alignment(store, mode, version, dev, test, n_errors=len(errors), notes=notes)
    typer.echo(f"dev  {dev.summary()}")
    if test:
        typer.echo(f"test {test.summary()}")
    typer.echo(f"written to {path}")


@app.command()
def promote(mode: str, version: int, repo: str = RepoOption) -> None:
    """Promote a judge to CI, if its test alignment clears the bar."""
    store = Store(repo)
    config = load_config(store)
    verdicts = store.read_jsonl(store.judge_dir(mode, version) / "verdicts.jsonl")
    try:
        do_promote(
            store, mode, version,
            alignment_from_verdicts(verdicts, "dev"),
            alignment_from_verdicts(verdicts, "test"),
            config.thresholds,
        )
    except AxialError as exc:
        # Print the refusal one line at a time rather than collapsing it to a
        # single `typer.echo(str(exc))` call: `check_promotion` can attach
        # several independent reasons (e.g. both `min_test_positives` and
        # `min_test_negatives` failing at once), and every one of them must be
        # legible in the output, not just the first.
        for line in str(exc).splitlines():
            typer.echo(line)
        raise typer.Exit(code=1) from None
    typer.echo(f"{mode} v{version} promoted")


@app.command()
def gate(
    mode: str,
    version: int = typer.Option(..., "--version"),
    baseline: int = typer.Option(None, "--baseline"),
    repo: str = RepoOption,
) -> None:
    """Fail the build on a regression outside the noise floor."""
    store = Store(repo)
    try:
        result = run_gate(store, mode, version, baseline_version=baseline)
    except AxialError as exc:
        _fail(str(exc))
    for message in result.messages:
        typer.echo(message)
    if not result.passed:
        raise typer.Exit(code=1)
