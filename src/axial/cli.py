"""The CLI. A thin shell over the library — no logic lives here."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, NoReturn

import typer
from pydantic import ValidationError

from axial.config import Config, load_config, load_env, write_default_config
from axial.dataset import (
    add_labels,
    class_balance_report,
    labels_from_reviews,
    make_splits,
    read_label_file,
    read_labels,
)
from axial.errors import AxialError, MalformedFile, validation_message
from axial.gate import gate as run_gate
from axial.gate import gate_batches
from axial.judge import (
    JudgeSpec,
    build_judge_model,
    kept_rows,
    read_errors,
    read_verdicts,
    run_judge,
)
from axial.measure import measure_batch, summary_lines
from axial.model import LiteLLMModel, is_decision_model, require_text_model
from axial.perplexity import DEGENERATE_SPREAD, score_traces
from axial.perplexity import spread as perplexity_spread
from axial.report import (
    alignment_from_verdicts,
    check_promotion,
    disagreement_counts,
    find_disagreements,
    write_alignment,
)
from axial.report import promote as do_promote
from axial.sample import (
    all_reviews,
    check_sample_options,
    create_session,
    read_session,
    refuse_existing_session,
    sample_batch,
)
from axial.store import Store, validate_name
from axial.taxonomy import Taxonomy, count_open_codes, propose_taxonomy, saturation_curve
from axial.traces import ingest as do_ingest

app = typer.Typer(help="Error-analysis-first eval toolkit: from traces to a judge you can trust.")

RepoOption = typer.Option(".", "--repo", help="Repository root holding .axial/")
# Hoisted out of `label_import`'s signature: ruff's B008 flags any function-call
# default on a parameter annotated `Path`, on the (here mistaken) assumption it
# might be `fastapi.Path` — a module-level singleton sidesteps that false positive,
# exactly as its own message suggests.
LabelPathOption = typer.Option(..., help="JSONL of {trace_id, label, critique, annotator}")
# Same B008 false positive, for `measure`'s batch file.
BatchTracesOption = typer.Option(
    ..., "--traces", help="JSONL or CSV of new traces to measure; never written to .axial/traces"
)


def _fail(message: str) -> NoReturn:
    typer.echo(message)
    raise typer.Exit(code=1)


def _one_line(exc: Exception) -> str:
    """`exc`'s message cut to one line, for a refusal that has to stay one line."""
    if isinstance(exc, ValidationError):
        return validation_message(exc)
    return (str(exc).splitlines() or [""])[0]


def _config(store: Store) -> Config:
    try:
        return load_config(store)
    except AxialError as exc:
        _fail(str(exc))


def _require_verdicts(verdicts: list[dict[str, Any]], split: str, mode: str, version: int) -> None:
    if not any(v.get("split") == split for v in verdicts):
        _fail(
            f"no {split} verdicts for {mode} v{version}; run `axial judge {mode} --split {split} "
            f"--version {version}` first"
        )


def _plain_mode(mode: str) -> str:
    try:
        return validate_name(mode, "mode")
    except AxialError as exc:
        _fail(str(exc))


# Every command that takes a mode checks it while parsing, before it reads or writes
# anything: a mode becomes a directory, and `../judges/x` once let `measure --replace`
# delete a promoted judge.
ModeArgument = typer.Argument(..., callback=_plain_mode)


def _plain_session(name: str) -> str:
    try:
        return validate_name(name, "session")
    except AxialError as exc:
        _fail(str(exc))


def _plain_sessions(names: list[str]) -> list[str]:
    return [_plain_session(name) for name in names]


# Review session names are checked the same way: a session becomes
# `.axial/reviews/<name>.jsonl`, and `sample --name ../escape` once wrote outside it.
# Hoisted for the same B008 false positive as LabelPathOption above.
SessionsArgument = typer.Argument(
    ..., help="Review session name(s) to convert (repeatable)", callback=_plain_sessions
)


@app.command()
def init(repo: str = RepoOption) -> None:
    """Create .axial/ and a default config."""
    store = Store(repo)
    if write_default_config(store):
        typer.echo(f"initialised {store.root}")
    else:
        typer.echo(
            f"{store.config_path} already exists; left it unchanged "
            "(delete it and run init again for the defaults)"
        )


@app.command()
def ingest(
    path: Path,
    adapter: str = typer.Option(None, "--adapter", help="Name of a registered adapter"),
    repo: str = RepoOption,
) -> None:
    """Load traces from a JSONL or CSV file, or through a registered adapter."""
    store = Store(repo)
    try:
        traces = do_ingest(path, adapter=adapter)
    except (OSError, ValueError) as exc:
        _fail(f"could not read {path}: {_one_line(exc)}")
    try:
        store.write_traces(traces)
    except OSError as exc:
        _fail(f"could not write {store.traces_path}: {exc}")
    typer.echo(f"ingested {len(traces)} traces into {store.traces_path}")


@app.command()
def sample(
    n: int = typer.Option(20, help="How many traces to put in the batch"),
    name: str = typer.Option(..., help="Name for this review session", callback=_plain_session),
    strategy: str = typer.Option(
        "random", help="random | outlier | longest | perplexity"
    ),
    random_fraction: float = typer.Option(
        0.3,
        "--random-fraction",
        help="Fraction of a targeted batch (outlier | longest | perplexity) to fill with a "
        "random draw instead of the targeted ranking. Every targeted strategy "
        "is biased toward the failures you already knew to look for, so a "
        "random slice is what catches the ones you didn't. Ignored by "
        "--strategy random. round(n * random_fraction) can be 0 for a small "
        "n, which drops the random slice entirely for that batch.",
    ),
    perplexity_field: str = typer.Option(
        "input",
        "--perplexity-field",
        help="Which side of the trace --strategy perplexity ranks on: input | "
        "output | both. Defaults to input, where an unusual value means an edge "
        "case the system was asked to handle. Ranking outputs sorts the most "
        "degenerate ones — a truncated stub, one sentence repeated — to the "
        "bottom, because repetition is the least surprising text there is.",
    ),
    seed: int = typer.Option(0),
    repo: str = RepoOption,
) -> None:
    """Draw a review batch. Targeted strategies mix in a random slice."""
    store = Store(repo)
    try:
        check_sample_options(n, strategy, random_fraction, perplexity_field)
    except ValueError as exc:
        _fail(str(exc))
    try:
        refuse_existing_session(store, name)
    except FileExistsError as exc:
        _fail(f"{exc}; choose another --name")
    try:
        traces = store.read_traces()
    except (OSError, ValueError) as exc:
        _fail(f"could not read the traces: {_one_line(exc)}")
    if strategy != "random" and random_fraction == 0:
        typer.echo(
            "warning: --random-fraction 0 with a targeted strategy disables invariant 10 "
            "(every batch keeps a random slice) — every trace in this batch was already "
            "suspected of a problem before it was chosen"
        )
    if strategy == "perplexity":
        ratio = perplexity_spread(score_traces(traces, field=perplexity_field))
        if math.isfinite(ratio):
            typer.echo(f"perplexity spread (max/median): {ratio:.2f}")
        if math.isfinite(ratio) and ratio < DEGENERATE_SPREAD:
            typer.echo(
                f"warning: a spread of {ratio:.2f} means the most surprising trace is barely "
                "more surprising than the typical one, so this ranking is close to arbitrary — "
                "it will be sorting on whatever incidental token happens to vary, such as an id "
                "or a timestamp. A corpus of near-identical traces has nothing for perplexity "
                "to find; --strategy random is the honest choice for it."
            )
    batch = sample_batch(
        traces,
        n,
        strategy=strategy,
        random_fraction=random_fraction,
        perplexity_field=perplexity_field,
        seed=seed,
    )
    try:
        create_session(store, batch, name=name)
    except OSError as exc:
        _fail(f"could not write review session {name!r}: {exc}")
    typer.echo(f"session {name}: {len(batch)} traces ready to annotate")


@app.command()
def annotate(
    name: str = typer.Option(..., help="Review session to open", callback=_plain_session),
    port: int = typer.Option(8000),
    repo: str = RepoOption,
) -> None:
    """Open the annotation app in a local server."""
    import uvicorn

    from axial.annotate.app import create_app

    store = Store(repo)
    config = _config(store)
    if not config.annotator:
        _fail("set `annotator` in .axial/config.toml first — every label records who made it")
    typer.echo(f"annotating {name} at http://127.0.0.1:{port}")
    uvicorn.run(create_app(store, name, config.annotator), host="127.0.0.1", port=port)


@app.command()
def taxonomy(repo: str = RepoOption) -> None:
    """Propose a failure taxonomy from your open codes."""
    store = Store(repo)
    load_env(repo)
    config = _config(store)
    try:
        require_text_model(config.judge_model)
    except AxialError as exc:
        _fail(str(exc))
    try:
        reviews = all_reviews(store)
    except (OSError, ValueError) as exc:
        _fail(f"could not read the review sessions: {_one_line(exc)}")
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
    except ValueError as exc:  # the reply held no JSON, or JSON of the wrong shape
        _fail(_one_line(exc))
    result.save(store)
    for mode in result.modes:
        typer.echo(f"  {mode.id}: {mode.name} (prevalence {mode.prevalence})")


@app.command("label-import")
def label_import(
    mode: str = ModeArgument,
    path: Path = LabelPathOption,
    repo: str = RepoOption,
) -> None:
    """Import human labels you made elsewhere."""
    store = Store(repo)
    try:
        labels, ignored = read_label_file(store, path)
        add_labels(store, mode, labels)
    except AxialError as exc:
        _fail(str(exc))
    except (OSError, ValueError) as exc:
        _fail(f"could not import labels: {_one_line(exc)}")
    note = f"; ignored columns: {', '.join(ignored)}" if ignored else ""
    typer.echo(f"{len(labels)} labels recorded for {mode}{note}")


@app.command("label-from-reviews")
def label_from_reviews(
    mode: str = ModeArgument,
    sessions: list[str] = SessionsArgument,
    repo: str = RepoOption,
) -> None:
    """Convert completed `annotate` reviews into labels for one failure mode.

    Every completed review already carries a verdict, a critique and an
    annotator -- a `Label` in all but name. This is the path from annotation
    to labels that `label-import` (a hand-built file) does not cover:
    `verdict == "fail"` becomes `label=True`, `"pass"` becomes `label=False`,
    and a `pending` review is skipped rather than silently counted.
    """
    store = Store(repo)
    try:
        reviews = [r for name in sessions for r in read_session(store, name, missing_ok=False)]
        labels, skipped = labels_from_reviews(reviews)
        add_labels(store, mode, labels)
    except AxialError as exc:
        _fail(str(exc))
    except (OSError, ValueError) as exc:
        _fail(f"could not record labels for {mode}: {_one_line(exc)}")
    typer.echo(f"{len(labels)} labels recorded for {mode}; {skipped} pending review(s) skipped")


@app.command()
def split(mode: str = ModeArgument, seed: int = typer.Option(0), repo: str = RepoOption) -> None:
    """Draw a stratified train/dev/test split and freeze the test manifest."""
    store = Store(repo)
    config = _config(store)
    try:
        labels = read_labels(store, mode)
    except (OSError, ValueError) as exc:
        _fail(f"could not split {mode}: {_one_line(exc)}")
    if not labels:
        # Saving an empty split would freeze an empty test manifest, and the test split
        # is write-once, so every split after labelling would then be refused.
        _fail(
            f"{mode} has no labels yet; record some with `axial label-import` or "
            f"`axial label-from-reviews` before `axial split {mode}`"
        )
    splits = make_splits(store, labels, seed=seed, fractions=config.split_fractions)
    try:
        splits.save(store, mode)
    except AxialError as exc:
        _fail(str(exc))
    except (OSError, ValueError) as exc:
        _fail(f"could not split {mode}: {_one_line(exc)}")
    typer.echo(json.dumps(class_balance_report(labels, splits), indent=2))


FewShotOption = typer.Option(
    [],
    "--few-shot",
    help="Trace id to use as a few-shot example (repeatable). The id must be in "
    "the train split and must carry a human label; run_judge raises "
    "InvariantViolation otherwise, so that an example from dev or test can't "
    "leak that split into the judge. Supplying this replaces the spec's "
    "existing few_shot_ids; omitting it leaves them as they are.",
)


NewOption = typer.Option(
    False,
    "--new",
    help="Start a new judge version instead of resuming the latest one for this "
    "mode. Reach for this after changing the judge prompt (or its few-shot "
    "examples) between runs — a version already scored against test must not "
    "silently keep being reused for a different judge.",
)


@app.command()
def judge(
    mode: str = ModeArgument,
    split_name: str = typer.Option("dev", "--split", help="dev | test"),
    version: int = typer.Option(
        0,
        help="Judge version to use. 0 (the default) resumes the latest existing "
        "version for this mode, or starts at 1 if none exists yet — this is what "
        "lets `judge --split dev` and a later `judge --split test` land on the same "
        "judge. Pass an explicit version to target one directly, or use --new to "
        "start the next one instead of resuming.",
    ),
    new: bool = NewOption,
    few_shot: list[str] = FewShotOption,
    repo: str = RepoOption,
) -> None:
    """Run the judge over one split."""
    from axial.dataset import Splits

    valid_splits = ("train", "dev", "test")
    if split_name not in valid_splits:
        _fail(f"unknown split {split_name!r}; expected one of {valid_splits}")

    store = Store(repo)
    load_env(repo)
    config = _config(store)
    if not (store.dataset_dir(mode) / "splits.json").exists():
        _fail(f"{mode} has no split yet; run `axial split {mode}` first")
    if not version:
        if new:
            version = store.next_judge_version(mode)
        else:
            version = store.latest_judge_version(mode) or 1

    # Everything the run reads is read before judge.yaml is written, so an input that
    # is missing or corrupt leaves no half-made judge behind.
    judge_yaml = store.judge_dir(mode, version) / "judge.yaml"
    try:
        taxonomy_modes = {m.id: m for m in Taxonomy.load(store).modes}
        existing = JudgeSpec.load(store, mode, version) if judge_yaml.exists() else None
        splits = Splits.load(store, mode)
        # `corpus` is read once, in full, and passed separately from `traces`: the
        # split under judgment is almost never where its own few-shot examples live
        # (those come from train), so `run_judge` needs the whole trace set to
        # resolve `few_shot_ids` against — passing only the split's traces here
        # would silently starve few-shot resolution.
        corpus = store.read_traces()
        labels = read_labels(store, mode)
        # run_judge reads the rows it keeps before grading, but by then judge.yaml is
        # saved; reading them here refuses a damaged verdicts or errors file first.
        kept_rows(store, mode, version, split_name)
    except (OSError, ValueError) as exc:
        _fail(f"could not judge {mode}: {_one_line(exc)}")
    entry = taxonomy_modes.get(mode)

    # Load an existing spec rather than overwriting it: a fresh `JudgeSpec` on
    # every run silently discarded `few_shot_ids` and any hand-edited
    # `prompt_template`. Only build a new spec when this mode/version has none
    # yet; a spec that already exists is the source of truth for everything it
    # already carries.
    if existing is not None:
        spec = existing
        if not spec.mode_name and entry:
            spec.mode_name = entry.name
        if not spec.mode_definition and entry:
            spec.mode_definition = entry.definition
    else:
        spec = JudgeSpec(
            mode=mode,
            model=config.judge_model,
            version=version,
            mode_name=entry.name if entry else mode,
            mode_definition=entry.definition if entry else "",
        )
    if few_shot:
        spec.few_shot_ids = few_shot
    spec.save(store)

    wanted = set(getattr(splits, split_name))
    traces = [t for t in corpus if t.trace_id in wanted]
    try:
        result = run_judge(
            store, spec, traces, labels, splits,
            build_judge_model(spec), split=split_name, corpus=corpus,
        )
    except AxialError as exc:
        _fail(str(exc))
    typer.echo(f"{len(result.verdicts)} verdicts, {len(result.errors)} quarantined")


@app.command()
def align(
    mode: str = ModeArgument, version: int = typer.Argument(...), repo: str = RepoOption
) -> None:
    """Report agreement between a judge and your labels."""
    store = Store(repo)
    config = _config(store)
    directory = store.judge_dir(mode, version)
    judge_yaml = directory / "judge.yaml"
    try:
        verdicts = read_verdicts(store, directory / "verdicts.jsonl")
        errors = read_errors(store, directory / "errors.jsonl")
        labels = read_labels(store, mode)
        traces = store.read_traces()
        spec = JudgeSpec.load(store, mode, version) if judge_yaml.exists() else None
    except (OSError, ValueError) as exc:
        _fail(f"could not align {mode} v{version}: {_one_line(exc)}")
    _require_verdicts(verdicts, "dev", mode, version)
    try:
        dev = alignment_from_verdicts(verdicts, "dev")
        has_test = any(v.get("split") == "test" for v in verdicts)
        test = alignment_from_verdicts(verdicts, "test") if has_test else None
    except ValueError as exc:  # no human label, or a recorded probability that is not one
        _fail(f"could not align {mode} v{version}: {_one_line(exc)}")
    notes: list[str] = []
    if test is not None:
        # `check_promotion` is the only place that decides a gap check was
        # skipped (e.g. because a rate is nan on dev or test); reuse its notes
        # here so that a skipped check is recorded in alignment.md rather than
        # only surfacing later, at `promote` time.
        notes = check_promotion(dev, test, config.thresholds).notes
    # judge.yaml names the model; a row records whatever id the provider served,
    # which can be a bare `jev-1.13` that no longer says it is a decision model.
    if spec is not None and is_decision_model(spec.model) and dev.brier is None:
        notes.append(
            "no Brier score: these decision-model verdicts carry no probability, so they "
            "were written before axial recorded one. Re-run "
            f"`axial judge {mode} --split dev --version {version}` to record it. The re-run "
            "replaces this version's dev verdicts and drops its PROMOTED marker, so it must "
            "be promoted again."
        )
    disagreements = find_disagreements(verdicts, labels, traces)
    path = write_alignment(
        store, mode, version, dev, test,
        n_errors=len(errors), notes=notes, disagreements=disagreements,
    )
    for name, result in (("dev ", dev), ("test", test)):
        if result is None:
            continue
        typer.echo(f"{name} {result.summary()}")
        if result.brier is not None:
            typer.echo(
                f"{name} Brier {result.brier:.3f} "
                f"(base-rate reference {result.brier_reference:.3f})"
            )
    typer.echo(disagreement_counts(disagreements))
    typer.echo(f"written to {path}")


@app.command("tune-threshold")
def tune_threshold(
    mode: str = ModeArgument,
    version: int = typer.Argument(...),
    folds: int = typer.Option(
        5, help="Stratified folds for the cross-validated estimate. Dev needs at least this "
        "many real failures and this many real passes."
    ),
    seed: int = typer.Option(0),
    write: bool = typer.Option(
        False, "--write", help="Save the chosen threshold as the next judge version."
    ),
    repo: str = RepoOption,
) -> None:
    """Choose a decision model's threshold from its dev verdicts. Calls no model."""
    from axial.threshold import tune_from_verdicts, write_tuned_spec

    store = Store(repo)
    config = _config(store)
    directory = store.judge_dir(mode, version)
    if not (directory / "judge.yaml").exists():
        _fail(f"no judge {mode} v{version}")
    try:
        spec = JudgeSpec.load(store, mode, version)
        verdicts = read_verdicts(store, directory / "verdicts.jsonl")
    except (OSError, ValueError) as exc:
        _fail(f"could not tune {mode} v{version}: {_one_line(exc)}")
    try:
        tuning = tune_from_verdicts(verdicts, spec, config.thresholds, folds=folds, seed=seed)
    except AxialError as exc:
        _fail(str(exc))

    floors = config.thresholds
    choice = tuning.choice
    typer.echo(
        f"dev: {tuning.n_failures} real failures, {tuning.n_passes} real passes, "
        f"scored by {spec.model}; floors TPR >= {floors.min_tpr:.2f}, TNR >= {floors.min_tnr:.2f}"
    )
    typer.echo(
        f"current threshold {tuning.current_threshold}: "
        f"TPR {tuning.current_tpr:.3f}  TNR {tuning.current_tnr:.3f}"
    )
    typer.echo(
        f"{'chosen' if choice.feasible else 'closest'} threshold {choice.threshold}: "
        f"TPR {choice.tpr:.3f}  TNR {choice.tnr:.3f} on dev, which it was chosen on"
    )
    held_out = tuning.cv.held_out
    typer.echo(
        f"{tuning.cv.folds}-fold cross-validation on dev: TPR {held_out.fmt('tpr')}  "
        f"TNR {held_out.fmt('tnr')}, each trace scored by a threshold chosen without it"
    )
    if tuning.cv.infeasible_folds:
        typer.echo(
            f"  no threshold cleared the floors in {tuning.cv.infeasible_folds} of "
            f"{tuning.cv.folds} training folds; those folds were scored at their closest one"
        )
    if not choice.feasible:
        _fail(
            "no threshold clears both floors on dev. Improve the judge prompt or the labels. "
            "The floors in .axial/config.toml are yours to set, and tune-threshold will not "
            "lower them."
        )
    if not write:
        typer.echo(f"pass --write to save it as {mode} v{store.next_judge_version(mode)}")
        return
    tuned = write_tuned_spec(store, spec, tuning)
    typer.echo(
        f"wrote {mode} v{tuned.version} with decision_threshold {tuned.decision_threshold}. "
        f"Run `axial judge {mode} --split dev --version {tuned.version}` to record its "
        "verdicts, then --split test once."
    )


@app.command()
def checks(
    mode: str = ModeArgument,
    version: int = typer.Argument(...),
    key: str = typer.Option(None, "--slice-by", help="Metadata key to break alignment down by"),
    split_name: str = typer.Option("test", "--split"),
    repo: str = RepoOption,
) -> None:
    """Slice breakdown, verbosity-bias correlation, cost, and the noise floor for this set."""
    from axial.checks import ci_half_width, cost_usd, slice_alignment, verbosity_bias

    store = Store(repo)
    try:
        verdicts = read_verdicts(store, store.judge_dir(mode, version) / "verdicts.jsonl")
        traces = store.read_traces()
    except (OSError, ValueError) as exc:
        _fail(f"could not check {mode} v{version}: {_one_line(exc)}")
    rows = [v for v in verdicts if v.get("split") == split_name]
    if not rows:
        _fail(f"no {split_name} verdicts for {mode} v{version}")
    typer.echo(f"noise floor at n={len(rows)}: +/-{ci_half_width(len(rows)):.3f}")

    # Bias and cost describe the same split as the noise floor, so all three are read
    # from `rows`; the other splits' verdicts would pool into numbers nobody asked for.
    bias = verbosity_bias(rows, traces)
    typer.echo(
        f"verbosity bias (length vs fail): {bias:.3f}"
        if not math.isnan(bias)
        else "verbosity bias: n/a (verdicts are all the same)"
    )

    # Cost from each row's own recorded model and usage -- never a flat assumed
    # rate. A row is reported as unknown, not folded into the total as zero,
    # for either of two reasons: its model has no entry in checks.PRICES, or
    # (invariant 9) it carries no usage at all -- a runner bug or a provider
    # that omitted it, not a genuinely free call. Both are exactly the
    # shortcut that can make a cost comparison come out backwards.
    known_cost = 0.0
    unknown_rows = 0
    unpriced_models: set[str] = set()
    for row in rows:
        usage = row.get("usage", {})
        model_name = str(row.get("model", "unknown"))
        priced = cost_usd(usage, model_name)
        if priced is None:
            unknown_rows += 1
            if usage:
                # Usage was recorded; it's the model's price that's missing.
                unpriced_models.add(model_name)
        else:
            known_cost += priced
    # The line names the split: the cost of one split is easily read as the whole run's.
    label = f"cost of the {split_name} split"
    if unknown_rows:
        detail = f"; unpriced model(s): {', '.join(sorted(unpriced_models))}" if unpriced_models else ""
        typer.echo(f"{label}: ${known_cost:.4f} known, {unknown_rows} row(s) unknown{detail}")
    else:
        typer.echo(f"{label}: ${known_cost:.4f}")

    if key:
        for value, result in sorted(slice_alignment(verdicts, traces, key, split=split_name).items()):
            typer.echo(f"  {key}={value}: {result.summary()}")


@app.command()
def promote(
    mode: str = ModeArgument, version: int = typer.Argument(...), repo: str = RepoOption
) -> None:
    """Promote a judge to CI, if its test alignment clears the bar."""
    store = Store(repo)
    config = _config(store)
    try:
        verdicts = read_verdicts(store, store.judge_dir(mode, version) / "verdicts.jsonl")
    except (OSError, ValueError) as exc:
        _fail(f"could not promote {mode} v{version}: {_one_line(exc)}")
    _require_verdicts(verdicts, "dev", mode, version)
    _require_verdicts(verdicts, "test", mode, version)
    try:
        dev = alignment_from_verdicts(verdicts, "dev")
        test = alignment_from_verdicts(verdicts, "test")
    except ValueError as exc:  # a verdict row that carries no human label
        _fail(f"could not promote {mode} v{version}: {_one_line(exc)}")
    try:
        do_promote(store, mode, version, dev, test, config.thresholds)
    except AxialError as exc:
        # Print the refusal one line at a time rather than collapsing it to a
        # single `typer.echo(str(exc))` call: `check_promotion` can attach
        # several independent reasons (e.g. both `min_test_positives` and
        # `min_test_negatives` failing at once), and every one of them must be
        # legible in the output, not just the first.
        for line in str(exc).splitlines():
            typer.echo(line)
        raise typer.Exit(code=1) from None
    except (OSError, MalformedFile) as exc:
        # A corrupt errors.jsonl, read to check every row came from the current prompt,
        # or a PROMOTED marker that cannot be written. Any other ValueError is a bug, so
        # it is left uncaught.
        _fail(f"could not promote {mode} v{version}: {_one_line(exc)}")
    typer.echo(f"{mode} v{version} promoted")


@app.command()
def measure(
    mode: str = ModeArgument,
    version: int = typer.Option(..., "--version", help="Promoted judge version to grade with"),
    traces: Path = BatchTracesOption,
    name: str = typer.Option(
        ..., "--name", help="Batch name; the measurement goes to .axial/measurements/<mode>/<name>/"
    ),
    adapter: str | None = typer.Option(None, "--adapter", help="Name of a registered adapter"),
    replace: bool = typer.Option(
        False, "--replace", help="Replace an existing measurement with the same name"
    ),
    repo: str = RepoOption,
) -> None:
    """Grade new traces with a promoted judge and report a corrected failure rate."""
    store = Store(repo)
    load_env(repo)
    try:
        batch = do_ingest(traces, adapter=adapter)
    except (OSError, ValueError) as exc:
        _fail(f"could not read {traces}: {exc}")
    try:
        result = measure_batch(
            store, mode, version, name, batch, build_judge_model, replace=replace
        )
    except AxialError as exc:
        _fail(str(exc))
    except OSError as exc:
        _fail(f"could not measure {name!r}: {exc}")
    for line in summary_lines(result):
        typer.echo(line)
    for note in result.notes:
        typer.echo(f"note: {note}")
    if result.rate is None:
        raise typer.Exit(code=1)


@app.command()
def gate(
    mode: str = ModeArgument,
    version: int | None = typer.Option(
        None, "--version", help="Promoted judge version to gate on (compares judge versions)"
    ),
    baseline: int | None = typer.Option(
        None, "--baseline", help="With --version: judge version to compare against"
    ),
    batch: str | None = typer.Option(
        None, "--batch", help="Measured batch to gate on (see `axial measure`)"
    ),
    baseline_batch: str | None = typer.Option(
        None, "--baseline-batch", help="With --batch: measured batch to compare against"
    ),
    max_rate: float | None = typer.Option(
        None,
        "--max-rate",
        help="With --batch: fail if the upper end of the corrected rate's 95% interval "
        "is above this",
    ),
    repo: str = RepoOption,
) -> None:
    """Fail the build on a regression outside the noise floor."""
    if (version is None) == (batch is None):
        raise typer.BadParameter(
            "give exactly one of --version (compare judge versions) or --batch (gate "
            "measured batches)"
        )
    if version is not None and (baseline_batch is not None or max_rate is not None):
        raise typer.BadParameter("--baseline-batch and --max-rate need --batch")
    if batch is not None and baseline is not None:
        raise typer.BadParameter(
            "--baseline compares judge versions and needs --version; to compare measured "
            "batches use --baseline-batch"
        )
    if max_rate is not None and not 0 < max_rate < 1:
        raise typer.BadParameter("--max-rate must lie strictly between 0 and 1")
    if batch is not None and batch == baseline_batch:
        raise typer.BadParameter(
            "--baseline-batch names the same batch as --batch; a batch compared with itself "
            "can never regress"
        )
    store = Store(repo)
    try:
        if version is not None:
            result = run_gate(store, mode, version, baseline_version=baseline)
        else:
            assert batch is not None  # exactly one of the two was checked above
            result = gate_batches(
                store, mode, batch, baseline_batch=baseline_batch, max_rate=max_rate
            )
    except AxialError as exc:
        _fail(str(exc))
    for message in result.messages:
        typer.echo(message)
    if not result.passed:
        raise typer.Exit(code=1)
