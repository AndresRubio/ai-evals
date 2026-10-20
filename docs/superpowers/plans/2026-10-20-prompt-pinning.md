# Prompt Pinning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A promoted judge counts as promoted only while the prompt it sends is the prompt its dev and test verdicts were produced with, including the few-shot examples resolved from the current traces and labels.

**Architecture:** `judge.py` gains a prompt fingerprint (SHA-256 of the spec's fields plus the rendered few-shot block). `run_judge` stamps it on every row, `promote` refuses dev or test rows with another stamp and records it in `PROMOTED`, `require_promoted` recomputes it from the current corpus and labels, `measure` records it in `measurement.json`, and `gate --batch` checks it. `measure` recognises a few-shot example in a batch by input hash.

**Tech Stack:** Python 3.13/3.14, pydantic, typer, pytest, ruff, mypy --strict, uv.

**Spec:** [`docs/superpowers/specs/2026-10-20-prompt-pinning-design.md`](../specs/2026-10-20-prompt-pinning-design.md)

## Global Constraints

- Work in `/Users/autobot/Desktop/projects/evals-prompt-pinning` (branch `prompt-pinning`, cut from `corrected-rates`). Never run `git checkout` or `git switch` in `/Users/autobot/Desktop/projects/evals`.
- Commit with `git -c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com commit`, every message ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Do not push.
- CI runs, and each task must pass before its commit: `uv run ruff check`, `uv run mypy --strict src/axial`, `uv run pytest -q`, `uv run python scripts/knowledge_index.py --check`. Baseline before Task 1: 1030 passed, 49 xfailed.
- The fingerprint is `prompt_sha256`, everywhere: verdict and error rows, the `PROMOTED` line `prompt_sha256=<64 hex>`, `PromotedHashes.prompt_sha256`, `Measurement.prompt_sha256` and the `measurement.json` key.
- `promote` checks every `dev` and `test` row in both `verdicts.jsonl` and `errors.jsonl`.
- A batch trace is a few-shot example when its `input_sha256` equals an example's; its `trace_id` plays no part.
- Prose in code comments, docstrings, error messages, docs and commit messages follows the user's writing rules: plain sentences, no em or en dashes, no "not X but Y".

## Files

- `src/axial/judge.py`: `render_few_shot`, `prompt_sha256`, `load_few_shot` (moved from `measure._read_few_shot`), `current_prompt_sha256`; `run_judge` stamps rows.
- `src/axial/report.py`: `PROMPT_HASH_KEY`, third marker line, stamp check in `promote`, prompt check in `require_promoted`, `PromotedHashes.prompt_sha256`.
- `src/axial/measure.py`: uses `load_few_shot`, re-checks the fingerprint, records it, matches examples by input hash.
- `src/axial/gate.py`: `_load_fresh` compares the fingerprint.
- `tests/test_prompt_pinning.py` (new): unit tests for everything above.
- `tests/promotion_helpers.py` (new): `stamp_run` for tests that promote hand-written runs.
- `tests/test_capabilities_prompt_pinning.py`: drop both expected-failure markers, rewrite the second test.
- Fixtures that promote hand-written runs: `tests/test_report.py`, `tests/test_gate.py`, `tests/test_capabilities_promote_gate.py`, `tests/test_end_to_end.py` (only those the suite shows failing).
- Docs: `README.md`, the corrected-rates spec, the prompt-pinning spec.

---

### Task 1: The fingerprint, and `run_judge` stamping rows

**Files:**
- Modify: `src/axial/judge.py` (`assemble_prompt`, `run_judge`, new functions after `resolve_few_shot`)
- Modify: `src/axial/measure.py` (`_read_few_shot` removed, its call site in `measure_batch`)
- Create: `tests/test_prompt_pinning.py`

**Interfaces:**
- Produces:
  - `render_few_shot(few_shot: list[tuple[Trace, Label]]) -> str`
  - `prompt_sha256(spec: JudgeSpec, few_shot: list[tuple[Trace, Label]]) -> str`
  - `load_few_shot(store: Store, spec: JudgeSpec) -> tuple[list[tuple[Trace, Label]], Splits | None]`: `([], None)` without reading anything when `spec.few_shot_ids` is empty; raises `AxialError("could not read ...")` on an unreadable traces, labels or splits file.
  - `current_prompt_sha256(store: Store, spec: JudgeSpec) -> str`
  - Every row `run_judge` writes carries `"prompt_sha256"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_prompt_pinning.py`:

```python
"""Prompt pinning: a promoted judge is bound to the prompt its verdicts came from.

See docs/superpowers/specs/2026-10-20-prompt-pinning-design.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from measure_helpers import MEASURE_MODE, build_promoted_judge

from axial.dataset import Label, Splits
from axial.judge import (
    JudgeSpec,
    current_prompt_sha256,
    load_few_shot,
    prompt_sha256,
    run_judge,
)
from axial.model import FakeModel
from axial.store import Store
from axial.traces import Trace

TEMPLATE = "Grade it.\n{few_shot}\nINPUT: {input}\nOUTPUT: {output}"


def spec(**kw: object) -> JudgeSpec:
    base: dict[str, object] = {
        "mode": "merged", "prompt_template": TEMPLATE, "model": "fake", "version": 1,
    }
    base.update(kw)
    return JudgeSpec.model_validate(base)


def example(
    i: int = 1, *, failed: bool = True, critique: str = "because",
    question: str | None = None, answer: str | None = None,
) -> tuple[Trace, Label]:
    return (
        Trace(trace_id=f"t{i}", input=question or f"in{i}", output=answer or f"out{i}"),
        Label(trace_id=f"t{i}", label=failed, critique=critique, annotator="andres"),
    )


# --- the fingerprint ---------------------------------------------------------------


def test_the_fingerprint_is_a_sha256_hex_digest():
    fingerprint = prompt_sha256(spec(), [])
    assert len(fingerprint) == 64 and set(fingerprint) <= set("0123456789abcdef")


def test_the_fingerprint_ignores_how_judge_yaml_is_formatted(tmp_path: Path):
    store = Store(tmp_path)
    original = spec(few_shot_ids=["t1"])
    original.save(store)
    path = store.judge_dir("merged", 1) / "judge.yaml"
    path.write_text(yaml.safe_dump(original.model_dump(), default_flow_style=True))
    reloaded = JudgeSpec.load(store, "merged", 1)
    assert prompt_sha256(reloaded, [example()]) == prompt_sha256(original, [example()])


@pytest.mark.parametrize(
    "changed",
    [
        {"prompt_template": "Other.\n{few_shot}\nINPUT: {input}\nOUTPUT: {output}"},
        {"model": "another-model"},
        {"decision_threshold": 0.7},
        {"mode_definition": "a different definition"},
    ],
)
def test_the_fingerprint_changes_with_the_spec(changed: dict[str, object]):
    assert prompt_sha256(spec(**changed), [example()]) != prompt_sha256(spec(), [example()])


@pytest.mark.parametrize(
    "changed",
    [
        example(failed=False),
        example(critique="another reason"),
        example(question="another input"),
        example(answer="another output"),
    ],
)
def test_the_fingerprint_changes_with_a_few_shot_example(changed: tuple[Trace, Label]):
    assert prompt_sha256(spec(), [changed]) != prompt_sha256(spec(), [example()])


def test_the_fingerprint_changes_with_the_order_of_the_examples():
    a, b = example(1), example(2)
    assert prompt_sha256(spec(), [a, b]) != prompt_sha256(spec(), [b, a])


def test_the_fingerprint_survives_a_lone_surrogate_in_a_critique():
    assert len(prompt_sha256(spec(), [example(critique="cut \ud83d")])) == 64


# --- resolving from the store ------------------------------------------------------


def test_current_fingerprint_reads_nothing_for_a_judge_without_examples(tmp_path: Path):
    store = Store(tmp_path)
    store.traces_path.parent.mkdir(parents=True, exist_ok=True)
    store.traces_path.write_text("not json\n")
    assert load_few_shot(store, spec()) == ([], None)
    assert current_prompt_sha256(store, spec()) == prompt_sha256(spec(), [])


def test_current_fingerprint_resolves_examples_from_the_store(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    judge = JudgeSpec.load(store, MEASURE_MODE, 1)
    few_shot, splits = load_few_shot(store, judge)
    assert [t.trace_id for t, _ in few_shot] == ["train-0"]
    assert splits is not None
    assert current_prompt_sha256(store, judge) == prompt_sha256(judge, few_shot)


# --- run_judge stamps its rows -----------------------------------------------------


def test_run_judge_stamps_every_verdict_and_error_row(tmp_path: Path):
    store = Store(tmp_path)
    judge = spec(few_shot_ids=["t1"])
    corpus = [example(1)[0], example(2)[0], example(3)[0]]
    labels = [example(1)[1], example(2, failed=False)[1]]  # t3 has no label: an error row
    splits = Splits(train=["t1"], dev=["t2", "t3"], test=[])
    run_judge(
        store, judge, corpus[1:], labels, splits,
        FakeModel(lambda _p: '{"verdict": "pass"}'), split="dev", corpus=corpus,
    )
    expected = prompt_sha256(judge, [(corpus[0], labels[0])])
    directory = store.judge_dir("merged", 1)
    verdicts = store.read_jsonl(directory / "verdicts.jsonl")
    errors = store.read_jsonl(directory / "errors.jsonl")
    assert len(verdicts) == 1 and len(errors) == 1
    assert {r["prompt_sha256"] for r in verdicts + errors} == {expected}
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_prompt_pinning.py -q`
Expected: collection error, `ImportError: cannot import name 'current_prompt_sha256' from 'axial.judge'`.

- [ ] **Step 3: Implement in `judge.py`**

Add `import hashlib` and `from collections.abc import Callable` to the imports, change `from axial.dataset import Label, Splits` to `from axial.dataset import Label, Splits, read_labels`, and `from axial.errors import InvariantViolation` to `from axial.errors import AxialError, InvariantViolation`.

In `assemble_prompt`, replace the loop that builds `blocks` and the `few_shot_text = "".join(blocks)` line with:

```python
    few_shot_text = render_few_shot(few_shot)
```

Add before `assemble_prompt`:

```python
def render_few_shot(few_shot: list[tuple[Trace, Label]]) -> str:
    """The few-shot block exactly as the prompt carries it.

    `assemble_prompt` and `prompt_sha256` both call this, so the fingerprint is taken
    over the text the model is sent.
    """
    blocks = []
    for example, label in few_shot:
        verdict = "fail" if label.label else "pass"
        blocks.append(
            f"\nEXAMPLE\nINPUT:\n{example.input}\nOUTPUT:\n{example.output}\n"
            f'ANSWER: {{"verdict": "{verdict}", "reason": "{label.critique}"}}\n'
        )
    return "".join(blocks)


def prompt_sha256(spec: JudgeSpec, few_shot: list[tuple[Trace, Label]]) -> str:
    """The fingerprint of everything in a judge's prompt except the trace it grades.

    It covers the spec's fields, not `judge.yaml`'s bytes, so reformatting the file
    leaves it alone, and the rendered few-shot block, so a relabelled or re-ingested
    example changes it. `ensure_ascii` writes a lone surrogate as an escape, which keeps
    the document encodable.
    """
    document = {"spec": spec.model_dump(), "few_shot": render_few_shot(few_shot)}
    text = json.dumps(document, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(text.encode("ascii")).hexdigest()
```

Add after `resolve_few_shot`, moving the body of `measure._read_few_shot` here:

```python
def load_few_shot(
    store: Store, spec: JudgeSpec
) -> tuple[list[tuple[Trace, Label]], Splits | None]:
    """The judge's few-shot examples as the store resolves them now, and the splits its
    prompt needs.

    Read only for a judge that has few-shot ids, so a damaged corpus does not stop a
    judge without examples. For one with examples, an unreadable file is refused here,
    before anything is graded and paid for.
    """
    if not spec.few_shot_ids:
        return [], None

    def read[T](what: str, reader: Callable[[], T]) -> T:
        try:
            return reader()
        except (OSError, ValueError) as exc:  # ValueError covers JSON and pydantic errors
            detail = (str(exc).splitlines() or [""])[0]
            raise AxialError(
                f"could not read {what}, which judge {spec.mode} v{spec.version}'s few-shot "
                f"examples come from: {detail}"
            ) from exc

    corpus = read(f"the traces ({store.traces_path})", store.read_traces)
    where = store.dataset_dir(spec.mode)
    labels = read(f"the labels (in {where})", lambda: read_labels(store, spec.mode))
    splits = read(f"the splits (in {where})", lambda: Splits.load(store, spec.mode))
    return resolve_few_shot(spec, corpus, labels), splits


def current_prompt_sha256(store: Store, spec: JudgeSpec) -> str:
    """The fingerprint of the prompt `spec` sends with the store's traces and labels now."""
    return prompt_sha256(spec, load_few_shot(store, spec)[0])
```

In `run_judge`, after `few_shot = resolve_few_shot(...)`:

```python
    fingerprint = prompt_sha256(spec, few_shot)
```

and add `"prompt_sha256": fingerprint,` as the last key of each of the three row dicts it appends (the set-aside error, the `unlabelled` error, and the verdict).

- [ ] **Step 4: Point `measure` at `load_few_shot`**

In `src/axial/measure.py`, delete `_read_few_shot`, change the import line to `from axial.judge import Graded, JudgeSpec, grade_trace, load_few_shot`, and replace `few_shot, splits = _read_few_shot(store, spec, mode, version)` with:

```python
    few_shot, splits = load_few_shot(store, spec)
```

Drop `read_labels`, `Label` and `Callable` from the measure imports only if ruff reports them unused.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_prompt_pinning.py -q` → all pass.
Run: `uv run pytest -q` → 1030 + the new tests pass, 49 xfailed.

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check && uv run mypy --strict src/axial
git add src/axial/judge.py src/axial/measure.py tests/test_prompt_pinning.py
git -c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com commit -m "Fingerprint a judge's prompt and stamp it on every row run_judge writes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `promote` checks the stamps, `require_promoted` checks the prompt

**Files:**
- Modify: `src/axial/report.py` (`format_promoted_marker`, `PromotedHashes`, `require_promoted`, `promote`)
- Create: `tests/promotion_helpers.py`
- Modify: `tests/test_prompt_pinning.py`, `tests/test_capabilities_prompt_pinning.py` (first test only)
- Modify, as the suite shows: `tests/test_report.py`, `tests/test_gate.py`, `tests/test_capabilities_promote_gate.py`, `tests/test_end_to_end.py`

**Interfaces:**
- Consumes: `current_prompt_sha256(store, spec)`, `JudgeSpec.load`, row key `prompt_sha256` (Task 1).
- Produces:
  - `PROMPT_HASH_KEY = "prompt_sha256"` in `report.py`
  - `format_promoted_marker(dev_summary: str, test_summary: str, judge_yaml_hash: str, verdicts_hash: str, prompt_hash: str) -> str`
  - `PromotedHashes(judge_yaml_sha256: str, verdicts_sha256: str, prompt_sha256: str)`
  - `promote` raises `InvariantViolation` for a missing `judge.yaml` and for any dev or test row whose `prompt_sha256` is missing or differs.
  - `require_promoted` raises `InvariantViolation` for a marker without `prompt_sha256`, for a missing `judge.yaml`, and when `current_prompt_sha256` differs from the marker.
  - `tests/promotion_helpers.py: stamp_run(store: Store, mode: str, version: int) -> None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prompt_pinning.py`. Extend its imports: `judge_reply` joins the `measure_helpers` import, `add_labels` and `read_labels` join the `axial.dataset` import, and these are new:

```python
from axial.config import Thresholds
from axial.errors import InvariantViolation
from axial.report import (
    PROMPT_HASH_KEY,
    alignment_from_verdicts,
    parse_promoted_marker,
    promote,
    require_promoted,
)
```

```python
# --- promote and require_promoted --------------------------------------------------


def judge_dir(store: Store) -> Path:
    return store.judge_dir(MEASURE_MODE, 1)


def promote_again(store: Store) -> None:
    verdicts = store.read_jsonl(judge_dir(store) / "verdicts.jsonl")
    promote(
        store, MEASURE_MODE, 1,
        alignment_from_verdicts(verdicts, "dev"), alignment_from_verdicts(verdicts, "test"),
        Thresholds(),
    )


def rerun(store: Store, split: str) -> None:
    """Re-run one split with the judge.yaml on disk, as `axial judge --split` does."""
    judge = JudgeSpec.load(store, MEASURE_MODE, 1)
    splits = Splits.load(store, MEASURE_MODE)
    corpus = store.read_traces()
    wanted = set(getattr(splits, split))
    run_judge(
        store, judge, [t for t in corpus if t.trace_id in wanted],
        read_labels(store, MEASURE_MODE), splits, FakeModel(judge_reply, model="fake-judge"),
        split=split, corpus=corpus,
    )


def reword(store: Store) -> None:
    judge = JudgeSpec.load(store, MEASURE_MODE, 1)
    judge.prompt_template = "Reworded.\n" + judge.prompt_template
    judge.save(store)


def relabel_train_0(store: Store) -> None:
    add_labels(
        store, MEASURE_MODE,
        [Label(trace_id="train-0", label=False, critique="actually fine", annotator="andres")],
    )


def test_promote_records_the_fingerprint_its_rows_carry(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    marker = parse_promoted_marker((judge_dir(store) / "PROMOTED").read_text())
    expected = current_prompt_sha256(store, JudgeSpec.load(store, MEASURE_MODE, 1))
    assert marker[PROMPT_HASH_KEY] == expected
    rows = store.read_jsonl(judge_dir(store) / "verdicts.jsonl")
    assert {r["prompt_sha256"] for r in rows} == {expected}
    assert require_promoted(store, MEASURE_MODE, 1).prompt_sha256 == expected


@pytest.mark.parametrize("stale", ["dev", "test"])
def test_promote_refuses_a_split_run_by_another_prompt(tmp_path: Path, stale: str):
    store = Store(tmp_path)
    build_promoted_judge(store)
    reword(store)
    rerun(store, "test" if stale == "dev" else "dev")
    with pytest.raises(InvariantViolation, match=f"--split {stale}`") as caught:
        promote_again(store)
    other = "test" if stale == "dev" else "dev"
    assert f"--split {other}`" not in str(caught.value)
    assert not (judge_dir(store) / "PROMOTED").exists()


def test_promote_refuses_a_split_after_a_few_shot_example_was_relabelled(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    relabel_train_0(store)
    rerun(store, "dev")
    with pytest.raises(InvariantViolation, match="--split test`"):
        promote_again(store)


def test_promote_refuses_rows_with_no_fingerprint(tmp_path: Path):
    """Rows written before prompt pinning carry no stamp; both splits must be re-run."""
    store = Store(tmp_path)
    build_promoted_judge(store)
    for name in ("verdicts.jsonl", "errors.jsonl"):
        path = judge_dir(store) / name
        store.write_jsonl(
            path, [{k: v for k, v in r.items() if k != "prompt_sha256"} for r in store.read_jsonl(path)]
        )
    with pytest.raises(InvariantViolation, match="dev and test"):
        promote_again(store)


def test_promote_refuses_a_stale_error_row(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store)
    path = judge_dir(store) / "errors.jsonl"
    store.write_jsonl(
        path,
        store.read_jsonl(path)
        + [{"trace_id": "test-000", "split": "test", "failure_class": "harness",
            "detail": "x", "prompt_sha256": "0" * 64}],
    )
    with pytest.raises(InvariantViolation, match="--split test`"):
        promote_again(store)


def test_promote_refuses_a_judge_with_no_judge_yaml(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store)
    (judge_dir(store) / "judge.yaml").unlink()
    with pytest.raises(InvariantViolation, match="has no judge.yaml"):
        promote_again(store)


def test_require_promoted_refuses_a_marker_without_a_fingerprint(tmp_path: Path):
    """A marker written before prompt pinning."""
    store = Store(tmp_path)
    build_promoted_judge(store)
    marker = judge_dir(store) / "PROMOTED"
    marker.write_text(
        "".join(line + "\n" for line in marker.read_text().splitlines()
                if not line.startswith(PROMPT_HASH_KEY))
    )
    with pytest.raises(InvariantViolation, match="'prompt_sha256'"):
        require_promoted(store, MEASURE_MODE, 1)


def test_require_promoted_refuses_a_relabelled_few_shot_example(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    relabel_train_0(store)
    with pytest.raises(InvariantViolation, match="not the prompt it was promoted with"):
        require_promoted(store, MEASURE_MODE, 1)


def test_require_promoted_refuses_a_reingested_few_shot_example(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    store.write_traces(
        [t.model_copy(update={"output": "SOMETHING ELSE"}) if t.trace_id == "train-0" else t
         for t in store.read_traces()]
    )
    with pytest.raises(InvariantViolation, match="not the prompt it was promoted with"):
        require_promoted(store, MEASURE_MODE, 1)


def test_require_promoted_reads_no_corpus_for_a_judge_without_examples(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store)
    store.traces_path.write_text("not json\n")
    require_promoted(store, MEASURE_MODE, 1)


def test_require_promoted_refuses_a_judge_whose_judge_yaml_was_deleted(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store)
    (judge_dir(store) / "judge.yaml").unlink()
    with pytest.raises(InvariantViolation, match="judge.yaml"):
        require_promoted(store, MEASURE_MODE, 1)
```

In `tests/test_capabilities_prompt_pinning.py`, delete the `@pytest.mark.xfail(...)` decorator above `test_repromoting_after_a_dev_only_rerun_binds_test_rows_from_another_prompt`, and rewrite the module docstring's last paragraph and the test's docstring from "Limit:" to state the behaviour now held (promote refuses test rows from an older prompt).

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_prompt_pinning.py tests/test_capabilities_prompt_pinning.py -q`
Expected: collection error, `ImportError: cannot import name 'PROMPT_HASH_KEY' from 'axial.report'`.

- [ ] **Step 3: Implement in `report.py`**

Add `from axial.judge import JudgeSpec, current_prompt_sha256` to the imports.

Add `PROMPT_HASH_KEY = "prompt_sha256"` after `VERDICTS_HASH_KEY`. Give `format_promoted_marker` a fifth parameter, `prompt_hash: str`, append `f"{PROMPT_HASH_KEY}={prompt_hash}\n"` to the text it returns, and add to its docstring that the third line binds the full prompt, few-shot examples included. Add a third field to `PromotedHashes`:

```python
class PromotedHashes(NamedTuple):
    judge_yaml_sha256: str
    verdicts_sha256: str
    prompt_sha256: str
```

Add above `require_promoted`:

```python
def _load_spec(store: Store, mode: str, version: int) -> JudgeSpec:
    if not (store.judge_dir(mode, version) / "judge.yaml").exists():
        raise InvariantViolation(
            f"judge {mode} v{version} has no judge.yaml, so the prompt its verdicts came "
            f"from cannot be checked. Run `axial judge {mode} --version {version} --split "
            "dev` and `--split test` first."
        )
    return JudgeSpec.load(store, mode, version)


def _splits_from_another_prompt(store: Store, directory: Path, fingerprint: str) -> list[str]:
    """The dev and test splits holding a row not stamped with `fingerprint`."""
    stale: set[str] = set()
    for name in ("verdicts.jsonl", "errors.jsonl"):
        for row in store.read_jsonl(directory / name):
            split = row.get("split")
            if split in ("dev", "test") and row.get(PROMPT_HASH_KEY) != fingerprint:
                stale.add(str(split))
    return sorted(stale)
```

In `require_promoted`, after the verdicts-hash check and before the `return`:

```python
    recorded_prompt_hash = _require_bound_hash(recorded, PROMPT_HASH_KEY, mode, version)
    if current_prompt_sha256(store, _load_spec(store, mode, version)) != recorded_prompt_hash:
        raise InvariantViolation(
            f"judge {mode} v{version} would send a prompt that is not the prompt it was "
            "promoted with: its few-shot examples, resolved from the current traces and "
            "labels, have changed (an example was relabelled or re-ingested). Re-run `axial "
            f"judge {mode} --version {version} --split dev` and `--split test`, then `axial "
            "promote` again."
        )
    return PromotedHashes(recorded_judge_hash, recorded_verdicts_hash, recorded_prompt_hash)
```

Update its docstring: it now also recomputes the prompt from the current corpus and labels.

In `promote`, before `check = check_promotion(...)`:

```python
    directory = store.judge_dir(mode, version)
    fingerprint = current_prompt_sha256(store, _load_spec(store, mode, version))
    stale = _splits_from_another_prompt(store, directory, fingerprint)
    if stale:
        rerun = " and ".join(
            f"`axial judge {mode} --version {version} --split {split}`" for split in stale
        )
        raise InvariantViolation(
            f"judge {mode} v{version} cannot be promoted: its {' and '.join(stale)} rows "
            "were not produced by the prompt its judge.yaml, traces and labels give now. "
            "judge.yaml was edited, or a few-shot example relabelled or re-ingested, after "
            f"they ran. Run {rerun}, then promote."
        )
```

Remove the later `directory = store.judge_dir(mode, version)` line, keep `directory.mkdir(...)`, and pass `fingerprint` as the fifth argument to `format_promoted_marker`.

- [ ] **Step 4: Run the new tests**

Run: `uv run pytest tests/test_prompt_pinning.py tests/test_capabilities_prompt_pinning.py::test_repromoting_after_a_dev_only_rerun_binds_test_rows_from_another_prompt -q`
Expected: all pass.

- [ ] **Step 5: Add the fixture helper and repair hand-written runs**

Create `tests/promotion_helpers.py`:

```python
"""For tests that promote a run they wrote by hand instead of through `run_judge`."""

from __future__ import annotations

from axial.judge import JudgeSpec, current_prompt_sha256
from axial.store import Store


def stamp_run(store: Store, mode: str, version: int) -> None:
    """Write a default judge.yaml if there is none, and stamp every verdict and error row
    with that judge's prompt fingerprint, as `run_judge` would have."""
    directory = store.judge_dir(mode, version)
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "judge.yaml").exists():
        JudgeSpec(mode=mode, model="fake", version=version).save(store)
    fingerprint = current_prompt_sha256(store, JudgeSpec.load(store, mode, version))
    for name in ("verdicts.jsonl", "errors.jsonl"):
        path = directory / name
        if path.exists():
            store.write_jsonl(
                path, [{**row, "prompt_sha256": fingerprint} for row in store.read_jsonl(path)]
            )
```

Run: `uv run pytest -q`. Expect failures in fixtures that promote a hand-written run, or write `PROMOTED` by hand. Repair each:

- A test or fixture that calls `promote` on a run it wrote by hand (`tests/test_report.py`'s promote tests, `promoted_run` in `tests/test_capabilities_promote_gate.py`, `_promote_with_real_hashes` in `tests/test_gate.py`, and any other the run shows): call `stamp_run(store, mode, version)` right before `promote`. Where the test then asserts on `judge.yaml`'s hash, nothing else changes, because `stamp_run` writes `judge.yaml` only when there is none.
- The fixture at the top of `tests/test_gate.py` that writes `PROMOTED` with `format_promoted_marker`: save `JudgeSpec(mode=mode, model="fake", version=version)` first, pass `current_prompt_sha256(store, JudgeSpec.load(store, mode, version))` as the fifth argument, and correct its docstring, which says no judge.yaml is written.
- A test that runs `run_judge` and then promotes, such as `tests/test_end_to_end.py`, needs no change. If one fails, the failure is real: read it before touching the test.

Do not weaken an assertion to make a test pass. Run `uv run pytest -q` until it reports the Task 1 count plus this task's new tests passed, 48 xfailed.

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check && uv run mypy --strict src/axial
git add src/axial/report.py tests/
git -c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com commit -m "Promote only rows from the current prompt, and refuse a judge whose prompt changed

promote refuses dev or test rows stamped with another prompt fingerprint, or none,
and records the fingerprint in PROMOTED. require_promoted recomputes it from the
current traces and labels, so a relabelled few-shot example withdraws the promotion.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `measure` records the fingerprint, `gate --batch` checks it

**Files:**
- Modify: `src/axial/measure.py` (`Measurement`, `measure_batch`)
- Modify: `src/axial/gate.py` (`_load_fresh`)
- Modify: `tests/test_prompt_pinning.py`, `tests/test_capabilities_prompt_pinning.py` (second test)

**Interfaces:**
- Consumes: `prompt_sha256`, `load_few_shot` (Task 1); `require_promoted` returning `PromotedHashes.prompt_sha256` (Task 2).
- Produces: `Measurement.prompt_sha256: str`, after `verdicts_sha256`; `measurement.json` key `"prompt_sha256"`. A `measurement.json` without it fails `load_measurement` with "is not a readable measurement". `_load_fresh` refuses a measurement whose fingerprint is not the marker's.

The spec also asks `gate_batches` to compare the candidate's and baseline's fingerprints. That check cannot fire: `_load_fresh` refuses each batch unless its fingerprint equals the marker's, and the existing version check ensures both batches name the same marker. Skip it; Task 5 records why in the spec.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prompt_pinning.py`, adding imports `import json`, `from axial.errors import AxialError`, `from axial.gate import gate_batches`, `from axial.measure import input_sha256, measure_batch`, and from `measure_helpers` also `batch_traces, fake_judge_builder`:

```python
# --- measure and gate --batch ------------------------------------------------------


def measure(store: Store, name: str, outputs: list[str] | None = None) -> None:
    measure_batch(
        store, MEASURE_MODE, 1, name, batch_traces(outputs or ["BAD"] * 60 + ["GOOD"] * 440),
        fake_judge_builder(), draws=2000,
    )


def measurement_json(store: Store, name: str) -> dict[str, object]:
    path = store.measurement_dir(MEASURE_MODE, name) / "measurement.json"
    data: dict[str, object] = json.loads(path.read_text())
    return data


def test_measurement_json_records_the_promoted_fingerprint(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    measure(store, "rel-1")
    assert (
        measurement_json(store, "rel-1")["prompt_sha256"]
        == require_promoted(store, MEASURE_MODE, 1).prompt_sha256
    )


def test_measure_refuses_after_a_few_shot_example_was_relabelled(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    relabel_train_0(store)
    prompts: list[str] = []
    with pytest.raises(InvariantViolation, match="not the prompt it was promoted with"):
        measure_batch(
            store, MEASURE_MODE, 1, "rel-1", batch_traces(["GOOD"] * 3),
            fake_judge_builder(prompts=prompts), draws=2000,
        )
    assert prompts == []


def test_measure_refuses_examples_that_change_after_the_promotion_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The labels change between `require_promoted` and grading."""
    import axial.measure

    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])

    def then_relabel(*args: object) -> object:
        hashes = require_promoted(*args)
        relabel_train_0(store)
        return hashes

    monkeypatch.setattr(axial.measure, "require_promoted", then_relabel)
    prompts: list[str] = []
    with pytest.raises(InvariantViolation, match="changed while"):
        measure_batch(
            store, MEASURE_MODE, 1, "rel-1", batch_traces(["GOOD"] * 3),
            fake_judge_builder(prompts=prompts), draws=2000,
        )
    assert prompts == []


def test_gate_refuses_a_batch_measured_before_a_relabel(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    measure(store, "rel-1")
    relabel_train_0(store)
    with pytest.raises(InvariantViolation, match="no longer promoted as it was"):
        gate_batches(store, MEASURE_MODE, "rel-1", max_rate=0.5, draws=2000)


def test_gate_refuses_a_measurement_with_another_fingerprint(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store)
    measure(store, "rel-1")
    path = store.measurement_dir(MEASURE_MODE, "rel-1") / "measurement.json"
    data = json.loads(path.read_text())
    data["prompt_sha256"] = "0" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(InvariantViolation, match="measured by an earlier"):
        gate_batches(store, MEASURE_MODE, "rel-1", max_rate=0.5, draws=2000)


def test_gate_refuses_a_measurement_written_before_prompt_pinning(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store)
    measure(store, "rel-1")
    path = store.measurement_dir(MEASURE_MODE, "rel-1") / "measurement.json"
    data = json.loads(path.read_text())
    del data["prompt_sha256"]
    path.write_text(json.dumps(data))
    with pytest.raises(AxialError, match="not a readable measurement.*--replace"):
        gate_batches(store, MEASURE_MODE, "rel-1", max_rate=0.5, draws=2000)
```

Rewrite `test_relabelling_a_few_shot_example_changes_the_measuring_prompt_unrecorded` in `tests/test_capabilities_prompt_pinning.py`. Delete its `xfail` decorator, rename it `test_a_changed_few_shot_example_stops_measure_and_gate_until_the_judge_is_retested`, and replace its body:

```python
    """`PROMOTED` binds the prompt the examples resolve to. Relabel the train example,
    or re-ingest it with another output, and `measure` refuses before grading, and the
    batch measured before the change no longer gates."""
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    batch = batch_traces(["BAD"] * 60 + ["GOOD"] * 440)
    measure_batch(store, MEASURE_MODE, 1, "before", batch, fake_judge_builder(), draws=2000)
    labels_path = store.dataset_dir(MEASURE_MODE)
    original = {p.name: p.read_bytes() for p in labels_path.iterdir() if p.is_file()}

    add_labels(
        store, MEASURE_MODE,
        [Label(trace_id="train-0", label=False, critique="actually fine", annotator="andres")],
    )
    with pytest.raises(AxialError, match="not the prompt it was promoted with"):
        measure_batch(store, MEASURE_MODE, 1, "after", batch, fake_judge_builder(), draws=2000)
    with pytest.raises(AxialError, match="no longer promoted as it was"):
        gate_batches(store, MEASURE_MODE, "before", max_rate=0.5, draws=2000)

    for name, data in original.items():
        (labels_path / name).write_bytes(data)
    store.write_traces(
        [
            Trace(trace_id=t.trace_id, input=t.input, output="SOMETHING ELSE")
            if t.trace_id == "train-0"
            else t
            for t in store.read_traces()
        ]
    )
    with pytest.raises(AxialError, match="not the prompt it was promoted with"):
        measure_batch(store, MEASURE_MODE, 1, "after", batch, fake_judge_builder(), draws=2000)
```

Restoring the labels by bytes isolates the re-ingest case from the relabel. Update the module docstring: both gaps are closed and the tests now hold the behaviour. Remove imports the file no longer uses (`json`, `run_judge` stays for the first test).

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_prompt_pinning.py tests/test_capabilities_prompt_pinning.py -q`
Expected: `KeyError: 'prompt_sha256'` in `test_measurement_json_records_the_promoted_fingerprint`, no refusal in the race test, `test_gate_refuses_a_measurement_with_another_fingerprint` passing gate, the rest already passing through Task 2's `require_promoted`.

- [ ] **Step 3: Implement in `measure.py`**

Add `prompt_sha256: str` to `Measurement` after `verdicts_sha256`; in `to_json` add `"prompt_sha256": self.prompt_sha256,` after `"verdicts_sha256"`; in `from_json` add `prompt_sha256=str(data["prompt_sha256"]),`. Import `InvariantViolation` from `axial.errors` and `prompt_sha256` from `axial.judge`.

In `measure_batch`, after `few_shot, splits = load_few_shot(store, spec)`:

```python
    # require_promoted resolved the examples too; resolve-then-compare here catches
    # labels or traces that changed in between, before anything is paid for.
    if prompt_sha256(spec, few_shot) != hashes.prompt_sha256:
        raise InvariantViolation(
            f"judge {mode} v{version}'s few-shot examples changed while this measurement "
            "was starting. Nothing was graded; run `axial measure` again."
        )
```

Pass `prompt_sha256=hashes.prompt_sha256,` to `Measurement(...)` after `verdicts_sha256`.

- [ ] **Step 4: Implement in `gate.py`**

In `_load_fresh`, compare all three hashes:

```python
    if (hashes.judge_yaml_sha256, hashes.verdicts_sha256, hashes.prompt_sha256) != (
        measurement.judge_yaml_sha256,
        measurement.verdicts_sha256,
        measurement.prompt_sha256,
    ):
        raise InvariantViolation(
            f"batch {batch!r} was measured by an earlier {mode} v{version}: its judge.yaml, "
            "verdicts or prompt changed and it was promoted again since. Re-measure the "
            f"batch with {remeasure}."
        )
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q`. Expect everything to pass with 47 xfailed. A test elsewhere that builds a `Measurement` dict by hand or edits `measurement.json` may now need `prompt_sha256`; add the field from `require_promoted(...).prompt_sha256` rather than loosening the check.

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check && uv run mypy --strict src/axial
git add src/axial/measure.py src/axial/gate.py tests/
git -c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com commit -m "Record the prompt fingerprint in measurement.json and refuse a batch from another prompt

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `measure` recognises a few-shot example by its input

**Files:**
- Modify: `src/axial/measure.py` (`measure_batch`, the few-shot exclusion)
- Modify: `tests/test_prompt_pinning.py`

**Interfaces:**
- Consumes: `input_sha256` (existing, `measure.py`), the resolved `few_shot` list.
- Produces: no new names. A batch trace is set aside as `few_shot_example` exactly when `input_sha256(trace.input)` is the input hash of a resolved example.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prompt_pinning.py`:

```python
# --- few-shot examples inside a batch ----------------------------------------------


def measured_rows(store: Store, name: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    directory = store.measurement_dir(MEASURE_MODE, name)
    return (
        store.read_jsonl(directory / "verdicts.jsonl"),
        store.read_jsonl(directory / "errors.jsonl"),
    )


def test_a_batch_trace_sharing_only_an_id_with_an_example_is_graded(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    collision = Trace(trace_id="train-0", input="an unrelated question", output="BAD")
    measure_batch(
        store, MEASURE_MODE, 1, "rel-1", [collision] + batch_traces(["GOOD"] * 3),
        fake_judge_builder(), draws=2000,
    )
    verdicts, errors = measured_rows(store, "rel-1")
    assert "train-0" in {r["trace_id"] for r in verdicts}
    assert errors == []


def test_an_example_under_another_id_is_set_aside(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    renamed = Trace(trace_id="fresh-1", input="question train-0", output="BAD")
    measure_batch(
        store, MEASURE_MODE, 1, "rel-1", [renamed] + batch_traces(["GOOD"] * 3),
        fake_judge_builder(), draws=2000,
    )
    verdicts, errors = measured_rows(store, "rel-1")
    assert [(e["trace_id"], e["failure_class"]) for e in errors] == [
        ("fresh-1", "few_shot_example")
    ]
    assert errors[0]["input_sha256"] == input_sha256("question train-0")
    assert "fresh-1" not in {r["trace_id"] for r in verdicts}
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_prompt_pinning.py -q -k "sharing_only_an_id or under_another_id"`
Expected: both fail; the first finds `train-0` in the errors, the second finds `fresh-1` graded.

- [ ] **Step 3: Implement**

In `measure_batch`, replace `few_shot_ids = set(spec.few_shot_ids)` with:

```python
    # By input, whatever the id: an example's answer is in the prompt under any id, and
    # a batch id that happens to equal an example's says nothing about its content.
    example_inputs = {input_sha256(example.input) for example, _ in few_shot}
```

and `if trace.trace_id in few_shot_ids:` with `if key in example_inputs:`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q`. Everything passes, 47 xfailed. `tests/test_measure.py::test_a_few_shot_example_in_the_batch_is_set_aside` and `tests/test_gate_batches.py::test_a_batch_with_set_aside_traces_and_a_few_shot_example_still_gates` keep passing: their example repeats both id and input.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check && uv run mypy --strict src/axial
git add src/axial/measure.py tests/test_prompt_pinning.py
git -c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com commit -m "Recognise a few-shot example in a batch by its input, not its id

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Documentation, and the full CI run

**Files:**
- Modify: `README.md` (the "corrected rate comes only from a promoted judge" bullet, around line 96)
- Modify: `docs/superpowers/specs/2026-10-07-corrected-rates-and-disagreements-design.md` ("Changes made during implementation", the entry around line 55)
- Modify: `docs/superpowers/specs/2026-10-20-prompt-pinning-design.md`
- Modify: `README.md` or other docs only where `grep -rn "few-shot\|PROMOTED" README.md docs knowledge llms.txt` shows a now-wrong claim

- [ ] **Step 1: README**

Replace the sentences from "Known limitation, deferred to a follow-up:" to "before measuring." with text that says, in plain sentences: each verdict row records a fingerprint of the full prompt (`judge.yaml` plus the few-shot examples as resolved from the current traces and labels); `axial promote` refuses unless every dev and test row carries the current fingerprint; `measure` and `gate` refuse a judge whose examples have since been relabelled or re-ingested, until dev and test are re-run and it is promoted again; and a judge promoted before this change must re-run both splits and be promoted again, with the three commands from the spec's "Upgrading" section in a `bash` block.

- [ ] **Step 2: Corrected-rates spec**

Rewrite the "Changes made during implementation" entry that describes the limitation so it says the limitation was closed on 2026-10-20 by prompt pinning, with a link to `2026-10-20-prompt-pinning-design.md`. Keep the record of what was true before.

- [ ] **Step 3: Prompt-pinning spec**

Set **Status** to `implemented (plan: docs/superpowers/plans/2026-10-20-prompt-pinning.md)`. In "`require_promoted`, `measure` and `gate`", replace the sentence about `gate_batches` comparing candidate and baseline fingerprints with: that comparison was dropped during implementation because it cannot fire, since `_load_fresh` already refuses each batch whose fingerprint is not the marker's and the version check makes both batches name the same marker. Remove "and a baseline with a different one" from the Testing section and add the race test (examples changing between `require_promoted` and grading).

- [ ] **Step 4: Full CI**

```bash
uv run ruff check
uv run mypy --strict src/axial
uv run pytest -q
uv run python scripts/knowledge_index.py --check
```

Expected: ruff and mypy clean; pytest all passed, 47 xfailed (49 before, minus the two prompt-pinning markers); the index check exits 0. If CI also runs Python 3.13, run `uv run --python 3.13 pytest -q` too.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/
git -c user.name=AndresRubio -c user.email=2109109+AndresRubio@users.noreply.github.com commit -m "Document prompt pinning and retire the known limitation

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
