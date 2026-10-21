# Pinning a promoted judge to the prompt it was tested with

**Date:** 2026-10-20
**Status:** implemented (plan: docs/superpowers/plans/2026-10-20-prompt-pinning.md)
**Builds on:** [`2026-10-07-corrected-rates-and-disagreements-design.md`](2026-10-07-corrected-rates-and-disagreements-design.md)

## Why

`PROMOTED` binds the hashes of `judge.yaml` and `verdicts.jsonl`. That leaves two ways for
the prompt a judge sends to differ from the prompt its test split measured. An adversarial
review found both on 2026-10-20, and `tests/test_capabilities_prompt_pinning.py` records
them as strict expected failures.

1. Test rows from an older prompt. `run_judge` merges verdict rows by split. Edit
   `judge.yaml` in place (as `axial judge --few-shot X` does on the latest version), re-run
   only dev, and `promote` succeeds. The test rows that supply the TPR and TNR every
   corrected rate uses came from the previous prompt.
2. Few-shot examples resolved at grading time. `judge.yaml` stores few-shot ids. Their
   text (input, output, label, critique) is looked up in `.axial/traces` and the labels
   each time the judge grades. Relabel or re-ingest an example after promotion and
   `axial measure` sends a different prompt, while `measurement.json` records nothing
   that says so. `gate --batch --baseline-batch` then compares two batches graded by
   different prompts as if one judge graded both.

A third, smaller gap came out of the corrected-rates final review. `measure` recognises a
few-shot example in a batch by `trace_id` alone. A batch trace whose id happens to match an
example is set aside although it is unrelated, and an example that arrives under another id
is graded with its own answer in the prompt.

## Decisions taken

| Decision | Choice | Why |
|---|---|---|
| What happens when a few-shot example changes after promotion | The judge stops counting as promoted until dev and test are re-run and it is promoted again | Chosen by the user on 2026-10-20 over freezing a snapshot of the examples in the judge directory. The judge always reflects the current labels, and a change to its prompt is never measured with numbers from another prompt. |
| Which rows `promote` checks | Every row in `verdicts.jsonl` and `errors.jsonl`, whatever its split | Asked 2026-10-20, when the answer was dev and test rows, since the promotion check reads both alignments. Widened on 2026-10-21 after the final review found that `axial judge --split train` writes into the same `verdicts.jsonl` and that `gate --version` counts every row whatever its split, so a train run from an older prompt was bound at promotion and compared by the judge-vs-judge gate. A row with no `split` key counts as stale too. |
| How `measure` recognises a few-shot example in a batch | By the SHA-256 of its input text only | Asked 2026-10-20. The answer sits in the prompt whatever id the trace carries, and an id alone says nothing about content. |

## The prompt fingerprint

`judge.prompt_sha256(spec, few_shot)` is the SHA-256 of a canonical JSON document
(`json.dumps(..., sort_keys=True, ensure_ascii=True)`) with two members:

- `spec`: `spec.model_dump()`, every field `judge.yaml` holds, so YAML formatting does not
  change the fingerprint and every field that does change a verdict does;
- `few_shot`: the few-shot block exactly as `assemble_prompt` pastes it into the prompt.

The block's rendering moves out of `assemble_prompt` into a helper, `render_few_shot`, that
both call, so the fingerprint and the prompt cannot drift apart. Together the two members
are the whole prompt apart from the trace under grading. `ensure_ascii` keeps a lone
surrogate in a label from making the document unencodable.

`judge.current_prompt_sha256(store, spec)` resolves the examples from the store's traces and
the mode's labels and returns the fingerprint. A judge with no few-shot ids does not read
the corpus or the labels.

## Stamping rows: `run_judge`

`run_judge` already resolves the examples once per run. It computes the fingerprint from
them and writes it as `prompt_sha256` on every row it produces, verdicts and errors. Merging
by split is unchanged, so rows of a split that was not re-run keep their old stamp.

## `promote`

Before writing the marker, `promote` loads `judge.yaml`, computes `current_prompt_sha256`,
and reads `verdicts.jsonl` and `errors.jsonl`. It refuses with `InvariantViolation` if any
row, whatever its split, has no `prompt_sha256` or a different one. The message names each
such split and the command to re-run it (`axial judge <mode> --version N --split <split>`);
a row with no `split` key is reported as such, with `axial judge --new` as the way out.
Train rows count because `axial judge --split train` writes into the same file and the
judge-vs-judge gate reads every row (decided 2026-10-21, see the table above). When
every row matches, the marker gains a third line, `prompt_sha256=<hex>`, after the two it
already has.

## `require_promoted`, `measure` and `gate`

`require_promoted` reads `prompt_sha256` from the marker with `_require_bound_hash`, so a
marker without it is refused the same way as one without the other hashes. It then
computes `current_prompt_sha256` and refuses when the two differ. The message says the
prompt resolved from the current `judge.yaml`, traces and labels is not the one that was
promoted, that a relabelled or re-ingested few-shot example is the usual cause, and that
re-running dev and test and promoting again fixes it. `PromotedHashes` gains a third
field, `prompt_sha256`.

`measure_batch` computes the fingerprint from the examples it resolved for grading and
refuses if it differs from `hashes.prompt_sha256`. This closes the window between the
check in `require_promoted` and the grading. `Measurement` gains `prompt_sha256`, written
to `measurement.json`.

`_load_fresh` in `gate.py` compares the measurement's `prompt_sha256` with the marker's,
beside the two hashes it compares today. A `measurement.json` with no `prompt_sha256` is
refused with the re-measure command.

`gate_batches` was meant to compare the candidate's and the baseline's fingerprints too.
That comparison was dropped during implementation because it cannot fire: `_load_fresh`
already refuses each batch whose fingerprint is not the marker's, and the version check
makes both batches name the same marker.

After a relabel, then, `measure` refuses before grading anything, and `gate --batch`
refuses every batch the judge measured, until the judge is re-tested and promoted.

## Few-shot examples inside a batch

`measure_batch` sets a batch trace aside as `few_shot_example` when its `input_sha256` is
the input hash of one of the resolved examples. The `trace_id` no longer matters: a batch
trace that shares an id with an example but has different input is graded, and one that
repeats an example's input under another id is set aside.

## Upgrading

A judge promoted before this change has a marker without `prompt_sha256` and rows without
stamps. `measure` and `gate` refuse it, and `promote` refuses until every split it has rows for is re-run, train included:

```bash
axial judge <mode> --version N --split dev
axial judge <mode> --version N --split test
axial judge <mode> --version N --split train   # only if it has train rows
axial promote <mode> N
```

Batches measured before the change are refused by `gate --batch` until they are
re-measured with `--replace`.

## Testing

Tests are written first, and each is seen to fail before the code that passes it.

- `test_capabilities_prompt_pinning.py`: drop the expected-failure marker from the
  dev-only re-run test; it asserts that `promote` refuses and needs no other change. The
  relabelling test is rewritten for this design: `measure` refuses after a relabel, it
  refuses after a re-ingest that changes the example's output, and `gate --batch` refuses
  the batch measured before the relabel.
- The fingerprint does not change when `judge.yaml` is reformatted, and does change with
  the template, the model, the decision threshold, an example's label, critique, input or
  output, and the order of the examples.
- `run_judge` stamps verdict and error rows. `promote` refuses a stale or missing stamp on
  dev, on test and on train, each, and a stale row with no split, and writes the third
  marker line.
- `promote` refuses a judge.yaml that is not valid YAML, or whose values the spec rejects,
  with an `InvariantViolation` that names the file; the CLI exits cleanly with no
  traceback.
- `require_promoted` refuses a marker without `prompt_sha256` and a corpus or labels that
  resolve to another prompt, and does not read the corpus for a judge with no few-shot ids.
- `measurement.json` records the fingerprint. `gate --batch` refuses a measurement without
  one and a measurement whose fingerprint is not the marker's.
- If the examples change between `require_promoted` and grading, `measure` refuses before
  grading anything.
- A batch trace sharing only an id with an example is graded; one repeating an example's
  input under a new id is set aside.

Before each commit: `ruff check`, `mypy --strict src/axial`, `pytest`, and
`scripts/knowledge_index.py --check`.

## Documentation

The README's known limitation became the rule above plus the upgrade steps. The
corrected-rates spec records in "Changes made during implementation" that the limitation
was closed, with a link here.
