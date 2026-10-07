# Knowledge base layout

**Date:** 2026-10-07
**Status:** approved design, not yet implemented
**Builds on:** [`2026-09-17-axial-evals-toolkit-design.md`](2026-09-17-axial-evals-toolkit-design.md)

## Why

This repository has two jobs. The first is `axial`, a Python library and CLI that other
projects import. The second is a knowledge base: everything read about evaluating LLM
applications, from methodology to the tools that run it, kept in one place where an
agent or a person can query it without opening a browser.

Today the second job is two files under `docs/research/`. They are good, and they are
not queryable: a question about one tool returns a survey of eighteen; a reader cannot
tell which claims are stable and which decayed; the only index is a README sentence.
The 2026-10-07 correction to the Inspect AI entry shows the cost. The survey's first
version mis-filed Inspect, nothing flagged it, and the fix landed as an inline
parenthesis three weeks later.

The shape agents already query well is a plain-text index pointing at plain-text files,
each with structured metadata. Evidently publishes its documentation that way as
`llms.txt`. Skills, memory files and Warp's factory definitions all use Markdown with
YAML frontmatter and validate it in CI. This design adopts that shape.

## Decisions taken

| Decision | Choice | Why |
|---|---|---|
| Where it lives | `knowledge/` at the repo root | Asked 2026-10-07. Keeps knowledge-for-readers apart from `docs/superpowers`, which holds design for contributors. |
| Index | `llms.txt` at the repo root, generated from frontmatter, checked in CI | Asked 2026-10-07. A hand-written index drifts. The Warp examples repo validates its tree the same way. |
| The existing landscape doc | Split into one file per tool plus one overview | Asked 2026-10-07. A query for one tool should return one tool. |
| Other tools' code | Depend on them, never vendor | Asked 2026-10-07. Vendoring makes the repo a maintenance and licence burden; the knowledge base explains each tool and a landscape entry names the extra that depends on it. |
| Shelves | Three fixed: `methodology`, `procedures`, `landscape` | They decay at different rates. A fourth shelf is a spec change, not a directory. |
| File granularity | One topic per file | What an agent fetches after reading the index is one file. |
| Query mechanism | None beyond the index | Fetch the index, pick, fetch the file. No search command, no embeddings, no server. |

## Scope

In:

1. The `knowledge/` tree, its three shelves, and the frontmatter schema every file
   carries.
2. `scripts/knowledge_index.py`: builds `llms.txt` from frontmatter and, with
   `--check`, validates the tree and fails when the committed index is stale.
3. A CI job that runs the check.
4. Migration of `docs/research/` into the shelves, and the first entries the research
   in this conversation produced: Warp Factories, Great Expectations, an updated
   Evidently entry, and four procedures.
5. README and `knowledge/README.md` text.

Out, each a possible later change:

- A `SKILL.md` wrapper so Claude Code, Codex or a Warp agent can load the index as a
  skill. One file, written once the tree is stable.
- Any adapter code for the tools the landscape describes. Each is its own spec and its
  own optional extra; this design only reserves the `extra` frontmatter field that
  names it.
- A public-API spec for the library half of the repo.
- Splitting the methodology spine into one file per step. The spine is a sequence and
  reads as one; the layout allows the split later without a schema change.
- Per-entry staleness thresholds. The index prints every entry's date; the reader
  decides.

## Conventions

- A file's `name` equals its filename stem, kebab-case, and is unique across all
  shelves. A file's `shelf` equals its parent directory. The check enforces both, so
  moving a file means editing its frontmatter, as in the Warp layout rules.
- Dates are ISO `YYYY-MM-DD`.
- Prose follows the existing research docs: claims cite a primary source; a correction
  stays in the file as a dated note rather than silently overwriting the claim; an
  unverified secondary source is marked as such.
- The failure is still the positive class wherever a file discusses TPR and TNR.

## 1. Tree

```
llms.txt                         generated; the query surface
knowledge/
  README.md                      the rules below, in reader-facing form
  methodology/                   how practitioners say to do evals; stable
    methodology-spine.md
    numbers-and-rules-of-thumb.md
    points-of-disagreement.md
    mechanizable-vs-human.md
  procedures/                    how to run one loop end to end; cites tools
    factory-lifecycle.md
    evals-in-ci.md
    production-monitoring.md
    data-contracts-at-ingest.md
  landscape/                     one file per tool; decays; dated
    landscape-overview.md
    promptfoo.md
    inspect-ai.md
    ...                          one per tool in section 5
scripts/
  knowledge_index.py
docs/
  superpowers/                   unchanged
```

`docs/research/` is deleted once migrated. Nothing else may live under `knowledge/`
except `README.md` and the three shelf directories; the check fails on any other path.

### What each shelf holds

- **methodology**: what the practitioners whose work `axial` implements say to do, and
  where they disagree. Sources are their writing. These files change when the
  practice changes, which is rarely.
- **procedures**: a loop a team runs end to end, written as steps with triggers,
  artifacts and human gates. A procedure cites landscape entries for the tools it uses
  and methodology entries for the numbers it quotes. It changes when a better way to run
  the loop appears.
- **landscape**: one tool per file. The claims are about a moving target, so every file
  carries the date its sources were read and the index shows it.

## 2. Frontmatter schema

Every file under a shelf begins with YAML frontmatter. Required keys:

```yaml
---
name: evidently
description: Open-source ML and LLM evaluation and monitoring; tests, LLM judges, tracing, scheduled evals
shelf: landscape
updated: 2026-10-07
sources:
  - url: https://docs.evidentlyai.com/llms.txt
    read: 2026-10-07
  - url: https://github.com/evidentlyai/evidently
    read: 2026-09-17
---
```

| Key | Rule |
|---|---|
| `name` | Equals the filename stem. Lowercase letters, digits and hyphens. Unique across shelves. |
| `description` | One line, at most 160 characters, no trailing period. This is the index line, so it must say what the file answers, not what it is about. |
| `shelf` | One of `methodology`, `procedures`, `landscape`. Equals the parent directory. |
| `updated` | Date of the last substantive change to the body. Typo fixes do not move it. |
| `sources` | Non-empty list of `{url, read}`. `read` is the date the source was read for the claims this file makes. |

Optional keys:

| Key | Rule |
|---|---|
| `related` | List of `name`s of other files. Each must exist. Rendered nowhere; it is for agents following links without parsing prose. |
| `extra` | Landscape only. The name of the optional-dependency group in `pyproject.toml` through which `axial` depends on this tool. Must exist under `[project.optional-dependencies]`. Absent means `axial` does not depend on the tool. |
| `license` | Landscape only. The tool's licence as an SPDX identifier or, for closed products, `proprietary`. |

Body links between knowledge files are relative Markdown links, `../landscape/evidently.md`,
so GitHub renders them. The check resolves every relative link in a body and fails on a
missing target. External links are not fetched.

### Landscape entry body

Each landscape file except `landscape-overview` has the same headings, in this order,
so an agent can read one section of any tool. A heading with nothing to say keeps a one-line "Not applicable" so
the structure stays constant.

1. **What it is.** One paragraph.
2. **Core abstraction.** What you write to use it.
3. **Where it sits.** Benchmarking, application evals, monitoring, data quality, or
   agent-platform scorer. Local-first or account-required.
4. **Maturity.** As of the read date. Figures that fluctuate, such as star counts, are
   marked directional.
5. **Biggest complaint.** From practitioners, with a source.
6. **Relation to axial.** What `axial` takes from it, what it lacks that `axial`
   provides, and the dependency, if any, by extra name.
7. **Corrections.** Dated notes, newest last. Empty until needed.

## 3. The index

`llms.txt` follows the llms.txt convention: an H1, a blockquote summary, one H2 per
shelf in the fixed order methodology, procedures, landscape, and one bullet per file,
sorted by `name` within a shelf.

```markdown
# axial knowledge base

> Everything this repository has read about evaluating LLM applications, as one file
> per topic. Fetch a file by its path. Generated by scripts/knowledge_index.py; edit the
> files, not this index.

## methodology

- [methodology-spine](knowledge/methodology/methodology-spine.md): The thirteen-step error-analysis-first loop from instrumentation to production monitoring (2026-09-17)

## procedures

- [evals-in-ci](knowledge/procedures/evals-in-ci.md): Gate a pull request on a promoted judge and schedule a batch measurement (2026-10-07)

## landscape

- [evidently](knowledge/landscape/evidently.md): Open-source ML and LLM evaluation and monitoring; tests, LLM judges, tracing, scheduled evals (2026-10-07)
```

The date in parentheses is `updated`. It appears on every line so an agent choosing
between two entries sees their ages without fetching either.

## 4. `scripts/knowledge_index.py`

A standalone script, not part of the `axial` package, so the wheel does not ship it.
It depends only on the standard library and PyYAML, which `axial` already requires.

```
python scripts/knowledge_index.py            # rewrite llms.txt
python scripts/knowledge_index.py --check    # validate; exit 1 on any violation
```

Functions, each testable on a temporary directory:

- `parse_entry(path) -> Entry`: splits frontmatter from body, validates the schema
  above, returns a dataclass or raises `KnowledgeError` listing every schema violation
  in that file.
- `load_tree(root) -> list[Entry]`: walks `knowledge/`, parses every file, checks
  cross-file rules: unique names, `related` targets exist, relative body links resolve,
  `extra` names exist in `pyproject.toml`, no stray paths.
- `render_index(entries) -> str`: the Markdown in section 3.
- `check(root) -> list[str]`: runs `load_tree`, renders, compares to the committed
  `llms.txt` byte for byte, and returns every violation found. The script prints them
  one per line as `path: message` and exits 1 if the list is non-empty.

The check collects every violation before exiting rather than stopping at the first, so
one CI run shows the whole list.

A CI job, `knowledge`, runs `uv run python scripts/knowledge_index.py --check` beside
the existing `test` and `compat` jobs.

## 5. Initial content

### Migration of `docs/research/`

- `2026-09-17-methodology.md` splits at its H2s into the four methodology files in the
  tree above. Its `## 5. Sources` section is distributed: each file's frontmatter lists
  the sources its body cites. `updated` is `2026-09-17` for all four, since the split
  changes no claim.
- `2026-09-17-tooling-landscape.md` splits into one landscape file per table row plus
  the Pydantic Evals addendum and the evals-skills note, and
  `landscape-overview.md` keeps "Two different problems" and the gap synthesis. The
  2026-10-07 Inspect correction moves into `inspect-ai.md`'s Corrections section with
  its date. `updated` is `2026-09-17` for rows unchanged since the survey and
  `2026-10-07` for Inspect, Pydantic Evals, evals-skills and Evidently, which is
  rewritten below.

Landscape files from the survey: `promptfoo`, `inspect-ai`, `deepeval`, `ragas`,
`openai-evals`, `braintrust-autoevals`, `langsmith`, `wandb-weave`, `arize-phoenix`,
`helm`, `lm-evaluation-harness`, `anthropic-cookbook`, `pytest-evals`, `dspy`,
`langfuse`, `opik`, `evidently`, `pydantic-evals`, `evals-skills`, `evalgen`.

### New landscape entries

- `warp-factories`: the agent-platform scorer. A factory definition declares scorers
  as Markdown with labels, scores, a sampling rate, a judge model and a self-improvement
  flag that feeds verdicts back into the agent. Nothing validates the scorer before
  self-improvement turns on, and the published example pins its scorer to an undated
  model alias. Sources: the
  `warpdotdev/warp-factory-examples` repository and its `02-sdlc-issue-to-pr` scorers,
  read 2026-10-07.
- `great-expectations`: the data-quality analogue of `axial gate`. Expectation suites
  are the contract, checkpoints run in the orchestrator before and after a step and
  stop the run, Data Docs are the human-readable record. Source: greatexpectations.io,
  read 2026-10-07.
- `evidently`: rewritten from the survey row with the docs read 2026-10-07: Reports
  versus Tests, descriptors, LLM judges, Tracely tracing, datasets, GitHub Actions,
  scheduled evals and batch monitoring. Relation to axial: `axial` produces the judge
  Evidently would run; a descriptor writer is a possible extra, not yet built.

### Procedures

Each is written as numbered steps, each step naming its trigger, its artifact and
whether a human gates it, in the form the factory lifecycle takes.

- `factory-lifecycle`: issue to merged PR as the Warp examples define it. Intake from
  Slack, Linear or GitHub; triage; spec with human approval; implement; adversarial
  review on a different model; UI verification; human merge; closed-loop automations
  for merged PRs and CI failures; scorers sampling the agents. Cites `warp-factories`.
- `evals-in-ci`: Husain's CI profile from the methodology, with deterministic
  assertions preferred over judges for cost, pass@k versus pass^k, saturation; then
  the `axial gate` form of it, including the `--batch` mode the corrected-rates spec
  adds, and Evidently's GitHub Actions form. Cites `methodology-spine`,
  `numbers-and-rules-of-thumb`, `evidently`.
- `production-monitoring`: step 12 of the spine as a procedure. Sample live traces,
  measure with a promoted judge, alert on the interval's lower bound, promote flagged
  traces into the review queue, re-check the judge's rates on a labelled slice. Cites
  `evidently` for the scheduled and batch forms and `warp-factories` for alert-to-ticket.
- `data-contracts-at-ingest`: expectation, checkpoint and data docs as Great
  Expectations defines them, mapped to contract, gate and alignment report in `axial`,
  with the checks `axial ingest` could run. Cites `great-expectations`.

## 6. Error handling

- A file with malformed YAML, or a body with no closing frontmatter fence, is one
  violation naming the file; the walk continues.
- Every schema rule in section 2 is its own violation message with the key it concerns,
  so a file with three problems reports three lines.
- A stale `llms.txt` reports one line: `llms.txt: stale; run scripts/knowledge_index.py`.
- Without `--check`, the script validates first and, when the tree has violations,
  prints the same messages, exits 1 and leaves `llms.txt` untouched. A broken tree
  never produces an index.

## 7. Testing

`tests/test_knowledge_index.py`, importing the script by path. The repository's
`testpaths` already covers `tests/`.

- `parse_entry` accepts the example in section 2 and rejects, each in its own test:
  a missing required key; a `name` not matching the stem; a `shelf` not matching the
  directory; a two-line or over-long `description`; a non-ISO date; an empty `sources`
  list; a source without `read`.
- `load_tree` rejects duplicate names across shelves, a `related` name that does not
  exist, a relative body link to a missing file, an `extra` absent from a given
  `pyproject.toml`, and a stray file at `knowledge/notes.md`.
- `render_index` produces the section 3 output for a three-entry fixture, shelves in
  fixed order, names sorted within a shelf.
- `check` returns an empty list on a fixture whose `llms.txt` matches and one
  violation when a byte differs.
- One test runs `check` over the repository's real `knowledge/` tree and asserts no
  violations. This is the test that keeps the committed knowledge valid on every CI
  run, independent of the `knowledge` job.

## 8. Documentation

- README: replace the `docs/research/` sentence under Credit with a short "Knowledge
  base" section: what `llms.txt` is, that agents and people query it by fetching the
  index then a file, and the one rule for contributors, which is to edit a file and run
  the script.
- `knowledge/README.md`: the three shelves and what belongs in each, the frontmatter
  schema, the landscape heading order, and the two commands. It is the contributor's
  version of this spec and must not drift from it; when the schema changes, both change
  in the same commit.
- The tooling-landscape addendum's note on the evals-skills misattribution survives in
  `evals-skills.md`'s Corrections section with its dates.
