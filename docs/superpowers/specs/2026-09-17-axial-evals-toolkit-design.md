# axial — an error-analysis-first eval toolkit

**Date:** 2026-09-17
**Status:** approved design, not yet implemented

## Why this exists

Every open-source Python eval tool surveyed — promptfoo, DeepEval, Ragas, autoevals,
Evidently, Inspect AI, lm-evaluation-harness, HELM, LangSmith, Langfuse, Opik, Weave,
Braintrust, Phoenix, DSPy — assumes **you already know your metrics before you open it**.
They are good at the last mile: run a fixed set, score it, gate CI, draw a dashboard.

The first mile is unserved. Deciding *what to measure at all*, from real failures, and
proving an automated judge actually agrees with your humans before you trust it, is taught
as methodology by Hamel Husain and Shreya Shankar and was prototyped once in research
(EvalGen, UIST 2024) — but it has never shipped as maintained, installable OSS. In practice
it is done in spreadsheets and notebooks.

`axial` is that first mile. It takes you from raw traces to a judge whose agreement with
your labels has been measured on a held-out set, and stops there — handing off an artifact
that any existing runner can use.

The name is after axial coding, the step the tool exists to support.

### Prior art worth knowing about

- Anthropic shipped an **error-discovery skill** (2026-08-23) that builds a review app over a
  diverse trace sample and clusters annotations into failure modes. Closest existing thing.
  It is a skill invoked inside an agent session, not a versioned artifact store with
  enforceable invariants and a CI gate; that difference is this project's reason to exist.
- **Braintrust Loop, Arize Alyx, LangSmith Engine** each ship an agent that reads production
  traces and reports failures. Benchmarked by Parlance Labs against 100 human-labelled
  traces from a production leasing agent: the best recovered **87%** of human-flagged
  failures, every system caught issues humans missed, and **all of them systematically
  missed "looked-correct-but-poor-UX" interactions**. They accelerate error analysis; they
  do not remove the human.
- **EvalGen / "Who Validates the Validators?"** (Shankar et al., UIST 2024) established
  criteria drift and demonstrated the judge-alignment loop. Research prototype, unmaintained.
- **Pydantic Evals** (MIT, part of Pydantic AI, usable offline — Logfire is optional) is the
  best-engineered instance of the pattern this project is a reaction to. `Case`, `Dataset`,
  `Evaluator`, an `LLMJudge` graded against a rubric you write, `generate_dataset()` for
  synthesis, and span-based evaluators over OpenTelemetry. Verified absent: any discovery of
  failure modes from traces, any human labelling surface, any train/dev/test split, and any
  judge/human agreement measurement. Its LLM-judge page advises validating judges against
  human-labelled sets in a single sentence of prose, with no API behind it. That sentence is
  this project's scope.

## Sources

Methodology: Hamel Husain's evals FAQ and his LLM-as-judge, field-guide and error-analysis
posts; Shreya Shankar's EvalGen / criteria-drift work and SPADE; Eugene Yan's task-specific
evals, product evals and AlignEval; the Parlance Labs *AI Evals for Engineers & PMs*
curriculum (Husain & Shankar, Oct–Nov 2026); Anthropic's bundled eval guides
(`build-eval`, `eval-audit`, `eval-hillclimb`) and *Demystifying evals for AI agents*.
Full research notes with per-claim attribution are committed alongside this spec, in
`docs/research/2026-09-17-methodology.md` and `docs/research/2026-09-17-tooling-landscape.md`;
the numbers below carry their source inline where a default depends on one.

## Scope

**In scope (v1)** — the spine from a trace pool to a promoted judge:
sample → annotate → taxonomy → label → split → judge → align → promote, plus a thin
`gate` command so a judge can fail a build.

**Out of scope (v1), deliberately:** instrumentation/tracing SDKs (use your existing one),
production drift monitoring, synthetic test-set generation, red-teaming, prompt
optimisation, dashboards. Each is real work covered by the Maven syllabus; each is a
separate spec. A thin `gate` is the single concession, because a judge that cannot fail a
build is a judge nobody reruns.

**Explicit non-goal:** competing with promptfoo / pytest / Langfuse on running fixed suites.
`axial` produces the artifact they consume.

## The invariants

This section is the product. Everything else is plumbing that could be rebuilt; these are
the rules that make the output trustworthy, and they are enforced in code, not documented
as advice. Each one exists because a source names the failure it prevents.

1. **`taxonomy propose` refuses until 30 traces carry human open codes.** Husain sets this
   floor explicitly: open coding is where you discover failures you did not know existed.
   LLMs scale what you learned; they do not replace the looking.
2. **No machine-generated label ever enters dev or test.** A judge is aligned *to* something;
   if that something is machine-generated, the construct is circular. Labels arrive only
   from the annotate app, or via an explicit import that records a human annotator.
3. **The test split is write-once.** Ids and a content hash are frozen in `splits.json`.
   Editing a judge prompt after a test run forces a new judge version rather than silently
   reusing the number.
4. **Few-shot examples are drawn from train only.** Enforced at prompt-assembly time.
5. **`align` never reports raw agreement alone.** TPR and TNR are always printed beside it,
   with a bootstrap confidence interval. At a 5% error rate an always-Pass judge scores 95%
   raw agreement; that number has misled teams into shipping a judge that detects nothing.
6. **`promote` fails** unless test TPR and TNR clear the configured bar and the dev→test gap
   is within tolerance. Defaults: TPR ≥ 0.85, TNR ≥ 0.85, gap ≤ 0.10 — all configurable.
   Honeycomb's judge reached >90% agreement in three iterations, so the bar is reachable.
7. **`gate` compares on a paired bootstrap CI, not a point estimate**, and warns when a suite
   hits 100% — a saturated eval has stopped carrying signal while still costing maintenance.
8. **Infra failures never score as model failures.** They go to `errors.jsonl` with a failure
   class. Truncated responses are marked `truncated`, counted and shown, never averaged in
   as wrong.
9. **A missing required field raises.** No silent zeros: a `cost_usd` of 0.00 or an empty
   `usage` is a runner bug, not a measurement.
10. **Every review batch contains a random slice.** Every targeted sampling method is biased
    toward problems you already know about.

## Architecture

### Artifact tree

All state is plain text under `.axial/` in the user's repo, diffable and reviewable in a PR.

```
.axial/
  config.toml                     # model, thresholds, annotator identity
  traces/traces.jsonl             # normalised; trace_id is the source's stable id, else a content hash
  reviews/<session>.jsonl         # a review batch: `sample` writes one pending row per trace,
                                  #   `annotate` fills in open code, verdict and critique in place
  taxonomy.yaml                   # failure modes: id, name, definition, examples[], prevalence
  datasets/<mode>/
    labels.jsonl                  # trace_id, label, critique, annotator, ts
    splits.json                   # train/dev/test ids, seed, hash
  judges/<mode>/vN/
    judge.yaml                    # prompt template, model, few-shot trace ids
    verdicts.jsonl                # one row per trace
    errors.jsonl                  # quarantined infra failures
    alignment.json                # statistics, machine-readable
    alignment.md                  # the same, for a human
  gates.yaml                      # which judges gate, at what thresholds
```

### Modules

Each is usable as a library without the CLI.

| Module | Owns |
|---|---|
| `traces` | the `Trace` model, ingest from JSONL/CSV, adapters, normalisation |
| `store` | the artifact tree: read, write, version, hash |
| `sample` | sampling strategies; enforces the random slice |
| `annotate` | FastAPI app + one static page — no npm, no build step |
| `taxonomy` | open codes → failure modes, counts, prevalence, saturation curve |
| `dataset` | labels, stratified splits, leakage detection, class balance |
| `judge` | prompt assembly, run harness, retry |
| `align` | **all statistics** — confusion matrix, TPR/TNR, κ, bootstrap CIs, bias checks |
| `gate` | CI comparison and exit code |
| `report` | `alignment.md`, taxonomy summaries |
| `cli` | Typer commands, thin over the above |

Statistics live in exactly one module so nothing else computes a rate.

### Model layer

`litellm` for provider reach, behind a thin `Model` protocol that owns retry with jittered
backoff, cost accounting from the response's own `usage`, caching, and — critically — a
deterministic `FakeModel` for tests. The protocol is what keeps every other module ignorant
of providers. Default judge model `claude-opus-5`, configurable per judge.

Where a call must return JSON — a verdict, a taxonomy proposal — it is requested as a
schema-validated structured output, falling back to a text parse for providers that cannot
enforce one. The fallback is not vestigial: `unparseable` has to remain a real error class,
because a judge that returned prose is a failed call, not a verdict.

The served model is asserted against the requested model on every call; a provider fallback
or capacity reroute fails the attempt loudly rather than quietly scoring a different model.

### Data flow

```
traces.jsonl
   └─ sample ──▶ reviews/<session>.jsonl (pending) ──▶ annotate (human) ──▶ same file, filled in
                                                    │
                          taxonomy.yaml ◀───────────┘  (LLM proposes, human accepts)
                                │
                                ▼
                          labels.jsonl (human) ──▶ splits.json
                                                       │
                                        judge vN ◀─────┘ (few-shot from train only)
                                            │
                                            ├─ align on dev ──▶ iterate prompt
                                            └─ align on test ──▶ promote ──▶ gate
```

## Statistics

`align` reports, per failure mode and per split: confusion matrix; TPR, TNR, precision,
recall, F1; Cohen's κ; bootstrap confidence intervals on each; the dev→test generalisation
gap. Slice breakdowns where trace metadata supports them.

Bias and robustness checks, run on demand: position swap for pairwise judges, verdict/length
correlation for verbosity bias, judge-model swap for self-preference, and context ablation —
drop one context element at a time and recompute alignment, since a judge given only the
segment its failure mode needs typically beats one given everything.

Sizing guidance is surfaced, not enforced: 100+ diverse traces per discovery cycle; 100–200
labels per failure mode with ~60 the floor below which CIs are uselessly wide; 30–50 Pass and
30–50 Fail in each of dev and test; a default split of 15/42/43. The tool prints the CI
half-width for the set you actually have and lets you decide.

## Error handling

Attempts that never produced a scorable output go to `errors.jsonl` with a failure class
(harness error, timeout, rate limit after retries, served-model mismatch, unparseable
output) and never occupy a verdict slot. Rows that did produce output carry `stop_reason`,
and `status: truncated` when the response hit the token cap. Retries are recorded per row so
they can be excluded from latency. A per-case wall-clock ceiling reclaims workers from hung
streams. Grader crashes are `status: error`, not a model failure.

## Worked example

`examples/newsroom/` — a toy story-extraction app (given newsletter text, emit discrete
stories with a topic label), with ~120 committed synthetic traces containing deliberately
planted failure modes: merged stories, hallucinated sources, wrong topic, promotional
content not filtered out.

Paired with the seeded `FakeModel`, the full tutorial and the entire integration test suite
run offline, with no API key and no spend. This keeps the repo honest, but not every README
command is CI-verified the same way: CI runs ruff, mypy and pytest, and the pipeline through
`sample` → `label-from-reviews` → `split` → `judge` → `align` → `promote` is exercised end to
end by `tests/test_end_to_end.py` against this corpus. `checks`, `annotate` and `label-import`
are covered as library-level tests, not as CLI invocations.

The corpus is synthetic and self-contained. It deliberately does not derive from the
ai-digest project's newsletter data, which is private email destined for a public repo.

## Testing

Test-driven, and hermetic by construction.

- **Statistics** get known-answer tests: confusion matrices with hand-computed TPR, TNR and κ,
  and bootstrap CIs checked against a fixed seed.
- **Splits** get property tests: no id appears in two splits, class balance holds within
  tolerance, the same seed reproduces the same split, and a frozen test manifest cannot be
  rewritten.
- **Invariants** get tests that assert the *refusals* — taxonomy proposal below 30 open codes
  raises; a machine label offered to dev raises; few-shot selection from dev raises; promote
  below the bar raises.
- **The pipeline** gets one end-to-end test over `examples/newsroom/` with `FakeModel`.
- **The annotate app** gets request-level tests against the FastAPI app; no browser driver.

## Decisions taken

- Package and CLI named `axial`; repo directory `evals`.
- Python 3.14, managed by uv and pinned in `.python-version`; the published floor is
  `>=3.13` so the package stays installable one version back, and CI tests both.
  3.15 is still at beta, so it is not a target.
- pytest, ruff, mypy. MIT licence.
- `litellm` for model access, per explicit instruction, wrapped in a local protocol.
- FastAPI + a single static HTML page for annotation — long multi-turn traces and JSON
  payloads read badly in a terminal, and annotation throughput is the real bottleneck.
