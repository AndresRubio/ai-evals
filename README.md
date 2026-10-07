# axial

From raw traces to an LLM judge you have actually measured.

Every tool surveyed here assumes you already know your metrics before you open it.
promptfoo wants an assertion. DeepEval and Ragas ship a catalogue of pre-named metrics.
Pydantic Evals wants a rubric. Inspect AI will run any dataset, agent or model-graded
scorer you give it, but the scorer arrives already written; lm-evaluation-harness
starts from a fixed benchmark whose ground truth is handed to you. The observability
platforms store traces beautifully and then, almost universally, ask you to write a
judge prompt and ship it.

None of the tools surveyed help with the part that decides whether any of it works:
reading real failures, deciding what "good" means for your system, and proving your
automated judge agrees with your humans before you trust it. That is taught as
methodology and done in spreadsheets. `axial` is that step, as code.

## Install

Not yet published to PyPI — `axial` is at 0.1.0, with no release workflow. From a
clone of this repository:

```bash
uv sync
```

Every command below runs as `uv run axial ...` from that clone.

### Credentials

```bash
cp .env.example .env    # then fill in the keys you need
```

`.env` is gitignored, read from the repo root by `axial taxonomy` and `axial judge`,
and never overrides a variable already exported in your shell. `.env.example` lists
which key each model path needs and why. Most of the pipeline — `init`, `ingest`,
`sample`, `annotate`, `label-from-reviews`, `split`, `align`, `promote`, `gate` —
calls no model at all and needs no key.

## The pipeline

Each command's output is the next one's input. Everything lands in `.axial/` as plain
text, so in your own repo, your eval state is reviewable in a pull request like any
other source. (This repo gitignores its own `.axial/` — it's generated fresh by the
example, not committed, so nothing here demonstrates that directly.)

| Command | Consumes | Produces |
|---|---|---|
| `axial ingest` | JSONL or CSV | normalised traces |
| `axial sample` | traces | a review batch (`random`, `outlier`, `longest`, `perplexity`); targeted strategies mix in a random slice (30% by default, `--random-fraction`) |
| `axial annotate` | a batch | open codes, verdicts and critiques |
| `axial taxonomy` | your open codes | a failure taxonomy with prevalence |
| `axial label-from-reviews` | one or more annotated review sessions + a failure mode | human labels, converted from `verdict`/`critique`/`annotator` |
| `axial label-import` | a labels JSONL + a mode name | human labels, for labels made elsewhere |
| `axial split` | labels | a stratified, seeded, frozen split |
| `axial judge` | a split + a prompt | verdicts, and errors kept separate |
| `axial align` | verdicts vs labels | an alignment report |
| `axial checks` | verdicts + traces | slice breakdown, verbosity bias, judged cost, noise floor |
| `axial promote` | alignment on test | a judge cleared for CI |
| `axial gate` | promoted judges | an exit code |

## What it refuses to do

The refusals are the product.

- **No taxonomy before 30 open codes.** Open coding is where you find failure modes
  nobody anticipated. A model cannot discover what it was not shown. This floor counts
  open codes; it does not check who wrote them — that check applies to labels, below,
  not to open codes.
- **Refuses annotators that look like model identifiers.** A name-pattern check
  catches the obvious cases — a model's own name, "bot", "synthetic" — before a
  label enters dev or test, because a judge aligned to machine labels is aligned to
  itself. It is a speed bump, not a security boundary: it cannot stop someone
  determined to record machine output under a human name.
- **The test split is write-once.** Redrawing it invalidates every number reported
  against it, so `axial` will not do it quietly.
- **Few-shot examples come from train only.**
- **Never raw agreement alone.** TPR and TNR are printed beside it, with a confidence
  interval. The positive class is the failure: TPR is the share of real failures the
  judge catches, TNR the share of real passes it lets through. At a 5% failure rate an always-pass judge scores 95% agreement and detects
  nothing.
- **No promotion below the bar.** Test TPR and TNR must clear your threshold and the
  dev→test gap must be within tolerance.
- **Gates compare intervals, not point estimates — when a `--baseline` is given.**
  Without one, `gate` only prints the failure rate and always passes; pass a
  baseline version to get a paired confidence interval and an actual regression
  check. Either way, it warns when a suite stops failing anything — a saturated
  eval has stopped carrying signal.
- **Infra failures never score as model failures.** They are quarantined in
  `errors.jsonl` with a failure class.
- **Perplexity is never a verdict.** It selects which traces a human reads and stops
  there. A fluent, confident, entirely fabricated answer has *low* perplexity, so the
  failure mode you care most about is the one it scores best. `axial sample` also
  reports the spread of its own ranking and warns when that ranking is arbitrary,
  rather than handing back a batch that looks considered and is not.
- **No floating model aliases.** A judge model must be pinned to an exact version.
  An alias that moves swaps the judge underneath a `PROMOTED` marker whose hashes
  still match, so the gate keeps attesting to an alignment run nobody performed.

## Quickstart

`init`, `ingest`, `sample`, and `annotate` run entirely offline against the
committed example corpus — no API key, no spend.

```bash
uv run axial init
uv run axial ingest examples/newsroom/traces.jsonl
uv run axial sample --n 40 --name first-pass
```

Uncomment and set `annotator = "your-name"` near the top of `.axial/config.toml`
(every label records who made it), then:

```bash
uv run axial annotate --name first-pass     # annotate 30+, then stop
```

`axial taxonomy` is the first command that calls a model, and therefore costs
money: it sends your open codes to the judge model configured in
`.axial/config.toml` (`claude-opus-5` by default), so you need working credentials
for it from here on.

```bash
uv run axial taxonomy
```

Pick one failure mode id from the taxonomy it just proposed (the newsroom corpus
plants `merged_stories`, `hallucinated_sources`, `wrong_topic` and
`promotional_content`, so it is likely one of those), then carry your own
annotations the rest of the way to a promoted, gate-ready judge:

```bash
uv run axial label-from-reviews merged_stories first-pass
uv run axial split merged_stories
uv run axial judge merged_stories --split dev
# iterate on .axial/judges/merged_stories/v1/judge.yaml, re-running `judge --split dev`
# and `axial align merged_stories 1` until dev alignment looks right, then:
uv run axial judge merged_stories --split test
uv run axial align merged_stories 1
uv run axial promote merged_stories 1
uv run axial gate merged_stories --version 1
```

`label-from-reviews` is what turns the `annotate` app's output into ground truth —
every completed review already carries a verdict, a critique and an annotator, so
there is no separate hand-built file to write. See
[`examples/newsroom/`](examples/newsroom/) for the fully offline, `FakeModel`-backed
version of this same walkthrough.

## Decision models

A judge answers one binary question per trace, which is what a *decision model* does
natively — and for a fraction of the cost. `axial` speaks to
[TypeSafe's Jev](https://openrouter.ai/typesafe/jev-1.13) through OpenRouter's
Decisions API:

```toml
# .axial/config.toml
judge_model = "typesafe/jev-1.13"   # pinned; an alias is refused
```

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
axial judge merged_stories --split dev
```

Jev is not reachable through litellm — it runs on `POST /api/alpha/decisions`, not the
chat-completions endpoint — so `JevModel` speaks that API directly. It asks one `noul`
(yes/no probability) question per trace, using the prompt `assemble_prompt` built as the
state, and thresholds the calibrated probability into the same verdict JSON an LLM judge
emits. `run_judge`, `align`, `promote` and `gate` are unchanged. Tune the threshold in
`judge.yaml` (`decision_threshold`, default `0.5`) and bump the judge version when you do —
it changes every verdict, which is why it lives in the hashed judge definition.

Two things it cannot do:

- **No critique.** Jev returns a number, so the recorded reason is the probability and the
  threshold. That is enough to audit a gate and not enough to run an alignment pass, where
  you need to read *why* the judge and a human disagree. Align with a text model; consider
  Jev for the high-volume gate afterwards, and only if its own TPR/TNR clear the bar.
- **No prose.** `axial taxonomy` refuses a decision model outright rather than proposing an
  empty taxonomy in silence.

Cheap labels are also the fastest route to breaking the second invariant. At roughly
$0.0004 a decision it becomes tempting to label the whole corpus — but a machine label in
dev or test means your alignment numbers measure the judge against itself. Machine labels
belong in triage and in train, nowhere else.

The adapter is exercised offline against a stubbed transport. It has not been run against
the live endpoint here.

## Perplexity

Perplexity is the exponentiated average negative log-likelihood a language model
assigns to a text — `exp(-1/N · Σ log p(xᵢ | x<ᵢ))`. It is the oldest metric in language
modelling, and it has exactly one job here: **deciding which traces a human reads.**

```bash
axial sample --n 40 --name first-pass --strategy perplexity
```

Traces are ranked most-surprising-first, because the failures worth finding sit in the
edge cases the corpus barely covers, not in the hundredth paraphrase of the commonest
question. `--perplexity-field` chooses the side to rank on (`input`, the default,
`output`, or `both`).

**It is not a quality metric, and the reason is not academic.** Perplexity measures how
*expected* a text was, not whether it is correct, grounded or useful. A fluent,
confident, completely fabricated answer is low-perplexity text. The literature is
consistent that perplexity correlates poorly with downstream task performance, degrades
as inputs get longer, and cannot be compared across tokenisers or corpora. So it never
becomes a verdict, a label or an input to alignment — invariant 2 already forbids a
machine score from becoming a label in dev or test, and this is a machine score.

**The model is fitted to your corpus, not pretrained.** The question is "unusual
compared to *my* production traffic", so your traffic is the reference distribution.
That also keeps `sample` free of an API key, a download and a torch dependency — and
the default judge model, Anthropic's, exposes no token logprobs at all, so a pretrained
reference could not be reached through the path axial already uses. The unit is
characters, via a Witten-Bell–interpolated n-gram model, so there is no vocabulary to
build and no unknown-word case to fudge on code, identifiers or other languages.

Two properties worth understanding before trusting a batch:

- **Scoring is leave-one-out, and that is load-bearing.** A trace left in the model that
  scores it teaches that model its own rare sequences. In a 7-trace corpus, a
  deliberately strange trace scored 199 held out and 1.21 left in — second place,
  indistinguishable from ordinary text. Self-inclusion does not flatten the ranking, it
  inverts it, so each trace's counts are removed before it is scored.
- **A flat corpus has nothing to rank, and `sample` says so.** It prints the spread
  between the most surprising trace and the typical one, and warns below 2.0:

  ```
  perplexity spread (max/median): 1.06
  warning: a spread of 1.06 means the most surprising trace is barely more surprising
  than the typical one, so this ranking is close to arbitrary — it will be sorting on
  whatever incidental token happens to vary, such as an id or a timestamp.
  ```

  That is this repo's own `examples/newsroom`: 400 traces that are 25 templates with an
  issue number substituted in, where the ranking was sorting on the issue number. On a
  varied support corpus with three planted oddities — an SQL injection string, an Arabic
  request, a raw JSON blob — the same statistic reads about 4,000 and the three rank
  first.
  The 2.0 threshold is a heuristic sitting in the wide empty gap between those two
  measurements, not a calibrated boundary.

## Prior art

- **promptfoo, DeepEval, Ragas, autoevals** — assertion and metric catalogues. `axial`
  produces a validated judge definition (`judge.yaml`: prompt template, model, few-shot
  trace ids, decision threshold) and the alignment report behind it. There is no
  exporter to any of these tools; to run the judge inside one, you port the prompt and
  threshold by hand, and the alignment numbers only carry over if the model and prompt
  do too.
- **Pydantic Evals** — the best-engineered instance of the pattern, and the most
  type-rigorous eval library in the ecosystem: typed `Case` and `Dataset`, an
  `LLMJudge` graded against a rubric you write, MIT, no account needed. Its own docs tell
  you to validate judges against human-labelled test sets, in one sentence, with nothing
  behind it. `axial` is that sentence, implemented.
- **Inspect AI** (UK AI Security Institute) — a full open harness for model and agent
  evals: custom datasets, tool use, sandboxed agents, `model_graded_qa` and custom
  scorers, a local log viewer, MIT. It runs a model-graded scorer, and its scoring
  workflow lets you re-score logs and hand-edit individual scores; its scorer and
  scoring-workflow docs say nothing about deriving a scorer from your own failures or
  measuring its agreement with human labels. `axial` sits upstream of it: the judge
  Inspect would run is the thing `axial` builds and validates.
- **lm-evaluation-harness** — foundation-model benchmarking, where the taxonomy is the
  benchmark and is given to you. Different problem.
- **Langfuse, Opik, LangSmith, Braintrust, Phoenix** — trace storage and judge running;
  Braintrust itself is closed-source SaaS built around the open `autoevals` scorer
  library, and Phoenix ships under Elastic License 2.0, which isn't OSI-approved.
  Langfuse and Opik are the ones with side-by-side human annotation queues, the closest
  any of the five get; none of them scaffold judge/human alignment as a gated step.
- **EvalGen** (Shankar et al., UIST 2024) — demonstrated the judge-alignment step
  specifically: generate candidate judge prompts, use human grades on a subset to select
  among them, and named criteria drift along the way. A research prototype, not a
  maintained package, and scoped to that one step — not the open coding, taxonomy
  building, splits, or gating around it. `axial` is the shipped, full-pipeline version.
- **The AI Evals course's `evals-skills`**
  ([ai-evals-course/evals-skills](https://github.com/ai-evals-course/evals-skills),
  Apache-2.0) — agent skills for the same methodology. Its `error-discovery` skill has
  an agent build a review app, sort your free-text notes into failure modes while you
  review, and propose new samples; `validate-evaluator` walks the split / TPR / TNR /
  bias-correction steps as instructions. `axial` is the versioned artifact store and
  CI gate around the same ideas. Note the opposite convention: the skills treat *Pass*
  as the positive class, so their TPR is `axial`'s TNR.

## Credit

The methodology is Hamel Husain's and Shreya Shankar's, with Eugene Yan's work on judge
alignment and Anthropic's published eval guides. `axial` implements it; it did not
invent it. See [`docs/research/`](docs/research/) for the sources behind every default.

## Licence

MIT.
