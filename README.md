# axial

From raw traces to an LLM judge you have actually measured.

Every tool surveyed here assumes you already know your metrics before you open it.
promptfoo wants an assertion. DeepEval and Ragas ship a catalogue of pre-named metrics.
Pydantic Evals wants a rubric. Inspect AI and lm-evaluation-harness solve a different
problem — a fixed benchmark whose ground truth is handed to you. The observability
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

## The pipeline

Each command's output is the next one's input. Everything lands in `.axial/` as plain
text, so in your own repo, your eval state is reviewable in a pull request like any
other source. (This repo gitignores its own `.axial/` — it's generated fresh by the
example, not committed, so nothing here demonstrates that directly.)

| Command | Consumes | Produces |
|---|---|---|
| `axial ingest` | JSONL or CSV | normalised traces |
| `axial sample` | traces | a review batch; targeted strategies mix in a random slice (30% by default, `--random-fraction`) |
| `axial annotate` | a batch | open codes, verdicts and critiques |
| `axial taxonomy` | your open codes | a failure taxonomy with prevalence |
| `axial label-import` | a labels JSONL + a mode name | human labels |
| `axial split` | labels | a stratified, seeded, frozen split |
| `axial judge` | a split + a prompt | verdicts, and errors kept separate |
| `axial align` | verdicts vs labels | an alignment report |
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
  interval. At a 5% failure rate an always-pass judge scores 95% agreement and detects
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

See [`examples/newsroom/`](examples/newsroom/) for the rest of the walkthrough.

## Prior art

- **promptfoo, DeepEval, Ragas, autoevals** — assertion and metric catalogues. `axial`
  produces a validated judge definition (`judge.yaml`) and the alignment report
  behind it, which you then wire into whichever of these you run.
- **Pydantic Evals** — the best-engineered instance of the pattern, and the most
  type-rigorous eval library in the ecosystem: typed `Case` and `Dataset`, an
  `LLMJudge` graded against a rubric you write, MIT, no account needed. Its own docs tell
  you to validate judges against human-labelled test sets, in one sentence, with nothing
  behind it. `axial` is that sentence, implemented.
- **Inspect AI, lm-evaluation-harness** — foundation-model benchmarking, where the
  taxonomy is the benchmark and is given to you. Different problem.
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
- **Anthropic's error-discovery skill** — clusters your annotations into failure modes
  inside an agent session. `axial` is the versioned artifact store and CI gate around
  the same idea. (Secondary source, not verified directly against Anthropic's own
  materials — see the "Other sources" note in
  [`docs/research/2026-09-17-tooling-landscape.md`](docs/research/2026-09-17-tooling-landscape.md).)

## Credit

The methodology is Hamel Husain's and Shreya Shankar's, with Eugene Yan's work on judge
alignment and Anthropic's published eval guides. `axial` implements it; it did not
invent it. See [`docs/research/`](docs/research/) for the sources behind every default.

## Licence

MIT.
