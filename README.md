# axial

From raw traces to an LLM judge you have actually measured.

Every open-source eval tool assumes you already know your metrics before you open it.
promptfoo wants an assertion. DeepEval and Ragas ship a catalogue of pre-named metrics.
Pydantic Evals wants a rubric. Inspect AI and lm-evaluation-harness solve a different
problem — a fixed benchmark whose ground truth is handed to you. The observability
platforms store traces beautifully and then ask you to write a judge prompt and ship it.

None of them help with the part that decides whether any of it works: reading real
failures, deciding what "good" means for your system, and proving your automated judge
agrees with your humans before you trust it. That is taught as methodology and done in
spreadsheets. `axial` is that step, as code.

## Install

```bash
uv add axial   # or: pip install axial
```

## The pipeline

Each command's output is the next one's input. Everything lands in `.axial/` as plain
text, so your eval state is reviewable in a pull request like any other source.

| Command | Consumes | Produces |
|---|---|---|
| `axial ingest` | JSONL or CSV | normalised traces |
| `axial sample` | traces | a review batch, always including a random slice |
| `axial annotate` | a batch | open codes, verdicts and critiques |
| `axial taxonomy` | your open codes | a failure taxonomy with prevalence |
| `axial label-import` | traces + one failure mode | human labels |
| `axial split` | labels | a stratified, seeded, frozen split |
| `axial judge` | a split + a prompt | verdicts, and errors kept separate |
| `axial align` | verdicts vs labels | an alignment report |
| `axial promote` | alignment on test | a judge cleared for CI |
| `axial gate` | promoted judges | an exit code |

## What it refuses to do

The refusals are the product.

- **No taxonomy before 30 human open codes.** Open coding is where you find failure modes
  nobody anticipated. A model cannot discover what it was not shown.
- **No machine-generated label in dev or test.** A judge aligned to machine labels is
  aligned to itself.
- **The test split is write-once.** Redrawing it invalidates every number reported
  against it, so `axial` will not do it quietly.
- **Few-shot examples come from train only.**
- **Never raw agreement alone.** TPR and TNR are printed beside it, with a confidence
  interval. At a 5% failure rate an always-pass judge scores 95% agreement and detects
  nothing.
- **No promotion below the bar.** Test TPR and TNR must clear your threshold and the
  dev→test gap must be within tolerance.
- **Gates compare intervals, not point estimates**, and warn when a suite stops failing
  anything — a saturated eval has stopped carrying signal.
- **Infra failures never score as model failures.** They are quarantined in
  `errors.jsonl` with a failure class.

## Quickstart

Runs entirely offline against the committed example corpus — no API key, no spend.

```bash
uv run axial init
uv run axial ingest examples/newsroom/traces.jsonl
uv run axial sample --n 40 --name first-pass
```

Uncomment and set `annotator = "your-name"` near the top of `.axial/config.toml`
(every label records who made it), then:

```bash
uv run axial annotate --name first-pass     # annotate 30+, then stop
uv run axial taxonomy
```

See [`examples/newsroom/`](examples/newsroom/) for the rest of the walkthrough.

## Prior art

- **promptfoo, DeepEval, Ragas, autoevals** — assertion and metric catalogues. `axial`
  produces the judge they run.
- **Pydantic Evals** — the most rigorous of these: typed `Case` and `Dataset`, an
  `LLMJudge` graded against a rubric you write, MIT, no account needed. Its own docs tell
  you to validate judges against human-labelled test sets, in one sentence, with nothing
  behind it. `axial` is that sentence, implemented.
- **Inspect AI, lm-evaluation-harness** — foundation-model benchmarking, where the
  taxonomy is the benchmark and is given to you. Different problem.
- **Langfuse, Opik, LangSmith, Braintrust, Phoenix** — trace storage and judge running.
  Closest on annotation queues; none scaffold judge/human alignment as a gated step.
- **EvalGen** (Shankar et al., UIST 2024) — demonstrated this loop and named criteria
  drift. A research prototype, never maintained. `axial` is the shipped version.
- **Anthropic's error-discovery skill** — clusters your annotations into failure modes
  inside an agent session. `axial` is the versioned artifact store and CI gate around
  the same idea.

## Credit

The methodology is Hamel Husain's and Shreya Shankar's, with Eugene Yan's work on judge
alignment and Anthropic's published eval guides. `axial` implements it; it did not
invent it. See [`docs/research/`](docs/research/) for the sources behind every default.

## Licence

MIT.
