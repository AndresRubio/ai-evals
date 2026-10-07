# The current landscape of open-source Python tooling for evaluating LLM applications (Sept 2026)

Research conducted via web search (WebSearch/WebFetch, GitHub, PyPI) in September 2026.

## Two different problems, one blurred vocabulary

Everything below calls itself "evals," but the tools split cleanly into two families that solve different problems:

1. **Foundation-model benchmarking** — "Is model X smart/safe/capable, relative to model Y, on a fixed, mostly-academic task suite (MMLU, GSM8K, HumanEval, GPQA, bio/cyber uplift, agentic CTFs)?" The consumer is a lab, a regulator, or someone picking a base model. The unit of work is a `Task`/benchmark registry entry.
2. **Application-level evaluation** — "Is *my* RAG pipeline / agent / support bot doing the right thing for *my* users, on *my* data?" The consumer is a product team. The unit of work is a trace, a dataset of your own examples, and a scorer you define.

Conflating the two is itself part of the tooling problem discussed in the synthesis.

## Comparison table

| Tool | What it is | Core abstraction you write | Benchmarking vs. application evals | Local-first vs. account | License | Maturity / last release (as of Sep 2026) | Biggest practitioner complaint |
|---|---|---|---|---|---|---|---|
| **promptfoo** | CLI + Node library (Python usable as a provider/wrapper) for prompt/model testing and LLM red-teaming | A **YAML config**: `prompts`, `providers`, `tests[]`, `assert[]` | Application evals (+ a strong red-teaming/security-scan mode) | Local-first CLI; optional hosted dashboard | MIT | Very active; v0.121.x, releases roughly weekly, e.g. 0.121.19 (Jul 2026) | It's a Node/JS-ecosystem tool at its core — Python teams have to context-switch into YAML + npm tooling, and there's no built-in dashboard for organizing datasets/results over time |
| **Inspect AI (`inspect_ai`)**, UK AISI | Python framework/CLI for frontier-model evaluations, built by the UK AI Security Institute (+ Meridian/Arcadia Impact/Vector Institute on `inspect_evals`) | A `@task` function combining a **Dataset + Solver(s) + Scorer** | Both: known for foundation-model and agent benchmarking (200+ pre-built evals in `inspect_evals`: coding, agentic, reasoning, safety/uplift), but it runs any custom dataset, tools, sandboxed agents, and model-graded or custom scorers — a general harness, not only a benchmark registry | Fully local/OSS; `inspect view` for local trace UI | MIT | Very active, becoming a de facto standard for gov/lab safety evals; frequent releases through 2026 | Steep learning curve — the Solver/Scorer/Task abstraction is powerful but heavier than teams need for a simple "grade my chatbot" use case; documentation assumes benchmark-authoring background |
| **DeepEval** (Confident AI) | Python library, "pytest for LLMs," with a paid hosted layer (Confident AI) for dashboards | A **pytest test** using `assert_test(test_case, metrics)`; also a plain Python API | Application evals (RAG, chatbot, agent, safety metrics — 60+ built-in) | Local-first OSS core; Confident AI platform is the account-required upsell | Apache-2.0 | Very active, one of the most popular OSS eval libs (GitHub stars reports vary 1.8k–18k depending on source/fork confusion, but it's a top-tier project); frequent PyPI releases | Metrics feel like a grab-bag/checkbox list decoupled from your own failure modes; heavy push toward the commercial Confident AI platform for anything beyond basic runs |
| **Ragas** | Python library for RAG-specific evaluation and synthetic test-set generation | A `dataset` (question/answer/contexts/ground_truth) + `evaluate(dataset, metrics=[...])` | Application evals (RAG-specific) | Local-first; `pip install ragas` | Apache-2.0 | Active; recently transferred/re-hosted under `vibrantlabsai/ragas` mirror of `explodinggradients/ragas` on GitHub | Metrics are somewhat opaque LLM-judge scores that are hard to debug when low; narrow to RAG (not useful for general agents/chat), and per-sample cost is real (multiple judge calls per metric, cited at $0.05–$0.15/trace on GPT-4o at scale) |
| **OpenAI Evals** (`openai/evals`) + hosted Evals API | GitHub OSS registry/framework (`pip install evals`) plus OpenAI's hosted Evals API | A **YAML/JSON eval spec** registered in the repo, or an `Eval`/`Run`/`data_source_config`/testing-criteria object via the API | Mixed — the OSS repo leans benchmark-registry style; the hosted API leans application evals for API customers | OSS repo is local; the Evals API requires an OpenAI account/key | MIT (repo) | OSS repo has slowed (community treats the hosted Evals API as the maintained path); ~17.6k stars, 2.9k forks | The open-source repo is semi-abandoned relative to the hosted API — OpenAI's own docs now steer users to the Evals API, so the "open source" story is really a thin, aging registry plus a vendor-locked product |
| **Braintrust / autoevals** | Braintrust = proprietary SaaS eval/observability platform; `autoevals` = its OSS scorer library (Python + TS) | An `Eval()` function: `data`, `task`, `scores[]` (autoevals scorers or custom) | Application evals | `autoevals` is local-first OSS; full Braintrust experience requires an account | `autoevals`: MIT/Apache-style OSS; Braintrust platform itself is closed-source SaaS | `autoevals` active, e.g. release Apr 2026, Python ≥3.10 | The genuinely open part (`autoevals`) is just a scorer library — the actual eval-running, dataset, and dashboard experience people mean when they say "Braintrust" is proprietary and pushes you toward their paid platform |
| **LangSmith / LangChain evaluators** | LangSmith = proprietary SaaS platform; `langchain`/`langsmith` ship OSS Python evaluator classes (string, comparison, trajectory) | An **evaluator function/class** run over a LangSmith `Dataset`, or a standalone `evaluate()` call | Application evals | Evaluator classes are local/OSS; the dataset store, dashboards, and annotation queues require a LangSmith account | LangChain/langsmith client libs: MIT; LangSmith platform: closed SaaS (has a generous free tier) | Very active given LangChain's overall momentum | It's marketed as "the" LangChain eval story but the useful parts (dataset versioning, annotation queues, dashboards) are gated behind the hosted product — not usable as a self-contained local OSS tool the way its docs imply |
| **W&B Weave** | Python SDK/toolkit from Weights & Biases for tracing + evaluation, tightly coupled to the W&B platform | `@weave.op` decorated functions + a `weave.Evaluation(dataset, scorers=[...])` | Application evals | Requires a W&B account for the UI/storage; instrumentation is local | Client SDK: Apache-2.0-style OSS; backend is SaaS (with self-hosted enterprise option) | Active, regularly updated on PyPI/GitHub | Fundamentally an "observability platform with evals bolted on" — evaluation runs and dashboards are only useful once you're logged into W&B, so it doesn't function as a standalone local eval tool |
| **Arize Phoenix** | Python OSS observability + eval library from Arize, OpenTelemetry-based tracing | `arize-phoenix-evals`: composable eval **functions over a dataframe of traces**, e.g. `llm_classify()` | Application evals (relevance, hallucination, toxicity, retrieval quality) | Local-first — can run entirely on your machine, no account required | **Elastic License 2.0** (source-available, not OSI-approved; forbids reselling as a hosted service) | Very active; 9k+ stars, frequent PyPI releases (e.g., v20.5.0) | License confusion — people call it "open source" but ELv2 isn't OSI-approved, and there's an open GitHub issue asking Arize to relicense the client under Apache-2.0 for compatibility reasons |
| **HELM** (Stanford CRFM) | Academic Python framework + public leaderboard for holistic foundation-model benchmarking | A **RunSpec** (scenario + adapter + metrics) registered in the framework's config | Foundation-model benchmarking (broad: accuracy, calibration, robustness, fairness, bias, toxicity, efficiency) | Local-first framework; results also published to a public leaderboard/account-free website | Apache-2.0 | **Entered maintenance mode June 1, 2026** — largely superseded in day-to-day use by Inspect AI and lm-eval-harness for new benchmarking work | It's heavy, academically oriented, and now unmaintained for new development — not something an app team would reach for; useful mainly as a leaderboard/reference, not a working tool |
| **lm-evaluation-harness** (EleutherAI) | Python CLI/library, the standard harness for reproducible few-shot benchmark evaluation (used to produce most public leaderboard numbers) | A **YAML task config** (prompt template, dataset, metric) registered in the task registry | Foundation-model benchmarking | Fully local OSS | MIT | Very active, ~13k stars, v0.4.x line, adds vLLM/HF-parallel support | Task-YAML system is powerful but arcane to extend correctly (templating/answer-extraction edge cases); installing from PyPI (`lm-eval`) is discouraged vs. an editable clone because of how fast internals move |
| **Anthropic eval tooling / cookbook** | Not a packaged library — a set of **notebooks/patterns** in `anthropic-cookbook`/`claude-cookbooks` (e.g. `building_evals.ipynb`, the evaluator-optimizer agent pattern) plus prose guidance (rubrics + LLM judges, IRT-based criterion vetting) | Whatever you copy out of the notebook — typically a hand-rolled **Python function that calls Claude as a judge** against a rubric | Application evals (guidance-first, not tool-first) | Fully local (notebooks); needs an Anthropic API key to run | Cookbook: MIT-style permissive (sample code) | Active as documentation, not as a product — no installable package, just recipes | There is no actual tool here — teams have to hand-build the harness the cookbook describes from scratch each time; nothing to `pip install` |
| **pytest-based approaches** (e.g. `pytest-evals`, or plain pytest + assertions) | A **pytest plugin** for running/aggregating LLM eval cases as tests | Ordinary `test_*` functions using pytest fixtures/parametrize, with `pytest-evals` adding aggregation/async/CI helpers | Application evals | Fully local, no account | MIT-style OSS (small community project) | Small, single-maintainer project (`AlmogBaku/pytest-evals`); low but real activity | Very thin — good for wiring evals into CI you already have, but you still have to write every metric/judge yourself; no dataset management, no dashboards, no built-in judge library |
| **DSPy** (`stanfordnlp/dspy`) evaluation/optimizers | Python framework for programming (not prompting) LLM pipelines; evaluation is a first-class primitive used to drive **optimization** | A **metric function** `(example, pred) -> score`, run via `dspy.Evaluate(devset, metric=...)`; optimizers (MIPROv2, BootstrapFewShot) search prompts/few-shots against that metric | Application evals, but purpose-built to *optimize* a pipeline rather than just *report* on it | Fully local OSS | MIT | Very active, ~34k stars, on the 3.x line | The metric doubles as an optimization objective, which tempts teams into over-fitting prompts to a small, under-validated metric before they've done real error analysis — evaluation is a means to auto-prompt-tuning, not an end in itself |
| **Langfuse** | Open-source LLM engineering platform: tracing + datasets + evals + prompt management, self-hostable | A `Dataset` of examples + evaluators defined via SDK/UI (LLM-as-judge, code evaluators) or run through experiments | Application evals | Self-hosted is fully local/OSS (as of June 2025 **all product features, including evals, are MIT-licensed**); only a few enterprise governance features (SCIM, audit log, RBAC) are commercial | MIT (core, including eval features since Jun 2025); enterprise add-ons commercial | Very active — now backed by ClickHouse (since Jan 2026), Python SDK rewritten (v4, Mar 2026) | Self-hosting Langfuse (ClickHouse + Postgres + Redis + S3-compatible storage) is nontrivial ops overhead for a team that just wants "run eval, see score" |
| **Opik** (Comet) | Open-source LLM observability + evaluation platform, self-hostable, with pytest integration | A `Dataset` + `evaluate()` call using built-in or custom **metrics** (hallucination, moderation, RAG); Opik Agent Optimizer SDK for prompt optimization | Application evals | Self-hostable fully free; also offers Comet-hosted cloud | Apache-2.0 | Very active, 20k+ stars, positioned as a leading "true open source" alternative to Langfuse/LangSmith | Younger platform relative to LangSmith/Langfuse — smaller integration ecosystem and community answers, even though the OSS story is arguably cleaner |
| **autoevals** | Already covered under Braintrust above — the standalone OSS scorer library | A `Score` object returned by a scorer function (heuristic, embedding, or LLM-judge) | Application evals | Fully local | MIT-style OSS | Active (Apr 2026 release) | Just a scorer toolkit — no dataset/run management of its own; meant to be embedded in Braintrust or another harness |
| **Evidently** | Open-source Python library for ML *and* LLM evaluation/monitoring — originated in classic ML data/drift monitoring, extended to LLM judges | A `Report`/`TestSuite` built from **100+ built-in metrics** (drift, text stats, LLM-judge) run over a dataframe | Mixed: application evals (LLM judges, RAG relevance) bolted onto a much larger classic-ML-monitoring product | Fully local OSS; optional Evidently Cloud | Apache-2.0 | Active, broad user base carried over from its ML-monitoring roots | LLM eval feels like a secondary feature grafted onto a data-drift-monitoring tool — less depth on LLM-specific failure modes (hallucination nuance, agent trajectories) than dedicated LLM-eval libraries |

Notes on things that were hard to pin to one exact number: GitHub star counts fluctuate across SEO-farmed "guide" sites that were the dominant search results in 2026 for several of these tools (a symptom of the space itself — see synthesis), so figures above are directional, not exact, and should be re-verified against the live repo before being quoted.

## Synthesis: where are the genuine gaps?

**Short answer: no, there is no maintained, installable open-source Python tool that supports the full error-analysis-first workflow (look at traces → open-code failure modes → axial-code into a taxonomy → label a sample → build and align an LLM judge against those human labels → gate CI on the aligned judge). The workflow is well-documented as *methodology* (Hamel Husain and Shreya Shankar's "AI Evals" course/FAQ, and Shankar et al.'s EvalGen / "Who Validates the Validators?" UIST 2024 paper on criteria drift), but it has never been productized as OSS. EvalGen itself was a research prototype, not a maintained package.**

Concretely, every tool surveyed above assumes you already know your metrics before you open the tool:

- **promptfoo / OpenAI Evals / lm-eval-harness / HELM**: you write the assertion or task spec *first*. There's no workflow step for "I don't yet know what 'good' means for my traces — help me discover it."
- **DeepEval / Ragas / autoevals / Evidently**: ship a fixed catalog of pre-named metrics (faithfulness, toxicity, hallucination, relevance…). This is useful, but it *presupposes* your failure modes match their taxonomy. Ragas's RAG metrics won't surface "the agent apologized instead of escalating" as a failure mode; DeepEval's catalog won't surface a domain-specific policy violation unless you already knew to name it.
- **LangSmith / Langfuse / Opik / Weave / Braintrust / Phoenix**: excellent at storing traces, running datasets, and wiring up "an" LLM-as-judge — but the judge-authoring UX is almost universally "write a judge prompt and ship it," not "systematically derive the judge's rubric from a labeled sample and measure judge/human agreement (Cohen's kappa, TPR/TNR) before trusting it." A few (Langfuse, Opik) let you attach human annotation queues and LLM judges *side by side*, which is the closest any tool gets — but none automate or even scaffold the *alignment* step (comparing judge verdicts to human labels, flagging disagreement, iterating the rubric) as a first-class, guided workflow. That alignment step is exactly what EvalGen's research prototype demonstrated and what practitioners currently do by hand in spreadsheets/notebooks.
- **lm-eval-harness / HELM**: solve a genuinely different problem (fixed benchmark, known ground truth) where error-analysis-first doesn't even apply the same way — there's no "discover your own failure taxonomy" step because the taxonomy (the benchmark) is handed to you.
- **Inspect AI**: not only a benchmark runner — it is a full harness for custom model and agent evals, with `model_graded_qa`/`model_graded_fact` and custom scorers. But it starts where the scorer already exists: its scorer and scoring-workflow docs (read 2026-10-07) cover re-scoring and hand-editing scores, but nothing on deriving a judge from reviewed failures or measuring judge/human agreement. (Corrected 2026-09-30; the first version of this survey filed Inspect with lm-eval-harness and HELM as benchmark-only.)
- **DSPy**: comes closest to *acting* on a metric (via optimizers), but again assumes the metric is already correct; if the metric encodes the wrong thing, DSPy will optimize a prompt straight into that blind spot faster than anything else surveyed.
- **The "pytest for evals" tools (DeepEval, pytest-evals, promptfoo)**: are good at the *last mile* (gate CI on a fixed judge/assertion) but contribute nothing to the *first mile* (how did you decide the assertion was right, and how do you know the judge agrees with your humans at an acceptable rate before you trust it in CI).

**What a new Python tool would have to do that these don't:**

1. **Make trace review the entry point, not an afterthought.** Ingest raw traces (from any of the above observability tools, or a CSV/JSONL) and present them for fast open coding — free-text notes per trace — before any metric exists. Today you'd do this in a spreadsheet or a notebook; nobody productized it.
2. **Support axial coding / taxonomy-building as a structured, revisable artifact**, not a one-off clustering script — i.e., let a taxonomy evolve as more traces are reviewed, track prevalence of each failure mode over time, and diff taxonomies across app versions.
3. **Turn labeled samples into a judge-alignment report as a first-class output**: given N human-labeled traces and a candidate judge prompt, compute agreement (accuracy, kappa, TPR/FPR per failure category), show *disagreement examples* to iterate the rubric on, and refuse to "graduate" a judge to CI gating until it clears an agreement bar the team sets — operationalizing exactly what EvalGen prototyped but generalized beyond a research demo and shipped as installable OSS.
4. **Version and diff taxonomies/judges alongside code**, so a CI gate is reproducible and auditable (which existing dataset/eval-run versioning in Langfuse/Opik/LangSmith gets partway to, but always for a metric assumed to already be correct).
5. **Be library-first and framework-agnostic** (a `pip install`-able Python package usable standalone, not requiring a hosted account or a specific observability vendor's trace format) — addressing the recurring complaint across the SaaS-backed tools (LangSmith, Weave, Braintrust, Langfuse's non-core features) that the genuinely open, local, no-account pieces are thin slices of a product whose real value sits behind a login.

In short: the ecosystem has matured a lot on *running* evals (datasets, scorers, CI hooks, dashboards) and on *benchmarking foundation models* (Inspect AI, lm-eval-harness, HELM), but the upstream, harder-to-automate step — figuring out, from real failures, what to measure at all, and proving your automated judge actually agrees with your humans — remains a manual, spreadsheet-and-notebook practice taught as methodology (Hamel Husain / Shreya Shankar's course, the EvalGen paper) rather than shipped as a tool. That is the real gap.

## Sources

- [promptfoo GitHub releases](https://github.com/promptfoo/promptfoo/releases)
- [promptfoo release notes](https://www.promptfoo.dev/docs/releases/)
- [Inspect AI — UK AISI](https://inspect.aisi.org.uk/)
- [UKGovernmentBEIS/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai)
- [UKGovernmentBEIS/inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals)
- [DeepEval PyPI](https://pypi.org/project/deepeval/)
- [Confident AI review 2026](https://techsy.io/en/blog/confident-ai-review)
- [Ragas docs](https://docs.ragas.io/en/stable/)
- [vibrantlabsai/ragas](https://github.com/vibrantlabsai/ragas)
- [openai/evals](https://github.com/openai/evals)
- [Getting Started with OpenAI Evals](https://developers.openai.com/cookbook/examples/evaluation/getting_started_with_openai_evals)
- [braintrustdata/autoevals](https://github.com/braintrustdata/autoevals)
- [autoevals PyPI](https://pypi.org/project/autoevals/)
- [LangSmith evaluation docs](https://docs.langchain.com/langsmith/evaluation)
- [LangSmith alternatives 2026 — Braintrust](https://www.braintrust.dev/articles/langsmith-alternatives-2026)
- [wandb/weave](https://github.com/wandb/weave)
- [weave PyPI](https://pypi.org/project/weave/)
- [Arize-ai/phoenix](https://github.com/arize-ai/phoenix)
- [Phoenix license docs](https://arize.com/docs/phoenix/self-hosting/license)
- [Phoenix relicensing issue #9890](https://github.com/Arize-ai/phoenix/issues/9890)
- [stanford-crfm/helm](https://github.com/stanford-crfm/helm)
- [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [lm-eval PyPI](https://pypi.org/project/lm-eval/)
- [anthropics/claude-cookbooks — building_evals.ipynb](https://github.com/anthropics/anthropic-cookbook/blob/main/misc/building_evals.ipynb)
- [AlmogBaku/pytest-evals](https://github.com/AlmogBaku/pytest-evals)
- [DSPy metrics and evaluation docs](https://dspy.ai/diving-deeper/metrics-and-evaluation/)
- [stanfordnlp/dspy metrics.md](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/learn/evaluation/metrics.md)
- [Langfuse — Doubling Down on Open Source](https://langfuse.com/blog/2025-06-04-open-sourcing-langfuse-product)
- [langfuse/langfuse](https://github.com/langfuse/langfuse)
- [comet-ml/opik](https://github.com/comet-ml/opik)
- [Opik PyPI](https://pypi.org/project/opik/)
- [evidentlyai/evidently](https://github.com/evidentlyai/evidently)
- [Evidently LLM evaluation guide](https://www.evidentlyai.com/llm-guide/llm-evaluation)
- [Hamel Husain — AI Evals FAQ: why error analysis matters](https://hamel.dev/blog/posts/evals-faq/why-is-error-analysis-so-important-in-llm-evals-and-how-is-it-performed.html)
- [Hamel Husain — AI Evals: Everything You Need to Know](https://hamel.dev/blog/posts/evals-faq/)
- [Who Validates the Validators? (EvalGen, Shankar et al., UIST 2024)](https://arxiv.org/abs/2404.12272)
- [Ragas/Promptfoo/LangSmith limitations discussion — Braintrust](https://www.braintrust.dev/articles/best-promptfoo-alternatives-2026)

---

## Addendum, 2026-09-18 — Pydantic Evals

The survey above missed `pydantic_evals` entirely. It belongs in the comparison, and it is
the most rigorous counter-example to the gap thesis, so it gets checked rather than
asserted. Verified against the docs and the `pydantic/pydantic-ai` repo.

**What it is.** The evals package of the Pydantic AI project. MIT, ~20k stars on the parent
repo, and — importantly — it does *not* depend on `pydantic-ai`, and Logfire is an optional
dependency, so it is fully usable locally with no account.

**Core abstractions.** `Case` (one scenario: inputs, optional `expected_output`, metadata,
case-specific evaluators), `Dataset` (cases plus dataset-level evaluators), `Evaluator`
(base class scoring an output; deterministic or LLM-based), `LLMJudge` (built-in evaluator
that grades against a rubric string, returning an assertion and/or a 0–1 score), and an
`EvaluationReport`. There is a `generate_dataset()` that synthesises cases against a typed
schema. Span-based evaluators can inspect OpenTelemetry spans to score tool calls and
execution flow.

**What it does not have.** Checked in the obvious places and not found:

- No discovery of failure modes from traces — no open coding, clustering or taxonomy
  building. Cases and evaluators are always supplied up front.
- No human annotation or labelling surface of any kind.
- **No judge/human agreement measurement.** No accuracy, TPR/TNR, kappa, calibration or
  disagreement review. The LLM-judge page's "Model Biases/Limitations" section advises
  validating judges against human-labelled test sets — as one sentence of prose. There is
  no API behind it.
- No train/dev/test split concept, and nothing about overfitting an evaluator to the data
  used to build it.

**Verdict.** Pydantic Evals is the best-engineered instance of the pattern the synthesis
above describes: excellent at the back half (define cases → run evaluators → report → gate
CI), silent on the front half. It does not close the gap. It is the strongest evidence for
it — the most type-rigorous eval library in the ecosystem tells you to validate your judge
against human labels, and then hands you nothing to do it with.

Sources: <https://pydantic.dev/docs/ai/evals/evals/>,
<https://pydantic.dev/docs/ai/evals/evaluators/llm-judge/>,
<https://pydantic.dev/docs/ai/evals/how-to/dataset-management/>,
<https://pydantic.dev/docs/ai/api/pydantic_evals/generation/>,
<https://github.com/pydantic/pydantic-ai>

---

## Other sources

Not independently verified against Anthropic's own materials — recorded here only so
the README's claim about it has a citation, not to assert it beyond what the source
itself says.

- **Anthropic's error-discovery skill.** Shipped 2026-08-23, installable via `npx
  skills add`. Builds a review app over a diverse trace sample and clusters
  annotations into failure modes. Reported by the ai-digest evals-and-benchmarks
  topic page: <https://andresrubio.github.io/ai-digest/topics/evals-and-benchmarks.html>.
  This is a secondary source (a digest of reporting on the skill, not Anthropic's own
  docs or repo), so treat the description above as reported, not confirmed firsthand.

  **Correction (2026-10-07):** a skill matching this description is published at
  <https://github.com/ai-evals-course/evals-skills> (`skills/error-discovery/`, installed
  with `npx skills add`). Its history there starts on 2026-08-15 with commits by Hamel
  Husain, the first of which moves the skill into a plugin layout, so it existed before
  that date somewhere else. We found no `error-discovery` skill in Anthropic's public
  `anthropics/skills` repo and no match in a GitHub code search of the `anthropics` org.
  That makes a misattribution in the digest likely, not proven; the README now credits
  the repo where the skill can actually be read.
