---
name: landscape-overview
description: Two families of eval tools, benchmarking and application evals, and why none closes the error-analysis-first gap
shelf: landscape
updated: 2026-10-07
sources:
  - url: https://hamel.dev/blog/posts/evals-faq/why-is-error-analysis-so-important-in-llm-evals-and-how-is-it-performed.html
    read: 2026-09-17
  - url: https://arxiv.org/abs/2404.12272
    read: 2026-09-17
  - url: https://www.braintrust.dev/articles/best-promptfoo-alternatives-2026
    read: 2026-09-17
related: [methodology-spine, evalgen, evals-skills]
---

# The landscape of eval tooling

## Two different problems, one blurred vocabulary

Everything below calls itself "evals," but the tools split cleanly into two families that solve different problems:

1. **Foundation-model benchmarking** — "Is model X smart/safe/capable, relative to model Y, on a fixed, mostly-academic task suite (MMLU, GSM8K, HumanEval, GPQA, bio/cyber uplift, agentic CTFs)?" The consumer is a lab, a regulator, or someone picking a base model. The unit of work is a `Task`/benchmark registry entry.
2. **Application-level evaluation** — "Is *my* RAG pipeline / agent / support bot doing the right thing for *my* users, on *my* data?" The consumer is a product team. The unit of work is a trace, a dataset of your own examples, and a scorer you define.

Conflating the two is itself part of the tooling problem discussed in the synthesis.

Notes on things that were hard to pin to one exact number: GitHub star counts fluctuate across SEO-farmed "guide" sites that were the dominant search results in 2026 for several of these tools (a symptom of the space itself — see synthesis), so figures in the per-tool entries are directional, not exact, and should be re-verified against the live repo before being quoted.

Two entries in this shelf sit outside both families: [Great Expectations](great-expectations.md), which is data quality for tabular pipelines, and [Warp Factories](warp-factories.md), whose scorers are LLM judges embedded in an agent platform.

## Synthesis: where are the genuine gaps?

**Is there a tool that supports the full workflow? No: there is no maintained, installable open-source Python tool that supports the full error-analysis-first workflow (look at traces → open-code failure modes → axial-code into a taxonomy → label a sample → build and align an LLM judge against those human labels → gate CI on the aligned judge). The workflow is well-documented as *methodology* (Hamel Husain and Shreya Shankar's "AI Evals" course/FAQ, and Shankar et al.'s [EvalGen](evalgen.md) / "Who Validates the Validators?" UIST 2024 paper on criteria drift), but it has never been productized as OSS. EvalGen itself was a research prototype, not a maintained package.**

Concretely, every tool surveyed in this shelf assumes you already know your metrics before you open the tool:

- **[promptfoo](promptfoo.md) / [OpenAI Evals](openai-evals.md) / [lm-eval-harness](lm-evaluation-harness.md) / [HELM](helm.md)**: you write the assertion or task spec *first*. There's no workflow step for "I don't yet know what 'good' means for my traces — help me discover it."
- **[DeepEval](deepeval.md) / [Ragas](ragas.md) / [autoevals](braintrust-autoevals.md) / [Evidently](evidently.md)**: ship a fixed catalog of pre-named metrics (faithfulness, toxicity, hallucination, relevance…). This is useful, but it *presupposes* your failure modes match their taxonomy. Ragas's RAG metrics won't surface "the agent apologized instead of escalating" as a failure mode; DeepEval's catalog won't surface a domain-specific policy violation unless you already knew to name it.
- **[LangSmith](langsmith.md) / [Langfuse](langfuse.md) / [Opik](opik.md) / [Weave](wandb-weave.md) / [Braintrust](braintrust-autoevals.md) / [Phoenix](arize-phoenix.md)**: excellent at storing traces, running datasets, and wiring up "an" LLM-as-judge — but the judge-authoring UX is almost universally "write a judge prompt and ship it," not "systematically derive the judge's rubric from a labeled sample and measure judge/human agreement (Cohen's kappa, TPR/TNR) before trusting it." A few (Langfuse, Opik) let you attach human annotation queues and LLM judges *side by side*, which is the closest any tool gets — but none automate or even scaffold the *alignment* step (comparing judge verdicts to human labels, flagging disagreement, iterating the rubric) as a first-class, guided workflow. That alignment step is exactly what EvalGen's research prototype demonstrated and what practitioners currently do by hand in spreadsheets/notebooks.
- **lm-eval-harness / HELM**: solve a genuinely different problem (fixed benchmark, known ground truth) where error-analysis-first doesn't even apply the same way — there's no "discover your own failure taxonomy" step because the taxonomy (the benchmark) is handed to you.
- **[Inspect AI](inspect-ai.md)**: not only a benchmark runner — it is a full harness for custom model and agent evals, with `model_graded_qa`/`model_graded_fact` and custom scorers. But it starts where the scorer already exists: its scorer and scoring-workflow docs (read 2026-10-07) cover re-scoring and hand-editing scores, but nothing on deriving a judge from reviewed failures or measuring judge/human agreement. (Corrected 2026-09-30; the first version of this survey filed Inspect with lm-eval-harness and HELM as benchmark-only.)
- **[DSPy](dspy.md)**: comes closest to *acting* on a metric (via optimizers), but again assumes the metric is already correct; if the metric encodes the wrong thing, DSPy will optimize a prompt straight into that blind spot faster than anything else surveyed.
- **The "pytest for evals" tools (DeepEval, [pytest-evals](pytest-evals.md), promptfoo)**: are good at the *last mile* (gate CI on a fixed judge/assertion) but contribute nothing to the *first mile* (how did you decide the assertion was right, and how do you know the judge agrees with your humans at an acceptable rate before you trust it in CI).

**What a new Python tool would have to do that these don't:**

1. **Make trace review the entry point, not an afterthought.** Ingest raw traces (from any of the above observability tools, or a CSV/JSONL) and present them for fast open coding — free-text notes per trace — before any metric exists. Today you'd do this in a spreadsheet or a notebook; nobody productized it.
2. **Support axial coding / taxonomy-building as a structured, revisable artifact**, not a one-off clustering script — i.e., let a taxonomy evolve as more traces are reviewed, track prevalence of each failure mode over time, and diff taxonomies across app versions.
3. **Turn labeled samples into a judge-alignment report as a first-class output**: given N human-labeled traces and a candidate judge prompt, compute agreement (accuracy, kappa, TPR/FPR per failure category), show *disagreement examples* to iterate the rubric on, and refuse to "graduate" a judge to CI gating until it clears an agreement bar the team sets — operationalizing exactly what EvalGen prototyped but generalized beyond a research demo and shipped as installable OSS.
4. **Version and diff taxonomies/judges alongside code**, so a CI gate is reproducible and auditable (which existing dataset/eval-run versioning in Langfuse/Opik/LangSmith gets partway to, but always for a metric assumed to already be correct).
5. **Be library-first and framework-agnostic** (a `pip install`-able Python package usable standalone, not requiring a hosted account or a specific observability vendor's trace format) — addressing the recurring complaint across the SaaS-backed tools (LangSmith, Weave, Braintrust, Langfuse's non-core features) that the genuinely open, local, no-account pieces are thin slices of a product whose real value sits behind a login.

In short: the ecosystem has matured a lot on *running* evals (datasets, scorers, CI hooks, dashboards) and on *benchmarking foundation models* (lm-eval-harness, HELM, and Inspect AI's benchmark registry), but the upstream, harder-to-automate step — figuring out, from real failures, what to measure at all, and proving your automated judge actually agrees with your humans — remains a manual, spreadsheet-and-notebook practice taught as methodology (Hamel Husain / Shreya Shankar's course, the EvalGen paper) rather than shipped as a tool. That is the real gap.
