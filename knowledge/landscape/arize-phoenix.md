---
name: arize-phoenix
description: Arize's OpenTelemetry-based observability and eval library, source-available under Elastic License 2.0
shelf: landscape
updated: 2026-09-17
sources:
  - url: https://github.com/arize-ai/phoenix
    read: 2026-09-17
  - url: https://arize.com/docs/phoenix/self-hosting/license
    read: 2026-09-17
  - url: https://github.com/Arize-ai/phoenix/issues/9890
    read: 2026-09-17
license: Elastic-2.0
related: [landscape-overview]
---

# Arize Phoenix

## What it is

Python OSS observability + eval library from Arize, OpenTelemetry-based tracing

## Core abstraction

`arize-phoenix-evals`: composable eval **functions over a dataframe of traces**, e.g. `llm_classify()`

## Where it sits

Application evals (relevance, hallucination, toxicity, retrieval quality)

Local-first — can run entirely on your machine, no account required

## Maturity

As of 2026-09-17: Very active; 9k+ stars, frequent PyPI releases (e.g., v20.5.0)

## Biggest complaint

License confusion — people call it "open source" but ELv2 isn't OSI-approved, and there's an open GitHub issue asking Arize to relicense the client under Apache-2.0 for compatibility reasons

## Relation to axial

- **LangSmith / Langfuse / Opik / Weave / Braintrust / Phoenix**: excellent at storing traces, running datasets, and wiring up "an" LLM-as-judge — but the judge-authoring UX is almost universally "write a judge prompt and ship it," not "systematically derive the judge's rubric from a labeled sample and measure judge/human agreement (Cohen's kappa, TPR/TNR) before trusting it." A few (Langfuse, Opik) let you attach human annotation queues and LLM judges *side by side*, which is the closest any tool gets — but none automate or even scaffold the *alignment* step (comparing judge verdicts to human labels, flagging disagreement, iterating the rubric) as a first-class, guided workflow. That alignment step is exactly what EvalGen's research prototype demonstrated and what practitioners currently do by hand in spreadsheets/notebooks.

No dependency.

## Corrections

None.
