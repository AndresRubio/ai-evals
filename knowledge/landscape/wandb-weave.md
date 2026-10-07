---
name: wandb-weave
description: Weights & Biases tracing and evaluation SDK, usable only with a W&B account for storage and the UI
shelf: landscape
updated: 2026-09-17
sources:
  - url: https://github.com/wandb/weave
    read: 2026-09-17
  - url: https://pypi.org/project/weave/
    read: 2026-09-17
license: Apache-2.0
related: [landscape-overview]
---

# W&B Weave

## What it is

Python SDK/toolkit from Weights & Biases for tracing + evaluation, tightly coupled to the W&B platform

## Core abstraction

`@weave.op` decorated functions + a `weave.Evaluation(dataset, scorers=[...])`

## Where it sits

Application evals

Requires a W&B account for the UI/storage; instrumentation is local

## Maturity

As of 2026-09-17: Active, regularly updated on PyPI/GitHub

## Biggest complaint

Fundamentally an "observability platform with evals bolted on" — evaluation runs and dashboards are only useful once you're logged into W&B, so it doesn't function as a standalone local eval tool

## Relation to axial

- **LangSmith / Langfuse / Opik / Weave / Braintrust / Phoenix**: excellent at storing traces, running datasets, and wiring up "an" LLM-as-judge — but the judge-authoring UX is almost universally "write a judge prompt and ship it," not "systematically derive the judge's rubric from a labeled sample and measure judge/human agreement (Cohen's kappa, TPR/TNR) before trusting it." A few (Langfuse, Opik) let you attach human annotation queues and LLM judges *side by side*, which is the closest any tool gets — but none automate or even scaffold the *alignment* step (comparing judge verdicts to human labels, flagging disagreement, iterating the rubric) as a first-class, guided workflow. That alignment step is exactly what EvalGen's research prototype demonstrated and what practitioners currently do by hand in spreadsheets/notebooks.

No dependency.

## Corrections

None.
