---
name: langsmith
description: LangChain's closed eval and tracing platform, with open evaluator classes in the client libraries
shelf: landscape
updated: 2026-09-17
sources:
  - url: https://docs.langchain.com/langsmith/evaluation
    read: 2026-09-17
  - url: https://www.braintrust.dev/articles/langsmith-alternatives-2026
    read: 2026-09-17
license: proprietary
related: [landscape-overview]
---

# LangSmith / LangChain evaluators

## What it is

LangSmith = proprietary SaaS platform; `langchain`/`langsmith` ship OSS Python evaluator classes (string, comparison, trajectory)

## Core abstraction

An **evaluator function/class** run over a LangSmith `Dataset`, or a standalone `evaluate()` call

## Where it sits

Application evals

Evaluator classes are local/OSS; the dataset store, dashboards, and annotation queues require a LangSmith account

## Maturity

As of 2026-09-17: Very active given LangChain's overall momentum

## Biggest complaint

It's marketed as "the" LangChain eval story but the useful parts (dataset versioning, annotation queues, dashboards) are gated behind the hosted product — not usable as a self-contained local OSS tool the way its docs imply

## Relation to axial

- **LangSmith / Langfuse / Opik / Weave / Braintrust / Phoenix**: excellent at storing traces, running datasets, and wiring up "an" LLM-as-judge — but the judge-authoring UX is almost universally "write a judge prompt and ship it," not "systematically derive the judge's rubric from a labeled sample and measure judge/human agreement (Cohen's kappa, TPR/TNR) before trusting it." A few (Langfuse, Opik) let you attach human annotation queues and LLM judges *side by side*, which is the closest any tool gets — but none automate or even scaffold the *alignment* step (comparing judge verdicts to human labels, flagging disagreement, iterating the rubric) as a first-class, guided workflow. That alignment step is exactly what EvalGen's research prototype demonstrated and what practitioners currently do by hand in spreadsheets/notebooks.

No dependency.

## Corrections

None.
