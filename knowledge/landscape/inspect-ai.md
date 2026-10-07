---
name: inspect-ai
description: UK AISI's open harness for model and agent evals, with datasets, solvers, scorers, sandboxed agents and a local log viewer
shelf: landscape
updated: 2026-10-07
sources:
  - url: https://inspect.aisi.org.uk/
    read: 2026-09-17
  - url: https://github.com/UKGovernmentBEIS/inspect_ai
    read: 2026-09-17
  - url: https://github.com/UKGovernmentBEIS/inspect_evals
    read: 2026-09-17
  - url: https://inspect.aisi.org.uk/scorers.html
    read: 2026-10-07
  - url: https://inspect.aisi.org.uk/scoring.html
    read: 2026-10-07
license: MIT
related: [landscape-overview]
---

# Inspect AI (`inspect_ai`), UK AISI

## What it is

Python framework/CLI for frontier-model evaluations, built by the UK AI Security Institute (+ Meridian/Arcadia Impact/Vector Institute on `inspect_evals`)

## Core abstraction

A `@task` function combining a **Dataset + Solver(s) + Scorer**

## Where it sits

Both: known for foundation-model and agent benchmarking (200+ pre-built evals in `inspect_evals`: coding, agentic, reasoning, safety/uplift), but it runs any custom dataset, tools, sandboxed agents, and model-graded or custom scorers — a general harness, not only a benchmark registry

Fully local/OSS; `inspect view` for local trace UI

## Maturity

As of 2026-09-17: Very active, becoming a de facto standard for gov/lab safety evals; frequent releases through 2026

## Biggest complaint

Steep learning curve — the Solver/Scorer/Task abstraction is powerful but heavier than teams need for a simple "grade my chatbot" use case; documentation assumes benchmark-authoring background

## Relation to axial

- **Inspect AI**: not only a benchmark runner — it is a full harness for custom model and agent evals, with `model_graded_qa`/`model_graded_fact` and custom scorers. But it starts where the scorer already exists: its scorer and scoring-workflow docs (read 2026-10-07) cover re-scoring and hand-editing scores, but nothing on deriving a judge from reviewed failures or measuring judge/human agreement. (Corrected 2026-09-30; the first version of this survey filed Inspect with lm-eval-harness and HELM as benchmark-only.)

**Inspect AI** (UK AI Security Institute) — a full open harness for model and agent evals: custom datasets, tool use, sandboxed agents, `model_graded_qa` and custom scorers, a local log viewer, MIT. It runs a model-graded scorer, and its scoring workflow lets you re-score logs and hand-edit individual scores; its scorer and scoring-workflow docs say nothing about deriving a scorer from your own failures or measuring its agreement with human labels. `axial` sits upstream of it: the judge Inspect would run is the thing `axial` builds and validates.

No dependency.

## Corrections

2026-10-07: the first version of this survey filed Inspect with lm-eval-harness and HELM as benchmark-only. Corrected after reading its scorer and scoring-workflow docs on 2026-10-07; see Relation to axial.
