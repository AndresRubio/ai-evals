---
name: langfuse
description: Open-source LLM engineering platform for tracing, datasets, evals and annotation queues; MIT since June 2025
shelf: landscape
updated: 2026-09-17
sources:
  - url: https://langfuse.com/blog/2025-06-04-open-sourcing-langfuse-product
    read: 2026-09-17
  - url: https://github.com/langfuse/langfuse
    read: 2026-09-17
license: MIT
related: [landscape-overview]
---

# Langfuse

## What it is

Open-source LLM engineering platform: tracing + datasets + evals + prompt management, self-hostable

## Core abstraction

A `Dataset` of examples + evaluators defined via SDK/UI (LLM-as-judge, code evaluators) or run through experiments

## Where it sits

Application evals

Self-hosted is fully local/OSS (as of June 2025 **all product features, including evals, are MIT-licensed**); only a few enterprise governance features (SCIM, audit log, RBAC) are commercial

## Maturity

As of 2026-09-17: Very active — now backed by ClickHouse (since Jan 2026), Python SDK rewritten (v4, Mar 2026)

## Biggest complaint

Self-hosting Langfuse (ClickHouse + Postgres + Redis + S3-compatible storage) is nontrivial ops overhead for a team that just wants "run eval, see score"

## Relation to axial

- **LangSmith / Langfuse / Opik / Weave / Braintrust / Phoenix**: excellent at storing traces, running datasets, and wiring up "an" LLM-as-judge — but the judge-authoring UX is almost universally "write a judge prompt and ship it," not "systematically derive the judge's rubric from a labeled sample and measure judge/human agreement (Cohen's kappa, TPR/TNR) before trusting it." A few (Langfuse, Opik) let you attach human annotation queues and LLM judges *side by side*, which is the closest any tool gets — but none automate or even scaffold the *alignment* step (comparing judge verdicts to human labels, flagging disagreement, iterating the rubric) as a first-class, guided workflow. That alignment step is exactly what EvalGen's research prototype demonstrated and what practitioners currently do by hand in spreadsheets/notebooks.

No dependency.

## Corrections

None.
