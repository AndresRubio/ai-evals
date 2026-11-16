---
name: arize-phoenix
description: Arize's OpenTelemetry-based observability and eval library; the server is source-available under Elastic License 2.0, the client Apache-2.0
shelf: landscape
updated: 2026-11-16
sources:
  - url: https://github.com/arize-ai/phoenix
    read: 2026-11-11
  - url: https://arize.com/docs/phoenix/self-hosting/license
    read: 2026-11-16
  - url: https://github.com/Arize-ai/phoenix/issues/9890
    read: 2026-11-11
  - url: https://pypi.org/project/arize-phoenix/
    read: 2026-11-11
  - url: https://pypi.org/project/arize-phoenix-client/
    read: 2026-11-11
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

Local-first: it can run entirely on your machine, with no account required

## Maturity

As of 2026-11-11: very active, about 12k stars. The server package `arize-phoenix` is at 21.0.0 (2026-10-30) and the client `arize-phoenix-client` at 3.7.0 (2026-11-05).

## Biggest complaint

License confusion: people call it open source, but the server ships under Elastic License 2.0, which isn't OSI-approved. Arize's licence page still gives Phoenix as a whole as ELv2 and names no package. A request to relicense the client for compatibility with open-source projects (issue #9890) was closed on 2025-11-19, and PyPI now lists `arize-phoenix-client` as Apache-2.0 and `arize-phoenix` as ELv2.

## Relation to axial

LangSmith, Langfuse, Opik, Weave, Braintrust and Phoenix are excellent at storing traces, running datasets and wiring up "an" LLM-as-judge. Their judge-authoring path is almost always "write a judge prompt and ship it"; deriving the judge's rubric from a labelled sample and measuring judge/human agreement (Cohen's kappa, TPR/TNR) before trusting it is rare. A few (Langfuse, Opik) let you attach human annotation queues and LLM judges *side by side*, the closest any tool gets, but none automates or scaffolds the *alignment* step (comparing judge verdicts to human labels, flagging disagreement, iterating the rubric) as a guided workflow. That step is what EvalGen's research prototype demonstrated and what practitioners do by hand in spreadsheets and notebooks.

No dependency. Code that reads traces out of a running Phoenix server needs only the Apache-2.0 client as a dependency; the server it talks to stays ELv2.

## Corrections

2026-11-16: the 2026-09-17 version said the issue asking Arize to relicense the client was open, and gave the whole project as Elastic License 2.0. The issue was closed on 2025-11-19 and the client is Apache-2.0; only the server is ELv2. The description, Maturity and Biggest complaint now say so.
