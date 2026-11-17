---
name: arize-phoenix
description: Arize's OpenTelemetry-based observability and eval library; the server and evals library are source-available under Elastic License 2.0, the client Apache-2.0
shelf: landscape
updated: 2026-11-17
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
  - url: https://pypi.org/project/arize-phoenix-evals/
    read: 2026-11-17
  - url: https://pypi.org/project/arize-phoenix-otel/
    read: 2026-11-17
license: Elastic-2.0
related: [landscape-overview]
---

# Arize Phoenix

## What it is

Python observability + eval library from Arize, with OpenTelemetry-based tracing. The server (`arize-phoenix`) and the evals library (`arize-phoenix-evals`) are source-available under Elastic License 2.0; the client (`arize-phoenix-client`) and the OpenTelemetry package (`arize-phoenix-otel`) are Apache-2.0. The licence above is the server's and the evals library's.

## Core abstraction

`arize-phoenix-evals`: composable eval **functions over a dataframe of traces**, e.g. `llm_classify()`

## Where it sits

Application evals (relevance, hallucination, toxicity, retrieval quality)

Local-first: it can run entirely on your machine, with no account required

## Maturity

As of 2026-11-11: very active, about 12k stars, directional. The server package `arize-phoenix` is at 21.0.0 (2026-10-30) and the client `arize-phoenix-client` at 3.7.0 (2026-11-05).

## Biggest complaint

License confusion: people call it open source, but the server and the evals library ship under Elastic License 2.0, which isn't OSI-approved. Arize's licence page still gives Phoenix as a whole as ELv2 and names no package. A request to relicense the client for compatibility with open-source projects (issue #9890) was closed on 2025-11-19, and PyPI now lists `arize-phoenix-client` as Apache-2.0, and `arize-phoenix` and `arize-phoenix-evals` as ELv2.

## Relation to axial

LangSmith, Langfuse, Opik, Weave, Braintrust and Phoenix are excellent at storing traces, running datasets and wiring up "an" LLM-as-judge. Their judge-authoring path is almost always "write a judge prompt and ship it"; deriving the judge's rubric from a labelled sample and measuring judge/human agreement (Cohen's kappa, TPR/TNR) before trusting it is rare. A few (Langfuse, Opik) let you attach human annotation queues and LLM judges *side by side*, the closest any tool gets, but none automates or scaffolds the *alignment* step (comparing judge verdicts to human labels, flagging disagreement, iterating the rubric) as a guided workflow. That step is what EvalGen's research prototype demonstrated and what practitioners do by hand in spreadsheets and notebooks.

No dependency. Code that reads traces out of a running Phoenix server needs only the Apache-2.0 client as a dependency. The server it talks to stays ELv2, and so does `arize-phoenix-evals`, Phoenix's own judge library.

## Corrections

2026-11-17: the 2026-09-17 version said the issue asking Arize to relicense the client was open, and gave the whole project as Elastic License 2.0. The issue was closed on 2025-11-19. The client and the OpenTelemetry package are Apache-2.0, and the server and the evals library are ELv2. The description, What it is, Biggest complaint and Relation to axial now say so.
