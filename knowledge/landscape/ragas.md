---
name: ragas
description: RAG-specific evaluation and synthetic test-set generation over question, answer, contexts and ground truth
shelf: landscape
updated: 2026-09-17
sources:
  - url: https://docs.ragas.io/en/stable/
    read: 2026-09-17
  - url: https://github.com/vibrantlabsai/ragas
    read: 2026-09-17
license: Apache-2.0
related: [landscape-overview]
---

# Ragas

## What it is

Python library for RAG-specific evaluation and synthetic test-set generation

## Core abstraction

A `dataset` (question/answer/contexts/ground_truth) + `evaluate(dataset, metrics=[...])`

## Where it sits

Application evals (RAG-specific)

Local-first; `pip install ragas`

## Maturity

As of 2026-09-17: Active; recently transferred/re-hosted under `vibrantlabsai/ragas` mirror of `explodinggradients/ragas` on GitHub

## Biggest complaint

Metrics are somewhat opaque LLM-judge scores that are hard to debug when low; narrow to RAG (not useful for general agents/chat), and per-sample cost is real (multiple judge calls per metric, cited at $0.05–$0.15/trace on GPT-4o at scale)

## Relation to axial

- **DeepEval / Ragas / autoevals / Evidently**: ship a fixed catalog of pre-named metrics (faithfulness, toxicity, hallucination, relevance…). This is useful, but it *presupposes* your failure modes match their taxonomy. Ragas's RAG metrics won't surface "the agent apologized instead of escalating" as a failure mode; DeepEval's catalog won't surface a domain-specific policy violation unless you already knew to name it.

No dependency.

## Corrections

None.
