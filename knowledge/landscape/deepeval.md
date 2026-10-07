---
name: deepeval
description: Pytest-style LLM eval library with 60+ built-in metrics and a paid Confident AI dashboard above it
shelf: landscape
updated: 2026-09-17
sources:
  - url: https://pypi.org/project/deepeval/
    read: 2026-09-17
  - url: https://techsy.io/en/blog/confident-ai-review
    read: 2026-09-17
license: Apache-2.0
related: [landscape-overview]
---

# DeepEval (Confident AI)

## What it is

Python library, "pytest for LLMs," with a paid hosted layer (Confident AI) for dashboards

## Core abstraction

A **pytest test** using `assert_test(test_case, metrics)`; also a plain Python API

## Where it sits

Application evals (RAG, chatbot, agent, safety metrics — 60+ built-in)

Local-first OSS core; Confident AI platform is the account-required upsell

## Maturity

As of 2026-09-17: Very active, one of the most popular OSS eval libs (GitHub stars reports vary 1.8k–18k depending on source/fork confusion, but it's a top-tier project); frequent PyPI releases

## Biggest complaint

Metrics feel like a grab-bag/checkbox list decoupled from your own failure modes; heavy push toward the commercial Confident AI platform for anything beyond basic runs

## Relation to axial

- **DeepEval / Ragas / autoevals / Evidently**: ship a fixed catalog of pre-named metrics (faithfulness, toxicity, hallucination, relevance…). This is useful, but it *presupposes* your failure modes match their taxonomy. Ragas's RAG metrics won't surface "the agent apologized instead of escalating" as a failure mode; DeepEval's catalog won't surface a domain-specific policy violation unless you already knew to name it.

- **The "pytest for evals" tools (DeepEval, pytest-evals, promptfoo)**: are good at the *last mile* (gate CI on a fixed judge/assertion) but contribute nothing to the *first mile* (how did you decide the assertion was right, and how do you know the judge agrees with your humans at an acceptable rate before you trust it in CI).

No dependency.

## Corrections

None.
