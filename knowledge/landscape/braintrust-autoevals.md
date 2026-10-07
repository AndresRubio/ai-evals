---
name: braintrust-autoevals
description: Braintrust's closed eval platform and autoevals, its open scorer library for heuristic, embedding and LLM-judge scores
shelf: landscape
updated: 2026-09-17
sources:
  - url: https://github.com/braintrustdata/autoevals
    read: 2026-09-17
  - url: https://pypi.org/project/autoevals/
    read: 2026-09-17
license: proprietary
related: [landscape-overview]
---

# Braintrust / autoevals

## What it is

Braintrust = proprietary SaaS eval/observability platform; `autoevals` = its OSS scorer library (Python + TS). autoevals itself is MIT; the licence above is the platform's.

autoevals, the standalone OSS scorer library, is covered here with the platform it ships with.

## Core abstraction

An `Eval()` function: `data`, `task`, `scores[]` (autoevals scorers or custom)

## Where it sits

Application evals

`autoevals` is local-first OSS; full Braintrust experience requires an account

## Maturity

As of 2026-09-17: `autoevals` active, e.g. release Apr 2026, Python ≥3.10

## Biggest complaint

The genuinely open part (`autoevals`) is just a scorer library — the actual eval-running, dataset, and dashboard experience people mean when they say "Braintrust" is proprietary and pushes you toward their paid platform

## Relation to axial

- **DeepEval / Ragas / autoevals / Evidently**: ship a fixed catalog of pre-named metrics (faithfulness, toxicity, hallucination, relevance…). This is useful, but it *presupposes* your failure modes match their taxonomy. Ragas's RAG metrics won't surface "the agent apologized instead of escalating" as a failure mode; DeepEval's catalog won't surface a domain-specific policy violation unless you already knew to name it.

- **LangSmith / Langfuse / Opik / Weave / Braintrust / Phoenix**: excellent at storing traces, running datasets, and wiring up "an" LLM-as-judge — but the judge-authoring UX is almost universally "write a judge prompt and ship it," not "systematically derive the judge's rubric from a labeled sample and measure judge/human agreement (Cohen's kappa, TPR/TNR) before trusting it." A few (Langfuse, Opik) let you attach human annotation queues and LLM judges *side by side*, which is the closest any tool gets — but none automate or even scaffold the *alignment* step (comparing judge verdicts to human labels, flagging disagreement, iterating the rubric) as a first-class, guided workflow. That alignment step is exactly what EvalGen's research prototype demonstrated and what practitioners currently do by hand in spreadsheets/notebooks.

No dependency.

## Corrections

None.
