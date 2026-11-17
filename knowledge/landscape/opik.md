---
name: opik
description: Comet's open-source observability and evaluation platform with pytest integration and a prompt optimizer
shelf: landscape
updated: 2026-09-17
sources:
  - url: https://github.com/comet-ml/opik
    read: 2026-09-17
  - url: https://pypi.org/project/opik/
    read: 2026-09-17
license: Apache-2.0
related: [landscape-overview]
---

# Opik (Comet)

## What it is

Open-source LLM observability + evaluation platform, self-hostable, with pytest integration

## Core abstraction

A `Dataset` + `evaluate()` call using built-in or custom **metrics** (hallucination, moderation, RAG); Opik Agent Optimizer SDK for prompt optimization

## Where it sits

Application evals

Self-hostable fully free; also offers Comet-hosted cloud

## Maturity

As of 2026-09-17: Very active, 20k+ stars, positioned as a leading "true open source" alternative to Langfuse/LangSmith

## Biggest complaint

Younger platform relative to LangSmith/Langfuse — smaller integration ecosystem and community answers, even though the OSS story is arguably cleaner

## Relation to axial

- **LangSmith / Langfuse / Opik / Weave / Braintrust / Phoenix**: excellent at storing traces, running datasets and wiring up "an" LLM-as-judge. Their judge-authoring path is almost always "write a judge prompt and ship it"; deriving the judge's rubric from a labelled sample and measuring judge/human agreement (Cohen's kappa, TPR/TNR) before trusting it is rare. A few (Langfuse, Opik) let you attach human annotation queues and LLM judges *side by side*, the closest any tool gets, but none automates or scaffolds the *alignment* step (comparing judge verdicts to human labels, flagging disagreement, iterating the rubric) as a guided workflow. That step is what EvalGen's research prototype demonstrated and what practitioners do by hand in spreadsheets and notebooks.

No dependency.

## Corrections

None.
