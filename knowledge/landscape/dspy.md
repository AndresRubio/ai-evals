---
name: dspy
description: Stanford's framework that treats an eval metric as the optimisation objective for prompts and few-shots
shelf: landscape
updated: 2026-09-17
sources:
  - url: https://dspy.ai/diving-deeper/metrics-and-evaluation/
    read: 2026-09-17
  - url: https://github.com/stanfordnlp/dspy/blob/main/docs/docs/learn/evaluation/metrics.md
    read: 2026-09-17
license: MIT
related: [landscape-overview]
---

# DSPy (`stanfordnlp/dspy`) evaluation/optimizers

## What it is

Python framework for programming (not prompting) LLM pipelines; evaluation is a first-class primitive used to drive **optimization**

## Core abstraction

A **metric function** `(example, pred) -> score`, run via `dspy.Evaluate(devset, metric=...)`; optimizers (MIPROv2, BootstrapFewShot) search prompts/few-shots against that metric

## Where it sits

Application evals, but purpose-built to *optimize* a pipeline rather than just *report* on it

Fully local OSS

## Maturity

As of 2026-09-17: Very active, ~34k stars, on the 3.x line

## Biggest complaint

The metric doubles as an optimization objective, which tempts teams into over-fitting prompts to a small, under-validated metric before they've done real error analysis — evaluation is a means to auto-prompt-tuning, not an end in itself

## Relation to axial

- **DSPy**: comes closest to *acting* on a metric (via optimizers), but again assumes the metric is already correct; if the metric encodes the wrong thing, DSPy will optimize a prompt straight into that blind spot faster than anything else surveyed.

No dependency.

## Corrections

None.
