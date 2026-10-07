---
name: evidently
description: Open-source ML and LLM evaluation and monitoring; Reports and Tests, descriptors, LLM judges, tracing, scheduled evals
shelf: landscape
updated: 2026-10-07
sources:
  - url: https://docs.evidentlyai.com/llms.txt
    read: 2026-10-07
  - url: https://www.evidentlyai.com/
    read: 2026-10-07
  - url: https://github.com/evidentlyai/evidently
    read: 2026-09-17
  - url: https://www.evidentlyai.com/llm-guide/llm-evaluation
    read: 2026-09-17
license: Apache-2.0
related: [landscape-overview, evals-in-ci, production-monitoring]
---
# Evidently

## What it is

An open-source Python library for evaluating and monitoring both classic ML systems
(data drift, data quality, predictive performance) and LLM applications (chatbots, RAG,
agents), with over 100 built-in metrics and a declarative testing API, plus a platform,
self-hosted or Evidently Cloud, for storing runs, datasets, traces and dashboards. Its
documentation is published as an `llms.txt` index of one page per topic, which this
knowledge base copies.

## Core abstraction

A Report runs metrics over a dataframe and summarises them. Tests are the same metrics
with pass or fail conditions attached. Descriptors are row-level evaluations of text,
including LLM-as-judge descriptors configured by a prompt. Datasets hold test cases, and
a synthetic data API generates edge-case and adversarial inputs. Tracely captures live
LLM inputs and outputs for later evaluation.

## Where it sits

Mixed: application evals bolted onto a larger ML-monitoring product, and the only one
in this shelf that pairs LLM evals with ML-style drift and data-quality monitoring. The
documented workflows are
evals in GitHub Actions, scheduled evals on the platform, batch monitoring jobs over
traced traffic, dashboards and alerts.

The library is fully local; the platform is self-hosted or cloud.

## Maturity

As of 2026-10-07: Apache-2.0, 7,500 or more GitHub stars and 40 million downloads by
its own count, directional; active, with a broad user base carried over from its
ML-monitoring roots.

## Biggest complaint

LLM eval feels like a secondary feature grafted onto a data-drift-monitoring tool, with
less depth on LLM-specific failure modes, such as hallucination nuance and agent
trajectories, than dedicated LLM-eval libraries.

## Relation to axial

Evidently ships a catalogue of pre-named metrics and asks you to write a judge prompt;
it does not derive the judge from your failures or measure its agreement with your
humans. axial produces the judge Evidently would run. What axial takes from it: the
index pattern for this knowledge base, Tests as the pass-or-fail form of a Report, and
the two procedures [evals-in-ci](../procedures/evals-in-ci.md) and
[production-monitoring](../procedures/production-monitoring.md). A writer that emits
axial verdicts as an Evidently descriptor column, so their dashboards can show a
validated judge, is a possible optional extra, not built. No dependency.

## Corrections

None.
