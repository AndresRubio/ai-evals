---
name: lm-evaluation-harness
description: EleutherAI's standard harness for reproducible few-shot benchmark evaluation of foundation models
shelf: landscape
updated: 2026-09-17
sources:
  - url: https://github.com/EleutherAI/lm-evaluation-harness
    read: 2026-09-17
  - url: https://pypi.org/project/lm-eval/
    read: 2026-09-17
license: MIT
related: [landscape-overview]
---

# lm-evaluation-harness (EleutherAI)

## What it is

Python CLI/library, the standard harness for reproducible few-shot benchmark evaluation (used to produce most public leaderboard numbers)

## Core abstraction

A **YAML task config** (prompt template, dataset, metric) registered in the task registry

## Where it sits

Foundation-model benchmarking

Fully local OSS

## Maturity

As of 2026-09-17: Very active, ~13k stars, v0.4.x line, adds vLLM/HF-parallel support

## Biggest complaint

Task-YAML system is powerful but arcane to extend correctly (templating/answer-extraction edge cases); installing from PyPI (`lm-eval`) is discouraged vs. an editable clone because of how fast internals move

## Relation to axial

- **promptfoo / OpenAI Evals / lm-eval-harness / HELM**: you write the assertion or task spec *first*. There's no workflow step for "I don't yet know what 'good' means for my traces — help me discover it."

- **lm-eval-harness / HELM**: solve a genuinely different problem (fixed benchmark, known ground truth) where error-analysis-first doesn't even apply the same way — there's no "discover your own failure taxonomy" step because the taxonomy (the benchmark) is handed to you.

No dependency.

## Corrections

None.
