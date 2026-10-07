---
name: helm
description: Stanford CRFM's holistic foundation-model benchmark framework, in maintenance mode since June 2026
shelf: landscape
updated: 2026-09-17
sources:
  - url: https://github.com/stanford-crfm/helm
    read: 2026-09-17
license: Apache-2.0
related: [landscape-overview]
---

# HELM (Stanford CRFM)

## What it is

Academic Python framework + public leaderboard for holistic foundation-model benchmarking

## Core abstraction

A **RunSpec** (scenario + adapter + metrics) registered in the framework's config

## Where it sits

Foundation-model benchmarking (broad: accuracy, calibration, robustness, fairness, bias, toxicity, efficiency)

Local-first framework; results also published to a public leaderboard/account-free website

## Maturity

As of 2026-09-17: **Entered maintenance mode June 1, 2026** — largely superseded in day-to-day use by Inspect AI and lm-eval-harness for new benchmarking work

## Biggest complaint

It's heavy, academically oriented, and now unmaintained for new development — not something an app team would reach for; useful mainly as a leaderboard/reference, not a working tool

## Relation to axial

- **promptfoo / OpenAI Evals / lm-eval-harness / HELM**: you write the assertion or task spec *first*. There's no workflow step for "I don't yet know what 'good' means for my traces — help me discover it."

- **lm-eval-harness / HELM**: solve a genuinely different problem (fixed benchmark, known ground truth) where error-analysis-first doesn't even apply the same way — there's no "discover your own failure taxonomy" step because the taxonomy (the benchmark) is handed to you.

No dependency.

## Corrections

None.
