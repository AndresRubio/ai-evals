---
name: helm
description: Stanford CRFM's holistic foundation-model benchmark framework, in maintenance mode since June 2026
shelf: landscape
updated: 2026-11-17
sources:
  - url: https://github.com/stanford-crfm/helm
    read: 2026-09-17
license: Apache-2.0
related: [landscape-overview, lm-evaluation-harness]
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

As of 2026-11-11, lm-eval-harness had no commits on `main` since 2026-09-14 and no release since v0.4.13 (2026-08-31); see [lm-evaluation-harness](lm-evaluation-harness.md).

## Biggest complaint

It's heavy, academically oriented, and now unmaintained for new development — not something an app team would reach for; useful mainly as a leaderboard/reference, not a working tool

## Relation to axial

- **promptfoo / OpenAI Evals / lm-eval-harness / HELM**: you write the assertion or task spec *first*. None has a workflow step for "I don't yet know what 'good' means for my traces; help me discover it."

- **lm-eval-harness / HELM**: solve a different problem (a fixed benchmark with known ground truth), where error analysis first doesn't apply the same way. There is no "discover your own failure taxonomy" step, because the taxonomy (the benchmark) is handed to you.

No dependency.

## Corrections

None.
