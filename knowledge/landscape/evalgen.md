---
name: evalgen
description: Shankar et al.'s research prototype that picks judge prompts by agreement with human grades and names criteria drift
shelf: landscape
updated: 2026-09-17
sources:
  - url: https://arxiv.org/abs/2404.12272
    read: 2026-09-17
related: [landscape-overview]
---

# EvalGen

## What it is

A research prototype (Shankar, Zamfirescu-Pereira, Hartmann, Parameswaran, Arawjo, UIST 2024) that generates candidate judge prompts, uses human grades on a subset to select among them, and names criteria drift: graders' criteria change as they read outputs.

## Core abstraction

A candidate set of judge criteria graded against human labels.

## Where it sits

Application evals. A paper and prototype, not a package.

## Maturity

As of 2026-09-17: not maintained as software.

## Biggest complaint

Not applicable.

## Relation to axial

**EvalGen** (Shankar et al., UIST 2024) — demonstrated the judge-alignment step specifically: generate candidate judge prompts, use human grades on a subset to select among them, and named criteria drift along the way. A research prototype, not a maintained package, and scoped to that one step — not the open coding, taxonomy building, splits, or gating around it. `axial` is the shipped, full-pipeline version.

No dependency.

## Corrections

None.
