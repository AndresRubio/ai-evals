---
name: pytest-evals
description: A thin pytest plugin for running and aggregating LLM eval cases as ordinary tests in existing CI
shelf: landscape
updated: 2026-09-17
sources:
  - url: https://github.com/AlmogBaku/pytest-evals
    read: 2026-09-17
license: MIT
related: [landscape-overview]
---

# pytest-based approaches (e.g. `pytest-evals`, or plain pytest + assertions)

## What it is

A **pytest plugin** for running/aggregating LLM eval cases as tests

## Core abstraction

Ordinary `test_*` functions using pytest fixtures/parametrize, with `pytest-evals` adding aggregation/async/CI helpers

## Where it sits

Application evals

Fully local, no account

## Maturity

As of 2026-09-17: Small, single-maintainer project (`AlmogBaku/pytest-evals`); low but real activity

## Biggest complaint

Very thin — good for wiring evals into CI you already have, but you still have to write every metric/judge yourself; no dataset management, no dashboards, no built-in judge library

## Relation to axial

- **The "pytest for evals" tools (DeepEval, pytest-evals, promptfoo)**: are good at the *last mile* (gate CI on a fixed judge/assertion) but contribute nothing to the *first mile* (how did you decide the assertion was right, and how do you know the judge agrees with your humans at an acceptable rate before you trust it in CI).

No dependency.

## Corrections

None.
