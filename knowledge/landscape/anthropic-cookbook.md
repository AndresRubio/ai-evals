---
name: anthropic-cookbook
description: Anthropic's eval notebooks and guidance, rubrics and Claude as judge; recipes to copy, nothing to install
shelf: landscape
updated: 2026-09-17
sources:
  - url: https://github.com/anthropics/anthropic-cookbook/blob/main/misc/building_evals.ipynb
    read: 2026-09-17
license: MIT
related: [landscape-overview]
---

# Anthropic eval tooling / cookbook

## What it is

Not a packaged library — a set of **notebooks/patterns** in `anthropic-cookbook`/`claude-cookbooks` (e.g. `building_evals.ipynb`, the evaluator-optimizer agent pattern) plus prose guidance (rubrics + LLM judges, IRT-based criterion vetting)

## Core abstraction

Whatever you copy out of the notebook — typically a hand-rolled **Python function that calls Claude as a judge** against a rubric

## Where it sits

Application evals (guidance-first, not tool-first)

Fully local (notebooks); needs an Anthropic API key to run

## Maturity

As of 2026-09-17: Active as documentation, not as a product — no installable package, just recipes

## Biggest complaint

There is no actual tool here — teams have to hand-build the harness the cookbook describes from scratch each time; nothing to `pip install`

## Relation to axial

axial implements the rubric-plus-judge guidance the cookbook gives as notebooks, and adds the alignment measurement the notebooks do not. No dependency.

## Corrections

None.
