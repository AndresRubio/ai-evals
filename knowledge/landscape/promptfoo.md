---
name: promptfoo
description: CLI and Node library for prompt and model testing and red-teaming, driven by a YAML config of prompts, providers, tests and asserts
shelf: landscape
updated: 2026-09-17
sources:
  - url: https://github.com/promptfoo/promptfoo/releases
    read: 2026-09-17
  - url: https://www.promptfoo.dev/docs/releases/
    read: 2026-09-17
license: MIT
related: [landscape-overview]
---

# promptfoo

## What it is

CLI + Node library (Python usable as a provider/wrapper) for prompt/model testing and LLM red-teaming

## Core abstraction

A **YAML config**: `prompts`, `providers`, `tests[]`, `assert[]`

## Where it sits

Application evals (+ a strong red-teaming/security-scan mode)

Local-first CLI; optional hosted dashboard

## Maturity

As of 2026-09-17: Very active; v0.121.x, releases roughly weekly, e.g. 0.121.19 (Jul 2026)

## Biggest complaint

It's a Node/JS-ecosystem tool at its core — Python teams have to context-switch into YAML + npm tooling, and there's no built-in dashboard for organizing datasets/results over time

## Relation to axial

- **promptfoo / OpenAI Evals / lm-eval-harness / HELM**: you write the assertion or task spec *first*. There's no workflow step for "I don't yet know what 'good' means for my traces — help me discover it."

- **The "pytest for evals" tools (DeepEval, pytest-evals, promptfoo)**: are good at the *last mile* (gate CI on a fixed judge/assertion) but contribute nothing to the *first mile* (how did you decide the assertion was right, and how do you know the judge agrees with your humans at an acceptable rate before you trust it in CI).

No dependency.

## Corrections

None.
