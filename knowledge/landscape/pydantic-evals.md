---
name: pydantic-evals
description: Pydantic AI's typed Case, Dataset, Evaluator and LLMJudge, with no judge-to-human agreement measurement
shelf: landscape
updated: 2026-10-07
sources:
  - url: https://pydantic.dev/docs/ai/evals/evals/
    read: 2026-09-18
  - url: https://pydantic.dev/docs/ai/evals/evaluators/llm-judge/
    read: 2026-09-18
  - url: https://pydantic.dev/docs/ai/evals/how-to/dataset-management/
    read: 2026-09-18
  - url: https://pydantic.dev/docs/ai/api/pydantic_evals/generation/
    read: 2026-09-18
  - url: https://github.com/pydantic/pydantic-ai
    read: 2026-09-18
license: MIT
related: [landscape-overview]
---

# Pydantic Evals

## What it is

The evals package of the Pydantic AI project. MIT, ~20k stars on the parent repo, and — importantly — it does *not* depend on `pydantic-ai`, and Logfire is an optional dependency, so it is fully usable locally with no account.

## Core abstraction

`Case` (one scenario: inputs, optional `expected_output`, metadata, case-specific evaluators), `Dataset` (cases plus dataset-level evaluators), `Evaluator` (base class scoring an output; deterministic or LLM-based), `LLMJudge` (built-in evaluator that grades against a rubric string, returning an assertion and/or a 0–1 score), and an `EvaluationReport`. There is a `generate_dataset()` that synthesises cases against a typed schema. Span-based evaluators can inspect OpenTelemetry spans to score tool calls and execution flow.

## Where it sits

Application evals. Fully local; Logfire is optional and pydantic-ai is not required.

## Maturity

As of 2026-09-18: MIT, about 20k stars on the parent repo, directional.

## Biggest complaint

Checked in the obvious places and not found:

- No discovery of failure modes from traces — no open coding, clustering or taxonomy
  building. Cases and evaluators are always supplied up front.
- No human annotation or labelling surface of any kind.
- **No judge/human agreement measurement.** No accuracy, TPR/TNR, kappa, calibration or
  disagreement review. The LLM-judge page's "Model Biases/Limitations" section advises
  validating judges against human-labelled test sets — as one sentence of prose. There is
  no API behind it.
- No train/dev/test split concept, and nothing about overfitting an evaluator to the data
  used to build it.

## Relation to axial

Pydantic Evals is the best-engineered instance of the pattern the synthesis above describes: excellent at the back half (define cases → run evaluators → report → gate CI), silent on the front half. It does not close the gap. It is the strongest evidence for it — the most type-rigorous eval library in the ecosystem tells you to validate your judge against human labels, and then hands you nothing to do it with.

No dependency.

## Corrections

None.
