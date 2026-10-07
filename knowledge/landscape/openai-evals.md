---
name: openai-evals
description: OpenAI's slowing open-source eval registry and the hosted Evals API its docs now steer users to
shelf: landscape
updated: 2026-09-17
sources:
  - url: https://github.com/openai/evals
    read: 2026-09-17
  - url: https://developers.openai.com/cookbook/examples/evaluation/getting_started_with_openai_evals
    read: 2026-09-17
license: MIT
related: [landscape-overview]
---

# OpenAI Evals (`openai/evals`) + hosted Evals API

## What it is

GitHub OSS registry/framework (`pip install evals`) plus OpenAI's hosted Evals API

## Core abstraction

A **YAML/JSON eval spec** registered in the repo, or an `Eval`/`Run`/`data_source_config`/testing-criteria object via the API

## Where it sits

Mixed — the OSS repo leans benchmark-registry style; the hosted API leans application evals for API customers

OSS repo is local; the Evals API requires an OpenAI account/key

## Maturity

As of 2026-09-17: OSS repo has slowed (community treats the hosted Evals API as the maintained path); ~17.6k stars, 2.9k forks

## Biggest complaint

The open-source repo is semi-abandoned relative to the hosted API — OpenAI's own docs now steer users to the Evals API, so the "open source" story is really a thin, aging registry plus a vendor-locked product

## Relation to axial

- **promptfoo / OpenAI Evals / lm-eval-harness / HELM**: you write the assertion or task spec *first*. There's no workflow step for "I don't yet know what 'good' means for my traces — help me discover it."

No dependency.

## Corrections

None.
