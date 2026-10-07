---
name: evals-skills
description: The AI Evals course's agent skills, error-discovery and validate-evaluator, which count Pass as the positive class
shelf: landscape
updated: 2026-10-07
sources:
  - url: https://github.com/ai-evals-course/evals-skills
    read: 2026-10-07
  - url: https://andresrubio.github.io/ai-digest/topics/evals-and-benchmarks.html
    read: 2026-09-17
license: Apache-2.0
related: [landscape-overview]
---

# evals-skills

## What it is

**The AI Evals course's `evals-skills`** ([ai-evals-course/evals-skills](https://github.com/ai-evals-course/evals-skills), Apache-2.0) — agent skills for the same methodology. Its `error-discovery` skill has an agent build a review app, sort your free-text notes into failure modes while you review, and propose new samples; `validate-evaluator` walks the split / TPR / TNR / bias-correction steps as instructions. `axial` is the versioned artifact store and CI gate around the same ideas. Note the opposite convention: the skills treat *Pass* as the positive class, so their TPR is `axial`'s TNR.

## Core abstraction

Agent skills (SKILL.md files) installed with npx skills add.

## Where it sits

Application evals, as instructions an agent follows rather than a library.

## Maturity

As of 2026-10-07: history in the repo starts 2026-08-15 with commits by Hamel Husain.

## Biggest complaint

Not applicable.

## Relation to axial

axial is the versioned artifact store and CI gate around the same ideas. The skills count Pass as the positive class, so their TPR is axial's TNR. No dependency.

## Corrections

2026-09-17: **Anthropic's error-discovery skill.** Shipped 2026-08-23, installable via `npx skills add`. Builds a review app over a diverse trace sample and clusters annotations into failure modes. Reported by the ai-digest evals-and-benchmarks topic page: <https://andresrubio.github.io/ai-digest/topics/evals-and-benchmarks.html>. This is a secondary source (a digest of reporting on the skill, not Anthropic's own docs or repo), so treat the description above as reported, not confirmed firsthand.

2026-10-07: a skill matching this description is published at <https://github.com/ai-evals-course/evals-skills> (`skills/error-discovery/`, installed with `npx skills add`). Its history there starts on 2026-08-15 with commits by Hamel Husain, the first of which moves the skill into a plugin layout, so it existed before that date somewhere else. We found no `error-discovery` skill in Anthropic's public `anthropics/skills` repo and no match in a GitHub code search of the `anthropics` org. That makes a misattribution in the digest likely, not proven; the README now credits the repo where the skill can actually be read.
