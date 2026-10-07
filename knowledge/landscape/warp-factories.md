---
name: warp-factories
description: Warp's agent factory as files, with foreman, triage, spec, implement and review agents, automations, runners and self-grading scorers
shelf: landscape
updated: 2026-10-07
sources:
  - url: https://github.com/warpdotdev/warp-factory-examples
    read: 2026-10-07
  - url: https://docs.warp.dev/factories/
    read: 2026-10-07
license: proprietary
related: [landscape-overview, factory-lifecycle, methodology-spine]
---
# Warp Factories

## What it is

Warp's hosted software factory: a team of agents that takes a request from Slack,
Linear or GitHub to a review-ready pull request, defined entirely as files in a Git
repository. The product is proprietary; the example definitions are MIT and are the
source for everything below. Warp's own account of running it internally reports the
cost per PR falling from 80 to 30 dollars; that figure is from a secondary write-up of
Ben Holmes's post and was not verified against Warp's own text.

## Core abstraction

A definition root containing `factory.yaml` (repositories, integrations, agent
defaults), `agents/<name>/agent.md` (a prompt with frontmatter: `agentType` of
FOREMAN, TRIAGE, SPEC, IMPLEMENT, REVIEW or CUSTOM, an optional per-agent `model` or
`harness`, an optional `runner`), `automations/<name>/automation.md` (event triggers
with AND-combined filters, plus a prompt), `runners/<name>.yaml` (compute shapes and
images), `scorers/<name>/scorer.md` and skills at factory or agent scope. A resource's
name is its path. Exactly one agent is the foreman. A validator script checks a tree
before it is registered.

## Where it sits

An agent-platform scorer and orchestration product, not an eval library. The scorers
are the part that concerns evals: each declares the agents it samples, a label set with
a score per label, a `passingScore`, a `samplingRate` in percent, a judge `model`, and
an optional `selfImprovement` flag that feeds verdicts back into the agent it grades.

Account-required and hosted; execution can run on a self-hosted worker.

## Maturity

As of 2026-10-07: schema `v1alpha1`, early access, eight published example trees.

## Biggest complaint

Not applicable; too new for a practitioner consensus. The design observation is in the
next section.

## Relation to axial

The scorers are LLM judges and nothing in the definition validates one against human
labels before `selfImprovement` is turned on. The full-lifecycle example samples a
quarter of implementer runs with a three-label task-compliance scorer, self-improvement
on, and a quarter of reviewer runs with a review-quality scorer, both on `model:
claude-4-5-haiku`, an undated provider alias whose underlying snapshot can move, which
is the hazard axial's no-floating-aliases invariant exists to name. A self-improving
agent steered by an
unmeasured judge drifts toward whatever the judge rewards. axial is the step that
measures the judge first. The two label sets are three-valued where axial's judge is
binary, so a port needs a human decision about the middle label; no exporter exists.
No dependency.

The procedure the examples define is written up in
[factory-lifecycle](../procedures/factory-lifecycle.md).

## Corrections

None.
