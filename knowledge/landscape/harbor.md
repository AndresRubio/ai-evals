---
name: harbor
description: Terminal-Bench's harness for running agents on tasks in fresh sandboxes, scored by per-task verifiers and LLM judges never checked against humans
shelf: landscape
updated: 2026-10-28
sources:
  - url: https://github.com/harbor-framework/harbor
    read: 2026-10-28
  - url: https://pypi.org/project/harbor/
    read: 2026-10-28
  - url: https://docs.harborframework.com/tasks/overview.md
    read: 2026-10-28
  - url: https://docs.harborframework.com/tasks/solution.md
    read: 2026-10-28
  - url: https://docs.harborframework.com/tasks/lint.md
    read: 2026-10-28
  - url: https://docs.harborframework.com/agents/atif.md
    read: 2026-10-28
  - url: https://docs.harborframework.com/rewardkit/judge-criteria.md
    read: 2026-10-28
  - url: https://docs.harborframework.com/rewardkit/motivation-and-design.md
    read: 2026-10-28
  - url: https://www.youtube.com/watch?v=kRJjSOYqW0I
    read: 2026-10-28
  - url: https://arize.com/blog/phoenix-mcp-sql-code-mode/
    read: 2026-10-28
license: Apache-2.0
related: [landscape-overview, inspect-ai, arize-phoenix, points-of-disagreement]
---

# Harbor

## What it is

Open-source framework for running agents on tasks in fresh sandboxes and scoring what they leave behind, from the creators of Terminal-Bench. It is the official harness for Terminal-Bench 2.0, and other benchmarks such as SWE-Bench load through `harbor datasets list`. Ships as a CLI (`uv tool install harbor`) that orchestrates from your laptop or a CI runner while the agent runs in the sandbox.

## Core abstraction

A **task** is a directory: `instruction.md` (what the agent is told), `task.toml`, `environment/` (a `Dockerfile`, `docker-compose.yaml` or `Apptainer.def`, seeded with the data the task needs), `tests/test.sh` (the verifier) and an optional `solution/solve.sh`. The verifier runs after the agent finishes and writes a number to `/logs/verifier/reward.txt`, or labelled rewards to `reward.json`. A **trial** is one attempt; a **job** runs every task in a dataset for k trials per agent and model, so pass@k falls out of it.

The **oracle** agent (`harbor run -a oracle`) runs `solve.sh` in place of a model, so an author can check a task is solvable and its verifier awards 1 before spending tokens. Agents can record each trial as an ATIF (Agent Trajectory Interchange Format, currently v1.7) `trajectory.json`: ordered steps with messages, tool calls and their observations, plus optional token, cost and step totals.

**Rewardkit** builds verifiers from 20+ built-in criteria (files, commands, JSON, CSV, HTTP, images, trajectories) and from LLM or agent judges defined in TOML. Judge scales are binary, Likert, numeric or rubric, normalised to [0, 1]; the default judge model is `anthropic/claude-opus-5-5`.

## Where it sits

Both families. With the registry it is a benchmark runner; with your own tasks it is an end-to-end regression suite for an agent and whatever surrounds it. The canonical task is a coding agent on a git repository at a commit, but the environment is anything that fits in a container. Arize puts the whole Phoenix server and a seeded SQLite file in one image, then tests Claude Code with their CLI, Claude Code with their MCP server, and their own server-side agent against the same tasks (talk, 2026-10-26).

Runs locally on Docker by default, or in parallel on a sandbox provider (Daytona, Modal and others the README lists); Arize reports about 200 concurrent workers on Daytona. Also generates rollouts for RL. Has a local results UI and a hosted service, Harbor Hub.

## Maturity

As of 2026-10-28: about 6.6k GitHub stars and 1,898 commits; PyPI 0.25.0 released 2026-10-25, first release 0.1.0 on 2025-11-02, plus dated dev builds between releases. Pre-1.0.

## Biggest complaint

Writing verifiers is the work. Rewardkit's own design page opens by calling Harbor verifiers tedious to write, and benchmark verifiers in general hard to read and reuse, which breeds copied boilerplate and subtle errors. `harbor lint` checks configuration, required files and sandbox safety, and its docs add that a clean lint says nothing about whether a verifier is correct.

The system under test also has to fit in a container. Asked about a distributed system backed by S3, the Arize speaker said that case is harder and that mocking those calls puts you back where Harbor was meant to take you away from.

## Relation to axial

Harbor starts where the scorer already exists, like the rest of this shelf: each task ships with a verifier written before the first run.

Its judges are never measured against people. Rewardkit's `samples` option runs a judge several times and reports how often it agrees with itself in `reward-details.json`; nothing in the judge-criteria or design pages compares a judge with human labels. The oracle does not close the gap. It shows that a correct solution scores 1, which tests that a verifier passes good output and never that it fails bad output: a true-positive check with no true-negative one. In the talk's case study some verifiers were LLM judges, and the speaker trusted them because they passed against the oracle. Arize's Phoenix team published a separate SQL-tools benchmark (Arize blog, August 2026) that shows the kind of error a pass-only check lets through: its scoring model marked a count of 622 tool spans correct when the true count was 626. `axial align` measures both rates against human labels, so `axial` sits upstream of a Rewardkit judge the way it sits upstream of an [Inspect AI](inspect-ai.md) scorer.

Its tasks come from wherever the author finds them. In the case study Claude brainstormed the questions across Phoenix's domains and wrote the oracle answers, and the speaker spot-checked the questions. The practitioner advice collected here is to draw an agent's first 20 to 50 tasks from real failures ([numbers-and-rules-of-thumb](../methodology/numbers-and-rules-of-thumb.md)).

It takes a side in the trajectory-versus-outcome argument ([points-of-disagreement](../methodology/points-of-disagreement.md), item h). A canonical verifier grades only the final state. When Arize wanted fewer tool calls they wrote verifiers that read the trajectory. The speaker agreed this is cheating by Harbor's own standard and makes the test more likely to break when the agent changes, and kept it because the goal was a metric; their correctness regression tests read only the final state.

Each trial's ATIF `trajectory.json` is documented JSON with messages, tool calls and observations, which makes Harbor jobs a possible source of agent traces for `axial ingest`. No adapter exists and none has been tried.

No dependency.

## Corrections

None.
