---
name: harbor
description: Terminal-Bench's harness for running agents on tasks in fresh sandboxes, scored by per-task verifiers and LLM judges with no documented human check
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
  - url: https://docs.harborframework.com/rewardkit/built-in-criteria.md
    read: 2026-10-28
  - url: https://docs.harborframework.com/rewardkit/quick-start.md
    read: 2026-10-28
  - url: https://docs.harborframework.com/tutorials/create-a-verifier-with-rewardkit.md
    read: 2026-10-28
  - url: https://docs.harborframework.com/tasks/verifier.md
    read: 2026-10-28
  - url: https://docs.harborframework.com/jobs/custom-verifiers.md
    read: 2026-10-28
  - url: https://docs.harborframework.com/jobs/regrade.md
    read: 2026-10-28
  - url: https://docs.harborframework.com/datasets/metrics.md
    read: 2026-10-28
  - url: https://docs.harborframework.com/agents/pre-integrated-agents.md
    read: 2026-10-28
  - url: https://github.com/harbor-framework/harbor/blob/main/src/harbor/agents/nop.py
    read: 2026-10-28
  - url: https://github.com/harbor-framework/harbor/blob/main/src/harbor/cli/analyze.py
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

Open-source framework for running agents on tasks in fresh sandboxes and scoring what they leave behind, from the creators of Terminal-Bench. It is the official harness for Terminal-Bench 2.0, and `harbor datasets list` shows the third-party benchmarks it can run, such as SWE-Bench and Aider Polyglot. Ships as a CLI (`uv tool install harbor`) that orchestrates from your laptop or a CI runner while the agent runs in the sandbox.

## Core abstraction

A **task** is a directory: `instruction.md` (what the agent is told), `task.toml`, `environment/` (a `Dockerfile`, `docker-compose.yaml` or `Apptainer.def`, seeded with the data the task needs), `tests/test.sh` (the verifier) and an optional `solution/solve.sh`. The verifier runs after the agent finishes and writes a number to `/logs/verifier/reward.txt`, or labelled rewards to `reward.json`. A **trial** is one attempt; a **job** runs every task in a dataset for k trials per agent and model, from which the Arize talk computes pass@k.

The **oracle** agent (`harbor run -a oracle`) runs `solve.sh` in place of a model, so an author can check a task is solvable and its verifier awards 1 before spending tokens. The **`nop`** agent's run does nothing (`src/harbor/agents/nop.py`), so running it shows whether a verifier fails an untouched environment; the docs list it without saying what it is for. Agents can record each trial as an ATIF (Agent Trajectory Interchange Format, currently v1.7) `trajectory.json`: ordered steps with messages, tool calls and their observations, plus optional token, cost and step totals.

**Rewardkit** builds verifiers from 20+ built-in criteria (files, commands, JSON, CSV, HTTP, images, trajectories) and from LLM or agent judges defined in TOML. Judge scales are binary, Likert, numeric or rubric, normalised to [0, 1]; the default judge model is `anthropic/claude-opus-5-5`.

Harbor's own quality checks are agent judges too. `harbor check` has an agent grade a task against a rubric, and `harbor analyze` has one review finished trials, by default for reward hacking and task-specification problems (`src/harbor/cli/analyze.py`).

## Where it sits

Both families. With the registry it is a benchmark runner; with your own tasks it is an end-to-end regression suite for an agent and whatever surrounds it. The canonical task is a coding agent on a git repository at a commit, but the environment is anything that fits in a container. Arize puts the whole Phoenix server and a seeded SQLite file in one image, then tests Claude Code with their CLI, Claude Code with their MCP server, and their own server-side agent against the same tasks (talk, 2026-10-26).

Runs locally on Docker by default, or in parallel on a sandbox provider (Daytona, Modal and others the README lists); Arize reports about 200 concurrent workers on Daytona. Also generates rollouts for RL. Has a local results UI and a hosted service, Harbor Hub.

## Maturity

As of 2026-10-28: 6,582 GitHub stars and 1,899 commits on `main`, repository created 2025-08-04. PyPI has 80 releases and 114 dev builds; 0.25.0 came out 2026-10-25 and the first, 0.1.0, on 2025-11-02. Pre-1.0.

## Biggest complaint

Writing verifiers is the work. Rewardkit's own design page opens by naming the verifier as often the most tedious part of writing a Harbor task, and benchmark verifiers in general as hard to read, extend and reuse, which leads to copied boilerplate and subtle errors. `harbor lint` checks configuration, required files and sandbox safety, and its docs add that linting does not guarantee the task or verifier is correct.

The system under test also has to fit in a container. Asked about a distributed system backed by S3, the Arize speaker said that case is harder and that mocking those calls puts you back where Harbor was meant to take you away from.

## Relation to axial

Harbor starts where the scorer already exists, like the rest of this shelf: each task ships with a verifier written before the first run.

The docs read for this entry describe no way to compare a judge with human labels. Rewardkit's `samples` option runs a judge several times and reports how often it agrees with itself in `reward-details.json`. None of the Rewardkit pages (judge criteria, built-in criteria, quick start, design, the verifier tutorial), the verifier, custom-verifier, regrade and metrics pages, or `analyze.py` mentions human labels or agreement with people, and neither does any title or summary in the docs index.

The oracle and `nop` agents cover the two easy cases. The oracle shows a correct solution scores 1, and `nop` shows whether an untouched environment scores 0. Neither tests a plausible wrong answer, which is where an LLM judge goes wrong. In the talk's case study some verifiers were LLM judges, and the speaker trusted them because they passed against the oracle. Arize published a separate SQL-tools benchmark (Arize blog, August 2026) that shows the kind of error this misses: its scoring model marked a count of 622 tool spans correct when the true count was 626. `axial align` measures both rates against human labels, so `axial` sits upstream of a Rewardkit judge the way it sits upstream of an [Inspect AI](inspect-ai.md) scorer.

Its tasks come from wherever the author finds them. In the case study Claude brainstormed the questions across Phoenix's domains and wrote the oracle answers, and the speaker spot-checked the questions. The practitioner advice collected here is to draw an agent's first 20 to 50 tasks from real failures ([numbers-and-rules-of-thumb](../methodology/numbers-and-rules-of-thumb.md)).

It takes a side in the trajectory-versus-outcome argument ([points-of-disagreement](../methodology/points-of-disagreement.md), item h). A canonical verifier grades only the final state. When Arize wanted fewer tool calls they wrote verifiers that read the trajectory. The speaker agreed this is cheating by Harbor's own standard and makes the test more likely to break when the agent changes, and kept it because the goal was a metric; their correctness regression tests read only the final state.

Each trial's ATIF `trajectory.json` is documented JSON with messages, tool calls and observations, which makes Harbor jobs a possible source of agent traces for `axial ingest`. No adapter exists and none has been tried.

No dependency.

## Corrections

2026-10-28: the first version, written the same day from tool summaries of these pages, said Harbor's judges are never checked against humans and that the oracle leaves no true-negative check. Rechecked against the raw docs and source. The human-label claim is now limited to the pages listed in the sources. The `nop` agent, which can show a verifier fails an untouched environment, and the `harbor check` and `harbor analyze` agent judges were missing and are added. Star, commit and release counts are now exact.
