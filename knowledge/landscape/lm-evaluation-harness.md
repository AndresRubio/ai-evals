---
name: lm-evaluation-harness
description: EleutherAI's standard harness for reproducible few-shot benchmark evaluation of foundation models
shelf: landscape
updated: 2026-11-16
sources:
  - url: https://github.com/EleutherAI/lm-evaluation-harness
    read: 2026-11-11
  - url: https://pypi.org/project/lm-eval/
    read: 2026-11-11
  - url: https://github.com/EleutherAI/lm-evaluation-harness/releases
    read: 2026-11-11
  - url: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/evaluator.py
    read: 2026-11-11
  - url: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/loggers/evaluation_tracker.py
    read: 2026-11-11
  - url: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/metrics.py
    read: 2026-11-11
  - url: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/decontamination.md
    read: 2026-11-11
  - url: https://arxiv.org/abs/2405.14782
    read: 2026-11-11
license: MIT
related: [landscape-overview]
---

# lm-evaluation-harness (EleutherAI)

## What it is

Python CLI/library, the standard harness for reproducible few-shot benchmark evaluation (used to produce most public leaderboard numbers)

## Core abstraction

A **YAML task config** (prompt template, dataset, metric) registered in the task registry

## Where it sits

Foundation-model benchmarking

Fully local OSS

## Maturity

As of 2026-11-11: about 14.5k stars, MIT. The latest release is v0.4.13 (2026-08-31). PyPI also holds a 0.5.0.dev1 pre-release, uploaded earlier, on 2026-05-11. Nothing has been committed to `main` since 2026-09-14.

## Biggest complaint

Task-YAML system is powerful but arcane to extend correctly (templating/answer-extraction edge cases); installing from PyPI (`lm-eval`) is discouraged vs. an editable clone because of how fast internals move

The decontamination docs describe a 13-gram overlap check run against a directory of n-gram files, but their usage example passes no such directory, and `evaluator.py` never calls the decontamination code.

## Relation to axial

With promptfoo, OpenAI Evals, lm-eval-harness and HELM you write the assertion or task spec *first*. None has a workflow step for "I don't yet know what 'good' means for my traces; help me discover it."

lm-eval-harness and HELM solve a different problem (a fixed benchmark with known ground truth), where error analysis first doesn't apply the same way. There is no "discover your own failure taxonomy" step, because the taxonomy (the benchmark) is handed to you.

Two things carry over. With `--log_samples` the harness writes a SHA-256 of each sample's document, prompt and target, and rolls the samples' hashes into one per task. It also records hashes of the chat template and the system instruction. That is the precedent for binding a judge's verdict to the exact input it judged. Its statistics stop at a standard error: closed-form for `mean` and `acc_all`, a bootstrap for a fixed list of metrics (median, F1, Matthews correlation, perplexity, BLEU and others) when `bootstrap_iters` is above zero, and none for the rest. It reports no confidence intervals and no paired comparisons, where `axial` reports Wilson intervals and paired differences.

The paper to cite is Biderman, Schoelkopf, Sutawika et al., "Lessons from the Trenches on Reproducible Evaluation of Language Models" (arXiv 2405.14782, v3 of 2026-05-31), written by the harness's maintainers. It reports, citing earlier work, that small implementation details change scores, "even down to whitespace usage" (§3.1). It tells authors to share their exact prompts (§3.2, Appendix D) and to provide their model outputs (Appendix D), describes saving outputs with the harness's `--log_samples` (§3.3), and recommends reporting variance or statistical significance over single point estimates (§4.2). Its evidence comes from benchmarking base models, so applying it to judges is `axial`'s inference.

No dependency.

## Corrections

2026-11-16: the 2026-09-17 version called the project very active on a v0.4.x line with about 13k stars. Nothing has been committed to `main` since 2026-09-14, so Maturity now gives the latest release, the pre-release and that date.
