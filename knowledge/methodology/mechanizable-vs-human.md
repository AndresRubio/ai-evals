---
name: mechanizable-vs-human
description: Which eval steps a tool should own outright, which must stay human, and the grey zone between
shelf: methodology
updated: 2026-09-17
sources:
  - url: https://hamel.dev/blog/posts/evals-faq/
    read: 2026-09-17
  - url: https://arxiv.org/abs/2404.12272
    read: 2026-09-17
  - url: https://eugeneyan.com/writing/aligneval/
    read: 2026-09-17
related: [methodology-spine]
---

# Mechanizable in Python vs irreducibly human

## Mechanizable — a tool should own these outright

**Trace plumbing.** Capture, normalize, version and index traces; link every eval run to a prompt version, dataset version and code commit. Git-backed prompt storage, not a SaaS indirection layer.

**Sampling.** Random, stratified, embedding-clustered (k-means/HDBSCAN over trace embeddings), outlier-ranked on computable signals (latency, token count, turn count, retry count, tool-error count), classifier-flagged, feedback-filtered, and semantic search. The tool should *enforce* that every review batch contains a random slice, since every targeted method is biased toward known problems.

**Annotation UI generation.** Progressive disclosure, domain-specific renderers, hotkeys, progress counters, filters, next/prev navigation, one-click note capture, resumable sessions. This is the highest-ROI thing to build and it is pure engineering — it holds no judgment.

**Saturation instrumentation.** Plot new-unique-codes per trace reviewed. Theoretical saturation is the stated stopping rule in every source but nobody gives a number; a tool can turn it into a visible curve and a suggested stop.

**Taxonomy bookkeeping.** Counts, frequency × severity pivot tables, ranked failure modes, coverage of taxonomy over the trace pool, drift in category proportions over time.

**Synthetic generation scaffolding.** Dimension registry (features × scenarios × personas), cross-product enumeration, sampling with coverage constraints, tuple → natural-language conversion, near-duplicate filtering, execution of generated inputs through the real system to capture traces, and a check that each generated case actually triggered its intended scenario.

**Dataset management.** Stratified train/dev/test splitting with class-balance constraints (10–20 / 40–45 / 40–45; ≥30 Pass and ≥30 Fail per split), frozen test manifests with hashes, leakage detection between splits, and a hard refusal to let a test row be touched by the optimization loop.

**Judge harness.** Prompt templating, few-shot injection drawn from labeled critiques in the train split only, concurrency, caching, retry, cost accounting.

**All measurement.** Confusion matrix, TPR, TNR, precision, recall, F1, Cohen's κ, bootstrap confidence intervals, per-slice breakdown by feature/scenario/persona, dev→test generalization gap. The tool should refuse to print raw agreement without TPR/TNR beside it.

**Bias and robustness checks.** Position-swap tests for pairwise judges, correlation of verdict with output length (verbosity bias), self-preference check by swapping the judge model, prompt-paraphrase sensitivity, and **context ablation** — drop one context element at a time and recompute alignment.

**Bounded judge-prompt optimization.** Search over judge prompt variants on dev only, with a trial budget, always reporting the dev→test gap. AlignEval proves this is tractable and also shows the honest failure mode (dev 0.722 vs test 0.727 is fine; a large gap means overfitting).

**Non-determinism handling.** k runs per task, pass@k and pass^k, variance reporting, environment isolation so flakiness isn't mistaken for regression.

**CI gating.** Fixed-set runner, per-metric thresholds, paired bootstrap CI on delta versus baseline (gate on the CI, not the point estimate), regression report naming which cases flipped, and a **saturation warning** when a suite hits 100%.

**Production monitoring.** Async sampled judging, rate estimates with confidence intervals and lower-bound alerting, input-distribution drift detection (embedding shift, new cluster emergence), auto-promotion of newly-flagged production traces into the review queue.

**Guardrails.** Inline deterministic validators — regex, schema, blocklist, PII, JSON/SQL well-formedness — with trigger logging and false-positive tracking. Architecturally separate from evaluators: guardrails block, evaluators measure.

## Irreducibly human — a tool must not automate these

**Defining what "good" means.** Criteria drift is the load-bearing finding: Shankar et al. establish that users need criteria to grade outputs, but grading outputs is how users define criteria. Criteria cannot be fully specified before observing outputs, so no tool can pre-populate them.

**Initial open coding.** Husain is explicit: do not outsource this to an LLM. It is where you discover failures you didn't know existed, build intuition about your data, and feel user pain. He sets a floor of **30 traces annotated by you personally** before any agent suggestions enter the loop.

**Validating the failure taxonomy.** LLMs mis-group. The proposal can be machine-generated; the acceptance cannot.

**Ground-truth labels.** Every label in the judge's dev and test sets must be hand-validated. A judge is aligned *to* something; if that something is machine-generated the whole construct is circular.

**Writing critiques.** The critique is the transmission medium for tacit domain knowledge and the raw material for few-shot examples. Terse or auto-generated critiques defeat the purpose.

**Adjudicating disagreement.** When two annotators differ, the resolution is a rubric edit — identifying which clause was ambiguous and adding a definition, rule or example, then relabeling. The tool computes κ and surfaces the disagreeing rows; a human writes the rule.

**Root-cause analysis and choosing the fix.** Only human review catches edge-case patterns and decides whether the answer is a prompt line, a retrieval change, decomposition, or a product redesign.

**Cost-benefit on each evaluator.** Deciding that a failure mode does *not* deserve a judge — the discipline Husain says most teams skip and where the ROI actually lives.

**Choosing the dimensions.** The features × scenarios × personas table is a domain artifact. An LLM can expand it; it cannot originate the right axes.

**Retirement decisions.** Knowing when a gold set is stale, when an always-passing eval should be dropped, and when the product itself should change so outputs become verifiable.

## The grey zone — LLM-assisted, human sign-off required
First-pass axial coding over 30–50 open-coded traces; mapping new annotations onto an existing taxonomy; drafting prompt improvements from open-coding notes; clustering and surfacing candidate traces; proposing judge prompt revisions; suggesting taxonomy merges. Husain's framing is the right design principle: **use LLMs to scale what you learned, not to avoid looking at data.**

## Design implications for a Python tool
1. Be a **state store plus a measurement engine plus an interface generator** — not a judgment engine.
2. Make the human steps *fast*, not optional. The bottleneck is reviewer throughput, so invest in rendering and hotkeys, not in auto-labeling.
3. **Refuse unsafe operations by construction**: no auto-labeling into dev/test, no test-set access during optimization, no raw-agreement-only reporting, no judge promoted without a test-set alignment report.
4. Emit **artifacts, not dashboards**: trace store → open codes → taxonomy → labeled set → splits → judge → alignment report → CI config. Each step's output is the next step's input, versioned.
5. Gate on **confidence intervals**, not point estimates, everywhere a decision is made.
6. Surface **saturation** and **staleness** as first-class signals, so the tool tells you when to stop reviewing and when an eval has stopped earning its maintenance cost.
