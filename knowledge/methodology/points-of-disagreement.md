---
name: points-of-disagreement
description: Where the practitioner sources disagree on method, numbers and judge design
shelf: methodology
updated: 2026-10-30
sources:
  - url: https://hamel.dev/blog/posts/evals-faq/
    read: 2026-09-17
  - url: https://hamelhusain.substack.com/p/its-hard-to-eval-is-a-product-smell
    read: 2026-09-17
  - url: https://hamel.dev/blog/posts/decision-model-evals/
    read: 2026-10-21
  - url: https://arxiv.org/abs/2404.12272
    read: 2026-09-17
  - url: https://arxiv.org/abs/2401.03038
    read: 2026-09-17
  - url: https://www.sh-reya.com/blog/in-defense-ai-evals/
    read: 2026-09-17
  - url: https://eugeneyan.com/writing/evals/
    read: 2026-09-17
  - url: https://eugeneyan.com/writing/llm-evaluators/
    read: 2026-09-17
  - url: https://eugeneyan.com/writing/product-evals/
    read: 2026-09-17
  - url: https://anthropic.com/engineering/demystifying-evals-for-ai-agents
    read: 2026-09-17
  - url: https://platform.claude.com/docs/en/docs/test-and-evaluate/develop-tests
    read: 2026-09-17
  - url: https://developers.openai.com/api/docs/guides/evaluation-best-practices
    read: 2026-09-17
  - url: https://developers.openai.com/cookbook/examples/partners/eval_driven_system_design/receipt_inspection
    read: 2026-09-17
  - url: https://www.sh-reya.com/blog/ai-engineering-short/
    read: 2026-09-17
related: [methodology-spine, decision-model-thresholds, evals-skills]
---

# Points of disagreement

**a) Eval-driven development: yes or no.**
Husain says **generally no** — LLM failure surface is effectively infinite, you can't anticipate what breaks, so write evaluators for *discovered* errors, not imagined ones. His only exception is a hard constraint with an unambiguous success criterion (e.g. "never mention competitors"). OpenAI's best-practices guide says the opposite outright, telling teams to evaluate early and often as a discipline, and its *Eval Driven System Design* cookbook is explicitly eval-centric rather than data-first: build a V0 system, generate draft outputs, have experts correct small batches, iterate on eval signal. Anthropic's agent guide sits nearer OpenAI — start before production, because waiting means reverse-engineering success criteria from a live system. **The reconciliation:** OpenAI and Anthropic mean "instrument and measure early"; Husain means "don't author rubrics before you've read outputs". They genuinely differ on whether the rubric precedes the data.

**b) Binary vs Likert.**
Husain, Shankar, and Yan are emphatic on **binary pass/fail**: the 3-vs-4 boundary is subjective and inconsistent across annotators, annotators centre-bias, and detecting differences on a scale needs bigger samples. Yan adds that binary outputs give you recall/precision, which map to production behavior, whereas correlation metrics do not. Anthropic's own docs, however, present a **1–5 Likert grader as a first-class pattern**, and its agent guide uses rubric scoring with **partial credit** across multi-component tasks. Husain's concession is to decompose instead of scaling: "4 of 5 expected facts present" as a set of binary checks.

**c) Sample size philosophy.**
Anthropic's docs say **prioritize volume over quality** — more questions with slightly noisier automated grading beats fewer hand-graded ones — and list 1,000+ test cases for exact-match evals. Husain says the opposite: **100 well-chosen traces beat thousands labeled superficially**, and careful investigation of a few failures beats rushing many. Anthropic's *agent* guide actually agrees with Husain (20–50 well-designed tasks; a good task beats hundreds of ambiguous ones), so Anthropic is internally split between its docs and its engineering blog.

**d) Same model as judge.**
Anthropic docs: generally best practice to use a **different** model for evaluation than generation. Husain: usually fine — the judge is doing a different task, and what actually matters is measured alignment (TPR/TNR) on a held-out labeled set; if alignment is poor, swap models then. Yan documents the self-enhancement bias (+10 to +25% self-preference) that motivates Anthropic's rule.

**e) Off-the-shelf / generic metrics.**
Husain: generic evaluations waste time and manufacture false confidence — useful only as a *sampling* signal to surface traces for review. Anthropic's docs include ROUGE-L and cosine-similarity graders. Yan splits the difference by task: he rejects ROUGE/METEOR/BERTScore/G-Eval for summarization, but endorses chrF, BLEURT, COMET and COMETKiwi for translation (BLEU is "bottom of the leaderboard" at WMT22/23) and fine-tuned NLI models for factual consistency. So "generic metrics are useless" is really "generic metrics are useless for *product* quality in open-ended domains".

**f) Who annotates, and how many.**
Husain: one **benevolent dictator** domain expert for most small/medium companies; multiple annotators only for genuinely multi-domain orgs, and then measure Cohen's κ and run alignment sessions. OpenAI recommends multiple review rounds with **consensus voting** aggregation and lists purchased/crowdsourced eval data as legitimate; Anthropic lists crowdsourcing as a human grader method. Husain calls outsourcing annotation "usually a big mistake" because it breaks the observe→understand→fix loop, allowing it only for mechanical tasks after an internal rubric exists.

**g) Automated prompt/assertion generation.**
Husain is skeptical of automated prompt optimization — writing prompts is how you externalize requirements, and optimizers hill-climb a fixed metric so they structurally cannot discover new failure modes; he permits it only for the "final mile" once good evals exist. Shankar's own **SPADE** does exactly the thing he's wary of — synthesizing assertions automatically from prompt version history — and reports real gains. Her **EvalGen** is a middle position: generate candidate Python functions and grader prompts, then use human grades on a subset to *select* among them.

**h) Trajectory vs outcome.**
Anthropic: grade what the agent produced, not the path, to avoid punishing valid unanticipated approaches. Husain's error-analysis method is fundamentally trajectory-based — find the **first upstream failure**. Not strictly contradictory (diagnosis vs scoring) but they pull tooling in different directions: one wants full-trace rendering, the other wants final-state assertions.

**i) Is rigorous eval for everyone?**
Shankar (*In Defense of AI Evals, for Everyone*, 2025-09-05) concedes ground to the anti-eval camp: teams in well-represented domains (coding agents) can lean on upstream model providers, and experienced teams with deep domain intuition can substitute continuous dogfooding plus error analysis for formal metrics. Her line is that newcomers and teams without data backgrounds cannot. Husain's framing in *"It's Hard to Eval" Is a Product Smell* (2026-06-29) goes further sideways: if verification is hard, that's a **product design defect** — surface intermediate artifacts, anchor outputs to vetted references with diffs, make outputs reviewable in small units — fix the product rather than building a heroic eval.

**j) Floors on precision, or on TPR and TNR.**
Husain's judge-alignment writing reports TPR and TNR beside raw agreement, and so does the course's `validate-evaluator` skill ([evals-skills](../landscape/evals-skills.md)). Husain and Saha's decision-model post (2026-10-20) sets its minimums on precision and recall instead and picks thresholds by F1. The choice matters when the labelled set's failure rate differs from production's. Recall is TPR, but precision moves with prevalence: their set is 62% failures, where a judge with TPR 0.9 and TNR 0.9 has precision 0.94, and at a 5% failure rate the same judge has precision 0.32. Precision is the right floor when the labelled set has production's failure rate; TPR and TNR are the floors that carry over when it does not. `axial` gates on TPR and TNR ([decision-model-thresholds](decision-model-thresholds.md)).
