---
name: decision-model-thresholds
description: Tune a decision model's threshold on held-out folds and check its calibration before it judges, per Husain and Saha (2026-10), and how axial does it
shelf: methodology
updated: 2026-11-24
sources:
  - url: https://hamel.dev/blog/posts/decision-model-evals/
    read: 2026-11-24
  - url: https://www.randalolson.com/2026/10/14/when-to-use-a-decision-model-instead-of-an-llm/
    read: 2026-11-24
  - url: https://hightouch.com/blog/llm-judges-identity-resolution
    read: 2026-11-24
  - url: https://strandsagents.com/blog/amazon-saves-millions-replacing-sonnet-with-strands-decider/
    read: 2026-11-24
  - url: https://hamel.dev/blog/posts/evals-faq/whats-the-difference-between-guardrails-evaluators.html
    read: 2026-11-24
  - url: https://hamel.dev/blog/posts/decision-model-evals/charts/chart-data.json
    read: 2026-10-22
  - url: https://hamel.dev/blog/posts/decision-model-evals/charts/threshold-selection-audit.json
    read: 2026-10-22
  - url: https://hamel.dev/blog/posts/decision-model-evals/charts/calibration-reliability-data.json
    read: 2026-10-22
related: [numbers-and-rules-of-thumb, points-of-disagreement, methodology-spine, guardrails]
---

# Decision model thresholds

A decision model is a classifier offered as a cheap judge. It reads a trace and returns
a score from 0 to 1 for "this trace has the failure", and the API turns the score into
a label at a threshold, 0.50 by default. Husain and Saha (*How To Use Decision Models
for Evals*, 2026-10-20) compare six of them on one task: Jev (TypeSafe), Decider
(Perplexity), d1 and d1-ultra (Liquid), Clef Flash (Cloudflare) and the Decisions API
(OpenAI, running GPT-6 Luna). They note that people use the default threshold without
checking it against their own data, which they say can cost a lot of performance. A
model that scores every failure 0.40 and every pass 0.10 separates them perfectly at
0.30 and catches nothing at 0.50.

## Their method

1. Finish the judge prompt before comparing models. Every model got the same criteria;
   Jev got them in its structured question format.
2. Label pass/fail by hand. The set is 98 production traces from an apartment-leasing
   agent (100 less two internal QA traces), 61 with a failure and 37 without. A footnote
   gives the 61 as 39 failures from the original human review plus 22 discoveries from
   Braintrust and Arize that the authors validated by hand. The post's text says the
   dataset and the broad failure-finding task are the ones from their prior post on
   automated evals.
3. Set minimums first, 80% precision and 80% recall, and drop any model that cannot
   reach both.
4. Tune each model's threshold before comparing models. For the three candidates, the
   models whose full-sample curves reach both minimums, each training split takes the
   highest-F1 threshold that meets both. When none does, it takes the unconstrained F1
   optimum and records the split as infeasible. The other three models are tuned to
   their unconstrained F1 optimum in every split. Each threshold is scored on the
   held-out fold. The published audit has 100 splits per model (five folds, repeated 20
   times), the same splits for every model.
5. Check calibration before reading a score as a probability. They group traces by
   predicted probability (about 20 per group), compare each group with its observed
   failure rate, try Platt scaling, and judge the result by the Brier score.
6. Weigh cost and latency last. Cost is per 100 runs, from token usage and October 2026
   prices. Latency came from separate runs, and the post says it does not compare
   throughput.

Their suggested tuning prompt says to report an infeasible requirement "rather than
relaxing either requirement". Their own fallback keeps infeasible splits in the
averages at the unconstrained optimum, so the two disagree.

## Results

Held-out figures are the means over the 100 splits in `chart-data.json`: the
constrained choice for the three candidates and the unconstrained one for the others.
"At 0.50" is the default threshold applied to all 98 traces. The Brier column is
computed here from the published scores; the post's text gives the Jev, d1-ultra and
OpenAI values and they match.

| Model | Met 80/80 | Held-out precision | Held-out recall | Held-out F1 | F1 at 0.50 | Recall at 0.50 | Cost per 100 runs | Brier |
|---|---|---|---|---|---|---|---|---|
| Jev | no | 0.711 | 0.867 | 0.781 | 0.788 | 0.885 | 1.90¢ | 0.187 |
| Perplexity Decider | yes | 0.754 | 0.825 | 0.787 | 0.729 | 0.639 | 0.82¢ | 0.227 |
| Liquid d1 | no | 0.697 | 0.923 | 0.794 | 0.577 | 0.459 | 0.72¢ | 0.282 |
| Liquid d1-ultra | yes | 0.759 | 0.817 | 0.787 | 0.803 | 0.803 | 1.52¢ | 0.173 |
| Clef Flash | no | 0.665 | 0.931 | 0.776 | 0.351 | 0.213 | 1.58¢ | 0.383 |
| OpenAI Decisions | yes | 0.869 | 0.862 | 0.865 | 0.840 | 0.902 | 3.72¢ | 0.159 |

What the numbers show:

- Leaving the default threshold in place costs two models much of their recall. Clef
  Flash catches 21% of failures at 0.50 and its tuned threshold is 0.055. Liquid d1
  catches 46% at 0.50 and tunes to about 0.10.
- "Met 80/80" was decided on thresholds fitted to all 98 traces. On held-out folds,
  Perplexity and d1-ultra fall below 80% precision (0.754 and 0.759), and their
  training splits were infeasible in 36 and 29 of 100. Only OpenAI met both minimums in
  all 100 training splits.
- Jev misses narrowly. On all 98 traces its best precision with recall still at 0.80 or
  more is 0.790, at a threshold of 0.68, and its training splits met both minimums in
  40 of 100. As a non-candidate it was scored at its F1 optimum, which holds out at
  0.781 against 0.788 at the default. A small manual prompt change, adding a checklist
  of failure types, raised its F1 at the default threshold from 0.78 to 0.80.
- Platt scaling moved Perplexity, Liquid d1 and Clef Flash closer to calibrated. Jev and
  d1-ultra barely needed it (Jev's Brier score went from 0.187 to 0.188), and it made
  OpenAI slightly worse (0.159 to 0.164). An exploratory refit directly on
  probabilities gave OpenAI 0.154, which the authors read as the result depending on
  the fitting method.
- Always predicting this set's failure rate (61 of 98) scores a Brier of 0.235. Liquid d1
  (0.282) and Clef Flash (0.383) score worse than that before calibration, so their raw
  probabilities should not be read as failure rates. They still rank traces: computed
  from the same published scores, their AUC is 0.715 and 0.709, where 0.5 is chance.
  The Brier score mixes calibration with how well the scores separate the classes.

## Caveats the authors give

One dataset and one broad task. They expect narrower tasks to narrow the gaps between
models. The calibration groups hold about 20 traces each, too few to pin down a failure
rate. Prices and model versions are from October 2026. The Jev prompt experiment was
small and manual, and they did not try GEPA (an automated prompt optimizer). Equal
weighting of precision and recall through F1 is a simplification they name. They point
to the F-beta score for weighting one over the other, beta below 1 for precision and
above 1 for recall, and say that optimizing a different metric could lead to different
thresholds and results.

## When a decision model fits

The post gives one habit for choosing a model, written in the first person: start
with a larger model that performs acceptably on the task, then try smaller ones to see
whether they match it for less. OpenAI Decisions gave their highest F1 at the highest cost, which they read
as a sign that harder eval tasks may justify paying more for an LLM. For a wider
comparison they cite Olson.

Olson (*When to use a decision model instead of an LLM*, published 2026-10-15) reads
the Artificial Analysis Decision Index as it stood on 2026-10-14. It ran 11 decision models and 15
LLM configurations on four tasks, with the LLMs answering through structured outputs.
Its blended score ranks most LLMs above every decision model. Split by task:

| Task | Median LLM against median decision model |
|---|---|
| Knowledge questions | 30 points higher |
| Hard reasoning | 19 points higher |
| Safety classification | within 2 points |
| Tool-call routing | within 2 points |

The best decision model beats every LLM tested at safety classification. On tool-call
routing only Claude Opus 5.5 beats the best decision models, at 50 or more times the
cost per task. On those two tasks every decision model answers faster than every LLM,
0.06 to 0.54 seconds against 1.1 to 17, and the median decision model costs 5 to 6
times less, though the cost ranges overlap.

One pair holds the model fixed. GPT-6 Luna as a decision model scores 48.5 on the
index in 0.17 seconds. As an LLM with reasoning off it scores 49.6 in 1.1 seconds, and
with reasoning on, 55 to 61 in 1.5 to 2.2 seconds. Olson concludes that the decision
format costs little accuracy and that what is given up is the accuracy reasoning buys.
The speed and Luna figures are from his chart descriptions.

His rule is to use a decision model for routing, triage, moderation, intent detection
and tool-or-no-tool calls, especially when the input holds everything needed to decide
and latency or volume matters, and an LLM when the answer depends on facts the input
does not contain or on multi-step reasoning. He reports that Artificial Analysis found
decision models overconfident on the knowledge and reasoning tasks and better
calibrated on tool calling and classification, and says their probabilities are most
useful for setting escalation thresholds on the tasks they are built for.

## Escalating the unsure cases

A decision model's score can route a case as well as label it: accept the confident
answers and send the rest to a stronger judge. Hightouch (Jin, Desai and Mackenzie,
2026-11-04) tested this for grading identity resolution, the question being whether a
merged profile holds more than one person. The set is 1,000 synthetic identities built
with pseudopeople, half of them over-merged, so accuracy is against the generator's
answer key.

| Setup | Accuracy | Cost per 1,000 | Time per identity |
|---|---|---|---|
| Claude Opus 5.5 alone | 99.0% | $3.86 | 2.2 s |
| GPT-6 Luna alone, 0.50 cutoff | 97.3% | $0.10 | 0.17 s |
| Jev alone, 0.50 cutoff | 95.6% | $0.05 | 0.15 s |
| Jev, then Opus on scores from 0.30 to 0.90 (22% of cases) | 98.8% | $0.88 | 0.63 s |
| Luna, then Opus on scores from 0.10 to 0.90 (6% of cases) | 98.5% | $0.33 | 0.30 s |

On a second set built only from hard cases, such as twins, a junior and a senior, and
typos in a birth date, Opus alone scores 95.1% at $5.39 per 1,000. Jev then Opus
scores 95.3% at $2.94 and Luna then Opus 91.6% at $1.15. Alone, at tuned cutoffs, Jev
scores 89.1% and Luna 88.8%.

The authors put the difference down to calibration. Most of Jev's mistakes come with
scores near 0.50, so the band sends them on. Most of Luna's come in the 0.90 to 1.00
bucket, so the band lets them through. On the harder set the Jev pair sends more cases
to Opus without any change to the harness, which costs more and keeps it level with
Opus alone.
This is the same check as step 5 of the method above, with a consequence attached: an
overconfident model cannot say which of its answers to escalate.

What the post does not establish: the data is synthetic, the cutoffs (0.60 for Jev,
0.70 for Luna) were chosen to maximise accuracy, and it mentions no held-out set for
them or for the escalation band. The costs and latencies are the authors' own setup.

Husain and Shankar's FAQ makes the same suggestion for an inline check: a slow LLM
judge might be feasible in a cascade that runs on the minority of borderline cases
([guardrails](../procedures/guardrails.md)).

## Running the old and the new side by side

Onstott and Jaiswal (Strands Agents blog, 2026-11-09) describe replacing Claude Sonnet
4.6 with Strands Decider for labelling tickets into 20 categories at Amazon. The post
is on the vendor's blog about its own model, and the task is production classification
and not judging. What carries over to changing a judge's model is the rollout.

A three-way setting, read on every message, chose between the old model only, both in
parallel with the old model's answer applied and the two compared, and the new model
only. The authors say that running both on real traffic and judging every
disagreement told them more than any offline eval would have, and that going back was
a 45-second configuration change. They report a cost per ticket of about $0.0005
against $0.0222 and 897 ms per call against 4,215 ms.

Two cautions from the same post. Sending the same content with its JSON keys in a
different order moved the output probabilities by up to 0.19, so they now sort the
keys, and they advise checking this before comparing runs. And they list when a
decision model probably cannot replace the LLM: the labels need explanation text, the
catalogue changes weekly, or the useful context does not fit in a few thousand tokens.

## In axial

`axial` keeps the method and changes two choices to fit its own rules.

- The floors are TPR and TNR, the same `min_tpr` and `min_tnr` that `promote` checks.
  Recall is TPR, but precision depends on how common failures are. This set is 62%
  failures. A judge with TPR 0.9 and TNR 0.9 has precision 0.94 at that rate and 0.32 at
  a 5% failure rate, so a precision floor measured here would not carry over to a rarer
  failure mode. See [points-of-disagreement](points-of-disagreement.md), item j.
- `axial tune-threshold` reads only dev verdicts and picks the threshold whose weaker rate
  clears its floor by the widest margin, where the post maximises F1. It cross-validates
  the choice on stratified dev folds, and when no threshold clears both floors it exits
  with an error and writes nothing, so it never saves a relaxed choice. Its
  cross-validated estimate does score a fold with no feasible threshold at the closest
  one, and says how many folds that was. The cross-validated rates take some of the
  in-sample optimism out on average, but on one dev split they cannot tell whether this
  threshold will clear the floors on test, and they move with the seed. With `--write`
  it saves the threshold as a new judge version, which still needs a real `axial judge`
  run on dev and then on test, and it writes nothing when the judge already uses the
  chosen threshold.
- `run_judge` records each decision model's probability in `verdicts.jsonl`, so tuning
  and the Brier score need no new model calls. `axial align` prints the Brier score
  beside the base-rate reference. Calibration curves and Platt scaling are not
  implemented.
- A judge is one model, named in its `judge.yaml`. Nothing routes a low-confidence
  verdict to a second model, so escalation is not implemented, and neither is a
  weighted tradeoff such as F-beta in `tune-threshold`.

Run on the post's published scores as dev verdicts (2026-10-22), `tune-threshold` finds
no threshold reaching TPR and TNR of 0.85 for any of the six models. OpenAI comes
closest, TPR 0.852 and TNR 0.811 on all 98 traces and TNR 0.757 cross-validated at the
default seed (0.757 to 0.784 over seeds 0 to 19), and
Perplexity's closest is TPR 0.770 and TNR 0.784. On a set that is 62% failures, 80%
precision asks much less of TNR, so all three candidates pass the post's bar and none
passes `axial`'s default one.
