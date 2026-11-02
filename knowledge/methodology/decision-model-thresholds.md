---
name: decision-model-thresholds
description: Tune a decision model's threshold on held-out folds and check its calibration before it judges, per Husain and Saha (2026-10), and how axial does it
shelf: methodology
updated: 2026-10-30
sources:
  - url: https://hamel.dev/blog/posts/decision-model-evals/
    read: 2026-10-21
  - url: https://hamel.dev/blog/posts/decision-model-evals/charts/chart-data.json
    read: 2026-10-22
  - url: https://hamel.dev/blog/posts/decision-model-evals/charts/threshold-selection-audit.json
    read: 2026-10-22
  - url: https://hamel.dev/blog/posts/decision-model-evals/charts/calibration-reliability-data.json
    read: 2026-10-22
related: [numbers-and-rules-of-thumb, points-of-disagreement, methodology-spine]
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
   agent (100 less two internal QA traces), 61 with a failure and 37 without.
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
weighting of precision and recall through F1 is a simplification they name.

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

Run on the post's published scores as dev verdicts (2026-10-22), `tune-threshold` finds
no threshold reaching TPR and TNR of 0.85 for any of the six models. OpenAI comes
closest, TPR 0.852 and TNR 0.811 on all 98 traces and TNR 0.757 cross-validated at the
default seed (0.757 to 0.784 over seeds 0 to 19), and
Perplexity's closest is TPR 0.770 and TNR 0.784. On a set that is 62% failures, 80%
precision asks much less of TNR, so all three candidates pass the post's bar and none
passes `axial`'s default one.
