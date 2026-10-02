# Testing against another model

The aim is a blind cross-model check of run 1.

## 1. Run the other model
1. Open a fresh conversation in the other model (no project files, no memory of this work).
2. Paste the whole of `prompts/scorer_prompt_deeptime_v1.2-OPT.txt`, then in a separate fresh conversation `prompts/scorer_prompt_deeptime_v1.2-OPT_windows2.txt`. Do not add context, hypotheses, or these scores.
3. Save its CSV output as `runs/<model-name>_pass1/raw_scores.csv`.
4. Repeat in up to five separate conversations for an ensemble (`_pass2`, `_pass3` ...).

Do not paste `cases.csv`, `HYPOTHESES.md` or `evidence_log.csv` into the other model; they reveal what each case is meant to test.

## 2. Claude reference
The five-scorer Claude ensemble (`runs/ensemble_claude-subagents_2026-10-02/`) is the reference. For a matching ensemble from the other model, run five passes and aggregate with:
```
python scripts/aggregate_ensemble.py runs/<model-name>_ensemble runs/<model-name>_pass1/raw_scores.csv ... runs/<model-name>_pass5/raw_scores.csv
```

## 3. Compare
```
python scripts/compare_models.py runs/ensemble_claude-subagents_2026-10-02/block1_ensemble_mean.csv runs/<model-name>_pass1/raw_scores.csv runs/claude-opus-5-5_pass1/raw_scores.csv --out results/comparison_<date>.md
python scripts/backstory_calc.py runs/<model-name>_pass1/raw_scores.csv runs/<model-name>_pass1
python scripts/hypothesis_check.py runs/<model-name>_pass1/node_metrics.csv runs/<model-name>_pass1/hypothesis_check.md
```
Requires Python 3 with pandas.

## 4. Read the result
| Signal | Reading |
| --- | --- |
| MAE under 1 and Spearman above 0.7 per dimension | Instrument is reasonably model-independent |
| Case ranking Spearman above 0.8 | Models agree on which societies were strong or failing |
| NA disagreement concentrated in B03, B07 | Gate is tracking genuine evidence gaps |
| Consistent mean diff on one dimension | Calibration offset between models, not disagreement |
| Hypothesis verdicts flip between models | Result depends on the scorer; do not publish as a finding |

Log every comparison in `RUNLOG.md`.

## Series tests
After combining both prompts' outputs (run-2 style), run `python scripts/hypothesis_check_series.py <node_metrics.csv> <out.md>` for the pre-registered H2–H4 series tests.
