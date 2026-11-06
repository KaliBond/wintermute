# Part 2 analysis v3: Kimi Work secondary sensitivity

Added 2026-11-06 (AEDT). The primary analysis remains **pool (a): 7 scorer-runs / 20 passes**, with **6 PASS / 9 FAIL and FC5 NOT TRIGGERED**. Analysis v2 and the sealed Section 7 are unchanged; this is an additive sensitivity analysis, not a redefinition of the primary comparison.

## Run classification

Kimi Work submitted five Mode A passes. Its claimed model, `k2d8-preview`, cannot be verified; differences in harness, evidence and NA behaviour mean it is treated as a run separate from Kimi web. Kimi Work is **secondary only**, in the same reporting class as Grok, and neither replaces any primary run. See `kimiwork_look_report.md` and `scorer_assessment.md` for the C+ assessment.

## Pool definitions

- **(a)** Primary: Claude agent, Codex, Kimi web, Gemini, Claude chat, Perplexity (unconfirmed), Manus; primary thresholds.
- **(b)** Pool (a) plus Grok; sealed secondary thresholds.
- **(c)** Pool (a) without Perplexity; primary thresholds.
- **(d)** Pool (a) without Manus; primary thresholds.
- **(e)** Pool (a) plus Kimi Work; **primary thresholds**.
- **(f)** Pool (a) minus Kimi web plus Kimi Work; **primary thresholds**. This is a hypothetical robustness check only, not a change of primary-pool membership.

Within each run, cells are averaged first, with the sealed `n_eff` rule unchanged (at least three passes for runs with three or more, at least one otherwise). Runs then count equally. Nothing is imputed. Scorers provide primitives only; `V = C + K − S`, Bond and all verdicts are recomputed centrally by `51_score.py`.

## Results of e/f

- **P6:** (a) FAIL at 13/16; (e) PASS at 14/16; (f) FAIL at 13/16.
- **P8:** (a) FAIL at 4.35/5.32 against the 5.0 limit; (e) PASS at 5.03/5.99; (f) PASS at 5.08/5.94. Pool (e) clears by only 0.03 on the first ratio.
- All other verdicts remain the same. **FC5 remains NOT TRIGGERED in (e) and (f).**

The primary verdicts are therefore unchanged. The secondary sensitivity makes P6 and P8 especially pool-dependent and should be read as such.

## Additional best-two pool

A post-hoc supplementary pool uses **Claude agent + Gemini**, the top overall grade then the first B-tier scorer with sound calibration. Run weighting, `n_eff`, primary thresholds and central derived formulas are unchanged. It gives **6 PASS / 9 FAIL, no indeterminates**, identical verdicts to pool (a), and FC5 NOT TRIGGERED. P6 fails at 13/16; P8 fails with rounded ratios 5.00/5.91: the unrounded Vanke ratio is 4.998819 (unrounded 4.998818728522337), below the sealed limit. Using Kimi web instead as the second seat would pass P6 (14/16; total 7 PASS / 8 FAIL) but not P8 (4.30/5.23). This does not alter pool (a), scorer grades, or count either Grok or Kimi Work as primary. Full results are in `best2_scorers_sensitivity.md` and `.csv`.

## Inputs and provenance

- `all_passes_long.csv`: complete primitive input to the v3 recomputation (300 KB).
- `kimiwork_passes/`: five supplied pass grids (not the shared evidence pack).
- `scoring_table.csv` / `.md`: all six pool verdicts, per-scorer verdicts and metrics.
- `kimiwork_vs_pool.csv`: Kimi Work per-pass and five-pass mean comparison against the primary pool.
- `kimiwork_look_report.md`: format, repeatability, carry-forward, bias and independence assessment.
- `51_score.py`: the centrally executed v2 scoring logic extended only with Kimi Work and pools e/f. P12 additionally requires the sealed Part 1 coupling baselines at `/workspace/evidence/ADVERSARY-02/analysis/section7/`.

The full five-pass run and evidence are ledger-filed outside the public site at `runs/part2/kimiwork/` (MANIFEST `95f9357d25ea1ca7c8610e3d33aaeddb9329b4426220ff92cd8a3b165e29fcdb`). The source analysis is `runs/part2/analysis_v3_kimiwork_2026-11-05/` (MANIFEST `4d3413ba51ac00e2c9f886d0af1d265c7e23e8c9fe73278bc6bc0a15696f9514`). The published slim inputs and code let the comparison be recomputed without publishing the full evidence pack.
