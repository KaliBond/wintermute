# ADVERSARY-02 Part 1, RUN 2 (2010–2020): run note

Second, independent five-pass run. Same scorer prompts, same rules, same aggregation script as run 1. Separate output folder. Scorers were Sonnet 5.5; run 1 scorers were Opus 5.5. Scorers were told not to read run 1 or each other's output.

Formulas: Node Value = C + K − S (`ensemble_mean_V_protocol.csv` has C + K − S + 0.5·A). Bond Strength B_ij = [0.6·C_i·C_j + 0.4·A_i·A_j]·exp(−(S_i+S_j)/20), mean over valid partner nodes, from unrounded ensemble means. Cells with n_eff < 3 are reported NA. SD uses ddof = 1.

Rules not fully followed (same as run 1):
- The 30 June 2021 evidence cutoff cannot be enforced; every pass reports hindsight contamination risk.
- All sources are cited from memory ("[memory, not retrieved]"), with approximate dates.
- Five passes share one model, so independence is structural, not statistical.
- Several scorers wrote files with Bash instead of the Write tool.
- Pass 5 reports setting each node's trajectory in one pass and a coarse re-check for repeated scores.
- NA gating differs by pass, so n_eff varies.
