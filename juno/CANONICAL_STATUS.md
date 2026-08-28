# JUNO canonical status

**Production classifier:** JUNO v1.2-Final.

The operator definitions, numerical policy (round aggregates to 6 decimal places before every threshold comparison), and regime-precedence rules are specified in [`JUNO_v1.2-Final_Formalism.md`](JUNO_v1.2-Final_Formalism.md). That file is the source of truth if any other document conflicts.

## Bond formula is algebraically rank-1

JUNO v1.2 bond strength is

```
q_i  = (0.6·C_i + 0.4·A_i) / 10
B_ij = √(q_i·q_j) · 2^(-(S_i+S_j)/10)   ∈ [0, 1],  B_ii = 0
```

which is identically `B_ij = w_i · w_j` for `i ≠ j` with `w_i = √(q_i) · 2^(-S_i/10)`. There is **no pair-specific coupling** in `B_ij`. The 8×8 matrix for a society-year is a rank-1 outer product (zero diagonal).

Network visuals that treat `B_ij` as measured topology (independent edge weights, community structure, “who is bonded to whom”) should be labelled **illustrative** until temporal inelasticity — or a successor that introduces genuine pair terms — is calibrated.

`Bond_Strength_Calc` on each node-row is the **per-node mean** of that node’s 7 off-diagonal edges. `SBD_Calc` is the mean of the 28 unique pairwise bonds (equal to the mean of the 8 per-node means). `Lambda2_Calc` is the second-smallest eigenvalue of the graph Laplacian of `W_ij = B_ij`.

## Bond column alignment fix (2026-08-28)

`Bond_Strength_Calc` previously stored the correct *multiset* of per-node mean bonds for each society-year, but those values were **misaligned to Node labels** in nearly every year (exact node alignment 5 / 5,425 complete society-years, and those 5 are years where all eight means are identical; sorted-value match 5,425 / 5,425).

This update recomputes per-node mean bonds from raw Coherence / Capacity / Stress / Abstraction and writes them into `Bond_Strength_Calc` aligned to the row’s `Node`. `SBD_Calc` and `Lambda2_Calc` already matched recomputation and were not changed.

- **Legacy column:** `Bond_Strength_Calc_legacy` is a copy of the pre-fix `Bond_Strength_Calc` values (rollback without restoring the file).
- **File backup:** [`backups/JUNO_Unified_Dataset_pre_bond_fix_20260828.csv`](backups/JUNO_Unified_Dataset_pre_bond_fix_20260828.csv)
- **Verification:** `python3 juno/verify_bond_alignment.py`

Incomplete society-years (missing C/K/S/A or not exactly eight nodes): none in this panel (48 societies, 43,400 node-rows, 5,425 complete society-years). If any appear later, leave `Bond_Strength_Calc` unchanged for those rows.

**Downstream analyses that slice `Bond_Strength_Calc` by node label must be re-run after this fix.** That includes Phase A tests such as Shield × SIPRI (`juno/PhaseA_Validation_Report_2026-08-16.md`). Year-level aggregates that only use `SBD_Calc` / `Lambda2_Calc` / the *multiset* of bond values are unaffected.

### Rollback

1. Revert the PR that introduced the aligned column, **or**
2. Restore `Bond_Strength_Calc` from `Bond_Strength_Calc_legacy` on every row, **or**
3. Replace `juno/JUNO_Unified_Dataset.csv` with `juno/backups/JUNO_Unified_Dataset_pre_bond_fix_20260828.csv` (that backup has no legacy column; it is the pre-fix file).
