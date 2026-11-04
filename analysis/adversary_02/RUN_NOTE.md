# ADVERSARY-02: Part 1 reconstruction (2010–2020): run note

Run date: 4 Nov 2026. Scope: China plus Evergrande, Vanke, Longfor, Huawei, SMIC, BYD and CATL. Years 2010–2020; CATL 2011–2020. Nothing for 2021–2024 was scored.

## How the run was done
- Five scorer passes. Each pass ran as a separate, fresh Claude subagent context. Each received only `scorer_prompts.txt` and the run rules. No pass could see another pass's output, the repository, or any archived scores.
- No web access was used. Search results returned today would leak post-2021 information. All evidence is therefore the scorer model's own knowledge. Each pass's `sources_passN.md` lists its sources with publication dates, all tagged `[memory, not retrieved]`.
- Aggregation is deterministic. `aggregate.py` is included and re-running it reproduces the CSVs exactly.

## Formulas
- **Node Value:** V = C + K − S, as rule 6 instructs.
  - The scorer-prompt file uses V = C + K − S + 0.5·A. Those values are in `ensemble_mean_V_protocol.csv` for comparison.
  - The per-scorer V_min, V_max and V_range in `envelope.csv` use C + K − S.
- **Bond Strength:** the formula from the scorer-prompt aggregation section, applied to the unrounded ensemble means within each entity-year:
  - B_ij = [0.6·C_i·C_j + 0.4·A_i·A_j] · exp(−(S_i+S_j)/20)
  - Bond Strength_i is the mean of B_ij over the partner nodes j ≠ i that have valid means. `bond_n` is the number of those partners.
- **NA handling:**
  - Means and SDs (ddof=1) use the non-NA scorers only (`n_eff`).
  - Node Value and bond participation require n_eff ≥ 3. Below that, the means are reported as NA.
  - `NA_rate` = 1 − n_eff/5. It appears in `envelope.csv` alongside the requested columns.

## Rules not fully followed, or followed with caveats
1. **Rule 2 (evidence cutoff): not enforceable.**
   - The scorer model was trained on data well past 2021. It knows the Evergrande default (late 2021), SMIC's 7 nm progress, and Huawei's later recovery.
   - Each pass was told to discipline itself to the cutoff. Every pass reported that it could not guarantee this.
   - This is the key threat to the validity of ADVERSARY-02 as a predictive test. All five scorers share one model, so they share the same contamination. A low V_range therefore does **not** show the scores are free of hindsight.
   - Cells most at risk:
     - Evergrande Board, Risk, Finance and Brand, 2016–2020. The 2020 ensemble V for Brand is 1.8 and for Risk is −2.0, which looks harsher than what was observable by mid-2021.
     - The Evergrande-versus-Vanke/Longfor gap in general.
     - BYD and CATL in 2020.
2. **Rule 7 (dated sources): followed, but unverified.**
   - Every source has a publication date, but the dates come from memory and were not retrieved. Treat them as approximate. Spot-check them before relying on them.
   - Every cited source falls on or before 30 Jun 2021. One pass flagged a borderline item: reports around June 2021 of overdue commercial bills at Evergrande.
3. **Rule 5 (independence):**
   - The passes are structurally independent: separate contexts, with no shared scratch work.
   - They are not independent in the deeper sense, because they share one model and one training corpus. Expect the envelope to understate the uncertainty that different scorers or human raters would produce.
   - Within a pass, all eight entities were scored in one context. Independence holds between passes, not between entities.
4. **NA cells:**
   - Company nodes use the evidence gate in the scorer prompts, so some node-years are NA. Rule 5 asked for primitives for every node; the gate takes precedence where it applies.
   - CATL 2011–2014 is mostly NA because its pre-IPO detail was published only in the 2017–2018 prospectus, after those years' cutoffs.
   - 93 of 696 ensemble cells have n_eff < 3. Most are CATL early years and RD or Workforce for the property developers.
5. **Header naming:** pass CSVs use `Entity,Year,Node,C,K,S,A` as requested, not the scorer prompts' `Society`/long column names.
6. **Model:** this run was produced by Claude, not Codex.
