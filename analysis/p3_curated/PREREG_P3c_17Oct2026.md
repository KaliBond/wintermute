# P3c frozen specification: curated ensemble panel (17 Oct 2026)

Written before any outcome metric was computed on a curated panel. Pre-freeze inspection covered only file structure and envelope quality (`run_inventory_prefreeze.csv`). It found that 11 envelopes are not genuine: the m5n block2 files for Australia, China, Russia, UK and USA (2000–2026), for Belgium, Morocco, Portugal and Sri Lanka (1900–2026), and for Japan 1850–2026 (both the m5n block2 and `cleaned_datasets/Japan_ENV_1850_2026_cleaned.csv`) all contain the same SD sequence.

## Candidate runs
Searched in KaliBond/wintermute `main` at 446928c: ENS/ENV pairs in `cleaned_datasets/`, `data/nations/` and `data/nations/5yr/` (the scope of `scripts/check_envelopes.py`); `juno/*_m5n_*_block1/2.csv`; `analysis/camnations5_*` plus `_envelope.csv`; `juno/USA_1995_2025_CAMNATIONS5_test1` plus `raw_envelope1`. Drive holds copies of the same families (the Drive search was titles only, no downloads); repo versions are used.

## Eligibility (all must hold)
1. Annual series (consecutive years; no quarterly or 5-year runs).
2. Complete: all 8 canonical nodes × 4 dimensions in every year, no duplicate cells.
3. Envelope covers exactly the same Society/Year/Node rows.
4. Envelope non-degenerate: ≤ 10% of rows with all four SDs = 0, and ≤ 5% missing SD values.
5. Envelope not a template: its SD sequence is not shared with another society (whole-file identity, or any 40-row window matching the m5n template).
6. Not flagged as test or not-an-ensemble: filename contains "test", or listed as "not an ensemble" or under Problems in `data/ENVELOPE_STATUS.md`.
7. At least 20 years within 1950–2022.

## Selection per society
Files with identical content count as one run. Among eligible runs: lowest mean envelope SD (mean of C/K/S/A SD); ties broken by latest commit date. Society names are normalised to JUNO names (United Kingdom→UK, United States→USA, Türkiye→Turkey).

## Panel
CURATED_CAMS.csv: Society, Year, Node, Coherence, Capacity, Stress, Abstraction.

## Outcomes and code
Coup events: the same rule and source as PREREG_P3_17Oct2026.md. Any curated society that is not among the 47 gets coverage from max(1950, independence) to 2025, with the start year recorded in the build script. V-Dem as in PREREG_P3b.

Reruns, with unchanged code paths where possible:
- P3: node vs aggregate, `p3_compare.py`, pooled with `p3_pooled.py`.
- P3b: C+B3 vs B3, C vs B3, C vs B1, the same feature definitions (logic from `p3b_benchmark.py`, re-pointed at the curated panel).
- P7: λ2 recomputed from the canonical bond formula, q_i = (0.6 C_i + 0.4 A_i)/10 and B_ij = √(q_i q_j)·2^(−(S_i+S_j)/10), then the Laplacian's second-smallest eigenvalue. It is verified against JUNO `Lambda2_Calc` on JUNO rows.

Horizon 3 is primary and horizon 5 a sensitivity. Society bootstrap: 2000 replicates, seed 20261017.

The success criteria are those of P3 and P3b. With fewer than 15 societies, power is declared low and the CI widths are reported.

## Curation check
On the society-years present in both panels, compare the curated and JUNO scores (correlations of the four aggregate means) and rerun P3 and P3b on JUNO restricted to those society-years.
