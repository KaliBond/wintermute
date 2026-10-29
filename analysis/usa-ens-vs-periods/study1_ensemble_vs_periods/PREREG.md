# USA ensemble vs 25 casebook periods: frozen pre-registration
DATASET STATUS (relabelled 2026-10-29 20:xx Sydney, before any run): USA_v12opt_hifi_block1.csv is a TEST (non-canonical) dataset. Kari has said it is no longer the canonical US file. Every use of "primary series" below means this test dataset. Results are test results, not canonical results. No threshold, rule or year table was changed.
Written 2026-10-29 (Sydney time, UTC+11), BEFORE run.py was run on the real data. Nothing in this folder has been computed from outcomes. Only file structure (columns, year coverage, row counts, checksums) was inspected.
Status: FROZEN on sign-off. Any change after freezing goes in the Deviation log (section 12), never by editing earlier sections.
Style follows /workspace/analysis/posture_acute/PREREG.md.

## 0. Purpose
Does the USA annual series show shifts in how the eight functions relate to each other, and do those shifts begin BEFORE the conventionally recognised transitions between US historical periods? A second question: do the period-level node profiles from the annual series agree with the independent blind-rater profiles for the same periods?

Premises (Kari's standing position, 29 Oct 2026): AI ratings are good-faith observations from a defined instrument, repeatable, inspectable and contestable. This study does not re-litigate rater reliability. Discovery comes first.

## 1. Disclosure of prior exposure (read before interpreting anything)
1. The rater-side results were already computed and read before this freeze (blind-rater note, PR #26): period mean Q differences for the three named transitions had the same sign across all three raters (1920s boom to Great Depression negative, New Deal to WWII positive, 1991-2001 to 2020-25 negative). The expected H3 signs below are therefore NOT blind predictions on the rater side. The new, untested side is the annual series.
2. No period mean, drift value, correlation or hit count has been computed from the annual series for this study.
3. The annual series (or earlier versions) may have been viewed in other CAMS tools and charts. Its general shape may be known to the author. This is stated, not corrected for.
4. The annual series was scored with a Claude model (header: claude-sonnet-4-6, five-pass ensemble). One rater (Claude) is from the same model family. Section 6 carries a sensitivity run without that rater.
5. The casebook has no sealed case-to-period map. The period identities and year boundaries in section 3 are the author's inference from cue text and labels (as in /workspace/analysis/blind_rater_1029/verify.py). They are frozen here as the working definition.

## 2. Data (frozen)
Primary series (TEST dataset, non-canonical): /workspace/wintermute/data/nations/USA_v12opt_hifi_block1.csv
- Repo: wintermute main 374bdf7 (2026-10-23). md5 cd8619ebddeef95925dd6ad5ec90a064.
- Header: 8 comment lines (# model: claude-sonnet-4-6, # method: in-session-ensemble-5n, # n_passes: 5, # year_range: 1900-2025), then columns Society, Year, Node, Coherence, Capacity, Stress, Abstraction, Node Value, Bond Strength.
- Structure seen: 1008 rows = 126 years (1900-2025) x 8 nodes, no missing cells, no duplicate year-node, Society = USA. Coherence 2.6 to 8.0, Capacity 2.8 to 8.2, Stress 2.0 to 9.2, Abstraction 3.0 to 7.4 (ranges only, no outcome).
- The brief's span "1900 to 2026" is not available in this file. 2026 is EXCLUDED. The analysis span is 1900 to 2025.
- Do NOT mix with, or substitute from, USA_ENS.csv (1900 to 2026, 208 rows, different layout), the v3 unified dataset, or other USA_v12opt_* passes. These are alternatives only and are not used in any run, including sensitivity runs.

Rater files (frozen copies in the PR #26 worktree): /workspace/wt-blindrater/analysis/blind_rater_accidental_test/data/
- BLIND_CASEBOOK_200.csv md5 6a3fdb6335d4d3fe475c63e614b9757a (25 cases x 8 nodes, columns CaseID, Node, System_evidence, Node_evidence, Scoring_basis; no year information).
- BLIND_RESPONSE_FROZEN_GPT_200.csv md5 3e30d87f7a993594f9ed0135296c0367 (no empty score cells).
- BLIND_RESPONSE_CLAUDE_200.csv md5 b263cb6b3d3318be080e9a938d85263d (25 empty Abstraction cells and 1 empty Capacity cell).
- BLIND_RESPONSE_GROK_200.csv md5 0a3f30b58e3e31490e13196fbf4ffa00 (no empty score cells).
- Columns: CaseID, Node, Coherence, Capacity, Stress, Abstraction (integer 1 to 10 or empty), Evidence_note, Confidence, Abstain_reason.
- run.py refuses to run if any checksum differs.

Node names (JUNO v1.2, in this order): Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow.

## 3. Mapping of years to periods (frozen)
Rule: one calendar year belongs to at most one period. Where labels overlap, the later period takes the boundary year, except where stated. Years inside no period are "gap years": they stay in the annual series (drift and coupling use them) but are in no period profile. Boundary years are set by the author from the casebook labels and cue text, before any annual result was seen.

| CaseID | Casebook label | Frozen years | In 1900-2025 series |
|---|---|---|---|
| CASE-18 | Progressive era | 1901-1916 | yes |
| CASE-01 | WWI | 1917-1918 | yes |
| CASE-16 | 1920s boom | 1921-1929 | yes |
| CASE-14 | Great Depression | 1930-1932 | yes |
| CASE-05 | New Deal | 1933-1940 | yes |
| CASE-09 | WWII | 1941-1945 | yes |
| CASE-04 | Postwar 1945-65 | 1946-1960 | yes (label truncated to remove overlap with CASE-21) |
| CASE-21 | 1960s | 1961-1972 | yes |
| CASE-17 | 1970s | 1973-1980 | yes |
| CASE-06 | Reagan 1980s | 1981-1989 | yes |
| CASE-23 | 1991-2001 | 1991-2000 | yes (2001 goes to CASE-10) |
| CASE-10 | Post-9/11 2001-08 | 2001-2008 | yes |
| CASE-08 | GFC aftermath | 2009-2015 | yes |
| CASE-03 | Realignment 2016-20 | 2016-2019 | yes (2020 goes to CASE-24) |
| CASE-24 | 2020-25 | 2020-2025 | yes |
| CASE-02, 07, 11, 12, 13, 15, 19, 20, 22, 25 | Civil War, Early Republic, Gilded Age, Antebellum 1845-61, Revolution/Confederation, Colonial 1700-63, Reconstruction, Mature colonial to imperial war, Jacksonian expansion, Early settlement | pre-1900 | NO: excluded |

- In scope: 15 of 25 periods. The other 10 end before the series starts (the Gilded Age overlaps 1900 by at most one year) and are EXCLUDED from every test. The study therefore cannot test the Civil War to Reconstruction disagreement or any pre-1900 transition.
- Gap years (frozen): 1900, 1919, 1920, 1990.
- Casebook identities are inferred, so a wrong mapping is a risk. It is carried as a stated limitation, not corrected after results.

Conventional transition years (frozen). T is the first year of the later period, for each adjacent pair of in-scope periods:
1917 (Progressive to WWI), 1921 (WWI to 1920s), 1930 (1920s to Depression), 1933 (Depression to New Deal), 1941 (New Deal to WWII), 1946 (WWII to Postwar), 1961 (Postwar to 1960s), 1973 (1960s to 1970s), 1981 (1970s to Reagan), 1991 (Reagan to 1991-2000), 2001 (1991-2000 to Post-9/11), 2009 (Post-9/11 to GFC aftermath), 2016 (GFC aftermath to Realignment), 2020 (Realignment to 2020-25).
That is 14 transitions. The period boundaries are conventional and approximate. This is acknowledged as a limit on any lead claim.

## 4. Metrics (frozen)
Node score per year: Q_i,t = Abstraction x Coherence + Capacity - Stress, computed from the Coherence, Capacity, Stress and Abstraction columns. The file's Node Value (C + K + A/2 - S) and Bond Strength are NOT used.
Rater Q: same formula on each rater's integer scores. A rater's Q is missing if any of its four scores is empty.
Rater consensus Q for a (period, node) = mean of the available rater Qs, requiring at least 2 of 3 raters present, else missing.
Period profile (annual series) = mean of Q_i,t over the period's frozen years, per node. Period mean Q = mean over the eight nodes. For comparisons with the raters, only nodes with a consensus value are used on the annual side too.
Scale note: raters score the whole period on integers, the annual series on decimals, so level differences are expected. Primary tests use rank or within-period correlation. Mean offset is reported descriptively only.

Relational measures.
- Annual tilt: theta_i,t = Q_i,t - mean over the eight nodes of Q_.,t. It sums to zero.
- Primary drift (series X): X_t = 1 - cos(theta_t, theta_bar_t), where theta_bar_t is the mean of the annual tilt vectors over t-5 to t-1. Defined for t = 1905 to 2025 (121 values). If either vector has zero norm, X_t is undefined.
- Why not the 20-year baseline tilt for the primary: a 20-year trailing mean cannot show a change until well after it starts, so it cannot test "before". It is run as a secondary (below).
- Regime tilt drift, posture convention (secondary): baseline Q_bar_i,t = mean of Q_i over t-20 to t-1 (excludes t, at least 10 observed years). Theta_i,t = Q_bar_i,t - mean over the eight nodes of Q_bar. D_t = 1 - cos(Theta_t, Theta_{t-10}). First evaluable D year is 1920. Descriptive classes: D < 0.15 within-regime, D >= 0.40 transition (inherited from the posture/acute pre-registration; those cut-offs were set for a different scale and may never be reached here. That outcome is reported, not tuned).
- Acute convention (secondary): per node, z_i,t = (Q_i,t - baseline mean) / baseline sd. Baseline = t-20 to t-1, excluding t, at least 10 observed years, sample sd (ddof 1) floored at 0.25, acute when |z| >= 1.96. First evaluable year 1910. The 0.25 floor is inherited unchanged. On the Q scale it is expected to be rarely binding. This is reported, not adjusted.
- Pairwise coupling (exploratory): for each of the 28 node pairs, Pearson correlation of annual Q over the rolling 10-year window t-9 to t. Coupling shift K_t = Frobenius norm of (corr matrix at t minus corr matrix at t-10), first evaluable t = 1919. Descriptive only. No hypothesis test is attached to it.

## 5. Hypotheses, thresholds, nulls (frozen; all one-sided unless stated)
Seed for every random procedure: 20261029. Resamples B = 10000 unless an exhaustive set is stated.

### H1: period profiles agree
Unit: the 15 in-scope periods x 8 nodes. Both criteria must be met.
- H1a (period level): Spearman rho between ensemble period mean Q and rater consensus period mean Q across the 15 periods, rho >= 0.60. P-value by permutation of period labels (one-sided, p = (1 + count(rho_perm >= rho_obs)) / (B + 1)).
- H1b (profile shape): for each period, Pearson r between the ensemble node profile and the rater consensus node profile across the nodes that have a consensus value (need at least 6 nodes and non-constant profiles on both sides, else the period is dropped from H1b and listed). Statistic = median r across periods, median r >= 0.30. P-value by permuting node labels of the rater profile within each period independently, B times (one-sided).
- H1 p-value = max(p_H1a, p_H1b) (both must hold).
- H1 SUPPORTED only if both thresholds are met AND the Holm-adjusted test (section 7) passes. Otherwise H1 is NOT SUPPORTED.
- Thresholds are the author's choices, not calibrated on the annual series. For context only (not used): earlier rater-to-rater correlations on node Q were 0.886 to 0.931.
- Descriptive extras, no threshold: ICC(3,1) and ICC(2,1) on the 120 period-node cells, mean offset per source.
- Sensitivity (reported, not confirmatory): H1 recomputed with the Claude rater removed from the consensus, and with each rater alone.

### H2: relational drift leads the conventional transitions
Frozen definitions:
- Series: X_t (section 4), t = 1905 to 2025.
- Threshold: thr = the 90th percentile (numpy linear interpolation) of X_t over all 121 values.
- Upcrossing in year t (t = 1906 to 2025): X_{t-1} < thr and X_t >= thr. Each upcrossing is a point event.
- Lead window for transition T: years T-5 to T-1 (5 years). A transition has a LEAD HIT if at least one upcrossing falls in its lead window.
- Lag window: years T to T+4. A LAG HIT is at least one upcrossing in it. (Comparison only.)
- Statistic: S_lead = number of the 14 transitions with a lead hit. S_lag likewise. Median lead time (years from the first upcrossing in the window to T) is reported but not tested.
- Null N1 (primary): rigid circular shift of the whole set of 14 transition years over the 120 index positions 1906 to 2025 (windows wrap around). All shifts s = 1 to 119 except |s| <= 2 on the circle, that is 115 shifts, exhaustive. p_N1 = (1 + count(S_lead_null >= S_lead_obs)) / (115 + 1). This keeps the spacing of the transitions and the autocorrelation of X.
- Null N2 (secondary): B random sets of 14 distinct years drawn from 1912 to 2021 with at least 4 years between any two, same statistic.
- H2 SUPPORTED only if ALL hold: (i) S_lead >= 7 of 14; (ii) S_lead > S_lag; (iii) p_N1 passes the Holm-adjusted test (section 7); (iv) p_N2 <= 0.05 (a confirmation check, not part of Holm).
- Otherwise H2 is NOT SUPPORTED. "Same as chance" is a failure of H2, not an inconclusive result.
- Sensitivity (reported, not confirmatory): thresholds P85 and P95; lead windows of 3 and 8 years; a causal threshold (90th percentile of X over the trailing 20 years excluding t, at least 10 values); node-level acute firings (section 4) counted in lead windows against N1; the 20-year Theta drift D_t and coupling shift K_t treated the same way with their own P90 thresholds.
- Interpretation limit: a lead hit shows the series moves before the author's boundary year. It does not show the boundary years are correct, and does not show cause.

### H3: named transitions have the same sign in both sources
Named transitions (frozen):
- N-a: 1920s boom (CASE-16) to Great Depression (CASE-14), expected sign negative.
- N-b: New Deal (CASE-05) to WWII (CASE-09), expected sign positive.
- N-c: 1991-2001 (CASE-23) to 2020-25 (CASE-24), expected sign negative. This pair is not adjacent. It is a comparison of period means, not a single transition year.
- Delta = period mean Q (later) minus period mean Q (earlier), for the ensemble and for the rater consensus (same nodes used on both sides).
- Per transition H3 PASSES if the ensemble Delta has the same sign as the rater consensus Delta AND each individual rater's Delta has that sign (if any rater disagrees with the consensus sign, the transition is reported as "rater-disputed" and not tested) AND the 95% block-bootstrap interval of the ensemble Delta excludes zero. Bootstrap: moving blocks of length min(3, n years) inside each period, B resamples, percentile interval. Two-sided p from the bootstrap distribution.
- H3 SUPPORTED only if all three transitions pass and the max of the three bootstrap p-values passes the Holm-adjusted test. If any transition fails or is rater-disputed, H3 is NOT SUPPORTED, with the per-transition table reported.
- Descriptive extra: per transition, count of the eight nodes with matching sign between sources (no threshold).

## 6. Exclusions
- The 10 pre-1900 periods (section 3). Gap years from period profiles.
- 2026 and every other file listed as an alternative in section 2.
- A period is dropped from H1 only if its ensemble years are fewer than 2 (not expected). WWI (2 years) and the Depression (3 years) are kept despite short length. Short periods are flagged in the results.
- No year or node is removed after seeing results. No outlier removal.

## 7. Multiple comparisons
- Confirmatory family: H1, H2, H3 (three tests). Holm correction at family alpha = 0.05 over the three p-values (p_H1, p_N1 for H2, p_H3). Smallest p must be <= 0.05/3, next <= 0.05/2, largest <= 0.05, stopping at the first failure.
- Within each hypothesis the sub-criteria are an intersection (all must pass), so no further correction there.
- Everything labelled sensitivity, secondary, exploratory or descriptive is reported with no inferential claim and no p-value language beyond what is listed. No new tests are added after seeing results; any added analysis is labelled post hoc.

## 8. What counts as failure
- H1 failure: either threshold missed, or the Holm test fails. Meaning: the annual series and the blind period profiles do not agree at the stated strength.
- H2 failure: any condition in section 5 not met. Meaning: no evidence that the series' relational drift precedes the conventional transitions beyond the circular-shift null. Reported plainly, not recast as partial support.
- H3 failure: any named transition with a different sign, a bootstrap interval spanning zero, or a disputed rater sign.
- Overall: the "relational change precedes conventional history" claim stands only if H2 is supported. H1 and H3 failing does not rescue or sink H2, but they are reported alongside it and limit how far H2 can be read as a CAMS result rather than a property of this one series.

## 9. Pipeline checks before any hypothesis test (stop if any fails)
1. md5 of the primary series and the five rater files match section 2.
2. 126 years x 8 nodes, no missing, no duplicates.
3. Q computes with no missing value for every annual node-year.
4. Frozen period table gives 15 periods, 14 transitions, gap years exactly {1900, 1919, 1920, 1990}.
5. The sha256 of this PREREG.md matches FROZEN.txt.

## 10. Outputs
results.md with every figure produced by run.py, plus: period_profiles.csv, drift_by_year.csv, upcrossings.csv, h1.csv, h2.csv, h3.csv. Every number in any later write-up must come from those files.

## 11. Open choices the author has decided here (flag for Kari's sign-off before the run)
1. Span ends 2025 (2026 not in the file).
2. Year boundaries and case identities in section 3 (inferred, no sealed map).
3. Thresholds: H1a rho 0.60, H1b median r 0.30, H2 P90 and 5-year lead and S_lead >= 7, series X uses a 5-year trailing tilt.
4. Postwar truncated to 1946-1960 to avoid overlap with the 1960s.

## 12. Deviation log
2026-10-29 (Sydney), before any run: relabelled the hi-fi series as a TEST (non-canonical) dataset per Kari. Thresholds, year table and 2025 end unchanged.
2026-10-29 (Sydney), before any run: secondary.py added to compute the section 5 sensitivity and exploratory items exactly as specified there; run.py unchanged.
