# Deviation log: canonical_usa_ens/
2026-10-29 (Sydney), 20:xx onward. PREREG.md itself is NOT edited (its sha256 3603e474...2fe still matches FROZEN.txt, so the hi-fi test run stays reproducible). Entries here apply to this subfolder and are cross-referenced from PREREG.md section 12 by this file.

1. Dataset swap. USA_ENS.csv (md5 2f51ff07a507324e6cfcc272e00e1fa4, repo wintermute last commit touching it 98ea7b4) replaces the hi-fi series as the canonical US file per Kari, AFTER the hi-fi test results (H1 pass, H2 fail, H3 pass) were seen. The swap was not pre-specified and is not blind to the earlier result. Thresholds, year table (2020-25 ends 2025), nulls, seed and Holm rule are unchanged.
2. 2026. USA_ENS has a 2026 row. The frozen year table has no 2026 period, so 2026 is a gap year: in the annual series, in no period profile.
3. Structure difference (not a rule change). USA_ENS is NOT annual: 26 sampled years, 1900 to 2020 every 5 years plus 2026, 8 nodes each, 208 rows, no missing cells, Society label "United States", no # header lines (no model, method or n_passes metadata). Columns are identical to the hi-fi file. The hi-fi file is annual 1900-2025 with 5-pass ensemble metadata.
4. Consequences, applied literally and not repaired by interpolation:
   - Period profiles use only the sampled years inside each frozen window. WWI (1917-18) and Realignment (2016-19) have none.
   - Section 6 drops periods with fewer than 2 ensemble years from H1. That leaves 7 of 15 periods for H1 (strict, confirmatory).
   - H2 needs 5 consecutive annual tilt vectors before t. None exist, so X, the upcrossings and both nulls are not evaluable. H2 is counted as not supported.
   - H3: each named transition includes a 1-year period, so none is testable under the frozen rule (no bootstrap CI possible). H3 is counted as not supported.
   - Secondaries needing annual data (Theta drift D, acute z, coupling, all H2 variants) are not evaluable.
   - Holm: p for non-evaluable tests set to 1.0 (convention added here, cannot pass).
5. Post hoc, labelled, not confirmatory: H1 relaxed (periods with at least 1 sampled year); H3 signs on 1-year periods; a control that subsamples the hi-fi annual series to the USA_ENS years.
6. Open: USA_ENV.csv (same 26 years, columns C_sd, K_sd, S_sd, A_sd, V_range, V_min, V_max, md5 f2d6706a35620d7407004b078809a9c3) is an envelope file, not a score file. It was not used. A message mentioned "ENS and ENV"; only USA_ENS was run.
