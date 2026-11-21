# SA-PEP-001a prospective timestamp bundle, 21 November 2026

Prepared 21 November 2026 (AEDT, UTC+11) for Kari McKern's CAMS framework, test SA-PEP-001a (Helm decoupling
predicts RIGIDITY). **Not deposited, not sealed.** The point of the bundle is to fix, before any of the outcome
years exist, the 2026 CAMS scores and the predictor values computed from them. It counts as prospective
evidence only once it carries a public timestamp (for example a Zenodo DOI or a signed commit to
GitHub KaliBond/wintermute). Kari decides where it goes.

## Contents

| File | What it is |
|---|---|
| prospective_scores_2024_2026.csv | Canonical 8 x 4 scores (C, K, S, A) for 2024, 2025 and 2026, as they stand, for the 32 societies with a complete 2024 to 2026 window. 768 rows. Column set: collation_25 = the 25 named in the collation of 21 November; R3_5_added_7 = DRC, Egypt, Eritrea, Ethiopia, Mauritania, Nigeria, Vietnam (found in the second fetch). Column src = source file. |
| prospective_predictor_D_Helm_2026.csv | D_Helm for window 2024 to 2026, tercile, and copy-forward flags. |
| DECISION_RULE.md | Predictor, outcome class (RIGIDITY, 2027 to 2029), decision rule, scope and stated weaknesses. Items marked KARI TO CONFIRM are open. |
| MANIFEST.txt | SHA-256 of every file here, plus each source file's origin path. |
| rebuild_prospective.py | Rebuilds both CSVs from sources/ (verified byte-identical on 21 November 2026). |
| census_2026_11_21.py, census_R3_5.py, decoupling_R3_4.py, pipeline_R3.py | Code used (normalisation, union rules, D_Helm, bond strength). Byte copies of the SA-PEP working files. |
| sources/ | The source score files themselves: JUNO_Unified_Dataset_v3_2026-10-19.csv, nation_blind_ensemble_primitives.csv with its KEY, six camnations5 files and the Vietnam block1 file. |

Outcomes: none. No outcome data was opened or coded. The outcome years 2027 to 2029 have not happened.

## Rebuild

    cd prospective_bundle_2026-11-21
    python3 rebuild_prospective.py     # needs Python 3, numpy, pandas
    cmp prospective_scores_2024_2026.csv prospective_scores_2024_2026_rebuilt.csv
    cmp prospective_predictor_D_Helm_2026.csv prospective_predictor_D_Helm_2026_rebuilt.csv

## Caveats

- 32 cases cannot show a CAMS-ontology advantage on their own (see DECISION_RULE.md, Scope).
- Kari asked for the 25; the 7 added societies are marked and can be dropped before deposit without
  changing the 25 rows.
- Copy-forward and low cross-scoring reliability of D_Helm are disclosed in DECISION_RULE.md section 4.
