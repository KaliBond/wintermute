# Model comparison

Reference: `block1_ensemble_mean.csv`

## vs `promptA_raw_scores.csv`

- Cells matched: 96 of 96; only in reference 0; only in other 0

| Dimension | n both scored | MAE | exact | within 1 | Spearman | mean diff (other - ref) | NA agree | NA only ref | NA only other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coherence | 93 | 0.90 | 6% | 69% | 0.84 | +0.82 | 2 | 1 | 0 |
| Capacity | 93 | 0.77 | 6% | 75% | 0.82 | +0.54 | 2 | 1 | 0 |
| Stress | 93 | 0.52 | 13% | 91% | 0.85 | +0.07 | 2 | 1 | 0 |
| Abstraction | 93 | 0.74 | 11% | 78% | 0.82 | -0.49 | 2 | 1 | 0 |

- Case-level ranking agreement (Spearman on mean Node Value, 12 cases): 0.77

### Ten largest Node Value disagreements

| Society | Node | NV ref | NV other | C K S A ref | C K S A other |
| --- | --- | --- | --- | --- | --- |
| B09_LBA_Collapse_Aegean | Archive | -6.2 | 0.0 | 1.1 1.0 9.1 1.7 | 3.0 3.0 7.0 2.0 |
| B10_Aboriginal_Australia | Stewards | 13.6 | 8.0 | 6.9 6.8 3.5 6.9 | 6.0 5.0 5.0 4.0 |
| B07_Minoan_Crete | Helm | 9.2 | 14.0 | 5.5 6.0 4.9 5.2 | 8.0 7.0 4.0 6.0 |
| B02_Shang_Anyang | Flow | 8.2 | 13.0 | 5.0 5.5 4.7 4.9 | 7.0 7.0 4.0 6.0 |
| B02_Shang_Anyang | Archive | 11.8 | 16.5 | 6.4 6.1 3.9 6.5 | 8.0 8.0 3.0 7.0 |
| B04_OldKingdom_Egypt | Craft | 18.2 | 14.0 | 8.3 9.0 3.2 8.3 | 8.0 8.0 5.0 6.0 |
| B07_Minoan_Crete | Archive | 9.0 | 13.0 | 5.3 5.1 4.3 5.7 | 7.0 7.0 4.0 6.0 |
| B02_Shang_Anyang | Shield | 10.0 | 14.0 | 6.1 7.0 5.9 5.5 | 8.0 8.0 5.0 6.0 |
| B02_Shang_Anyang | Helm | 11.6 | 15.5 | 6.7 7.0 5.1 6.1 | 8.0 8.0 4.0 7.0 |
| B01_Longshan_YellowRiver | Shield | 5.1 | 9.0 | 4.4 5.0 6.2 3.8 | 7.0 6.0 6.0 4.0 |

