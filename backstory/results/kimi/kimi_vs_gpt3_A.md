# Model comparison

Reference: `block1_ensemble_mean.csv`

## vs `promptA_raw_scores.csv`

- Cells matched: 96 of 96; only in reference 0; only in other 0

| Dimension | n both scored | MAE | exact | within 1 | Spearman | mean diff (other - ref) | NA agree | NA only ref | NA only other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coherence | 77 | 0.44 | 48% | 95% | 0.87 | -0.02 | 2 | 17 | 0 |
| Capacity | 77 | 0.55 | 23% | 91% | 0.90 | -0.40 | 2 | 17 | 0 |
| Stress | 77 | 0.77 | 18% | 75% | 0.77 | -0.60 | 2 | 17 | 0 |
| Abstraction | 77 | 2.09 | 3% | 18% | 0.81 | -2.09 | 2 | 17 | 0 |

- Case-level ranking agreement (Spearman on mean Node Value, 12 cases): 0.80

### Ten largest Node Value disagreements

| Society | Node | NV ref | NV other | C K S A ref | C K S A other |
| --- | --- | --- | --- | --- | --- |
| B09_LBA_Collapse_Aegean | Helm | 4.9 | -0.5 | 4.7 4.7 7.0 5.0 | 3.0 3.0 8.0 3.0 |
| B10_Aboriginal_Australia | Stewards | 12.4 | 8.0 | 6.7 7.0 5.0 7.3 | 6.0 5.0 5.0 4.0 |
| B05_FirstIntermediate_Egypt | Hands | 5.6 | 1.5 | 5.0 5.3 7.3 5.3 | 4.0 4.0 8.0 3.0 |
| B12_Xiongnu_Steppe | Hands | 10.5 | 6.5 | 6.0 7.0 6.0 7.0 | 5.0 6.0 6.0 3.0 |
| B09_LBA_Collapse_Aegean | Stewards | 4.9 | 1.0 | 4.7 4.7 7.0 5.0 | 3.0 4.0 7.0 2.0 |
| B10_Aboriginal_Australia | Craft | 12.8 | 9.0 | 7.0 7.0 5.0 7.7 | 6.0 5.0 4.0 4.0 |
| B10_Aboriginal_Australia | Hands | 11.1 | 7.5 | 6.3 7.0 5.7 7.0 | 6.0 5.0 5.0 3.0 |
| B05_FirstIntermediate_Egypt | Helm | 6.6 | 3.0 | 5.0 5.3 6.7 6.0 | 4.0 4.0 7.0 4.0 |
| B05_FirstIntermediate_Egypt | Shield | 7.0 | 3.5 | 5.0 6.0 7.0 6.0 | 4.0 5.0 7.0 3.0 |
| B09_LBA_Collapse_Aegean | Hands | 4.5 | 1.0 | 4.7 5.0 7.7 5.0 | 4.0 4.0 8.0 2.0 |

