# Model comparison

Reference: `block1_ensemble_mean.csv`

## vs `promptB_raw_scores.csv`

- Cells matched: 80 of 80; only in reference 0; only in other 0

| Dimension | n both scored | MAE | exact | within 1 | Spearman | mean diff (other - ref) | NA agree | NA only ref | NA only other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coherence | 79 | 1.08 | 1% | 57% | 0.89 | +0.98 | 0 | 1 | 0 |
| Capacity | 79 | 0.73 | 13% | 82% | 0.92 | +0.67 | 0 | 1 | 0 |
| Stress | 79 | 0.69 | 6% | 80% | 0.84 | +0.38 | 0 | 1 | 0 |
| Abstraction | 79 | 0.65 | 13% | 84% | 0.92 | -0.49 | 0 | 1 | 0 |

- Case-level ranking agreement (Spearman on mean Node Value, 10 cases): 0.93

### Ten largest Node Value disagreements

| Society | Node | NV ref | NV other | C K S A ref | C K S A other |
| --- | --- | --- | --- | --- | --- |
| B20_LateOldBabylonian | Shield | 3.4 | 9.5 | 4.1 4.0 6.9 4.4 | 6 7 6 5 |
| B09_LBA_Collapse_Aegean | Archive | -6.0 | 0.0 | 1.4 1.0 9.0 1.2 | 3 3 7 2 |
| B20_LateOldBabylonian | Helm | 5.6 | 11.0 | 4.7 4.7 6.7 5.8 | 7 7 6 6 |
| B20_LateOldBabylonian | Craft | 7.4 | 12.0 | 5.0 5.0 5.3 5.5 | 7 7 5 6 |
| B05_FirstIntermediate_Egypt | Flow | 0.6 | 5.0 | 2.9 3.0 6.9 3.1 | 5 5 7 4 |
| B05_FirstIntermediate_Egypt | Archive | 2.0 | 6.0 | 3.1 3.7 6.9 4.1 | 5 5 6 4 |
| B20_LateOldBabylonian | Flow | 5.6 | 9.5 | 4.3 4.7 6.2 5.7 | 7 6 6 5 |
| B05_FirstIntermediate_Egypt | Helm | -0.9 | 3.0 | 2.1 3.0 8.0 4.0 | 4 4 7 4 |
| B17_Greece_Protogeometric | Stewards | 3.8 | 0.0 | 4.0 3.2 4.9 2.9 | 3 3 7 2 |
| B13_Egypt_5thDynasty | Craft | 15.5 | 12.0 | 7.3 7.8 3.3 7.4 | 7 7 5 6 |

