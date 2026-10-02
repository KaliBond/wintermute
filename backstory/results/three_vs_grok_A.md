# Model comparison

Reference: `block1_ensemble_mean.csv`

## vs `block1_ensemble_mean.csv`

- Cells matched: 96 of 96; only in reference 0; only in other 0

| Dimension | n both scored | MAE | exact | within 1 | Spearman | mean diff (other - ref) | NA agree | NA only ref | NA only other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coherence | 77 | 0.73 | 16% | 83% | 0.91 | +0.71 | 4 | 0 | 15 |
| Capacity | 77 | 0.59 | 13% | 91% | 0.94 | +0.56 | 4 | 0 | 15 |
| Stress | 77 | 0.72 | 14% | 83% | 0.86 | +0.56 | 4 | 0 | 15 |
| Abstraction | 77 | 1.40 | 0% | 39% | 0.87 | +1.40 | 4 | 0 | 15 |

- Case-level ranking agreement (Spearman on mean Node Value, 12 cases): 0.85

### Ten largest Node Value disagreements

| Society | Node | NV ref | NV other | C K S A ref | C K S A other |
| --- | --- | --- | --- | --- | --- |
| B05_FirstIntermediate_Egypt | Craft | 3.7 | 9.4 | 4.0 4.0 6.3 4.0 | 6.0 5.7 5.7 6.7 |
| B09_LBA_Collapse_Aegean | Helm | -0.7 | 4.9 | 3.0 3.0 8.3 3.3 | 4.7 4.7 7.0 5.0 |
| B05_FirstIntermediate_Egypt | Helm | 1.0 | 6.6 | 3.0 4.0 8.0 4.0 | 5.0 5.3 6.7 6.0 |
| B09_LBA_Collapse_Aegean | Craft | 3.0 | 8.5 | 4.0 4.0 7.0 4.0 | 5.7 6.0 6.7 7.0 |
| B05_FirstIntermediate_Egypt | Shield | 1.7 | 7.0 | 3.0 4.0 7.3 4.0 | 5.0 6.0 7.0 6.0 |
| B05_FirstIntermediate_Egypt | Lore | 5.5 | 10.2 | 4.0 5.0 6.0 5.0 | 6.0 6.0 5.3 7.0 |
| B09_LBA_Collapse_Aegean | Lore | 3.6 | 7.6 | 3.7 4.3 6.3 3.7 | 5.3 5.3 6.0 6.0 |
| B05_FirstIntermediate_Egypt | Flow | 2.6 | 6.3 | 3.3 4.0 6.7 4.0 | 5.0 5.0 6.7 6.0 |
| B05_FirstIntermediate_Egypt | Archive | 5.0 | 8.5 | 4.0 4.7 6.0 4.7 | 6.0 5.3 6.3 7.0 |
| B09_LBA_Collapse_Aegean | Flow | 2.0 | 5.4 | 3.7 3.7 7.3 3.7 | 4.7 5.0 7.3 6.0 |

