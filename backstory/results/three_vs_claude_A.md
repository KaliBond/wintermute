# Model comparison

Reference: `block1_ensemble_mean.csv`

## vs `block1_ensemble_mean.csv`

- Cells matched: 96 of 96; only in reference 0; only in other 0

| Dimension | n both scored | MAE | exact | within 1 | Spearman | mean diff (other - ref) | NA agree | NA only ref | NA only other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coherence | 77 | 0.93 | 5% | 68% | 0.93 | +0.90 | 3 | 0 | 16 |
| Capacity | 77 | 0.95 | 4% | 60% | 0.93 | +0.94 | 3 | 0 | 16 |
| Stress | 77 | 0.89 | 1% | 66% | 0.88 | +0.64 | 3 | 0 | 16 |
| Abstraction | 77 | 1.64 | 0% | 21% | 0.92 | +1.64 | 3 | 0 | 16 |

- Case-level ranking agreement (Spearman on mean Node Value, 12 cases): 0.81

### Ten largest Node Value disagreements

| Society | Node | NV ref | NV other | C K S A ref | C K S A other |
| --- | --- | --- | --- | --- | --- |
| B09_LBA_Collapse_Aegean | Helm | -2.9 | 4.9 | 2.1 2.3 8.7 2.8 | 4.7 4.7 7.0 5.0 |
| B05_FirstIntermediate_Egypt | Helm | -0.1 | 6.6 | 2.7 3.1 8.0 4.1 | 5.0 5.3 6.7 6.0 |
| B09_LBA_Collapse_Aegean | Craft | 2.9 | 8.5 | 3.9 3.9 6.9 4.1 | 5.7 6.0 6.7 7.0 |
| B09_LBA_Collapse_Aegean | Stewards | -0.6 | 4.9 | 2.9 2.9 7.8 2.9 | 4.7 4.7 7.0 5.0 |
| B05_FirstIntermediate_Egypt | Craft | 4.0 | 9.4 | 4.0 4.0 6.2 4.3 | 6.0 5.7 5.7 6.7 |
| B05_FirstIntermediate_Egypt | Shield | 1.7 | 7.0 | 3.1 4.4 7.7 3.8 | 5.0 6.0 7.0 6.0 |
| B05_FirstIntermediate_Egypt | Flow | 1.4 | 6.3 | 3.2 3.4 6.9 3.5 | 5.0 5.0 6.7 6.0 |
| B05_FirstIntermediate_Egypt | Hands | 0.7 | 5.6 | 3.2 4.0 7.9 2.9 | 5.0 5.3 7.3 5.3 |
| B05_FirstIntermediate_Egypt | Archive | 3.8 | 8.5 | 3.8 3.9 6.2 4.7 | 6.0 5.3 6.3 7.0 |
| B09_LBA_Collapse_Aegean | Shield | 0.4 | 4.9 | 3.1 3.6 8.0 3.4 | 4.7 5.0 7.3 5.0 |

