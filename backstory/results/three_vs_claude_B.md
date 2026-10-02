# Model comparison

Reference: `block1_ensemble_mean.csv`

## vs `block1_ensemble_mean.csv`

- Cells matched: 80 of 80; only in reference 0; only in other 0

| Dimension | n both scored | MAE | exact | within 1 | Spearman | mean diff (other - ref) | NA agree | NA only ref | NA only other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coherence | 74 | 1.24 | 1% | 45% | 0.95 | +1.23 | 1 | 0 | 5 |
| Capacity | 74 | 1.16 | 3% | 46% | 0.92 | +1.15 | 1 | 0 | 5 |
| Stress | 74 | 0.84 | 3% | 68% | 0.88 | +0.57 | 1 | 0 | 5 |
| Abstraction | 74 | 1.80 | 0% | 12% | 0.91 | +1.80 | 1 | 0 | 5 |

- Case-level ranking agreement (Spearman on mean Node Value, 10 cases): 0.99

### Ten largest Node Value disagreements

| Society | Node | NV ref | NV other | C K S A ref | C K S A other |
| --- | --- | --- | --- | --- | --- |
| B05_FirstIntermediate_Egypt | Helm | -0.9 | 6.6 | 2.1 3.0 8.0 4.0 | 5.0 5.3 6.7 6.0 |
| B09_LBA_Collapse_Aegean | Helm | -2.2 | 4.9 | 2.3 2.5 8.3 2.7 | 4.7 4.7 7.0 5.0 |
| B05_FirstIntermediate_Egypt | Archive | 2.0 | 8.5 | 3.1 3.7 6.9 4.1 | 6.0 5.3 6.3 7.0 |
| B17_Greece_Protogeometric | Hands | 3.0 | 9.0 | 4.0 3.1 5.4 2.6 | 5.7 5.7 5.3 5.7 |
| B05_FirstIntermediate_Egypt | Flow | 0.6 | 6.3 | 2.9 3.0 6.9 3.1 | 5.0 5.0 6.7 6.0 |
| B05_FirstIntermediate_Egypt | Craft | 3.8 | 9.4 | 3.8 4.0 6.1 4.1 | 6.0 5.7 5.7 6.7 |
| B05_FirstIntermediate_Egypt | Shield | 1.4 | 7.0 | 3.0 4.3 7.7 3.6 | 5.0 6.0 7.0 6.0 |
| B09_LBA_Collapse_Aegean | Craft | 3.2 | 8.5 | 3.9 4.0 6.7 4.0 | 5.7 6.0 6.7 7.0 |
| B09_LBA_Collapse_Aegean | Stewards | -0.3 | 4.9 | 2.9 2.9 7.6 2.9 | 4.7 4.7 7.0 5.0 |
| B17_Greece_Protogeometric | Craft | 7.8 | 13.0 | 4.9 4.5 4.0 4.8 | 7.0 7.0 5.0 8.0 |

