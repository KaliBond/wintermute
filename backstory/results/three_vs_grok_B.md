# Model comparison

Reference: `block1_ensemble_mean.csv`

## vs `block1_ensemble_mean.csv`

- Cells matched: 80 of 80; only in reference 0; only in other 0

| Dimension | n both scored | MAE | exact | within 1 | Spearman | mean diff (other - ref) | NA agree | NA only ref | NA only other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coherence | 74 | 0.87 | 14% | 72% | 0.93 | +0.87 | 3 | 0 | 3 |
| Capacity | 74 | 0.81 | 11% | 77% | 0.94 | +0.80 | 3 | 0 | 3 |
| Stress | 74 | 0.51 | 23% | 95% | 0.83 | +0.14 | 3 | 0 | 3 |
| Abstraction | 74 | 1.45 | 5% | 35% | 0.90 | +1.45 | 3 | 0 | 3 |

- Case-level ranking agreement (Spearman on mean Node Value, 10 cases): 0.99

### Ten largest Node Value disagreements

| Society | Node | NV ref | NV other | C K S A ref | C K S A other |
| --- | --- | --- | --- | --- | --- |
| B05_FirstIntermediate_Egypt | Helm | 1.2 | 6.6 | 3.0 4.0 8.0 4.3 | 5.0 5.3 6.7 6.0 |
| B05_FirstIntermediate_Egypt | Stewards | 2.4 | 7.7 | 3.3 4.3 7.3 4.3 | 5.7 6.0 7.0 6.0 |
| B05_FirstIntermediate_Egypt | Shield | 1.7 | 7.0 | 3.3 4.3 7.7 3.7 | 5.0 6.0 7.0 6.0 |
| B05_FirstIntermediate_Egypt | Archive | 3.2 | 8.5 | 3.7 4.3 7.0 4.3 | 6.0 5.3 6.3 7.0 |
| B20_LateOldBabylonian | Stewards | 4.2 | 9.3 | 4.3 5.0 8.0 5.7 | 6.0 6.3 7.0 8.0 |
| B20_LateOldBabylonian | Flow | 4.6 | 9.6 | 4.3 4.7 7.0 5.3 | 6.0 6.3 6.7 8.0 |
| B17_Greece_Protogeometric | Flow | 4.6 | 9.5 | 4.3 4.0 5.7 4.0 | 5.7 5.7 5.3 6.7 |
| B05_FirstIntermediate_Egypt | Craft | 4.8 | 9.4 | 4.7 4.3 6.3 4.3 | 6.0 5.7 5.7 6.7 |
| B17_Greece_Protogeometric | Hands | 4.5 | 9.0 | 4.7 4.0 5.7 3.0 | 5.7 5.7 5.3 5.7 |
| B17_Greece_Protogeometric | Stewards | 4.5 | 9.0 | 4.3 4.0 5.3 3.0 | 5.7 5.7 5.0 5.3 |

