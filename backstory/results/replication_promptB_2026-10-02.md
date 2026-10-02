# Model comparison

Reference: `block1_ensemble_mean.csv`

## vs `block1_ensemble_mean.csv`

- Cells matched: 80 of 80; only in reference 0; only in other 0

| Dimension | n both scored | MAE | exact | within 1 | Spearman | mean diff (other - ref) | NA agree | NA only ref | NA only other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coherence | 76 | 0.19 | 36% | 100% | 0.98 | +0.04 | 1 | 2 | 1 |
| Capacity | 76 | 0.11 | 62% | 100% | 0.99 | -0.07 | 1 | 2 | 1 |
| Stress | 76 | 0.19 | 36% | 100% | 0.99 | -0.11 | 1 | 2 | 1 |
| Abstraction | 76 | 0.22 | 42% | 99% | 0.97 | +0.01 | 1 | 2 | 1 |

- Case-level ranking agreement (Spearman on mean Node Value, 10 cases): 1.00

### Ten largest Node Value disagreements

| Society | Node | NV ref | NV other | C K S A ref | C K S A other |
| --- | --- | --- | --- | --- | --- |
| B15_Egypt_MiddleKingdom | Hands | 8.5 | 10.2 | 5.4 7.0 6.0 4.2 | 6.0 7.0 5.6 5.6 |
| B13_Egypt_5thDynasty | Hands | 9.0 | 10.4 | 5.8 6.8 5.6 4.0 | 6.2 7.0 5.2 4.8 |
| B19_UrIII_Mesopotamia | Hands | 8.0 | 9.3 | 5.6 7.2 7.0 4.4 | 5.6 7.6 6.6 5.4 |
| B09_LBA_Collapse_Aegean | Lore | 3.9 | 2.6 | 4.0 4.0 6.0 3.8 | 3.3 4.0 6.3 3.3 |
| B20_LateOldBabylonian | Helm | 5.1 | 6.1 | 4.4 4.6 6.8 5.8 | 5.0 4.8 6.6 5.8 |
| B14_Egypt_6thDynasty | Shield | 6.1 | 5.1 | 4.4 5.2 5.8 4.6 | 4.0 4.8 5.8 4.2 |
| B19_UrIII_Mesopotamia | Archive | 17.8 | 18.8 | 8.0 9.0 3.4 8.4 | 8.4 9.0 3.0 8.8 |
| B18_Greece_LateGeometric | Hands | 6.8 | 7.7 | 5.0 6.0 6.0 3.6 | 5.0 6.0 5.4 4.2 |
| B19_UrIII_Mesopotamia | Flow | 11.6 | 12.5 | 6.0 6.6 4.4 6.8 | 6.2 6.8 4.0 7.0 |
| B05_FirstIntermediate_Egypt | Stewards | 3.5 | 2.6 | 3.8 4.6 7.0 4.2 | 3.6 4.4 7.4 4.0 |

## vs `promptB_raw_scores.csv`

- Cells matched: 80 of 80; only in reference 0; only in other 0

| Dimension | n both scored | MAE | exact | within 1 | Spearman | mean diff (other - ref) | NA agree | NA only ref | NA only other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coherence | 77 | 0.36 | 40% | 97% | 0.95 | +0.28 | 2 | 1 | 0 |
| Capacity | 77 | 0.46 | 35% | 92% | 0.94 | +0.39 | 2 | 1 | 0 |
| Stress | 77 | 0.47 | 21% | 96% | 0.91 | +0.25 | 2 | 1 | 0 |
| Abstraction | 77 | 0.55 | 16% | 87% | 0.91 | +0.29 | 2 | 1 | 0 |

- Case-level ranking agreement (Spearman on mean Node Value, 10 cases): 1.00

### Ten largest Node Value disagreements

| Society | Node | NV ref | NV other | C K S A ref | C K S A other |
| --- | --- | --- | --- | --- | --- |
| B05_FirstIntermediate_Egypt | Archive | 1.9 | 5.5 | 3.0 3.8 7.0 4.2 | 4.0 5.0 6.0 5.0 |
| B14_Egypt_6thDynasty | Helm | 4.8 | 8.0 | 4.0 5.0 6.8 5.2 | 5.0 6.0 6.0 6.0 |
| B17_Greece_Protogeometric | Hands | 3.0 | 6.0 | 4.0 3.2 5.4 2.4 | 4.0 5.0 5.0 4.0 |
| B05_FirstIntermediate_Egypt | Hands | 0.7 | 3.5 | 3.2 4.0 8.0 3.0 | 4.0 5.0 7.0 3.0 |
| B16_Mycenaean_LHIIIA | Lore | 8.4 | 11.0 | 5.0 5.0 4.0 4.8 | 6.0 6.0 4.0 6.0 |
| B15_Egypt_MiddleKingdom | Hands | 8.5 | 11.0 | 5.4 7.0 6.0 4.2 | 6.0 7.0 5.0 6.0 |
| B05_FirstIntermediate_Egypt | Flow | 0.5 | 3.0 | 3.0 3.0 7.0 3.0 | 4.0 4.0 7.0 4.0 |
| B09_LBA_Collapse_Aegean | Hands | -0.0 | 2.5 | 3.0 3.6 7.8 2.4 | 4.0 4.0 7.0 3.0 |
| B20_LateOldBabylonian | Hands | 3.1 | 5.5 | 3.6 4.8 7.2 3.8 | 5.0 5.0 7.0 5.0 |
| B13_Egypt_5thDynasty | Lore | 14.8 | 12.5 | 7.0 7.6 3.6 7.6 | 7.0 7.0 5.0 7.0 |

