# Model comparison

Reference: `block1_ensemble_mean.csv`

## vs `block1_ensemble_mean.csv`

- Cells matched: 80 of 80; only in reference 0; only in other 0

| Dimension | n both scored | MAE | exact | within 1 | Spearman | mean diff (other - ref) | NA agree | NA only ref | NA only other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coherence | 77 | 0.22 | 40% | 100% | 0.98 | -0.19 | 1 | 0 | 2 |
| Capacity | 77 | 0.14 | 45% | 100% | 0.99 | -0.04 | 1 | 0 | 2 |
| Stress | 77 | 0.17 | 35% | 100% | 0.99 | +0.08 | 1 | 0 | 2 |
| Abstraction | 77 | 0.31 | 29% | 96% | 0.97 | -0.27 | 1 | 0 | 2 |

- Case-level ranking agreement (Spearman on mean Node Value, 10 cases): 1.00

### Ten largest Node Value disagreements

| Society | Node | NV ref | NV other | C K S A ref | C K S A other |
| --- | --- | --- | --- | --- | --- |
| B19_UrIII_Mesopotamia | Hands | 9.8 | 8.0 | 5.8 7.8 6.6 5.6 | 5.6 7.2 7.0 4.4 |
| B13_Egypt_5thDynasty | Hands | 10.7 | 9.0 | 6.2 7.0 5.0 5.0 | 5.8 6.8 5.6 4.0 |
| B13_Egypt_5thDynasty | Stewards | 12.4 | 10.8 | 6.8 7.0 4.6 6.4 | 5.8 7.0 4.8 5.6 |
| B19_UrIII_Mesopotamia | Flow | 13.1 | 11.6 | 6.6 7.0 4.0 7.0 | 6.0 6.6 4.4 6.8 |
| B15_Egypt_MiddleKingdom | Hands | 9.9 | 8.5 | 5.8 7.0 5.8 5.8 | 5.4 7.0 6.0 4.2 |
| B15_Egypt_MiddleKingdom | Lore | 15.9 | 14.5 | 7.4 7.6 3.2 8.2 | 7.0 7.0 3.4 7.8 |
| B13_Egypt_5thDynasty | Lore | 16.2 | 14.8 | 7.8 7.8 3.4 8.0 | 7.0 7.6 3.6 7.6 |
| B20_LateOldBabylonian | Flow | 6.3 | 5.0 | 4.4 5.0 6.2 6.2 | 4.2 4.6 6.4 5.2 |
| B13_Egypt_5thDynasty | Archive | 14.9 | 13.7 | 7.2 7.4 3.4 7.4 | 6.8 7.2 3.8 7.0 |
| B20_LateOldBabylonian | Stewards | 5.8 | 4.6 | 4.4 5.2 6.8 6.0 | 4.0 5.0 7.0 5.2 |

## vs `block1_ensemble_mean.csv`

- Cells matched: 80 of 80; only in reference 0; only in other 0

| Dimension | n both scored | MAE | exact | within 1 | Spearman | mean diff (other - ref) | NA agree | NA only ref | NA only other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coherence | 78 | 0.21 | 33% | 100% | 0.98 | -0.16 | 1 | 0 | 1 |
| Capacity | 78 | 0.16 | 54% | 100% | 0.99 | -0.11 | 1 | 0 | 1 |
| Stress | 78 | 0.14 | 44% | 100% | 0.99 | -0.03 | 1 | 0 | 1 |
| Abstraction | 78 | 0.31 | 24% | 100% | 0.98 | -0.26 | 1 | 0 | 1 |

- Case-level ranking agreement (Spearman on mean Node Value, 10 cases): 1.00

### Ten largest Node Value disagreements

| Society | Node | NV ref | NV other | C K S A ref | C K S A other |
| --- | --- | --- | --- | --- | --- |
| B13_Egypt_5thDynasty | Lore | 16.2 | 14.0 | 7.8 7.8 3.4 8.0 | 7.2 7.0 3.8 7.2 |
| B05_FirstIntermediate_Egypt | Stewards | 4.3 | 2.6 | 3.8 5.0 7.0 5.0 | 3.6 4.4 7.4 4.0 |
| B15_Egypt_MiddleKingdom | Flow | 12.7 | 11.0 | 6.6 6.8 3.8 6.2 | 6.0 6.0 4.0 6.0 |
| B13_Egypt_5thDynasty | Flow | 11.7 | 10.3 | 6.2 6.4 3.8 5.8 | 5.6 6.0 4.0 5.4 |
| B13_Egypt_5thDynasty | Stewards | 12.4 | 11.1 | 6.8 7.0 4.6 6.4 | 6.0 7.0 4.8 5.8 |
| B05_FirstIntermediate_Egypt | Shield | 2.0 | 0.8 | 3.0 4.6 7.6 4.0 | 3.0 4.0 7.8 3.2 |
| B09_LBA_Collapse_Aegean | Lore | 3.7 | 2.6 | 4.0 4.0 6.2 3.8 | 3.3 4.0 6.3 3.3 |
| B13_Egypt_5thDynasty | Shield | 9.2 | 8.1 | 5.6 5.2 4.0 4.8 | 5.0 5.0 4.0 4.2 |
| B13_Egypt_5thDynasty | Archive | 14.9 | 13.9 | 7.2 7.4 3.4 7.4 | 7.0 7.0 3.6 7.0 |
| B14_Egypt_6thDynasty | Shield | 6.1 | 5.1 | 4.6 5.0 6.0 5.0 | 4.0 4.8 5.8 4.2 |

## vs `promptB_raw_scores.csv`

- Cells matched: 80 of 80; only in reference 0; only in other 0

| Dimension | n both scored | MAE | exact | within 1 | Spearman | mean diff (other - ref) | NA agree | NA only ref | NA only other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coherence | 78 | 0.36 | 32% | 100% | 0.95 | +0.09 | 1 | 0 | 1 |
| Capacity | 78 | 0.46 | 37% | 91% | 0.94 | +0.35 | 1 | 0 | 1 |
| Stress | 78 | 0.53 | 21% | 94% | 0.91 | +0.34 | 1 | 0 | 1 |
| Abstraction | 78 | 0.48 | 21% | 92% | 0.91 | +0.03 | 1 | 0 | 1 |

- Case-level ranking agreement (Spearman on mean Node Value, 10 cases): 1.00

### Ten largest Node Value disagreements

| Society | Node | NV ref | NV other | C K S A ref | C K S A other |
| --- | --- | --- | --- | --- | --- |
| B05_FirstIntermediate_Egypt | Archive | 1.7 | 5.5 | 3.0 3.6 7.0 4.2 | 4.0 5.0 6.0 5.0 |
| B13_Egypt_5thDynasty | Lore | 16.2 | 12.5 | 7.8 7.8 3.4 8.0 | 7.0 7.0 5.0 7.0 |
| B14_Egypt_6thDynasty | Helm | 4.8 | 8.0 | 4.2 4.8 7.0 5.6 | 5.0 6.0 6.0 6.0 |
| B17_Greece_Protogeometric | Hands | 2.9 | 6.0 | 4.0 3.0 5.5 2.8 | 4.0 5.0 5.0 4.0 |
| B09_LBA_Collapse_Aegean | Hands | -0.3 | 2.5 | 3.0 3.4 8.0 2.6 | 4.0 4.0 7.0 3.0 |
| B13_Egypt_5thDynasty | Stewards | 12.4 | 10.0 | 6.8 7.0 4.6 6.4 | 6.0 7.0 6.0 6.0 |
| B15_Egypt_MiddleKingdom | Lore | 15.9 | 13.5 | 7.4 7.6 3.2 8.2 | 7.0 7.0 4.0 7.0 |
| B13_Egypt_5thDynasty | Craft | 15.9 | 13.5 | 7.6 7.8 3.2 7.4 | 7.0 7.0 4.0 7.0 |
| B16_Mycenaean_LHIIIA | Lore | 8.6 | 11.0 | 5.0 5.0 3.7 4.7 | 6.0 6.0 4.0 6.0 |
| B05_FirstIntermediate_Egypt | Flow | 0.6 | 3.0 | 3.0 3.0 7.0 3.2 | 4.0 4.0 7.0 4.0 |

