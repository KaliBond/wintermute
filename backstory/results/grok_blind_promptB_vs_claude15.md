# Model comparison

Reference: `block1_ensemble_mean.csv`

## vs `promptB_raw_scores.csv`

- Cells matched: 80 of 80; only in reference 0; only in other 0

| Dimension | n both scored | MAE | exact | within 1 | Spearman | mean diff (other - ref) | NA agree | NA only ref | NA only other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coherence | 77 | 0.53 | 9% | 90% | 0.93 | +0.42 | 1 | 0 | 2 |
| Capacity | 77 | 0.44 | 27% | 94% | 0.94 | +0.33 | 1 | 0 | 2 |
| Stress | 77 | 0.47 | 8% | 96% | 0.91 | +0.22 | 1 | 0 | 2 |
| Abstraction | 77 | 0.42 | 10% | 95% | 0.93 | +0.20 | 1 | 0 | 2 |

- Case-level ranking agreement (Spearman on mean Node Value, 10 cases): 0.98

### Ten largest Node Value disagreements

| Society | Node | NV ref | NV other | C K S A ref | C K S A other |
| --- | --- | --- | --- | --- | --- |
| B17_Greece_Protogeometric | Craft | 7.8 | 11.0 | 4.9 4.5 4.0 4.8 | 6.0 6.0 4.0 6.0 |
| B09_LBA_Collapse_Aegean | Stewards | -0.3 | 2.5 | 2.9 2.9 7.6 2.9 | 4.0 4.0 7.0 3.0 |
| B09_LBA_Collapse_Aegean | Helm | -2.2 | 0.5 | 2.3 2.5 8.3 2.7 | 3.0 4.0 8.0 3.0 |
| B05_FirstIntermediate_Egypt | Shield | 1.4 | 4.0 | 3.0 4.3 7.7 3.6 | 4.0 5.0 7.0 4.0 |
| B09_LBA_Collapse_Aegean | Hands | -0.1 | 2.5 | 3.0 3.5 7.9 2.7 | 4.0 4.0 7.0 3.0 |
| B16_Mycenaean_LHIIIA | Lore | 8.4 | 11.0 | 5.0 5.0 3.9 4.7 | 6.0 6.0 4.0 6.0 |
| B15_Egypt_MiddleKingdom | Hands | 9.5 | 12.0 | 5.7 7.0 5.8 5.2 | 7.0 7.0 5.0 6.0 |
| B05_FirstIntermediate_Egypt | Stewards | 3.5 | 1.0 | 3.7 4.7 7.1 4.4 | 3.0 4.0 8.0 4.0 |
| B17_Greece_Protogeometric | Hands | 3.0 | 5.5 | 4.0 3.1 5.4 2.6 | 5.0 4.0 5.0 3.0 |
| B05_FirstIntermediate_Egypt | Archive | 2.0 | 4.5 | 3.1 3.7 6.9 4.1 | 4.0 5.0 7.0 5.0 |

## vs `promptB_raw_scores.csv`

- Cells matched: 80 of 80; only in reference 0; only in other 0

| Dimension | n both scored | MAE | exact | within 1 | Spearman | mean diff (other - ref) | NA agree | NA only ref | NA only other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coherence | 78 | 0.36 | 14% | 96% | 0.95 | +0.21 | 1 | 0 | 1 |
| Capacity | 78 | 0.47 | 27% | 90% | 0.94 | +0.40 | 1 | 0 | 1 |
| Stress | 78 | 0.52 | 8% | 92% | 0.91 | +0.32 | 1 | 0 | 1 |
| Abstraction | 78 | 0.51 | 9% | 92% | 0.93 | +0.21 | 1 | 0 | 1 |

- Case-level ranking agreement (Spearman on mean Node Value, 10 cases): 1.00

### Ten largest Node Value disagreements

| Society | Node | NV ref | NV other | C K S A ref | C K S A other |
| --- | --- | --- | --- | --- | --- |
| B05_FirstIntermediate_Egypt | Archive | 2.0 | 5.5 | 3.1 3.7 6.9 4.1 | 4.0 5.0 6.0 5.0 |
| B14_Egypt_6thDynasty | Helm | 4.8 | 8.0 | 4.1 4.9 6.9 5.3 | 5.0 6.0 6.0 6.0 |
| B17_Greece_Protogeometric | Hands | 3.0 | 6.0 | 4.0 3.1 5.4 2.6 | 4.0 5.0 5.0 4.0 |
| B05_FirstIntermediate_Egypt | Hands | 0.7 | 3.5 | 3.1 4.0 7.9 3.0 | 4.0 5.0 7.0 3.0 |
| B09_LBA_Collapse_Aegean | Hands | -0.1 | 2.5 | 3.0 3.5 7.9 2.7 | 4.0 4.0 7.0 3.0 |
| B16_Mycenaean_LHIIIA | Lore | 8.4 | 11.0 | 5.0 5.0 3.9 4.7 | 6.0 6.0 4.0 6.0 |
| B13_Egypt_5thDynasty | Lore | 15.0 | 12.5 | 7.3 7.5 3.6 7.6 | 7.0 7.0 5.0 7.0 |
| B05_FirstIntermediate_Egypt | Flow | 0.6 | 3.0 | 2.9 3.0 6.9 3.1 | 4.0 4.0 7.0 4.0 |
| B17_Greece_Protogeometric | Craft | 7.8 | 10.0 | 4.9 4.5 4.0 4.8 | 5.0 6.0 4.0 6.0 |
| B17_Greece_Protogeometric | Helm | 2.9 | 5.0 | 3.5 3.0 5.0 2.8 | 4.0 4.0 5.0 4.0 |

