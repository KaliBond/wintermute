# Model comparison

Reference: `block1_ensemble_mean.csv`

## vs `block1_ensemble_mean.csv`

- Cells matched: 80 of 80; only in reference 0; only in other 0

| Dimension | n both scored | MAE | exact | within 1 | Spearman | mean diff (other - ref) | NA agree | NA only ref | NA only other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coherence | 77 | 0.44 | 8% | 97% | 0.96 | +0.37 | 1 | 0 | 2 |
| Capacity | 77 | 0.43 | 14% | 95% | 0.96 | +0.36 | 1 | 0 | 2 |
| Stress | 77 | 0.55 | 8% | 91% | 0.94 | +0.43 | 1 | 0 | 2 |
| Abstraction | 77 | 0.43 | 8% | 92% | 0.95 | +0.34 | 1 | 0 | 2 |

- Case-level ranking agreement (Spearman on mean Node Value, 10 cases): 1.00

### Ten largest Node Value disagreements

| Society | Node | NV ref | NV other | C K S A ref | C K S A other |
| --- | --- | --- | --- | --- | --- |
| B09_LBA_Collapse_Aegean | Helm | -2.2 | 0.5 | 2.3 2.5 8.3 2.7 | 3.0 4.0 8.0 3.0 |
| B17_Greece_Protogeometric | Craft | 7.8 | 10.3 | 4.9 4.5 4.0 4.8 | 6.0 6.0 4.7 6.0 |
| B16_Mycenaean_LHIIIA | Lore | 8.4 | 10.8 | 5.0 5.0 3.9 4.7 | 6.3 6.0 4.3 5.7 |
| B13_Egypt_5thDynasty | Craft | 15.5 | 13.2 | 7.3 7.8 3.3 7.4 | 7.0 7.0 4.3 7.0 |
| B09_LBA_Collapse_Aegean | Hands | -0.1 | 2.2 | 3.0 3.5 7.9 2.7 | 4.0 4.0 7.3 3.0 |
| B05_FirstIntermediate_Egypt | Hands | 0.7 | 3.0 | 3.1 4.0 7.9 3.0 | 4.0 4.7 7.3 3.3 |
| B14_Egypt_6thDynasty | Helm | 4.8 | 7.0 | 4.1 4.9 6.9 5.3 | 5.0 5.3 6.0 5.3 |
| B05_FirstIntermediate_Egypt | Helm | -0.9 | 1.2 | 2.1 3.0 8.0 4.0 | 3.0 4.0 8.0 4.3 |
| B15_Egypt_MiddleKingdom | Hands | 9.5 | 11.5 | 5.7 7.0 5.8 5.2 | 6.7 7.0 5.3 6.3 |
| B09_LBA_Collapse_Aegean | Craft | 3.2 | 5.1 | 3.9 4.0 6.7 4.0 | 4.3 4.7 6.3 4.7 |

