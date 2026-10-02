# Model comparison

Reference: `block1_ensemble_mean.csv`

## vs `block1_ensemble_mean.csv`

- Cells matched: 80 of 80; only in reference 0; only in other 0

| Dimension | n both scored | MAE | exact | within 1 | Spearman | mean diff (other - ref) | NA agree | NA only ref | NA only other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coherence | 77 | 0.43 | 5% | 95% | 0.96 | +0.36 | 1 | 0 | 2 |
| Capacity | 77 | 0.44 | 12% | 95% | 0.96 | +0.37 | 1 | 0 | 2 |
| Stress | 77 | 0.57 | 1% | 94% | 0.95 | +0.43 | 1 | 0 | 2 |
| Abstraction | 77 | 0.41 | 5% | 95% | 0.95 | +0.28 | 1 | 0 | 2 |

- Case-level ranking agreement (Spearman on mean Node Value, 10 cases): 1.00

### Ten largest Node Value disagreements

| Society | Node | NV ref | NV other | C K S A ref | C K S A other |
| --- | --- | --- | --- | --- | --- |
| B05_FirstIntermediate_Egypt | Hands | 0.7 | 3.4 | 3.1 4.0 7.9 3.0 | 4.0 4.8 7.2 3.5 |
| B05_FirstIntermediate_Egypt | Helm | -0.9 | 1.7 | 2.1 3.0 8.0 4.0 | 3.2 4.2 7.8 4.2 |
| B09_LBA_Collapse_Aegean | Helm | -2.2 | 0.3 | 2.3 2.5 8.3 2.7 | 3.0 3.8 8.0 3.0 |
| B16_Mycenaean_LHIIIA | Lore | 8.4 | 10.8 | 5.0 5.0 3.9 4.7 | 6.2 6.0 4.2 5.5 |
| B09_LBA_Collapse_Aegean | Hands | -0.1 | 2.3 | 3.0 3.5 7.9 2.7 | 4.0 4.0 7.2 3.0 |
| B17_Greece_Protogeometric | Craft | 7.8 | 10.2 | 4.9 4.5 4.0 4.8 | 6.0 6.0 4.8 6.0 |
| B13_Egypt_5thDynasty | Craft | 15.5 | 13.2 | 7.3 7.8 3.3 7.4 | 7.0 7.0 4.2 6.8 |
| B14_Egypt_6thDynasty | Helm | 4.8 | 7.1 | 4.1 4.9 6.9 5.3 | 5.0 5.5 6.0 5.2 |
| B09_LBA_Collapse_Aegean | Craft | 3.2 | 5.5 | 3.9 4.0 6.7 4.0 | 4.5 4.8 6.2 4.8 |
| B15_Egypt_MiddleKingdom | Hands | 9.5 | 11.7 | 5.7 7.0 5.8 5.2 | 6.8 7.0 5.2 6.2 |

