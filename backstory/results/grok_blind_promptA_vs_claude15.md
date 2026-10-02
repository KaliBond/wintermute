# Model comparison

Reference: `block1_ensemble_mean.csv`

## vs `promptA_raw_scores.csv`

- Cells matched: 96 of 96; only in reference 0; only in other 0

| Dimension | n both scored | MAE | exact | within 1 | Spearman | mean diff (other - ref) | NA agree | NA only ref | NA only other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coherence | 91 | 0.37 | 16% | 98% | 0.90 | +0.03 | 2 | 1 | 2 |
| Capacity | 91 | 0.50 | 12% | 93% | 0.92 | +0.35 | 2 | 1 | 2 |
| Stress | 91 | 0.41 | 18% | 96% | 0.92 | +0.00 | 2 | 1 | 2 |
| Abstraction | 91 | 0.43 | 13% | 93% | 0.92 | +0.20 | 2 | 1 | 2 |

- Case-level ranking agreement (Spearman on mean Node Value, 12 cases): 0.87

### Ten largest Node Value disagreements

| Society | Node | NV ref | NV other | C K S A ref | C K S A other |
| --- | --- | --- | --- | --- | --- |
| B05_FirstIntermediate_Egypt | Hands | 0.7 | 4.5 | 3.2 4.0 7.9 2.9 | 4.0 5.0 6.0 3.0 |
| B02_Shang_Anyang | Archive | 11.8 | 15.0 | 6.4 6.1 3.9 6.5 | 7.0 8.0 4.0 8.0 |
| B08_Mycenaean_Greece | Helm | 9.8 | 12.5 | 6.1 6.6 5.9 6.0 | 7.0 7.0 5.0 7.0 |
| B05_FirstIntermediate_Egypt | Flow | 1.4 | 4.0 | 3.2 3.4 6.9 3.5 | 4.0 4.0 6.0 4.0 |
| B08_Mycenaean_Greece | Hands | 6.0 | 8.5 | 4.7 6.0 6.5 3.7 | 5.0 7.0 6.0 5.0 |
| B11_Maori_Aotearoa | Archive | 13.0 | 10.5 | 6.9 6.7 3.9 6.7 | 6.0 6.0 5.0 7.0 |
| B07_Minoan_Crete | Stewards | 11.6 | 14.0 | 6.1 6.9 4.3 5.8 | 7.0 8.0 4.0 6.0 |
| B08_Mycenaean_Greece | Lore | 8.6 | 11.0 | 5.3 5.5 4.7 5.1 | 6.0 6.0 4.0 6.0 |
| B09_LBA_Collapse_Aegean | Helm | -2.9 | -0.5 | 2.1 2.3 8.7 2.8 | 3.0 3.0 8.0 3.0 |
| B03_Harappan_Indus | Lore | 9.7 | 12.0 | 6.0 5.0 3.8 5.0 | 7.0 6.0 4.0 6.0 |

## vs `promptA_raw_scores.csv`

- Cells matched: 96 of 96; only in reference 0; only in other 0

| Dimension | n both scored | MAE | exact | within 1 | Spearman | mean diff (other - ref) | NA agree | NA only ref | NA only other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coherence | 89 | 0.43 | 17% | 93% | 0.88 | +0.12 | 3 | 0 | 4 |
| Capacity | 89 | 0.43 | 18% | 94% | 0.93 | +0.36 | 3 | 0 | 4 |
| Stress | 89 | 0.57 | 12% | 89% | 0.89 | +0.33 | 3 | 0 | 4 |
| Abstraction | 89 | 0.47 | 13% | 91% | 0.92 | +0.29 | 3 | 0 | 4 |

- Case-level ranking agreement (Spearman on mean Node Value, 12 cases): 0.92

### Ten largest Node Value disagreements

| Society | Node | NV ref | NV other | C K S A ref | C K S A other |
| --- | --- | --- | --- | --- | --- |
| B08_Mycenaean_Greece | Hands | 6.0 | 9.5 | 4.7 6.0 6.5 3.7 | 6.0 7.0 6.0 5.0 |
| B04_OldKingdom_Egypt | Helm | 16.4 | 13.5 | 8.1 8.4 3.7 7.2 | 7.0 8.0 5.0 7.0 |
| B02_Shang_Anyang | Flow | 8.2 | 11.0 | 5.0 5.5 4.7 4.9 | 6.0 7.0 5.0 6.0 |
| B08_Mycenaean_Greece | Archive | 10.7 | 13.5 | 6.3 6.3 4.9 6.0 | 7.0 7.0 4.0 7.0 |
| B12_Xiongnu_Steppe | Shield | 13.8 | 11.0 | 7.2 8.1 4.7 6.5 | 6.0 8.0 6.0 6.0 |
| B05_FirstIntermediate_Egypt | Hands | 0.7 | 3.5 | 3.2 4.0 7.9 2.9 | 4.0 5.0 7.0 3.0 |
| B09_LBA_Collapse_Aegean | Hands | 0.0 | 2.5 | 3.1 3.5 7.9 2.7 | 4.0 4.0 7.0 3.0 |
| B10_Aboriginal_Australia | Craft | 12.0 | 14.5 | 6.1 6.0 3.1 6.1 | 7.0 7.0 3.0 7.0 |
| B01_Longshan_YellowRiver | Hands | 6.6 | 9.0 | 5.0 5.9 6.0 3.5 | 6.0 7.0 6.0 4.0 |
| B08_Mycenaean_Greece | Lore | 8.6 | 11.0 | 5.3 5.5 4.7 5.1 | 6.0 6.0 4.0 6.0 |

