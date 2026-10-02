# Model comparison

Reference: `block1_ensemble_mean.csv`

## vs `block1_ensemble_mean.csv`

- Cells matched: 96 of 96; only in reference 0; only in other 0

| Dimension | n both scored | MAE | exact | within 1 | Spearman | mean diff (other - ref) | NA agree | NA only ref | NA only other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coherence | 88 | 0.23 | 28% | 100% | 0.97 | -0.08 | 5 | 0 | 3 |
| Capacity | 88 | 0.17 | 47% | 100% | 0.99 | -0.05 | 5 | 0 | 3 |
| Stress | 88 | 0.20 | 35% | 100% | 0.98 | +0.12 | 5 | 0 | 3 |
| Abstraction | 88 | 0.23 | 38% | 99% | 0.97 | -0.05 | 5 | 0 | 3 |

- Case-level ranking agreement (Spearman on mean Node Value, 12 cases): 0.98

### Ten largest Node Value disagreements

| Society | Node | NV ref | NV other | C K S A ref | C K S A other |
| --- | --- | --- | --- | --- | --- |
| B04_OldKingdom_Egypt | Hands | 12.3 | 9.9 | 7.0 8.6 6.0 5.4 | 6.2 8.0 6.2 3.8 |
| B02_Shang_Anyang | Hands | 6.0 | 4.1 | 5.0 6.2 7.0 3.6 | 4.0 6.0 7.4 3.0 |
| B04_OldKingdom_Egypt | Flow | 12.6 | 11.1 | 6.2 6.8 3.4 6.0 | 6.0 6.2 3.8 5.4 |
| B05_FirstIntermediate_Egypt | Stewards | 2.8 | 4.2 | 3.8 4.0 7.2 4.4 | 4.0 4.8 7.0 4.8 |
| B12_Xiongnu_Steppe | Craft | 8.5 | 9.8 | 5.0 5.2 4.2 5.0 | 5.6 5.6 4.2 5.6 |
| B09_LBA_Collapse_Aegean | Helm | -2.3 | -3.5 | 2.2 2.6 8.6 3.0 | 2.0 2.0 8.8 2.6 |
| B11_Maori_Aotearoa | Stewards | 8.6 | 7.4 | 5.8 5.8 5.8 5.6 | 5.2 5.8 6.4 5.6 |
| B01_Longshan_YellowRiver | Lore | 7.5 | 8.7 | 5.2 5.0 5.2 5.0 | 5.6 5.2 4.8 5.4 |
| B05_FirstIntermediate_Egypt | Hands | 1.5 | 0.4 | 3.6 4.0 7.6 3.0 | 3.0 4.0 8.0 2.8 |
| B08_Mycenaean_Greece | Hands | 6.5 | 5.4 | 4.8 6.0 6.4 4.2 | 4.4 6.0 6.6 3.2 |

## vs `block1_ensemble_mean.csv`

- Cells matched: 96 of 96; only in reference 0; only in other 0

| Dimension | n both scored | MAE | exact | within 1 | Spearman | mean diff (other - ref) | NA agree | NA only ref | NA only other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coherence | 91 | 0.14 | 48% | 100% | 0.98 | -0.10 | 4 | 1 | 0 |
| Capacity | 91 | 0.17 | 38% | 100% | 0.99 | -0.14 | 4 | 1 | 0 |
| Stress | 91 | 0.15 | 44% | 100% | 0.99 | +0.01 | 4 | 1 | 0 |
| Abstraction | 91 | 0.17 | 41% | 100% | 0.99 | -0.10 | 4 | 1 | 0 |

- Case-level ranking agreement (Spearman on mean Node Value, 12 cases): 0.99

### Ten largest Node Value disagreements

| Society | Node | NV ref | NV other | C K S A ref | C K S A other |
| --- | --- | --- | --- | --- | --- |
| B11_Maori_Aotearoa | Archive | 13.4 | 11.7 | 7.0 6.8 3.8 6.8 | 6.4 6.2 4.0 6.2 |
| B07_Minoan_Crete | Archive | 9.8 | 8.2 | 5.4 5.4 4.0 6.0 | 5.0 5.0 4.4 5.2 |
| B11_Maori_Aotearoa | Flow | 8.9 | 7.6 | 5.4 5.4 4.4 5.0 | 5.0 5.0 4.8 4.8 |
| B09_LBA_Collapse_Aegean | Flow | 2.2 | 0.9 | 3.6 3.6 7.0 4.0 | 3.2 3.2 7.4 3.8 |
| B11_Maori_Aotearoa | Stewards | 8.6 | 7.3 | 5.8 5.8 5.8 5.6 | 5.4 5.4 6.0 5.0 |
| B05_FirstIntermediate_Egypt | Hands | 1.5 | 0.5 | 3.6 4.0 7.6 3.0 | 3.0 4.0 8.0 3.0 |
| B06_OldBabylonian_Mesopotamia | Flow | 12.3 | 11.3 | 6.6 7.0 4.8 7.0 | 6.0 6.8 5.0 7.0 |
| B04_OldKingdom_Egypt | Hands | 12.3 | 11.3 | 7.0 8.6 6.0 5.4 | 6.8 8.0 6.0 5.0 |
| B10_Aboriginal_Australia | Shield | 8.2 | 7.2 | 5.2 5.0 4.0 4.0 | 5.0 4.4 4.2 4.0 |
| B03_Harappan_Indus | Archive | 9.8 | 8.9 | 5.8 5.0 3.8 5.5 | 6.0 4.7 4.3 5.0 |

## vs `promptA_raw_scores.csv`

- Cells matched: 96 of 96; only in reference 0; only in other 0

| Dimension | n both scored | MAE | exact | within 1 | Spearman | mean diff (other - ref) | NA agree | NA only ref | NA only other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coherence | 87 | 0.42 | 30% | 94% | 0.88 | +0.08 | 3 | 2 | 4 |
| Capacity | 87 | 0.42 | 33% | 98% | 0.93 | +0.31 | 3 | 2 | 4 |
| Stress | 87 | 0.60 | 16% | 92% | 0.88 | +0.37 | 3 | 2 | 4 |
| Abstraction | 87 | 0.45 | 37% | 98% | 0.91 | +0.25 | 3 | 2 | 4 |

- Case-level ranking agreement (Spearman on mean Node Value, 12 cases): 0.92

### Ten largest Node Value disagreements

| Society | Node | NV ref | NV other | C K S A ref | C K S A other |
| --- | --- | --- | --- | --- | --- |
| B04_OldKingdom_Egypt | Helm | 17.0 | 13.5 | 8.2 8.6 3.6 7.6 | 7.0 8.0 5.0 7.0 |
| B12_Xiongnu_Steppe | Shield | 14.4 | 11.0 | 7.4 8.2 4.4 6.4 | 6.0 8.0 6.0 6.0 |
| B08_Mycenaean_Greece | Hands | 6.5 | 9.5 | 4.8 6.0 6.4 4.2 | 6.0 7.0 6.0 5.0 |
| B08_Mycenaean_Greece | Archive | 10.8 | 13.5 | 6.4 6.4 5.0 6.0 | 7.0 7.0 4.0 7.0 |
| B12_Xiongnu_Steppe | Flow | 11.2 | 8.5 | 6.2 7.0 5.0 6.0 | 5.0 7.0 6.0 5.0 |
| B06_OldBabylonian_Mesopotamia | Helm | 14.1 | 11.5 | 7.6 8.0 5.0 7.0 | 6.0 8.0 6.0 7.0 |
| B02_Shang_Anyang | Flow | 8.5 | 11.0 | 5.0 5.6 4.6 5.0 | 6.0 7.0 5.0 6.0 |
| B10_Aboriginal_Australia | Craft | 12.0 | 14.5 | 6.0 6.0 3.0 6.0 | 7.0 7.0 3.0 7.0 |
| B08_Mycenaean_Greece | Lore | 8.7 | 11.0 | 5.4 5.6 4.8 5.0 | 6.0 6.0 4.0 6.0 |
| B02_Shang_Anyang | Archive | 11.7 | 14.0 | 6.4 6.0 3.8 6.2 | 7.0 7.0 4.0 8.0 |

