# Model comparison

Reference: `block1_ensemble_mean.csv`

## vs `promptA_raw_scores.csv`

- Cells matched: 96 of 96; only in reference 0; only in other 0

| Dimension | n both scored | MAE | exact | within 1 | Spearman | mean diff (other - ref) | NA agree | NA only ref | NA only other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coherence | 85 | 0.42 | 31% | 96% | 0.89 | +0.14 | 4 | 4 | 3 |
| Capacity | 85 | 0.42 | 36% | 99% | 0.93 | +0.36 | 4 | 4 | 3 |
| Stress | 85 | 0.53 | 26% | 95% | 0.89 | +0.27 | 4 | 4 | 3 |
| Abstraction | 85 | 0.46 | 29% | 92% | 0.91 | +0.29 | 4 | 4 | 3 |

- Case-level ranking agreement (Spearman on mean Node Value, 12 cases): 0.89

### Ten largest Node Value disagreements

| Society | Node | NV ref | NV other | C K S A ref | C K S A other |
| --- | --- | --- | --- | --- | --- |
| B08_Mycenaean_Greece | Hands | 5.4 | 9.5 | 4.4 6.0 6.6 3.2 | 6.0 7.0 6.0 5.0 |
| B05_FirstIntermediate_Egypt | Hands | 0.4 | 3.5 | 3.0 4.0 8.0 2.8 | 4.0 5.0 7.0 3.0 |
| B04_OldKingdom_Egypt | Hands | 9.9 | 13.0 | 6.2 8.0 6.2 3.8 | 7.0 8.0 5.0 6.0 |
| B09_LBA_Collapse_Aegean | Helm | -3.5 | -0.5 | 2.0 2.0 8.8 2.6 | 3.0 3.0 8.0 3.0 |
| B08_Mycenaean_Greece | Archive | 10.6 | 13.5 | 6.4 6.0 4.8 6.0 | 7.0 7.0 4.0 7.0 |
| B02_Shang_Anyang | Flow | 8.2 | 11.0 | 5.0 5.6 4.8 4.8 | 6.0 7.0 5.0 6.0 |
| B09_LBA_Collapse_Aegean | Hands | -0.3 | 2.5 | 3.0 3.4 8.0 2.6 | 4.0 4.0 7.0 3.0 |
| B08_Mycenaean_Greece | Lore | 8.4 | 11.0 | 5.2 5.4 4.8 5.2 | 6.0 6.0 4.0 6.0 |
| B07_Minoan_Crete | Lore | 10.5 | 13.0 | 6.4 6.0 5.0 6.2 | 7.0 7.0 4.0 6.0 |
| B02_Shang_Anyang | Hands | 4.1 | 6.5 | 4.0 6.0 7.4 3.0 | 5.0 7.0 8.0 5.0 |

