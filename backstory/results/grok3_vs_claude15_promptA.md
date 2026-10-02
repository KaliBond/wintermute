# Model comparison

Reference: `block1_ensemble_mean.csv`

## vs `block1_ensemble_mean.csv`

- Cells matched: 96 of 96; only in reference 0; only in other 0

| Dimension | n both scored | MAE | exact | within 1 | Spearman | mean diff (other - ref) | NA agree | NA only ref | NA only other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coherence | 91 | 0.36 | 10% | 100% | 0.94 | +0.17 | 2 | 1 | 2 |
| Capacity | 91 | 0.47 | 13% | 97% | 0.94 | +0.38 | 2 | 1 | 2 |
| Stress | 91 | 0.40 | 9% | 98% | 0.95 | +0.12 | 2 | 1 | 2 |
| Abstraction | 91 | 0.42 | 14% | 95% | 0.94 | +0.26 | 2 | 1 | 2 |

- Case-level ranking agreement (Spearman on mean Node Value, 12 cases): 0.92

### Ten largest Node Value disagreements

| Society | Node | NV ref | NV other | C K S A ref | C K S A other |
| --- | --- | --- | --- | --- | --- |
| B02_Shang_Anyang | Archive | 11.8 | 15.4 | 6.4 6.1 3.9 6.5 | 7.3 8.0 3.7 7.7 |
| B05_FirstIntermediate_Egypt | Hands | 0.7 | 4.2 | 3.2 4.0 7.9 2.9 | 4.0 4.7 6.3 3.7 |
| B09_LBA_Collapse_Aegean | Stewards | -0.6 | 2.4 | 2.9 2.9 7.8 2.9 | 3.7 4.0 7.0 3.3 |
| B08_Mycenaean_Greece | Hands | 6.0 | 8.9 | 4.7 6.0 6.5 3.7 | 5.7 7.0 6.3 5.0 |
| B01_Longshan_YellowRiver | Shield | 5.1 | 7.4 | 4.4 5.0 6.2 3.8 | 5.0 6.0 6.0 4.7 |
| B08_Mycenaean_Greece | Lore | 8.6 | 10.8 | 5.3 5.5 4.7 5.1 | 6.3 6.0 4.3 5.7 |
| B09_LBA_Collapse_Aegean | Helm | -2.9 | -0.7 | 2.1 2.3 8.7 2.8 | 3.0 3.0 8.3 3.3 |
| B12_Xiongnu_Steppe | Hands | 6.8 | 9.0 | 5.0 5.8 5.9 3.9 | 5.7 6.3 5.3 4.7 |
| B08_Mycenaean_Greece | Archive | 10.7 | 12.8 | 6.3 6.3 4.9 6.0 | 7.0 7.0 4.7 7.0 |
| B07_Minoan_Crete | Helm | 9.2 | 11.3 | 5.5 6.0 4.9 5.2 | 6.0 7.0 4.7 6.0 |

