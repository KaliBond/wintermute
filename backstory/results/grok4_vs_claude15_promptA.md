# Model comparison

Reference: `block1_ensemble_mean.csv`

## vs `block1_ensemble_mean.csv`

- Cells matched: 96 of 96; only in reference 0; only in other 0

| Dimension | n both scored | MAE | exact | within 1 | Spearman | mean diff (other - ref) | NA agree | NA only ref | NA only other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coherence | 91 | 0.39 | 7% | 99% | 0.93 | +0.21 | 2 | 1 | 2 |
| Capacity | 91 | 0.53 | 5% | 91% | 0.93 | +0.47 | 2 | 1 | 2 |
| Stress | 91 | 0.42 | 7% | 97% | 0.95 | +0.19 | 2 | 1 | 2 |
| Abstraction | 91 | 0.45 | 10% | 92% | 0.94 | +0.34 | 2 | 1 | 2 |

- Case-level ranking agreement (Spearman on mean Node Value, 12 cases): 0.92

### Ten largest Node Value disagreements

| Society | Node | NV ref | NV other | C K S A ref | C K S A other |
| --- | --- | --- | --- | --- | --- |
| B05_FirstIntermediate_Egypt | Hands | 0.7 | 4.4 | 3.2 4.0 7.9 2.9 | 4.2 4.8 6.5 3.8 |
| B02_Shang_Anyang | Archive | 11.8 | 15.3 | 6.4 6.1 3.9 6.5 | 7.2 8.0 3.8 7.8 |
| B09_LBA_Collapse_Aegean | Stewards | -0.6 | 2.4 | 2.9 2.9 7.8 2.9 | 3.8 4.0 7.0 3.2 |
| B09_LBA_Collapse_Aegean | Helm | -2.9 | -0.0 | 2.1 2.3 8.7 2.8 | 3.2 3.2 8.2 3.5 |
| B08_Mycenaean_Greece | Hands | 6.0 | 8.6 | 4.7 6.0 6.5 3.7 | 5.5 7.0 6.5 5.2 |
| B12_Xiongnu_Steppe | Hands | 6.8 | 9.2 | 5.0 5.8 5.9 3.9 | 5.8 6.5 5.5 4.8 |
| B08_Mycenaean_Greece | Lore | 8.6 | 10.8 | 5.3 5.5 4.7 5.1 | 6.2 6.2 4.5 5.8 |
| B09_LBA_Collapse_Aegean | Hands | 0.0 | 2.2 | 3.1 3.5 7.9 2.7 | 3.8 4.2 7.5 3.5 |
| B05_FirstIntermediate_Egypt | Archive | 3.8 | 5.9 | 3.8 3.9 6.2 4.7 | 4.2 5.0 5.8 5.0 |
| B01_Longshan_YellowRiver | Shield | 5.1 | 7.2 | 4.4 5.0 6.2 3.8 | 5.0 6.0 6.0 4.5 |

