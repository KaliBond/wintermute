# Model comparison

Reference: `block1_ensemble_mean.csv`

## vs `promptA_raw_scores.csv`

- Cells matched: 96 of 96; only in reference 0; only in other 0

| Dimension | n both scored | MAE | exact | within 1 | Spearman | mean diff (other - ref) | NA agree | NA only ref | NA only other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coherence | 92 | 0.73 | 11% | 86% | 0.84 | +0.60 | 2 | 2 | 0 |
| Capacity | 92 | 0.46 | 30% | 93% | 0.85 | +0.06 | 2 | 2 | 0 |
| Stress | 92 | 0.45 | 20% | 93% | 0.85 | -0.10 | 2 | 2 | 0 |
| Abstraction | 92 | 0.92 | 8% | 71% | 0.80 | -0.82 | 2 | 2 | 0 |

- Case-level ranking agreement (Spearman on mean Node Value, 12 cases): 0.83

### Ten largest Node Value disagreements

| Society | Node | NV ref | NV other | C K S A ref | C K S A other |
| --- | --- | --- | --- | --- | --- |
| B02_Shang_Anyang | Shield | 9.8 | 14.0 | 6.0 7.0 6.2 6.0 | 8.0 8.0 5.0 6.0 |
| B10_Aboriginal_Australia | Stewards | 12.2 | 8.0 | 7.0 6.2 4.5 7.0 | 6.0 5.0 5.0 4.0 |
| B12_Xiongnu_Steppe | Craft | 10.9 | 7.0 | 5.8 6.2 4.2 6.2 | 5.0 5.0 5.0 4.0 |
| B02_Shang_Anyang | Flow | 9.1 | 13.0 | 5.8 6.2 5.5 5.2 | 7.0 7.0 4.0 6.0 |
| B10_Aboriginal_Australia | Helm | 10.6 | 7.0 | 6.0 5.5 4.0 6.2 | 5.0 5.0 5.0 4.0 |
| B10_Aboriginal_Australia | Craft | 12.3 | 9.0 | 6.2 6.5 3.5 6.2 | 6.0 5.0 4.0 4.0 |
| B02_Shang_Anyang | Helm | 12.5 | 15.5 | 6.5 7.8 5.2 6.8 | 8.0 8.0 4.0 7.0 |
| B07_Minoan_Crete | Helm | 11.0 | 14.0 | 6.0 7.0 5.0 6.0 | 8.0 7.0 4.0 6.0 |
| B05_FirstIntermediate_Egypt | Hands | 4.4 | 1.5 | 4.2 4.8 6.5 3.8 | 4.0 4.0 8.0 3.0 |
| B03_Harappan_Indus | Archive | 10.2 | 13.0 | 6.0 5.5 4.2 5.8 | 7.0 7.0 4.0 6.0 |

