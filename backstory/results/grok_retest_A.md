# Model comparison

Reference: `promptA_raw_scores.csv`

## vs `promptA_raw_scores.csv`

- Cells matched: 96 of 96; only in reference 0; only in other 0

| Dimension | n both scored | MAE | exact | within 1 | Spearman | mean diff (other - ref) | NA agree | NA only ref | NA only other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coherence | 89 | 0.28 | 73% | 99% | 0.87 | -0.10 | 4 | 3 | 0 |
| Capacity | 89 | 0.22 | 78% | 100% | 0.92 | -0.02 | 4 | 3 | 0 |
| Stress | 89 | 0.42 | 60% | 99% | 0.89 | -0.33 | 4 | 3 | 0 |
| Abstraction | 89 | 0.25 | 75% | 100% | 0.94 | -0.11 | 4 | 3 | 0 |

- Case-level ranking agreement (Spearman on mean Node Value, 12 cases): 0.83

### Ten largest Node Value disagreements

| Society | Node | NV ref | NV other | C K S A ref | C K S A other |
| --- | --- | --- | --- | --- | --- |
| B02_Shang_Anyang | Lore | 11.5 | 15.0 | 7.0 7.0 6.0 7.0 | 7.0 8.0 4.0 8.0 |
| B03_Harappan_Indus | Lore | 8.5 | 12.0 | 5.0 5.0 4.0 5.0 | 7.0 6.0 4.0 6.0 |
| B07_Minoan_Crete | Stewards | 10.5 | 14.0 | 6.0 7.0 5.0 5.0 | 7.0 8.0 4.0 6.0 |
| B06_OldBabylonian_Mesopotamia | Shield | 10.0 | 13.0 | 6.0 7.0 6.0 6.0 | 7.0 8.0 5.0 6.0 |
| B11_Maori_Aotearoa | Archive | 13.5 | 10.5 | 7.0 7.0 4.0 7.0 | 6.0 6.0 5.0 7.0 |
| B10_Aboriginal_Australia | Craft | 14.5 | 12.0 | 7.0 7.0 3.0 7.0 | 6.0 6.0 3.0 6.0 |
| B02_Shang_Anyang | Flow | 11.0 | 8.5 | 6.0 7.0 5.0 6.0 | 5.0 6.0 5.0 5.0 |
| B11_Maori_Aotearoa | Flow | 10.0 | 7.5 | 6.0 6.0 5.0 6.0 | 5.0 5.0 5.0 5.0 |
| B10_Aboriginal_Australia | Flow | 13.5 | 11.0 | 7.0 7.0 4.0 7.0 | 6.0 6.0 4.0 6.0 |
| B09_LBA_Collapse_Aegean | Flow | 1.5 | -0.5 | 3.0 4.0 7.0 3.0 | 3.0 3.0 8.0 3.0 |

