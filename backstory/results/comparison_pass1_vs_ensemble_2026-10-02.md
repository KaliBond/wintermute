# Model comparison

Reference: `runs/claude-opus-5-5_pass1/raw_scores.csv`

## vs `runs/ensemble_claude-subagents_2026-10-02/block1_ensemble_mean.csv`

- Cells matched: 96 of 96; only in reference 0; only in other 0

| Dimension | n both scored | MAE | exact | within 1 | Spearman | mean diff (other - ref) | NA agree | NA only ref | NA only other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coherence | 85 | 0.44 | 33% | 96% | 0.94 | -0.09 | 6 | 3 | 2 |
| Capacity | 85 | 0.33 | 48% | 99% | 0.94 | -0.09 | 6 | 3 | 2 |
| Stress | 85 | 0.46 | 27% | 93% | 0.93 | +0.12 | 6 | 3 | 2 |
| Abstraction | 85 | 0.48 | 21% | 99% | 0.95 | +0.34 | 6 | 3 | 2 |

- Case-level ranking agreement (Spearman on mean Node Value, 12 cases): 0.93

### Ten largest Node Value disagreements

| Society | Node | NV ref | NV other | C K S A ref | C K S A other |
| --- | --- | --- | --- | --- | --- |
| B03_Harappan_Indus | Archive | 15.0 | 8.8 | 8.0 7.0 3.0 6.0 | 6.0 4.5 4.2 5.0 |
| B09_LBA_Collapse_Aegean | Flow | -1.5 | 2.2 | 2.0 3.0 8.0 3.0 | 3.4 3.8 7.0 4.0 |
| B09_LBA_Collapse_Aegean | Stewards | -4.0 | -0.3 | 2.0 2.0 9.0 2.0 | 3.0 3.0 7.8 3.0 |
| B05_FirstIntermediate_Egypt | Hands | -3.0 | 0.4 | 2.0 3.0 9.0 2.0 | 3.0 4.0 8.0 2.8 |
| B09_LBA_Collapse_Aegean | Shield | -2.5 | 0.8 | 2.0 3.0 9.0 3.0 | 3.0 3.8 7.8 3.6 |
| B05_FirstIntermediate_Egypt | Stewards | 1.0 | 4.2 | 3.0 4.0 8.0 4.0 | 4.0 4.8 7.0 4.8 |
| B04_OldKingdom_Egypt | Stewards | 16.5 | 13.6 | 8.0 8.0 3.0 7.0 | 7.0 7.6 4.0 6.0 |
| B05_FirstIntermediate_Egypt | Lore | 4.0 | 6.5 | 3.0 5.0 7.0 6.0 | 4.4 5.2 6.0 5.8 |
| B07_Minoan_Crete | Lore | 13.0 | 10.5 | 7.0 7.0 4.0 6.0 | 6.4 6.0 5.0 6.2 |
| B11_Maori_Aotearoa | Flow | 10.5 | 8.1 | 6.0 6.0 4.0 5.0 | 5.2 5.4 5.0 5.0 |

