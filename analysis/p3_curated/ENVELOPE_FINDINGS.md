# Envelope quality findings (17 Oct 2026, repo at 446928c)

## Template envelopes (11 files, identical SD sequence)
The same C/K/S/A SD sequence appears across different societies, so these files do not record scorer disagreement:
- `juno/{Australia,China,Russia,UK,USA}_CAMS_m5n_2000-2026_block2.csv`: byte-identical SD columns
- `juno/{Belgium,Morocco,Portugal,Sri_Lanka}_CAMS_m5n_1900-2026_block2.csv`: byte-identical SD columns
- `juno/Japan_CAMS_m5n_1850-2026_block2.csv` and `cleaned_datasets/Japan_ENV_1850_2026_cleaned.csv`: contain the same sequence

The two groups share the sequence with each other (40-row window match), so all 11 hold one template. `scripts/check_envelopes.py` checks row keys and that spread is non-empty, so it does not catch this.

## Mostly-zero envelopes (share of rows with all four SDs = 0)
- Argentina 0.721
- Poland 0.813
- United Kingdom 0.804
- Italy 0.73
- Colombia 0.592
- Australia 0.125

Also above the 10% threshold, and additionally not annual: Canada 0.693, China 1850–2026 0.868, Sweden 0.474, Thailand 0.635, Türkiye 0.169, USA 0.582, France 5-yr 0.915, Japan 5-yr 0.125, Sweden 5-yr 0.617.

Full table: `run_eligibility.csv`.
