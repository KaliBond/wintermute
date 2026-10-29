# Blind-rater accidental repeatability test (29 Oct 2026)

Research note plus the data and script that reproduce its figures. Run `python3 verify.py` from this folder; it reproduces `output.txt` (and writes three helper CSVs, not committed).

In `verify.py` and `output.txt` the rater keys are `FROZEN` = GPT, `CLAUDE_files27` = Claude, `UPLOAD_FILLED` = Grok.

| File | Origin / original name | SHA-256 |
|---|---|---|
| `data/BLIND_RESPONSE_FROZEN_GPT_200.csv` | BLIND_RESPONSE_FROZEN_200.csv (frozen response, GPT) | `cc2d67ab07e3a166359d6b8263236460aa1906ae979e559fa08e0e6950cf8a0b` |
| `data/BLIND_RESPONSE_CLAUDE_200.csv` | BLIND_RESPONSE_FILLED_200.csv in files28.zip (identical in files27.zip), Claude | `af6caee554144183cc9e84640ebe4aca9ea2167d5e327e0fc61766d9b60e2a31` |
| `data/BLIND_RESPONSE_GROK_200.csv` | BLIND_RESPONSE_FILLED_200.csv (loose upload), Grok | `19f6df66e34a699ef25f50668657ecd073f64b567c4221d4f8ce5960515fd81a` |
| `data/RATER_DECLARATION.md` | RATER_DECLARATION.md (files27.zip / files28.zip) | `aa3504db21e6aa642571312caa8f869c11bec4c8e78a237fb3a12332fac6a579` |
| `data/BLIND_CASEBOOK_200.csv` | rebuilt.zip | `89eb51e75c105f199fa0e033913e80dd0a54b9cd124b056911a84d9c457e104f` |
| `data/BLIND_RESPONSE_TEMPLATE_200.csv` | rebuilt.zip | `d00925e710fbd91d12734093a06e77db95b20f0b6908f5dbfd0e1e34ebda78a3` |
| `data/NAIVE_SCORER_INSTRUCTIONS.md` | rebuilt.zip | `33064c0b7a7d7e87d3e704fbd781aab8e6b0659d41fe8693e00575bba86d9759` |
| `data/README_FIRST.md` | rebuilt.zip | `8110db8f324cd21e3f37d908d172bb6678343ee9a787ef570e9b878c19f48641` |
| `CAMS_Accidental_Repeatability_Test_GPT_2026-10-29.md` | research note | `2f045779236d2be08290437f3a8d8015fa2c13dda5d20073102e24063defd3d4` |
| `verify.py` | verification script | `b385fabece944be982d3b23cbcc4217fe74e629310beb494f5be31a3c4161a8b` |
| `output.txt` | output of verify.py | `b3d0606fc3c44819cc236e342b94be2a25133fa16d4705fdc3a68e235fdba776` |
