# CAMS v1.0-Final Batch Recompute Report

**Engine:** `cams_framework_v2_4.py` (v2.4)
**Source doc:** `papers/CAMS_v1.0-Final_Framework_Conclusion.md`

## Canonical operators

```
V_i  = C + K - S + 0.5*A
q_i  = (0.6*C + 0.4*A) / 10
B_ij = sqrt(q_i * q_j) * 2**(-(S_i + S_j)/10)   ∈ [0, 1]  for scores in [1,10]
BS_i = mean_j≠i B_ij
F_G  = (V̄, V_std, V_min, B̄, λ₂, σ_min)
```

## Summary

| Metric | Value |
|--------|------:|
| Series recomputed OK | 91 |
| In-range (C/K/S/A ∈ [1,10]) | 68 |
| Out-of-range (DEV / legacy Stress) | 23 |
| Primary ENS corpus (cross-society) | 34 |
| Series failed | 0 |
| Society-years (all) | 9058 |
| Society-years (in-range) | 6778 |
| Output directory | `data/v1.0_final_recompute/series/` |
| F_G corpus | `data/v1.0_final_recompute/FG_corpus.csv` |

### Primary ENS series (cross-society comparison corpus)

| Society | Years | Span | BS min | BS max | Source |
|---------|------:|------|-------:|-------:|--------|
| Argentina | 77 | 1950–2026 | 0.0551 | 0.3338 | `Argentina_ENS.csv` |
| Australia | 152 | 1875–2026 | 0.0845 | 0.4906 | `Australia_ENS.csv` |
| Brazil | 146 | 1880–2025 | 0.0997 | 0.4685 | `Brazil_ENS.csv` |
| Canada | 33 | 1850–2026 | 0.1075 | 0.5324 | `Canada_ENS.csv` |
| Chile | 152 | 1875–2026 | 0.1084 | 0.4652 | `Chile_ENS.csv` |
| China | 226 | 1800–2025 | 0.0723 | 0.4426 | `China_ENS.csv` |
| Colombia | 152 | 1875–2026 | 0.0875 | 0.2806 | `Colombia_ENS.csv` |
| France | 177 | 1850–2026 | 0.0545 | 0.5220 | `France_ENS.csv` |
| Germany | 147 | 1880–2026 | 0.0477 | 0.5057 | `Germany_ENS.csv` |
| Hong Kong | 116 | 1900–2015 | 0.0522 | 0.4261 | `Hong_Kong_ENS.csv` |
| India | 31 | 1875–2026 | 0.0738 | 0.3584 | `India_ENS.csv` |
| Indonesia | 85 | 1941–2025 | 0.0343 | 0.2914 | `Indonesia_ENS.csv` |
| Iran | 152 | 1875–2026 | 0.0741 | 0.3960 | `Iran_ENS.csv` |
| Iraq | 118 | 1900–2025 | 0.0650 | 0.2847 | `Iraq_ENS.csv` |
| Israel | 80 | 1946–2025 | 0.1594 | 0.4241 | `Israel_ENS.csv` |
| Italy | 124 | 1900–2024 | 0.0361 | 0.3973 | `Italy_ENS.csv` |
| Japan | 152 | 1875–2026 | 0.0468 | 0.4715 | `Japan_ENS.csv` |
| Latium Vetus | 79 | 460–2010 | 0.0665 | 0.4361 | `LatimVetus_ENS_460_2010_cleaned.csv` |
| Lebanon | 83 | 1943–2025 | 0.0681 | 0.3704 | `Lebanon_ENS.csv` |
| Norway | 147 | 1880–2026 | 0.0634 | 0.5617 | `Norway_ENS.csv` |
| Pakistan | 78 | 1947–2025 | 0.0755 | 0.3867 | `Pakistan_ENS.csv` |
| Poland | 152 | 1875–2026 | 0.0860 | 0.3426 | `Poland_ENS.csv` |
| Russia | 227 | 1800–2026 | 0.0702 | 0.4024 | `Russia_ENS.csv` |
| Saudi Arabia | 126 | 1900–2026 | 0.1611 | 0.4015 | `Saudi_Arabia_ENS.csv` |
| Singapore | 96 | 1930–2025 | 0.0577 | 0.6167 | `Singapore_ENS.csv` |
| South Africa | 145 | 1880–2025 | 0.0418 | 0.7609 | `South_Africa_ENS.csv` |
| Sweden | 145 | 1880–2025 | 0.1806 | 0.7372 | `Sweden_ENS.csv` |
| Syria | 115 | 1893–2024 | 0.0512 | 0.2780 | `Syria_ENS.csv` |
| Thailand | 36 | 1850–2025 | 0.1264 | 0.4000 | `Thailand_ENS.csv` |
| Türkiye | 31 | 1875–2026 | 0.0767 | 0.3498 | `Turkiye_ENS.csv` |
| UAE | 51 | 1970–2026 | 0.2198 | 0.7029 | `UAE_ENS.csv` |
| United Kingdom | 147 | 1880–2026 | 0.1594 | 0.5087 | `UK_ENS.csv` |
| United States | 26 | 1900–2026 | 0.1119 | 0.4900 | `USA_ENS.csv` |
| Venezuela | 56 | 1970–2025 | 0.0443 | 0.5381 | `Venezuela_ENS.csv` |

### All recomputed series

| Society | Tier | Range | Years | Span | BS min | BS max | Source |
|---------|------|-------|------:|------|-------:|-------:|--------|
| Afghanistan | usp_or_legacy | in_range | 126 | 1900–2025 | 0.0468 | 0.3015 | `Afghanistan_cleaned.csv` |
| Argentina | usp_or_legacy | in_range | 77 | 1950–2026 | 0.0551 | 0.3338 | `Argentina_CAMS5_calc_1950_2026_cleaned.csv` |
| Argentina | usp_or_legacy | in_range | 77 | 1950–2026 | 0.0551 | 0.3338 | `Argentina_CAMS5_ensemble_1950_2026_cleaned.csv` |
| Argentina | usp_or_legacy | in_range | 77 | 1950–2026 | 0.0551 | 0.3338 | `Argentina_cam5_1950_2026_cleaned.csv` |
| Argentina | canonical_ens | in_range | 77 | 1950–2026 | 0.0551 | 0.3338 | `Argentina_ENS.csv` |
| Australia | usp_or_legacy | in_range | 131 | 1895–2026 | 0.1423 | 0.5747 | `Australia_2026_cleaned.csv` |
| Australia | usp_or_legacy | in_range | 31 | 1996–2026 | 0.1873 | 0.4948 | `Australia_cam5_1996_2026_cleaned.csv` |
| Australia | usp_or_legacy | out_of_range | 123 | 1900–2024 | 0.2820 | 1.0633 | `Australia_cleaned.csv` |
| Australia | canonical_ens | in_range | 152 | 1875–2026 | 0.0845 | 0.4906 | `Australia_ENS.csv` |
| Brazil | canonical_ens | in_range | 146 | 1880–2025 | 0.0997 | 0.4685 | `Brazil_ENS.csv` |
| Canada | usp_or_legacy | in_range | 6 | 1900–1905 | 0.2271 | 0.3355 | `Canada_cleaned.csv` |
| Canada | canonical_ens | in_range | 33 | 1850–2026 | 0.1075 | 0.5324 | `Canada_ENS.csv` |
| Chile | canonical_ens | in_range | 152 | 1875–2026 | 0.1084 | 0.4652 | `Chile_ENS.csv` |
| China | usp_or_legacy | in_range | 19 | 1850–2026 | 0.0859 | 0.3972 | `China_CAMS5_ensemble_1850_2026_cleaned.csv` |
| China | usp_or_legacy | in_range | 127 | 1900–2026 | 0.0644 | 0.5024 | `China_CAMS_1900_2026_cleaned.csv` |
| China | canonical_ens | in_range | 226 | 1800–2025 | 0.0723 | 0.4426 | `China_ENS.csv` |
| Colombia | canonical_ens | in_range | 152 | 1875–2026 | 0.0875 | 0.2806 | `Colombia_ENS.csv` |
| Denmark | usp_or_legacy | out_of_range | 114 | 1752–2025 | 0.1360 | 1.1861 | `Denmark_cleaned.csv` |
| Denmark | canonical_ens | out_of_range | 114 | 1752–2025 | 0.1246 | 0.8946 | `Denmark_ENS.csv` |
| England | usp_or_legacy | in_range | 151 | 1750–1900 | 0.2242 | 0.5847 | `England_cleaned.csv` |
| France | usp_or_legacy | out_of_range | 49 | 1785–2024 | 0.1920 | 1.0367 | `France_1785_1800_cleaned.csv` |
| France | usp_or_legacy | in_range | 31 | 1800–1830 | 0.1111 | 0.4952 | `France_1800_1830_GEM_cleaned.csv` |
| France | usp_or_legacy | in_range | 31 | 1800–1830 | 0.0572 | 0.5040 | `France_1800_1830_claude_cleaned.csv` |
| France | usp_or_legacy | in_range | 127 | 1900–2026 | 0.0723 | 0.4609 | `France_1900_2026_cleaned.csv` |
| France | usp_or_legacy | out_of_range | 49 | 1785–2024 | 0.1920 | 1.0367 | `France_Master_3_France_1785_1790_1795_1800_cleaned.csv` |
| France | usp_or_legacy | out_of_range | 49 | 1785–2024 | 0.1920 | 1.0367 | `France_cleaned.csv` |
| France | canonical_ens | in_range | 177 | 1850–2026 | 0.0545 | 0.5220 | `France_ENS.csv` |
| Germany | usp_or_legacy | in_range | 86 | 1900–2026 | 0.0347 | 0.5419 | `Germany_CAMS_1900_2026_cleaned.csv` |
| Germany | usp_or_legacy | out_of_range | 274 | 1750–2025 | 0.0370 | 0.6871 | `Germany_cleaned.csv` |
| Germany | canonical_ens | in_range | 147 | 1880–2026 | 0.0477 | 0.5057 | `Germany_ENS.csv` |
| Hong Kong | usp_or_legacy | in_range | 116 | 1900–2015 | 0.0522 | 0.4261 | `Hong_Kong_cleaned.csv` |
| Hong Kong | usp_or_legacy | in_range | 116 | 1900–2015 | 0.0522 | 0.4261 | `Hongkong_Manual_cleaned.csv` |
| Hong Kong | canonical_ens | in_range | 116 | 1900–2015 | 0.0522 | 0.4261 | `Hong_Kong_ENS.csv` |
| India | usp_or_legacy | out_of_range | 73 | 1950–2024 | 0.5401 | 1.6833 | `India_cleaned.csv` |
| India | canonical_ens | in_range | 31 | 1875–2026 | 0.0738 | 0.3584 | `India_ENS.csv` |
| Indigenous Australia | usp_or_legacy | in_range | 18 | 1600–2025 | 0.1135 | 0.5142 | `IndigenousAustralia_cleaned.csv` |
| Indonesia | usp_or_legacy | in_range | 85 | 1941–2025 | 0.0343 | 0.2914 | `Indonesia_cleaned.csv` |
| Indonesia | canonical_ens | in_range | 85 | 1941–2025 | 0.0343 | 0.2914 | `Indonesia_ENS.csv` |
| Iran | usp_or_legacy | out_of_range | 115 | 1900–2025 | 0.0995 | 1.7068 | `Iran_cleaned.csv` |
| Iran | canonical_ens | in_range | 152 | 1875–2026 | 0.0741 | 0.3960 | `Iran_ENS.csv` |
| Iraq | usp_or_legacy | in_range | 124 | 1900–2025 | 0.0650 | 0.2847 | `Iraq_cleaned.csv` |
| Iraq | canonical_ens | in_range | 118 | 1900–2025 | 0.0650 | 0.2847 | `Iraq_ENS.csv` |
| Israel | canonical_ens | in_range | 80 | 1946–2025 | 0.1594 | 0.4241 | `Israel_ENS.csv` |
| Italy | usp_or_legacy | out_of_range | 124 | 1900–2024 | 0.0337 | 0.3973 | `Italy19002025_cleaned.csv` |
| Italy | usp_or_legacy | out_of_range | 124 | 1900–2024 | 0.0337 | 0.3973 | `Italy_cleaned.csv` |
| Italy | canonical_ens | in_range | 124 | 1900–2024 | 0.0361 | 0.3973 | `Italy_ENS.csv` |
| Japan | usp_or_legacy | in_range | 166 | 1850–2025 | 0.0984 | 0.6947 | `Japan_cleaned.csv` |
| Japan | canonical_ens | in_range | 152 | 1875–2026 | 0.0468 | 0.4715 | `Japan_ENS.csv` |
| Latium Vetus | cleaned_ens | in_range | 79 | 460–2010 | 0.0665 | 0.4361 | `LatimVetus_ENS_460_2010_cleaned.csv` |
| Lebanon | usp_or_legacy | in_range | 83 | 1943–2025 | 0.0681 | 0.3704 | `Lebanon_cleaned.csv` |
| Lebanon | canonical_ens | in_range | 83 | 1943–2025 | 0.0681 | 0.3704 | `Lebanon_ENS.csv` |
| Netherlands | usp_or_legacy | out_of_range | 54 | 1750–2024 | 0.4526 | 1.0406 | `Netherlands_cleaned.csv` |
| Netherlands | canonical_ens | out_of_range | 54 | 1750–2024 | 0.3965 | 0.8528 | `Netherlands_ENS.csv` |
| New Zealand | usp_or_legacy | in_range | 127 | 1900–2026 | 0.1318 | 0.5320 | `NewZealand_Gem_April_cleaned.csv` |
| New Zealand | usp_or_legacy | in_range | 127 | 1900–2026 | 0.1279 | 0.4763 | `New_Zealand_1900_2026_claude_cleaned.csv` |
| Norway | canonical_ens | in_range | 147 | 1880–2026 | 0.0634 | 0.5617 | `Norway_ENS.csv` |
| Pakistan | usp_or_legacy | in_range | 78 | 1947–2025 | 0.0755 | 0.3867 | `Pakistan_cleaned.csv` |
| Pakistan | canonical_ens | in_range | 78 | 1947–2025 | 0.0755 | 0.3867 | `Pakistan_ENS.csv` |
| Poland | canonical_ens | in_range | 152 | 1875–2026 | 0.0860 | 0.3426 | `Poland_ENS.csv` |
| Roman Empire | usp_or_legacy | out_of_range | 83 | 5–425 | 0.0358 | 1.2638 | `New_Rome_Ad_5Y_Rome_0_Bce_5Ad_10Ad_15Ad_20_Ad_cleaned.csv` |
| Rome | usp_or_legacy | in_range | 45 | 10–450 | 0.0708 | 0.5371 | `Rome_CAMS_recalculated_cleaned.csv` |
| Russia | usp_or_legacy | in_range | 31 | 1800–1830 | 0.1361 | 0.4456 | `Russia_1800_1830_claude_cleaned.csv` |
| Russia | usp_or_legacy | in_range | 126 | 1900–2025 | 0.0451 | 0.3037 | `Russia_cleaned.csv` |
| Russia | canonical_ens | in_range | 227 | 1800–2026 | 0.0702 | 0.4024 | `Russia_ENS.csv` |
| Saudi Arabia | usp_or_legacy | out_of_range | 107 | 1918–2025 | 0.3556 | 2.6056 | `Saudi_Arabia_cleaned.csv` |
| Saudi Arabia | canonical_ens | in_range | 126 | 1900–2026 | 0.1611 | 0.4015 | `Saudi_Arabia_ENS.csv` |
| Singapore | usp_or_legacy | out_of_range | 90 | 1935–2025 | 0.0662 | 1.4522 | `Singapore_cleaned.csv` |
| Singapore | canonical_ens | in_range | 96 | 1930–2025 | 0.0577 | 0.6167 | `Singapore_ENS.csv` |
| South Africa | canonical_ens | in_range | 145 | 1880–2025 | 0.0418 | 0.7609 | `South_Africa_ENS.csv` |
| SpaceX | usp_or_legacy | in_range | 21 | 2006–2026 | 0.1529 | 0.4794 | `SpaceX_cam5_2006_2026_cleaned.csv` |
| Spain | usp_or_legacy | in_range | 31 | 1800–1830 | 0.0805 | 0.3784 | `Spain_1800_1830_cleaned.csv` |
| Sweden | canonical_ens | in_range | 145 | 1880–2025 | 0.1806 | 0.7372 | `Sweden_ENS.csv` |
| Syria | usp_or_legacy | in_range | 115 | 1893–2024 | 0.0512 | 0.2780 | `Syria_cleaned.csv` |
| Syria | canonical_ens | in_range | 115 | 1893–2024 | 0.0512 | 0.2780 | `Syria_ENS.csv` |
| Thailand | usp_or_legacy | in_range | 174 | 1850–2025 | 0.1320 | 0.3804 | `Thailand_1850_2025_Thailand_1850_2025_cleaned.csv` |
| Thailand | canonical_ens | in_range | 36 | 1850–2025 | 0.1264 | 0.4000 | `Thailand_ENS.csv` |
| Türkiye | canonical_ens | in_range | 31 | 1875–2026 | 0.0767 | 0.3498 | `Turkiye_ENS.csv` |
| UAE | canonical_ens | in_range | 51 | 1970–2026 | 0.2198 | 0.7029 | `UAE_ENS.csv` |
| USA | usp_or_legacy | in_range | 127 | 1900–2026 | 0.1171 | 0.4780 | `MARKER_USA_1900_2026_ENSEMBLE_MEAN_cleaned.csv` |
| USA | usp_or_legacy | out_of_range | 119 | 1790–2025 | 0.2845 | 1.5524 | `USA_HighRes_cleaned.csv` |
| USA | usp_or_legacy | out_of_range | 119 | 1790–2025 | 0.2845 | 1.5524 | `USA_Reconstructed_cleaned.csv` |
| USA | usp_or_legacy | in_range | 31 | 1996–2026 | 0.1233 | 0.4972 | `USA_cam5_1996_2026_cleaned.csv` |
| Ukraine | canonical_ens | out_of_range | 84 | 1931–2025 | 0.0000 | 0.4631 | `Ukraine_ENS.csv` |
| United Kingdom | canonical_ens | in_range | 147 | 1880–2026 | 0.1594 | 0.5087 | `UK_ENS.csv` |
| United States | canonical_ens | in_range | 26 | 1900–2026 | 0.1119 | 0.4900 | `USA_ENS.csv` |
| Venezuela | canonical_ens | in_range | 56 | 1970–2025 | 0.0443 | 0.5381 | `Venezuela_ENS.csv` |
| WorldCom | usp_or_legacy | out_of_range | 12 | 1990–2001 | 0.0000 | 0.4840 | `WorldCom_cleaned.csv` |
| israel | usp_or_legacy | in_range | 80 | 1946–2025 | 0.1594 | 0.4241 | `Israel_cleaned.csv` |
| usa | usp_or_legacy | out_of_range | 119 | 1790–2025 | 0.2845 | 1.5524 | `USA_cleaned.csv` |
| usa_master | usp_or_legacy | out_of_range | 112 | 1790–2023 | 0.4593 | 1.6520 | `USA_Master_cleaned.csv` |
| usa_maximum_1790-2025_us_high_res | usp_or_legacy | out_of_range | 119 | 1790–2025 | 0.2845 | 1.5524 | `Usa_Maximum_1790-2025_Us_High_Res_2025_(1)_cleaned.csv` |

## Regime distribution (in-range society-years only)

| Regime | Count | Share |
|--------|------:|------:|
| Stable adaptive | 2299 | 33.9% |
| Strained | 1781 | 26.3% |
| Local Node Failure | 1641 | 24.2% |
| Systemic crisis | 850 | 12.5% |
| Freeze/Collapse | 199 | 2.9% |
| Phantom Type II | 8 | 0.1% |

## Sanity checks (in-range)

- B̄ ∈ [0,1]: **PASS** (range [0.0437, 0.7220])
- λ₂ range: [0.2650, 5.3537]
- V̄ range: [-5.69, 22.38]
- σ_min range: [-2.756, 6.000]

## Notes

- Per-node Bond Strength is the **mean of the 7 off-diagonal** B_ij edges.
- Legacy Army/Executive/… labels mapped to the 8-node CAMS set; dual aliases (e.g. Trades/Prof. vs Trades/Professions) resolved by preference rank.
- Incomplete society-years (≠ 8 nodes after mapping) are dropped.
- ENV (variance) files are not recomputed.
- **Out-of-range** series (negative Stress legacy scale, C>10, etc.) are recomputed for archival completeness but **must not** be used for cross-society F_G threshold recalibration — B_ij can exceed 1 when S < 0.
- Primary corpus for cross-society work = `canonical_ens` + `cleaned_ens` with `range_tier=in_range`.
- Next critical-path step: F_G threshold recalibration on the primary corpus.
