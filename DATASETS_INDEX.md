# DATASETS_INDEX.md
## Neural Nations — CAMS Dataset Registry
**32+ societies · 30,856 records · GPT + Grok + Gemini ensemble scoring**

All datasets live in [`data/cleaned/`](./data/cleaned/). Schema: `Society, Year, Node, Coherence, Capacity, Stress, Abstraction, Node Value, Bond Strength`

See [DATASET_VALIDATION_SUMMARY.md](./DATASET_VALIDATION_SUMMARY.md) for scoring methodology and inter-rater agreement stats.

---

### North America

| File | Society | Years | Records | Notes |
|------|---------|-------|---------|-------|
| `USA_Master_cleaned.csv` | United States (master) | 1790–2025 | 960 | Primary series; includes 1861 Civil War benchmark |
| `USA_HighRes_cleaned.csv` | United States (high-res) | 1790–2025 | 2,168 | Maximum temporal resolution |
| `USA_Reconstructed_cleaned.csv` | United States (reconstructed) | 1790–2025 | 1,248 | Alternative scoring run for cross-validation |
| `Usa_Maximum_1790-2025_Us_High_Res_2025_(1)_cleaned.csv` | United States (maximum) | 1790–2025 | 1,248 | Extended high-res variant |
| `USA_cleaned.csv` | United States (standard) | — | 1,248 | Standard cleaned series |
| `usa2010_2025.csv` | United States (2010–2025) | 2010–2025 | 36 | Recent window only |
| `US_gem_jan8.csv` | United States (Gemini scorer) | 1970–2025 | 349 | Raw Gemini scoring run |
| `us_grok_jan8.csv` | United States (Grok scorer) | 1970–2025 | 458 | Raw Grok scoring run |
| `Canada_cleaned.csv` | Canada | 1900–1905 | 48 | Short series |
| `brazil_grok_jan.csv` | Brazil | 1880–2025 | 1,176 | South America |

---

### Europe

| File | Society | Years | Records | Notes |
|------|---------|-------|---------|-------|
| `England_cleaned.csv` | England | 1750–1900 | 1,224 | Pre-UK series |
| `France_cleaned.csv` | France | 1785–2024 | 400 | Standard series |
| `France_1785_1800_cleaned.csv` | France 1785–1800 | 1785–2024 | 400 | French Revolution benchmark |
| `France_Master_3_France_1785_1790_1795_1800_cleaned.csv` | France (master) | 1785–2024 | 400 | 5-year interval master |
| `Germany_cleaned.csv` | Germany | 1750–2025 | 2,199 | Full series |
| `Germany_gem_jan_2025.csv` | Germany (Gemini scorer) | 1880–2025 | 1,295 | Raw Gemini run |
| `Germany_grok_jan21.csv` | Germany (Grok scorer) | 1880–2025 | 1,170 | Raw Grok run |
| `germany1750 2025.csv` | Germany (extended) | 1750–2025 | 2,211 | Full extended series |
| `Italy_cleaned.csv` | Italy | 1900–2024 | 1,080 | Standard series |
| `Italy19002025_cleaned.csv` | Italy 1900–2025 | 1900–2024 | 1,080 | High-res modern series |
| `Netherlands_cleaned.csv` | Netherlands | 1750–2024 | 440 | |
| `Denmark_cleaned.csv` | Denmark | 1752–2025 | 920 | |
| `Finland_Gem_21Jan.csv` | Finland (Gemini) | 1900–2025 | 920 | Raw Gemini run |
| `Finland_Grok_jan21.csv` | Finland (Grok) | 1900–2025 | 984 | Raw Grok run |
| `norway_gem_jan.csv` | Norway | 1881–2025 | 1,160 | |
| `sweden_gem_jan26.csv` | Sweden | 1880–2025 | 1,160 | |
| `Russia_cleaned.csv` | Russia | 1900–2025 | 1,208 | |
| `Ukraine.csv` | Ukraine | 1980–2025 | 368 | Post-Soviet series |
| `Ukraine_gem_1930_Jan26.csv` | Ukraine (extended) | 1930–2025 | 687 | Extended back to 1930 |

---

### Asia-Pacific

| File | Society | Years | Records | Notes |
|------|---------|-------|---------|-------|
| `Japan_cleaned.csv` | Japan | 1850–2025 | 1,632 | |
| `Thailand_1850_2025_Thailand_1850_2025_cleaned.csv` | Thailand | 1850–2025 | 1,486 | |
| `Australia_cleaned.csv` | Australia | 1900–2024 | 984 | |
| `Australian_Gem_Jan30.csv` | Australia (Gemini) | 1900–2025 | 1,008 | Raw Gemini run |
| `India_cleaned.csv` | India | 1950–2024 | 592 | Post-independence |
| `Indonesia_cleaned.csv` | Indonesia | 1941–2025 | 680 | |
| `Singapore_cleaned.csv` | Singapore | 1935–2025 | 744 | |
| `Hong_Kong_cleaned.csv` | Hong Kong | 1900–2015 | 952 | |
| `Hongkong_Manual_cleaned.csv` | Hong Kong (manual) | 1900–2015 | 952 | Manual scoring variant |
| `Pakistan_cleaned.csv` | Pakistan | 1947–2025 | 642 | Post-independence |

---

### Middle East

| File | Society | Years | Records | Notes |
|------|---------|-------|---------|-------|
| `Iran_cleaned.csv` | Iran | 1900–2025 | 920 | Standard series |
| `Iran_gem_jan_13.csv` | Iran (Gemini) | 1900–2025 | 1,008 | Raw Gemini run |
| `iran_grok_feb_cleaned.csv.csv` | Iran (Grok) | 1900–2025 | 1,008 | Raw Grok run |
| `Iraq_cleaned.csv` | Iraq | ~1900–2025 | 1,202 | |
| `Israel_cleaned.csv` | Israel | 1946–2025 | 664 | Post-statehood |
| `Lebanon_cleaned.csv` | Lebanon | 1943–2025 | 664 | Post-independence |
| `Saudi_Arabia_cleaned.csv` | Saudi Arabia | 1918–2025 | 856 | |
| `Syria_cleaned.csv` | Syria | 1893–2025 | 943 | |
| `uae_gem_marc.csv` | UAE | 1970–2026 | 408 | |

---

### Historical

| File | Society | Years | Records | Notes |
|------|---------|-------|---------|-------|
| `New_Rome_Ad_5Y_Rome_0_Bce_5Ad_10Ad_15Ad_20_Ad_cleaned.csv` | Rome (terminal) | 5–425 AD | 672 | 5-year resolution; collapse benchmark |

---

### Organisations

| File | Organisation | Years | Records | Notes |
|------|-------------|-------|---------|-------|
| `Qantas_grok_91.csv` | Qantas | 1960–2025 | 547 | Corporate CAMS |
| `markerq_grok_91.csv` | Qantas (marker) | 1960–2025 | 547 | Validation marker |
| `Nexperia_jan_26.csv` | Nexperia | 2010–2025 | 132 | Corporate CAMS |
| `SydneyCC_GEM_jan.csv` | Sydney City Council | 1990–2025 | 292 | Government/institutional |
| `markercnxax_jan_26.csv` | Marker (Nexperia-type) | 2010–2025 | 128 | Validation marker |

---

### Methodological / Validation Markers

| File | Years | Records | Notes |
|------|-------|---------|-------|
| `Markerixnx_grok_feb_cleaned_society_first.csv` | 1900–2025 | 1,008 | Inter-rater validation |
| `Markerxaxa_grok_feb.csv` | 1900–2025 | 944 | Inter-rater validation |
| `Markerxaxu_Gem_Jan30_CLEAN.csv` | 1900–2025 | 896 | Inter-rater validation |
| `Markerxixn_gem_jan_13.csv` | 1900–2025 | 1,008 | Inter-rater validation |

---

**Total records across primary datasets: 30,856+**

For full validation methodology see [`CAMS_Validation_Formulation.md`](./CAMS_Validation_Formulation.md) and [`DATASET_VALIDATION_SUMMARY.md`](./DATASET_VALIDATION_SUMMARY.md).

*Open Science — Common Property — KaliBond/wintermute*
