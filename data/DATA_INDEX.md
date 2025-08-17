# Wintermute Data Directory Index

**Complex Adaptive Management Systems - Catch-All Network v3.4**
**Data Repository Organization**

**Generated:** 2025-08-11 14:48:34
**Total Datasets:** 32
**Total Records:** 30,856

## Directory Structure

```
data/
├── cleaned/          # Validated CAMS-CAN v3.4 datasets (32 files)
├── raw/              # Original unprocessed CSV files
├── processed/        # Intermediate processing files
├── backup/           # Backup and archive files
└── input/            # Data input staging area
```

## Geographic Coverage

### Asia-Pacific
**Countries:** 9 | **Datasets:** Australia, Hongkong Manual, Hong Kong, India, Indonesia, Japan, Pakistan, Singapore, Thailand 1850 2025 Thailand 1850 2025

### North America
**Countries:** 6 | **Datasets:** Canada, USA, USA HighRes, USA Master, Usa Maximum 1790-2025 Us High Res 2025 (1), USA Reconstructed

### Europe
**Countries:** 9 | **Datasets:** Denmark, England, France 1785 1800, France, France Master 3 France 1785 1790 1795 1800, Germany, Italy19002025, Italy, Netherlands

### Middle East
**Countries:** 6 | **Datasets:** Iran, Iraq, Israel, Lebanon, Saudi Arabia, Syria

### Historical
**Countries:** 1 | **Datasets:** New Rome Ad 5Y Rome 0 Bce 5Ad 10Ad 15Ad 20 Ad

### Other
**Countries:** 1 | **Datasets:** Russia

## Dataset Details

| Country | Records | Time Span | Institutions | File |
|---------|---------|-----------|--------------|------|
| Australia | 984 | 1900-2024 | 8 | `Australia_cleaned.csv` |
| Canada | 48 | 1900-1905 | 8 | `Canada_cleaned.csv` |
| Denmark | 920 | 1752-2025 | 8 | `Denmark_cleaned.csv` |
| England | 1,224 | 1750.0-1900.0 | 8 | `England_cleaned.csv` |
| France | 400 | 1785-2024 | 8 | `France_cleaned.csv` |
| France 1785 1800 | 400 | 1785-2024 | 8 | `France_1785_1800_cleaned.csv` |
| France Master 3 France 1785 1790 1795 1800 | 400 | 1785-2024 | 8 | `France_Master_3_France_1785_1790_1795_1800_cleaned.csv` |
| Germany | 2,199 | 1750.0-2025.0 | 11 | `Germany_cleaned.csv` |
| Hong Kong | 952 | 1900-2015 | 8 | `Hong_Kong_cleaned.csv` |
| Hongkong Manual | 952 | 1900-2015 | 8 | `Hongkong_Manual_cleaned.csv` |
| India | 592 | 1950-2024 | 8 | `India_cleaned.csv` |
| Indonesia | 680 | 1941-2025 | 8 | `Indonesia_cleaned.csv` |
| Iran | 920 | 1900-2025 | 8 | `Iran_cleaned.csv` |
| Iraq | 1,202 | 3-2025 | 13 | `Iraq_cleaned.csv` |
| Israel | 664 | 1946.0-2025.0 | 8 | `Israel_cleaned.csv` |
| Italy | 1,080 | 1900-2024 | 8 | `Italy_cleaned.csv` |
| Italy19002025 | 1,080 | 1900.0-2024.0 | 8 | `Italy19002025_cleaned.csv` |
| Japan | 1,632 | 1850.0-2025.0 | 8 | `Japan_cleaned.csv` |
| Lebanon | 664 | 1943-2025 | 8 | `Lebanon_cleaned.csv` |
| Netherlands | 440 | 1750-2024 | 8 | `Netherlands_cleaned.csv` |
| New Rome Ad 5Y Rome 0 Bce 5Ad 10Ad 15Ad 20 Ad | 672 | 5.0-425.0 | 8 | `New_Rome_Ad_5Y_Rome_0_Bce_5Ad_10Ad_15Ad_20_Ad_cleaned.csv` |
| Pakistan | 642 | 1947.0-2025.0 | 8 | `Pakistan_cleaned.csv` |
| Russia | 1,208 | 1900.0-2025.0 | 8 | `Russia_cleaned.csv` |
| Saudi Arabia | 856 | 1918-2025 | 8 | `Saudi_Arabia_cleaned.csv` |
| Singapore | 744 | 1935-2025 | 8 | `Singapore_cleaned.csv` |
| Syria | 943 | 1893-2025 | 8 | `Syria_cleaned.csv` |
| Thailand 1850 2025 Thailand 1850 2025 | 1,486 | 1850-2025 | 8 | `Thailand_1850_2025_Thailand_1850_2025_cleaned.csv` |
| USA | 1,248 | 1790-2025 | 10 | `USA_cleaned.csv` |
| USA HighRes | 2,168 | 1790.0-2025.0 | 10 | `USA_HighRes_cleaned.csv` |
| USA Master | 960 | 1790-2023 | 10 | `USA_Master_cleaned.csv` |
| USA Reconstructed | 1,248 | 1790-2025 | 10 | `USA_Reconstructed_cleaned.csv` |
| Usa Maximum 1790-2025 Us High Res 2025 (1) | 1,248 | 1790-2025 | 10 | `Usa_Maximum_1790-2025_Us_High_Res_2025_(1)_cleaned.csv` |

## Organization Changes Made

- Copied: Australia_cleaned.csv -> data/cleaned/
- Copied: Canada_cleaned.csv -> data/cleaned/
- Copied: Denmark_cleaned.csv -> data/cleaned/
- Copied: England_cleaned.csv -> data/cleaned/
- Copied: France_1785_1800_cleaned.csv -> data/cleaned/
- Copied: France_cleaned.csv -> data/cleaned/
- Copied: France_Master_3_France_1785_1790_1795_1800_cleaned.csv -> data/cleaned/
- Copied: Germany_cleaned.csv -> data/cleaned/
- Copied: Hongkong_Manual_cleaned.csv -> data/cleaned/
- Copied: Hong_Kong_cleaned.csv -> data/cleaned/
- Copied: India_cleaned.csv -> data/cleaned/
- Copied: Indonesia_cleaned.csv -> data/cleaned/
- Copied: Iran_cleaned.csv -> data/cleaned/
- Copied: Iraq_cleaned.csv -> data/cleaned/
- Copied: Israel_cleaned.csv -> data/cleaned/
- Copied: Italy19002025_cleaned.csv -> data/cleaned/
- Copied: Italy_cleaned.csv -> data/cleaned/
- Copied: Japan_cleaned.csv -> data/cleaned/
- Copied: Lebanon_cleaned.csv -> data/cleaned/
- Copied: Netherlands_cleaned.csv -> data/cleaned/
- ... and 54 more files organized

## Usage Instructions

### Primary Analysis Interface
```bash
streamlit run cams_can_v34_explorer.py
```

### Data Access
- **Validated Data:** `data/cleaned/` - Ready for analysis
- **Raw Data:** `data/raw/` - Original files for reference
- **Backup Data:** `data/backup/` - Archive copies

## Data Standards

### CAMS-CAN v3.4 Format
```csv
Nation,Year,Node,Coherence,Capacity,Stress,Abstraction,Node value,Bond strength
```

### Standard Institutional Nodes
1. Executive
2. Army
3. Priesthood
4. Property
5. Trades
6. Proletariat
7. StateMemory
8. Merchants

## Quality Metrics

- **Validation Status:** All cleaned datasets CAMS-CAN v3.4 compatible
- **Time Coverage:** 3 CE - 2025 CE (2,022 years)
- **Geographic Scope:** 6 regions, 32 countries/civilizations
- **Data Integrity:** Validated numeric ranges, standardized node names

---

*This index is automatically maintained. For technical details, see CAMS_INDEX.md*