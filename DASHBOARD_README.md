# CAMS GTS EV v2.0 Dashboard

**Thermodynamic Cognitive Analysis Platform**
Live multi-society thermodynamic analysis using the CAMS GTS EV v1.95 canonical engine.

## Overview

The CAMS GTS EV v2.0 Dashboard provides real-time thermodynamic analysis of civilizations through a powerful Streamlit interface. It processes historical and contemporary data to compute:

- **Dual-Mode Cognitive Load** (Ψ Deliberative vs Φ Reactive)
- **R Ratio** (Reactive/Deliberative balance)
- **Health Index H%** (System battery capacity)
- **Bond Strength ⟨B⟩** (Inter-institutional coupling)
- **Crisis Probability** (10-year risk assessment)
- **System Classification** (Resilient Frontier, Stable Core, Transitional, Fragile, Terminal)

## Quick Start

### 1. Installation

```bash
# Navigate to the wintermute directory
cd wintermute

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place your CSV files in the `cleaned_datasets/` directory (or create a `data/` directory).

**Expected CSV format:**
```csv
Society,Year,Node,Coherence,Capacity,Stress,Abstraction,Node_Value,Bond_Strength
USA,2025,Executive,5.2,6.1,-2.3,7.8,16.8,4.2
USA,2025,Army,4.8,5.9,-1.2,6.5,16.0,3.8
...
```

**Required nodes (8 total):**
1. Executive (Helm)
2. Army (Shield)
3. Priesthood / Knowledge Workers (Lore)
4. Property Owners (Stewards)
5. Trades/Professions (Craft)
6. Proletariat (Hands)
7. State Memory (Archive)
8. Merchants (Flow)

### 3. Launch Dashboard

```bash
streamlit run streamlit_app.py
```

Open your browser to **http://localhost:8501**

## Features

### 🎯 Single Society Analysis
- **Time-Series Visualization**: Track Ψ, Φ, R ratio, Health H%, and Bond Strength over decades
- **Bond Strength Heatmap**: Interactive 8×8 matrix showing institutional coupling
- **Real-Time Metrics**: Live deltas showing direction of change
- **Year Range Filtering**: Analyze any historical period (e.g., 1900-1950)

### 🌍 Multi-Society Comparison
- **Health-Normalized Trajectories**: Compare civilizations at different stages
- **Overlay Multiple Societies**: USA, Rome, China, Russia, UK, Ukraine, Australia
- **Peak Detection**: Automatically normalizes to historical maximum (100%)

### 📊 Data Export
- **CSV Download**: Full time-series results for publications
- **JSON Export**: Machine-readable format for further analysis
- **One-Row Canonical Output**: Condensed format for rapid comparison
- **Full Results Table**: Interactive dataframe with all computed metrics

### 🔧 Thermodynamic Controls
- **EROEI Adjustment**: Set Energy Return on Energy Invested (2.0 - 50.0)
- **Φ_export Parameter**: Model entropy offload (0.0 - 10.0 W·K⁻¹·cap⁻¹)
- **Population Override**: Manual population input (millions)

## Dashboard Tabs

### Tab 1: Single Society Analysis
- Dual-mode time series (4-panel visualization)
- Threshold indicators (crisis zones highlighted)
- Bond strength heatmap (latest year)
- Classification state display

### Tab 2: Multi-Society Comparison
- Health trajectory overlays
- Peak-normalized curves
- Historical pattern recognition
- Comparative crisis analysis

### Tab 3: Data Export
- CSV/JSON downloads
- Canonical one-row output
- Full results dataframe
- Timestamp-stamped exports

## Thermodynamic Engine (v1.95)

### Physical Constants
```python
SIGMA_0 = 1.0
EPS_REF = 1.2e3   # W/capita reference
EPS_CRIT = 12e3   # W/capita critical
KAPPA = 1e-14     # Abstraction coupling
OMEGA = 2.5       # Bond strength modifier
CHI = 0.08        # Entropy export coefficient
```

### Key Equations

**Node Value:**
```
NV = C + K - S + 0.5·A
```

**Bond Strength:**
```
B_ij = C_i · C_j · exp(-|K_i - K_j|) · (1 - Ω·D_KL)
```

**Deliberative Mode (Ψ):**
```
Ψ = ln((E_net - P_Abs) / E_star) · Σ(C_i · A_i)
```

**Reactive Mode (Φ):**
```
Φ = Σ(K_i · (11 - S_i)) - χ · φ_export · pop/1e9
```

**R Ratio (Collapse Indicator):**
```
R = Φ / Ψ
```

**Health Index:**
```
H% = 100 · Σ(C_i · K_i) / (max(C_i · K_i) · 8)
```

### Classification Thresholds

| R Ratio | H% | ⟨B⟩ | Classification |
|---------|-----|-----|----------------|
| < 1.0 | > 65 | > 2.0 | **Resilient Frontier** |
| < 1.0 | > 65 | any | **Stable Core** |
| 1.0–2.2 | 35–65 | any | **Transitional** |
| 2.2–4.5 | < 35 | any | **Fragile** |
| > 4.5 | any | any | **Terminal** |

## Example Use Cases

### Academic Research
```python
# Compare USA 1861 (Civil War) vs 2025
# Set year range: 1850-1870
# Observe R ratio spike, H% collapse
# Export CSV for publication
```

### Policy Analysis
```python
# Monitor Ukraine 2022-2025
# Track crisis probability evolution
# Compare with historical collapse patterns (Rome 470)
# Generate JSON for predictive models
```

### Historical Validation
```python
# Load Rome dataset
# Analyze 400-500 CE collapse
# Compare bond strength matrices
# Validate against known phase transitions
```

## Data Requirements

### Minimum Dataset
- **Temporal Coverage**: At least 10 years of data
- **Node Completeness**: All 8 nodes per year
- **Variable Range**: C, K ∈ [0,10]; S ∈ [-5,5]; A ∈ [0,10]

### Optional Fields
- `Pop_M`: Population in millions (defaults to 330 for USA, 26 others)
- `EROEI`: Society-specific energy return (defaults to 10.0)

## Troubleshooting

### "No data files found"
- Ensure CSV files are in `cleaned_datasets/` or `data/` directory
- Check file extension is `.csv`

### "Engine failed"
- Verify exactly 8 nodes per year
- Check column names: `Coherence`, `Capacity`, `Stress`, `Abstraction`

### "No data in selected range"
- Adjust year slider to match available data
- Check `Year` column contains integers

## Performance Notes

- **Loading Time**: ~2-3 seconds for 30,000+ records
- **Computation**: Real-time for 100+ year ranges
- **Export Speed**: <1 second for typical datasets
- **Browser Compatibility**: Chrome, Firefox, Edge (latest versions)

## Files

```
wintermute/
├── streamlit_app.py          # Main dashboard application
├── cams_engine.py            # Canonical v1.95 thermodynamic engine
├── cleaned_datasets/         # Primary data directory
├── exports/                  # Auto-generated download folder
└── requirements.txt          # Python dependencies
```

## Citation

```bibtex
@software{cams_gts_ev_2025,
  title = {CAMS GTS EV v2.0 Dashboard},
  author = {McKern, Kari},
  year = {2025},
  month = {12},
  url = {https://neuralnations.org},
  version = {2.0},
  note = {Thermodynamic Cognitive Dual-Mode Theory}
}
```

## Support

**Website**: [neuralnations.org](https://neuralnations.org)
**Email**: kari.freyr.4@gmail.com
**Framework Version**: GTS EV v1.95 / Dashboard v2.0
**Last Updated**: December 2025

## Changelog

### v2.0 (Dec 2025)
- ✅ Multi-society comparison tab
- ✅ Year range filtering (vs single year)
- ✅ Bond strength heatmap visualization
- ✅ CSV/JSON export functionality
- ✅ Health-normalized trajectory comparison
- ✅ Real-time delta metrics
- ✅ Auto-detection of data directories

### v1.95 (Nov 2025)
- Initial canonical engine implementation
- Single society analysis
- Dual-mode Ψ/Φ calculations
- Classification system

---

**Built with Streamlit • Powered by Thermodynamic Physics • December 2025**
