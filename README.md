# CAMS Framework: Complex Adaptive Model of Societies
## Version 2.3 — Formal Specification with Coordination Phase Space

![CAMS Logo](https://img.shields.io/badge/CAMS-v2.3-blue) ![License](https://img.shields.io/badge/License-Open%20Science-green) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

**Purpose**: CAMS represents a society as a dynamic 8×4 state matrix, bridging the mythic layer (Lore, Archive) with executive interfaces (Helm, Stewards) through to material foundations (Shield, Craft, Hands, Flow). Effective coupling coordination between these three ontological layers defines societal health and predicts civilisational resilience or collapse.

## 🌡️ Core Architecture

### **State Representation**

Society S at time t is the state matrix **X**(t) ∈ ℝ^(8×4), where rows are 8 nodes and columns are [C, K, S, A] ∈ [1,10] integers.

### **Three Ontological Layers**

| Layer | Nodes | Function |
|-------|-------|----------|
| **Mythic** | Lore (3), Archive (4) | Cognitive coherence & legitimacy |
| **Interface** | Helm (1), Stewards (5) | Executive conduit |
| **Material** | Shield (2), Craft (6), Hands (7), Flow (8) | Physical/metabolic base |

### **Node Value & Health**

```
V_i(t) = C_i + K_i + (A_i / 2) - S_i    range: [-7.5, 24.0]

V̄(t)  = (1/8) Σ V_i(t)
σ_V(t) = sqrt( (1/8) Σ (V_i - V̄)² )
```

### **Bond Strength (negative-domain safe)**

```
B_ij(t) = sqrt( max(V_i+8, 0) · max(V_j+8, 0) ) / 32    ∈ [0, 1]
```

Canonical directed graph with Helm as universal hub and full Mythic ↔ Interface ↔ Material bridges.

### **Mythic–Material Coupling Index** (primary leading indicator)

```
Λ(t) = mean B_ij over all cross-layer edges
```

### **Coordination Phase Space & CPT**

Coordination state: Φ(t) = (V̄(t), σ_V(t))

| Regime | Condition |
|--------|-----------|
| Coherent-Capable | V̄ ≥ V_θ, σ_V ≤ σ_θ |
| Strained-Coherent | V̄ ≥ V_θ, σ_V > σ_θ |
| Polarised-Capable | V̄ < V_θ, σ_V ≤ σ_θ |
| **Crisis** | V̄ < V_θ, σ_V > σ_θ |

*(V_θ ≈ 12, σ_θ ≈ 3.5)*

**Coordination Phase Transition (CPT)** at t* when Φ enters Crisis with dσ_V/dt > 0, dV̄/dt < 0, and Λ(t*) < 0.45.

### **Dynamics**

```
V_i(t+1) = V_i(t) + α Σ_j B_ij(V_j - V_i) + ε_i(t+1)
```

Graph Laplacian diffusion plus shocks. Order parameters: ξ₁ = V̄, ξ₂ = σ_V, ξ₃ = Λ.

## ⚠️ Failure Modes

Systemic failure = layer decoupling → CPT, signalled by falling Λ(t). Societies do not fail by single-node collapse but by severed bonds between Mythic, Interface, and Material layers.

**Macro-Coupling Failures (low Λ)**
- **Chaotic Fragmentation**: Material layer runs without Mythic coherence
- **Regime Rigidity**: Mythic layer captures Helm and freezes Material

**Node-Specific Taxonomy**
- **Helm Isolation**: Executive fragmentation or capture
- **Mythic Decoupling**: Narrative–material gap
- **Flow Collapse**: Supply/currency failure cascading to Hands/Craft/Shield
- **Late Abstraction Collapse**: A_i drops after prolonged S_i rise
- **Shield Inversion**: Praetorian turn — coercion dominates collapsing Helm/Lore
- **Archive Amnesia**: Memory collapse → policy incoherence and narrative drift

## 🔭 Universality & Falsification

All societies obey identical (V̄, σ_V, Λ) geometry. Crisis = mythic–material decoupling, detectable ≥5 years pre-CPT.

Falsification protocol: cross-LLM r > 0.7, Λ(t−5) AUC > 0.75, universal ρ(s, k) < −0.3.

## 🚀 Interactive Dashboard

Live at **[cams-advanced-analysis.streamlit.app](https://cams-advanced-analysis.streamlit.app/)** — four specialised analysis tools:

| Tab | Tool | Purpose |
|-----|------|---------|
| 📊 Tab 1 | **dDIG Analysis** | Directed Information Gain — measures institutional influence via conditional mutual information I(X→Y\|Z) |
| 🌀 Tab 2 | **Dyad Field Analysis** | Tracks M (metabolic load), Y (mythic integration), D (mismatch), R (risk), Ω (stress volatility) |
| 📈 Tab 3 | **Combined Insights** | Unified cognitive + affective influence rankings, heatmaps, radar charts |
| 🌌 Tab 4 | **Phase-Space Attractor** | 3D trajectory in M-Y-B space, regime detection, phase velocity, density projections |

**Data input**: Upload CSV or select from 50+ pre-loaded societies.

**Submit this framework to any AI for thesis testing**: [Download one-shot test file](cams-one-shot-test.txt)

## 📊 Analysis Capabilities

### **System Classification**
- **Optimisation Engine**: Ψ≥0.60, low stress, high abstraction
- **Steady Climber**: Ψ≥0.60, low stress, moderate abstraction  
- **Phoenix Transformer**: High asymmetry, rebuilding phase
- **Resilient Innovator**: Cycling through moderate stress
- **Stable Core / Fragile High-Stress**: Based on stress thresholds

### **Early Warning System**
- Multi-threshold detection with validated bands
- Phase transition identification
- Coherence entropy monitoring
- Stress variance analysis
- Network synchronization tracking

## 🗂️ Repository Structure

```
wintermute/
├── 📁 data/
│   ├── cleaned/          # 32+ validated CAMS datasets (30,856 records)
│   ├── raw/              # Original unprocessed files
│   ├── processed/        # Intermediate analysis outputs
│   └── backup/           # Archive and backup files
├── 📄 cams_framework_v2_1.py     # Production framework (v2.1)
├── 📄 cams_can_v34_explorer.py   # Interactive dashboard
├── 📄 organize_data_directory.py # Data organization toolkit
└── 📊 Dashboard & Analysis Tools
```

## 🚀 Quick Start

### **1. Run Analysis Dashboard**
```bash
streamlit run cams_can_v34_explorer.py --server.port 8501
```
Access at: `http://localhost:8501`

### **2. Analyze Single Dataset**
```python
from cams_framework_v2_1 import CAMSNetwork, CAMSDataProcessor

# Load data
processor = CAMSDataProcessor()
df = processor.load_society_data("data/cleaned/USA_cleaned.csv")

# Create network
network = processor.create_network_from_year(df, 2025)

# Get metrics
psi = network.compute_psi_metric()
health = network.compute_system_health()
classification = network.classify_system_type()

print(f"Ψ: {psi:.3f}, Health: {health:.3f}, Type: {classification}")
```

### **3. Batch Analysis**
```python
from cams_framework_v2_1 import run_cams_analysis

data_files = ["data/cleaned/USA_cleaned.csv", "data/cleaned/China_cleaned.csv"]
results = run_cams_analysis(data_files)
```

## 📈 Datasets & Coverage

### **Geographic Coverage** (32 societies, 6 regions)
- **Americas**: USA (multiple high-res), Canada, Australia
- **Europe**: UK, France, Germany, Denmark, Netherlands, Italy, Russia  
- **Middle East**: Iran, Iraq, Israel, Lebanon, Saudi Arabia, Syria
- **Asia**: China, Japan, India, Indonesia, Pakistan, Singapore, Thailand
- **Historical**: Ancient Rome, Hong Kong

### **Temporal Range**
- **Ancient**: Rome (0 BCE - 20 CE)
- **Historical**: Various societies from 1750+
- **Contemporary**: High-resolution modern analysis (1900-2025)

## 🛠️ Advanced Features

### **Validation & Testing**
- **USA 1861 Benchmark**: All node values validated to exact specification
- **Mathematical Consistency**: Normalized end-to-end calculations
- **Production Safety**: Robust error handling and edge case management

## 📋 Requirements

```bash
pip install numpy pandas streamlit plotly scipy scikit-learn networkx matplotlib
```

## 🎯 Applications

### **Academic Research**
- Comparative civilizational analysis
- Historical transition studies  
- Institutional resilience modeling
- Social complexity quantification

### **Policy Analysis**
- Early warning system development
- Institutional health monitoring
- Cross-national comparisons
- Crisis prediction and prevention

### **Corporate Analysis**
- Organizational stress assessment
- Leadership network dynamics
- Change management optimization
- Resilience planning

## 📖 Documentation

- **[CAMS_INDEX.md](CAMS_INDEX.md)**: Complete framework overview
- **[DATA_INDEX.md](data/DATA_INDEX.md)**: Dataset documentation
- **[CAMS_Validation_Formulation.md](CAMS_Validation_Formulation.md)**: Mathematical specifications
- **[Advanced_CAMS_Integration_Summary.md](Advanced_CAMS_Integration_Summary.md)**: Integration guide

## 🤝 Contributing

We welcome contributions to the CAMS Framework:

1. **Data Contributions**: New society datasets in CAMS format
2. **Algorithm Improvements**: Enhanced metrics and analysis methods  
3. **Visualization Tools**: New dashboard components and charts
4. **Validation Studies**: Cross-validation with historical events

### **Data Format**
```csv
Society,Year,Node,Coherence,Capacity,Stress,Abstraction
USA,2025,Executive,5.2,6.1,-2.3,7.8
```

## 📧 Contact & Collaboration

**Lead Researcher**: Kari McKern
**Email**: [kari.freyr.4@gmail.com](mailto:kari.freyr.4@gmail.com)
**Framework Version**: 2.3 (February 2026)
**License**: Open Science - Common Property
**Co-authored with**: Claude Sonnet 4.5 (Anthropic)

---

*CAMS v2.3 provides a formally specified, falsifiable model of societal dynamics. Societies do not fail by single-node collapse but by severed bonds between Mythic, Interface, and Material layers — a universal geometry detectable years before crisis.*
