# CAMS Framework: Complex Adaptive Metrics of Society
## Version 2.1 - Production Ready Thermodynamic Implementation

![CAMS Logo](https://img.shields.io/badge/CAMS-v2.1-blue) ![License](https://img.shields.io/badge/License-Open%20Science-green) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

**Purpose**: CAMS analyzes societal structures as thermodynamic Complex Adaptive Systems, focusing on the mathematical dynamics between 8 institutional nodes and their emergent behaviors across civilizations and time periods.

> **Note**: The neural network hypothesis has been falsified (December 2025). CAMS now focuses on thermodynamic principles, entropy flows, and phase transitions.

## 🌡️ Core Architecture

### **Thermodynamic Foundation**
- **8 Institutional Nodes**: Executive (Helm), Army (Shield), Knowledge Workers (Lore), Property Owners (Stewards), Trades/Professions (Craft), Proletariat (Hands), State Memory (Archive), Merchants (Flow)
- **4 State Variables per Node**: Coherence (C), Capacity (K), Stress (S), Abstraction (A)
- **Thermodynamic Dynamics**: Entropy flows, phase transitions, and bistability (Ψ vs Φ modes)
- **Normalized Mathematics**: End-to-end unit scaling for mathematical consistency

### **Grand System Metric (Ψ)**
```
Ψ = 0.35 H' + 0.25 P_S + 0.20 P_C + 0.20 P_A
```
Where:
- `H'` = System Health (stress-free)
- `P_S` = Stress Penalty (1 - S̄')
- `P_C` = Coherence Penalty (1 - dispersion)
- `P_A` = Abstraction Penalty (1 - Ā')

## 🚀 CAMS-CAN Integrated Dashboard

**NEW**: Access the complete CAMS-CAN analysis system at [cams-integrated-dashboard.html](cams-integrated-dashboard.html)

### **Key Features**
- **Interactive Data Upload**: CSV files or manual entry
- **Real-time Calculations**: All CAMS-CAN formulas implemented
- **Critical Threshold Alerts**: Automatic risk assessment
- **Multi-tab Interface**: Dashboard, Data, Formulas, Samples, Validation
- **Time Series Visualization**: Track system evolution
- **Network Analysis**: Node relationship mapping
- **Sample Datasets**: Norway historical, crisis scenarios
- **Validation Protocol**: Data quality checks

### **Getting Started**
1. Visit the dashboard
2. Upload your CSV data (format: Society,Year,Node,Coherence,Capacity,Stress,Abstraction,Node_Value,Bond_Strength)
3. View automated calculations and visualizations
4. Download sample datasets for testing
5. Run validation checks for data quality

### **For Researchers & Practitioners**
- Use the Model Reference tab for formula documentation
- Access sample datasets for benchmarking
- Follow validation protocols for rigorous analysis
- Export results for further study

**Submit this framework to any AI for thesis testing**: [Download one-shot test file](cams-one-shot-test.txt) - provides complete framework description with Norway data for independent validation.

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

### **Neural Network Dynamics**
- **Stress-Gated Plasticity**: `Δw_ij = η·tanh(C_i C_j/100)·(1 - S̄'_i) - γ·w_ij + ζ·ε_ij`
- **Error-Driven Learning**: One-step capacity forecasting with persistence
- **Range-Safe Activation**: Weighted neighbor averaging with adaptive thresholds
- **Discrete-Time Integration**: Euler method with configurable time steps

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
**Framework Version**: 2.1 (August 2025)  
**License**: Open Science - Common Property

---

*CAMS Framework v2.1 represents a production-ready implementation of thermodynamic Complex Adaptive Systems analysis for societal dynamics, validated against historical data and ready for rigorous academic and policy research.*
