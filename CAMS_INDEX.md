# CAMS-CAN v3.4 Comprehensive Index

**Complex Adaptive Management Systems - Catch-All Network v3.4**  
**Stress as Societal Meta-Cognition Framework**

---

## 📚 TABLE OF CONTENTS

### [A. Core Applications](#core-applications)
### [B. Mathematical Framework](#mathematical-framework)  
### [C. Datasets & Data Management](#datasets-data-management)
### [D. Analysis Tools](#analysis-tools)
### [E. Visualization Components](#visualization-components)
### [F. Documentation](#documentation)
### [G. Validation & Testing](#validation-testing)
### [H. Utility Scripts](#utility-scripts)

---

## A. CORE APPLICATIONS

### 🧠 Primary Analysis Interface
- **`cams_can_v34_explorer.py`** - **[MAIN APPLICATION]**
  - Complete CAMS-CAN v3.4 Stress Dynamics Explorer
  - 8 Institutional Nodes Analysis
  - 32-Dimensional Phase Space Visualization
  - 5 Key Metrics Dashboard
  - Interactive visualizations with PCA/t-SNE/UMAP
  - Time filtering and institution filtering
  - Attractor behavior detection

### 🎛️ Dashboard Applications
- **`fixed_cams_dashboard.py`** - Enhanced CAMS dashboard with proper calculations
- **`cams_stress_dynamics_app.py`** - Main stress dynamics application
- **`simple_dashboard.py`** - Simplified analysis interface
- **`dashboard.py`** - Basic CAMS analysis dashboard

### 🔬 Specialized Analysis Tools
- **`iran_transition_analysis_fixed.py`** - Iran phase transition analysis (working version)
- **`stress_dynamics_engine.py`** - Core mathematical engine
- **`formal_cams_can_framework.py`** - Formal mathematical implementation
- **`unified_cams_13laws_framework.py`** - Unified CAMS + 13 Laws framework

---

## B. MATHEMATICAL FRAMEWORK

### 📐 Core Mathematical Components
- **Node Fitness:** `Hi(t) = Ci*Ki / (1+exp((|Si|-τ)/λ)) * (1+Ai/10)`
- **System Health:** `Ω(S,t) = (∏Hi)^(1/8) * (1+Φ_network) * R(t)`
- **Coherence Asymmetry:** `CA = std(Ci*Ki) / mean(Ci*Ki)`
- **Critical Transition Risk:** `CTR = (stress_var * (1 + CA)) / (system_health + ε)`
- **Network Coherence:** `NC = correlation(Coherence, Capacity)`

### 🎯 Phase Attractors
- **A₁ Adaptive:** Ω>3.5, CA<0.3 - High coordination, efficient processing
- **A₂ Authoritarian:** Ω∈[2.5,3.5] - Centralized control, moderate efficiency  
- **A₃ Fragmented:** Ω∈[1.5,2.5], CA>0.4 - Distributed but inefficient
- **A₄ Collapse:** Ω<1.5 - System breakdown, meta-cognitive failure

### 🔬 Advanced Framework Files
- **`advanced_cams_laws.py`** - Advanced CAMS with constraint laws
- **`gtsc_stsc_toolkit.py`** - General/Specific Transition State Complexity toolkit

---

## C. DATASETS & DATA MANAGEMENT

### 📊 Validated Datasets (32 Total)
Located in `/cleaned_datasets/` - All CAMS-CAN v3.4 compatible

#### Major Civilizations:
- **`Australia_cleaned.csv`** - 984 records (1900-2024)
- **`USA_HighRes_cleaned.csv`** - 2,168 records (1790-2025) 
- **`Iran_cleaned.csv`** - 920 records (1900-2025)
- **`Germany_cleaned.csv`** - 2,199 records (1750-2025)
- **`Japan_cleaned.csv`** - 1,632 records (1850-2025)
- **`Russia_cleaned.csv`** - 1,208 records (1900-2025)
- **`Italy_cleaned.csv`** - 1,080 records (1900-2024)

#### Regional Coverage:
- **Europe:** Germany, France, Denmark, Italy, Netherlands, England
- **North America:** USA (multiple versions), Canada
- **Middle East:** Iran, Iraq, Saudi Arabia, Israel, Lebanon, Syria  
- **Asia-Pacific:** Japan, Thailand, Hong Kong, Singapore, Indonesia, India, Pakistan
- **Historical:** Roman Empire datasets

### 🔧 Data Management Tools
- **`dataset_validator.py`** - Comprehensive dataset validation and cleaning
- **`verify_all_datasets.py`** - Dataset verification tool
- **`fix_data_files.py`** - Data repair utilities
- **`regenerate_clean_data.py`** - Data regeneration tools

---

## D. ANALYSIS TOOLS

### 📈 Core Analysis Scripts
- **`simple_analysis.py`** - Basic CAMS analysis
- **`enhanced_stress_analysis.py`** - Advanced stress analysis
- **`simple_stress_analysis.py`** - Simplified stress calculations
- **`cams_validation_suite.py`** - Comprehensive validation suite

### 🔍 Specialized Analysis
- **`inter_institutional_bond_network.py`** - Network analysis tools
- **`cams_analysis_charts.py`** - Chart generation utilities
- **`run_analysis.py`** - Analysis execution script
- **`run_advanced_analysis.py`** - Advanced analysis runner

### 🧪 Testing & Validation
- **`test_cleaned_datasets.py`** - Dataset compatibility testing
- **`test_complete.py`** - Comprehensive testing suite
- **`test_simple.py`** - Basic functionality tests
- **`standalone_test.py`** - Standalone testing utility

---

## E. VISUALIZATION COMPONENTS

### 📊 Interactive Visualizations (CAMS-CAN v3.4)

#### 1. **Node Stress Trajectories**
- Time series visualization of institutional stress evolution
- Interactive filtering by time window and institutions

#### 2. **32-Dimensional Phase Space Analysis** 
- PCA/t-SNE/UMAP dimensionality reduction
- Attractor trajectory visualization
- Critical transition highlighting

#### 3. **Attractor Basin Visualization**
- Phase space regions for A₁-A₄ attractors
- Current civilization positioning
- Transition probability landscapes

#### 4. **Bond Strength Matrix Heatmap**
- Inter-institutional coherence correlations
- Network coherence visualization
- Relationship strength mapping

#### 5. **System Health & Risk Trends**
- Multi-metric dashboard (Ω, CA, CTR, NC)
- Historical trend analysis
- Risk threshold monitoring

#### 6. **Coordinated Interactive Visualization**
- Synchronized time controls
- Multi-panel analysis views
- Real-time parameter adjustment

### 🎨 Visualization Files
- **`src/visualizations.py`** - Core visualization components
- **`show_outputs.py`** - Output display utilities

---

## F. DOCUMENTATION

### 📋 Core Documentation
- **`DATASET_VALIDATION_SUMMARY.md`** - **[DATASET STATUS]**
- **`validation_report.md`** - Detailed validation results
- **`CAMS_INDEX.md`** - **[THIS FILE]** - Comprehensive index
- **`README.md`** - Project overview
- **`STATUS.md`** - Current project status

### 📚 Technical Documentation
- **`CAMS_Validation_Formulation.md`** - Mathematical formulation
- **`CAMS_Validation_Quick_Reference.md`** - Quick reference guide
- **`Advanced_CAMS_Integration_Summary.md`** - Integration summary

### 📁 Data Documentation
- **`data_input/README.md`** - Data input instructions
- **`data_input/INSTRUCTIONS.txt`** - Data formatting guidelines

---

## G. VALIDATION & TESTING

### ✅ Validation Status
- **32 datasets validated** and CAMS-CAN v3.4 compatible
- **Mathematical framework verified** across multiple test cases
- **All core functions tested** with real data
- **Visualization components validated** with interactive testing

### 🧪 Test Files
- **`test_node_matrix.py`** - Node matrix functionality test
- **`test_dashboard_functionality.py`** - Dashboard testing
- **`minimal_test.py`** - Minimal functionality verification
- **`verify_simple.py`** - Simple verification checks

### 📊 Working Examples
- **`simple_dynamic_evolution.py`** - **[USER CONFIRMED WORKING]**
- **`simple_working_simulation.py`** - Basic simulation example

---

## H. UTILITY SCRIPTS

### 🔧 Import & Data Processing
- **`import_github_data.py`** - GitHub data import
- **`github_importer_app.py`** - GitHub import application
- **`process_data_input.py`** - Data input processing
- **`quick_import.py`** - Quick data import utility

### 🛠️ Maintenance & Fixes
- **`fix_monitor_syntax.py`** - Monitor syntax repairs
- **`fix_thermo_section.py`** - Thermodynamics section fixes
- **`clean_iran_data.py`** - Iran-specific data cleaning
- **`debug_*.py`** - Various debugging utilities

### ⚙️ Configuration
- **`setup.py`** - Project setup
- **`requirements.txt`** - Python dependencies
- **`src/__init__.py`** - Package initialization

---

## 🎯 QUICK START GUIDE

### 1. **Primary Analysis Interface**
```bash
streamlit run cams_can_v34_explorer.py
```
**Features:** Full CAMS-CAN v3.4 interface, 32 validated datasets, interactive analysis

### 2. **Dataset Information**
- **Location:** `/cleaned_datasets/`
- **Count:** 32 validated civilizations  
- **Coverage:** 3 CE - 2025 CE
- **Status:** ✅ All ready for analysis

### 3. **Key Metrics Available**
- Individual Node Fitness H_i(t)
- System Health Ω(S,t)  
- Coherence Asymmetry CA(t)
- Critical Transition Risk CTR(t)
- Network Coherence NC(t)

### 4. **Analysis Capabilities**
- ✅ Time series analysis
- ✅ Cross-civilization comparison
- ✅ 32D phase space dynamics
- ✅ Attractor detection
- ✅ Crisis period analysis

---

## 📊 FILE STATISTICS

| Category | File Count | Status |
|----------|------------|--------|
| **Core Applications** | 8 | ✅ Ready |
| **Analysis Tools** | 12 | ✅ Validated |
| **Datasets (Cleaned)** | 32 | ✅ CAMS-CAN v3.4 Compatible |
| **Visualization Components** | 6 | ✅ Interactive |
| **Documentation** | 6 | ✅ Comprehensive |
| **Test Files** | 10 | ✅ Passing |
| **Utility Scripts** | 15+ | ✅ Functional |

---

## 🏆 COMPLETION STATUS

### ✅ **FULLY IMPLEMENTED:**
- CAMS-CAN v3.4 Mathematical Framework
- Interactive Stress Dynamics Explorer  
- 32 Validated and Cleaned Datasets
- Comprehensive Analysis Tools
- Phase Space Visualization System
- Attractor Behavior Detection

### 🎯 **READY FOR:**
- Advanced civilizational analysis
- Historical crisis period research
- Cross-civilization comparison studies
- Real-time stress dynamics monitoring
- Academic research applications
- Policy analysis and recommendations

---

**Last Updated:** 2025-08-11  
**Framework Version:** CAMS-CAN v3.4  
**Total Project Files:** 100+  
**Analysis-Ready Datasets:** 32  
**Time Span Covered:** 2,022 years (3 CE - 2025 CE)

---

*For detailed information on any component, refer to the individual files or contact the development team.*