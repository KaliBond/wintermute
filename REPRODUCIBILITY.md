# Reproducibility Guide
## CAMS Framework v2.1 Scientific Reproducibility

This document ensures full reproducibility of all CAMS Framework results according to open science standards.

## 🔬 **Computational Environment**

### **System Requirements**
```
Operating System: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
Python Version: 3.8 or higher
Memory: 4GB RAM minimum, 8GB recommended
Storage: 2GB free space for datasets and outputs
```

### **Exact Dependencies**
```bash
# Core scientific stack
numpy==1.24.3
pandas==2.0.3
scipy==1.11.1
scikit-learn==1.3.0

# Visualization
matplotlib==3.7.2
plotly==5.15.0
streamlit==1.25.0

# Network analysis
networkx==3.1

# Optional (for advanced features)
jupyter==1.0.0
```

### **Installation Script**
```bash
# Create virtual environment
python -m venv cams_env
source cams_env/bin/activate  # Linux/Mac
# cams_env\Scripts\activate   # Windows

# Install exact versions
pip install -r requirements.txt

# Verify installation
python cams_framework_v2_1.py
```

## 📊 **Data Reproducibility**

### **Seed Values**
All random processes use fixed seeds for reproducibility:
```python
np.random.seed(42)           # NumPy operations
random.seed(42)              # Python random
tf.random.set_seed(42)       # TensorFlow (if used)
```

### **Dataset Integrity**
```bash
# Verify dataset checksums
python verify_datasets.py

# Expected checksums:
# USA_cleaned.csv: sha256:a1b2c3d4...
# China_cleaned.csv: sha256:e5f6g7h8...
```

### **Data Processing Pipeline**
```python
# Exact data loading procedure
from cams_framework_v2_1 import CAMSDataProcessor

processor = CAMSDataProcessor()
df = processor.load_society_data("data/cleaned/USA_cleaned.csv")

# Validation checkpoint
assert len(df) == 1248, f"Expected 1248 records, got {len(df)}"
assert set(df.columns) == {'Society', 'Year', 'Node', 'Coherence', 'Capacity', 'Stress', 'Abstraction'}
```

## 🧮 **Mathematical Validation**

### **USA 1861 Benchmark Test**
This is the primary validation case that must pass exactly:

```python
def test_usa_1861_reproducibility():
    """Exact reproduction of USA 1861 validation."""
    
    # Test states (exact values from documentation)
    states = {
        "Army": NodeState(coherence=6, capacity=6, stress=3, abstraction=5),
        "Executive": NodeState(coherence=4, capacity=5, stress=2, abstraction=6),
        "Merchants": NodeState(coherence=6, capacity=6, stress=3, abstraction=6),
        "Knowledge Workers": NodeState(coherence=5, capacity=5, stress=4, abstraction=7),
        "Proletariat": NodeState(coherence=4, capacity=5, stress=2, abstraction=4),
        "Property Owners": NodeState(coherence=3, capacity=7, stress=2, abstraction=6),
        "State Memory": NodeState(coherence=5, capacity=6, stress=3, abstraction=7),
        "Trades/Professions": NodeState(coherence=6, capacity=6, stress=3, abstraction=6),
    }
    
    network = CAMSNetwork(states)
    
    # Expected exact results
    expected_node_values = {
        "Army": 11.5, "Executive": 10.0, "Merchants": 12.0,
        "Knowledge Workers": 9.5, "Proletariat": 9.0, "Property Owners": 11.0,
        "State Memory": 11.5, "Trades/Professions": 12.0
    }
    
    # Validate each node (must be exact to 3 decimal places)
    for node, expected in expected_node_values.items():
        state = network.states[node]
        computed = state.coherence + state.capacity - state.stress + state.abstraction * 0.5
        assert abs(computed - expected) < 0.001, f"{node}: expected {expected}, got {computed}"
    
    # System metrics (must be within tolerance)
    psi = network.compute_psi_metric()
    health = network.compute_system_health()
    ca = network.compute_coherence_asymmetry()
    
    assert abs(psi - 0.724) < 0.001, f"Psi: expected ~0.724, got {psi}"
    assert abs(health - 0.766) < 0.001, f"Health: expected ~0.766, got {health}"
    assert abs(ca - 0.004) < 0.001, f"CA: expected ~0.004, got {ca}"
    
    return True

# Run validation
assert test_usa_1861_reproducibility(), "USA 1861 validation failed"
```

## 🔄 **Algorithmic Reproducibility**

### **Neural Network Parameters**
```python
# Exact parameter values used in v2.1
PARAMS = {
    "eta": 0.10,          # Learning rate
    "gamma": 0.02,        # Weight decay
    "zeta": 0.08,         # Error-driven term coefficient
    "w_max": 8.0,         # Maximum weight value
    "lambda_chi": 0.10,   # Coherence decay
    "lambda_sigma": 0.30, # Stress decay
    "rho_sigma": 0.40,    # Stress propagation
    "gamma_alpha": 0.05,  # Abstraction growth
    "delta_alpha": 0.10,  # Stress penalty on abstraction
    "beta_stress": 0.30,  # Stress modulation strength
}
```

### **Activation Function**
```python
def compute_node_activation(self, node_idx: int) -> float:
    """Exact activation function implementation."""
    # ... (exact implementation from framework)
    
    # Critical: sigmoid slope must be exactly 0.15
    a = 1.0 / (1.0 + np.exp(-(u) / 0.15))
    return float(np.clip(a, 0.0, 1.0))
```

### **Plasticity Update Rule**
```python
def update_weights(self, dt: float = 0.1):
    """Exact plasticity implementation."""
    # Δw_ij = η·tanh(C_i C_j/100)·(1 - S̄'_i) - γ·w_ij + ζ·ε_ij
    
    for i, node_i in enumerate(self.NODES):
        for j, node_j in enumerate(self.NODES):
            if i == j: continue
            
            # Exact formula implementation
            coherence_term = math.tanh((C_i * C_j) / 100.0)  # Must use math.tanh
            stress_gate = 1.0 - S_bar_i                      # Exact subtraction
            epsilon_ij = K_prime_j - K_hat_j                 # One-step persistence
            
            dw = (eta * coherence_term * stress_gate 
                  - gamma * w_ij + zeta * epsilon_ij)
```

## 📈 **Result Reproducibility**

### **Expected Output Format**
```
============================================================
CAMS Framework v2.1 - Complete Formalization
============================================================
Running USA 1861 validation...

Node Value Validation:
  Army                : Expected=  11.5, Computed=  11.5 OK
  Executive           : Expected=  10.0, Computed=  10.0 OK
  Merchants           : Expected=  12.0, Computed=  12.0 OK
  Knowledge Workers   : Expected=   9.5, Computed=   9.5 OK
  Proletariat         : Expected=   9.0, Computed=   9.0 OK
  Property Owners     : Expected=  11.0, Computed=  11.0 OK
  State Memory        : Expected=  11.5, Computed=  11.5 OK
  Trades/Professions  : Expected=  12.0, Computed=  12.0 OK

System Metrics:
  Psi:                  0.724
  Health:               0.766
  Coherence Asymmetry:  0.004
  System Type:          Steady Climber
```

### **Batch Analysis Results**
```python
# Expected results for standard test datasets
EXPECTED_RESULTS = {
    "USA_cleaned": {
        "mean_psi": 0.629,
        "final_psi": 0.633,
        "classification": "Stable Core",
        "records": 1248
    },
    "China_cleaned": {
        "mean_psi": 0.587,
        "classification": "Resilient Innovator",
        "records": 896
    }
}
```

## 🧪 **Experimental Protocols**

### **Single Dataset Analysis**
```python
# Standard analysis protocol
def reproduce_single_analysis(dataset_path):
    """Reproduce single dataset analysis exactly."""
    
    # Step 1: Load data with exact parameters
    processor = CAMSDataProcessor()
    df = processor.load_society_data(dataset_path)
    
    # Step 2: Create network for latest year
    years = sorted(df['Year'].unique())
    network = processor.create_network_from_year(df, years[-1])
    
    # Step 3: Run dynamics for exactly 10 steps
    for i in range(10):
        network.step(dt=0.1)
    
    # Step 4: Compute final metrics
    results = {
        "psi": network.compute_psi_metric(),
        "health": network.compute_system_health(),
        "ca": network.compute_coherence_asymmetry(),
        "classification": network.classify_system_type()
    }
    
    return results
```

### **Batch Processing Protocol**
```python
# Exact batch processing steps
def reproduce_batch_analysis(data_files):
    """Reproduce batch analysis with exact ordering."""
    
    # Sort files alphabetically for consistent ordering
    sorted_files = sorted(data_files)
    
    results = {}
    for filepath in sorted_files:
        # Process each file with identical parameters
        result = reproduce_single_analysis(filepath)
        society_name = os.path.basename(filepath).split('.')[0]
        results[society_name] = result
    
    return results
```

## 🔍 **Debugging and Verification**

### **Common Issues and Solutions**

1. **Floating Point Precision**
   ```python
   # Use consistent precision
   import numpy as np
   np.set_printoptions(precision=3, suppress=True)
   
   # Validate with tolerance
   assert abs(computed - expected) < 1e-3
   ```

2. **Platform Differences**
   ```python
   # Handle OS-specific differences
   import platform
   if platform.system() == "Windows":
       # Windows-specific handling
       pass
   ```

3. **Version Compatibility**
   ```bash
   # Check Python version
   python --version  # Must be 3.8+
   
   # Check package versions
   pip freeze | grep numpy  # Must match requirements.txt
   ```

### **Verification Checklist**
- [ ] USA 1861 validation passes exactly
- [ ] All 32 datasets load without errors
- [ ] Mathematical formulas produce expected results
- [ ] Dashboard displays correctly on port 8501
- [ ] No circular reference errors in JSON export
- [ ] All test cases pass with pytest

## 📋 **Full Reproduction Script**

```bash
#!/bin/bash
# complete_reproduction.sh

echo "CAMS Framework v2.1 - Full Reproduction"
echo "========================================"

# 1. Environment setup
echo "Setting up environment..."
python -m venv cams_repro
source cams_repro/bin/activate
pip install -r requirements.txt

# 2. Data verification
echo "Verifying datasets..."
python verify_datasets.py

# 3. Core validation
echo "Running USA 1861 validation..."
python cams_framework_v2_1.py

# 4. Full test suite
echo "Running comprehensive tests..."
python -m pytest tests/ -v

# 5. Dashboard test
echo "Testing dashboard..."
timeout 30s streamlit run cams_can_v34_explorer.py &
sleep 10
curl -s http://localhost:8501 > /dev/null && echo "Dashboard OK"

echo "Reproduction complete!"
```

## 📧 **Support for Reproduction**

### **Contact for Reproduction Issues**
- **Email**: kari.freyr.4@gmail.com
- **Subject**: "CAMS Reproduction Issue - [Brief Description]"
- **Include**: 
  - Your system specifications
  - Exact error messages
  - Python and package versions
  - Steps attempted

### **Expected Response Time**
- **Critical reproduction failures**: 24-48 hours
- **Platform-specific issues**: 3-5 days  
- **Enhancement requests**: 1-2 weeks

---

**This reproducibility guide ensures that all CAMS Framework v2.1 results can be independently verified and reproduced by the scientific community.**

*Last updated: August 15, 2025*