# Contributing to CAMS Framework

We welcome contributions to the CAMS Framework! This document provides guidelines for contributing to this open science project.

## 🤝 Ways to Contribute

### 1. **Data Contributions**
- **New Society Datasets**: Historical or contemporary CAMS data
- **Data Validation**: Cross-verification of existing datasets
- **Data Quality**: Cleaning and standardization improvements

### 2. **Algorithm Improvements**
- **Mathematical Enhancements**: Improved metrics and formulations
- **Neural Network Advances**: Better plasticity rules or activation functions
- **Optimization**: Performance improvements and efficiency gains

### 3. **Visualization & Tools**
- **Dashboard Components**: New interactive elements
- **Analysis Tools**: Additional statistical or graphical capabilities
- **User Interface**: Improved usability and accessibility

### 4. **Research & Validation**
- **Historical Validation**: Testing against known historical events
- **Cross-Cultural Studies**: Applications to different societal contexts
- **Methodological Research**: Comparative studies with other frameworks

## 📋 Contribution Process

### **Before You Start**
1. **Check Issues**: Look for existing issues or feature requests
2. **Discussion**: Open an issue to discuss major changes
3. **Fork Repository**: Create your own fork for development

### **Development Workflow**
1. **Clone your fork**:
   ```bash
   git clone https://github.com/[YOUR-USERNAME]/wintermute.git
   cd wintermute
   ```

2. **Create feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make changes**: Follow coding standards (see below)

4. **Test thoroughly**:
   ```bash
   python cams_framework_v2_1.py  # Run validation
   python -m pytest tests/        # Run test suite
   ```

5. **Commit with clear messages**:
   ```bash
   git commit -m "Add: Brief description of your changes
   
   - Detailed explanation of what changed
   - Why the change was necessary
   - Any breaking changes or new requirements"
   ```

6. **Push and create Pull Request**:
   ```bash
   git push origin feature/your-feature-name
   ```

## 💻 Coding Standards

### **Python Code Style**
- **PEP 8**: Follow Python style guidelines
- **Type Hints**: Use type annotations where appropriate
- **Docstrings**: Document all public functions and classes
- **Comments**: Explain complex mathematical formulations

### **Mathematical Documentation**
- **Formula Comments**: Include mathematical formulas in docstrings
- **Variable Naming**: Use descriptive names matching mathematical notation
- **Units**: Clearly specify units and ranges for all variables
- **References**: Cite relevant papers or theoretical foundations

### **Example Code Style**
```python
def compute_psi_metric(self) -> float:
    """
    Compute grand system metric Ψ (Section 2.2).
    
    Formula: Ψ = 0.35 H' + 0.25 P_S + 0.20 P_C + 0.20 P_A
    
    Returns:
        float: System metric in range [0, 1]
        
    References:
        McKern, K. (2025). CAMS Framework Formalization
    """
    # Implementation here...
```

## 📊 Data Contribution Guidelines

### **Dataset Format**
```csv
Society,Year,Node,Coherence,Capacity,Stress,Abstraction
USA,2025,Executive,5.2,6.1,-2.3,7.8
```

### **Data Quality Standards**
- **Completeness**: All required columns present
- **Validation**: Cross-check against historical sources
- **Documentation**: Include methodology and sources
- **Metadata**: Provide context and collection methods

### **Data Sources**
- **Primary Sources**: Government records, historical documents
- **Academic Sources**: Peer-reviewed research and datasets
- **Attribution**: Proper citation of all data sources
- **Permissions**: Ensure appropriate usage rights

## 🧪 Testing Requirements

### **Mandatory Tests**
- **USA 1861 Validation**: Must pass benchmark test
- **Mathematical Consistency**: Verify formulas and calculations
- **Data Processing**: Test with various input formats
- **Regression Tests**: Ensure existing functionality unchanged

### **Test Categories**
```bash
# Unit tests
python -m pytest tests/test_neural_network.py

# Integration tests  
python -m pytest tests/test_data_processing.py

# Validation tests
python -m pytest tests/test_validation.py

# Performance tests
python -m pytest tests/test_performance.py
```

## 📝 Documentation Standards

### **Required Documentation**
- **README Updates**: Reflect new features or changes
- **API Documentation**: Document all public interfaces
- **Mathematical Formulations**: Explain theoretical foundations
- **Usage Examples**: Provide clear examples for new features

### **Academic Standards**
- **Peer Review**: Welcome feedback from domain experts
- **Reproducibility**: Ensure all results can be reproduced
- **Open Data**: Share datasets when legally permissible
- **Version Control**: Maintain clear versioning and changelogs

## 🔬 Research Collaboration

### **Academic Partnerships**
- **University Collaborations**: Welcome institutional partnerships
- **Interdisciplinary Work**: Encourage cross-field applications
- **Publication Support**: Assist with academic paper preparation
- **Conference Presentations**: Support presenting framework at conferences

### **Research Areas**
- **Historical Analysis**: Applications to historical societies
- **Contemporary Studies**: Modern institutional analysis
- **Predictive Modeling**: Forecasting societal transitions
- **Comparative Studies**: Cross-cultural and cross-temporal analysis

## 📧 Communication

### **Contact Methods**
- **Email**: kari.freyr.4@gmail.com (Lead Researcher)
- **Issues**: GitHub issue tracker for bugs and features
- **Discussions**: GitHub discussions for general questions
- **Academic**: Contact for research collaborations

### **Response Time**
- **Bug Reports**: 48-72 hours
- **Feature Requests**: 1-2 weeks
- **Academic Inquiries**: 1 week
- **Data Requests**: 2-4 weeks

## 🏆 Recognition

### **Contributor Recognition**
- **CONTRIBUTORS.md**: All contributors acknowledged
- **Academic Citations**: Co-authorship for significant contributions
- **Conference Presentations**: Opportunity to present work
- **Research Credits**: Recognition in publications

### **Types of Recognition**
- **Code Contributors**: Technical improvements and features
- **Data Contributors**: Dataset provision and validation
- **Research Contributors**: Theoretical advances and applications
- **Community Contributors**: Documentation, outreach, and support

## 📜 License and Attribution

### **License Terms**
- **MIT License**: Permissive open-source license
- **Academic Use**: Free for research and educational purposes
- **Commercial Use**: Contact for commercial licensing
- **Attribution**: Required citation in derivative works

### **Citation Requirements**
Please cite this work as:
```
McKern, K. (2025). CAMS Framework: Complex Adaptive Metrics of Society (Version 2.1) 
[Computer software]. https://github.com/[USERNAME]/wintermute
```

---

**Thank you for contributing to open science and the advancement of complex systems research!**

*For questions about contributing, please contact: kari.freyr.4@gmail.com*