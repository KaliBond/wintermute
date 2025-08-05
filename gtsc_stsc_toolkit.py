"""
🛠️ GTSC-STSC Implementation Toolkit
From Theoretical Framework to Practical Analysis

A comprehensive implementation of the GTSC-STSC framework for evidence-based societal diagnosis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Essential Metrics Framework
REQUIRED_METRICS = {
    "Coherence (C)": {
        "scale": "1-10",
        "indicators": [
            "institutional_unity_index",
            "policy_consistency_score", 
            "leadership_stability_measure",
            "public_confidence_ratings"
        ],
        "data_sources": ["polling", "institutional_analysis", "policy_tracking"]
    },
    
    "Capacity (K)": {
        "scale": "1-10", 
        "indicators": [
            "resource_mobilization_efficiency",
            "operational_effectiveness",
            "infrastructure_quality",
            "human_capital_metrics"
        ],
        "data_sources": ["economic_statistics", "performance_indices", "infrastructure_assessments"]
    },
    
    "Stress (S)": {
        "scale": "1-10",
        "indicators": [
            "internal_pressures",
            "external_challenges", 
            "resource_constraints",
            "social_tensions"
        ],
        "data_sources": ["conflict_databases", "economic_stress_indices", "social_monitoring"]
    },
    
    "Abstraction (A)": {
        "scale": "1-10",
        "indicators": [
            "communication_complexity",
            "institutional_sophistication",
            "information_processing_capacity",
            "narrative_elaboration"
        ],
        "data_sources": ["media_analysis", "institutional_complexity_measures", "education_metrics"]
    },
    
    "Adaptability (D)": {
        "scale": "1-10",
        "indicators": [
            "response_flexibility",
            "learning_capacity",
            "innovation_adoption",
            "change_management_effectiveness"
        ],
        "data_sources": ["policy_adaptation_tracking", "innovation_indices", "response_time_analysis"]
    }
}

class GTSCAnalyzer:
    """Core GTSC-STSC analysis engine"""
    
    def __init__(self, data):
        self.data = data
        self.results = {}
        
    def calculate_node_values(self):
        """Calculate NV_i for each node using refined formula"""
        NV = []
        for _, row in self.data.iterrows():
            C = row.get('Coherence', 0)
            K = row.get('Capacity', 0) 
            S = row.get('Stress', 0)
            A = row.get('Abstraction', 0)
            
            # Enhanced node value calculation
            nv = (C * K) / (1 + S/10) + 0.5 * A
            NV.append(nv)
            
        self.data['Node_Value'] = NV
        return NV
    
    def calculate_system_health(self):
        """Calculate H_t system health with adaptive weighting"""
        node_values = self.data['Node_Value']
        
        # Adaptive weights based on node criticality
        node_weights = {
            'Executive': 0.15,  # Leadership critical
            'Army': 0.12,       # Security essential
            'Archive': 0.13,    # Memory preservation vital
            'Lore': 0.12,       # Cultural continuity
            'Stewards': 0.12,   # Administrative function
            'Craft': 0.12,      # Economic production
            'Flow': 0.12,       # Resource distribution
            'Hands': 0.12       # Labor foundation
        }
        
        # Calculate weighted system health
        H_t = 0
        for _, row in self.data.iterrows():
            node = row.get('Node', 'Unknown')
            weight = node_weights.get(node, 1/8)  # Default equal weight
            H_t += weight * row['Node_Value']
            
        self.results['System_Health'] = H_t
        return H_t
        
    def calculate_synchronization(self):
        """Calculate coordination index with robustness measures"""
        coherence_values = self.data['Coherence'].values
        
        if len(coherence_values) == 0:
            sync = 0
        else:
            variance = np.var(coherence_values)
            mean = np.mean(coherence_values)
            
            # Robust synchronization calculation
            if mean > 0:
                sync = 1 - (variance / (mean**2))  # Normalized by mean squared
                sync = max(0, min(1, sync))  # Bound between 0 and 1
            else:
                sync = 0
                
        self.results['Synchronization'] = sync
        return sync
        
    def calculate_stress_asymmetry(self):
        """Calculate SAR stress distribution with edge case handling"""
        stress_values = self.data['Stress'].values
        
        if len(stress_values) == 0:
            sar = 0
        else:
            max_stress = np.max(stress_values)
            min_stress = np.min(stress_values)
            mean_stress = np.mean(stress_values)
            
            if mean_stress > 0:
                sar = (max_stress - min_stress) / mean_stress
            else:
                sar = 0
                
        self.results['Stress_Asymmetry'] = sar
        return sar
        
    def calculate_narrative_coherence(self):
        """Calculate NCI for Archive and Lore nodes"""
        try:
            archive_row = self.data[self.data['Node'] == 'Archive'].iloc[0]
            lore_row = self.data[self.data['Node'] == 'Lore'].iloc[0]
            
            avg_coherence = (archive_row['Coherence'] + lore_row['Coherence']) / 2
            avg_abstraction = (archive_row['Abstraction'] + lore_row['Abstraction']) / 2
            
            nci = avg_coherence * avg_abstraction
        except (IndexError, KeyError):
            # Fallback if Archive/Lore nodes not found
            coherence_mean = self.data['Coherence'].mean()
            abstraction_mean = self.data['Abstraction'].mean()
            nci = coherence_mean * abstraction_mean
            
        self.results['Narrative_Coherence'] = nci
        return nci
        
    def run_full_analysis(self):
        """Execute complete GTSC analysis"""
        self.calculate_node_values()
        self.calculate_system_health()
        self.calculate_synchronization()
        self.calculate_stress_asymmetry()
        self.calculate_narrative_coherence()
        
        # Calculate derived metrics
        sar = self.results['Stress_Asymmetry']
        nci = self.results['Narrative_Coherence']
        self.results['Fragility_Factor'] = sar * (1 - nci/100)
        
        return self.results

class CAMSTRESSAnalyzer:
    """CAMSTRESS Thermodynamic Extension"""
    
    def __init__(self, data, alpha=1.2, beta=1.0, gamma=0.8, delta=0.9):
        self.data = data
        self.alpha = alpha  # Coherence-to-energy conversion
        self.beta = beta    # Capacity work efficiency  
        self.gamma = gamma  # Stress cost
        self.delta = delta  # Dissipation multiplier
        
    def calculate_node_energy(self, C, K, S):
        """Calculate thermodynamic energy for each node"""
        return self.alpha * (C**2) + self.beta * K - self.gamma * S
        
    def calculate_dissipation(self, S, A):
        """Calculate energy dissipation"""
        return self.delta * S * np.log(1 + A)
        
    def calculate_free_energy(self, K, S, A):
        """Calculate available energy for work"""
        return (K - S) * (1 - A / 10)
        
    def identify_heat_sinks(self, K, S, A):
        """Identify nodes operating below capacity"""
        return (K < S) and (A < 5)
        
    def run_thermodynamic_analysis(self):
        """Complete CAMSTRESS thermodynamic assessment"""
        results = []
        
        for _, row in self.data.iterrows():
            C = row.get('Coherence', 0)
            K = row.get('Capacity', 0)
            S = row.get('Stress', 0)
            A = row.get('Abstraction', 0)
            
            node_result = {
                'Node': row.get('Node', 'Unknown'),
                'Node_Energy': self.calculate_node_energy(C, K, S),
                'Dissipation': self.calculate_dissipation(S, A),
                'Free_Energy': self.calculate_free_energy(K, S, A),
                'Heat_Sink': self.identify_heat_sinks(K, S, A)
            }
            results.append(node_result)
            
        thermo_df = pd.DataFrame(results)
        
        # System-level metrics
        total_dissipation = thermo_df['Dissipation'].sum()
        total_free_energy = thermo_df['Free_Energy'].sum()
        
        system_metrics = {
            'Total_Entropy': total_dissipation,
            'Total_Free_Energy': total_free_energy,
            'Heat_Sink_Count': thermo_df['Heat_Sink'].sum(),
            'Energy_Efficiency': total_free_energy / (total_dissipation + 1e-6) if total_dissipation > 0 else 0
        }
        
        return thermo_df, system_metrics

def classify_system_health(H_t):
    """Classify societal health status"""
    if H_t >= 8.0:
        return "Transcendent", "Exceptional resilience and adaptation capacity"
    elif H_t >= 6.0:
        return "High Performance", "Strong coordination and stability"  
    elif H_t >= 4.0:
        return "Stable", "Adequate functioning with moderate resilience"
    elif H_t >= 2.5:
        return "Stressed", "Coordination challenges, intervention recommended"
    else:
        return "Critical", "System breakdown risk, urgent intervention required"

def assess_risk_level(metrics):
    """Assess overall risk level based on multiple indicators"""
    sar = metrics.get('Stress_Asymmetry', 0)
    nci = metrics.get('Narrative_Coherence', 0)
    h_t = metrics.get('System_Health', 0)
    
    if sar < 0.4 and nci > 60 and h_t > 6.0:
        return "Green"
    elif sar < 0.6 and nci > 60 and h_t > 4.0:
        return "Yellow" 
    elif sar < 0.6 and nci > 40 and h_t > 2.5:
        return "Orange"
    elif sar > 0.6 or nci < 40 or h_t < 2.5:
        return "Red"
    else:
        return "Critical"

def identify_system_patterns(metrics):
    """Recognize characteristic societal patterns"""
    patterns = []
    
    # Strong coordination pattern
    if metrics.get('Synchronization', 0) > 0.8:
        patterns.append("Strong_Coordination")
        
    # Narrative coherence pattern  
    if metrics.get('Narrative_Coherence', 0) > 65 and metrics.get('Stress_Asymmetry', 0) < 0.4:
        patterns.append("Strong_Social_Unity")
        
    # Stress cascade pattern
    if metrics.get('Stress_Asymmetry', 0) > 0.7:
        patterns.append("Cascade_Risk")
        
    # System fragility pattern
    if metrics.get('Fragility_Factor', 0) > 0.5:
        patterns.append("High_Fragility")
        
    return patterns

def create_gtsc_dashboard(data, metrics, patterns):
    """Create comprehensive GTSC-STSC dashboard"""
    
    st.markdown("## 🛠️ GTSC-STSC Analysis Dashboard")
    st.markdown("*Evidence-based societal diagnosis using thermodynamic principles*")
    
    # System Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        h_t = metrics.get('System_Health', 0)
        status, description = classify_system_health(h_t)
        st.metric("System Health", f"{h_t:.2f}", help=description)
        
    with col2:
        sync = metrics.get('Synchronization', 0)
        st.metric("Synchronization", f"{sync:.2f}", help="Coordination efficiency across nodes")
        
    with col3:
        sar = metrics.get('Stress_Asymmetry', 0)
        st.metric("Stress Asymmetry", f"{sar:.2f}", help="Distribution of stress across system")
        
    with col4:
        nci = metrics.get('Narrative_Coherence', 0)
        st.metric("Narrative Coherence", f"{nci:.1f}", help="Cultural and informational unity")
    
    # Risk Assessment
    risk_level = assess_risk_level(metrics)
    risk_colors = {
        "Green": "🟢", "Yellow": "🟡", "Orange": "🟠", 
        "Red": "🔴", "Critical": "⚫"
    }
    
    st.markdown(f"### Risk Assessment: {risk_colors.get(risk_level, '⚪')} {risk_level}")
    
    # Pattern Recognition
    if patterns:
        st.markdown("### 🔍 Identified Patterns")
        for pattern in patterns:
            st.info(f"**{pattern.replace('_', ' ')}** detected")
    
    # Node Analysis Table
    st.markdown("### 📊 Node Performance Analysis")
    display_df = data[['Node', 'Coherence', 'Capacity', 'Stress', 'Abstraction', 'Node_Value']].copy()
    display_df = display_df.round(2)
    st.dataframe(display_df, use_container_width=True)
    
    # Thermodynamic Analysis
    if 'thermodynamic_metrics' in st.session_state:
        thermo_metrics = st.session_state.thermodynamic_metrics
        
        st.markdown("### 🔬 CAMSTRESS Thermodynamic Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Entropy", f"{thermo_metrics['Total_Entropy']:.1f}")
        with col2:
            st.metric("Free Energy", f"{thermo_metrics['Total_Free_Energy']:.1f}")
        with col3:
            st.metric("Efficiency", f"{thermo_metrics['Energy_Efficiency']:.2f}")
    
    return metrics

def generate_analysis_report(society_name, time_period, metrics, patterns):
    """Generate standardized GTSC-STSC analysis report"""
    
    status, description = classify_system_health(metrics.get('System_Health', 0))
    risk_level = assess_risk_level(metrics)
    
    report = f"""
# GTSC-STSC Analysis Report: {society_name}
## Assessment Period: {time_period}

### Executive Summary
- **System Health (H_t)**: {metrics.get('System_Health', 0):.2f} - {status}
- **Risk Level**: {risk_level}  
- **Key Patterns**: {', '.join(patterns) if patterns else 'None detected'}

### Detailed Findings

#### Coordination Assessment
- **Synchronization Index**: {metrics.get('Synchronization', 0):.2f}
- **Stress Asymmetry Ratio**: {metrics.get('Stress_Asymmetry', 0):.2f}  
- **Narrative Coherence**: {metrics.get('Narrative_Coherence', 0):.1f}
- **Fragility Factor**: {metrics.get('Fragility_Factor', 0):.2f}

#### System Interpretation
{description}

#### Strategic Recommendations
Based on the analysis, the following interventions are recommended:
"""
    
    # Add specific recommendations based on patterns
    if 'Cascade_Risk' in patterns:
        report += "\n- **Urgent**: Address stress concentration in high-stress nodes"
    if 'High_Fragility' in patterns:
        report += "\n- **Priority**: Strengthen narrative coherence and institutional unity"
    if metrics.get('System_Health', 0) < 2.5:
        report += "\n- **Critical**: Implement crisis management protocols"
        
    return report

# Integration functions for existing CAMS systems
def integrate_with_cams_monitor(df):
    """Integrate GTSC-STSC analysis with existing CAMS real-time monitor"""
    
    # Run GTSC analysis
    gtsc_analyzer = GTSCAnalyzer(df)
    gtsc_results = gtsc_analyzer.run_full_analysis()
    
    # Run CAMSTRESS thermodynamic analysis
    camstress_analyzer = CAMSTRESSAnalyzer(df)
    thermo_df, thermo_metrics = camstress_analyzer.run_thermodynamic_analysis()
    
    # Identify patterns
    patterns = identify_system_patterns(gtsc_results)
    
    # Store in session state for dashboard
    st.session_state.gtsc_metrics = gtsc_results
    st.session_state.thermodynamic_metrics = thermo_metrics
    st.session_state.system_patterns = patterns
    
    return gtsc_results, thermo_metrics, patterns

if __name__ == "__main__":
    st.title("🛠️ GTSC-STSC Implementation Toolkit")
    st.markdown("*Advanced societal analysis using thermodynamic principles*")
    
    # Sample data for demonstration
    sample_data = pd.DataFrame({
        'Node': ['Executive', 'Army', 'Archive', 'Lore', 'Stewards', 'Craft', 'Flow', 'Hands'],
        'Coherence': [6.5, 7.2, 6.8, 5.9, 6.1, 7.0, 5.8, 4.9],
        'Capacity': [7.1, 8.2, 6.5, 5.8, 6.9, 7.8, 6.2, 5.5],
        'Stress': [6.8, 5.1, 4.2, 5.9, 6.5, 4.8, 7.2, 8.1],
        'Abstraction': [7.5, 4.8, 8.2, 8.9, 6.1, 5.2, 4.9, 3.8]
    })
    
    # Run analysis
    gtsc_results, thermo_metrics, patterns = integrate_with_cams_monitor(sample_data)
    
    # Display dashboard
    create_gtsc_dashboard(sample_data, gtsc_results, patterns)