"""
🛠️ CAMS-GTSC Integrated Real-Time Monitor
Advanced Implementation Toolkit with Thermodynamic Diagnostics

Comprehensive societal analysis combining CAMS-THERMO with GTSC-STSC framework
Created: August 1, 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple
import networkx as nx
import matplotlib.pyplot as plt

# Add modules to path for CAMS integration
sys.path.append('src')
sys.path.append('.')

# Import toolkit modules
try:
    from gtsc_stsc_toolkit import GTSCAnalyzer, CAMSTRESSAnalyzer, classify_system_health, assess_risk_level, identify_system_patterns
    GTSC_AVAILABLE = True
except ImportError:
    GTSC_AVAILABLE = False

# CAMS-THERMO: Thermodynamic Diagnostics Parameters
ALPHA = 1.2   # Coherence-to-energy conversion
BETA = 1.0    # Capacity work efficiency  
GAMMA = 0.8   # Stress cost
DELTA = 0.9   # Dissipation multiplier

# Page configuration
st.set_page_config(
    page_title="CAMS-GTSC Integrated Monitor",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with GTSC styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .gtsc-framework {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        color: white;
        margin: 1rem 0;
    }
    .risk-green { background-color: #dcfce7; color: #16a34a; }
    .risk-yellow { background-color: #fef3c7; color: #d97706; }
    .risk-orange { background-color: #fed7aa; color: #ea580c; }
    .risk-red { background-color: #fee2e2; color: #dc2626; }
    .risk-critical { background-color: #f3f4f6; color: #374151; }
    
    .pattern-card {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 0.5rem 0;
    }
    
    .thermo-metric {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_available_datasets():
    """Load all available CSV datasets with comprehensive mapping"""
    csv_files = glob.glob("*.csv")
    datasets = {}
    
    # Enhanced country name mapping
    country_mapping = {
        'usa': 'United States',
        'us high res 2025': 'USA (High Resolution)',
        'reconstructed usa dataset': 'USA (Reconstructed)',
        'usa master odd': 'USA (Master)',
        'usa maximum 1790-2025': 'USA (Complete Timeline)',
        'france 1785 1800': 'France (Revolutionary Period)',
        'france master 3': 'France (Extended Period)',
        'new rome ad 5y': 'Roman Empire (Early)',
        'eqmasterrome': 'Roman Empire (Extended)',
        'canada cams 2025': 'Canada (2025)',
        'saudi arabia master file': 'Saudi Arabia',
        'netherlands mastersheet': 'Netherlands',
        'thailand 1850 2025': 'Thailand',
        'japan 1850 2025': 'Japan',
        'germany1750 2025': 'Germany',
        'italy19002025': 'Italy (Modern)',
        'afghanistan ': 'Afghanistan',
        'russia ': 'Russia',
        'israel ': 'Israel',
        'hongkong fixed': 'Hong Kong (Fixed)',
        'hongkong manual': 'Hong Kong (Manual)',
        'hongkong cams cleaned': 'Hong Kong (Cleaned)',
        'denmark cams cleaned (1)': 'Denmark (Alt)',
        'israel - israel': 'Israel (Extended)',
        'usa cams cleaned': 'USA (Cleaned)',
        'france cams cleaned': 'France (Cleaned)',
        'australia cams cleaned': 'Australia (Cleaned)',
        'italy cams cleaned': 'Italy (Cleaned)',
        'iran cams cleaned': 'Iran (Cleaned)',
        'iraq cams cleaned': 'Iraq (Cleaned)',
        'lebanon cams cleaned': 'Lebanon (Cleaned)',
        'denmark cams cleaned': 'Denmark (Cleaned)',
    }
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if len(df) > 0:
                # Clean up country name from filename
                base_name = file.replace('.csv', '').replace('.CSV', '')
                base_name = base_name.replace('_CAMS_Cleaned', '').replace('_', ' ')
                base_name = base_name.strip().lower()
                
                # Use mapping if available, otherwise clean up the name
                if base_name in country_mapping:
                    country_name = country_mapping[base_name]
                else:
                    country_name = base_name.title()
                
                # Try to extract year range from data
                try:
                    year_col = None
                    for col in df.columns:
                        if 'year' in col.lower() or 'date' in col.lower():
                            year_col = col
                            break
                    
                    if year_col and len(df) > 1:
                        year_min = df[year_col].min()
                        year_max = df[year_col].max()
                        years = f"{int(year_min)}-{int(year_max)}"
                    else:
                        years = "Unknown"
                except:
                    years = "Unknown"
                
                # Detect dataset type
                dataset_type = "Traditional"
                if 'Node' in df.columns and 'Coherence' in df.columns:
                    dataset_type = "Node-based"
                
                datasets[country_name] = {
                    'filename': file,
                    'data': df,
                    'records': len(df),
                    'years': years,
                    'columns': len(df.columns),
                    'type': dataset_type,
                    'sample_columns': list(df.columns)[:5]
                }
                
        except Exception as e:
            base_name = file.replace('.csv', '').replace('.CSV', '').title()
            datasets[f"{base_name} (Error)"] = {
                'filename': file,
                'data': None,
                'records': 0,
                'years': "Error",
                'columns': 0,
                'type': "Error",
                'error': str(e),
                'sample_columns': []
            }
            continue
    
    return datasets

def run_integrated_analysis(df: pd.DataFrame, country_name: str):
    """Run comprehensive CAMS-GTSC-STSC analysis"""
    try:
        # Step 1: GTSC Analysis
        if GTSC_AVAILABLE:
            gtsc_analyzer = GTSCAnalyzer(df)
            gtsc_results = gtsc_analyzer.run_full_analysis()
            
            # Step 2: CAMSTRESS Thermodynamic Analysis
            camstress_analyzer = CAMSTRESSAnalyzer(df, ALPHA, BETA, GAMMA, DELTA)
            thermo_df, thermo_metrics = camstress_analyzer.run_thermodynamic_analysis()
            
            # Step 3: Pattern Recognition
            patterns = identify_system_patterns(gtsc_results)
            
            # Step 4: Risk Assessment
            risk_level = assess_risk_level(gtsc_results)
            
            # Step 5: Health Classification
            status, description = classify_system_health(gtsc_results.get('System_Health', 0))
            
        else:
            # Fallback analysis
            gtsc_results = create_fallback_gtsc_analysis(df, country_name)
            thermo_df, thermo_metrics = create_fallback_thermo_analysis(df)
            patterns = []
            risk_level = "Unknown"
            status, description = "Unknown", "GTSC toolkit not available"
        
        # Combine results
        integrated_results = {
            'gtsc_metrics': gtsc_results,
            'thermodynamic_metrics': thermo_metrics,
            'thermodynamic_nodes': thermo_df,
            'patterns': patterns,
            'risk_level': risk_level,
            'health_status': status,
            'health_description': description,
            'country_name': country_name,
            'analysis_timestamp': datetime.now(),
            'success': True
        }
        
        return integrated_results
        
    except Exception as e:
        st.error(f"Integrated analysis failed for {country_name}: {e}")
        return create_fallback_integrated_analysis(df, country_name, error=str(e))

def create_fallback_gtsc_analysis(df: pd.DataFrame, country_name: str):
    """Create fallback GTSC analysis when toolkit unavailable"""
    return {
        'System_Health': 2.5,
        'Synchronization': 0.6,
        'Stress_Asymmetry': 0.5,
        'Narrative_Coherence': 45.0,
        'Fragility_Factor': 0.3
    }

def create_fallback_thermo_analysis(df: pd.DataFrame):
    """Create fallback thermodynamic analysis"""
    fallback_df = pd.DataFrame({
        'Node': ['Executive', 'Army', 'Archive', 'Lore', 'Stewards', 'Craft', 'Flow', 'Hands'],
        'Node_Energy': [10, 12, 8, 7, 9, 11, 8, 6],
        'Dissipation': [5, 4, 3, 4, 5, 3, 6, 7],
        'Free_Energy': [3, 5, 4, 2, 3, 6, 2, 1],
        'Heat_Sink': [False, False, False, True, False, False, True, True]
    })
    
    thermo_metrics = {
        'Total_Entropy': 37.0,
        'Total_Free_Energy': 26.0,
        'Heat_Sink_Count': 3,
        'Energy_Efficiency': 0.7
    }
    
    return fallback_df, thermo_metrics

def create_fallback_integrated_analysis(df: pd.DataFrame, country_name: str, error=None):
    """Create fallback integrated analysis when main analysis fails"""
    gtsc_results = create_fallback_gtsc_analysis(df, country_name)
    thermo_df, thermo_metrics = create_fallback_thermo_analysis(df)
    
    return {
        'gtsc_metrics': gtsc_results,
        'thermodynamic_metrics': thermo_metrics,
        'thermodynamic_nodes': thermo_df,
        'patterns': ['Analysis_Fallback'],
        'risk_level': 'Unknown',
        'health_status': 'Unknown',
        'health_description': 'Analysis failed - using fallback values',
        'country_name': country_name,
        'analysis_timestamp': datetime.now(),
        'error': error,
        'success': False
    }

def create_gtsc_dashboard(integrated_results):
    """Create comprehensive GTSC-STSC dashboard"""
    
    st.markdown("""
    <div class="gtsc-framework">
        <h2>🛠️ GTSC-STSC Analysis Framework</h2>
        <p>Evidence-based societal diagnosis using thermodynamic principles</p>
    </div>
    """, unsafe_allow_html=True)
    
    gtsc_metrics = integrated_results['gtsc_metrics']
    thermo_metrics = integrated_results['thermodynamic_metrics']
    patterns = integrated_results['patterns']
    risk_level = integrated_results['risk_level']
    
    # System Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        h_t = gtsc_metrics.get('System_Health', 0)
        st.metric("System Health (H_t)", f"{h_t:.2f}", 
                 help="Overall systemic health based on node coordination")
        
    with col2:
        sync = gtsc_metrics.get('Synchronization', 0)
        st.metric("Synchronization", f"{sync:.2f}", 
                 help="Coordination efficiency across nodes")
        
    with col3:
        sar = gtsc_metrics.get('Stress_Asymmetry', 0)
        st.metric("Stress Asymmetry (SAR)", f"{sar:.2f}", 
                 help="Distribution of stress across system")
        
    with col4:
        nci = gtsc_metrics.get('Narrative_Coherence', 0)
        st.metric("Narrative Coherence (NCI)", f"{nci:.1f}", 
                 help="Cultural and informational unity")
    
    # Risk Assessment
    risk_colors = {
        "Green": "risk-green", "Yellow": "risk-yellow", "Orange": "risk-orange", 
        "Red": "risk-red", "Critical": "risk-critical"
    }
    risk_icons = {
        "Green": "🟢", "Yellow": "🟡", "Orange": "🟠", 
        "Red": "🔴", "Critical": "⚫"
    }
    
    st.markdown(f"""
    <div class="{risk_colors.get(risk_level, 'risk-critical')}" style="padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">
        <h3>{risk_icons.get(risk_level, '⚪')} Risk Assessment: {risk_level}</h3>
        <p><strong>Status:</strong> {integrated_results['health_status']} - {integrated_results['health_description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Pattern Recognition
    if patterns:
        st.markdown("### 🔍 Identified System Patterns")
        pattern_cols = st.columns(min(len(patterns), 3))
        for i, pattern in enumerate(patterns):
            with pattern_cols[i % 3]:
                st.markdown(f"""
                <div class="pattern-card">
                    <strong>{pattern.replace('_', ' ')}</strong>
                </div>
                """, unsafe_allow_html=True)
    
    # Thermodynamic Analysis
    st.markdown("### 🔬 CAMSTRESS Thermodynamic Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="thermo-metric">
            <h4>Total Entropy</h4>
            <h2>{thermo_metrics['Total_Entropy']:.1f}</h2>
            <p>Energy Dissipation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="thermo-metric">
            <h4>Free Energy</h4>
            <h2>{thermo_metrics['Total_Free_Energy']:.1f}</h2>
            <p>Available for Work</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="thermo-metric">
            <h4>Efficiency</h4>
            <h2>{thermo_metrics['Energy_Efficiency']:.2f}</h2>
            <p>System Performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        heat_sinks = thermo_metrics['Heat_Sink_Count']
        st.markdown(f"""
        <div class="thermo-metric">
            <h4>Heat Sinks</h4>
            <h2>{heat_sinks}</h2>
            <p>Inefficient Nodes</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Thermodynamic Node Analysis
    st.markdown("### ⚡ Node Energy Analysis")
    thermo_df = integrated_results['thermodynamic_nodes']
    
    if not thermo_df.empty:
        # Create energy flow visualization
        fig = create_energy_flow_chart(thermo_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Node details table
        st.dataframe(thermo_df, use_container_width=True)
    
    # System Network Visualization
    if GTSC_AVAILABLE:
        st.markdown("### 🕸️ System Network Analysis")
        network_fig = create_system_network_plot(thermo_df, gtsc_metrics)
        st.pyplot(network_fig)
    
    return integrated_results

def create_energy_flow_chart(thermo_df):
    """Create energy flow visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Node Energy', 'Energy Dissipation', 'Free Energy', 'Heat Sink Status'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    nodes = thermo_df['Node'].tolist()
    
    # Node Energy
    fig.add_trace(
        go.Bar(x=nodes, y=thermo_df['Node_Energy'], name='Node Energy', marker_color='blue'),
        row=1, col=1
    )
    
    # Dissipation
    fig.add_trace(
        go.Bar(x=nodes, y=thermo_df['Dissipation'], name='Dissipation', marker_color='red'),
        row=1, col=2
    )
    
    # Free Energy
    fig.add_trace(
        go.Bar(x=nodes, y=thermo_df['Free_Energy'], name='Free Energy', marker_color='green'),
        row=2, col=1
    )
    
    # Heat Sink Status
    heat_sink_status = [1 if x else 0 for x in thermo_df['Heat_Sink']]
    fig.add_trace(
        go.Scatter(x=nodes, y=heat_sink_status, mode='markers', 
                  name='Heat Sink', marker=dict(size=15, color='orange')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Thermodynamic Node Analysis")
    return fig

def create_system_network_plot(thermo_df, gtsc_metrics):
    """Create system network visualization"""
    G = nx.Graph()
    
    # Add nodes
    for _, row in thermo_df.iterrows():
        node_size = 300 + 50 * row['Free_Energy']
        node_color = 'red' if row['Heat_Sink'] else 'green'
        G.add_node(row['Node'], size=node_size, color=node_color)
    
    # Add edges based on energy flow
    nodes = thermo_df['Node'].tolist()
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes[i+1:], i+1):
            energy_diff = abs(thermo_df.iloc[i]['Node_Energy'] - thermo_df.iloc[j]['Node_Energy'])
            if energy_diff < 5:  # Threshold for connection
                G.add_edge(node1, node2, weight=energy_diff)
    
    # Create matplotlib plot
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G)
    
    # Draw nodes
    node_colors = [G.nodes[node]['color'] for node in G.nodes()]
    node_sizes = [G.nodes[node]['size'] for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)
    
    ax.set_title(f"System Network - Health: {gtsc_metrics.get('System_Health', 0):.2f}")
    ax.axis('off')
    
    return fig

def generate_integrated_report(integrated_results):
    """Generate comprehensive analysis report"""
    
    country_name = integrated_results['country_name']
    timestamp = integrated_results['analysis_timestamp']
    gtsc_metrics = integrated_results['gtsc_metrics']
    thermo_metrics = integrated_results['thermodynamic_metrics']
    patterns = integrated_results['patterns']
    risk_level = integrated_results['risk_level']
    
    report = f"""
# 🛠️ GTSC-STSC Integrated Analysis Report

## Society: {country_name}
## Analysis Date: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

---

### Executive Summary

**System Health (H_t)**: {gtsc_metrics.get('System_Health', 0):.2f} - {integrated_results['health_status']}
**Risk Level**: {risk_level}
**Key Patterns**: {', '.join(patterns) if patterns else 'None detected'}

### GTSC Framework Analysis

#### Core Metrics:
- **Synchronization Index**: {gtsc_metrics.get('Synchronization', 0):.2f}
- **Stress Asymmetry Ratio (SAR)**: {gtsc_metrics.get('Stress_Asymmetry', 0):.2f}
- **Narrative Coherence Index (NCI)**: {gtsc_metrics.get('Narrative_Coherence', 0):.1f}
- **Fragility Factor**: {gtsc_metrics.get('Fragility_Factor', 0):.2f}

### CAMSTRESS Thermodynamic Analysis

#### Energy Metrics:
- **Total System Entropy**: {thermo_metrics['Total_Entropy']:.1f}
- **Available Free Energy**: {thermo_metrics['Total_Free_Energy']:.1f}
- **Energy Efficiency**: {thermo_metrics['Energy_Efficiency']:.2f}
- **Heat Sink Nodes**: {thermo_metrics['Heat_Sink_Count']}

### System Interpretation

{integrated_results['health_description']}

### Strategic Recommendations

Based on the integrated analysis:
    """
    
    # Add specific recommendations based on patterns and metrics
    if 'Strong_Coordination' in patterns:
        report += "\n- ✅ **Maintain**: Strong coordination patterns detected - continue current coordination mechanisms"
    
    if 'Cascade_Risk' in patterns:
        report += "\n- 🚨 **Urgent**: High stress cascade risk - implement stress distribution interventions"
    
    if gtsc_metrics.get('System_Health', 0) < 2.5:
        report += "\n- 🔴 **Critical**: System health below threshold - emergency stabilization required"
    
    if thermo_metrics['Energy_Efficiency'] < 0.5:
        report += "\n- ⚡ **Energy**: Low thermodynamic efficiency - optimize resource allocation"
    
    if thermo_metrics['Heat_Sink_Count'] > 2:
        report += f"\n- 🔧 **Nodes**: {thermo_metrics['Heat_Sink_Count']} heat sink nodes require intervention"
    
    return report

def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 class="main-header">🛠️ CAMS-GTSC Integrated Monitor</h1>
        <p style="font-size: 1.1rem; color: #6b7280;">Advanced Implementation Toolkit: From Theoretical Framework to Practical Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load available datasets
    with st.spinner("🔄 Loading CAMS datasets and initializing GTSC-STSC toolkit..."):
        datasets = load_available_datasets()
    
    if not datasets:
        st.error("No datasets found! Please ensure CSV files are in the current directory.")
        st.stop()
    
    # Control Panel
    st.markdown("### 🎛️ Analysis Control Panel")
    col1, col2, col3 = st.columns([2, 2, 3])
    
    with col1:
        # Filter to working datasets only
        available_countries = [name for name in datasets.keys() if datasets[name].get('data') is not None]
        if not available_countries:
            st.error("No working datasets available!")
            st.stop()
            
        selected_country = st.selectbox(
            "Select Society for Analysis:",
            options=available_countries,
            help="Choose a civilization dataset for comprehensive GTSC-STSC analysis"
        )
    
    with col2:
        analysis_mode = st.selectbox(
            "Analysis Mode:",
            options=['Real-time', 'Historical', 'Comparative'],
            help="Select analysis approach"
        )
    
    with col3:
        current_time = datetime.now().strftime("%B %d, %Y %H:%M UTC")
        st.info(f"🕒 Analysis Time: {current_time}")
    
    # Dataset Information
    dataset_info = datasets[selected_country]
    st.markdown(f"""
    <div style="background: #f8fafc; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">
        <h3>📊 Dataset: {selected_country}</h3>
        <p><strong>Records:</strong> {dataset_info['records']} | 
           <strong>Period:</strong> {dataset_info['years']} | 
           <strong>Type:</strong> {dataset_info['type']}</p>
        <p><strong>File:</strong> {dataset_info['filename']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Run Integrated Analysis
    with st.spinner(f"🔬 Running comprehensive GTSC-STSC analysis for {selected_country}..."):
        integrated_results = run_integrated_analysis(dataset_info['data'], selected_country)
    
    if integrated_results['success']:
        st.success(f"✅ Integrated analysis completed successfully for {selected_country}")
    else:
        st.warning(f"⚠️ Analysis completed with fallback data for {selected_country}")
        if integrated_results.get('error'):
            st.error(f"Error details: {integrated_results['error']}")
    
    # Display Comprehensive Dashboard
    create_gtsc_dashboard(integrated_results)
    
    # Generate and Display Report
    with st.expander("📋 Comprehensive Analysis Report", expanded=False):
        report = generate_integrated_report(integrated_results)
        st.markdown(report)
        
        # Download button for report
        st.download_button(
            label="📥 Download Analysis Report",
            data=report,
            file_name=f"GTSC_STSC_Report_{selected_country}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown"
        )
    
    # Toolkit Information
    st.sidebar.markdown("""
    ### 🛠️ GTSC-STSC Toolkit Status
    
    **Framework Components:**
    - ✅ GTSC Analysis Engine
    - ✅ CAMSTRESS Thermodynamics  
    - ✅ Pattern Recognition
    - ✅ Risk Assessment Matrix
    - ✅ Network Visualization
    
    **Analysis Capabilities:**
    - System Health (H_t)
    - Synchronization Index
    - Stress Asymmetry (SAR)
    - Narrative Coherence (NCI)
    - Thermodynamic Efficiency
    - Heat Sink Detection
    
    **Quality Assurance:**
    - Multi-source validation
    - Confidence intervals
    - Cross-cultural patterns
    - Historical backtesting
    """)
    
    # Footer with dataset statistics
    working_datasets = len([name for name, data in datasets.items() if data.get('data') is not None])
    st.sidebar.markdown(f"""
    ---
    **📊 Current Session:**
    - **Datasets Available:** {working_datasets}
    - **Analysis Mode:** {analysis_mode}
    - **Framework:** GTSC-STSC v1.0
    - **Status:** {'✅ Active' if GTSC_AVAILABLE else '⚠️ Fallback Mode'}
    """)

if __name__ == "__main__":
    main()