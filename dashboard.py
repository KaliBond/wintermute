"""
CAMS Framework Interactive Dashboard
Streamlit dashboard for Complex Adaptive Model State framework analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Clear Python cache and force reload
if hasattr(sys, '_clear_type_cache'):
    sys._clear_type_cache()

# Force reload modules if already imported
module_names = ['src.cams_analyzer', 'src.visualizations', 'cams_analyzer', 'visualizations']
for module_name in module_names:
    if module_name in sys.modules:
        del sys.modules[module_name]

from src.cams_analyzer import CAMSAnalyzer
from src.visualizations import CAMSVisualizer
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="CAMS Framework Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize classes with debugging
def get_analyzer():
    try:
        analyzer = CAMSAnalyzer()
        st.sidebar.success("✅ Analyzer initialized successfully")
        return analyzer
    except Exception as e:
        st.sidebar.error(f"❌ Analyzer initialization failed: {e}")
        return None

def get_visualizer():
    try:
        visualizer = CAMSVisualizer()
        st.sidebar.success("✅ Visualizer initialized successfully")
        return visualizer
    except Exception as e:
        st.sidebar.error(f"❌ Visualizer initialization failed: {e}")
        return None

analyzer = get_analyzer()
visualizer = get_visualizer()

if analyzer is None or visualizer is None:
    st.error("Failed to initialize components. Please refresh the page.")
    st.stop()

# Title and description
st.title("🏛️ CAMS Framework Analysis Dashboard")
st.markdown("""
**Complex Adaptive Model State (CAMS) Framework** - Analyzing societies as Complex Adaptive Systems
Developed by Kari McKern | Quantifying Coherence, Capacity, Stress, and Abstraction across societal nodes
""")

# Sidebar for data upload and configuration
st.sidebar.header("📊 Data Configuration")

# File upload
uploaded_files = st.sidebar.file_uploader(
    "Upload CAMS Data Files (CSV)",
    type=['csv'],
    accept_multiple_files=True,
    help="Upload your CAMS framework CSV files containing nation data"
)

# Load default data if no files uploaded
def load_default_data():
    """Load default datasets"""
    datasets = {}
    analyzer = CAMSAnalyzer()
    
    # All available CAMS datasets
    available_datasets = [
        ('Australia_CAMS_Cleaned.csv', 'Australia'),
        ('USA_CAMS_Cleaned.csv', 'USA'),
        ('Denmark_CAMS_Cleaned.csv', 'Denmark'),
        ('Iraq_CAMS_Cleaned.csv', 'Iraq'),
        ('Lebanon_CAMS_Cleaned.csv', 'Lebanon'),
        ('Iran_CAMS_Cleaned.csv', 'Iran'),
        ('France_CAMS_Cleaned.csv', 'France'),
        ('HongKong_CAMS_Cleaned.csv', 'Hong Kong')
    ]
    
    loaded_count = 0
    for filename, country_name in available_datasets:
        try:
            if os.path.exists(filename):
                df = analyzer.load_data(filename)
                if not df.empty:
                    datasets[country_name] = df
                    loaded_count += 1
                    st.sidebar.success(f"✅ Loaded {country_name}: {len(df)} records")
                else:
                    st.sidebar.warning(f"⚠️ {country_name}: Empty dataset")
            else:
                st.sidebar.info(f"ℹ️ {country_name}: File not found")
        except Exception as e:
            st.sidebar.error(f"❌ {country_name}: {str(e)}")
    
    if loaded_count > 0:
        st.sidebar.success(f"🎉 Loaded {loaded_count} countries successfully!")
    else:
        st.warning("No default data files found. Please upload your own data.")
    
    return datasets

# Process uploaded files or use defaults
if uploaded_files:
    datasets = {}
    analyzer_for_upload = CAMSAnalyzer()
    
    for file in uploaded_files:
        try:
            # Save uploaded file temporarily and load with analyzer
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='w+b', suffix='.csv', delete=False) as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name
            
            df = analyzer_for_upload.load_data(tmp_path)
            filename = file.name.replace('.csv', '').replace('_CAMS_Cleaned', '')
            
            if not df.empty:
                datasets[filename] = df
                st.sidebar.success(f"Loaded {filename}: {len(df)} records")
            else:
                st.sidebar.error(f"Failed to load {filename}")
            
            # Clean up temp file
            os.unlink(tmp_path)
            
        except Exception as e:
            st.sidebar.error(f"Error loading {file.name}: {e}")
else:
    datasets = load_default_data()

if not datasets:
    st.error("No data available. Please upload CAMS CSV files.")
    st.stop()

# Nation selection
selected_nations = st.sidebar.multiselect(
    "Select Nations to Analyze",
    options=list(datasets.keys()),
    default=list(datasets.keys())[:2] if len(datasets) >= 2 else list(datasets.keys()),
    help="Choose which nations to include in the analysis"
)

if not selected_nations:
    st.error("Please select at least one nation to analyze.")
    st.stop()

# Analysis type selection
analysis_type = st.sidebar.selectbox(
    "Analysis Type",
    ["Overview Dashboard", "Comparative Analysis", "Deep Dive", "Phase Transitions"],
    help="Choose the type of analysis to perform"
)

# Main dashboard content
if analysis_type == "Overview Dashboard":
    
    # Create tabs for each nation
    if len(selected_nations) == 1:
        nation = selected_nations[0]
        df = datasets[nation]
        
        # Generate summary report
        report = analyzer.generate_summary_report(df, nation)
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current System Health", 
                f"{report['current_health']:.2f}",
                help="H(t) = N(t)/D(t) * (1 - P(t))"
            )
        
        with col2:
            st.metric(
                "Civilization Type",
                report['civilization_type'].split(':')[0],
                help="Classification based on system health and bond strength"
            )
        
        with col3:
            st.metric(
                "Health Trajectory",
                report['health_trajectory'],
                delta=f"{report['recent_health_trend'][-1] - report['recent_health_trend'][0]:.2f}" if len(report['recent_health_trend']) > 1 else None
            )
        
        with col4:
            stress_esd = report['stress_analysis']['esd']
            st.metric(
                "Stress Distribution (ESD)",
                f"{stress_esd:.3f}",
                help="Evenness of Stress Distribution - lower values indicate more uneven stress"
            )
        
        # Visualizations
        st.header("📈 System Health Timeline")
        health_fig = visualizer.plot_system_health_timeline(df, nation)
        st.plotly_chart(health_fig, use_container_width=True)
        
        # Two column layout for additional charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Four Dimensions Profile")
            radar_fig = visualizer.plot_four_dimensions_radar(df, nation=nation)
            st.plotly_chart(radar_fig, use_container_width=True)
        
        with col2:
            st.subheader("🔗 Node Network")
            network_fig = visualizer.plot_node_network(df, nation=nation)
            st.plotly_chart(network_fig, use_container_width=True)
        
        # Heatmaps
        st.header("🌡️ Node Analysis Heatmaps")
        
        heatmap_metric = st.selectbox(
            "Select Metric for Heatmap",
            ["Coherence", "Capacity", "Stress", "Abstraction", "Node value", "Bond strength"]
        )
        
        heatmap_fig = visualizer.plot_node_heatmap(df, heatmap_metric, nation)
        st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Stress distribution
        st.header("⚡ Current Stress Distribution")
        stress_fig = visualizer.plot_stress_distribution(df, nation=nation)
        st.plotly_chart(stress_fig, use_container_width=True)
        
    else:
        # Multi-nation tabs
        tabs = st.tabs(selected_nations)
        
        for i, nation in enumerate(selected_nations):
            with tabs[i]:
                df = datasets[nation]
                report = analyzer.generate_summary_report(df, nation)
                
                # Key metrics for this nation
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("System Health", f"{report['current_health']:.2f}")
                with col2:
                    st.metric("Type", report['civilization_type'].split(':')[0])
                with col3:
                    st.metric("Trajectory", report['health_trajectory'])
                
                # Health timeline
                health_fig = visualizer.plot_system_health_timeline(df, nation)
                st.plotly_chart(health_fig, use_container_width=True)

elif analysis_type == "Comparative Analysis":
    st.header("🔄 Comparative Analysis")
    
    if len(selected_nations) < 2:
        st.error("Please select at least 2 nations for comparative analysis.")
    else:
        # Comparison metrics
        comparison_data = []
        
        for nation in selected_nations:
            df = datasets[nation]
            report = analyzer.generate_summary_report(df, nation)
            comparison_data.append({
                'Nation': nation,
                'System Health': report['current_health'],
                'Civilization Type': report['civilization_type'],
                'Health Trajectory': report['health_trajectory'],
                'Time Period': report['time_period'],
                'Phase Transitions': len(report['phase_transitions'])
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Comparative health timeline
        fig = go.Figure()
        
        for nation in selected_nations:
            df = datasets[nation]
            year_col = analyzer._get_column_name(df, 'year')
            years = sorted(df[year_col].unique())
            health_values = []
            
            for year in years:
                health = analyzer.calculate_system_health(df, year)
                health_values.append(health)
            
            fig.add_trace(go.Scatter(
                x=years,
                y=health_values,
                mode='lines+markers',
                name=nation,
                line=dict(width=3),
                marker=dict(size=6)
            ))
        
        fig.add_hline(y=2.3, line_dash="dash", line_color="red")
        fig.add_hline(y=2.5, line_dash="dash", line_color="orange") 
        fig.add_hline(y=5.0, line_dash="dash", line_color="yellow")
        
        fig.update_layout(
            title="Comparative System Health Timeline",
            xaxis_title="Year",
            yaxis_title="System Health H(t)",
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Deep Dive":
    st.header("🔍 Deep Dive Analysis")
    
    nation = st.selectbox("Select Nation for Deep Dive", selected_nations)
    df = datasets[nation]
    
    # Year range selector
    year_col = analyzer._get_column_name(df, 'year')
    years = sorted(df[year_col].unique())
    year_range = st.slider(
        "Select Year Range",
        min_value=min(years),
        max_value=max(years),
        value=(min(years), max(years))
    )
    
    # Filter data
    filtered_df = df[(df[year_col] >= year_range[0]) & (df[year_col] <= year_range[1])]
    
    # Deep analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Node Performance Over Time")
        node = st.selectbox("Select Node", analyzer.nodes)
        node_col = analyzer._get_column_name(filtered_df, 'node')
        node_data = filtered_df[filtered_df[node_col] == node]
        
        fig = go.Figure()
        
        # Get column names dynamically
        year_col = analyzer._get_column_name(node_data, 'year')
        coherence_col = analyzer._get_column_name(node_data, 'coherence')
        capacity_col = analyzer._get_column_name(node_data, 'capacity')
        abstraction_col = analyzer._get_column_name(node_data, 'abstraction')
        stress_col = analyzer._get_column_name(node_data, 'stress')
        
        for metric_name, col_name in [('Coherence', coherence_col), ('Capacity', capacity_col), ('Abstraction', abstraction_col)]:
            fig.add_trace(go.Scatter(
                x=node_data[year_col],
                y=node_data[col_name],
                mode='lines+markers',
                name=metric_name
            ))
        
        # Stress on secondary y-axis
        fig.add_trace(go.Scatter(
            x=node_data[year_col],
            y=np.abs(node_data[stress_col]),
            mode='lines+markers',
            name='Stress (abs)',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title=f"{node} Performance Metrics",
            xaxis_title="Year",
            yaxis_title="Metric Value",
            yaxis2=dict(title="Stress Level", overlaying='y', side='right'),
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("System Health Decomposition")
        
        # Calculate health components for selected year
        selected_year = st.selectbox("Select Year for Analysis", years)
        year_col = analyzer._get_column_name(filtered_df, 'year')
        year_data = filtered_df[filtered_df[year_col] == selected_year]
        
        if not year_data.empty:
            # Calculate components using dynamic column detection
            node_value_col = analyzer._get_column_name(year_data, 'node_value')
            bond_strength_col = analyzer._get_column_name(year_data, 'bond_strength')
            stress_col = analyzer._get_column_name(year_data, 'stress')
            abstraction_col = analyzer._get_column_name(year_data, 'abstraction')
            
            node_values = year_data[node_value_col].values
            bond_strengths = year_data[bond_strength_col].values
            
            # Convert to numeric, handling any non-numeric values
            node_values = pd.to_numeric(node_values, errors='coerce')
            bond_strengths = pd.to_numeric(bond_strengths, errors='coerce')
            
            # Remove NaN values
            valid_mask = ~(np.isnan(node_values) | np.isnan(bond_strengths))
            node_values = node_values[valid_mask]
            bond_strengths = bond_strengths[valid_mask]
            
            n_t = np.sum(node_values * bond_strengths) / len(node_values) if len(node_values) > 0 else 0
            
            stress_values = np.abs(pd.to_numeric(year_data[stress_col].values, errors='coerce'))
            abstraction_values = pd.to_numeric(year_data[abstraction_col].values, errors='coerce')
            stress_values = stress_values[~np.isnan(stress_values)]
            abstraction_values = abstraction_values[~np.isnan(abstraction_values)]
            d_t = 1 + np.mean(stress_values) * np.std(abstraction_values) if len(stress_values) > 0 and len(abstraction_values) > 0 else 1
            
            coherence_col = analyzer._get_column_name(year_data, 'coherence')
            coherence_values = pd.to_numeric(year_data[coherence_col].values, errors='coerce')
            coherence_values = coherence_values[~np.isnan(coherence_values)]
            coherence_asymmetry = np.std(coherence_values) / (np.mean(coherence_values) + 1e-6) if len(coherence_values) > 0 else 0
            p_t = min(coherence_asymmetry / 10, 0.9)
            
            h_t = (n_t / d_t) * (1 - p_t)
            
            st.metric("N(t) - Node Fitness", f"{n_t:.2f}")
            st.metric("D(t) - Stress-Abstraction Penalty", f"{d_t:.2f}")
            st.metric("P(t) - Polarization Penalty", f"{p_t:.3f}")
            st.metric("H(t) - System Health", f"{h_t:.2f}")

elif analysis_type == "Phase Transitions":
    st.header("⚠️ Phase Transition Analysis")
    
    for nation in selected_nations:
        st.subheader(f"📍 {nation}")
        df = datasets[nation]
        transitions = analyzer.detect_phase_transitions(df, nation)
        
        if transitions:
            transition_df = pd.DataFrame(transitions)
            
            # Color code by severity
            def color_severity(val):
                if val == 'Critical':
                    return 'background-color: #ff4444'
                elif val == 'High': 
                    return 'background-color: #ff8800'
                elif val == 'Medium':
                    return 'background-color: #ffbb00'
                return ''
            
            styled_df = transition_df.style.applymap(color_severity, subset=['severity'])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.success("No significant phase transitions detected.")

# Footer
st.markdown("---")
st.markdown("""
**CAMS Framework Dashboard** | Developed based on Kari McKern's Complex Adaptive Model State framework  
📧 Contact: kari.freyr.4@gmail.com | 📚 Learn more: [Pearls and Irritations](https://johnmenadue.com/)
""")