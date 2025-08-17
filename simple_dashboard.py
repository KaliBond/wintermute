"""
Simple CAMS Framework Dashboard - No Caching
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Force module reload
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import after path setup
from cams_analyzer import CAMSAnalyzer
from visualizations import CAMSVisualizer

# Configure page
st.set_page_config(
    page_title="CAMS Framework Dashboard",
    page_icon="🏛️",
    layout="wide"
)

st.title("🏛️ CAMS Framework Analysis Dashboard")
st.markdown("**Complex Adaptive Model State (CAMS) Framework** - Analyzing societies as Complex Adaptive Systems")

# Initialize (no caching)
try:
    analyzer = CAMSAnalyzer()
    visualizer = CAMSVisualizer()
    st.sidebar.success("✅ Components initialized successfully")
except Exception as e:
    st.error(f"❌ Initialization failed: {e}")
    st.stop()

# Load data directly
st.sidebar.header("📊 Data Status")

datasets = {}

# Try to load Australia data
try:
    au_df = analyzer.load_data('Australia_CAMS_Cleaned.csv')
    if not au_df.empty:
        datasets['Australia'] = au_df
        st.sidebar.success(f"✅ Australia: {len(au_df)} records")
    else:
        st.sidebar.warning("⚠️ Australia data empty")
except Exception as e:
    st.sidebar.error(f"❌ Australia load error: {e}")

# Try to load USA data
try:
    usa_df = analyzer.load_data('USA_CAMS_Cleaned.csv')
    if not usa_df.empty:
        datasets['USA'] = usa_df
        st.sidebar.success(f"✅ USA: {len(usa_df)} records")
    else:
        st.sidebar.warning("⚠️ USA data empty")
except Exception as e:
    st.sidebar.error(f"❌ USA load error: {e}")

if not datasets:
    st.error("No data available. Please ensure CSV files are in the directory.")
    st.stop()

# Nation selection
selected_nation = st.sidebar.selectbox(
    "Select Nation to Analyze",
    options=list(datasets.keys()),
    index=0
)

df = datasets[selected_nation]

# Analysis section
st.header(f"📊 {selected_nation} Analysis")

# Generate report with error handling
try:
    st.info("🔍 Generating analysis report...")
    
    # Debug info
    st.write("**Debug Info:**")
    st.write(f"- Dataframe shape: {df.shape}")
    st.write(f"- Columns: {list(df.columns)}")
    
    # Test individual components
    st.write("**Component Tests:**")
    
    # Test system health
    try:
        latest_year = df['Year'].max()
        health = analyzer.calculate_system_health(df, latest_year)
        st.write(f"- System Health: {health:.2f} ✅")
    except Exception as e:
        st.write(f"- System Health: Error - {e}")
    
    # Test civilization type
    try:
        civ_type = analyzer.analyze_civilization_type(df, selected_nation)
        st.write(f"- Civilization Type: {civ_type} ✅")
    except Exception as e:
        st.write(f"- Civilization Type: Error - {e}")
    
    # Full report generation
    st.write("**Full Report Generation:**")
    report = analyzer.generate_summary_report(df, selected_nation)
    st.success("✅ Report generated successfully!")
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("System Health", f"{report['current_health']:.2f}")
    
    with col2:
        st.metric("Civilization Type", report['civilization_type'].split(':')[0])
    
    with col3:
        st.metric("Health Trajectory", report['health_trajectory'])
    
    with col4:
        st.metric("Time Period", report['time_period'])

    # Visualizations
    st.header("📈 Visualizations")
    
    try:
        # System health timeline
        health_fig = visualizer.plot_system_health_timeline(df, selected_nation)
        st.plotly_chart(health_fig, use_container_width=True)
        
        # Two column layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Four Dimensions Profile")
            radar_fig = visualizer.plot_four_dimensions_radar(df, nation=selected_nation)
            st.plotly_chart(radar_fig, use_container_width=True)
        
        with col2:
            st.subheader("⚡ Stress Distribution")
            stress_fig = visualizer.plot_stress_distribution(df, nation=selected_nation)
            st.plotly_chart(stress_fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Visualization error: {e}")

except Exception as e:
    st.error(f"❌ Report generation failed: {e}")
    st.error(f"Error type: {type(e)}")
    
    # Show full traceback
    import traceback
    st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("**CAMS Framework Dashboard** | Simplified Version | No Caching")