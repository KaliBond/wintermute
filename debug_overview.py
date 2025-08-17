"""
Debug the overview page issue
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from cams_analyzer import CAMSAnalyzer

st.title("🔍 Debug Overview Page")

# Initialize
analyzer = CAMSAnalyzer()

# Load data
datasets = {}
try:
    au_df = analyzer.load_data('Australia_CAMS_Cleaned.csv')
    datasets['Australia'] = au_df
    st.success(f"Australia loaded: {len(au_df)} records")
except Exception as e:
    st.error(f"Australia load error: {e}")

try:
    usa_df = analyzer.load_data('USA_CAMS_Cleaned.csv')
    datasets['USA'] = usa_df  
    st.success(f"USA loaded: {len(usa_df)} records")
except Exception as e:
    st.error(f"USA load error: {e}")

if datasets:
    nation = st.selectbox("Select Nation", list(datasets.keys()))
    df = datasets[nation]
    
    st.subheader("DataFrame Info")
    st.write(f"Shape: {df.shape}")
    st.write(f"Columns: {list(df.columns)}")
    st.dataframe(df.head())
    
    st.subheader("Column Detection Test")
    try:
        for col_type in ['year', 'nation', 'node', 'coherence', 'capacity', 'stress', 'abstraction', 'node_value', 'bond_strength']:
            detected_col = analyzer._get_column_name(df, col_type)
            st.write(f"✅ {col_type} -> '{detected_col}'")
    except Exception as e:
        st.error(f"Column detection error: {e}")
    
    st.subheader("System Health Calculation Test")
    try:
        # Test with latest year
        year_col = analyzer._get_column_name(df, 'year')
        latest_year = df[year_col].max()
        st.write(f"Latest year: {latest_year}")
        
        health = analyzer.calculate_system_health(df, latest_year)
        st.write(f"✅ System Health: {health:.2f}")
        
        # Test year-by-year for last 5 years
        years = sorted(df[year_col].unique())[-5:]
        st.write("Recent health values:")
        for year in years:
            h = analyzer.calculate_system_health(df, year)
            st.write(f"  {year}: {h:.2f}")
            
    except Exception as e:
        st.error(f"System health error: {e}")
        import traceback
        st.code(traceback.format_exc())
    
    st.subheader("Full Report Generation Test")
    try:
        report = analyzer.generate_summary_report(df, nation)
        st.success("✅ Report generated!")
        
        st.write("Report contents:")
        for key, value in report.items():
            if key == 'phase_transitions':
                st.write(f"  {key}: {len(value)} transitions")
            elif key == 'recent_health_trend':
                st.write(f"  {key}: {value}")
            else:
                st.write(f"  {key}: {value}")
                
    except Exception as e:
        st.error(f"Report generation error: {e}")
        import traceback
        st.code(traceback.format_exc())
        
else:
    st.error("No datasets loaded")