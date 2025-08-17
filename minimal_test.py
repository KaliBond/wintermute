"""
Minimal test to isolate the exact issue
"""

import streamlit as st
import pandas as pd
import sys
import os

# Force fresh imports
for module in ['src.cams_analyzer', 'cams_analyzer']:
    if module in sys.modules:
        del sys.modules[module]

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from cams_analyzer import CAMSAnalyzer

st.title("🔍 Minimal CAMS Test")

# Test direct file loading
st.header("Direct File Test")

try:
    # Load with pandas directly
    st.subheader("1. Direct Pandas Load")
    df_direct = pd.read_csv('Australia_CAMS_Cleaned.csv')
    st.write(f"Shape: {df_direct.shape}")
    st.write(f"Columns: {list(df_direct.columns)}")
    st.dataframe(df_direct.head())
    
    # Load with analyzer
    st.subheader("2. Analyzer Load")
    analyzer = CAMSAnalyzer()
    df_analyzer = analyzer.load_data('Australia_CAMS_Cleaned.csv')
    st.write(f"Shape: {df_analyzer.shape}")
    st.write(f"Columns: {list(df_analyzer.columns)}")
    st.dataframe(df_analyzer.head())
    
    # Test column detection
    st.subheader("3. Column Detection Test")
    try:
        year_col = analyzer._get_column_name(df_analyzer, 'year')
        st.write(f"Year column detected: '{year_col}' ✅")
    except Exception as e:
        st.error(f"Year column detection failed: {e}")
    
    # Test report generation
    st.subheader("4. Report Generation Test")
    try:
        report = analyzer.generate_summary_report(df_analyzer, "Australia")
        st.success("Report generated successfully!")
        st.json(report)
    except Exception as e:
        st.error(f"Report generation failed: {e}")
        import traceback
        st.code(traceback.format_exc())

except Exception as e:
    st.error(f"Test failed: {e}")
    import traceback
    st.code(traceback.format_exc())