"""
Debug version of the dashboard to identify the exact issue
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from cams_analyzer import CAMSAnalyzer
    st.success("✅ CAMS Analyzer imported successfully")
except Exception as e:
    st.error(f"❌ Error importing CAMS Analyzer: {e}")
    st.stop()

st.title("🔧 CAMS Framework Debug Dashboard")

# Initialize analyzer
analyzer = CAMSAnalyzer()
st.success("✅ Analyzer initialized")

# Try to load data
st.header("Data Loading Test")

try:
    # Load Australia data
    au_df = analyzer.load_data('Australia_CAMS_Cleaned.csv')
    if not au_df.empty:
        st.success(f"✅ Australia data loaded: {len(au_df)} records")
        st.write("Australia columns:", list(au_df.columns))
        
        # Test system health calculation
        try:
            au_health = analyzer.calculate_system_health(au_df, au_df['Year'].max())
            st.success(f"✅ Australia system health: {au_health:.2f}")
        except Exception as e:
            st.error(f"❌ Australia system health error: {e}")
        
        # Test report generation - this is where the error occurs
        try:
            st.info("🔍 Testing report generation...")
            au_report = analyzer.generate_summary_report(au_df, "Australia")
            st.success(f"✅ Australia report generated: {au_report['civilization_type']}")
        except Exception as e:
            st.error(f"❌ Australia report generation error: {e}")
            st.error(f"Error type: {type(e)}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.error("❌ Failed to load Australia data")
        
except Exception as e:
    st.error(f"❌ Error loading Australia data: {e}")

try:
    # Load USA data
    usa_df = analyzer.load_data('USA_CAMS_Cleaned.csv')
    if not usa_df.empty:
        st.success(f"✅ USA data loaded: {len(usa_df)} records")
        st.write("USA columns:", list(usa_df.columns))
        
        # Test system health calculation
        try:
            usa_health = analyzer.calculate_system_health(usa_df, usa_df['Year'].max())
            st.success(f"✅ USA system health: {usa_health:.2f}")
        except Exception as e:
            st.error(f"❌ USA system health error: {e}")
        
        # Test report generation
        try:
            st.info("🔍 Testing USA report generation...")
            usa_report = analyzer.generate_summary_report(usa_df, "USA")
            st.success(f"✅ USA report generated: {usa_report['civilization_type']}")
        except Exception as e:
            st.error(f"❌ USA report generation error: {e}")
            st.error(f"Error type: {type(e)}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.error("❌ Failed to load USA data")
        
except Exception as e:
    st.error(f"❌ Error loading USA data: {e}")

st.header("System Information")
st.write("Python version:", sys.version)
st.write("Working directory:", os.getcwd())
st.write("Files in directory:", os.listdir('.'))

# Show column mappings
st.header("Column Mappings")
st.write("Analyzer column mappings:", analyzer.column_mappings)