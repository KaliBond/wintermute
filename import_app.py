"""
Streamlit App for CAMS Data Import
Easy-to-use interface for importing new CAMS data files
"""

import streamlit as st
import pandas as pd
import requests
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from cams_analyzer import CAMSAnalyzer

st.set_page_config(
    page_title="CAMS Data Importer",
    page_icon="📥",
    layout="wide"
)

st.title("📥 CAMS Data Import Tool")
st.markdown("Import new CAMS data files from GitHub URLs or upload local files")

# Initialize session state
if 'imported_files' not in st.session_state:
    st.session_state.imported_files = []

def download_github_file(github_url):
    """Download file from GitHub URL"""
    try:
        # Convert GitHub URL to raw content URL if needed
        if 'github.com' in github_url and '/blob/' in github_url:
            raw_url = github_url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
        else:
            raw_url = github_url
        
        response = requests.get(raw_url)
        response.raise_for_status()
        
        return response.text, True
        
    except Exception as e:
        return str(e), False

def clean_and_process_data(df, filename):
    """Clean and process dataframe"""
    try:
        original_len = len(df)
        
        # Remove header rows mixed in data
        for col in df.columns:
            if col in df[col].astype(str).values:
                df = df[df[col].astype(str) != col]
        
        # Remove rows where Year column contains 'Year'
        year_cols = [col for col in df.columns if 'year' in col.lower()]
        for year_col in year_cols:
            df = df[df[year_col].astype(str) != year_col]
            df = df[df[year_col].astype(str) != 'Year']
        
        # Convert numeric columns
        numeric_keywords = ['year', 'coherence', 'capacity', 'stress', 'abstraction', 'node value', 'bond strength']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in numeric_keywords):
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Fill NaN values with median
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
        
        # Ensure Year is integer if found
        year_cols = [col for col in df.columns if 'year' in col.lower()]
        for year_col in year_cols:
            if df[year_col].dtype in ['float64', 'int64']:
                df[year_col] = df[year_col].astype(int)
        
        return df, True, f"Cleaned {original_len - len(df)} invalid rows"
        
    except Exception as e:
        return None, False, str(e)

def test_cams_compatibility(df, country_name):
    """Test CAMS framework compatibility"""
    try:
        analyzer = CAMSAnalyzer()
        
        # Test column detection
        column_mappings = {}
        required_cols = ['year', 'nation', 'node', 'coherence', 'capacity', 'stress', 'abstraction']
        
        for col_type in required_cols:
            try:
                detected_col = analyzer._get_column_name(df, col_type)
                column_mappings[col_type] = detected_col
            except Exception:
                return False, f"Missing required column: {col_type}", None
        
        # Test calculations
        year_col = column_mappings['year']
        years = sorted(df[year_col].unique())
        latest_year = years[-1]
        
        health = analyzer.calculate_system_health(df, latest_year)
        report = analyzer.generate_summary_report(df, country_name)
        
        return True, "Compatible", {
            'health': health,
            'year_range': f"{years[0]}-{years[-1]}",
            'civilization_type': report['civilization_type'],
            'records': len(df)
        }
        
    except Exception as e:
        return False, f"Compatibility test failed: {e}", None

# Main interface
tab1, tab2, tab3 = st.tabs(["🌐 GitHub Import", "📁 File Upload", "📋 Import Status"])

with tab1:
    st.header("Import from GitHub")
    st.markdown("Paste GitHub URLs for CAMS data files")
    
    # URL inputs
    urls = []
    countries = []
    
    for i in range(5):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            url = st.text_input(
                f"GitHub URL {i+1}", 
                key=f"url_{i}",
                placeholder="https://github.com/user/repo/blob/main/data.csv"
            )
        
        with col2:
            country = st.text_input(
                f"Country {i+1}", 
                key=f"country_{i}",
                placeholder="Auto-detect"
            )
        
        if url.strip():
            urls.append(url.strip())
            countries.append(country.strip() if country.strip() else None)
    
    if st.button("🚀 Import from GitHub", disabled=len(urls) == 0):
        st.header("Import Results")
        
        for i, (url, country) in enumerate(zip(urls, countries)):
            with st.expander(f"Importing {url.split('/')[-1]}", expanded=True):
                
                # Auto-detect country if not provided
                if not country:
                    url_parts = url.split('/')
                    filename = url_parts[-1]
                    country = filename.split('_')[0].split('.')[0].title()
                
                st.write(f"**Country:** {country}")
                st.write(f"**URL:** {url}")
                
                # Download
                st.write("📥 Downloading...")
                content, download_success = download_github_file(url)
                
                if download_success:
                    st.success("✅ Download successful")
                    
                    # Parse CSV
                    st.write("📊 Processing data...")
                    try:
                        from io import StringIO
                        df = pd.read_csv(StringIO(content))
                        
                        # Show initial data info
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Shape:** {df.shape}")
                            st.write(f"**Columns:** {list(df.columns)}")
                        with col2:
                            st.dataframe(df.head(3))
                        
                        # Clean data
                        df_clean, clean_success, clean_msg = clean_and_process_data(df, url.split('/')[-1])
                        
                        if clean_success:
                            st.info(f"🧹 {clean_msg}")
                            
                            # Test compatibility
                            st.write("🔍 Testing CAMS compatibility...")
                            compatible, compat_msg, stats = test_cams_compatibility(df_clean, country)
                            
                            if compatible:
                                st.success("✅ CAMS Compatible!")
                                
                                # Show stats
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Records", stats['records'])
                                with col2:
                                    st.metric("Year Range", stats['year_range'])
                                with col3:
                                    st.metric("System Health", f"{stats['health']:.2f}")
                                
                                st.info(f"**Type:** {stats['civilization_type']}")
                                
                                # Save file
                                clean_filename = f"{country}_CAMS_Cleaned.csv"
                                df_clean.to_csv(clean_filename, index=False)
                                
                                st.success(f"💾 Saved as: {clean_filename}")
                                
                                # Add to session state
                                st.session_state.imported_files.append({
                                    'country': country,
                                    'filename': clean_filename,
                                    'records': stats['records'],
                                    'year_range': stats['year_range'],
                                    'health': stats['health'],
                                    'type': stats['civilization_type']
                                })
                                
                            else:
                                st.error(f"❌ {compat_msg}")
                        else:
                            st.error(f"❌ Data cleaning failed: {clean_msg}")
                            
                    except Exception as e:
                        st.error(f"❌ CSV parsing failed: {e}")
                
                else:
                    st.error(f"❌ Download failed: {content}")

with tab2:
    st.header("Upload Local Files")
    st.markdown("Upload CSV/TXT files from your computer")
    
    uploaded_files = st.file_uploader(
        "Choose CAMS data files",
        type=['csv', 'txt', 'tsv'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.header("Processing Uploaded Files")
        
        for file in uploaded_files:
            with st.expander(f"Processing {file.name}", expanded=True):
                
                # Auto-detect country from filename
                country = file.name.split('_')[0].split('.')[0].title()
                country = st.text_input(f"Country name for {file.name}", value=country, key=f"upload_{file.name}")
                
                try:
                    # Read file
                    df = pd.read_csv(file)
                    
                    # Show initial info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Shape:** {df.shape}")
                        st.write(f"**Columns:** {list(df.columns)}")
                    with col2:
                        st.dataframe(df.head(3))
                    
                    # Clean data
                    df_clean, clean_success, clean_msg = clean_and_process_data(df, file.name)
                    
                    if clean_success:
                        st.info(f"🧹 {clean_msg}")
                        
                        # Test compatibility
                        compatible, compat_msg, stats = test_cams_compatibility(df_clean, country)
                        
                        if compatible:
                            st.success("✅ CAMS Compatible!")
                            
                            # Show stats
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Records", stats['records'])
                            with col2:
                                st.metric("Year Range", stats['year_range'])
                            with col3:
                                st.metric("System Health", f"{stats['health']:.2f}")
                            
                            # Save file
                            clean_filename = f"{country}_CAMS_Cleaned.csv"
                            df_clean.to_csv(clean_filename, index=False)
                            
                            st.success(f"💾 Saved as: {clean_filename}")
                            
                            # Add to session state
                            st.session_state.imported_files.append({
                                'country': country,
                                'filename': clean_filename,
                                'records': stats['records'],
                                'year_range': stats['year_range'],
                                'health': stats['health'],
                                'type': stats['civilization_type']
                            })
                            
                        else:
                            st.error(f"❌ {compat_msg}")
                    else:
                        st.error(f"❌ Data cleaning failed: {clean_msg}")
                        
                except Exception as e:
                    st.error(f"❌ File processing failed: {e}")

with tab3:
    st.header("Import Status")
    
    if st.session_state.imported_files:
        st.success(f"✅ {len(st.session_state.imported_files)} files imported successfully!")
        
        # Create summary dataframe
        summary_data = []
        for file_info in st.session_state.imported_files:
            summary_data.append({
                'Country': file_info['country'],
                'Filename': file_info['filename'],
                'Records': file_info['records'],
                'Year Range': file_info['year_range'],
                'System Health': f"{file_info['health']:.2f}",
                'Civilization Type': file_info['type']
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        st.info("🚀 Files are ready for analysis! You can now run the main dashboard.")
        
        if st.button("🔄 Launch Dashboard"):
            st.write("Please run: `streamlit run dashboard.py` in your terminal")
        
        if st.button("🗑️ Clear Import History"):
            st.session_state.imported_files = []
            st.rerun()
    
    else:
        st.info("No files imported yet. Use the tabs above to import data.")

# Help section
st.markdown("---")
with st.expander("📖 Help & Instructions"):
    st.markdown("""
    **How to use this import tool:**
    
    1. **GitHub Import**: Paste GitHub URLs for CAMS data files
       - Supports both blob URLs and raw URLs
       - Automatically detects country names from filenames
       - Downloads and processes files automatically
    
    2. **File Upload**: Upload CSV/TXT files from your computer
       - Drag and drop or browse for files
       - Supports multiple file upload
       - Specify country names for each file
    
    **Required data format:**
    Your files should contain these columns (names may vary):
    - Year/Date
    - Nation/Country/Society
    - Node/Component
    - Coherence
    - Capacity  
    - Stress
    - Abstraction
    - Node Value (optional)
    - Bond Strength (optional)
    
    **The tool will:**
    - Automatically detect column variations
    - Clean and validate data
    - Test CAMS framework compatibility
    - Save cleaned files for dashboard use
    """)