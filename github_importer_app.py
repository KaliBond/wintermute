"""
Streamlit app to import CAMS data from GitHub
"""

import streamlit as st
import requests
import pandas as pd
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from cams_analyzer import CAMSAnalyzer

st.set_page_config(
    page_title="CAMS GitHub Importer",
    page_icon="📥",
    layout="wide"
)

st.title("📥 CAMS GitHub Data Importer")
st.markdown("Import CAMS data files directly from your GitHub repository")

def download_github_file(github_url):
    """Download a file from GitHub and return the content"""
    try:
        # Convert GitHub URL to raw content URL if needed
        if 'github.com' in github_url and '/blob/' in github_url:
            github_url = github_url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
        
        response = requests.get(github_url)
        response.raise_for_status()
        
        return response.text, True
        
    except Exception as e:
        return str(e), False

def process_csv_content(csv_content, filename):
    """Process CSV content and return DataFrame"""
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(csv_content))
        return df, True
    except Exception as e:
        return str(e), False

# Input section
st.header("📂 Enter GitHub File URLs")

st.markdown("""
**Supported URL formats:**
- `https://github.com/username/repo/blob/main/data.csv`
- `https://raw.githubusercontent.com/username/repo/main/data.csv`
""")

# URL inputs
urls = []
for i in range(5):  # Allow up to 5 files
    url = st.text_input(f"GitHub File URL {i+1}", key=f"url_{i}", placeholder="https://github.com/KaliBond/wintermute/blob/main/data.csv")
    if url.strip():
        urls.append(url.strip())

if st.button("🚀 Import Data Files", disabled=len(urls) == 0):
    
    if urls:
        st.header("📊 Import Results")
        
        processed_files = []
        analyzer = CAMSAnalyzer()
        
        for i, url in enumerate(urls):
            with st.expander(f"Processing File {i+1}: {url.split('/')[-1]}", expanded=True):
                
                # Download file
                st.write("📥 Downloading...")
                csv_content, download_success = download_github_file(url)
                
                if download_success:
                    st.success("✅ Download successful")
                    
                    # Process CSV
                    st.write("🔄 Processing CSV...")
                    df, process_success = process_csv_content(csv_content, url.split('/')[-1])
                    
                    if process_success:
                        st.success(f"✅ CSV processed: {df.shape[0]} rows, {df.shape[1]} columns")
                        
                        # Show data info
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Columns:**")
                            st.write(list(df.columns))
                        
                        with col2:
                            st.write("**Sample Data:**")
                            st.dataframe(df.head(3))
                        
                        # Test with CAMS analyzer
                        st.write("🔍 Testing CAMS compatibility...")
                        
                        compatible = True
                        column_mappings = {}
                        
                        for col_type in ['year', 'nation', 'node', 'coherence', 'capacity', 'stress', 'abstraction']:
                            try:
                                detected_col = analyzer._get_column_name(df, col_type)
                                column_mappings[col_type] = detected_col
                                st.write(f"✅ {col_type} → '{detected_col}'")
                            except Exception as e:
                                st.write(f"❌ {col_type} → {str(e)}")
                                compatible = False
                        
                        if compatible:
                            # Test calculations
                            try:
                                year_col = column_mappings['year']
                                latest_year = df[year_col].max()
                                health = analyzer.calculate_system_health(df, latest_year)
                                st.success(f"🎯 System Health ({latest_year}): {health:.2f}")
                                
                                # Save file
                                filename = url.split('/')[-1]
                                if not filename.endswith('.csv'):
                                    filename += '.csv'
                                
                                clean_filename = filename.replace('.csv', '_CAMS_Cleaned.csv')
                                df.to_csv(clean_filename, index=False)
                                
                                st.success(f"💾 Saved as: {clean_filename}")
                                processed_files.append(clean_filename)
                                
                            except Exception as e:
                                st.error(f"❌ Calculation test failed: {e}")
                        else:
                            st.error("❌ File not compatible with CAMS framework")
                    
                    else:
                        st.error(f"❌ CSV processing failed: {df}")
                
                else:
                    st.error(f"❌ Download failed: {csv_content}")
        
        # Summary
        if processed_files:
            st.header("🎉 Import Complete!")
            st.success(f"Successfully imported {len(processed_files)} files:")
            
            for filename in processed_files:
                st.write(f"✅ {filename}")
            
            st.info("🚀 Files are ready! Restart your dashboard to see the new data.")
            
            if st.button("🔄 Restart Dashboard"):
                st.write("Please manually restart the dashboard at your preferred port")
        
        else:
            st.error("❌ No files were successfully imported")

# Help section
st.header("❓ Help")
with st.expander("How to get GitHub file URLs"):
    st.markdown("""
    **To get the URL for your GitHub files:**
    
    1. Go to your GitHub repository
    2. Navigate to the file you want to import
    3. Click on the file to view it
    4. Copy the URL from your browser address bar
    
    **Example:**
    - Repository: `https://github.com/KaliBond/wintermute`
    - File: `data/Australia_CAMS.csv`
    - URL to paste: `https://github.com/KaliBond/wintermute/blob/main/data/Australia_CAMS.csv`
    
    The importer will automatically convert this to the raw file URL for downloading.
    """)

with st.expander("Supported data formats"):
    st.markdown("""
    **Your CSV files should contain columns for:**
    
    - **Year/Date**: Time period (1900, 2000, etc.)
    - **Nation/Country**: Nation name  
    - **Node/Component**: Societal element
    - **Coherence**: Alignment measure
    - **Capacity**: Resource measure
    - **Stress**: Pressure measure  
    - **Abstraction**: Innovation measure
    
    **Optional columns:**
    - **Node Value**: Calculated composite score
    - **Bond Strength**: Inter-node relationship strength
    
    The importer will automatically detect column name variations and test compatibility.
    """)