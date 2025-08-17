"""
Import CAMS data files from GitHub repository
"""

import requests
import pandas as pd
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from cams_analyzer import CAMSAnalyzer

def download_github_file(github_url, local_filename):
    """Download a file from GitHub"""
    print(f"Downloading: {github_url}")
    
    try:
        # Convert GitHub URL to raw content URL if needed
        if 'github.com' in github_url and '/blob/' in github_url:
            github_url = github_url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
        
        response = requests.get(github_url)
        response.raise_for_status()
        
        with open(local_filename, 'wb') as f:
            f.write(response.content)
        
        print(f"SUCCESS: Downloaded {local_filename}")
        return True
        
    except Exception as e:
        print(f"ERROR downloading {github_url}: {e}")
        return False

def process_github_csv(local_filename):
    """Process and clean a downloaded CSV file"""
    print(f"\nProcessing: {local_filename}")
    
    try:
        # Try to read the file
        df = pd.read_csv(local_filename)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Show first few rows
        print("First 3 rows:")
        print(df.head(3))
        
        # Test with analyzer
        analyzer = CAMSAnalyzer()
        
        # Test column detection
        print("\nColumn Detection:")
        column_mappings = {}
        for col_type in ['year', 'nation', 'node', 'coherence', 'capacity', 'stress', 'abstraction']:
            try:
                detected_col = analyzer._get_column_name(df, col_type)
                column_mappings[col_type] = detected_col
                print(f"  {col_type} -> '{detected_col}' SUCCESS")
            except Exception as e:
                print(f"  {col_type} -> ERROR: {e}")
        
        # Test basic calculations
        if 'year' in column_mappings:
            year_col = column_mappings['year']
            years = sorted(df[year_col].unique())
            print(f"\nYear range: {years[0]} to {years[-1]} ({len(years)} years)")
            
            # Test system health calculation
            try:
                latest_year = years[-1]
                health = analyzer.calculate_system_health(df, latest_year)
                print(f"Latest system health ({latest_year}): {health:.2f}")
            except Exception as e:
                print(f"System health calculation error: {e}")
        
        return df, True
        
    except Exception as e:
        print(f"ERROR processing {local_filename}: {e}")
        return None, False

def main():
    print("CAMS GitHub Data Importer")
    print("=" * 40)
    
    # Ask for GitHub repository information
    print("\nPlease provide your GitHub data files:")
    print("You can provide:")
    print("1. Direct file URLs (raw.githubusercontent.com links)")
    print("2. GitHub blob URLs (github.com/user/repo/blob/main/file.csv)")
    print("3. Repository URL and I'll help find the files")
    print()
    
    # Get user input for files
    files_to_download = []
    
    print("Enter file URLs (one per line, empty line to finish):")
    while True:
        url = input("GitHub file URL: ").strip()
        if not url:
            break
        
        # Generate local filename from URL
        filename = url.split('/')[-1]
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        files_to_download.append((url, filename))
    
    if not files_to_download:
        print("No files specified. Exiting.")
        return
    
    # Download and process files
    processed_files = []
    
    for github_url, local_filename in files_to_download:
        print(f"\n" + "="*50)
        
        # Download file
        if download_github_file(github_url, local_filename):
            # Process file
            df, success = process_github_csv(local_filename)
            
            if success and df is not None:
                processed_files.append((local_filename, df))
                
                # Create clean version
                clean_filename = local_filename.replace('.csv', '_CAMS_Cleaned.csv')
                df.to_csv(clean_filename, index=False)
                print(f"Saved clean version: {clean_filename}")
    
    # Summary
    print(f"\n" + "="*50)
    print("IMPORT SUMMARY")
    print("-" * 20)
    print(f"Files processed: {len(processed_files)}")
    
    for filename, df in processed_files:
        print(f"✅ {filename}: {df.shape[0]} records, {df.shape[1]} columns")
    
    if processed_files:
        print(f"\n🚀 Ready for dashboard!")
        print("Files are now available in the wintermute directory.")
        print("Restart your dashboard to see the new data.")
    else:
        print(f"\n❌ No files were successfully processed.")

if __name__ == "__main__":
    main()