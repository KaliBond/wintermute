"""
Comprehensive CAMS Data Import Tool
Supports GitHub URLs, local files, and various formats
"""

import pandas as pd
import requests
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from cams_analyzer import CAMSAnalyzer

class CAMSDataImporter:
    def __init__(self):
        self.analyzer = CAMSAnalyzer()
        self.imported_files = []
    
    def download_from_github(self, github_url):
        """Download file from GitHub URL"""
        print(f"Downloading: {github_url}")
        
        try:
            # Convert GitHub URL to raw content URL if needed
            if 'github.com' in github_url and '/blob/' in github_url:
                raw_url = github_url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
            else:
                raw_url = github_url
            
            response = requests.get(raw_url)
            response.raise_for_status()
            
            # Generate filename from URL
            filename = github_url.split('/')[-1]
            if not any(filename.endswith(ext) for ext in ['.csv', '.txt', '.tsv']):
                filename += '.csv'
            
            # Save file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            print(f"SUCCESS: Downloaded {filename}")
            return filename, True
            
        except Exception as e:
            print(f"ERROR downloading {github_url}: {e}")
            return None, False
    
    def clean_and_process_data(self, filename):
        """Clean and process a data file"""
        print(f"Processing: {filename}")
        
        try:
            # Try different separators
            df = None
            for sep in [',', '\t', ';', None]:
                try:
                    if sep is None:
                        df = pd.read_csv(filename, sep=None, engine='python')
                    else:
                        df = pd.read_csv(filename, sep=sep)
                    break
                except:
                    continue
            
            if df is None:
                raise Exception("Could not read file with any separator")
            
            print(f"Initial shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Clean data
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
            
            if len(df) < original_len:
                print(f"Removed {original_len - len(df)} header/invalid rows")
            
            # Convert numeric columns
            numeric_keywords = ['year', 'coherence', 'capacity', 'stress', 'abstraction', 'node value', 'bond strength']
            
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in numeric_keywords):
                    print(f"Converting {col} to numeric...")
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Fill NaN values with median
                    nan_count = df[col].isna().sum()
                    if nan_count > 0:
                        median_val = df[col].median()
                        df[col].fillna(median_val, inplace=True)
                        print(f"  Filled {nan_count} NaN values with median {median_val:.2f}")
            
            # Ensure Year is integer if found
            year_cols = [col for col in df.columns if 'year' in col.lower()]
            for year_col in year_cols:
                if df[year_col].dtype in ['float64', 'int64']:
                    df[year_col] = df[year_col].astype(int)
            
            print(f"Cleaned shape: {df.shape}")
            return df, True
            
        except Exception as e:
            print(f"ERROR processing {filename}: {e}")
            return None, False
    
    def test_cams_compatibility(self, df, country_name):
        """Test if data is compatible with CAMS framework"""
        print(f"Testing CAMS compatibility for {country_name}...")
        
        try:
            # Test column detection
            column_mappings = {}
            required_cols = ['year', 'nation', 'node', 'coherence', 'capacity', 'stress', 'abstraction']
            
            for col_type in required_cols:
                try:
                    detected_col = self.analyzer._get_column_name(df, col_type)
                    column_mappings[col_type] = detected_col
                    print(f"  {col_type} -> '{detected_col}' SUCCESS")
                except Exception as e:
                    print(f"  {col_type} -> ERROR: {e}")
                    return False, f"Missing column: {col_type}"
            
            # Test system health calculation
            year_col = column_mappings['year']
            years = sorted(df[year_col].unique())
            latest_year = years[-1]
            
            health = self.analyzer.calculate_system_health(df, latest_year)
            print(f"  System Health ({latest_year}): {health:.2f}")
            
            # Test report generation
            report = self.analyzer.generate_summary_report(df, country_name)
            print(f"  Civilization Type: {report['civilization_type']}")
            
            print(f"  Year range: {years[0]} to {years[-1]} ({len(years)} years)")
            print(f"  Records: {len(df)}")
            
            return True, "All tests passed"
            
        except Exception as e:
            return False, f"CAMS test failed: {e}"
    
    def import_file(self, source, country_name=None):
        """Import a single file from URL or local path"""
        print(f"\n{'='*60}")
        print(f"IMPORTING: {source}")
        print('='*60)
        
        # Determine if it's a URL or local file
        if source.startswith('http'):
            # Download from URL
            filename, success = self.download_from_github(source)
            if not success:
                return False
            
            # Guess country name from URL if not provided
            if country_name is None:
                url_parts = source.split('/')
                for part in url_parts:
                    if any(part.lower().startswith(country.lower()) for country in ['australia', 'usa', 'denmark', 'iraq', 'lebanon', 'france', 'germany', 'uk', 'canada', 'japan']):
                        country_name = part.split('_')[0].split('.')[0].title()
                        break
                
                if country_name is None:
                    country_name = filename.split('_')[0].split('.')[0].title()
        
        else:
            # Local file
            filename = source
            if not os.path.exists(filename):
                print(f"ERROR: File {filename} not found")
                return False
            
            if country_name is None:
                country_name = Path(filename).stem.split('_')[0].title()
        
        print(f"Country: {country_name}")
        print(f"File: {filename}")
        
        # Process the data
        df, success = self.clean_and_process_data(filename)
        if not success:
            return False
        
        # Test CAMS compatibility
        compatible, message = self.test_cams_compatibility(df, country_name)
        if not compatible:
            print(f"ERROR: {message}")
            return False
        
        # Save cleaned version
        clean_filename = f"{country_name}_CAMS_Cleaned.csv"
        df.to_csv(clean_filename, index=False)
        print(f"SUCCESS: Saved as {clean_filename}")
        
        self.imported_files.append((clean_filename, country_name, len(df)))
        return True
    
    def import_multiple(self, sources):
        """Import multiple files"""
        print("CAMS Data Import Tool")
        print("=" * 60)
        
        success_count = 0
        
        for source in sources:
            if isinstance(source, tuple):
                url, country = source
                success = self.import_file(url, country)
            else:
                success = self.import_file(source)
            
            if success:
                success_count += 1
        
        # Summary
        print(f"\n{'='*60}")
        print("IMPORT SUMMARY")
        print('='*60)
        print(f"Files processed: {len(sources)}")
        print(f"Successfully imported: {success_count}")
        print(f"Failed: {len(sources) - success_count}")
        
        if self.imported_files:
            print(f"\nImported datasets:")
            for filename, country, records in self.imported_files:
                print(f"  - {country}: {filename} ({records} records)")
            
            print(f"\nReady for dashboard! Run: streamlit run dashboard.py")
        
        return success_count

def main():
    """Interactive import session"""
    importer = CAMSDataImporter()
    
    print("CAMS Data Import Tool")
    print("=" * 60)
    print("This tool can import CAMS data from:")
    print("1. GitHub URLs (github.com/user/repo/blob/main/file.csv)")
    print("2. Raw URLs (raw.githubusercontent.com/user/repo/main/file.csv)")
    print("3. Local files (path/to/file.csv)")
    print()
    
    sources = []
    
    while True:
        print("\nEnter data source (empty line to start import):")
        source = input("URL or file path: ").strip()
        
        if not source:
            break
        
        # Optional country name
        country = input("Country name (optional): ").strip()
        
        if country:
            sources.append((source, country))
        else:
            sources.append(source)
    
    if not sources:
        print("No sources provided. Exiting.")
        return
    
    # Import all sources
    importer.import_multiple(sources)

if __name__ == "__main__":
    main()