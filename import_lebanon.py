"""
Import Lebanon CAMS data from GitHub
"""

import requests
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from cams_analyzer import CAMSAnalyzer

def main():
    print('Importing Lebanon CAMS data from GitHub...')
    
    # GitHub URL
    url = 'https://github.com/KaliBond/wintermute/blob/main/lebanon.csv'
    raw_url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
    
    print(f'Downloading from: {raw_url}')
    
    try:
        # Download the file
        response = requests.get(raw_url)
        response.raise_for_status()
        
        # Save the file
        with open('lebanon.csv', 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print('SUCCESS: Lebanon data downloaded')
        
        # Test the file
        df = pd.read_csv('lebanon.csv')
        print(f'Shape: {df.shape}')
        print(f'Columns: {list(df.columns)}')
        print('First 3 rows:')
        print(df.head(3))
        
        # Clean data if needed (similar to Iraq cleaning)
        print('\nCleaning data...')
        
        # Remove any header rows mixed in data
        original_len = len(df)
        for col in df.columns:
            if col in df[col].values:
                df = df[df[col] != col]
        
        if len(df) < original_len:
            print(f'Removed {original_len - len(df)} header rows from data')
        
        # Convert numeric columns
        numeric_cols = []
        for col in df.columns:
            if col.lower() in ['year', 'coherence', 'capacity', 'stress', 'abstraction', 'node value', 'bond strength']:
                numeric_cols.append(col)
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    print(f'  {col}: filled {nan_count} NaN values with median {median_val:.2f}')
        
        print(f'Cleaned shape: {df.shape}')
        
        # Test with analyzer
        analyzer = CAMSAnalyzer()
        print('\nColumn Detection Test:')
        
        column_mappings = {}
        for col_type in ['year', 'nation', 'node', 'coherence', 'capacity', 'stress', 'abstraction']:
            try:
                detected_col = analyzer._get_column_name(df, col_type)
                column_mappings[col_type] = detected_col
                print(f'  {col_type} -> {detected_col} SUCCESS')
            except Exception as e:
                print(f'  {col_type} -> ERROR: {e}')
        
        # Test system health if columns were detected
        if 'year' in column_mappings:
            try:
                year_col = column_mappings['year']
                years = sorted(df[year_col].unique())
                latest_year = years[-1]
                health = analyzer.calculate_system_health(df, latest_year)
                print(f'\nYear range: {years[0]} to {years[-1]}')
                print(f'System Health ({latest_year}): {health:.2f}')
                
                # Test report generation
                report = analyzer.generate_summary_report(df, "Lebanon")
                print(f'Civilization Type: {report["civilization_type"]}')
                
                # Save cleaned file
                df.to_csv('Lebanon_CAMS_Cleaned.csv', index=False)
                print('SUCCESS: Saved as Lebanon_CAMS_Cleaned.csv')
                print('SUCCESS: Lebanon data is ready for analysis!')
                
            except Exception as e:
                print(f'Analysis error: {e}')
                import traceback
                traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f'ERROR: {e}')
        return False

if __name__ == "__main__":
    main()