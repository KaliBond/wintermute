"""
Import Iraq CAMS data from GitHub
"""

import requests
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from cams_analyzer import CAMSAnalyzer

def main():
    print('Importing Iraq CAMS data from GitHub...')
    
    # GitHub URL for Iraq data
    url = 'https://github.com/KaliBond/wintermute/blob/main/Iraq%201900-2025%20.txt'
    raw_url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
    
    print(f'Downloading from: {raw_url}')
    
    try:
        # Download the file
        response = requests.get(raw_url)
        response.raise_for_status()
        
        # Save the file
        with open('Iraq_1900-2025.txt', 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print('SUCCESS: Iraq data downloaded')
        
        # Try to read as CSV (might be tab-separated or comma-separated)
        try:
            # First try comma-separated
            df = pd.read_csv('Iraq_1900-2025.txt')
            print('SUCCESS: Read as CSV')
        except:
            try:
                # Try tab-separated
                df = pd.read_csv('Iraq_1900-2025.txt', sep='\t')
                print('SUCCESS: Read as TSV (tab-separated)')
            except:
                # Try space-separated or other delimiters
                df = pd.read_csv('Iraq_1900-2025.txt', sep=None, engine='python')
                print('SUCCESS: Read with auto-detected separator')
        
        print(f'Shape: {df.shape}')
        print(f'Columns: {list(df.columns)}')
        print('First 3 rows:')
        print(df.head(3))
        
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
                
                # Save as CSV
                df.to_csv('Iraq_CAMS_Cleaned.csv', index=False)
                print('SUCCESS: Saved as Iraq_CAMS_Cleaned.csv')
                
                # Test report generation
                report = analyzer.generate_summary_report(df, "Iraq")
                print(f'Civilization Type: {report["civilization_type"]}')
                print('SUCCESS: Iraq data is ready for analysis!')
                
            except Exception as e:
                print(f'Analysis error: {e}')
        
        return True
        
    except Exception as e:
        print(f'ERROR: {e}')
        return False

if __name__ == "__main__":
    main()