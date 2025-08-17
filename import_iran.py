"""
Import Iran CAMS data from GitHub
"""

import requests
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from cams_analyzer import CAMSAnalyzer

def main():
    print('Importing Iran CAMS data from GitHub...')
    
    # GitHub URL
    url = 'https://github.com/KaliBond/wintermute/blob/main/Iran_CAMS_Cleaned.csv'
    raw_url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
    
    print(f'Downloading from: {raw_url}')
    
    try:
        # Download the file
        response = requests.get(raw_url)
        response.raise_for_status()
        
        # Save the file
        with open('Iran_CAMS_Cleaned.csv', 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print('SUCCESS: Iran data downloaded')
        
        # Test the file
        df = pd.read_csv('Iran_CAMS_Cleaned.csv')
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
                
                # Test report generation
                report = analyzer.generate_summary_report(df, "Iran")
                print(f'Civilization Type: {report["civilization_type"]}')
                print('SUCCESS: Iran data is ready for analysis!')
                
            except Exception as e:
                print(f'Analysis error: {e}')
        
        return True
        
    except Exception as e:
        print(f'ERROR: {e}')
        return False

if __name__ == "__main__":
    main()