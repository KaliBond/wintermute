"""
Debug script to check column issues
"""

import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cams_analyzer import CAMSAnalyzer

print("COLUMN DEBUG ANALYSIS")
print("=" * 50)

analyzer = CAMSAnalyzer()

# Check both data files
files_to_check = ['Australia_CAMS_Cleaned.csv', 'USA_CAMS_Cleaned.csv']

for filename in files_to_check:
    print(f"\nCHECKING: {filename}")
    print("-" * 30)
    
    try:
        # Load using pandas directly first
        df_direct = pd.read_csv(filename)
        print(f"Direct pandas load successful: {df_direct.shape}")
        print(f"Direct columns: {list(df_direct.columns)}")
        
        # Load using analyzer
        df_analyzer = analyzer.load_data(filename)
        print(f"Analyzer load successful: {df_analyzer.shape}")
        print(f"Analyzer columns: {list(df_analyzer.columns)}")
        
        # Test column detection
        print("\nColumn Detection Test:")
        for col_type in ['year', 'nation', 'node', 'coherence', 'capacity', 'stress', 'abstraction']:
            try:
                detected_col = analyzer._get_column_name(df_analyzer, col_type)
                print(f"  {col_type} -> {detected_col} SUCCESS")
            except KeyError as e:
                print(f"  {col_type} -> ERROR: {e}")
        
        # Test specific operations that are failing
        print("\nOperation Tests:")
        try:
            year_col = analyzer._get_column_name(df_analyzer, 'year')
            min_year = df_analyzer[year_col].min()
            max_year = df_analyzer[year_col].max()
            print(f"  Year range: {min_year}-{max_year} SUCCESS")
        except Exception as e:
            print(f"  Year range: ERROR - {e}")
        
        try:
            health = analyzer.calculate_system_health(df_analyzer, df_analyzer['Year'].max() if 'Year' in df_analyzer.columns else 2024)
            print(f"  System health: {health:.2f} SUCCESS")
        except Exception as e:
            print(f"  System health: ERROR - {e}")
            
    except Exception as e:
        print(f"ERROR loading {filename}: {e}")

print(f"\nANALYZER COLUMN MAPPINGS:")
print("-" * 30)
for key, values in analyzer.column_mappings.items():
    print(f"{key}: {values}")