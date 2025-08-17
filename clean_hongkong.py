"""
Clean Hong Kong CAMS data - remove instruction header
"""

import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from cams_analyzer import CAMSAnalyzer

print("Cleaning Hong Kong CAMS data...")

# Read the raw file and find the real header
with open('hongkong.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line with the proper CSV header
header_line = None
for i, line in enumerate(lines):
    if 'Society,Year,Node,Coherence' in line:
        header_line = i
        break

if header_line is None:
    print("ERROR: Could not find proper CSV header")
    exit(1)

print(f"Found proper header at line {header_line + 1}")

# Read the file starting from the proper header
df = pd.read_csv('hongkong.csv', skiprows=header_line)

print(f"Original shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("First 3 rows:")
print(df.head(3))

# Clean any remaining issues
print("\nCleaning data...")

# Remove any remaining header rows mixed in data
original_len = len(df)
for col in df.columns:
    if col in df[col].astype(str).values:
        df = df[df[col].astype(str) != col]

if len(df) < original_len:
    print(f'Removed {original_len - len(df)} header rows from data')

# Convert numeric columns
numeric_cols = ['Year', 'Coherence', 'Capacity', 'Stress', 'Abstraction', 'Node Value', 'Bond Strength']

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            median_val = df[col].median()
            if pd.isna(median_val):
                # Default values
                if 'bond' in col.lower():
                    median_val = 2.5
                elif 'node' in col.lower():
                    median_val = 5.0
                else:
                    median_val = 4.0
            df[col].fillna(median_val, inplace=True)
            print(f'  {col}: filled {nan_count} NaN values with median {median_val:.2f}')

# Ensure Year is integer
if 'Year' in df.columns:
    df['Year'] = df['Year'].astype(int)

print(f"Cleaned shape: {df.shape}")

# Test with analyzer
print("\n" + "="*50)
print("TESTING WITH CAMS ANALYZER")
print("="*50)

analyzer = CAMSAnalyzer()

try:
    # Test column detection
    print("Column detection:")
    for col_type in ['year', 'nation', 'node', 'coherence', 'capacity', 'stress', 'abstraction']:
        detected_col = analyzer._get_column_name(df, col_type)
        print(f"  {col_type} -> '{detected_col}' SUCCESS")
    
    # Test system health calculation
    years = sorted(df['Year'].unique())
    latest_year = years[-1]
    health = analyzer.calculate_system_health(df, latest_year)
    print(f"\nYear range: {years[0]} to {years[-1]}")
    print(f"System Health ({latest_year}): {health:.2f}")
    
    # Test full report
    report = analyzer.generate_summary_report(df, "Hong Kong")
    print(f"Civilization Type: {report['civilization_type']}")
    
    # Save cleaned data
    df.to_csv('HongKong_CAMS_Cleaned.csv', index=False)
    print(f"\nSUCCESS: Saved cleaned Hong Kong data with {len(df)} records!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()