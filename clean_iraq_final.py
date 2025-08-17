"""
Clean Iraq CAMS data properly
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from cams_analyzer import CAMSAnalyzer

print("Cleaning Iraq CAMS data...")

# Load the raw data
df = pd.read_csv('Iraq_1900-2025.txt')

print(f"Original shape: {df.shape}")

# Remove rows where Year is 'Year' (header rows mixed in data)
print("Removing header rows mixed in data...")
df = df[df['Year'] != 'Year']

# Remove rows where key columns contain their column names
for col in ['Coherence', 'Capacity', 'Stress', 'Abstraction', 'Node Value', 'Bond Strength']:
    df = df[df[col] != col]

print(f"After removing header rows: {df.shape}")

# Convert Year to numeric
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Remove rows with NaN Year (these were probably corrupted)
df = df.dropna(subset=['Year'])

print(f"After cleaning Year column: {df.shape}")

# Convert numeric columns
numeric_cols = ['Coherence', 'Capacity', 'Stress', 'Abstraction', 'Node Value', 'Bond Strength']

print("Converting numeric columns...")
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill any remaining NaN with column median
    nan_count = df[col].isna().sum()
    if nan_count > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  {col}: filled {nan_count} NaN values with median {median_val:.2f}")

# Ensure Year is integer
df['Year'] = df['Year'].astype(int)

print(f"\nFinal cleaned shape: {df.shape}")
print("Data types:")
print(df.dtypes)

print(f"\nYear range: {df['Year'].min()} to {df['Year'].max()}")
print(f"Countries: {df['Society'].unique()}")
print(f"Nodes: {df['Node'].unique()}")

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
    latest_year = df['Year'].max()
    health = analyzer.calculate_system_health(df, latest_year)
    print(f"\nSystem Health ({latest_year}): {health:.2f}")
    
    # Test full report
    report = analyzer.generate_summary_report(df, "Iraq")
    print(f"Civilization Type: {report['civilization_type']}")
    
    # Save cleaned data
    df.to_csv('Iraq_CAMS_Cleaned.csv', index=False)
    print(f"\nSUCCESS: Saved cleaned Iraq data with {len(df)} records!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()