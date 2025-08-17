"""
Debug Iraq data import issues
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from cams_analyzer import CAMSAnalyzer

# Load Iraq data
print("Loading Iraq data...")
df = pd.read_csv('Iraq_1900-2025.txt')

print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nData types:")
print(df.dtypes)

print("\nFirst few rows:")
print(df.head())

print("\nUnique values check:")
for col in ['Coherence', 'Capacity', 'Stress', 'Abstraction', 'Node Value', 'Bond Strength']:
    print(f"\n{col}:")
    unique_vals = df[col].unique()[:10]  # First 10 unique values
    print(f"  Unique values (first 10): {unique_vals}")
    print(f"  Data type: {df[col].dtype}")
    
    # Check for non-numeric values
    if df[col].dtype == 'object':
        non_numeric = df[~pd.to_numeric(df[col], errors='coerce').notna()]
        if len(non_numeric) > 0:
            print(f"  Non-numeric entries found:")
            print(non_numeric[[col]].head())

# Try to clean the data
print("\n" + "="*50)
print("CLEANING DATA")
print("="*50)

# Convert numeric columns
numeric_cols = ['Coherence', 'Capacity', 'Stress', 'Abstraction', 'Node Value', 'Bond Strength']

for col in numeric_cols:
    print(f"\nProcessing {col}...")
    original_dtype = df[col].dtype
    
    # Convert to numeric, coercing errors to NaN
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Check for NaN values
    nan_count = df[col].isna().sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values created")
        # Fill NaN with median
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  Filled NaN with median: {median_val}")
    
    print(f"  {original_dtype} -> {df[col].dtype}")

print("\nCleaned data types:")
print(df.dtypes)

# Test system health calculation
print("\n" + "="*50)
print("TESTING SYSTEM HEALTH CALCULATION")
print("="*50)

analyzer = CAMSAnalyzer()

try:
    latest_year = df['Year'].max()
    print(f"Testing with year: {latest_year}")
    
    # Test step by step
    year_data = df[df['Year'] == latest_year]
    print(f"Records for {latest_year}: {len(year_data)}")
    
    print("\nNode Value stats:")
    print(f"  Min: {year_data['Node Value'].min()}")
    print(f"  Max: {year_data['Node Value'].max()}")
    print(f"  Mean: {year_data['Node Value'].mean()}")
    
    print("\nBond Strength stats:")
    print(f"  Min: {year_data['Bond Strength'].min()}")
    print(f"  Max: {year_data['Bond Strength'].max()}")
    print(f"  Mean: {year_data['Bond Strength'].mean()}")
    
    # Calculate system health
    health = analyzer.calculate_system_health(df, latest_year)
    print(f"\nSUCCESS: System Health ({latest_year}): {health:.2f}")
    
    # Save cleaned data
    df.to_csv('Iraq_CAMS_Cleaned.csv', index=False)
    print("\nSUCCESS: Saved cleaned data as Iraq_CAMS_Cleaned.csv")
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()