"""
Clean Iran CAMS data - fix NaN values
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from cams_analyzer import CAMSAnalyzer

print("Cleaning Iran CAMS data...")

# Load the data
df = pd.read_csv('Iran_CAMS_Cleaned.csv')

print(f"Original shape: {df.shape}")
print("Data types:")
print(df.dtypes)

# Check for NaN values
print("\nNaN values per column:")
for col in df.columns:
    nan_count = df[col].isna().sum()
    if nan_count > 0:
        print(f"  {col}: {nan_count} NaN values")

# Fix NaN values in numeric columns
numeric_cols = ['Coherence', 'Capacity', 'Stress', 'Abstraction', 'Node value', 'Bond strength']

for col in numeric_cols:
    if col in df.columns:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            # Calculate median from non-NaN values
            median_val = df[col].median()
            if pd.isna(median_val):
                # If all values are NaN, use a default based on column type
                if 'bond' in col.lower():
                    median_val = 2.5  # Typical bond strength
                elif 'node' in col.lower():
                    median_val = 5.0  # Typical node value
                else:
                    median_val = 4.0  # Default for other metrics
            
            df[col].fillna(median_val, inplace=True)
            print(f"Filled {nan_count} NaN values in {col} with {median_val:.2f}")

print(f"\nCleaned shape: {df.shape}")

# Verify data types
print("\nFinal data types:")
print(df.dtypes)

# Test with analyzer
print("\n" + "="*50)
print("TESTING WITH CAMS ANALYZER")
print("="*50)

analyzer = CAMSAnalyzer()

try:
    # Test system health calculation
    latest_year = df['Year'].max()
    health = analyzer.calculate_system_health(df, latest_year)
    print(f"System Health ({latest_year}): {health:.2f}")
    
    # Test full report
    report = analyzer.generate_summary_report(df, "Iran")
    print(f"Civilization Type: {report['civilization_type']}")
    
    # Show some statistics
    print(f"\nData summary:")
    print(f"  Records: {len(df)}")
    print(f"  Year range: {df['Year'].min()} to {df['Year'].max()}")
    print(f"  Countries: {df['Nation'].unique()}")
    print(f"  Nodes: {len(df['Node'].unique())} unique nodes")
    
    # Save cleaned data
    df.to_csv('Iran_CAMS_Cleaned.csv', index=False)
    print(f"\nSUCCESS: Saved cleaned Iran data!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()