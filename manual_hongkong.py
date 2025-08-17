"""
Manually extract Hong Kong CAMS data
"""

import pandas as pd
import sys
import os

print("Manually extracting Hong Kong CAMS data...")

# Read the raw file and manually extract data lines
with open('hongkong.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find all lines that look like data (start with "Hong Kong,")
data_lines = []
header = "Society,Year,Node,Coherence,Capacity,Stress,Abstraction,Node Value,Bond Strength"

for line in lines:
    stripped = line.strip()
    if stripped.startswith('Hong Kong,') and len(stripped.split(',')) >= 9:
        data_lines.append(stripped)

print(f"Found {len(data_lines)} valid data lines")

if len(data_lines) == 0:
    print("No valid data lines found!")
    exit(1)

print("First 3 data lines:")
for i, line in enumerate(data_lines[:3]):
    print(f"  {line}")

# Create proper CSV content
csv_content = header + "\n" + "\n".join(data_lines)

# Write to file
with open('HongKong_manual.csv', 'w', encoding='utf-8') as f:
    f.write(csv_content)

print("Created HongKong_manual.csv")

# Test the file
df = pd.read_csv('HongKong_manual.csv')
print(f"\nLoaded shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("First 3 rows:")
print(df.head(3))

# Convert numeric columns
numeric_cols = ['Year', 'Coherence', 'Capacity', 'Stress', 'Abstraction', 'Node Value', 'Bond Strength']

print("\nConverting numeric columns...")
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  {col}: filled {nan_count} NaN values with median {median_val:.2f}")

# Ensure Year is integer
if 'Year' in df.columns:
    df['Year'] = df['Year'].astype(int)

print(f"Final shape: {df.shape}")

# Test with CAMS analyzer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from cams_analyzer import CAMSAnalyzer

analyzer = CAMSAnalyzer()

try:
    print("\nCAMS Analyzer Test:")
    
    # Test column detection
    for col_type in ['year', 'nation', 'node', 'coherence', 'capacity', 'stress', 'abstraction']:
        detected_col = analyzer._get_column_name(df, col_type)
        print(f"  {col_type} -> '{detected_col}' SUCCESS")
    
    # Test calculations
    years = sorted(df['Year'].unique())
    latest_year = years[-1]
    health = analyzer.calculate_system_health(df, latest_year)
    
    print(f"\nYear range: {years[0]} to {years[-1]} ({len(years)} years)")
    print(f"System Health ({latest_year}): {health:.2f}")
    
    # Test report
    report = analyzer.generate_summary_report(df, "Hong Kong")
    print(f"Civilization Type: {report['civilization_type']}")
    
    # Save final clean file
    df.to_csv('HongKong_CAMS_Cleaned.csv', index=False)
    print(f"\nSUCCESS: Saved HongKong_CAMS_Cleaned.csv with {len(df)} records!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()