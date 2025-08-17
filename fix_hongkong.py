"""
Fix Hong Kong CAMS data properly
"""

import pandas as pd
import sys
import os

print("Fixing Hong Kong CAMS data...")

# Read the raw file line by line to find the correct header
with open('hongkong.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("Looking for proper CSV structure...")
for i, line in enumerate(lines[:15]):
    print(f"Line {i+1}: {line.strip()[:100]}...")

# The proper header should be "Society,Year,Node,Coherence,Capacity,Stress,Abstraction,Node Value,Bond Strength"
# Let's find it
header_line = None
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped.startswith('Society,Year,Node') or 'Society,Year,Node,Coherence' in stripped:
        header_line = i
        print(f"Found header at line {i+1}: {stripped}")
        break

if header_line is None:
    print("Could not find proper header. Trying manual reconstruction...")
    
    # Let's look for data lines that start with "Hong Kong"
    data_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('Hong Kong,'):
            data_lines.append(stripped)
    
    if data_lines:
        print(f"Found {len(data_lines)} data lines starting with 'Hong Kong,'")
        print("First few data lines:")
        for i, line in enumerate(data_lines[:3]):
            print(f"  {line}")
        
        # Create a proper CSV with header
        csv_content = "Society,Year,Node,Coherence,Capacity,Stress,Abstraction,Node Value,Bond Strength\n"
        csv_content += "\n".join(data_lines)
        
        # Write to a new file
        with open('hongkong_fixed.csv', 'w', encoding='utf-8') as f:
            f.write(csv_content)
        
        print("Created hongkong_fixed.csv")
        
    else:
        print("No data lines found starting with 'Hong Kong,'")
        exit(1)

else:
    # Use pandas to read from the found header line
    print(f"Reading from line {header_line + 1}")
    try:
        df = pd.read_csv('hongkong.csv', skiprows=header_line)
        df.to_csv('hongkong_fixed.csv', index=False)
        print("Created hongkong_fixed.csv using pandas")
    except Exception as e:
        print(f"Pandas read failed: {e}")
        exit(1)

# Now read and test the fixed file
print("\nTesting fixed file...")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from cams_analyzer import CAMSAnalyzer

df = pd.read_csv('hongkong_fixed.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("First 3 rows:")
print(df.head(3))

# Test with analyzer
analyzer = CAMSAnalyzer()

try:
    # Test column detection
    print("\nColumn detection:")
    for col_type in ['year', 'nation', 'node', 'coherence', 'capacity', 'stress', 'abstraction']:
        detected_col = analyzer._get_column_name(df, col_type)
        print(f"  {col_type} -> '{detected_col}' SUCCESS")
    
    # Test system health
    years = sorted(df['Year'].unique())
    latest_year = years[-1]
    health = analyzer.calculate_system_health(df, latest_year)
    print(f"\nYear range: {years[0]} to {years[-1]}")
    print(f"System Health ({latest_year}): {health:.2f}")
    
    # Save as final clean file
    df.to_csv('HongKong_CAMS_Cleaned.csv', index=False)
    print(f"\nSUCCESS: Saved as HongKong_CAMS_Cleaned.csv with {len(df)} records!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()