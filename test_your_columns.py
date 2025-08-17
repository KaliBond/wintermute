"""
Test with the exact column structure from your error message
"""

import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cams_analyzer import CAMSAnalyzer

# Create test data with your exact column structure
test_data = {
    'Society': ['TestNation'] * 8,
    'Year': [2000] * 8,
    'Node': ['Executive', 'Army', 'Priests', 'Property Owners', 'Trades/Professions', 'Proletariat', 'State Memory', 'Shopkeepers/Merchants'],
    'Coherence': [6.5, 5.0, 7.0, 7.5, 6.0, 5.5, 4.0, 6.5],
    'Capacity': [6.0, 4.5, 6.0, 7.0, 5.5, 6.0, 3.5, 6.0],
    'Stress': [-4, -5, -3, -2, -3, -2, -4, -2],
    'Abstraction': [5.5, 4.0, 6.0, 6.5, 5.0, 4.5, 3.0, 5.5],
    'Node Value': [18.0, 14.0, 19.0, 20.0, 16.5, 17.0, 10.5, 18.0],
    'Bond Strength': [10.8, 8.4, 11.4, 12.0, 9.9, 10.2, 6.3, 10.8],
    'Nation': ['TestNation'] * 8
}

df = pd.DataFrame(test_data)

print("TEST: Your Exact Column Structure")
print("=" * 40)
print(f"Columns: {list(df.columns)}")
print(f"Shape: {df.shape}")

# Test with analyzer
analyzer = CAMSAnalyzer()

print("\nColumn Detection Test:")
try:
    for col_type in ['year', 'nation', 'node', 'coherence', 'capacity', 'stress', 'abstraction', 'node_value', 'bond_strength']:
        detected_col = analyzer._get_column_name(df, col_type)
        print(f"  {col_type} -> '{detected_col}' SUCCESS")
except Exception as e:
    print(f"  ERROR: {e}")

print("\nSystem Health Test:")
try:
    health = analyzer.calculate_system_health(df, 2000)
    print(f"  System health: {health:.2f} SUCCESS")
except Exception as e:
    print(f"  ERROR: {e}")

print("\nCivilization Type Test:")
try:
    civ_type = analyzer.analyze_civilization_type(df, "TestNation")
    print(f"  Civilization type: {civ_type} SUCCESS")
except Exception as e:
    print(f"  ERROR: {e}")

print("\nFull Report Test:")
try:
    report = analyzer.generate_summary_report(df, "TestNation")
    print(f"  Report generated: SUCCESS")
    print(f"  Keys: {list(report.keys())}")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()

print("\nTest complete!")