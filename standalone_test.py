"""
Standalone test to isolate the issue
"""

import pandas as pd
import sys
import os

# Clear any Python cache
if hasattr(sys, '_clear_type_cache'):
    sys._clear_type_cache()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Force reload if already imported
module_names = ['cams_analyzer', 'visualizations']
for module_name in module_names:
    if module_name in sys.modules:
        del sys.modules[module_name]

try:
    from cams_analyzer import CAMSAnalyzer
    print("SUCCESS: CAMS Analyzer imported")
except Exception as e:
    print(f"ERROR importing CAMS Analyzer: {e}")
    sys.exit(1)

# Test with fresh analyzer
analyzer = CAMSAnalyzer()
print("SUCCESS: Analyzer initialized")

# Load data
try:
    au_df = analyzer.load_data('Australia_CAMS_Cleaned.csv')
    print(f"SUCCESS: Australia loaded - {len(au_df)} records")
    print(f"Columns: {list(au_df.columns)}")
    
    # The problematic line - let's isolate it
    print("Testing report generation step by step...")
    
    # Step 1: Check column detection
    year_col = analyzer._get_column_name(au_df, 'year')
    print(f"Year column detected as: '{year_col}'")
    
    # Step 2: Check if the column exists
    if year_col not in au_df.columns:
        print(f"ERROR: Year column '{year_col}' not found in dataframe!")
        print(f"Available columns: {list(au_df.columns)}")
    else:
        print(f"SUCCESS: Year column '{year_col}' found")
    
    # Step 3: Test system health calculation
    latest_year = au_df[year_col].max()
    print(f"Latest year: {latest_year}")
    
    health = analyzer.calculate_system_health(au_df, latest_year)
    print(f"System health: {health:.2f}")
    
    # Step 4: Test civilization type analysis
    civ_type = analyzer.analyze_civilization_type(au_df, "Australia")
    print(f"Civilization type: {civ_type}")
    
    # Step 5: Full report generation
    report = analyzer.generate_summary_report(au_df, "Australia")
    print(f"SUCCESS: Full report generated")
    print(f"Report keys: {list(report.keys())}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    print("Full traceback:")
    traceback.print_exc()

print("Standalone test complete.")