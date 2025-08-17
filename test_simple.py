"""
Simple test to verify the CAMS analyzer fix
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from cams_analyzer import CAMSAnalyzer

print("Testing CAMS Analyzer fixes...")

analyzer = CAMSAnalyzer()

# Test Australia data
print("\n1. Testing Australia data:")
au_df = analyzer.load_data('Australia_CAMS_Cleaned.csv')
if not au_df.empty:
    print(f"SUCCESS: Australia loaded - {len(au_df)} records")
    print(f"Columns: {list(au_df.columns)}")
    
    # Test system health
    try:
        health = analyzer.calculate_system_health(au_df, au_df['Year'].max())
        print(f"SUCCESS: System health = {health:.2f}")
    except Exception as e:
        print(f"ERROR in system health: {e}")
        
    # Test report
    try:
        report = analyzer.generate_summary_report(au_df, "Australia")
        print(f"SUCCESS: Report generated - {report['civilization_type']}")
    except Exception as e:
        print(f"ERROR in report: {e}")

# Test USA data
print("\n2. Testing USA data:")
usa_df = analyzer.load_data('USA_CAMS_Cleaned.csv')
if not usa_df.empty:
    print(f"SUCCESS: USA loaded - {len(usa_df)} records")
    print(f"Columns: {list(usa_df.columns)}")
    
    # Test system health
    try:
        health = analyzer.calculate_system_health(usa_df, usa_df['Year'].max())
        print(f"SUCCESS: System health = {health:.2f}")
    except Exception as e:
        print(f"ERROR in system health: {e}")

print("\nTest complete! Dashboard should work now.")
print("Access dashboard at: http://localhost:8502")