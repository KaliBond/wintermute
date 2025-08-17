"""
Quick test to verify the CAMS analyzer fix
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from cams_analyzer import CAMSAnalyzer

def test_analyzer():
    print("Testing CAMS Analyzer fixes...")
    
    analyzer = CAMSAnalyzer()
    
    # Test loading Australia data
    print("\n1. Testing Australia data loading:")
    au_df = analyzer.load_data('Australia_CAMS_Cleaned.csv')
    if not au_df.empty:
        print(f"✅ Australia: {len(au_df)} records loaded")
        print(f"   Columns: {list(au_df.columns)}")
        print(f"   Nation column present: {'Nation' in au_df.columns}")
        
        # Test system health calculation
        try:
            health = analyzer.calculate_system_health(au_df, au_df['Year'].max())
            print(f"✅ System health calculation: {health:.2f}")
        except Exception as e:
            print(f"❌ System health error: {e}")
            
        # Test report generation
        try:
            report = analyzer.generate_summary_report(au_df, "Australia")
            print(f"✅ Report generated: {report['civilization_type']}")
        except Exception as e:
            print(f"❌ Report generation error: {e}")
    else:
        print("❌ Failed to load Australia data")
    
    # Test loading USA data
    print("\n2. Testing USA data loading:")
    usa_df = analyzer.load_data('USA_CAMS_Cleaned.csv')
    if not usa_df.empty:
        print(f"✅ USA: {len(usa_df)} records loaded")
        print(f"   Columns: {list(usa_df.columns)}")
        print(f"   Nation column present: {'Nation' in usa_df.columns}")
        
        # Test system health calculation
        try:
            health = analyzer.calculate_system_health(usa_df, usa_df['Year'].max())
            print(f"✅ System health calculation: {health:.2f}")
        except Exception as e:
            print(f"❌ System health error: {e}")
            
        # Test report generation
        try:
            report = analyzer.generate_summary_report(usa_df, "USA")
            print(f"✅ Report generated: {report['civilization_type']}")
        except Exception as e:
            print(f"❌ Report generation error: {e}")
    else:
        print("❌ Failed to load USA data")

if __name__ == "__main__":
    test_analyzer()
    print("\n🎉 Test complete! Dashboard should now work properly.")