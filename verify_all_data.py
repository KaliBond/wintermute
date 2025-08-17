"""
Verify all imported CAMS datasets
"""

import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from cams_analyzer import CAMSAnalyzer

def verify_dataset(filename, country_name):
    """Verify a single dataset"""
    print(f"\n{'='*50}")
    print(f"VERIFYING: {country_name}")
    print(f"File: {filename}")
    print('='*50)
    
    try:
        # Load with analyzer
        analyzer = CAMSAnalyzer()
        df = analyzer.load_data(filename)
        
        if df.empty:
            print("❌ FAILED: Dataset is empty")
            return False
        
        print(f"✅ Loaded successfully: {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        
        # Get year range
        year_col = analyzer._get_column_name(df, 'year')
        years = sorted(df[year_col].unique())
        print(f"Year range: {years[0]} to {years[-1]} ({len(years)} years)")
        
        # Get nation info
        nation_col = analyzer._get_column_name(df, 'nation')
        nations = df[nation_col].unique()
        print(f"Nations: {list(nations)}")
        
        # Get nodes
        node_col = analyzer._get_column_name(df, 'node')
        nodes = df[node_col].unique()
        print(f"Nodes: {list(nodes)}")
        
        # Test system health calculation
        latest_year = years[-1]
        health = analyzer.calculate_system_health(df, latest_year)
        print(f"Latest System Health ({latest_year}): {health:.2f}")
        
        # Generate summary report
        report = analyzer.generate_summary_report(df, country_name)
        print(f"Civilization Type: {report['civilization_type']}")
        print(f"Health Trajectory: {report['health_trajectory']}")
        
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def main():
    print("CAMS Framework - Dataset Verification")
    print("=" * 60)
    
    # List of datasets to verify
    datasets = [
        ('Australia_CAMS_Cleaned.csv', 'Australia'),
        ('USA_CAMS_Cleaned.csv', 'USA'),
        ('Denmark_CAMS_Cleaned.csv', 'Denmark'),
        ('Iraq_CAMS_Cleaned.csv', 'Iraq'),
        ('Lebanon_CAMS_Cleaned.csv', 'Lebanon')
    ]
    
    verified_count = 0
    
    for filename, country in datasets:
        if os.path.exists(filename):
            success = verify_dataset(filename, country)
            if success:
                verified_count += 1
        else:
            print(f"\n❌ {filename} not found!")
    
    # Summary
    print(f"\n{'='*60}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total datasets tested: {len(datasets)}")
    print(f"Successfully verified: {verified_count}")
    print(f"Failed/Missing: {len(datasets) - verified_count}")
    
    if verified_count == len(datasets):
        print("\n🎉 ALL DATASETS READY FOR DASHBOARD!")
        print("You can now run: streamlit run dashboard.py")
        print("\nAvailable countries for analysis:")
        for _, country in datasets:
            print(f"  • {country}")
    else:
        print(f"\n⚠️  {len(datasets) - verified_count} datasets need attention")

if __name__ == "__main__":
    main()