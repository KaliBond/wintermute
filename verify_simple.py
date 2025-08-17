"""
Simple verification of all imported CAMS datasets
"""

import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from cams_analyzer import CAMSAnalyzer

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
    
    analyzer = CAMSAnalyzer()
    verified = []
    
    for filename, country in datasets:
        print(f"\nTesting {country} ({filename})...")
        
        if not os.path.exists(filename):
            print(f"  ERROR: File not found")
            continue
            
        try:
            # Load dataset
            df = analyzer.load_data(filename)
            
            if df.empty:
                print(f"  ERROR: Dataset is empty")
                continue
            
            # Get basic info
            year_col = analyzer._get_column_name(df, 'year')
            years = sorted(df[year_col].unique())
            
            # Test system health
            latest_year = years[-1]
            health = analyzer.calculate_system_health(df, latest_year)
            
            # Test report generation
            report = analyzer.generate_summary_report(df, country)
            
            print(f"  SUCCESS: {len(df)} records, {years[0]}-{years[-1]}, Health: {health:.2f}")
            print(f"           Type: {report['civilization_type']}")
            
            verified.append(country)
            
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Datasets tested: {len(datasets)}")
    print(f"Successfully verified: {len(verified)}")
    
    if verified:
        print(f"\nReady for dashboard:")
        for country in verified:
            print(f"  - {country}")
    
    if len(verified) == len(datasets):
        print(f"\nAll datasets ready! Run: streamlit run dashboard.py")
    else:
        print(f"\n{len(datasets) - len(verified)} datasets need attention")

if __name__ == "__main__":
    main()