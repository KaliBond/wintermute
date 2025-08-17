"""
Comprehensive verification of all CAMS datasets
"""

import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from cams_analyzer import CAMSAnalyzer

def main():
    print("CAMS Framework - Complete Dataset Verification")
    print("=" * 70)
    
    # All datasets including newly imported ones
    datasets = [
        ('Australia_CAMS_Cleaned.csv', 'Australia'),
        ('USA_CAMS_Cleaned.csv', 'USA'),
        ('Denmark_CAMS_Cleaned.csv', 'Denmark'),
        ('Iraq_CAMS_Cleaned.csv', 'Iraq'),
        ('Lebanon_CAMS_Cleaned.csv', 'Lebanon'),
        ('Iran_CAMS_Cleaned.csv', 'Iran'),
        ('France_CAMS_Cleaned.csv', 'France'),
        ('HongKong_CAMS_Cleaned.csv', 'Hong Kong')
    ]
    
    analyzer = CAMSAnalyzer()
    verified = []
    
    for filename, country in datasets:
        print(f"\nTesting {country} ({filename})...")
        
        if not os.path.exists(filename):
            print(f"  MISSING: File not found")
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
            
            print(f"  SUCCESS: {len(df)} records")
            print(f"           Years: {years[0]}-{years[-1]} ({len(years)} years)")
            print(f"           Health: {health:.2f}")
            print(f"           Type: {report['civilization_type']}")
            
            verified.append({
                'country': country,
                'records': len(df),
                'year_start': years[0],
                'year_end': years[-1],
                'year_count': len(years),
                'health': health,
                'type': report['civilization_type']
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("COMPLETE DATASET SUMMARY")
    print("=" * 70)
    
    if verified:
        print(f"{'Country':<12} {'Records':<8} {'Years':<12} {'Health':<8} {'Type':<25}")
        print("-" * 70)
        
        for data in verified:
            year_range = f"{data['year_start']}-{data['year_end']}"
            type_short = data['type'].split(':')[0] if ':' in data['type'] else data['type']
            print(f"{data['country']:<12} {data['records']:<8} {year_range:<12} {data['health']:<8.1f} {type_short:<25}")
        
        total_records = sum(d['records'] for d in verified)
        min_year = min(d['year_start'] for d in verified)
        max_year = max(d['year_end'] for d in verified)
        
        print("-" * 70)
        print(f"{'TOTAL':<12} {total_records:<8} {min_year}-{max_year:<6} {'N/A':<8} {len(verified)} countries")
        
        print(f"\n🎉 ALL {len(verified)} DATASETS READY FOR ANALYSIS!")
        print("\nAvailable for comparative analysis:")
        for data in verified:
            print(f"  • {data['country']}")
        
        print(f"\n🚀 Launch dashboard: streamlit run dashboard.py")
    
    else:
        print("❌ No datasets successfully verified")

if __name__ == "__main__":
    main()