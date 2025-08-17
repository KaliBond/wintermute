"""
CAMS Framework Standalone Analysis Runner
Run analysis without requiring Streamlit dashboard
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from cams_analyzer import CAMSAnalyzer
    print("CAMS Analyzer loaded successfully")
except ImportError as e:
    print(f"Error importing CAMS Analyzer: {e}")
    sys.exit(1)

def run_basic_analysis():
    """Run basic CAMS analysis on available data"""
    print("[BUILDING] CAMS Framework Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = CAMSAnalyzer()
    print("[SUCCESS] Analyzer initialized")
    
    # Try to load data files
    datasets = {}
    data_files = ['Australia_CAMS_Cleaned.csv', 'USA_CAMS_Cleaned.csv']
    
    for filename in data_files:
        try:
            df = pd.read_csv(filename)
            nation_name = filename.replace('_CAMS_Cleaned.csv', '')
            datasets[nation_name] = df
            print(f"[SUCCESS] Loaded {nation_name}: {len(df)} records")
        except FileNotFoundError:
            print(f"[WARNING] {filename} not found, skipping...")
        except Exception as e:
            print(f"[ERROR] Error loading {filename}: {e}")
    
    if not datasets:
        print("[ERROR] No data files found. Please ensure CSV files are in the current directory.")
        return
    
    print("\n[CHART] ANALYSIS RESULTS")
    print("=" * 50)
    
    # Analyze each dataset
    for nation, df in datasets.items():
        print(f"\n[TARGET] {nation.upper()} ANALYSIS")
        print("-" * 30)
        
        # Generate comprehensive report
        try:
            report = analyzer.generate_summary_report(df, nation)
            
            print(f"[CALENDAR] Time Period: {report['time_period']}")
            print(f"[CHART] Total Records: {report['total_records']:,}")
            print(f"[BUILDING] Civilization Type: {report['civilization_type']}")
            print(f"[HEART] Current System Health: {report['current_health']:.2f}")
            print(f"[TREND_UP] Health Trajectory: {report['health_trajectory']}")
            
            # Stress analysis
            stress_info = report['stress_analysis']
            print(f"[LIGHTNING] Stress ESD: {stress_info['esd']:.3f}")
            print(f"[LIGHTNING] Total Stress: {stress_info['total_stress']:.1f}")
            print(f"[LIGHTNING] Mean Stress: {stress_info['mean_stress']:.2f}")
            
            # Phase transitions
            transitions = report['phase_transitions']
            print(f"[WARNING] Phase Transitions: {len(transitions)}")
            
            if transitions:
                print("   Recent Critical Periods:")
                for t in transitions[-3:]:  # Last 3 transitions
                    severity_icon = "[CRITICAL]" if t['severity'] == 'Critical' else "[HIGH]" if t['severity'] == 'High' else "[MEDIUM]"
                    print(f"   {t['year']}: {t['type']} (H={t['health']:.2f}) {severity_icon}")
            
            # Recent health trend
            if len(report['recent_health_trend']) > 1:
                trend_change = report['recent_health_trend'][-1] - report['recent_health_trend'][0]
                trend_icon = "[UP]" if trend_change > 0 else "[DOWN]" if trend_change < 0 else "[FLAT]"
                print(f"[CHART] Recent Trend: {trend_change:+.2f} {trend_icon}")
                
        except Exception as e:
            print(f"[ERROR] Error analyzing {nation}: {e}")
    
    # Comparative analysis if multiple datasets
    if len(datasets) > 1:
        print(f"\n[REFRESH] COMPARATIVE ANALYSIS")
        print("-" * 30)
        
        nations = list(datasets.keys())
        nation1, nation2 = nations[0], nations[1]
        df1, df2 = datasets[nation1], datasets[nation2]
        
        try:
            # Compare system health
            health1 = analyzer.calculate_system_health(df1, df1['Year'].max())
            health2 = analyzer.calculate_system_health(df2, df2['Year'].max())
            
            print(f"[BUILDING] System Health Comparison:")
            print(f"   {nation1}: {health1:.2f}")
            print(f"   {nation2}: {health2:.2f}")
            print(f"   Difference: {abs(health1 - health2):.2f}")
            
            # Compare civilization types
            type1 = analyzer.analyze_civilization_type(df1, nation1)
            type2 = analyzer.analyze_civilization_type(df2, nation2)
            print(f"[TARGET] Civilization Types:")
            print(f"   {nation1}: {type1}")
            print(f"   {nation2}: {type2}")
            
            # DTW similarity (if available)
            try:
                similarity = analyzer.calculate_dtw_similarity(df1, df2, 'Coherence')
                print(f"[LINK] Coherence Similarity: {similarity:.3f}")
            except Exception:
                print("[LINK] Similarity analysis: Not available (requires fastdtw)")
                
        except Exception as e:
            print(f"[ERROR] Error in comparative analysis: {e}")
    
    print(f"\n[SUCCESS] Analysis Complete!")
    print("[INFO] For interactive visualizations, install streamlit and run: streamlit run dashboard.py")

def create_simple_plots():
    """Create basic matplotlib plots"""
    print(f"\n[CHART_UP] CREATING BASIC VISUALIZATIONS")
    print("-" * 40)
    
    analyzer = CAMSAnalyzer()
    
    try:
        # Load Australia data for plotting
        df = pd.read_csv('Australia_CAMS_Cleaned.csv')
        print("[SUCCESS] Creating system health timeline...")
        
        # Calculate system health over time
        years = sorted(df['Year'].unique())
        health_values = []
        
        for year in years:
            health = analyzer.calculate_system_health(df, year)
            health_values.append(health)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(years, health_values, 'b-', linewidth=3, marker='o', markersize=4)
        plt.axhline(y=2.3, color='red', linestyle='--', alpha=0.7, label='Critical: Forced Reorganization')
        plt.axhline(y=2.5, color='orange', linestyle='--', alpha=0.7, label='Warning: Collapse Risk') 
        plt.axhline(y=5.0, color='yellow', linestyle='--', alpha=0.7, label='Instability Threshold')
        
        plt.title('Australia - System Health Timeline (CAMS Framework)', fontsize=16, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('System Health H(t)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig('australia_system_health.png', dpi=300, bbox_inches='tight')
        print("[SUCCESS] Saved: australia_system_health.png")
        
        # Create stress distribution plot
        latest_year = df['Year'].max()
        latest_data = df[df['Year'] == latest_year]
        
        plt.figure(figsize=(10, 6))
        stress_values = np.abs(latest_data['Stress'])
        nodes = latest_data['Node']
        
        bars = plt.bar(range(len(nodes)), stress_values, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
                            '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'][:len(nodes)])
        
        plt.title(f'Australia - Stress Distribution by Node ({latest_year})', fontsize=16, fontweight='bold')
        plt.xlabel('Societal Node', fontsize=12)
        plt.ylabel('Stress Level (Absolute)', fontsize=12)
        plt.xticks(range(len(nodes)), nodes, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        plt.savefig('australia_stress_distribution.png', dpi=300, bbox_inches='tight')
        print("[SUCCESS] Saved: australia_stress_distribution.png")
        
        plt.show()  # This will display the plots if running interactively
        
    except FileNotFoundError:
        print("[WARNING] Australia_CAMS_Cleaned.csv not found for plotting")
    except Exception as e:
        print(f"[ERROR] Error creating plots: {e}")

if __name__ == "__main__":
    # Run the analysis
    run_basic_analysis()
    
    # Create basic plots
    try:
        create_simple_plots()
    except Exception as e:
        print(f"[WARNING] Plotting skipped: {e}")
    
    print(f"\n[CELEBRATION] CAMS Framework analysis complete!")
    print("[CLIPBOARD] Check the generated PNG files for visualizations.")
    print("[BOOKS] See analysis_examples.py for more detailed usage examples.")