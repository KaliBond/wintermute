"""
CAMS Framework Analysis Examples
Example scripts demonstrating how to use the CAMS analysis tools
"""

import pandas as pd
import numpy as np
from src.cams_analyzer import CAMSAnalyzer
from src.visualizations import CAMSVisualizer
import matplotlib.pyplot as plt

def basic_analysis_example():
    """Basic analysis workflow example"""
    print("=== CAMS Framework Basic Analysis Example ===\n")
    
    # Initialize analyzer
    analyzer = CAMSAnalyzer()
    
    # Load data
    try:
        australia_df = analyzer.load_data('Australia_CAMS_Cleaned.csv')
        usa_df = analyzer.load_data('USA_CAMS_Cleaned.csv')
    except FileNotFoundError:
        print("Data files not found. Please ensure CSV files are in the current directory.")
        return
    
    # Generate reports
    print("1. AUSTRALIA ANALYSIS")
    print("-" * 30)
    au_report = analyzer.generate_summary_report(australia_df, "Australia")
    
    print(f"Nation: {au_report['nation']}")
    print(f"Time Period: {au_report['time_period']}")
    print(f"Civilization Type: {au_report['civilization_type']}")
    print(f"Current System Health: {au_report['current_health']:.2f}")
    print(f"Health Trajectory: {au_report['health_trajectory']}")
    print(f"Stress ESD: {au_report['stress_analysis']['esd']:.3f}")
    print(f"Phase Transitions Detected: {len(au_report['phase_transitions'])}")
    
    if au_report['phase_transitions']:
        print("\nCritical Periods:")
        for transition in au_report['phase_transitions'][-3:]:  # Last 3
            print(f"  {transition['year']}: {transition['type']} (H={transition['health']:.2f})")
    
    print("\n2. USA ANALYSIS")
    print("-" * 30)
    usa_report = analyzer.generate_summary_report(usa_df, "USA")
    
    print(f"Nation: {usa_report['nation']}")
    print(f"Time Period: {usa_report['time_period']}")
    print(f"Civilization Type: {usa_report['civilization_type']}")
    print(f"Current System Health: {usa_report['current_health']:.2f}")
    print(f"Health Trajectory: {usa_report['health_trajectory']}")
    print(f"Stress ESD: {usa_report['stress_analysis']['esd']:.3f}")
    print(f"Phase Transitions Detected: {len(usa_report['phase_transitions'])}")
    
    # Comparative analysis
    print("\n3. COMPARATIVE ANALYSIS")
    print("-" * 30)
    
    # DTW similarity
    similarity = analyzer.calculate_dtw_similarity(australia_df, usa_df, 'Coherence')
    print(f"Coherence Trajectory Similarity: {similarity:.3f}")
    
    capacity_similarity = analyzer.calculate_dtw_similarity(australia_df, usa_df, 'Capacity')
    print(f"Capacity Trajectory Similarity: {capacity_similarity:.3f}")
    
    # System health comparison
    au_latest_health = analyzer.calculate_system_health(australia_df, australia_df['Year'].max())
    usa_latest_health = analyzer.calculate_system_health(usa_df, usa_df['Year'].max())
    
    print(f"\nCurrent System Health Comparison:")
    print(f"Australia: {au_latest_health:.2f}")
    print(f"USA: {usa_latest_health:.2f}")
    print(f"Difference: {abs(au_latest_health - usa_latest_health):.2f}")

def visualization_example():
    """Example of creating visualizations"""
    print("\n=== Visualization Example ===\n")
    
    analyzer = CAMSAnalyzer()
    visualizer = CAMSVisualizer()
    
    try:
        df = analyzer.load_data('Australia_CAMS_Cleaned.csv')
    except FileNotFoundError:
        print("Australia data file not found.")
        return
    
    # Create dashboard
    dashboard = visualizer.create_dashboard_layout(df, "Australia")
    
    print("Generated visualizations:")
    for viz_name in dashboard.keys():
        print(f"  - {viz_name}")
    
    # Example: Save one visualization as HTML
    dashboard['health_timeline'].write_html("australia_health_timeline.html")
    print("\nSaved 'australia_health_timeline.html'")

def stress_analysis_example():
    """Detailed stress analysis example"""
    print("\n=== Stress Analysis Example ===\n")
    
    analyzer = CAMSAnalyzer()
    
    try:
        df = analyzer.load_data('Australia_CAMS_Cleaned.csv')
    except FileNotFoundError:
        print("Data file not found.")
        return
    
    # Analyze stress for different time periods
    years_to_analyze = [1900, 1950, 2000, df['Year'].max()]
    
    print("Stress Distribution Analysis:")
    print("-" * 40)
    
    for year in years_to_analyze:
        if year in df['Year'].values:
            stress_analysis = analyzer.calculate_node_stress_distribution(df, year)
            
            print(f"\n{year}:")
            print(f"  ESD (Evenness): {stress_analysis['esd']:.3f}")
            print(f"  Total Stress: {stress_analysis['total_stress']:.1f}")
            print(f"  Mean Stress: {stress_analysis['mean_stress']:.2f}")
            print(f"  Std Stress: {stress_analysis['std_stress']:.2f}")
            
            # Find most/least stressed nodes
            year_data = df[df['Year'] == year]
            most_stressed = year_data.loc[year_data['Stress'].abs().idxmax()]
            least_stressed = year_data.loc[year_data['Stress'].abs().idxmin()]
            
            print(f"  Most Stressed Node: {most_stressed['Node']} ({abs(most_stressed['Stress']):.1f})")
            print(f"  Least Stressed Node: {least_stressed['Node']} ({abs(least_stressed['Stress']):.1f})")

def phase_transition_example():
    """Phase transition detection example"""
    print("\n=== Phase Transition Detection Example ===\n")
    
    analyzer = CAMSAnalyzer()
    
    try:
        df = analyzer.load_data('USA_CAMS_Cleaned.csv')
    except FileNotFoundError:
        print("Data file not found.")
        return
    
    # Detect phase transitions
    transitions = analyzer.detect_phase_transitions(df, "USA")
    
    print(f"Detected {len(transitions)} potential phase transitions:")
    print("-" * 50)
    
    # Group by severity
    by_severity = {}
    for t in transitions:
        severity = t['severity']
        if severity not in by_severity:
            by_severity[severity] = []
        by_severity[severity].append(t)
    
    for severity in ['Critical', 'High', 'Medium']:
        if severity in by_severity:
            print(f"\n{severity} Risk Periods:")
            for t in by_severity[severity]:
                print(f"  {t['year']}: {t['type']} (H={t['health']:.2f})")
    
    # Calculate system health trajectory
    print(f"\nSystem Health Trajectory (last 10 years):")
    years = sorted(df['Year'].unique())[-10:]
    
    for year in years:
        health = analyzer.calculate_system_health(df, year)
        status = "🔴" if health < 2.5 else "🟡" if health < 5.0 else "🟢"
        print(f"  {year}: {health:.2f} {status}")

def node_performance_example():
    """Node performance analysis example"""
    print("\n=== Node Performance Analysis Example ===\n")
    
    analyzer = CAMSAnalyzer()
    
    try:
        df = analyzer.load_data('Australia_CAMS_Cleaned.csv')
    except FileNotFoundError:
        print("Data file not found.")
        return
    
    # Analyze each node's performance over time
    latest_year = df['Year'].max()
    earliest_year = df['Year'].min()
    
    print(f"Node Performance Comparison ({earliest_year} vs {latest_year}):")
    print("-" * 60)
    
    for node in analyzer.nodes:
        early_data = df[(df['Year'] == earliest_year) & (df['Node'] == node)]
        late_data = df[(df['Year'] == latest_year) & (df['Node'] == node)]
        
        if not early_data.empty and not late_data.empty:
            early_health = early_data['Node value'].iloc[0]
            late_health = late_data['Node value'].iloc[0]
            change = late_health - early_health
            
            trend = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            
            print(f"{node:20} | {early_health:5.1f} → {late_health:5.1f} ({change:+5.1f}) {trend}")
    
    # Identify best and worst performing nodes currently
    latest_data = df[df['Year'] == latest_year]
    best_node = latest_data.loc[latest_data['Node value'].idxmax()]
    worst_node = latest_data.loc[latest_data['Node value'].idxmin()]
    
    print(f"\nCurrent Performance Leaders:")
    print(f"Best:  {best_node['Node']} (Value: {best_node['Node value']:.1f})")
    print(f"Worst: {worst_node['Node']} (Value: {worst_node['Node value']:.1f})")

if __name__ == "__main__":
    print("🏛️ CAMS Framework Analysis Examples")
    print("=" * 50)
    
    # Run all examples
    basic_analysis_example()
    stress_analysis_example()
    phase_transition_example()
    node_performance_example()
    visualization_example()
    
    print("\n" + "=" * 50)
    print("Analysis complete! Check generated HTML files for visualizations.")
    print("Run 'streamlit run dashboard.py' to launch the interactive dashboard.")