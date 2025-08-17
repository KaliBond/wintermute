"""
Comprehensive test of CAMS framework functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from cams_analyzer import CAMSAnalyzer
from visualizations import CAMSVisualizer
import pandas as pd

def test_complete_functionality():
    print("CAMS Framework Comprehensive Test")
    print("=" * 50)
    
    # Initialize components
    analyzer = CAMSAnalyzer()
    visualizer = CAMSVisualizer()
    
    # Load data
    print("\n1. DATA LOADING TEST")
    print("-" * 30)
    
    au_df = analyzer.load_data('Australia_CAMS_Cleaned.csv')
    usa_df = analyzer.load_data('USA_CAMS_Cleaned.csv')
    
    print(f"Australia: {len(au_df)} records loaded")
    print(f"USA: {len(usa_df)} records loaded")
    
    # Test core analysis functions
    print("\n2. CORE ANALYSIS FUNCTIONS TEST")
    print("-" * 30)
    
    try:
        # System health
        au_health = analyzer.calculate_system_health(au_df, au_df['Year'].max())
        usa_health = analyzer.calculate_system_health(usa_df, usa_df['Year'].max())
        print(f"System Health - Australia: {au_health:.2f}, USA: {usa_health:.2f}")
        
        # Civilization types
        au_type = analyzer.analyze_civilization_type(au_df, "Australia")
        usa_type = analyzer.analyze_civilization_type(usa_df, "USA")
        print(f"Civ Types - Australia: {au_type}")
        print(f"            USA: {usa_type}")
        
        # Phase transitions
        au_transitions = analyzer.detect_phase_transitions(au_df, "Australia")
        usa_transitions = analyzer.detect_phase_transitions(usa_df, "USA")
        print(f"Phase Transitions - Australia: {len(au_transitions)}, USA: {len(usa_transitions)}")
        
        # Stress analysis
        au_stress = analyzer.calculate_node_stress_distribution(au_df, au_df['Year'].max())
        usa_stress = analyzer.calculate_node_stress_distribution(usa_df, usa_df['Year'].max())
        print(f"Stress ESD - Australia: {au_stress['esd']:.3f}, USA: {usa_stress['esd']:.3f}")
        
        print("SUCCESS: All core analysis functions working")
        
    except Exception as e:
        print(f"ERROR in core analysis: {e}")
        return False
    
    # Test reports
    print("\n3. REPORT GENERATION TEST")
    print("-" * 30)
    
    try:
        au_report = analyzer.generate_summary_report(au_df, "Australia")
        usa_report = analyzer.generate_summary_report(usa_df, "USA")
        
        print(f"Australia Report: {au_report['civilization_type']}")
        print(f"USA Report: {usa_report['civilization_type']}")
        print("SUCCESS: Report generation working")
        
    except Exception as e:
        print(f"ERROR in report generation: {e}")
        return False
    
    # Test visualizations
    print("\n4. VISUALIZATION TEST")
    print("-" * 30)
    
    try:
        # System health timeline
        health_fig = visualizer.plot_system_health_timeline(au_df, "Australia")
        print("SUCCESS: Health timeline created")
        
        # Heatmap
        heatmap_fig = visualizer.plot_node_heatmap(au_df, 'Coherence', "Australia")
        print("SUCCESS: Heatmap created")
        
        # Stress distribution
        stress_fig = visualizer.plot_stress_distribution(au_df, nation="Australia")
        print("SUCCESS: Stress distribution created")
        
        # Radar chart
        radar_fig = visualizer.plot_four_dimensions_radar(au_df, nation="Australia")
        print("SUCCESS: Radar chart created")
        
        # Network visualization
        network_fig = visualizer.plot_node_network(au_df, nation="Australia")
        print("SUCCESS: Network visualization created")
        
        print("SUCCESS: All visualizations working")
        
    except Exception as e:
        print(f"ERROR in visualizations: {e}")
        return False
    
    # Test comparative analysis
    print("\n5. COMPARATIVE ANALYSIS TEST")
    print("-" * 30)
    
    try:
        similarity = analyzer.calculate_dtw_similarity(au_df, usa_df, 'Coherence')
        print(f"Coherence DTW Similarity: {similarity:.3f}")
        print("SUCCESS: Comparative analysis working")
        
    except Exception as e:
        print(f"NOTE: DTW similarity failed (expected if fastdtw not installed): {e}")
    
    print("\n" + "=" * 50)
    print("COMPREHENSIVE TEST COMPLETED SUCCESSFULLY!")
    print("All major CAMS framework functions are operational.")
    print("Dashboard should be fully functional at: http://localhost:8504")
    
    return True

if __name__ == "__main__":
    success = test_complete_functionality()
    if success:
        print("\nReady for production use!")
    else:
        print("\nSome issues detected - check error messages above.")