"""
Test script to verify all CAMS dashboard functionality is working
"""
import sys
import os
sys.path.append('src')

from cams_analyzer import CAMSAnalyzer
from visualizations import CAMSVisualizer
import pandas as pd

def test_dashboard_functionality():
    """Test all core dashboard functions"""
    print("=== CAMS Dashboard Functionality Test ===\n")
    
    # Load test data
    print("1. Loading test data...")
    try:
        df = pd.read_csv('Australia_CAMS_Cleaned.csv')
        print(f"   [SUCCESS] Loaded {len(df)} records from Australia data")
    except Exception as e:
        print(f"   [ERROR] Error loading data: {e}")
        return False
    
    # Initialize components
    print("\n2. Initializing components...")
    try:
        analyzer = CAMSAnalyzer()
        visualizer = CAMSVisualizer()
        print("   [SUCCESS] Analyzer and visualizer initialized")
    except Exception as e:
        print(f"   [ERROR] Error initializing: {e}")
        return False
    
    # Test summary report generation
    print("\n3. Testing summary report generation...")
    try:
        report = analyzer.generate_summary_report(df, 'Australia')
        print(f"   [SUCCESS] System Health: {report['current_health']:.2f}")
        print(f"   [SUCCESS] Civilization Type: {report['civilization_type']}")
        print(f"   [SUCCESS] Health Trajectory: {report['health_trajectory']}")
        print(f"   [SUCCESS] Total Records: {report['total_records']}")
        print(f"   [SUCCESS] Stress ESD: {report['stress_analysis']['esd']:.3f}")
    except Exception as e:
        print(f"   [ERROR] Error generating report: {e}")
        return False
    
    # Test system health calculation
    print("\n4. Testing system health calculation...")
    try:
        year_col = analyzer._get_column_name(df, 'year')
        latest_year = df[year_col].max()
        health = analyzer.calculate_system_health(df, latest_year)
        print(f"   [SUCCESS] System health for {latest_year}: {health:.2f}")
    except Exception as e:
        print(f"   [ERROR] Error calculating health: {e}")
        return False
    
    # Test stress distribution
    print("\n5. Testing stress distribution analysis...")
    try:
        stress_analysis = analyzer.calculate_node_stress_distribution(df, latest_year)
        print(f"   [SUCCESS] ESD: {stress_analysis['esd']:.3f}")
        print(f"   [SUCCESS] Mean Stress: {stress_analysis['mean_stress']:.3f}")
        print(f"   [SUCCESS] Total Stress: {stress_analysis['total_stress']:.2f}")
    except Exception as e:
        print(f"   [ERROR] Error analyzing stress: {e}")
        return False
    
    # Test phase transitions
    print("\n6. Testing phase transition detection...")
    try:
        transitions = analyzer.detect_phase_transitions(df, 'Australia')
        print(f"   [SUCCESS] Detected {len(transitions)} phase transitions")
    except Exception as e:
        print(f"   [ERROR] Error detecting transitions: {e}")
        return False
    
    # Test visualizations
    print("\n7. Testing visualization generation...")
    try:
        # Test timeline visualization
        fig_timeline = visualizer.plot_system_health_timeline(df)
        print("   [SUCCESS] System health timeline created")
        
        # Test stress distribution plot
        fig_stress = visualizer.plot_stress_distribution(df)
        print("   [SUCCESS] Stress distribution plot created")
        
        # Test four dimensions radar
        fig_radar = visualizer.plot_four_dimensions_radar(df, latest_year)
        print("   [SUCCESS] Four dimensions radar created")
        
        # Test node heatmap
        fig_heatmap = visualizer.plot_node_heatmap(df, 'coherence', latest_year)
        print("   [SUCCESS] Node heatmap created")
        
    except Exception as e:
        print(f"   [ERROR] Error creating visualizations: {e}")
        return False
    
    print("\n=== ALL TESTS PASSED! Dashboard functionality is working correctly ===")
    return True

if __name__ == "__main__":
    success = test_dashboard_functionality()
    sys.exit(0 if success else 1)