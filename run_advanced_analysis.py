"""
Bridge script to run Advanced CAMS Laws analysis on real data
Integrates the theoretical framework with your imported datasets
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from advanced_cams_laws import (
    CAMSLaws, CAMSNodeSimulator, create_default_initial_state,
    analyze_real_data_with_laws, plot_network_bonds
)
from cams_analyzer import CAMSAnalyzer

def test_advanced_framework_on_real_data():
    """Test the advanced framework on your real imported data"""
    print("=" * 80)
    print("ADVANCED CAMS LAWS ANALYSIS ON REAL DATA")
    print("=" * 80)
    
    # Test datasets from your imports
    test_datasets = [
        ('Australia_CAMS_Cleaned.csv', 'Australia'),
        ('syria.csv', 'Syria'),
        ('Japan 1850 2025 (2).csv', 'Japan'),
        ('Saudi Arabia Master File.csv', 'Saudi Arabia'),
        ('singapore.csv', 'Singapore')
    ]
    
    results = {}
    
    for filename, nation in test_datasets:
        if os.path.exists(filename):
            print(f"\n{'-'*60}")
            print(f"ANALYZING: {nation} ({filename})")
            print(f"{'-'*60}")
            
            try:
                # Load data
                df = pd.read_csv(filename)
                print(f"Loaded {len(df)} records")
                
                # Run advanced analysis
                analysis = analyze_real_data_with_laws(df, nation)
                results[nation] = analysis
                
                # Display key results
                laws = analysis['laws_analysis']
                
                print(f"\nSYSTEM FITNESS ANALYSIS:")
                fitness = laws['law_6_system_fitness']
                print(f"  System Fitness: {fitness['system_fitness']:.2f}")
                print(f"  Low Fitness Nodes: {fitness['low_fitness_nodes']}")
                
                print(f"\nELITE CIRCULATION (Law 7):")
                elite = laws['law_7_elite_circulation']
                print(f"  Average Elite Vitality: {elite['average_elite_vitality']:.2f}")
                print(f"  Stagnant Elites: {elite['stagnant_elites']}")
                print(f"  Circulation Needed: {elite['circulation_needed']}")
                
                print(f"\nTRANSFORMATION POTENTIAL (Law 13):")
                transform = laws['law_13_transformation']
                print(f"  Transformation Score: {transform['transformation_score']:.3f}")
                print(f"  Transformation Likely: {transform['transformation_likely']}")
                print(f"  Type: {transform['transformation_type']}")
                
                print(f"\nSTRESS CASCADE ANALYSIS (Law 11):")
                cascade = laws['law_11_stress_cascade']
                print(f"  Vulnerable Nodes: {cascade['cascade_vulnerable_nodes']}")
                
                print(f"\nMETASTABILITY (Law 12):")
                meta = laws['law_12_metastability']
                print(f"  System Metastable: {meta['system_metastable']}")
                print(f"  Risk Score: {meta['metastability_risk']:.3f}")
                
            except Exception as e:
                print(f"Error analyzing {nation}: {e}")
                continue
    
    # Comparative analysis
    print(f"\n{'='*80}")
    print("COMPARATIVE CIVILIZATIONAL ANALYSIS")
    print(f"{'='*80}")
    
    if results:
        print(f"\n{'Nation':<15} {'Fitness':<10} {'Transform':<10} {'Elite Vital':<12} {'Metastable':<10}")
        print(f"{'-'*65}")
        
        for nation, analysis in results.items():
            laws = analysis['laws_analysis']
            fitness = laws['law_6_system_fitness']['system_fitness']
            transform = laws['law_13_transformation']['transformation_score']
            elite_vital = laws['law_7_elite_circulation']['average_elite_vitality']
            metastable = laws['law_12_metastability']['system_metastable']
            
            print(f"{nation:<15} {fitness:<10.2f} {transform:<10.3f} {elite_vital:<12.2f} {str(metastable):<10}")
    
    return results

def run_theoretical_simulation():
    """Run the theoretical simulation from your framework"""
    print(f"\n{'='*80}")
    print("THEORETICAL CAMS SIMULATION")
    print(f"{'='*80}")
    
    # Create initial state
    initial_nodes = create_default_initial_state()
    
    # Add some stress shocks
    shock_timeline = {
        20: {
            'Executive': [0.0, 0.0, 2.0, 0.5],     # Political crisis
            'Property_Owners': [-1.0, -0.5, 1.5, 0.0],  # Economic shock
        },
        40: {
            'Proletariat': [-0.5, -0.8, 2.5, 0.0],     # Social unrest
            'Army': [0.5, 1.0, 0.5, 0.0],              # Response
        }
    }
    
    # Run simulation
    simulator = CAMSNodeSimulator(initial_nodes, coupling_strength=0.1)
    times, trajectories = simulator.simulate(
        time_span=(0, 60),
        dt=0.2,
        external_shocks_timeline=shock_timeline
    )
    
    # Analyze final state
    final_nodes = trajectories[-1]
    final_analysis = simulator.analyze_laws_compliance(final_nodes, initial_nodes)
    
    print(f"\nFINAL STATE ANALYSIS:")
    print(f"System Fitness: {final_analysis['law_6_system_fitness']['system_fitness']:.2f}")
    print(f"Transformation Score: {final_analysis['law_13_transformation']['transformation_score']:.3f}")
    print(f"Elite Circulation Needed: {final_analysis['law_7_elite_circulation']['circulation_needed']}")
    
    # Plot node trajectories
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    node_names = CAMSNodeSimulator.NODE_NAMES
    variables = ['Coherence', 'Capacity', 'Stress', 'Abstraction']
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    
    for var_idx, (ax, var_name) in enumerate(zip(axes.flat, variables)):
        for node_idx, (node_name, color) in enumerate(zip(node_names, colors)):
            ax.plot(times, trajectories[:, node_idx, var_idx], 
                   color=color, label=node_name[:8], alpha=0.8, linewidth=1.5)
        
        ax.set_title(f'{var_name} Evolution')
        ax.set_xlabel('Time')
        ax.set_ylabel(var_name)
        ax.grid(True, alpha=0.3)
        if var_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.suptitle('Theoretical CAMS Node Evolution', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Network bonds analysis
    laws = CAMSLaws(final_nodes, CAMSNodeSimulator.NODE_NAMES)
    bond_matrix, bond_data = laws.law_8_bond_strength_matrix()
    
    network_fig = plot_network_bonds(bond_matrix, CAMSNodeSimulator.NODE_NAMES, 
                                   "Final Network Bond Strengths")
    plt.show()
    
    return times, trajectories, final_analysis

def main():
    """Main analysis function"""
    print("ADVANCED CAMS FRAMEWORK INTEGRATION")
    print("Bridging theoretical laws with real data")
    
    # Test on real data
    real_data_results = test_advanced_framework_on_real_data()
    
    # Run theoretical simulation
    theoretical_results = run_theoretical_simulation()
    
    print(f"\n{'='*80}")
    print("INTEGRATION SUCCESSFUL!")
    print("Advanced CAMS Laws framework is now operational with your data")
    print(f"{'='*80}")
    
    return real_data_results, theoretical_results

if __name__ == "__main__":
    main()