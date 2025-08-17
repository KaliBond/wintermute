"""
CAMS Framework Simple Analysis - No External Dependencies
Simplified version that works with just pandas, numpy, matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

class SimpleCAMSAnalyzer:
    """Simplified CAMS analyzer without external dependencies"""
    
    def __init__(self):
        self.nodes = [
            'Executive', 'Army', 'Priests', 'Property Owners',
            'Trades/Professions', 'Proletariat', 'State Memory', 
            'Shopkeepers/Merchants'
        ]
        
    def load_data(self, filepath):
        """Load CAMS data from CSV file"""
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded {len(df)} records from {filepath}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def calculate_system_health(self, df, year=None):
        """Calculate System Health H(t) = N(t)/D(t) * (1 - P(t))"""
        if year:
            year_data = df[df['Year'] == year]
        else:
            year_data = df
            
        if year_data.empty:
            return 0.0
            
        # Calculate weighted node fitness N(t)
        node_values = year_data['Node value'].values
        bond_strengths = year_data['Bond strength'].values
        n_t = np.sum(node_values * bond_strengths) / len(node_values)
        
        # Calculate stress-abstraction penalty D(t)
        stress_values = np.abs(year_data['Stress'].values)
        abstraction_values = year_data['Abstraction'].values
        d_t = 1 + np.mean(stress_values) * np.std(abstraction_values)
        
        # Calculate polarization penalty P(t)
        coherence_values = year_data['Coherence'].values
        coherence_asymmetry = np.std(coherence_values) / (np.mean(coherence_values) + 1e-6)
        p_t = min(coherence_asymmetry / 10, 0.9)
        
        h_t = (n_t / d_t) * (1 - p_t)
        return h_t
    
    def analyze_civilization_type(self, df):
        """Classify civilization type based on system health"""
        latest_year = df['Year'].max()
        h_t = self.calculate_system_health(df[df['Year'] == latest_year])
        
        if 10.5 <= h_t <= 20.0:
            return "Type I: Adaptive-Expansive"
        elif 10.4 <= h_t <= 10.9:
            return "Type II: Stable Core"
        elif 8.6 <= h_t <= 10.4:
            return "Type III: Resilient Frontier"
        elif 2.3 <= h_t <= 8.5:
            return "Type IV: Fragile High-Stress"
        else:
            return "Unclassified/Critical State"
    
    def detect_critical_periods(self, df):
        """Find years with critical system health"""
        years = sorted(df['Year'].unique())
        critical_periods = []
        
        for year in years:
            h_t = self.calculate_system_health(df, year)
            
            if h_t < 2.3:
                critical_periods.append((year, 'Forced Reorganization', h_t, 'Critical'))
            elif h_t < 2.5:
                critical_periods.append((year, 'Collapse Risk', h_t, 'High'))
            elif h_t < 5.0:
                critical_periods.append((year, 'Instability', h_t, 'Medium'))
                
        return critical_periods
    
    def calculate_stress_stats(self, df, year=None):
        """Calculate stress distribution statistics"""
        if year:
            year_data = df[df['Year'] == year]
        else:
            year_data = df
            
        stress_values = np.abs(year_data['Stress'].values)
        
        return {
            'total_stress': float(np.sum(stress_values)),
            'mean_stress': float(np.mean(stress_values)),
            'std_stress': float(np.std(stress_values)),
            'max_stress': float(np.max(stress_values)),
            'min_stress': float(np.min(stress_values))
        }

def main():
    print("CAMS Framework Simple Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SimpleCAMSAnalyzer()
    print("Analyzer initialized")
    
    # Try to load data files
    datasets = {}
    data_files = ['Australia_CAMS_Cleaned.csv', 'USA_CAMS_Cleaned.csv']
    
    for filename in data_files:
        try:
            df = pd.read_csv(filename)
            nation_name = filename.replace('_CAMS_Cleaned.csv', '')
            datasets[nation_name] = df
        except FileNotFoundError:
            print(f"{filename} not found, skipping...")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    if not datasets:
        print("No data files found. Please ensure CSV files are in the current directory.")
        return
    
    print("\nANALYSIS RESULTS")
    print("=" * 50)
    
    # Analyze each dataset
    for nation, df in datasets.items():
        print(f"\n{nation.upper()} ANALYSIS")
        print("-" * 30)
        
        try:
            # Basic statistics
            time_period = f"{df['Year'].min()}-{df['Year'].max()}"
            total_records = len(df)
            
            # System health analysis
            current_health = analyzer.calculate_system_health(df, df['Year'].max())
            civ_type = analyzer.analyze_civilization_type(df)
            
            print(f"Time Period: {time_period}")
            print(f"Total Records: {total_records:,}")
            print(f"Civilization Type: {civ_type}")
            print(f"Current System Health: {current_health:.2f}")
            
            # Health trajectory (last 5 years)
            years = sorted(df['Year'].unique())
            recent_years = years[-5:] if len(years) >= 5 else years
            recent_health = []
            
            for year in recent_years:
                health = analyzer.calculate_system_health(df, year)
                recent_health.append(health)
            
            if len(recent_health) > 1:
                trend_change = recent_health[-1] - recent_health[0]
                trajectory = "Improving" if trend_change > 0 else "Declining" if trend_change < 0 else "Stable"
                print(f"Health Trajectory: {trajectory} ({trend_change:+.2f})")
            
            # Stress analysis
            stress_stats = analyzer.calculate_stress_stats(df, df['Year'].max())
            print(f"Current Total Stress: {stress_stats['total_stress']:.1f}")
            print(f"Current Mean Stress: {stress_stats['mean_stress']:.2f}")
            
            # Critical periods
            critical_periods = analyzer.detect_critical_periods(df)
            print(f"Critical Periods Detected: {len(critical_periods)}")
            
            if critical_periods:
                print("Recent Critical Periods:")
                for year, event_type, health, severity in critical_periods[-3:]:
                    print(f"  {year}: {event_type} (H={health:.2f}, {severity})")
            
            # Node performance
            latest_data = df[df['Year'] == df['Year'].max()]
            best_node = latest_data.loc[latest_data['Node value'].idxmax()]
            worst_node = latest_data.loc[latest_data['Node value'].idxmin()]
            
            print(f"Best Performing Node: {best_node['Node']} ({best_node['Node value']:.1f})")
            print(f"Worst Performing Node: {worst_node['Node']} ({worst_node['Node value']:.1f})")
                
        except Exception as e:
            print(f"Error analyzing {nation}: {e}")
    
    # Comparative analysis
    if len(datasets) > 1:
        print(f"\nCOMPARATIVE ANALYSIS")
        print("-" * 30)
        
        nations = list(datasets.keys())
        for i, nation1 in enumerate(nations):
            for nation2 in nations[i+1:]:
                df1, df2 = datasets[nation1], datasets[nation2]
                
                health1 = analyzer.calculate_system_health(df1, df1['Year'].max())
                health2 = analyzer.calculate_system_health(df2, df2['Year'].max())
                
                print(f"{nation1} vs {nation2}:")
                print(f"  System Health: {health1:.2f} vs {health2:.2f}")
                print(f"  Difference: {abs(health1 - health2):.2f}")
    
    # Create visualizations
    print(f"\nCREATING VISUALIZATIONS")
    print("-" * 30)
    
    for nation, df in datasets.items():
        try:
            # System health timeline
            years = sorted(df['Year'].unique())
            health_values = []
            
            for year in years:
                health = analyzer.calculate_system_health(df, year)
                health_values.append(health)
            
            plt.figure(figsize=(12, 6))
            plt.plot(years, health_values, 'b-', linewidth=3, marker='o', markersize=4)
            plt.axhline(y=2.3, color='red', linestyle='--', alpha=0.7, label='Critical: Forced Reorganization')
            plt.axhline(y=2.5, color='orange', linestyle='--', alpha=0.7, label='Warning: Collapse Risk')
            plt.axhline(y=5.0, color='yellow', linestyle='--', alpha=0.7, label='Instability Threshold')
            
            plt.title(f'{nation} - System Health Timeline (CAMS Framework)', fontsize=16, fontweight='bold')
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('System Health H(t)', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            filename = f'{nation.lower()}_system_health.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            
            # Stress distribution
            latest_year = df['Year'].max()
            latest_data = df[df['Year'] == latest_year]
            
            plt.figure(figsize=(10, 6))
            stress_values = np.abs(latest_data['Stress'])
            nodes = latest_data['Node']
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
                     '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
            
            plt.bar(range(len(nodes)), stress_values, color=colors[:len(nodes)])
            plt.title(f'{nation} - Stress Distribution by Node ({latest_year})', fontsize=16, fontweight='bold')
            plt.xlabel('Societal Node', fontsize=12)
            plt.ylabel('Stress Level (Absolute)', fontsize=12)
            plt.xticks(range(len(nodes)), nodes, rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            filename = f'{nation.lower()}_stress_distribution.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            
        except Exception as e:
            print(f"Error creating plots for {nation}: {e}")
    
    print(f"\nAnalysis Complete!")
    print("Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()