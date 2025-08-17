"""
CAMS Framework Analyzer
Core analysis functions for Complex Adaptive Model State framework
Developed by Kari McKern
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class CAMSAnalyzer:
    """Main analyzer class for CAMS framework data"""
    
    def __init__(self):
        self.nodes = [
            'Executive', 'Army', 'Priests', 'Property Owners',
            'Trades/Professions', 'Proletariat', 'State Memory', 
            'Shopkeepers/Merchants'
        ]
        self.dimensions = ['Coherence', 'Capacity', 'Stress', 'Abstraction']
        
        # Column name mappings to handle variations
        self.column_mappings = {
            'node_value': ['Node Value', 'Node value', 'Node_value', 'NodeValue', 'node_value'],
            'bond_strength': ['Bond Strength', 'Bond strength', 'Bond_strength', 'BondStrength', 'bond_strength'],
            'nation': ['Nation', 'nation', 'Country', 'country', 'Society', 'society'],
            'year': ['Year', 'year'],
            'node': ['Node', 'node'],
            'coherence': ['Coherence', 'coherence'],
            'capacity': ['Capacity', 'capacity'],
            'stress': ['Stress', 'stress'],
            'abstraction': ['Abstraction', 'abstraction']
        }
    
    def _get_column_name(self, df: pd.DataFrame, column_type: str) -> str:
        """Get the actual column name from the dataframe"""
        possible_names = self.column_mappings.get(column_type, [])
        
        for name in possible_names:
            if name in df.columns:
                return name
        
        # If not found, raise a clear error with debugging info
        available_cols = list(df.columns)
        raise KeyError(f"Could not find column for '{column_type}'. Possible names: {possible_names}. Available columns: {available_cols}")
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load CAMS data from CSV file"""
        try:
            df = pd.read_csv(filepath)
            
            # Check if Nation column exists, if not, extract from filename
            if 'Nation' not in df.columns:
                # Extract nation name from filename
                import os
                filename = os.path.basename(filepath)
                if 'Australia' in filename:
                    df['Nation'] = 'Australia'
                elif 'USA' in filename:
                    df['Nation'] = 'USA'
                else:
                    df['Nation'] = 'Unknown'
            
            print(f"Loaded {len(df)} records from {filepath}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def calculate_system_health(self, df: pd.DataFrame, year: int = None) -> float:
        """
        Calculate System Health H(t) = N(t)/D(t) * (1 - P(t))
        Where N(t) is weighted node fitness, D(t) is stress-abstraction penalty,
        and P(t) is polarization penalty
        """
        if year:
            year_col = self._get_column_name(df, 'year')
            year_data = df[df[year_col] == year]
        else:
            year_data = df
            
        if year_data.empty:
            return 0.0
            
        # Calculate weighted node fitness N(t)
        node_value_col = self._get_column_name(year_data, 'node_value')
        bond_strength_col = self._get_column_name(year_data, 'bond_strength')
        
        node_values = pd.to_numeric(year_data[node_value_col].values, errors='coerce')
        bond_strengths = pd.to_numeric(year_data[bond_strength_col].values, errors='coerce')
        
        # Remove NaN values
        valid_mask = ~(np.isnan(node_values) | np.isnan(bond_strengths))
        node_values = node_values[valid_mask]
        bond_strengths = bond_strengths[valid_mask]
        
        n_t = np.sum(node_values * bond_strengths) / len(node_values) if len(node_values) > 0 else 0
        
        # Calculate stress-abstraction penalty D(t)
        stress_col = self._get_column_name(year_data, 'stress')
        abstraction_col = self._get_column_name(year_data, 'abstraction')
        coherence_col = self._get_column_name(year_data, 'coherence')
        
        stress_values = np.abs(pd.to_numeric(year_data[stress_col].values, errors='coerce'))
        abstraction_values = pd.to_numeric(year_data[abstraction_col].values, errors='coerce')
        stress_values = stress_values[~np.isnan(stress_values)]
        abstraction_values = abstraction_values[~np.isnan(abstraction_values)]
        d_t = 1 + np.mean(stress_values) * np.std(abstraction_values) if len(stress_values) > 0 and len(abstraction_values) > 0 else 1
        
        # Calculate polarization penalty P(t)
        coherence_values = pd.to_numeric(year_data[coherence_col].values, errors='coerce')
        coherence_values = coherence_values[~np.isnan(coherence_values)]
        coherence_asymmetry = np.std(coherence_values) / (np.mean(coherence_values) + 1e-6) if len(coherence_values) > 0 else 0
        p_t = min(coherence_asymmetry / 10, 0.9)  # Cap at 0.9
        
        h_t = (n_t / d_t) * (1 - p_t) if d_t != 0 else 0
        return h_t
    
    def detect_phase_transitions(self, df: pd.DataFrame, 
                               nation: str = None) -> List[Dict]:
        """Detect potential phase transitions based on system health thresholds"""
        if nation:
            nation_col = self._get_column_name(df, 'nation')
            df = df[df[nation_col] == nation] if nation_col in df.columns else df
            
        year_col = self._get_column_name(df, 'year')
        years = sorted(df[year_col].unique())
        transitions = []
        
        for year in years:
            h_t = self.calculate_system_health(df, year)
            
            if h_t < 2.3:
                transitions.append({
                    'year': year,
                    'type': 'Forced Reorganization',
                    'health': h_t,
                    'severity': 'Critical'
                })
            elif h_t < 2.5:
                transitions.append({
                    'year': year,
                    'type': 'Collapse Risk',
                    'health': h_t,
                    'severity': 'High'
                })
            elif h_t < 5.0:
                transitions.append({
                    'year': year,
                    'type': 'Instability',
                    'health': h_t,
                    'severity': 'Medium'
                })
                
        return transitions
    
    def calculate_node_stress_distribution(self, df: pd.DataFrame, 
                                         year: int = None) -> Dict:
        """Calculate Evenness of Stress Distribution (ESD)"""
        if year:
            year_col = self._get_column_name(df, 'year')
            year_data = df[df[year_col] == year]
        else:
            year_data = df
            
        stress_col = self._get_column_name(year_data, 'stress')
        stress_values = np.abs(pd.to_numeric(year_data[stress_col].values, errors='coerce'))
        stress_values = stress_values[~np.isnan(stress_values)]
        
        # Calculate Shannon's evenness index
        if len(stress_values) > 0:
            total_stress = np.sum(stress_values)
            if total_stress > 0:
                proportions = stress_values / total_stress
                # Remove zeros to avoid log(0)
                proportions = proportions[proportions > 0]
                shannon_h = -np.sum(proportions * np.log(proportions))
                max_h = np.log(len(proportions))
                esd = shannon_h / max_h if max_h > 0 else 0
            else:
                esd = 0
        else:
            esd = 0
            
        return {
            'esd': esd,
            'stress_values': stress_values.tolist(),
            'total_stress': float(np.sum(stress_values)) if len(stress_values) > 0 else 0.0,
            'mean_stress': float(np.mean(stress_values)) if len(stress_values) > 0 else 0.0,
            'std_stress': float(np.std(stress_values)) if len(stress_values) > 1 else 0.0
        }
    
    def analyze_civilization_type(self, df: pd.DataFrame, 
                                nation: str = None) -> str:
        """Classify civilization type based on CAMS metrics"""
        if nation:
            nation_col = self._get_column_name(df, 'nation')
            df = df[df[nation_col] == nation] if nation_col in df.columns else df
            
        # Use most recent year for classification
        year_col = self._get_column_name(df, 'year')
        latest_year = df[year_col].max()
        latest_data = df[df[year_col] == latest_year]
        
        h_t = self.calculate_system_health(latest_data)
        
        # Calculate bond strength average
        bond_strength_col = self._get_column_name(latest_data, 'bond_strength')
        avg_bond_strength = latest_data[bond_strength_col].mean()
        
        # Classify based on thresholds from CAMS framework
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
    
    def calculate_dtw_similarity(self, df1: pd.DataFrame, df2: pd.DataFrame,
                               metric: str = 'Coherence') -> float:
        """Calculate Dynamic Time Warping similarity between two time series"""
        try:
            from fastdtw import fastdtw
            
            year_col1 = self._get_column_name(df1, 'year')
            year_col2 = self._get_column_name(df2, 'year')
            
            series1 = df1.groupby(year_col1)[metric].mean().values
            series2 = df2.groupby(year_col2)[metric].mean().values
            
            distance, _ = fastdtw(series1, series2)
            
            # Normalize to similarity score (0-1)
            max_len = max(len(series1), len(series2))
            similarity = 1 - (distance / (max_len * 20))  # Assuming max metric range ~20
            return max(0, min(1, similarity))
            
        except ImportError:
            print("fastdtw not available, using simple correlation")
            # Fallback to correlation-based similarity
            year_col1 = self._get_column_name(df1, 'year')
            year_col2 = self._get_column_name(df2, 'year')
            
            min_len = min(len(df1[year_col1].unique()), len(df2[year_col2].unique()))
            series1 = df1.groupby(year_col1)[metric].mean().values[:min_len]
            series2 = df2.groupby(year_col2)[metric].mean().values[:min_len]
            
            correlation = np.corrcoef(series1, series2)[0, 1]
            return (correlation + 1) / 2  # Convert from [-1,1] to [0,1]
    
    def generate_summary_report(self, df: pd.DataFrame, 
                              nation: str = None) -> Dict:
        """Generate comprehensive analysis summary"""
        if nation:
            nation_col = self._get_column_name(df, 'nation')
            if nation_col and nation_col in df.columns:
                df = df[df[nation_col] == nation]
        
        year_col = self._get_column_name(df, 'year')
        
        if df.empty or year_col not in df.columns:
            return {
                'nation': nation or 'Unknown',
                'time_period': 'No data',
                'total_records': 0,
                'civilization_type': 'Unclassified/Critical State',
                'current_health': 0.0,
                'phase_transitions': [],
                'stress_analysis': {'esd': 0.0, 'stress_values': [], 'total_stress': 0.0, 'mean_stress': 0.0, 'std_stress': 0.0},
                'recent_health_trend': [],
                'health_trajectory': 'Declining'
            }
        
        report = {
            'nation': nation or 'Unknown',
            'time_period': f"{df[year_col].min()}-{df[year_col].max()}",
            'total_records': len(df),
            'civilization_type': self.analyze_civilization_type(df, nation),
            'current_health': self.calculate_system_health(df, df[year_col].max()),
            'phase_transitions': self.detect_phase_transitions(df, nation),
            'stress_analysis': self.calculate_node_stress_distribution(df, df[year_col].max())
        }
        
        # Add trend analysis
        years = sorted(df[year_col].unique())
        health_trend = []
        for year in years[-5:]:  # Last 5 years
            health_trend.append(self.calculate_system_health(df, year))
            
        report['recent_health_trend'] = health_trend
        report['health_trajectory'] = 'Improving' if len(health_trend) > 1 and health_trend[-1] > health_trend[0] else 'Declining'
        
        return report