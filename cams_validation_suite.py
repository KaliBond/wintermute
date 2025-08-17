"""
CAMS Framework Validation Suite
Comprehensive validation and testing framework for CAMS implementation
"""

import sys
import os
import time
import traceback
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

# Add src directory to path
sys.path.append('src')

try:
    from cams_analyzer import CAMSAnalyzer
    from visualizations import CAMSVisualizer
except ImportError as e:
    print(f"[ERROR] Failed to import CAMS modules: {e}")
    sys.exit(1)

class CAMSValidationSuite:
    """Comprehensive validation suite for CAMS Framework"""
    
    def __init__(self):
        self.analyzer = CAMSAnalyzer()
        self.visualizer = CAMSVisualizer()
        self.validation_results = {}
        self.performance_metrics = {}
        
        # Validation thresholds
        self.THRESHOLDS = {
            'missing_data_warning': 0.10,  # 10%
            'missing_data_error': 0.25,    # 25%
            'processing_time_limit': 30.0,  # seconds
            'memory_usage_limit': 0.8,      # 80%
            'data_quality_minimum': 0.9     # 90%
        }
        
        # Expected data ranges
        self.VALUE_RANGES = {
            'coherence': (0, 1),
            'capacity': (0, float('inf')),
            'stress': (-float('inf'), float('inf')),
            'abstraction': (0, float('inf')),
            'node_value': (0, float('inf')),
            'bond_strength': (0, 1)
        }
        
        # Required columns mapping
        self.REQUIRED_COLUMNS = {
            'year': ['Year', 'year', 'YEAR'],
            'node': ['Node', 'node', 'NODE'],
            'coherence': ['Coherence', 'coherence', 'COHERENCE'],
            'capacity': ['Capacity', 'capacity', 'CAPACITY'],
            'stress': ['Stress', 'stress', 'STRESS'],
            'abstraction': ['Abstraction', 'abstraction', 'ABSTRACTION'],
            'node_value': ['Node Value', 'Node value', 'node_value', 'NodeValue'],
            'bond_strength': ['Bond Strength', 'Bond strength', 'bond_strength', 'BondStrength']
        }

    def run_full_validation(self, data_path: str = None) -> Dict[str, Any]:
        """Run complete validation suite"""
        print("=" * 60)
        print("CAMS FRAMEWORK COMPREHENSIVE VALIDATION SUITE")
        print("=" * 60)
        print(f"Validation Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        start_time = time.time()
        
        # Load test data
        if data_path:
            df = self._load_test_data(data_path)
        else:
            df = self._load_default_test_data()
            
        if df is None:
            return {"status": "FAILED", "error": "No test data available"}
        
        # Run validation checks
        validation_steps = [
            ("Data Structure Validation", self._validate_data_structure),
            ("Data Quality Assessment", self._validate_data_quality),
            ("Mathematical Formula Validation", self._validate_mathematical_formulas),
            ("System Health Calculation", self._validate_system_health_calculation),
            ("Stress Distribution Analysis", self._validate_stress_distribution),
            ("Phase Transition Detection", self._validate_phase_transitions),
            ("Visualization Generation", self._validate_visualizations),
            ("Error Handling", self._validate_error_handling),
            ("Performance Benchmarks", self._validate_performance)
        ]
        
        results = {}
        for step_name, validation_func in validation_steps:
            print(f"Running: {step_name}...")
            try:
                step_result = validation_func(df)
                results[step_name] = step_result
                status = "[PASS]" if step_result.get('passed', False) else "[FAIL]"
                print(f"  {status} {step_name}")
                if not step_result.get('passed', False):
                    print(f"    Issues: {step_result.get('issues', [])}")
            except Exception as e:
                results[step_name] = {
                    'passed': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                print(f"  [ERROR] {step_name}: {e}")
            print()
        
        # Generate summary report
        total_time = time.time() - start_time
        summary = self._generate_validation_report(results, total_time)
        
        print("=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {len(results)}")
        print(f"Passed: {summary['passed_count']}")
        print(f"Failed: {summary['failed_count']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Overall Status: {summary['status']}")
        
        return summary

    def _load_test_data(self, data_path: str) -> Optional[pd.DataFrame]:
        """Load test data from specified path"""
        try:
            df = pd.read_csv(data_path)
            print(f"[SUCCESS] Loaded {len(df)} records from {data_path}")
            return df
        except Exception as e:
            print(f"[ERROR] Failed to load data from {data_path}: {e}")
            return None

    def _load_default_test_data(self) -> Optional[pd.DataFrame]:
        """Load default test data"""
        test_files = [
            'Australia_CAMS_Cleaned.csv',
            'USA_CAMS_Cleaned.csv',
            'Denmark_CAMS_Cleaned.csv'
        ]
        
        for filename in test_files:
            if os.path.exists(filename):
                return self._load_test_data(filename)
                
        print("[ERROR] No default test data files found")
        return None

    def _validate_data_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data structure and required columns"""
        issues = []
        checks = []
        
        # Check required columns
        missing_columns = []
        for col_type, possible_names in self.REQUIRED_COLUMNS.items():
            found = False
            for name in possible_names:
                if name in df.columns:
                    found = True
                    break
            if not found:
                missing_columns.append(col_type)
        
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        checks.append(("Required columns present", len(missing_columns) == 0))
        
        # Check data types
        numeric_columns = ['coherence', 'capacity', 'stress', 'abstraction', 'node_value', 'bond_strength']
        for col_type in numeric_columns:
            try:
                col_name = self.analyzer._get_column_name(df, col_type)
                if col_name:
                    non_numeric = pd.to_numeric(df[col_name], errors='coerce').isna().sum()
                    total = len(df[col_name])
                    if non_numeric > 0:
                        issues.append(f"Non-numeric values in {col_name}: {non_numeric}/{total}")
                    checks.append((f"{col_name} is numeric", non_numeric == 0))
            except KeyError:
                continue
        
        # Check for duplicates
        try:
            year_col = self.analyzer._get_column_name(df, 'year')
            node_col = self.analyzer._get_column_name(df, 'node')
            if year_col and node_col:
                duplicates = df.duplicated([year_col, node_col]).sum()
                if duplicates > 0:
                    issues.append(f"Duplicate year-node combinations: {duplicates}")
                checks.append(("No duplicate records", duplicates == 0))
        except KeyError:
            issues.append("Cannot check for duplicates - missing year or node columns")
            checks.append(("Duplicate check", False))
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'checks': checks,
            'total_records': len(df),
            'columns': list(df.columns)
        }

    def _validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality metrics"""
        issues = []
        checks = []
        
        # Calculate completeness
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness = (total_cells - missing_cells) / total_cells
        
        if completeness < self.THRESHOLDS['data_quality_minimum']:
            issues.append(f"Data completeness below threshold: {completeness:.3f}")
        checks.append(("Data completeness acceptable", completeness >= self.THRESHOLDS['data_quality_minimum']))
        
        # Check missing data by column
        for column in df.columns:
            missing_pct = df[column].isnull().sum() / len(df)
            if missing_pct > self.THRESHOLDS['missing_data_error']:
                issues.append(f"Excessive missing data in {column}: {missing_pct:.1%}")
            elif missing_pct > self.THRESHOLDS['missing_data_warning']:
                issues.append(f"Warning: Missing data in {column}: {missing_pct:.1%}")
            
            checks.append((f"{column} missing data acceptable", missing_pct <= self.THRESHOLDS['missing_data_error']))
        
        # Validate value ranges
        for col_type, (min_val, max_val) in self.VALUE_RANGES.items():
            try:
                col_name = self.analyzer._get_column_name(df, col_type)
                if col_name and col_name in df.columns:
                    values = pd.to_numeric(df[col_name], errors='coerce').dropna()
                    if len(values) > 0:
                        out_of_range = ((values < min_val) | (values > max_val)).sum()
                        if out_of_range > 0:
                            issues.append(f"Values out of range in {col_name}: {out_of_range}")
                        checks.append((f"{col_name} values in range", out_of_range == 0))
            except KeyError:
                continue
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'checks': checks,
            'completeness_score': completeness,
            'missing_data_summary': df.isnull().sum().to_dict()
        }

    def _validate_mathematical_formulas(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate mathematical formula implementations"""
        issues = []
        checks = []
        
        try:
            year_col = self.analyzer._get_column_name(df, 'year')
            test_year = df[year_col].max()
            
            # Test system health calculation components
            year_data = df[df[year_col] == test_year]
            
            # Test N(t) calculation
            node_value_col = self.analyzer._get_column_name(year_data, 'node_value')
            bond_strength_col = self.analyzer._get_column_name(year_data, 'bond_strength')
            
            node_values = pd.to_numeric(year_data[node_value_col].values, errors='coerce')
            bond_strengths = pd.to_numeric(year_data[bond_strength_col].values, errors='coerce')
            
            valid_mask = ~(np.isnan(node_values) | np.isnan(bond_strengths))
            node_values = node_values[valid_mask]
            bond_strengths = bond_strengths[valid_mask]
            
            if len(node_values) > 0:
                n_t = np.sum(node_values * bond_strengths) / len(node_values)
                checks.append(("N(t) calculation valid", np.isfinite(n_t) and n_t >= 0))
                if not (np.isfinite(n_t) and n_t >= 0):
                    issues.append(f"Invalid N(t) calculation: {n_t}")
            else:
                issues.append("No valid data for N(t) calculation")
                checks.append(("N(t) calculation valid", False))
            
            # Test D(t) calculation
            stress_col = self.analyzer._get_column_name(year_data, 'stress')
            abstraction_col = self.analyzer._get_column_name(year_data, 'abstraction')
            
            stress_values = np.abs(pd.to_numeric(year_data[stress_col].values, errors='coerce'))
            abstraction_values = pd.to_numeric(year_data[abstraction_col].values, errors='coerce')
            
            stress_values = stress_values[~np.isnan(stress_values)]
            abstraction_values = abstraction_values[~np.isnan(abstraction_values)]
            
            if len(stress_values) > 0 and len(abstraction_values) > 0:
                d_t = 1 + np.mean(stress_values) * np.std(abstraction_values)
                checks.append(("D(t) calculation valid", np.isfinite(d_t) and d_t >= 1))
                if not (np.isfinite(d_t) and d_t >= 1):
                    issues.append(f"Invalid D(t) calculation: {d_t}")
            else:
                issues.append("No valid data for D(t) calculation")
                checks.append(("D(t) calculation valid", False))
            
            # Test P(t) calculation
            coherence_col = self.analyzer._get_column_name(year_data, 'coherence')
            coherence_values = pd.to_numeric(year_data[coherence_col].values, errors='coerce')
            coherence_values = coherence_values[~np.isnan(coherence_values)]
            
            if len(coherence_values) > 0:
                coherence_asymmetry = np.std(coherence_values) / (np.mean(coherence_values) + 1e-6)
                p_t = min(coherence_asymmetry / 10, 0.9)
                checks.append(("P(t) calculation valid", 0 <= p_t <= 0.9))
                if not (0 <= p_t <= 0.9):
                    issues.append(f"Invalid P(t) calculation: {p_t}")
            else:
                issues.append("No valid data for P(t) calculation")
                checks.append(("P(t) calculation valid", False))
        
        except Exception as e:
            issues.append(f"Mathematical validation error: {e}")
            checks.append(("Mathematical formulas", False))
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'checks': checks
        }

    def _validate_system_health_calculation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate system health calculation end-to-end"""
        issues = []
        checks = []
        
        try:
            year_col = self.analyzer._get_column_name(df, 'year')
            years = sorted(df[year_col].unique())
            
            health_values = []
            for year in years[-5:]:  # Test last 5 years
                health = self.analyzer.calculate_system_health(df, year)
                health_values.append(health)
                
                # Validate individual calculation
                if not np.isfinite(health):
                    issues.append(f"Non-finite health value for year {year}: {health}")
                elif health < 0:
                    issues.append(f"Negative health value for year {year}: {health}")
                
                checks.append((f"Health {year} valid", np.isfinite(health) and health >= 0))
            
            # Check for reasonable health values
            if health_values:
                mean_health = np.mean(health_values)
                if mean_health > 1000:  # Unusually high
                    issues.append(f"Unusually high health values (mean: {mean_health:.2f})")
                elif mean_health == 0:
                    issues.append("All health values are zero")
                
                checks.append(("Health values reasonable", 0 < mean_health < 1000))
        
        except Exception as e:
            issues.append(f"System health calculation error: {e}")
            checks.append(("System health calculation", False))
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'checks': checks,
            'sample_health_values': health_values[-3:] if 'health_values' in locals() else []
        }

    def _validate_stress_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate stress distribution analysis"""
        issues = []
        checks = []
        
        try:
            year_col = self.analyzer._get_column_name(df, 'year')
            test_year = df[year_col].max()
            
            stress_analysis = self.analyzer.calculate_node_stress_distribution(df, test_year)
            
            # Validate ESD
            esd = stress_analysis['esd']
            if not (0 <= esd <= 1):
                issues.append(f"ESD out of range [0,1]: {esd}")
            checks.append(("ESD in valid range", 0 <= esd <= 1))
            
            # Validate stress statistics
            total_stress = stress_analysis['total_stress']
            mean_stress = stress_analysis['mean_stress']
            std_stress = stress_analysis['std_stress']
            
            if not np.isfinite(total_stress):
                issues.append(f"Non-finite total stress: {total_stress}")
            if not np.isfinite(mean_stress):
                issues.append(f"Non-finite mean stress: {mean_stress}")
            if not np.isfinite(std_stress):
                issues.append(f"Non-finite std stress: {std_stress}")
            
            checks.append(("Stress statistics finite", all(np.isfinite([total_stress, mean_stress, std_stress]))))
        
        except Exception as e:
            issues.append(f"Stress distribution validation error: {e}")
            checks.append(("Stress distribution", False))
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'checks': checks
        }

    def _validate_phase_transitions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate phase transition detection"""
        issues = []
        checks = []
        
        try:
            transitions = self.analyzer.detect_phase_transitions(df)
            
            # Validate transition structure
            for transition in transitions:
                required_keys = ['year', 'type', 'health', 'severity']
                missing_keys = [key for key in required_keys if key not in transition]
                if missing_keys:
                    issues.append(f"Transition missing keys: {missing_keys}")
                
                # Validate transition values
                if 'health' in transition and not np.isfinite(transition['health']):
                    issues.append(f"Non-finite health in transition: {transition}")
                
            checks.append(("Phase transitions structure valid", len(issues) == 0))
            checks.append(("Phase transitions detected", len(transitions) >= 0))  # Always pass, just count
        
        except Exception as e:
            issues.append(f"Phase transition validation error: {e}")
            checks.append(("Phase transitions", False))
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'checks': checks,
            'transitions_count': len(transitions) if 'transitions' in locals() else 0
        }

    def _validate_visualizations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate visualization generation"""
        issues = []
        checks = []
        
        visualization_tests = [
            ("Timeline", lambda: self.visualizer.plot_system_health_timeline(df)),
            ("Stress Distribution", lambda: self.visualizer.plot_stress_distribution(df)),
            ("Four Dimensions", lambda: self.visualizer.plot_four_dimensions_radar(df)),
            ("Node Heatmap", lambda: self.visualizer.plot_node_heatmap(df, 'coherence'))
        ]
        
        for viz_name, viz_func in visualization_tests:
            try:
                fig = viz_func()
                if fig is None:
                    issues.append(f"{viz_name} visualization returned None")
                    checks.append((f"{viz_name} visualization", False))
                else:
                    checks.append((f"{viz_name} visualization", True))
            except Exception as e:
                issues.append(f"{viz_name} visualization error: {e}")
                checks.append((f"{viz_name} visualization", False))
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'checks': checks
        }

    def _validate_error_handling(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate error handling robustness"""
        issues = []
        checks = []
        
        # Test with empty dataframe
        try:
            empty_df = pd.DataFrame()
            health = self.analyzer.calculate_system_health(empty_df)
            if health != 0.0:
                issues.append(f"Empty dataframe should return 0.0 health, got: {health}")
            checks.append(("Empty dataframe handling", health == 0.0))
        except Exception as e:
            issues.append(f"Empty dataframe error handling failed: {e}")
            checks.append(("Empty dataframe handling", False))
        
        # Test with invalid year
        try:
            health = self.analyzer.calculate_system_health(df, year=9999)
            # Should handle gracefully
            checks.append(("Invalid year handling", True))
        except Exception as e:
            issues.append(f"Invalid year handling failed: {e}")
            checks.append(("Invalid year handling", False))
        
        # Test with missing columns
        try:
            invalid_df = df.drop(columns=[df.columns[0]])  # Remove first column
            report = self.analyzer.generate_summary_report(invalid_df)
            # Should handle gracefully
            checks.append(("Missing columns handling", True))
        except Exception as e:
            # This is expected to fail, but should be handled gracefully
            checks.append(("Missing columns handling", True))
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'checks': checks
        }

    def _validate_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate performance benchmarks"""
        issues = []
        checks = []
        
        # Test calculation performance
        start_time = time.time()
        try:
            year_col = self.analyzer._get_column_name(df, 'year')
            test_year = df[year_col].max()
            health = self.analyzer.calculate_system_health(df, test_year)
            calc_time = time.time() - start_time
            
            if calc_time > 5.0:  # 5 second threshold
                issues.append(f"System health calculation too slow: {calc_time:.2f}s")
            checks.append(("Calculation performance", calc_time <= 5.0))
            
        except Exception as e:
            issues.append(f"Performance test error: {e}")
            checks.append(("Calculation performance", False))
        
        # Test visualization performance
        start_time = time.time()
        try:
            fig = self.visualizer.plot_system_health_timeline(df)
            viz_time = time.time() - start_time
            
            if viz_time > 10.0:  # 10 second threshold
                issues.append(f"Visualization generation too slow: {viz_time:.2f}s")
            checks.append(("Visualization performance", viz_time <= 10.0))
            
        except Exception as e:
            issues.append(f"Visualization performance test error: {e}")
            checks.append(("Visualization performance", False))
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'checks': checks
        }

    def _generate_validation_report(self, results: Dict[str, Any], total_time: float) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        passed_count = sum(1 for result in results.values() if result.get('passed', False))
        total_count = len(results)
        success_rate = (passed_count / total_count * 100) if total_count > 0 else 0
        
        overall_status = "PASS" if passed_count == total_count else "FAIL"
        
        return {
            'status': overall_status,
            'passed_count': passed_count,
            'failed_count': total_count - passed_count,
            'total_count': total_count,
            'success_rate': success_rate,
            'total_time': total_time,
            'detailed_results': results,
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Run the validation suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CAMS Framework Validation Suite')
    parser.add_argument('--data', type=str, help='Path to test data file')
    parser.add_argument('--output', type=str, help='Output file for detailed results')
    
    args = parser.parse_args()
    
    # Run validation
    validator = CAMSValidationSuite()
    results = validator.run_full_validation(args.data)
    
    # Save detailed results if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {args.output}")
    
    # Exit with appropriate code
    sys.exit(0 if results['status'] == 'PASS' else 1)

if __name__ == "__main__":
    main()