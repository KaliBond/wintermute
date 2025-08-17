"""
CAMS Dataset Validator and Cleaner
Comprehensive validation and cleaning tool for CAMS-CAN v3.4 compatibility
"""

import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime

class CAMSDatasetValidator:
    def __init__(self):
        self.required_columns = ['Node', 'Coherence', 'Capacity', 'Stress', 'Abstraction']
        self.optional_columns = ['Year']
        self.standard_nodes = [
            'Executive', 'Army', 'Priesthood', 'Property', 
            'Trades', 'Proletariat', 'StateMemory', 'Merchants'
        ]
        
        self.country_mapping = {
            'australia cams cleaned': 'Australia',
            'usa cams cleaned': 'USA',
            'reconstructed usa dataset': 'USA_Reconstructed', 
            'france cams cleaned': 'France',
            'denmark cams cleaned': 'Denmark',
            'germany1750 2025': 'Germany',
            'italy cams cleaned': 'Italy',
            'iran cams cleaned': 'Iran',
            'iraq cams cleaned': 'Iraq',
            'lebanon cams cleaned': 'Lebanon',
            'japan 1850 2025': 'Japan',
            'thailand 1850_2025': 'Thailand',
            'netherlands mastersheet': 'Netherlands',
            'hongkong cams cleaned': 'Hong Kong',
            'saudi arabia master file': 'Saudi Arabia',
            'new rome ad 5y': 'Early Rome',
            'eqmasterrome': 'Roman Empire',
            'us high res 2025': 'USA_HighRes',
            'usa master odd': 'USA_Master',
            'usa maximum': 'USA_Maximum'
        }
        
        self.validation_results = {}
    
    def detect_column_alternatives(self, df):
        """Detect alternative column names and suggest mappings"""
        column_alternatives = {
            'Node': ['node', 'Node_Name', 'Institution', 'institution', 'Name', 'name'],
            'Coherence': ['coherence', 'Coh', 'C', 'Coherance'],
            'Capacity': ['capacity', 'Cap', 'K', 'Capability', 'capability'],
            'Stress': ['stress', 'S', 'Stress_Level'],
            'Abstraction': ['abstraction', 'Abstract', 'A', 'Abs', 'abstraction_level'],
            'Year': ['year', 'YEAR', 'Date', 'date', 'Time', 'time']
        }
        
        mappings = {}
        for standard, alternatives in column_alternatives.items():
            for alt in alternatives:
                if alt in df.columns and standard not in df.columns:
                    mappings[alt] = standard
                    break
        
        return mappings
    
    def standardize_node_names(self, df):
        """Standardize node names to match CAMS-CAN v3.4 specification"""
        if 'Node' not in df.columns:
            return df
            
        node_mappings = {
            # Executive variations
            'executive': 'Executive',
            'Executive Branch': 'Executive',
            'Government': 'Executive',
            'gov': 'Executive',
            
            # Army variations  
            'army': 'Army',
            'military': 'Army',
            'Military': 'Army',
            'Armed Forces': 'Army',
            'Defense': 'Army',
            
            # Priesthood variations
            'priesthood': 'Priesthood',
            'religion': 'Priesthood',
            'Religion': 'Priesthood',
            'Religious': 'Priesthood',
            'Church': 'Priesthood',
            
            # Property variations
            'property': 'Property',
            'property owners': 'Property',
            'Property Owners': 'Property',
            'Owners': 'Property',
            'Stewards': 'Property',
            'stewards': 'Property',
            
            # Trades variations
            'trades': 'Trades',
            'craft': 'Trades',
            'Craft': 'Trades',
            'Artisans': 'Trades',
            'artisans': 'Trades',
            
            # Proletariat variations
            'proletariat': 'Proletariat',
            'workers': 'Proletariat',
            'Workers': 'Proletariat',
            'Labor': 'Proletariat',
            'labor': 'Proletariat',
            'flow': 'Proletariat',
            'Flow': 'Proletariat',
            'hands': 'Proletariat',
            'Hands': 'Proletariat',
            
            # StateMemory variations
            'statememory': 'StateMemory',
            'state memory': 'StateMemory',
            'State Memory': 'StateMemory',
            'Memory': 'StateMemory',
            'memory': 'StateMemory',
            'Records': 'StateMemory',
            
            # Merchants variations
            'merchants': 'Merchants',
            'merchant': 'Merchants',
            'Merchant': 'Merchants',
            'Trade': 'Merchants',
            'trade': 'Merchants',
            'Commerce': 'Merchants',
            'commerce': 'Merchants'
        }
        
        df['Node'] = df['Node'].replace(node_mappings)
        return df
    
    def validate_dataset(self, filepath):
        """Validate a single dataset file"""
        result = {
            'filepath': filepath,
            'filename': os.path.basename(filepath),
            'valid': False,
            'issues': [],
            'warnings': [],
            'info': {},
            'cleaned_data': None
        }
        
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(filepath, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                result['issues'].append("Could not read file with any encoding")
                return result
            
            result['info']['original_shape'] = df.shape
            result['info']['original_columns'] = list(df.columns)
            
            # Check for empty file
            if len(df) == 0:
                result['issues'].append("File is empty")
                return result
            
            # Detect and apply column mappings
            mappings = self.detect_column_alternatives(df)
            if mappings:
                df = df.rename(columns=mappings)
                result['info']['column_mappings'] = mappings
            
            # Check required columns
            missing_required = [col for col in self.required_columns if col not in df.columns]
            if missing_required:
                result['issues'].append(f"Missing required columns: {missing_required}")
                return result
            
            # Standardize node names
            df = self.standardize_node_names(df)
            
            # Validate data types and clean
            numeric_columns = ['Coherence', 'Capacity', 'Stress', 'Abstraction']
            for col in numeric_columns:
                if col in df.columns:
                    original_count = len(df)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    null_count = df[col].isnull().sum()
                    if null_count > 0:
                        result['warnings'].append(f"{col}: {null_count} non-numeric values converted to NaN")
            
            # Handle Year column if present
            if 'Year' in df.columns:
                df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
                year_nulls = df['Year'].isnull().sum()
                if year_nulls > 0:
                    result['warnings'].append(f"Year: {year_nulls} invalid year values")
            
            # Remove rows with missing critical data
            critical_cols = ['Node'] + numeric_columns
            before_clean = len(df)
            df = df.dropna(subset=critical_cols)
            after_clean = len(df)
            
            if before_clean != after_clean:
                result['warnings'].append(f"Removed {before_clean - after_clean} rows with missing critical data")
            
            if len(df) == 0:
                result['issues'].append("No valid data rows after cleaning")
                return result
            
            # Validate node names
            unique_nodes = df['Node'].unique()
            result['info']['unique_nodes'] = list(unique_nodes)
            result['info']['node_count'] = len(unique_nodes)
            
            # Check for standard CAMS nodes
            standard_nodes_present = [node for node in self.standard_nodes if node in unique_nodes]
            result['info']['standard_nodes_present'] = standard_nodes_present
            result['info']['standard_coverage'] = len(standard_nodes_present) / len(self.standard_nodes)
            
            if len(standard_nodes_present) < 4:
                result['warnings'].append(f"Only {len(standard_nodes_present)} standard CAMS nodes found")
            
            # Validate numeric ranges
            for col in numeric_columns:
                if col in df.columns:
                    col_min, col_max = df[col].min(), df[col].max()
                    result['info'][f'{col}_range'] = (col_min, col_max)
                    
                    # Check for reasonable ranges
                    if col in ['Coherence', 'Capacity', 'Abstraction']:
                        if col_min < 0 or col_max > 10:
                            result['warnings'].append(f"{col} values outside expected range (0-10): {col_min:.2f} to {col_max:.2f}")
                    elif col == 'Stress':
                        if abs(col_min) > 10 or abs(col_max) > 10:
                            result['warnings'].append(f"{col} values outside expected range (-10 to +10): {col_min:.2f} to {col_max:.2f}")
            
            # Check time series properties
            if 'Year' in df.columns:
                years = sorted(df['Year'].dropna().unique())
                if len(years) > 1:
                    result['info']['year_range'] = f"{int(years[0])}-{int(years[-1])}"
                    result['info']['time_series'] = True
                    
                    # Check for consistent time coverage
                    year_gaps = []
                    for i in range(1, len(years)):
                        if years[i] - years[i-1] > 1:
                            year_gaps.append((int(years[i-1]), int(years[i])))
                    
                    if year_gaps:
                        result['warnings'].append(f"Time gaps detected: {year_gaps}")
                else:
                    result['info']['time_series'] = False
            else:
                result['info']['time_series'] = False
            
            # Final validation
            result['valid'] = True
            result['cleaned_data'] = df
            result['info']['final_shape'] = df.shape
            
        except Exception as e:
            result['issues'].append(f"Error processing file: {str(e)}")
        
        return result
    
    def generate_country_name(self, filename):
        """Generate standardized country name from filename"""
        base_name = filename.replace('.csv', '').replace('.CSV', '').lower().strip()
        base_name = base_name.replace('_', ' ').replace(' (2)', '').replace(' - ', ' ')
        
        return self.country_mapping.get(base_name, base_name.title())
    
    def validate_all_datasets(self, directory="."):
        """Validate all CSV files in directory"""
        csv_files = glob.glob(os.path.join(directory, "*.csv")) + glob.glob(os.path.join(directory, "*.CSV"))
        
        print(f"Found {len(csv_files)} CSV files to validate\n")
        
        valid_datasets = {}
        invalid_datasets = {}
        
        for filepath in csv_files:
            filename = os.path.basename(filepath)
            print(f"Validating: {filename}")
            
            result = self.validate_dataset(filepath)
            country_name = self.generate_country_name(filename)
            
            if result['valid']:
                valid_datasets[country_name] = result
                print(f"  Valid - {result['info']['final_shape'][0]} rows, {result['info']['final_shape'][1]} columns")
                if result['warnings']:
                    print(f"  Warnings: {len(result['warnings'])}")
            else:
                invalid_datasets[country_name] = result
                print(f"  Invalid - Issues: {len(result['issues'])}")
            print()
        
        self.validation_results = {
            'valid': valid_datasets,
            'invalid': invalid_datasets,
            'summary': {
                'total_files': len(csv_files),
                'valid_count': len(valid_datasets),
                'invalid_count': len(invalid_datasets),
                'validation_date': datetime.now().isoformat()
            }
        }
        
        return self.validation_results
    
    def save_cleaned_datasets(self, output_dir="cleaned_datasets"):
        """Save cleaned datasets to output directory"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        saved_count = 0
        for country_name, result in self.validation_results['valid'].items():
            if result['cleaned_data'] is not None:
                output_path = os.path.join(output_dir, f"{country_name.replace(' ', '_')}_cleaned.csv")
                result['cleaned_data'].to_csv(output_path, index=False)
                print(f"Saved: {output_path}")
                saved_count += 1
        
        print(f"\nSaved {saved_count} cleaned datasets to {output_dir}/")
        return saved_count
    
    def generate_validation_report(self, output_path="validation_report.md"):
        """Generate comprehensive validation report"""
        if not self.validation_results:
            print("No validation results available. Run validate_all_datasets() first.")
            return
        
        report = []
        report.append("# CAMS Dataset Validation Report")
        report.append(f"**Generated:** {self.validation_results['summary']['validation_date']}")
        report.append(f"**Total Files:** {self.validation_results['summary']['total_files']}")
        report.append(f"**Valid Datasets:** {self.validation_results['summary']['valid_count']}")
        report.append(f"**Invalid Datasets:** {self.validation_results['summary']['invalid_count']}")
        report.append("")
        
        # Valid datasets summary
        if self.validation_results['valid']:
            report.append("## ✅ Valid Datasets")
            report.append("")
            
            for country, result in self.validation_results['valid'].items():
                report.append(f"### {country}")
                report.append(f"- **File:** {result['filename']}")
                report.append(f"- **Shape:** {result['info']['final_shape']}")
                report.append(f"- **Nodes:** {result['info']['node_count']} ({', '.join(result['info']['unique_nodes'])})")
                report.append(f"- **Standard Coverage:** {result['info']['standard_coverage']:.1%}")
                report.append(f"- **Time Series:** {result['info']['time_series']}")
                
                if result['info']['time_series'] and 'year_range' in result['info']:
                    report.append(f"- **Years:** {result['info']['year_range']}")
                
                # Numeric ranges
                for col in ['Coherence', 'Capacity', 'Stress', 'Abstraction']:
                    if f'{col}_range' in result['info']:
                        min_val, max_val = result['info'][f'{col}_range']
                        report.append(f"- **{col} Range:** {min_val:.2f} to {max_val:.2f}")
                
                if result['warnings']:
                    report.append("- **Warnings:**")
                    for warning in result['warnings']:
                        report.append(f"  - {warning}")
                
                report.append("")
        
        # Invalid datasets
        if self.validation_results['invalid']:
            report.append("## ❌ Invalid Datasets")
            report.append("")
            
            for country, result in self.validation_results['invalid'].items():
                report.append(f"### {country}")
                report.append(f"- **File:** {result['filename']}")
                report.append("- **Issues:**")
                for issue in result['issues']:
                    report.append(f"  - {issue}")
                report.append("")
        
        # Recommendations
        report.append("## 📋 Recommendations")
        report.append("")
        
        valid_count = len(self.validation_results['valid'])
        invalid_count = len(self.validation_results['invalid'])
        
        if valid_count > 0:
            report.append(f"✅ **{valid_count} datasets are ready for CAMS-CAN v3.4 analysis**")
        
        if invalid_count > 0:
            report.append(f"⚠️ **{invalid_count} datasets need attention before use**")
        
        # Node coverage analysis
        all_standard_nodes = set()
        for result in self.validation_results['valid'].values():
            all_standard_nodes.update(result['info']['standard_nodes_present'])
        
        missing_nodes = set(self.standard_nodes) - all_standard_nodes
        if missing_nodes:
            report.append(f"📊 **Standard nodes missing across all datasets:** {', '.join(missing_nodes)}")
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"Validation report saved to: {output_path}")
        
        return '\n'.join(report)

def main():
    """Main validation and cleaning process"""
    print("CAMS Dataset Validator and Cleaner")
    print("=" * 50)
    
    validator = CAMSDatasetValidator()
    
    # Validate all datasets
    results = validator.validate_all_datasets()
    
    # Print summary
    print("VALIDATION SUMMARY")
    print("=" * 30)
    print(f"Total files processed: {results['summary']['total_files']}")
    print(f"Valid datasets: {results['summary']['valid_count']}")
    print(f"Invalid datasets: {results['summary']['invalid_count']}")
    print(f"Success rate: {results['summary']['valid_count']/results['summary']['total_files']*100:.1f}%")
    print()
    
    # Save cleaned datasets
    if results['valid']:
        print("SAVING CLEANED DATASETS")
        print("=" * 30)
        validator.save_cleaned_datasets()
        print()
    
    # Generate report
    print("GENERATING VALIDATION REPORT")
    print("=" * 30)
    validator.generate_validation_report()
    
    print("\nDataset validation and cleaning completed!")
    
    return validator

if __name__ == "__main__":
    validator = main()