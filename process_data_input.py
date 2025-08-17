"""
CAMS Data Input Processor
Cleans and processes raw data files from the data_input directory
"""

import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime
import glob
import sys

class CAMSDataProcessor:
    """Process and clean CAMS data files"""
    
    def __init__(self):
        self.input_dir = "data_input"
        self.output_dir = "."
        self.backup_dir = os.path.join(self.input_dir, "originals")
        
        # Column mapping for flexible input formats
        self.column_mappings = {
            'year': ['Year', 'year', 'DATE', 'Date', 'date', 'TIME', 'Period', 'period'],
            'nation': ['Nation', 'nation', 'Country', 'country', 'State', 'state'],
            'node': ['Node', 'node', 'Component', 'component', 'Societal_Node', 'societal_node'],
            'coherence': ['Coherence', 'coherence', 'Alignment', 'alignment', 'Unity', 'unity'],
            'capacity': ['Capacity', 'capacity', 'Resources', 'resources', 'Capability', 'capability'],
            'stress': ['Stress', 'stress', 'Pressure', 'pressure', 'Tension', 'tension'],
            'abstraction': ['Abstraction', 'abstraction', 'Innovation', 'innovation', 'Complexity', 'complexity']
        }
        
        # Standard node names mapping
        self.node_mappings = {
            'executive': ['Executive', 'Government', 'Leadership', 'Political', 'Administration', 'Ruling'],
            'army': ['Army', 'Military', 'Defense', 'Armed Forces', 'Security', 'Forces'],
            'priests': ['Priests', 'Religious', 'Clergy', 'Ideology', 'Scientists', 'Intellectuals'],
            'property_owners': ['Property Owners', 'Capitalists', 'Landowners', 'Wealthy', 'Elite', 'Owners'],
            'trades': ['Trades/Professions', 'Skilled Workers', 'Middle Class', 'Professionals', 'Craftsmen'],
            'proletariat': ['Proletariat', 'Workers', 'Labor', 'Working Class', 'Laborers', 'Masses'],
            'state_memory': ['State Memory', 'Archives', 'Records', 'Bureaucracy', 'Memory', 'Documentation'],
            'shopkeepers': ['Shopkeepers/Merchants', 'Commerce', 'Trade', 'Merchants', 'Traders', 'Business']
        }
        
    def create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        
    def find_column(self, df, column_type):
        """Find the actual column name in the dataframe"""
        possible_names = self.column_mappings.get(column_type, [])
        
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    def standardize_node_name(self, node_name):
        """Standardize node names to CAMS format"""
        node_name = str(node_name).strip()
        
        for standard_name, variations in self.node_mappings.items():
            for variation in variations:
                if variation.lower() in node_name.lower() or node_name.lower() in variation.lower():
                    if standard_name == 'executive':
                        return 'Executive'
                    elif standard_name == 'army':
                        return 'Army'
                    elif standard_name == 'priests':
                        return 'Priests'
                    elif standard_name == 'property_owners':
                        return 'Property Owners'
                    elif standard_name == 'trades':
                        return 'Trades/Professions'
                    elif standard_name == 'proletariat':
                        return 'Proletariat'
                    elif standard_name == 'state_memory':
                        return 'State Memory'
                    elif standard_name == 'shopkeepers':
                        return 'Shopkeepers/Merchants'
        
        return node_name  # Return original if no match found
    
    def calculate_derived_metrics(self, df):
        """Calculate Node Value and Bond Strength if missing"""
        # Calculate Node Value if missing
        if 'Node value' not in df.columns:
            coherence_col = self.find_column(df, 'coherence')
            capacity_col = self.find_column(df, 'capacity')
            stress_col = self.find_column(df, 'stress')
            abstraction_col = self.find_column(df, 'abstraction')
            
            if all(col is not None for col in [coherence_col, capacity_col, stress_col, abstraction_col]):
                df['Node value'] = (df[coherence_col] + df[capacity_col] + 
                                  np.abs(df[stress_col]) + df[abstraction_col])
                print(f"  ✓ Calculated Node value from component dimensions")
        
        # Calculate Bond Strength if missing
        if 'Bond strength' not in df.columns and 'Node value' in df.columns:
            df['Bond strength'] = df['Node value'] * 0.6  # Standard multiplier
            print(f"  ✓ Calculated Bond strength from Node value")
            
        return df
    
    def clean_dataframe(self, df, filename):
        """Clean and standardize a dataframe"""
        print(f"\n📊 Processing: {filename}")
        print(f"   Original shape: {df.shape}")
        print(f"   Original columns: {list(df.columns)}")
        
        # Create standardized column mapping
        column_map = {}
        
        # Map standard columns
        for col_type, _ in self.column_mappings.items():
            found_col = self.find_column(df, col_type)
            if found_col:
                if col_type == 'year':
                    column_map[found_col] = 'Year'
                elif col_type == 'nation':
                    column_map[found_col] = 'Nation'
                elif col_type == 'node':
                    column_map[found_col] = 'Node'
                elif col_type == 'coherence':
                    column_map[found_col] = 'Coherence'
                elif col_type == 'capacity':
                    column_map[found_col] = 'Capacity'
                elif col_type == 'stress':
                    column_map[found_col] = 'Stress'
                elif col_type == 'abstraction':
                    column_map[found_col] = 'Abstraction'
        
        # Rename columns
        df = df.rename(columns=column_map)
        
        # Add Nation column if missing (extract from filename)
        if 'Nation' not in df.columns:
            # Try to extract nation from filename
            base_name = os.path.splitext(filename)[0].upper()
            if 'AUSTRALIA' in base_name or 'AU' in base_name:
                df['Nation'] = 'Australia'
            elif 'USA' in base_name or 'US' in base_name or 'AMERICA' in base_name:
                df['Nation'] = 'USA'
            elif 'UK' in base_name or 'BRITAIN' in base_name or 'ENGLAND' in base_name:
                df['Nation'] = 'UK'
            elif 'CHINA' in base_name or 'CN' in base_name:
                df['Nation'] = 'China'
            else:
                df['Nation'] = 'Unknown'
            print(f"  ✓ Added Nation column: {df['Nation'].iloc[0]}")
        
        # Standardize node names
        if 'Node' in df.columns:
            df['Node'] = df['Node'].apply(self.standardize_node_name)
            print(f"  ✓ Standardized node names")
        
        # Calculate derived metrics
        df = self.calculate_derived_metrics(df)
        
        # Ensure proper column order
        desired_columns = ['Nation', 'Year', 'Node', 'Coherence', 'Capacity', 'Stress', 'Abstraction']
        if 'Node value' in df.columns:
            desired_columns.append('Node value')
        if 'Bond strength' in df.columns:
            desired_columns.append('Bond strength')
        
        # Reorder columns
        available_columns = [col for col in desired_columns if col in df.columns]
        other_columns = [col for col in df.columns if col not in available_columns]
        df = df[available_columns + other_columns]
        
        print(f"   Final shape: {df.shape}")
        print(f"   Final columns: {list(df.columns)}")
        
        return df
    
    def validate_data(self, df, filename):
        """Validate data quality and ranges"""
        issues = []
        
        # Check required columns
        required_cols = ['Year', 'Node', 'Coherence', 'Capacity', 'Stress', 'Abstraction']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check data ranges
        if 'Coherence' in df.columns:
            coherence_range = (df['Coherence'].min(), df['Coherence'].max())
            if coherence_range[0] < -10 or coherence_range[1] > 10:
                issues.append(f"Coherence values outside expected range [-10,10]: {coherence_range}")
        
        if 'Capacity' in df.columns:
            capacity_range = (df['Capacity'].min(), df['Capacity'].max())
            if capacity_range[0] < -10 or capacity_range[1] > 10:
                issues.append(f"Capacity values outside expected range [-10,10]: {capacity_range}")
        
        if 'Stress' in df.columns:
            stress_range = (df['Stress'].min(), df['Stress'].max())
            if stress_range[0] < -10 or stress_range[1] > 10:
                issues.append(f"Stress values outside expected range [-10,10]: {stress_range}")
        
        if 'Abstraction' in df.columns:
            abstraction_range = (df['Abstraction'].min(), df['Abstraction'].max())
            if abstraction_range[0] < 0 or abstraction_range[1] > 10:
                issues.append(f"Abstraction values outside expected range [0,10]: {abstraction_range}")
        
        # Check for missing data
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            issues.append(f"Missing data points: {dict(missing_data[missing_data > 0])}")
        
        return issues
    
    def process_file(self, filepath):
        """Process a single file"""
        filename = os.path.basename(filepath)
        print(f"\n🔍 Found file: {filename}")
        
        try:
            # Read file based on extension
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filepath)
            elif filepath.endswith('.txt'):
                # Try common delimiters
                try:
                    df = pd.read_csv(filepath, delimiter='\t')
                except:
                    try:
                        df = pd.read_csv(filepath, delimiter=',')
                    except:
                        df = pd.read_csv(filepath, delimiter=';')
            else:
                print(f"  ❌ Unsupported file format: {filename}")
                return None, []
            
            # Clean the dataframe
            cleaned_df = self.clean_dataframe(df, filename)
            
            # Validate data
            issues = self.validate_data(cleaned_df, filename)
            
            return cleaned_df, issues
            
        except Exception as e:
            print(f"  ❌ Error processing {filename}: {e}")
            return None, [f"Processing error: {str(e)}"]
    
    def process_all_files(self):
        """Process all files in the input directory"""
        print("🏛️ CAMS Data Input Processor")
        print("=" * 50)
        
        # Create directories
        self.create_directories()
        
        # Find all data files
        file_patterns = ['*.csv', '*.xlsx', '*.xls', '*.txt']
        input_files = []
        
        for pattern in file_patterns:
            input_files.extend(glob.glob(os.path.join(self.input_dir, pattern)))
        
        if not input_files:
            print(f"\n📂 No data files found in {self.input_dir}")
            print("Please add your data files and run again.")
            return
        
        print(f"\n📂 Found {len(input_files)} files to process")
        
        # Process each file
        processed_files = []
        all_issues = {}
        
        for filepath in input_files:
            # Backup original file
            filename = os.path.basename(filepath)
            backup_path = os.path.join(self.backup_dir, filename)
            shutil.copy2(filepath, backup_path)
            
            # Process file
            cleaned_df, issues = self.process_file(filepath)
            
            if cleaned_df is not None:
                # Generate output filename
                base_name = os.path.splitext(filename)[0]
                nation = cleaned_df['Nation'].iloc[0] if 'Nation' in cleaned_df.columns else 'Unknown'
                output_filename = f"{nation}_CAMS_Cleaned.csv"
                output_path = os.path.join(self.output_dir, output_filename)
                
                # Save cleaned file
                cleaned_df.to_csv(output_path, index=False)
                processed_files.append(output_filename)
                print(f"  ✅ Saved: {output_filename}")
                
                # Store issues
                if issues:
                    all_issues[filename] = issues
            
        # Generate processing report
        self.generate_report(processed_files, all_issues)
        
        # Clean up input directory (move processed files to backup)
        for filepath in input_files:
            filename = os.path.basename(filepath)
            if not os.path.exists(os.path.join(self.backup_dir, filename)):
                shutil.move(filepath, self.backup_dir)
    
    def generate_report(self, processed_files, issues):
        """Generate processing report"""
        report_path = "data_processing_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("CAMS Data Processing Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"PROCESSED FILES ({len(processed_files)}):\n")
            f.write("-" * 30 + "\n")
            for filename in processed_files:
                f.write(f"✅ {filename}\n")
            
            if issues:
                f.write(f"\nDATA QUALITY ISSUES:\n")
                f.write("-" * 30 + "\n")
                for filename, file_issues in issues.items():
                    f.write(f"\n📋 {filename}:\n")
                    for issue in file_issues:
                        f.write(f"  ⚠️  {issue}\n")
            else:
                f.write(f"\n✅ No data quality issues detected.\n")
            
            f.write(f"\nNEXT STEPS:\n")
            f.write("-" * 30 + "\n")
            f.write("1. Review any data quality issues above\n")
            f.write("2. Run the dashboard: streamlit run dashboard.py\n")
            f.write("3. Cleaned files are ready for analysis\n")
            f.write("4. Original files backed up in data_input/originals/\n")
        
        print(f"\n📋 Processing report saved: {report_path}")
        print(f"✅ Processed {len(processed_files)} files successfully")
        
        if issues:
            print(f"⚠️  {len(issues)} files had data quality issues - check report")
        
        print("\n🚀 Ready for dashboard analysis!")

def main():
    processor = CAMSDataProcessor()
    processor.process_all_files()

if __name__ == "__main__":
    main()