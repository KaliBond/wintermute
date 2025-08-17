"""
Fix corrupted CAMS data files
"""

import pandas as pd
import os

def fix_corrupted_csv(filename):
    """Fix CSV files with corrupted headers"""
    print(f"Fixing: {filename}")
    
    try:
        # Read the raw file to examine structure
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"Total lines: {len(lines)}")
        print(f"First line (header): {lines[0][:100]}...")
        
        # Find the actual data start
        data_start_idx = 0
        expected_columns = ['Nation', 'Year', 'Node', 'Coherence', 'Capacity', 'Stress', 'Abstraction']
        
        for i, line in enumerate(lines):
            # Look for a line that contains expected column patterns
            if any(col in line for col in expected_columns):
                # Check if this looks like a proper header
                parts = line.strip().split(',')
                clean_parts = [part.strip().strip('"') for part in parts]
                
                # If we find columns that match our expected format
                if len(clean_parts) >= 7:  # At least 7 columns expected
                    data_start_idx = i
                    print(f"Found proper header at line {i+1}: {clean_parts[:5]}...")
                    break
        
        if data_start_idx > 0:
            # Skip the corrupted lines and use the proper header
            print(f"Skipping {data_start_idx} corrupted lines")
            df = pd.read_csv(filename, skiprows=data_start_idx)
        else:
            # If we can't find a proper header, try to reconstruct
            print("No proper header found, attempting reconstruction...")
            df = pd.read_csv(filename, header=None)
            
            # Try to find the row with actual data
            for idx, row in df.iterrows():
                if pd.notna(row[1]) and str(row[1]).isdigit() and len(str(row[1])) == 4:  # Year column
                    print(f"Found data starting at row {idx}")
                    # Use this row onwards and set proper headers
                    data_df = df.iloc[idx:].copy()
                    data_df.columns = ['Nation', 'Year', 'Node', 'Coherence', 'Capacity', 'Stress', 'Abstraction', 'Node value', 'Bond strength'][:len(data_df.columns)]
                    df = data_df
                    break
        
        # Clean column names
        df.columns = [col.strip().strip('"') for col in df.columns]
        
        # Map common column variations to standard names
        column_mapping = {}
        for col in df.columns:
            clean_col = col.strip().lower()
            if 'nation' in clean_col or 'country' in clean_col:
                column_mapping[col] = 'Nation'
            elif 'year' in clean_col or 'date' in clean_col:
                column_mapping[col] = 'Year'
            elif 'node' in clean_col or 'component' in clean_col:
                column_mapping[col] = 'Node'
            elif 'coherence' in clean_col:
                column_mapping[col] = 'Coherence'
            elif 'capacity' in clean_col:
                column_mapping[col] = 'Capacity'
            elif 'stress' in clean_col:
                column_mapping[col] = 'Stress'
            elif 'abstraction' in clean_col:
                column_mapping[col] = 'Abstraction'
            elif 'node value' in clean_col or 'nodevalue' in clean_col:
                column_mapping[col] = 'Node value'
            elif 'bond strength' in clean_col or 'bondstrength' in clean_col:
                column_mapping[col] = 'Bond strength'
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
            print(f"Mapped columns: {column_mapping}")
        
        # Remove any completely empty rows
        df = df.dropna(how='all')
        
        # Remove rows where Year is not a valid number
        if 'Year' in df.columns:
            df = df[pd.to_numeric(df['Year'], errors='coerce').notna()]
        
        print(f"Final shape: {df.shape}")
        print(f"Final columns: {list(df.columns)}")
        
        # Create backup of original
        backup_filename = filename + '.backup'
        if not os.path.exists(backup_filename):
            os.rename(filename, backup_filename)
            print(f"Original backed up as: {backup_filename}")
        
        # Save fixed file
        df.to_csv(filename, index=False)
        print(f"SUCCESS: Fixed file saved: {filename}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error fixing {filename}: {e}")
        return None

def main():
    print("CAMS Data File Fixer")
    print("=" * 40)
    
    # Check for corrupted files
    files_to_check = ['Australia_CAMS_Cleaned.csv', 'USA_CAMS_Cleaned.csv']
    
    for filename in files_to_check:
        if os.path.exists(filename):
            # Quick check if file is corrupted
            try:
                df = pd.read_csv(filename)
                columns = list(df.columns)
                
                # Check if first column looks corrupted (very long or contains instruction text)
                if len(columns[0]) > 50 or 'analyze' in columns[0].lower() or 'dataset' in columns[0].lower():
                    print(f"ALERT: Detected corrupted file: {filename}")
                    fix_corrupted_csv(filename)
                else:
                    print(f"SUCCESS: File looks good: {filename}")
                    print(f"   Columns: {columns}")
            except Exception as e:
                print(f"ALERT: Error reading {filename}: {e}")
                fix_corrupted_csv(filename)
        else:
            print(f"WARNING: File not found: {filename}")
    
    print("\nFile fixing complete!")

if __name__ == "__main__":
    main()