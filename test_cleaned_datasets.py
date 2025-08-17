"""
Test cleaned datasets with CAMS-CAN v3.4 Explorer
"""

import pandas as pd
import glob
import os

def test_cleaned_datasets():
    """Test cleaned datasets for CAMS-CAN v3.4 compatibility"""
    print("Testing cleaned datasets for CAMS-CAN v3.4 compatibility")
    print("=" * 60)
    
    # Load cleaned datasets
    cleaned_dir = "cleaned_datasets"
    csv_files = glob.glob(os.path.join(cleaned_dir, "*.csv"))
    
    valid_datasets = {}
    issues = []
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        country_name = filename.replace('_cleaned.csv', '').replace('_', ' ')
        
        try:
            df = pd.read_csv(filepath)
            
            # Check required columns
            required_cols = ['Node', 'Coherence', 'Capacity', 'Stress', 'Abstraction']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                issues.append(f"{country_name}: Missing columns {missing_cols}")
                continue
            
            # Check data types
            numeric_issues = []
            for col in ['Coherence', 'Capacity', 'Stress', 'Abstraction']:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    numeric_issues.append(col)
            
            if numeric_issues:
                issues.append(f"{country_name}: Non-numeric columns {numeric_issues}")
                continue
            
            # Check for null values in critical columns
            null_counts = df[required_cols].isnull().sum()
            if null_counts.sum() > 0:
                issues.append(f"{country_name}: Null values detected - {null_counts.to_dict()}")
                continue
            
            # Check node count
            unique_nodes = len(df['Node'].unique())
            node_list = df['Node'].unique().tolist()
            
            # Check year availability
            has_year = 'Year' in df.columns
            year_range = ""
            if has_year:
                years = sorted(df['Year'].unique())
                year_range = f"{int(years[0])}-{int(years[-1])}" if len(years) > 1 else str(int(years[0]))
            
            valid_datasets[country_name] = {
                'filepath': filepath,
                'shape': df.shape,
                'nodes': unique_nodes,
                'node_list': node_list,
                'has_year': has_year,
                'year_range': year_range,
                'time_points': len(df['Year'].unique()) if has_year else 1
            }
            
            print(f"✅ {country_name}: {df.shape[0]} rows, {unique_nodes} nodes, {year_range}")
            
        except Exception as e:
            issues.append(f"{country_name}: Error loading - {str(e)}")
    
    print(f"\nSUMMARY:")
    print(f"Valid datasets: {len(valid_datasets)}")
    print(f"Issues found: {len(issues)}")
    
    if issues:
        print(f"\nISSUES:")
        for issue in issues:
            print(f"  - {issue}")
    
    # Test with sample dataset
    if valid_datasets:
        print(f"\nTesting sample calculations with Australia...")
        australia_path = os.path.join(cleaned_dir, "Australia_cleaned.csv")
        if os.path.exists(australia_path):
            test_sample_calculations(australia_path)
    
    return valid_datasets, issues

def test_sample_calculations(filepath):
    """Test CAMS calculations with sample data"""
    import numpy as np
    
    df = pd.read_csv(filepath)
    
    # Get latest year data
    if 'Year' in df.columns:
        latest_year = df['Year'].max()
        latest_data = df[df['Year'] == latest_year]
    else:
        latest_data = df
    
    print(f"Sample data from {os.path.basename(filepath)} (Year: {latest_year if 'Year' in df.columns else 'N/A'})")
    print(f"Institutions: {', '.join(latest_data['Node'].tolist())}")
    
    # Calculate sample node fitness
    def calculate_fitness(row):
        C, K, S, A = row['Coherence'], row['Capacity'], row['Stress'], row['Abstraction']
        tau, lam = 3.0, 0.5
        stress_impact = 1 + np.exp((abs(S) - tau) / lam)
        return (C * K / stress_impact) * (1 + A / 10)
    
    # Test calculations
    fitness_values = []
    for _, row in latest_data.iterrows():
        fitness = calculate_fitness(row)
        fitness_values.append(fitness)
        print(f"  {row['Node']}: Fitness = {fitness:.3f}")
    
    # System health (geometric mean)
    system_health = np.exp(np.mean(np.log(fitness_values)))
    print(f"\nSystem Health (geometric mean): {system_health:.3f}")
    
    # Coherence asymmetry
    coherence_capacity = latest_data['Coherence'] * latest_data['Capacity']
    coherence_asymmetry = np.std(coherence_capacity) / (np.mean(coherence_capacity) + 1e-9)
    print(f"Coherence Asymmetry: {coherence_asymmetry:.3f}")
    
    print("✅ Sample calculations completed successfully!")

if __name__ == "__main__":
    valid_datasets, issues = test_cleaned_datasets()
    
    if len(valid_datasets) > 0:
        print(f"\n🎉 {len(valid_datasets)} datasets are ready for CAMS-CAN v3.4!")
    else:
        print(f"\n❌ No valid datasets available.")