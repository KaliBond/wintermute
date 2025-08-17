#!/usr/bin/env python3
"""
Simple test of the data loading logic from CAMS dashboard
"""

import pandas as pd
import glob

def load_available_datasets():
    """Load all available CAMS datasets - copied from dashboard"""
    # Look for CSV files in multiple locations
    csv_files = (glob.glob("*.csv") + 
                 glob.glob("data/cleaned/*.csv") + 
                 glob.glob("cleaned_datasets/*.csv"))
    datasets = {}
    
    # Enhanced country name mapping for cleaned datasets
    country_mapping = {
        'australia cleaned': 'Australia',
        'usa cleaned': 'USA', 
        'france cleaned': 'France',
        'italy cleaned': 'Italy',
        'italy19002025 cleaned': 'Italy (1900-2025)',
        'germany cleaned': 'Germany',
        'denmark cleaned': 'Denmark',
        'iran cleaned': 'Iran',
        'iraq cleaned': 'Iraq',
        'lebanon cleaned': 'Lebanon',
        'japan cleaned': 'Japan',
        'thailand 1850 2025 thailand 1850 2025 cleaned': 'Thailand',
        'netherlands cleaned': 'Netherlands',
        'canada cleaned': 'Canada',
        'saudi arabia cleaned': 'Saudi Arabia',
        'hong kong cleaned': 'Hong Kong',
        'hongkong manual cleaned': 'Hong Kong (Manual)',
        'england cleaned': 'England',
        'france 1785 1800 cleaned': 'France (1785-1800)',
        'france master 3 france 1785 1790 1795 1800 cleaned': 'France (Master)',
        'india cleaned': 'India',
        'indonesia cleaned': 'Indonesia',
        'israel cleaned': 'Israel',
        'new rome ad 5y rome 0 bce 5ad 10ad 15ad 20 ad cleaned': 'Ancient Rome',
        'pakistan cleaned': 'Pakistan',
        'russia cleaned': 'Russia',
        'singapore cleaned': 'Singapore',
        'syria cleaned': 'Syria',
        'usa highres cleaned': 'USA (HighRes)',
        'usa master cleaned': 'USA (Master)',
        'usa reconstructed cleaned': 'USA (Reconstructed)',
        'usa maximum 1790-2025 us high res 2025 (1) cleaned': 'USA (Maximum)',
        # Legacy mappings
        'australia cams cleaned': 'Australia',
        'usa cams cleaned': 'USA', 
        'france cams cleaned': 'France',
        'germany1750 2025': 'Germany',
        'denmark cams cleaned': 'Denmark',
        'iran cams cleaned': 'Iran',
        'iraq cams cleaned': 'Iraq',
        'lebanon cams cleaned': 'Lebanon',
        'japan 1850 2025': 'Japan',
        'thailand 1850_2025': 'Thailand',
        'netherlands mastersheet': 'Netherlands',
        'canada_cams_2025': 'Canada',
        'saudi arabia master file': 'Saudi Arabia',
        'eqmasterrome': 'Roman Empire',
        'new rome ad 5y': 'Early Rome'
    }
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if len(df) > 0:
                # Extract filename and remove path
                filename = file.split('/')[-1].split('\\')[-1]
                base_name = filename.replace('.csv', '').replace('.CSV', '').lower().strip()
                base_name = base_name.replace('_', ' ').replace(' (2)', '').replace(' - ', ' ')
                
                country_name = country_mapping.get(base_name, base_name.title())
                
                # Check if it's a node-based dataset with required columns
                required_cols = ['Node', 'Coherence', 'Capacity', 'Stress', 'Abstraction']
                has_required = all(col in df.columns for col in required_cols)
                has_year = 'Year' in df.columns
                
                if has_required and has_year:
                    # Standardize column names
                    df_clean = df.copy()
                    df_clean = df_clean.rename(columns={
                        'Coherence': 'C',
                        'Capacity': 'K', 
                        'Stress': 'S',
                        'Abstraction': 'A'
                    })
                    
                    # Ensure numeric values
                    for col in ['C', 'K', 'S', 'A']:
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    
                    # Remove rows with missing data
                    df_clean = df_clean.dropna(subset=['C', 'K', 'S', 'A'])
                    
                    if len(df_clean) > 0:
                        datasets[country_name] = {
                            'filename': file,
                            'data': df_clean,
                            'records': len(df_clean),
                            'years': f"{int(df_clean['Year'].min())}-{int(df_clean['Year'].max())}",
                            'nodes': sorted(df_clean['Node'].unique())
                        }
                        
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    return datasets

# Test the function
print("TESTING CAMS DASHBOARD DATA LOADING")
print("=" * 40)

datasets = load_available_datasets()

print(f"Total datasets loaded: {len(datasets)}")
print()

# Look for Hong Kong specifically
hk_datasets = {k: v for k, v in datasets.items() if 'hong kong' in k.lower()}

if hk_datasets:
    print("HONG KONG DATASETS FOUND:")
    print("-" * 30)
    for name, info in hk_datasets.items():
        print(f"Dataset: {name}")
        print(f"  Records: {info['records']}")
        print(f"  Years: {info['years']}")
        print(f"  Nodes: {len(info['nodes'])}")
        print(f"  File: {info['filename']}")
        
        # Test data structure
        df = info['data']
        print(f"  Columns: {list(df.columns)}")
        
        # Calculate basic metrics like dashboard would
        if all(col in df.columns for col in ['C', 'K', 'S', 'A']):
            print(f"  Mean Coherence: {df['C'].mean():.2f}")
            print(f"  Mean Capacity: {df['K'].mean():.2f}")
            print(f"  Mean Stress: {df['S'].mean():.2f}")
            print(f"  Mean Abstraction: {df['A'].mean():.2f}")
            
            # System health calculation (simplified)
            system_health = (df['C'].mean() + df['K'].mean() + df['A'].mean() - abs(df['S'].mean()))
            print(f"  System Health: {system_health:.2f}")
        print()
else:
    print("NO HONG KONG DATASETS FOUND!")
    print("\nAll available datasets:")
    for name in sorted(datasets.keys()):
        print(f"  - {name}")

print("Test completed!")