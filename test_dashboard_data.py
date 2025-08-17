#!/usr/bin/env python3
"""
Test script to verify CAMS dashboard data loading works correctly
"""

import sys
sys.path.append('.')

# Import the data loading function from the dashboard
import importlib.util
spec = importlib.util.spec_from_file_location("cams_explorer", "cams_can_v34_explorer.py")
cams_explorer = importlib.util.module_from_spec(spec)

# Mock streamlit for testing
class MockStreamlit:
    def cache_data(self, func):
        return func
    def warning(self, msg):
        print(f"Warning: {msg}")

import sys
sys.modules['streamlit'] = MockStreamlit()
sys.modules['st'] = MockStreamlit()

spec.loader.exec_module(cams_explorer)

# Test the data loading
print("TESTING CAMS DASHBOARD DATA LOADING")
print("=" * 40)

datasets = cams_explorer.load_available_datasets()

print(f"Total datasets loaded: {len(datasets)}")
print()

# Look for Hong Kong specifically
hk_datasets = {k: v for k, v in datasets.items() if 'hong kong' in k.lower()}

if hk_datasets:
    print("HONG KONG DATASETS FOUND:")
    print("-" * 25)
    for name, info in hk_datasets.items():
        print(f"Dataset: {name}")
        print(f"  Records: {info['records']}")
        print(f"  Years: {info['years']}")
        print(f"  Nodes: {len(info['nodes'])}")
        print(f"  File: {info['filename']}")
        
        # Test data structure
        df = info['data']
        print(f"  Columns: {list(df.columns)}")
        
        # Test some calculations
        if 'C' in df.columns:
            sample_data = df.head()
            print(f"  Sample coherence values: {sample_data['C'].tolist()}")
            print(f"  Mean coherence: {df['C'].mean():.2f}")
            print(f"  Mean capacity: {df['K'].mean():.2f}")
            print(f"  Mean stress: {df['S'].mean():.2f}")
        print()
else:
    print("❌ NO HONG KONG DATASETS FOUND!")
    print("\nAll available datasets:")
    for name in sorted(datasets.keys()):
        print(f"  - {name}")

print("\n" + "=" * 40)
print("Test completed!")