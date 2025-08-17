"""
Automatic GitHub Data Import Script for CAMS Framework
Automatically discovers and imports CSV files from a GitHub repository
"""

import os
import sys
import requests
import pandas as pd
from datetime import datetime
import json

def get_github_repo_files(owner, repo, path=""):
    """Get all files from a GitHub repository using the API"""
    print(f"Scanning GitHub repository: {owner}/{repo}")
    
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        
        files = response.json()
        all_files = []
        
        for item in files:
            if item['type'] == 'file':
                all_files.append(item)
            elif item['type'] == 'dir':
                # Recursively scan subdirectories
                subdir_files = get_github_repo_files(owner, repo, item['path'])
                all_files.extend(subdir_files)
                
        return all_files
        
    except requests.exceptions.RequestException as e:
        print(f"Error accessing repository: {e}")
        return []

def download_file(download_url, local_path):
    """Download a file from a URL"""
    try:
        response = requests.get(download_url)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            f.write(response.content)
        
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def validate_cams_file(filepath):
    """Validate if a CSV file is suitable for CAMS analysis"""
    try:
        df = pd.read_csv(filepath)
        
        # Check for CAMS-relevant columns
        columns = [col.lower() for col in df.columns]
        required_keywords = ['year', 'node', 'coherence', 'capacity', 'stress']
        
        found_keywords = sum(1 for keyword in required_keywords 
                           if any(keyword in col for col in columns))
        
        is_cams_file = found_keywords >= 3  # At least 3 CAMS keywords
        
        return {
            'valid': is_cams_file,
            'records': len(df),
            'columns': list(df.columns),
            'cams_score': found_keywords,
            'size_mb': os.path.getsize(filepath) / 1024 / 1024
        }
        
    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }

def auto_import_from_github(owner, repo):
    """Automatically import CSV files from a GitHub repository"""
    print("=" * 60)
    print("AUTOMATIC GITHUB DATA IMPORT FOR CAMS")
    print("=" * 60)
    print(f"Repository: https://github.com/{owner}/{repo}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Get all files from repository
    all_files = get_github_repo_files(owner, repo)
    
    if not all_files:
        print("No files found or repository not accessible")
        return
    
    # Filter for CSV files
    csv_files = [f for f in all_files if f['name'].lower().endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in repository")
        return
    
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f['name']} ({f['size']} bytes)")
    print()
    
    # Download and process files
    successful_imports = []
    failed_imports = []
    
    for file_info in csv_files:
        filename = file_info['name']
        download_url = file_info['download_url']
        
        print(f"Processing: {filename}")
        
        # Download file
        if download_file(download_url, filename):
            print(f"  [SUCCESS] Downloaded successfully")
            
            # Validate file
            validation = validate_cams_file(filename)
            
            if validation['valid']:
                print(f"  [VALID] CAMS validation passed")
                print(f"    Records: {validation['records']:,}")
                print(f"    Columns: {len(validation['columns'])}")
                print(f"    CAMS score: {validation['cams_score']}/5")
                
                successful_imports.append({
                    'filename': filename,
                    'records': validation['records'],
                    'columns': validation['columns']
                })
                
                # Test with CAMS analyzer if available
                try:
                    sys.path.append('src')
                    from cams_analyzer import CAMSAnalyzer
                    
                    analyzer = CAMSAnalyzer()
                    df = pd.read_csv(filename)
                    
                    # Test system health calculation
                    year_col = analyzer._get_column_name(df, 'year')
                    latest_year = df[year_col].max()
                    health = analyzer.calculate_system_health(df, latest_year)
                    
                    print(f"    System Health ({latest_year}): {health:.2f}")
                    
                except Exception as e:
                    print(f"    CAMS test failed: {e}")
                
            else:
                print(f"  [WARNING] Not a CAMS file (score: {validation.get('cams_score', 0)}/5)")
                if 'error' in validation:
                    print(f"    Error: {validation['error']}")
                failed_imports.append(filename)
        else:
            print(f"  [ERROR] Download failed")
            failed_imports.append(filename)
        
        print()
    
    # Summary
    print("=" * 60)
    print("IMPORT SUMMARY")
    print("=" * 60)
    print(f"Total CSV files found: {len(csv_files)}")
    print(f"Successfully imported: {len(successful_imports)}")
    print(f"Failed/Invalid: {len(failed_imports)}")
    
    if successful_imports:
        total_records = sum(f['records'] for f in successful_imports)
        print(f"Total records imported: {total_records:,}")
        print()
        print("Successfully imported files:")
        for f in successful_imports:
            print(f"  [SUCCESS] {f['filename']}: {f['records']:,} records")
            
        print()
        print("[COMPLETED] Import finished! Your new data is ready for analysis.")
        print("Run the dashboard to explore your updated datasets:")
        print("  streamlit run dashboard.py")
    
    if failed_imports:
        print()
        print("Files that couldn't be imported:")
        for f in failed_imports:
            print(f"  [FAILED] {f}")

def main():
    """Main function with repository discovery"""
    print("CAMS Framework - Automatic GitHub Import")
    print("=" * 45)
    
    # Check if we can auto-detect from known patterns
    common_repos = [
        ("your_username", "wintermute"),
        ("julie", "wintermute"), 
        ("juliemckern", "wintermute")
    ]
    
    # Try to get repository info from user
    print("Please provide your GitHub repository information:")
    print()
    
    owner = input("GitHub username: ").strip()
    if not owner:
        print("ERROR: GitHub username is required")
        return
    
    repo = input("Repository name [wintermute]: ").strip() or "wintermute"
    
    print(f"\nAttempting to import from: https://github.com/{owner}/{repo}")
    print("Press Enter to continue or Ctrl+C to cancel...")
    input()
    
    # Run the import
    auto_import_from_github(owner, repo)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nImport cancelled by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Please check your repository information and try again.")