"""
Quick import script for CAMS GitHub data
Usage: python quick_import.py <github_username> [repo_name]
"""

import sys
import os

# Add arguments parsing
if len(sys.argv) < 2:
    print("Usage: python quick_import.py <github_username> [repo_name]")
    print("Example: python quick_import.py julie wintermute")
    sys.exit(1)

github_user = sys.argv[1]
repo_name = sys.argv[2] if len(sys.argv) > 2 else "wintermute"

print(f"Quick importing from: https://github.com/{github_user}/{repo_name}")

# Import the auto import function
from auto_import_github import auto_import_from_github

# Run the import
auto_import_from_github(github_user, repo_name)