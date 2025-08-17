"""
Setup script for CAMS Framework
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cams-framework",
    version="1.0.0",
    author="Kari McKern",
    author_email="kari.freyr.4@gmail.com",
    description="Complex Adaptive Model State framework for analyzing societies as Complex Adaptive Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KaliBond/wintermute",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "dash>=2.10.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0",
        "jupyter>=1.0.0",
        "streamlit>=1.25.0",
        "dash-bootstrap-components>=1.4.0",
        "fastdtw>=0.3.4",
        "networkx>=3.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "cams-dashboard=dashboard:main",
            "cams-analyze=analysis_examples:main",
        ],
    },
    keywords="complex-adaptive-systems, civilization, analysis, societies, systems-theory",
    project_urls={
        "Bug Reports": "https://github.com/KaliBond/wintermute/issues",
        "Source": "https://github.com/KaliBond/wintermute",
        "Documentation": "https://github.com/KaliBond/wintermute/blob/main/FAQ",
    },
)