# CAMS Data Input Directory

## 📁 Purpose
This directory is for uploading new CAMS data files that need cleaning and processing.

## 📋 Supported File Formats
- **CSV files** (.csv)
- **Excel files** (.xlsx, .xls)
- **Text files** (.txt) with delimiter-separated values

## 🏛️ Expected Data Structure
Your files should contain data about societal nodes with these types of columns:

### Required Columns (any naming variation):
- **Year/Date**: Time period (1900, 2000, etc.)
- **Nation/Country**: Nation name (Australia, USA, etc.) 
- **Node**: Societal component (Executive, Army, Priests, Property Owners, Trades/Professions, Proletariat, State Memory, Shopkeepers/Merchants)
- **Coherence**: Alignment measure (-10 to 10)
- **Capacity**: Resource/competence measure (-10 to 10)
- **Stress**: Pressure measure (-10 to 10, usually negative)
- **Abstraction**: Innovation/complexity measure (0 to 10)

### Optional Columns:
- **Node Value**: Calculated composite score
- **Bond Strength**: Inter-node relationship strength

## 📥 How to Use This Directory

### Step 1: Place Your Files
Copy your raw data files into this directory:
```
data_input/
├── your_country_data.csv
├── historical_analysis.xlsx
├── raw_societal_metrics.txt
└── any_other_data_files.*
```

### Step 2: Run the Data Processor
From the wintermute directory, run:
```bash
python process_data_input.py
```

This will:
- ✅ Detect and analyze all files in this directory
- ✅ Clean and standardize column names
- ✅ Validate data ranges and formats
- ✅ Calculate missing Node Value and Bond Strength if needed
- ✅ Export cleaned files to the main directory
- ✅ Generate a processing report

### Step 3: Automatic Dashboard Integration
Cleaned files will be automatically available in the dashboard at:
http://localhost:8507

## 🔧 Data Cleaning Features

### Automatic Column Detection
- Handles various naming conventions (Year/year/DATE, Nation/Country/nation, etc.)
- Maps common variations to standard CAMS format
- Preserves original data while creating standardized versions

### Data Validation
- Checks value ranges for each dimension
- Identifies and flags potential data quality issues
- Suggests corrections for common problems

### Missing Data Handling  
- Calculates Node Value if missing: `Coherence + Capacity + |Stress| + Abstraction`
- Estimates Bond Strength if missing: `Node Value * 0.6`
- Interpolates missing years where possible

### Format Standardization
- Converts all files to consistent CSV format
- Standardizes column names and order
- Ensures compatibility with CAMS analysis framework

## 📊 Example Input Formats

### Format 1: Basic Structure
```csv
Country,Year,Component,Coherence,Capacity,Stress,Innovation
Australia,1900,Government,6.5,6.0,-4.0,5.5
Australia,1900,Military,5.0,4.5,-5.0,4.0
```

### Format 2: Full Structure  
```csv
Nation,Year,Node,Coherence,Capacity,Stress,Abstraction,Node value,Bond strength
USA,1790,Executive,5.0,4.0,-3.0,4.0,10.0,7.0
USA,1790,Army,4.0,3.0,-2.0,3.0,8.0,5.0
```

### Format 3: Alternative Naming
```csv
date,nation,societal_node,alignment,resources,pressure,complexity
2000,Germany,Leadership,7.2,8.1,-2.5,6.8
2000,Germany,Defense,6.0,7.5,-3.0,5.5
```

## ⚠️ Important Notes

### Data Quality Guidelines
- **Coherence/Capacity**: Usually positive values (-10 to +10)
- **Stress**: Usually negative values (-10 to +10)  
- **Abstraction**: Usually positive values (0 to 10)
- **Years**: Use 4-digit format (1900, 2000, not 00, 1900.0)

### Node Names
Standard CAMS nodes (flexible matching):
- Executive/Government/Leadership/Political
- Army/Military/Defense/Armed Forces
- Priests/Religious/Clergy/Ideology/Scientists
- Property Owners/Capitalists/Landowners/Wealthy
- Trades/Professions/Skilled Workers/Middle Class
- Proletariat/Workers/Labor/Working Class
- State Memory/Archives/Records/Bureaucracy
- Shopkeepers/Merchants/Commerce/Trade

## 🎯 Processing Results

After processing, you'll get:
- **Cleaned CSV files** in main directory
- **Processing report** with data quality assessment
- **Automatic dashboard integration**  
- **Backup of original files** (in `data_input/originals/`)

## 📞 Support

If you encounter issues with data processing:
1. Check the processing report for specific error messages
2. Verify your data matches the expected format
3. Run the debug processor: `python debug_data_input.py`

---

**Ready to process your CAMS data!** 🚀  
*Drop your files here and run the processor.*