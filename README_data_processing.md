# ABIDE Data Processing

This repository contains code for processing and analyzing the ABIDE (Autism Brain Imaging Data Exchange) dataset, focusing on statistical analysis and appropriate preprocessing techniques based on the data distribution.

## Overview

The `data_processing.py` script provides a complete pipeline for:

1. Loading and exploring the ABIDE brain imaging dataset
2. Analyzing data distributions and statistical properties
3. Detecting group differences
4. Identifying feature correlations
5. Determining optimal preprocessing methods
6. Applying appropriate transformations and preprocessing

## Requirements

To run this code, you need the following Python packages:
```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
```

You can install them using:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

To run the complete data processing pipeline:

```bash
python data_processing.py
```

This will:
1. Load the ABIDE dataset (`ABIDE2_sample.csv`)
2. Analyze the data characteristics
3. Perform statistical tests
4. Generate visualizations in the `plots/` directory
5. Apply appropriate preprocessing
6. Save the processed data to `ABIDE2_processed.csv`

### Key Functions

The script includes several key functions:

- `load_data()`: Loads the ABIDE dataset
- `analyze_basic_stats()`: Calculates and displays basic statistics
- `check_distribution()`: Analyzes data distributions using Shapiro-Wilk test and visualizations
- `analyze_group_differences()`: Detects differences between groups using Mann-Whitney U test
- `analyze_feature_correlations()`: Identifies correlations between brain features
- `get_optimal_preprocessing()`: Determines the best preprocessing methods based on data characteristics
- `preprocess_data()`: Applies the optimal preprocessing pipeline to the data

## Preprocessing Methods

The script automatically selects appropriate preprocessing techniques based on data characteristics:

1. **Power Transformation** (Yeo-Johnson) for highly skewed features
2. **Robust Scaling** when outliers are detected, or **Standard Scaling** otherwise
3. **Variance Threshold** to remove low-variance features
4. **Principal Component Analysis (PCA)** for dimensionality reduction when dealing with many features

## Output

The script generates several outputs:

1. Statistical analysis in the console output
2. Visualizations in the `plots/` directory:
   - Distribution analysis plots
   - Group differences boxplots
   - Feature correlation heatmap
   - PCA explained variance plot
3. Processed dataset saved as `ABIDE2_processed.csv`

## Customization

You can modify the script to:

- Change the input file path by editing the `file_path` parameter in `load_data()`
- Adjust thresholds for skewness detection or outlier handling
- Modify the number of features used in visualizations
- Change the dimensionality reduction parameters

## Notes

- The script is designed to handle the specific structure of the ABIDE dataset, which includes brain region measurements prefixed with "fs"
- The preprocessing pipeline is optimized for clustering and classification tasks
- Visualization outputs help understand the data distribution and validate preprocessing choices 