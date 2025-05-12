# 🧠 Math for Machine Learning: PCA & Clustering Analysis

![Machine Learning Banner](https://img.shields.io/badge/Machine%20Learning-Brain%20Connectivity%20Analysis-blue?style=for-the-badge&logo=brain)
![Python](https://img.shields.io/badge/Python-3.8+-yellow?style=for-the-badge&logo=python)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

> 📊 A comprehensive implementation of PCA and clustering techniques for data analysis, featuring custom implementations and advanced feature engineering methods.

## 📑 Table of Contents

- [Overview](#-overview)
- [Task 1: PCA on Iris Dataset](#-task-1-pca-on-iris-dataset)
- [Task 2: ABIDE Brain Imaging Clustering](#-task-2-abide-brain-imaging-clustering)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [The Enhanced Clustering Pipeline](#-the-enhanced-clustering-pipeline)
- [Usage Examples](#-usage-examples)
- [Advanced Features](#-advanced-features)
- [Troubleshooting](#-troubleshooting)
- [License & Acknowledgments](#-license--acknowledgments)

## 🔭 Overview

This project implements various machine learning techniques focusing on dimensionality reduction (PCA) and unsupervised clustering. It consists of two main tasks:

1. **PCA implementation and analysis on the Iris dataset** - A classical dataset for demonstrating dimensionality reduction
2. **Advanced clustering analysis on ABIDE brain imaging data** - Identifying natural groupings within brain connectivity data that might correspond to different autism spectrum disorder subtypes

## 🌸 Task 1: PCA on Iris Dataset

### Description

Task 1 involves implementing Principal Component Analysis (PCA) from scratch and applying it to the well-known Iris dataset. The custom PCA implementation is validated against scikit-learn's implementation.

### Features

- ✅ Custom PCA implementation without relying on scikit-learn
- ✅ Preprocessing and normalization of data
- ✅ Visualization of explained variance
- ✅ 2D data projection and visualization
- ✅ Validation against scikit-learn's PCA

### Running the Iris PCA Analysis

To run the PCA analysis on the Iris dataset:

```bash
python iris_pca.py
```

This script will:
1. Load the Iris dataset
2. Preprocess the data 
3. Apply our custom PCA implementation
4. Visualize the results
5. Compare with scikit-learn's PCA implementation

### Output and Evaluation

The script generates two plots in the `plots/` directory:

1. **Explained Variance Plot** (`iris_variance_explained.png`): Shows how much variance is explained by each principal component

   | Component | Explained Variance (%) | Cumulative Variance (%) |
   |-----------|------------------------|-------------------------|
   | PC1       | ~72.9%                | ~72.9%                 |
   | PC2       | ~22.8%                | ~95.7%                 |
   | PC3       | ~3.7%                 | ~99.4%                 |
   | PC4       | ~0.6%                 | 100%                   |

2. **2D Projection Plot** (`iris_pca_2d.png`): Visualizes the data projected onto the first two principal components, with points colored by class

   The script also outputs Mean Squared Error (MSE) between our implementation and scikit-learn's implementation, which should be very close to zero (<1e-15) if our implementation is correct.

## 🧠 Task 2: ABIDE Brain Imaging Clustering

This project implements unsupervised clustering techniques for brain imaging data from the ABIDE (Autism Brain Imaging Data Exchange) dataset. The goal is to identify natural groupings within the brain connectivity data that might correspond to different phenotypes or subtypes of autism spectrum disorder.

### Key Features

- **🧰 Advanced Feature Engineering**: Specialized preprocessing for brain connectivity data
  - Outlier detection and handling
  - Normalization techniques
  - Feature interactions
  - Brain region ratio creation

- **📉 Multiple Dimensionality Reduction Techniques**:
  - Standard PCA (Principal Component Analysis)
  - Advanced PCA with Whitening
  - Double PCA (sequential PCA application)
  - Kernel PCA with various kernels (RBF, polynomial, sigmoid)
  - Various whitening methods (ZCA, Cholesky, Mahalanobis, Adaptive, Hybrid)
  - Graph-based Feature Selection

- **🔍 Multiple Clustering Algorithms**:
  - K-Means (standard implementation)
  - Enhanced K-Means (with feature weighting, outlier handling, distance metrics)
  - Gaussian Mixture Models (GMM) with various covariance types

- **📊 Comprehensive Evaluation Metrics**:
  - Adjusted Rand Index (ARI)
  - Normalized Mutual Information (NMI)
  - Accuracy, Precision, Recall, and F1-score
  - Silhouette Score

## 📁 Project Structure

```
├── src/                    # Source code
│   ├── __init__.py         # Package initialization
│   ├── my_pca.py           # Custom PCA implementation
│   ├── my_kmeans.py        # Custom K-Means implementation
│   ├── enhanced_kmeans.py  # Enhanced K-Means implementation
│   ├── feature_engineering.py  # Basic feature engineering methods
│   ├── abide_feature_engineering.py  # Advanced feature engineering
│   ├── data_processing.py  # Data loading and preprocessing
│   ├── clustering_algorithms.py  # Various clustering algorithms
│   └── gmm_clustering.py   # GMM clustering implementation
├── utils/                  # Utility functions
│   ├── evaluation.py       # Clustering evaluation metrics
│   └── visualization.py    # Visualization utilities
├── plots/                  # Generated plots and visualizations
├── data/                   # Datasets
│   ├── ABIDE2_sample.csv   # Sample ABIDE dataset
│   ├── ABIDE2_processed.csv # Processed ABIDE dataset
│   └── ABIDE2(updated).csv # Updated ABIDE dataset
├── iris_pca.py             # Task 1: PCA on Iris dataset
├── abide_clustering.py     # Basic clustering script
├── abide_clustering_enhanced.py  # Enhanced clustering script
├── abide_clustering_compare.py   # Script for comparing methods
├── abide_feature_engineering_demo.py  # Feature engineering demo
├── abide_optimal_clustering.py   # Optimal clustering approach
├── compare_whitening_methods.py  # Compare whitening methods
└── README.md               # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Required packages:

  ```
  numpy
  pandas
  matplotlib
  scikit-learn
  networkx
  ```

### Installation

1. Clone the repository
2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the desired script:

   ```bash
   # For Task 1:
   python iris_pca.py
   
   # For Task 2:
   python abide_clustering_enhanced.py
   ```

## 🔄 The Enhanced Clustering Pipeline

The `abide_clustering_enhanced.py` script is the main component for Task 2, providing a comprehensive pipeline for clustering brain imaging data.

### Pipeline Overview

![Pipeline Overview](https://via.placeholder.com/800x200?text=Pipeline+Diagram)

The pipeline consists of the following steps:

1. **📥 Data Loading and Preprocessing**:
   - Loads the ABIDE dataset
   - Removes non-numeric columns
   - Handles missing values
   - Optional: Skips data processing if processed file exists

2. **⚠️ Outlier Detection and Handling**:
   - Identifies outliers using Z-score method
   - Either removes outliers or applies robust scaling
   - Customizable threshold

3. **🛠️ Feature Engineering**:
   - Applies specified feature engineering method
   - Multiple options from basic to advanced techniques
   - Customizable number of components

4. **🧩 Clustering**:
   - Applies multiple clustering algorithms
   - Compares clustering performance
   - Evaluates against true labels for validation

5. **📈 Results Analysis**:
   - Calculates evaluation metrics
   - Generates visualization plots
   - Saves results to CSV files

### Command-Line Arguments

<details>
<summary>🔍 Click to view all available arguments</summary>

#### Input/Output Options:
```bash
--input-file FILENAME      # Path to input CSV file (default: ABIDE2(updated).csv)
--output-dir DIRECTORY     # Directory to save output files and plots (default: plots)
```

#### Data Processing Options:
```bash
--skip-process             # Skip data processing if processed file exists
--force-process            # Force data processing even if processed file exists
```

#### Feature Engineering Options:
```bash
--components N             # Number of components for dimensionality reduction (default: 20)
--outlier-threshold T      # Z-score threshold for outlier detection (default: 5)
--whitening-method METHOD  # Whitening method to use for dimensionality reduction
                           # Choices: pca, zca, cholesky, mahalanobis, adaptive, graph, hybrid
--adaptive-alpha ALPHA     # Alpha parameter for adaptive whitening (0-1, default: 1.0)
--zca-weight WEIGHT        # Weight for ZCA in hybrid whitening (0-1, default: 0.5)
--advanced-fe METHOD       # Advanced feature engineering method to use
                           # Choices: standard, advanced_pca, optimal_kernel_pca, 
                           # feature_interactions_pca, region_ratios_pca, ensemble_reduction, 
                           # variance_interactions, optimize_for_clustering, combined_optimal, 
                           # zca_whitening_pca, whitening_graph_selection_pca, cholesky_whitening, 
                           # mahalanobis_whitening, adaptive_whitening, hybrid_whitening, double_pca
```

#### Clustering Options:
```bash
--method METHOD            # Clustering method to use (default: compare)
                           # Choices: kmeans, enhanced_kmeans, gmm, compare
--n-clusters N             # Number of clusters (default: 2)
```

#### Testing and Analysis Options:
```bash
--run-all                  # Run all analysis methods
--test-advanced-pca-range  # Test advanced_pca with components from 1 to 20
--test-standard-pca-range  # Test standard PCA with components from 1 to 30
--test-whitening-range M   # Test a specific whitening method with components from 1 to 30
                           # Choices: zca, graph, cholesky, mahalanobis, adaptive, hybrid
--test-double-pca-range    # Test double PCA with components from 1 to 30
```

#### Visualization Options:
```bash
--plot-pca                 # Plot PCA components variance
--plot-tsne                # Plot t-SNE visualization of clusters
```
</details>

### Feature Engineering Methods

<details>
<summary>🔍 Click to expand feature engineering methods</summary>

#### Basic Methods:
- **Standard PCA**: Standard Principal Component Analysis for dimensionality reduction. 
- **Advanced PCA with Whitening**: PCA with whitening transformation to decorrelate features. (THE BEST!!!)
- **Kernel PCA**: Non-linear dimensionality reduction using various kernels.
- **Feature Agglomeration**: Groups similar features to reduce dimensionality.
- **Variance-based Feature Selection**: Selects features based on their variance.
- **Select Percentile**: Selects features with highest scores according to a statistical test.

#### Advanced Methods:
- **Feature Interactions + PCA**: Creates interaction features and applies PCA.
- **Region Ratio Features + PCA**: Creates ratio features between brain regions.
- **Ensemble Dimensionality Reduction**: Combines multiple techniques.
- **Variance Selection + Interaction Features**: Selects features based on variance and adds interactions.
- **Optimized for Clustering**: Feature engineering optimized specifically for clustering tasks.
- **Combined Optimal Approach**: Multi-stage approach combining several techniques.
- **Double PCA**: Sequential application of PCA - first without components restriction, then with specified components.

#### Whitening Methods:
- **ZCA Whitening + PCA**: Zero-phase Component Analysis whitening followed by PCA.
- **Whitening + Graph Feature Selection + PCA**: Graph-based feature selection after whitening.
- **Cholesky Whitening**: Uses Cholesky decomposition for whitening.
- **Mahalanobis Whitening**: Applies Mahalanobis distance-based whitening.
- **Adaptive PCA Whitening**: PCA whitening with adjustable strength.
- **Hybrid Whitening**: Combines ZCA and adaptive whitening methods.
</details>

### Clustering Methods

<details>
<summary>🔍 Click to expand clustering methods</summary>

The pipeline implements and compares multiple clustering algorithms:

1. **Standard K-means**: Classic K-means clustering algorithm.
2. **Enhanced K-means**: Improved K-means with:
   - Feature weighting based on brain region importance
   - Outlier handling during clustering
   - Multiple distance metrics (euclidean, manhattan, RBF kernel)
   - Momentum for faster convergence
   - Early stopping for efficiency
3. **Gaussian Mixture Models (GMM)**: Probabilistic model with:
   - Different covariance types (full, tied)
   - Optimal component detection
   - BIC criterion for model selection
</details>

## 📋 Usage Examples

### Basic Usage

```bash
# Run with default settings
python abide_clustering_enhanced.py

# Use advanced PCA with 30 components
python abide_clustering_enhanced.py --advanced-fe advanced_pca --components (THE BEST!!!)

# Use double PCA with 25 components
python abide_clustering_enhanced.py --advanced-fe double_pca --components 25
```

### Testing Component Ranges

```bash
# Test standard PCA with components from 1 to 30
python abide_clustering_enhanced.py --test-standard-pca-range

# Test double PCA with components from 1 to 30
python abide_clustering_enhanced.py --test-double-pca-range

# Test ZCA whitening with components from 1 to 30
python abide_clustering_enhanced.py --test-whitening-range zca
```

### Whitening Method Examples

```bash
# Use ZCA whitening with 20 components
python abide_clustering_enhanced.py --whitening-method zca --components 20

# Use adaptive whitening with custom alpha parameter
python abide_clustering_enhanced.py --whitening-method adaptive --adaptive-alpha 0.7 --components 15

# Use hybrid whitening with custom parameters
python abide_clustering_enhanced.py --whitening-method hybrid --zca-weight 0.6 --adaptive-alpha 0.8
```

### Testing Multiple Configurations

```bash
# Test all whitening methods with 20 components
for method in pca zca cholesky mahalanobis adaptive graph hybrid; do
  python abide_clustering_enhanced.py --whitening-method $method --components 20
done
```

## 🔍 Advanced Features

### Component Range Testing

A key feature of the pipeline is testing different numbers of components for dimensionality reduction:

- `--test-standard-pca-range`: Tests standard PCA with components from 1 to 30
- `--test-advanced-pca-range`: Tests advanced PCA with components from 1 to 20
- `--test-double-pca-range`: Tests double PCA with components from 1 to 30
- `--test-whitening-range METHOD`: Tests a specific whitening method with components from 1 to 30

### Double PCA

One of the notable feature engineering methods is Double PCA, which applies PCA sequentially:

1. First PCA: Applied without restricting the number of components
2. Second PCA: Applied with specified number of components to the output of the first PCA

This approach is useful when:
- The dataset has very high dimensionality
- There are complex noise patterns
- The data contains hierarchical structures

To use Double PCA:
```bash
python abide_clustering_enhanced.py --advanced-fe double_pca --components 20
```

To find the optimal components for Double PCA:
```bash
python abide_clustering_enhanced.py --test-double-pca-range
```

## ⚠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| **Memory errors** | Reduce the number of samples or features using `--components` |
| **Runtime errors with specific methods** | Try alternative methods or adjust parameters |
| **Poor clustering performance** | Experiment with different feature engineering methods and components |
| **File not found errors** | Ensure all data files are in the correct location |
| **Library errors** | Verify all dependencies are installed with correct versions |

## 📜 License & Acknowledgments

This project is licensed under the MIT License.

### Acknowledgments

- ABIDE dataset: [http://fcon_1000.projects.nitrc.org/indi/abide/](http://fcon_1000.projects.nitrc.org/indi/abide/)
- The scikit-learn team for their excellent machine learning implementations
- The Matplotlib team for visualization capabilities 