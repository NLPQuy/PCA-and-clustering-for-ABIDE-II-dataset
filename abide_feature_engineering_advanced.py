#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced feature engineering specialized for ABIDE brain imaging data.
This module implements techniques to enhance class separation in ABIDE dataset
prior to unsupervised clustering.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD, FastICA
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import FeatureAgglomeration
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
import os
import argparse

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

def load_and_prepare_abide_data(filepath='ABIDE2_sample.csv', target_column='DX_GROUP'):
    """
    Load ABIDE data and prepare it for analysis.
    
    Parameters:
    -----------
    filepath : str
        Path to the ABIDE dataset CSV file
    target_column : str
        Name of the target column (diagnostic group)
        
    Returns:
    --------
    X : numpy array
        Features (brain connectivity data)
    y : numpy array
        Target (diagnostic group labels)
    feature_names : list
        Names of the brain connectivity features
    """
    # Load the data
    print(f"Loading data from {filepath}...")
    data = pd.read_csv(filepath)
    
    # Extract target column
    if target_column in data.columns:
        y = data[target_column].values
    else:
        print(f"Warning: Target column '{target_column}' not found. Using first column as target.")
        y = data.iloc[:, 0].values
    
    # Extract features (all columns except those that are not connectivity data)
    non_feature_columns = ['Unnamed: 0', target_column, 'site', 'subject', 'age', 'group'] 
    feature_columns = [col for col in data.columns if col not in non_feature_columns]
    
    # If no feature columns are found, use all columns except the first and target
    if not feature_columns:
        feature_columns = [col for col in data.columns if col not in [data.columns[0], target_column]]
    
    # Convert to numeric, forcing error columns to NaN
    X_df = data[feature_columns].apply(pd.to_numeric, errors='coerce')
    
    # Fill any NaN values with column means
    X_df = X_df.fillna(X_df.mean())
    
    # Convert to numpy array
    X = X_df.values
    
    print(f"Loaded dataset with {X.shape[0]} samples and {X.shape[1]} features")
    
    # Convert y to binary (if it's not already)
    if target_column == 'group':
        # Map 'Cancer'/'Normal' to 1/0
        y = np.array([1 if val == 'Cancer' else 0 for val in y])
    else:
        # Try to convert to numeric if possible
        try:
            y = y.astype(int)
        except:
            # If conversion fails, create binary encoding
            unique_vals = np.unique(y)
            y_map = {val: i for i, val in enumerate(unique_vals)}
            y = np.array([y_map[val] for val in y])
    
    print(f"Class distribution: {y}")
    
    return X, y, feature_columns

def normalize_connectivity_data(X, method='robust'):
    """
    Normalize brain connectivity data using various methods.
    
    Parameters:
    -----------
    X : array-like
        Brain connectivity data
    method : str, default='robust'
        Normalization method: 'standard', 'robust', or 'minmax'
        
    Returns:
    --------
    X_normalized : array-like
        Normalized data
    scaler : object
        Fitted scaler object
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    X_normalized = scaler.fit_transform(X)
    return X_normalized, scaler

def remove_site_effects(X, site_labels):
    """
    Remove site effects from connectivity data using regression.
    
    Parameters:
    -----------
    X : array-like
        Brain connectivity data
    site_labels : array-like
        Site IDs for each sample
        
    Returns:
    --------
    X_corrected : array-like
        Data with site effects removed
    """
    from sklearn.linear_model import LinearRegression
    
    # Create site dummy variables
    site_dummies = pd.get_dummies(site_labels, prefix='site')
    
    # For each feature, regress out site effects
    X_corrected = np.zeros_like(X)
    
    for i in range(X.shape[1]):
        # Fit linear model: feature ~ site
        model = LinearRegression().fit(site_dummies.values, X[:, i])
        
        # Predict site effects
        site_effects = model.predict(site_dummies.values)
        
        # Subtract site effects from the original data
        X_corrected[:, i] = X[:, i] - site_effects
    
    return X_corrected

def detect_and_handle_outliers(X, threshold=3.0, method='winsorize'):
    """
    Detect and handle outliers in brain connectivity data.
    
    Parameters:
    -----------
    X : array-like
        Brain connectivity data
    threshold : float, default=3.0
        Z-score threshold for outlier detection
    method : str, default='winsorize'
        Method for handling outliers: 'remove', 'winsorize', or 'none'
        
    Returns:
    --------
    X_processed : array-like
        Data with outliers handled
    outlier_mask : array-like
        Boolean mask indicating outlier samples (if method='remove')
    """
    # Calculate z-scores
    z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
    
    # Find samples with any feature having z-score > threshold
    outlier_mask = np.any(z_scores > threshold, axis=1)
    outlier_count = np.sum(outlier_mask)
    
    print(f"Detected {outlier_count} samples ({outlier_count/len(X)*100:.1f}%) with outliers")
    
    if method == 'none':
        return X, outlier_mask
    
    elif method == 'remove':
        if outlier_count < len(X) * 0.3:  # Only remove if < 30% are outliers
            X_processed = X[~outlier_mask]
            print(f"Removed {outlier_count} outlier samples")
            return X_processed, outlier_mask
        else:
            print("Too many outliers to remove. Consider using 'winsorize' instead.")
            return X, outlier_mask
    
    elif method == 'winsorize':
        # Winsorize: cap extreme values at the threshold
        X_processed = X.copy()
        
        for col in range(X.shape[1]):
            mean = np.mean(X[:, col])
            std = np.std(X[:, col])
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            
            # Cap values outside bounds
            X_processed[:, col] = np.clip(X_processed[:, col], lower_bound, upper_bound)
        
        print(f"Winsorized outliers in {outlier_count} samples")
        return X_processed, outlier_mask
    
    else:
        raise ValueError(f"Unknown outlier handling method: {method}")

def create_connectivity_ratios(X, feature_names=None, top_n=20):
    """
    Create ratio features from brain connectivity data to enhance separability.
    
    Parameters:
    -----------
    X : array-like
        Brain connectivity data
    feature_names : list, optional
        Names of the brain connectivity features
    top_n : int, default=20
        Number of top ratio features to return
        
    Returns:
    --------
    X_with_ratios : array-like
        Data with added ratio features
    ratio_feature_names : list
        Names of the new ratio features
    """
    n_samples, n_features = X.shape
    
    # If no feature names provided, create generic names
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Find the most variable features
    variances = np.var(X, axis=0)
    top_indices = np.argsort(-variances)[:min(100, n_features)]
    
    # Create ratios between top features
    ratio_features = []
    ratio_feature_names = []
    
    for i, idx1 in enumerate(top_indices[:10]):  # Limit to avoid combinatorial explosion
        for j, idx2 in enumerate(top_indices[i+1:15]):
            # Avoid division by zero by adding small constant
            ratio = X[:, idx1] / (X[:, idx2] + 1e-10)
            ratio_features.append(ratio)
            ratio_feature_names.append(f"ratio_{feature_names[idx1]}_{feature_names[idx2]}")
    
    # Convert to array and combine with original features
    ratio_features = np.column_stack(ratio_features)
    X_with_ratios = np.hstack((X, ratio_features))
    
    print(f"Added {ratio_features.shape[1]} ratio features")
    
    # Return original features plus top ratio features
    return X_with_ratios, ratio_feature_names

def apply_advanced_pca(X, n_components=None, transform_method='standard'):
    """
    Apply advanced PCA and variants to brain connectivity data.
    
    Parameters:
    -----------
    X : array-like
        Brain connectivity data
    n_components : int or None
        Number of components to keep. If None, determine automatically.
    transform_method : str, default='standard'
        Method for dimensionality reduction:
        - 'pca': Standard PCA
        - 'kernel_pca': Kernel PCA with RBF kernel
        - 'polynomial_pca': PCA with polynomial features
        - 'ica': Independent Component Analysis
        - 'truncated_svd': Truncated SVD
        - 'tsne': t-SNE
        
    Returns:
    --------
    X_transformed : array-like
        Data with dimensionality reduction applied
    """
    # Set default number of components if not provided
    if n_components is None:
        n_components = min(20, X.shape[0] // 5, X.shape[1])
    
    try:
        # Apply the selected method
        if transform_method == 'pca':
            model = PCA(n_components=n_components, random_state=42)
            X_transformed = model.fit_transform(X)
            explained_variance_ratio = model.explained_variance_ratio_
            
            # Plot explained variance
            plt.figure(figsize=(10, 6))
            plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
            plt.plot(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio), 'r-o')
            plt.xlabel('Principal Component')
            plt.ylabel('Explained Variance Ratio')
            plt.title('PCA Explained Variance')
            plt.axhline(y=0.8, linestyle='--', color='g', label='80% Threshold')
            plt.grid(True, alpha=0.3)
            plt.savefig('plots/pca_explained_variance.png')
            plt.close()
        
        elif transform_method == 'kernel_pca':
            # Use a smaller gamma value for stability
            model = KernelPCA(n_components=n_components, kernel='rbf', gamma=0.01, 
                             random_state=42, fit_inverse_transform=True, alpha=0.1)
            X_transformed = model.fit_transform(X)
        
        elif transform_method == 'polynomial_pca':
            # First apply polynomial transformation, then PCA
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            X_poly = poly.fit_transform(X)
            
            # Apply PCA to polynomial features
            model = PCA(n_components=n_components, random_state=42)
            X_transformed = model.fit_transform(X_poly)
        
        elif transform_method == 'ica':
            model = FastICA(n_components=n_components, random_state=42, max_iter=1000, tol=0.01)
            X_transformed = model.fit_transform(X)
        
        elif transform_method == 'truncated_svd':
            model = TruncatedSVD(n_components=n_components, random_state=42)
            X_transformed = model.fit_transform(X)
        
        elif transform_method == 'tsne':
            from sklearn.manifold import TSNE
            model = TSNE(n_components=min(n_components, 3), 
                        perplexity=min(30, X.shape[0] // 10), 
                        learning_rate='auto',
                        init='pca', 
                        random_state=42)
            X_transformed = model.fit_transform(X)
        
        else:
            raise ValueError(f"Unknown transform method: {transform_method}")
        
        print(f"Applied {transform_method} dimensionality reduction: {X.shape} -> {X_transformed.shape}")
        
        return X_transformed
    
    except Exception as e:
        print(f"Error applying {transform_method}: {str(e)}")
        # Fall back to standard PCA
        print(f"Falling back to standard PCA")
        model = PCA(n_components=n_components, random_state=42)
        X_transformed = model.fit_transform(X)
        return X_transformed

def optimize_clustering_preprocessing(X, y, methods=None, n_components=20):
    """
    Find the optimal preprocessing pipeline for clustering.
    
    Parameters:
    -----------
    X : array-like
        Brain connectivity data
    y : array-like
        Target labels (only used for evaluation)
    methods : list or None
        List of preprocessing methods to try
    n_components : int
        Number of components to keep
        
    Returns:
    --------
    best_method : str
        Best preprocessing method
    best_components : int
        Best number of components
    """
    if methods is None:
        methods = ['pca', 'kernel_pca', 'polynomial_pca', 'ica', 'truncated_svd']
    
    print("\nOptimizing preprocessing for clustering...")
    
    results = {}
    max_score = -1
    best_method = None
    best_components = None
    
    # Find optimal preprocessing method
    for method in methods:
        print(f"Testing {method}...")
        
        # Try different numbers of components
        component_scores = {}
        for n in [2, 5, 10, 15, 20, 30, 40]:
            if n > min(X.shape[0], X.shape[1]):
                continue
                
            # Apply dimensionality reduction
            X_reduced = apply_advanced_pca(X, n_components=n, transform_method=method)
            
            # Apply K-means clustering
            from sklearn.cluster import KMeans
            n_clusters = len(np.unique(y))
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = km.fit_predict(X_reduced)
            
            # Evaluate clustering (silhouette score)
            try:
                silhouette = silhouette_score(X_reduced, cluster_labels)
                component_scores[n] = silhouette
                
                # Check if this is the best score so far
                if silhouette > max_score:
                    max_score = silhouette
                    best_method = method
                    best_components = n
                
                print(f"  {method} with {n} components: silhouette = {silhouette:.4f}")
            except:
                print(f"  {method} with {n} components: Failed to compute silhouette")
        
        results[method] = component_scores
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for method, scores in results.items():
        components = list(scores.keys())
        silhouettes = list(scores.values())
        plt.plot(components, silhouettes, 'o-', label=method)
    
    plt.xlabel('Number of Components')
    plt.ylabel('Silhouette Score')
    plt.title('Preprocessing Optimization Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/preprocessing_optimization.png')
    plt.close()
    
    print(f"Best preprocessing: {best_method} with {best_components} components (silhouette = {max_score:.4f})")
    
    return best_method, best_components

def plot_dimensionality_reduction_comparison(X, y, methods=None, n_components=2):
    """
    Compare and visualize different dimensionality reduction methods on brain data.
    
    Parameters:
    -----------
    X : array-like
        Brain connectivity data
    y : array-like
        Target labels (for coloring)
    methods : list or None
        List of dimensionality reduction methods to compare
    n_components : int, default=2
        Number of components for visualization
    """
    if methods is None:
        methods = ['pca', 'kernel_pca', 'tsne', 'polynomial_pca', 'ica']
    
    # Limit to 2 or 3 components for visualization
    n_components = min(n_components, 3)
    
    # Create subplots
    fig = plt.figure(figsize=(15, 10))
    
    for i, method in enumerate(methods):
        try:
            # Apply dimensionality reduction
            X_reduced = apply_advanced_pca(X, n_components=n_components, transform_method=method)
            
            # Create plot
            if n_components == 2:
                ax = fig.add_subplot(2, 3, i+1)
                scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, 
                                    cmap='viridis', alpha=0.7, s=50)
                ax.set_title(f'{method}')
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
                ax.grid(True, alpha=0.3)
            else:  # 3D plot for 3 components
                ax = fig.add_subplot(2, 3, i+1, projection='3d')
                scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], 
                                    c=y, cmap='viridis', alpha=0.7, s=50)
                ax.set_title(f'{method}')
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
                ax.set_zlabel('Component 3')
            
            # Add legend
            legend1 = ax.legend(*scatter.legend_elements(),
                                loc="upper right", title="Classes")
            ax.add_artist(legend1)
        
        except Exception as e:
            print(f"Error plotting {method}: {str(e)}")
    
    plt.tight_layout()
    plt.savefig('plots/dimensionality_reduction_comparison.png')
    plt.close()
    
    print(f"Dimensionality reduction comparison saved to plots/dimensionality_reduction_comparison.png")

def main(components=20):
    """
    Main function to run the advanced feature engineering pipeline.
    
    Parameters:
    -----------
    components : int, default=20
        Số components mặc định cho các phương pháp giảm chiều dữ liệu
    """
    print("=== Advanced Feature Engineering for ABIDE Dataset ===")
    print(f"Using {components} components for dimensionality reduction")
    
    try:
        # Load and prepare data
        print("Loading and preparing data...")
        X, y, feature_names = load_and_prepare_abide_data('ABIDE2(updated).csv', target_column='group')
        print(f"Data loaded successfully. X shape: {X.shape}, y shape: {y.shape}")
        
        # Handle outliers
        print("Handling outliers...")
        X_clean, outlier_mask = detect_and_handle_outliers(X, threshold=3.0, method='winsorize')
        print("Outliers handled successfully")
        
        # Normalize data
        print("Normalizing data...")
        X_norm, _ = normalize_connectivity_data(X_clean, method='robust')
        print("Data normalized successfully")
        
        # Create connectivity ratio features
        # Only use this if you need more features - can make computation intensive
        # X_ratios = create_connectivity_ratios(X_norm, feature_names, top_n=100)
        # X_enhanced = np.hstack((X_norm, X_ratios))
        # print(f"Enhanced feature matrix shape: {X_enhanced.shape}")
        
        # Compare dimensionality reduction methods
        print("Comparing dimensionality reduction methods...")
        plot_dimensionality_reduction_comparison(
            X_norm, y, 
            methods=['pca', 'kernel_pca', 'tsne', 'polynomial_pca', 'ica'],
            n_components=min(components, 3)  # Giới hạn số components cho visualization
        )
        
        # Find optimal preprocessing pipeline
        print("Finding optimal preprocessing pipeline...")
        best_method, best_components = optimize_clustering_preprocessing(
            X_norm, y, 
            methods=['pca', 'kernel_pca', 'polynomial_pca', 'ica', 'truncated_svd'],
            n_components=components
        )
        
        print(f"\nBest preprocessing method: {best_method}")
        print(f"Best number of components: {best_components}")
        
        # Apply the best method
        print("Applying best method...")
        X_reduced = apply_advanced_pca(
            X_norm, 
            n_components=best_components,
            transform_method=best_method
        )
        
        # Save the engineered features
        print("Saving engineered features...")
        np.save('abide_engineered_features.npy', X_reduced)
        np.save('abide_labels.npy', y)
        
        print(f"\nEngineered features saved to abide_engineered_features.npy")
        print(f"Labels saved to abide_labels.npy")
        
        return X_reduced, y
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    print("Script starting...", flush=True)
    # Parse command line arguments for components
    parser = argparse.ArgumentParser(description="Advanced feature engineering for ABIDE dataset")
    parser.add_argument('--components', type=int, default=20, 
                      help='Number of components for dimensionality reduction (default: 20)')
    args = parser.parse_args()
    
    try:
        main(components=args.components)
        print("Script completed successfully.", flush=True)
    except Exception as e:
        print(f"ERROR in main execution: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1) 