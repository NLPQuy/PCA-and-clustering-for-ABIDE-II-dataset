#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compare whitening methods on ABIDE dataset.
This script visualizes the effect of different whitening methods on brain imaging data.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Import necessary functions from our code
from src.abide_feature_engineering import (
    apply_zca_whitening_pca,
    apply_cholesky_whitening,
    apply_mahalanobis_whitening,
    apply_pca_adaptive_whitening,
    apply_whitening_graph_feature_selection_pca,
    apply_hybrid_whitening
)

def load_abide_sample(file_path='ABIDE2_sample.csv', n_samples=100):
    """
    Load a sample of the ABIDE dataset.
    
    Parameters:
    -----------
    file_path : str
        Path to the ABIDE dataset file
    n_samples : int
        Number of samples to load
        
    Returns:
    --------
    X : array-like
        Feature data
    y : array-like
        Labels
    feature_names : list
        Names of features
    """
    try:
        df = pd.read_csv(file_path)
        
        # Check if the dataframe is empty
        if df.empty:
            print(f"Error: Empty dataframe loaded from {file_path}")
            return None, None, None
            
        # Identify label column (expecting 'DX_GROUP' or similar)
        label_cols = [col for col in df.columns if 'DX' in col.upper()]
        if not label_cols:
            print(f"Error: No label column found in {file_path}")
            return None, None, None
            
        label_col = label_cols[0]
        
        # Extract features and labels
        X = df.drop(columns=[label_col]).values
        y = df[label_col].values
        feature_names = df.drop(columns=[label_col]).columns.tolist()
        
        # Take a sample if requested
        if n_samples and n_samples < X.shape[0]:
            indices = np.random.choice(X.shape[0], n_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
        return X, y, feature_names
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def apply_standard_pca(X, n_components=2):
    """
    Apply standard PCA without whitening.
    
    Parameters:
    -----------
    X : array-like
        Data to transform
    n_components : int
        Number of components
        
    Returns:
    --------
    X_pca : array-like
        Transformed data
    """
    pca = PCA(n_components=n_components, whiten=False)
    return pca.fit_transform(X)

def apply_pca_whitening(X, n_components=2):
    """
    Apply PCA with whitening.
    
    Parameters:
    -----------
    X : array-like
        Data to transform
    n_components : int
        Number of components
        
    Returns:
    --------
    X_pca : array-like
        Transformed data
    """
    pca = PCA(n_components=n_components, whiten=True)
    return pca.fit_transform(X)

def visualize_covariance_matrices(data_dict):
    """
    Visualize the covariance matrices of different transformed datasets.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary mapping method names to transformed data
    """
    n_methods = len(data_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 4))
    
    if n_methods == 1:
        axes = [axes]
    
    for i, (method, data) in enumerate(data_dict.items()):
        # Calculate covariance matrix
        cov = np.cov(data, rowvar=False)
        
        # Plot
        im = axes[i].imshow(cov, cmap='coolwarm')
        axes[i].set_title(f'{method} Covariance')
        fig.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig('plots/whitening_covariance_comparison.png')
    plt.close()
    print("Saved covariance visualization to 'plots/whitening_covariance_comparison.png'")

def visualize_transformations(data_dict, labels=None):
    """
    Visualize 2D projections of the transformed data.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary mapping method names to transformed data
    labels : array-like, optional
        Class labels for coloring points
    """
    n_methods = len(data_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 4))
    
    if n_methods == 1:
        axes = [axes]
    
    for i, (method, data) in enumerate(data_dict.items()):
        # Only use first 2 dimensions for plotting
        X_plot = data[:, :2]
        
        # Plot with or without labels
        if labels is not None:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                axes[i].scatter(X_plot[mask, 0], X_plot[mask, 1], alpha=0.7, label=f'Class {label}')
            axes[i].legend()
        else:
            axes[i].scatter(X_plot[:, 0], X_plot[:, 1], alpha=0.7)
        
        axes[i].set_title(method)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/whitening_transformation_comparison.png')
    plt.close()
    print("Saved transformation visualization to 'plots/whitening_transformation_comparison.png'")

def compare_whitening_methods(X, y=None, n_components=2):
    """
    Compare different whitening methods.
    
    Parameters:
    -----------
    X : array-like
        Data for transformation
    y : array-like, optional
        Class labels
    n_components : int
        Number of components for dimensionality reduction
    """
    # Standard scale the data first
    X_std = StandardScaler().fit_transform(X)
    
    # Apply different whitening methods
    transformed_data = {}
    
    # 1. No whitening (standard PCA)
    transformed_data['Standard PCA'] = apply_standard_pca(X_std, n_components)
    
    # 2. PCA whitening
    transformed_data['PCA Whitening'] = apply_pca_whitening(X_std, n_components)
    
    # 3. ZCA whitening
    try:
        transformed_data['ZCA Whitening'] = apply_zca_whitening_pca(X_std, n_components)
    except Exception as e:
        print(f"Error applying ZCA whitening: {e}")
    
    # 4. Cholesky whitening
    try:
        transformed_data['Cholesky Whitening'] = apply_cholesky_whitening(X_std, n_components)
    except Exception as e:
        print(f"Error applying Cholesky whitening: {e}")
    
    # 5. Mahalanobis whitening
    try:
        transformed_data['Mahalanobis Whitening'] = apply_mahalanobis_whitening(X_std, n_components)
    except Exception as e:
        print(f"Error applying Mahalanobis whitening: {e}")
    
    # 6. Adaptive whitening with different alpha values
    alphas = [0.3, 0.7, 1.0]
    for alpha in alphas:
        try:
            transformed_data[f'Adaptive (α={alpha})'] = apply_pca_adaptive_whitening(
                X_std, n_components, alpha=alpha)
        except Exception as e:
            print(f"Error applying adaptive whitening with α={alpha}: {e}")
    
    # 7. Graph-based selection with whitening
    try:
        transformed_data['Graph + Whitening'] = apply_whitening_graph_feature_selection_pca(
            X_std, n_components)
    except Exception as e:
        print(f"Error applying graph-based whitening: {e}")
    
    # 8. Hybrid whitening with different zca_weight values
    zca_weights = [0.3, 0.5, 0.7]
    for zca_weight in zca_weights:
        try:
            transformed_data[f'Hybrid (ZCA={zca_weight}, α=0.7)'] = apply_hybrid_whitening(
                X_std, n_components, zca_weight=zca_weight, adaptive_alpha=0.7)
        except Exception as e:
            print(f"Error applying hybrid whitening with ZCA weight={zca_weight}: {e}")
    
    # Visualize covariance matrices
    visualize_covariance_matrices(transformed_data)
    
    # Visualize transformations
    visualize_transformations(transformed_data, y)
    
    return transformed_data

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Compare whitening methods on ABIDE data")
    parser.add_argument("--input-file", type=str, default="ABIDE2_sample.csv",
                        help="Path to input CSV file")
    parser.add_argument("--components", type=int, default=10,
                        help="Number of components for whitening (default: 10)")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples to use (default: 100)")
    parser.add_argument("--zca-weight", type=float, default=0.5,
                        help="Weight for ZCA in hybrid whitening (0-1, default: 0.5)")
    parser.add_argument("--adaptive-alpha", type=float, default=0.7,
                        help="Alpha parameter for adaptive whitening (0-1, default: 0.7)")
    
    args = parser.parse_args()
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Load data
    X, y, feature_names = load_abide_sample(args.input_file, args.samples)
    
    if X is None:
        print("Error loading data. Exiting.")
        return
    
    # Compare whitening methods
    print(f"\nComparing whitening methods with {args.components} components...")
    transformed_data = compare_whitening_methods(X, y, args.components)
    
    print(f"\nWhitening comparison completed! Results saved in the 'plots' directory.")

if __name__ == "__main__":
    main() 