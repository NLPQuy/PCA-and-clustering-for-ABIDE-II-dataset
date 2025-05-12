#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Đề bài 1: Lập trình PCA và áp dụng cho tập dữ liệu Iris
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA as SklearnPCA

# Import custom modules
from src import MyPCA, load_iris_data, preprocess_data
from utils import plot_pca_components_variance, plot_pca_2d

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

def main():
    """
    Main function for PCA analysis on the Iris dataset.
    """
    print("=== Lab 2 - Part 1: PCA on Iris Dataset ===")
    
    # Load Iris dataset
    X, y, feature_names, target_names = load_iris_data()
    
    # Preprocess data
    X_preprocessed, _ = preprocess_data(X)
    
    # Apply custom PCA
    n_components = 4  # Use all components for Iris dataset
    my_pca = MyPCA(n_components=n_components)
    my_pca.fit(X_preprocessed)
    
    # Transform the data
    X_transformed = my_pca.transform(X_preprocessed)
    
    # Display PCA results
    print("\nPCA Analysis Results:")
    print("Explained Variance Ratio:", my_pca.explained_variance_ratio_)
    print("Cumulative Explained Variance Ratio:", my_pca.cumulative_explained_variance_ratio_)
    
    # Plot explained variance and cumulative explained variance
    plot_pca_components_variance(
        my_pca.explained_variance_ratio_,
        my_pca.cumulative_explained_variance_ratio_,
        save_path='plots/iris_variance_explained.png'
    )
    
    # Plot first two principal components with class labels
    plot_pca_2d(X_transformed, y, 
                title="Iris Dataset: First Two Principal Components",
                save_path='plots/iris_pca_2d.png')
    
    # Verify with sklearn PCA
    print("\nVerification with scikit-learn PCA:")
    sklearn_pca = SklearnPCA(n_components=n_components)
    sklearn_pca.fit(X_preprocessed)
    
    # Compare results
    print("Custom PCA Explained Variance Ratio:", my_pca.explained_variance_ratio_)
    print("Sklearn PCA Explained Variance Ratio:", sklearn_pca.explained_variance_ratio_)
    
    # Calculate MSE between the two implementations to verify correctness
    mse = np.mean((my_pca.explained_variance_ratio_ - sklearn_pca.explained_variance_ratio_) ** 2)
    print(f"Mean Squared Error between implementations: {mse:.8f}")
    
    print("\nPCA Analysis Completed!")

if __name__ == "__main__":
    main() 