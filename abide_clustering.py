#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Đề bài 3: Bài toán phân cụm trên tập dữ liệu ABIDE
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from sklearn.preprocessing import RobustScaler

# Import custom modules
from src import MyPCA, MyKMeans, load_abide_data, preprocess_data
from utils import plot_pca_components_variance, plot_pca_2d, plot_clusters_2d
from utils import calculate_cluster_metrics, print_metrics

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

def detect_outliers(X, threshold=3):
    """
    Detect outliers in the data using Z-score method.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data to check for outliers
    threshold : float, default=3
        Z-score threshold for outlier detection
        
    Returns:
    --------
    outlier_mask : array, shape (n_samples,)
        Boolean mask of outliers (True for outliers)
    """
    # Calculate Z-scores for each feature
    z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
    
    # Find samples with any feature having Z-score > threshold
    outlier_mask = np.any(z_scores > threshold, axis=1)
    
    return outlier_mask

def main():
    """
    Main function for clustering analysis on the ABIDE dataset.
    """
    print("=== Lab 2 - Part 3: Clustering on ABIDE Dataset ===")
    
    # Load ABIDE dataset
    X, y, feature_names = load_abide_data("ABIDE2(updated).csv")
    
    if X is None:
        print("Error loading the dataset. Exiting.")
        return
    
    # Check for and handle outliers
    print("\nChecking for outliers...")
    outlier_mask = detect_outliers(X, threshold=5)
    outlier_count = np.sum(outlier_mask)
    print(f"Detected {outlier_count} outliers out of {X.shape[0]} samples ({outlier_count/X.shape[0]*100:.2f}%)")
    
    if outlier_count > 0:
        # Option 1: Remove outliers
        if outlier_count < X.shape[0] * 0.3:  # Only remove if less than 30% are outliers
            X_clean = X[~outlier_mask]
            y_clean = y[~outlier_mask]
            print(f"Removed {outlier_count} outliers. New dataset shape: {X_clean.shape}")
            X = X_clean
            y = y_clean
        else:
            # Option 2: Use robust scaling instead of standard scaling
            print("Too many outliers to remove. Using robust scaling instead of standard scaling.")
            scaler = RobustScaler()
            X = scaler.fit_transform(X)
            print("Data robust-scaled.")
    else:
        # Preprocess data with standard scaling
        X_preprocessed, _ = preprocess_data(X)
        X = X_preprocessed
    
    # PCA for dimensionality reduction
    print("\nPerforming PCA for dimensionality reduction...")
    
    # Initialize with a large number of components first
    # We'll examine the cumulative explained variance to choose the optimal number
    initial_n_components = min(100, X.shape[1])
    
    try:
        my_pca = MyPCA(n_components=initial_n_components)
        my_pca.fit(X)
        
        # Plot explained variance and cumulative explained variance
        plot_pca_components_variance(
            my_pca.explained_variance_ratio_,
            my_pca.cumulative_explained_variance_ratio_,
            save_path='plots/abide_variance_explained.png'
        )
        
        # Select optimal number of components based on explained variance
        # Let's find where cumulative explained variance exceeds 80%
        cumulative_var = my_pca.cumulative_explained_variance_ratio_
        n_components_80 = np.argmax(cumulative_var >= 0.8) + 1
        n_components_90 = np.argmax(cumulative_var >= 0.9) + 1
        
        # If we don't reach 80% or 90% with our initial components, use a reasonable default
        if n_components_80 == 0 or n_components_80 > initial_n_components:
            n_components_80 = min(50, initial_n_components)
        if n_components_90 == 0 or n_components_90 > initial_n_components:
            n_components_90 = min(75, initial_n_components)
        
        print(f"\nComponents needed for 80% variance: {n_components_80}")
        print(f"Components needed for 90% variance: {n_components_90}")
        
        # Choose component counts to evaluate
        components_to_evaluate = [10, 20, 50, n_components_80, n_components_90]
        
        # If n_components_80 is large, let's also evaluate a smaller number
        if n_components_80 > 30:
            components_to_evaluate.append(30)
        
        # Sort the list of components to evaluate and remove duplicates
        components_to_evaluate = sorted(list(set(components_to_evaluate)))
        
        # Make sure all component numbers are valid
        components_to_evaluate = [n for n in components_to_evaluate if n <= X.shape[1]]
        
        # Create a table to store results
        results = []
        
        # Try different numbers of components
        for n_components in components_to_evaluate:
            print(f"\nEvaluating with {n_components} principal components...")
            
            # Fit PCA with the selected number of components
            my_pca = MyPCA(n_components=n_components)
            my_pca.fit(X)
            
            # Transform the data
            X_reduced = my_pca.transform(X)
            
            # Apply K-Means clustering (with k=2 because we know there are 2 classes)
            start_time = time.time()
            kmeans = MyKMeans(n_clusters=2, random_state=42, n_init=10)
            kmeans.fit(X_reduced)
            elapsed_time = time.time() - start_time
            
            # Evaluate clustering results
            metrics = calculate_cluster_metrics(y, kmeans.labels_)
            
            print(f"Clustering with {n_components} components completed in {elapsed_time:.2f} seconds")
            print_metrics(metrics)
            
            # Plot clustering results (only for first 2 components)
            if X_reduced.shape[1] >= 2:
                plot_clusters_2d(X_reduced[:, :2], kmeans.labels_, y, 
                                title=f"Clustering with {n_components} Components",
                                save_path=f'plots/abide_clustering_{n_components}_components.png')
            
            # Store results
            results.append({
                'n_components': n_components,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'ari': metrics['ari'],
                'nmi': metrics['nmi'],
                'time': elapsed_time
            })
        
        # Print summary table
        print("\n=== Summary of Results ===")
        print(f"{'Components':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} "
              f"{'F1 Score':<10} {'ARI':<10} {'NMI':<10} {'Time (s)':<10}")
        print("-" * 80)
        
        for result in results:
            print(f"{result['n_components']:<10} {result['accuracy']:<10.4f} {result['precision']:<10.4f} "
                  f"{result['recall']:<10.4f} {result['f1']:<10.4f} {result['ari']:<10.4f} "
                  f"{result['nmi']:<10.4f} {result['time']:<10.2f}")
        
        # Save results to CSV file
        import pandas as pd
        results_df = pd.DataFrame(results)
        results_df.to_csv('plots/abide_clustering_results.csv', index=False)
        print("Results saved to 'plots/abide_clustering_results.csv'")
        
        # Find best result based on F1 score
        best_result = max(results, key=lambda x: x['f1'])
        print(f"\nBest result by F1 score with {best_result['n_components']} components:")
        print(f"Accuracy: {best_result['accuracy']:.4f}, F1 Score: {best_result['f1']:.4f}")
        
        # Find best result based on ARI
        best_ari_result = max(results, key=lambda x: x['ari'])
        print(f"\nBest result by ARI with {best_ari_result['n_components']} components:")
        print(f"ARI: {best_ari_result['ari']:.4f}, NMI: {best_ari_result['nmi']:.4f}")
        
        print("\nClustering Analysis Completed!")
    
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 