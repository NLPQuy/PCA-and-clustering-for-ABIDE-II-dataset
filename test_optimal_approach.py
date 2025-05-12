#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the Combined Optimal Approach on ABIDE dataset.
This script tests the performance of the recommended combined approach for feature engineering.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# Import custom modules
from src import MyKMeans, load_abide_data, preprocess_data
from src.abide_feature_engineering import (
    apply_combined_optimal_approach, 
    apply_advanced_pca,
    apply_ensemble_dimensionality_reduction
)
from utils import plot_clusters_2d, calculate_cluster_metrics, print_metrics

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

def main():
    """
    Main function to test the Combined Optimal Approach.
    """
    print("=== Testing Combined Optimal Approach on ABIDE Dataset ===")
    
    # Load ABIDE dataset
    X, y, feature_names = load_abide_data("ABIDE2(updated).csv")
    
    if X is None:
        print("Error loading the dataset. Exiting.")
        return
    
    print(f"Loaded dataset with {X.shape[0]} samples and {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Number of points to be visualized: {X.shape[0]}")
    
    # Apply the combined optimal approach on the entire dataset
    X_opt = apply_combined_optimal_approach(X, feature_names, n_components=20)
    
    # Apply K-Means clustering
    kmeans = MyKMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_opt)
    
    # Evaluate clustering results
    metrics = calculate_cluster_metrics(y, kmeans.labels_)
    print("\n=== Clustering Performance ===")
    print_metrics(metrics)
    
    # Plot clustering results with lower alpha to see overlapping points
    plt.figure(figsize=(20, 10))
    
    # Plot based on clusters
    plt.subplot(1, 2, 1)
    for i in range(2):
        plt.scatter(X_opt[kmeans.labels_ == i, 0], X_opt[kmeans.labels_ == i, 1], 
                   alpha=0.4, s=10, label=f'Cluster {i}')
    plt.title("Clustering with Combined Optimal Approach (Clusters)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot based on true labels
    plt.subplot(1, 2, 2)
    unique_labels = np.unique(y)
    for label in unique_labels:
        class_name = 'Cancer' if label == 1 else 'Normal'
        plt.scatter(X_opt[y == label, 0], X_opt[y == label, 1], 
                   alpha=0.4, s=10, label=class_name)
    plt.title("Clustering with Combined Optimal Approach (True Labels)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('plots/abide_clustering_combined_optimal_approach_full_data.png')
    plt.close()
    
    print(f"Visualization saved to 'plots/abide_clustering_combined_optimal_approach_full_data.png'")
    
    # Compare with other results from previous runs
    print("\n=== Comparative Analysis ===")
    print("Previous best ARI was with 'Region Ratio Features + PCA': 0.0363")
    print(f"Combined Optimal Approach ARI on full dataset: {metrics['ari']:.4f}")
    
    print("\nPrevious best F1 Score was with 'Ensemble Dimensionality Reduction': 0.4828")
    print(f"Combined Optimal Approach F1 Score on full dataset: {metrics['f1']:.4f}")
    
    # Save the detailed results to CSV
    results = [{
        'method': 'Combined Optimal Approach (Full Dataset)',
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'ari': metrics['ari'],
        'nmi': metrics['nmi']
    }]
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('plots/combined_optimal_approach_results_full_data.csv', index=False)
    print("Detailed results saved to 'plots/combined_optimal_approach_results_full_data.csv'")
    
    # Additional visualization with hexbin to show density
    plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 2, 1)
    # Use hexbin for better visualization of point density
    hb = plt.hexbin(X_opt[:, 0], X_opt[:, 1], gridsize=50, cmap='viridis')
    plt.colorbar(hb, label='Count')
    plt.title("Point Density Visualization")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, alpha=0.3)
    
    # 3D scatter plot for better visualization
    from mpl_toolkits.mplot3d import Axes3D
    
    if X_opt.shape[1] >= 3:  # Make sure we have at least 3 components
        ax = plt.subplot(1, 2, 2, projection='3d')
        scatter = ax.scatter(X_opt[:, 0], X_opt[:, 1], X_opt[:, 2], 
                            c=y, cmap='viridis', alpha=0.6, s=10)
        plt.colorbar(scatter, label='Class')
        ax.set_title("3D Visualization (First 3 Components)")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
    
    plt.tight_layout()
    plt.savefig('plots/abide_clustering_density_3d_visualization.png')
    plt.close()
    
    print(f"Additional visualizations saved to 'plots/abide_clustering_density_3d_visualization.png'")
    
    print("\n=== Combined Optimal Approach Testing Complete ===")

if __name__ == "__main__":
    main() 