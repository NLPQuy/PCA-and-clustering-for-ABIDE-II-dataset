#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced visualization script for the ABIDE dataset results.
This script provides multiple visualization techniques to better understand 
the clustering results and data distribution.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import matplotlib.cm as cm

# Import custom modules
from src import MyKMeans, load_abide_data, preprocess_data
from src.abide_feature_engineering import apply_combined_optimal_approach

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

def plot_zoomed_regions(X_opt, y, kmeans_labels, save_path):
    """
    Create multiple plots with different zoom levels to focus on dense regions.
    
    Parameters:
    -----------
    X_opt : array-like
        Transformed data with first two components for visualization
    y : array-like
        True labels
    kmeans_labels : array-like
        Clustering labels
    save_path : str
        Path to save the output plots
    """
    # Define regions to zoom in (based on data distribution)
    regions = [
        {'name': 'full', 'xlim': None, 'ylim': None},
        {'name': 'center', 'xlim': (-3, 3), 'ylim': (-3, 3)},
        {'name': 'dense_cluster', 'xlim': (-1, 1), 'ylim': (-1, 1)},
        {'name': 'outlier_region', 'xlim': (3, 7), 'ylim': (3, 7)}
    ]
    
    for region in regions:
        plt.figure(figsize=(20, 10))
        
        # Plot clusters
        plt.subplot(1, 2, 1)
        for i in range(2):  # Assuming 2 clusters
            cluster_points = X_opt[kmeans_labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       alpha=0.5, s=15, label=f'Cluster {i}')
        
        if region['xlim']:
            plt.xlim(region['xlim'])
        if region['ylim']:
            plt.ylim(region['ylim'])
            
        plt.title(f"Clusters - {region['name'].replace('_', ' ').title()} View")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot true labels
        plt.subplot(1, 2, 2)
        unique_labels = np.unique(y)
        for label in unique_labels:
            class_name = 'Cancer' if label == 1 else 'Normal'
            class_points = X_opt[y == label]
            plt.scatter(class_points[:, 0], class_points[:, 1], 
                       alpha=0.5, s=15, label=class_name)
        
        if region['xlim']:
            plt.xlim(region['xlim'])
        if region['ylim']:
            plt.ylim(region['ylim'])
            
        plt.title(f"True Labels - {region['name'].replace('_', ' ').title()} View")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_path}_{region['name']}.png")
        plt.close()
        
        print(f"Saved {region['name']} view visualization")

def plot_density_visualization(X_opt, y, kmeans_labels, save_path):
    """
    Create density-based visualizations using hexbin and KDE plots.
    
    Parameters:
    -----------
    X_opt : array-like
        Transformed data with first two components for visualization
    y : array-like
        True labels
    kmeans_labels : array-like
        Clustering labels
    save_path : str
        Path to save the output plots
    """
    # Hexbin plot - separate for clusters and true labels
    plt.figure(figsize=(20, 10))
    
    # Hexbin for clusters
    plt.subplot(1, 2, 1)
    hb = plt.hexbin(X_opt[:, 0], X_opt[:, 1], gridsize=75, cmap='viridis', bins='log')
    plt.colorbar(hb, label='log(Count)')
    plt.title("Point Density using Hexbin (Log Scale)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, alpha=0.3)
    
    # KDE plot by class
    plt.subplot(1, 2, 2)
    
    # Create a DataFrame for seaborn
    df = pd.DataFrame({
        'PC1': X_opt[:, 0],
        'PC2': X_opt[:, 1],
        'Class': ['Cancer' if l == 1 else 'Normal' for l in y]
    })
    
    # Plot KDE for each class
    sns.kdeplot(data=df, x='PC1', y='PC2', hue='Class', fill=True, alpha=0.5, levels=10)
    plt.title("Kernel Density Estimation by Class")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}_density.png")
    plt.close()
    
    print("Saved density visualization")
    
    # Create separate density plots for each class
    plt.figure(figsize=(20, 10))
    
    # Separate KDE for each class
    for i, class_name in enumerate(['Normal', 'Cancer']):
        plt.subplot(1, 2, i+1)
        class_df = df[df['Class'] == class_name]
        sns.kdeplot(data=class_df, x='PC1', y='PC2', fill=True, alpha=0.7, levels=10, 
                   cmap='viridis' if class_name == 'Normal' else 'magma')
        plt.title(f"Density Distribution - {class_name}")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}_density_by_class.png")
    plt.close()
    
    print("Saved class-specific density visualizations")
    
    # Now create density plots for specific regions
    plt.figure(figsize=(15, 15))
    # Focus on center region
    df_center = df[(df['PC1'] > -3) & (df['PC1'] < 3) & (df['PC2'] > -3) & (df['PC2'] < 3)]
    
    # Plot KDE for center region
    sns.kdeplot(data=df_center, x='PC1', y='PC2', hue='Class', fill=True, alpha=0.5, levels=15)
    plt.title("Kernel Density Estimation - Center Region")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}_density_center.png")
    plt.close()
    
    print("Saved center region density visualization")

def plot_3d_visualizations(X_opt, y, kmeans_labels, save_path):
    """
    Create 3D visualizations with different angles and perspectives.
    
    Parameters:
    -----------
    X_opt : array-like
        Transformed data with at least 3 components for 3D visualization
    y : array-like
        True labels
    kmeans_labels : array-like
        Clustering labels
    save_path : str
        Path to save the output plots
    """
    if X_opt.shape[1] < 3:
        print("Not enough components for 3D visualization.")
        return
    
    # Create 3D plots with different angles
    angles = [
        (30, 45),  # Front-angled view
        (0, 0),    # Front view
        (0, 90),   # Top view
        (90, 0)    # Side view
    ]
    
    # Create colormap with better visibility
    cluster_colors = ['#1f77b4', '#ff7f0e']  # Blue and orange
    class_colors = ['#2ca02c', '#d62728']    # Green and red
    
    for i, (elev, azim) in enumerate(angles):
        fig = plt.figure(figsize=(20, 10))
        
        # Plot by clusters
        ax1 = fig.add_subplot(121, projection='3d')
        for cluster in range(2):  # Assuming 2 clusters
            ax1.scatter(X_opt[kmeans_labels == cluster, 0], 
                        X_opt[kmeans_labels == cluster, 1], 
                        X_opt[kmeans_labels == cluster, 2],
                        alpha=0.7, s=25, c=[cluster_colors[cluster]], 
                        edgecolors='w', linewidth=0.4,
                        label=f'Cluster {cluster}')
        
        ax1.set_title("Clusters in 3D Space")
        ax1.set_xlabel("Component 1")
        ax1.set_ylabel("Component 2")
        ax1.set_zlabel("Component 3")
        ax1.view_init(elev=elev, azim=azim)
        ax1.legend()
        
        # Plot by true labels
        ax2 = fig.add_subplot(122, projection='3d')
        for j, label in enumerate(np.unique(y)):
            class_name = 'Cancer' if label == 1 else 'Normal'
            ax2.scatter(X_opt[y == label, 0], 
                        X_opt[y == label, 1], 
                        X_opt[y == label, 2],
                        alpha=0.7, s=25, c=[class_colors[j]], 
                        edgecolors='w', linewidth=0.4,
                        label=class_name)
        
        ax2.set_title("True Classes in 3D Space")
        ax2.set_xlabel("Component 1")
        ax2.set_ylabel("Component 2")
        ax2.set_zlabel("Component 3")
        ax2.view_init(elev=elev, azim=azim)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_path}_3d_angle_{i}.png")
        plt.close()
        
        print(f"Saved 3D visualization with angle {elev}, {azim}")
    
    # Create animated 3D visualization (if needed)
    # Uncomment and run if you want to create an animated GIF
    """
    from matplotlib.animation import FuncAnimation
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot data points
    scatter = ax.scatter(X_opt[:, 0], X_opt[:, 1], X_opt[:, 2],
                         c=y, cmap='coolwarm', alpha=0.7, s=30)
    
    ax.set_title("3D Visualization - Rotating View")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    
    # Add a colorbar
    plt.colorbar(scatter, label='Class')
    
    def update(frame):
        ax.view_init(elev=30, azim=frame)
        return scatter,
    
    ani = FuncAnimation(fig, update, frames=range(0, 360, 5), blit=True)
    ani.save(f"{save_path}_3d_animated.gif", writer='pillow', fps=10)
    plt.close()
    
    print("Saved animated 3D visualization")
    """

def plot_parallel_coordinates(X_opt, y, kmeans_labels, save_path):
    """
    Create parallel coordinates plots to visualize multiple dimensions simultaneously.
    
    Parameters:
    -----------
    X_opt : array-like
        Transformed data
    y : array-like
        True labels
    kmeans_labels : array-like
        Clustering labels
    save_path : str
        Path to save the output plots
    """
    # Use first 10 components or all if less than 10
    n_components = min(10, X_opt.shape[1])
    
    # Create DataFrames for visualization
    df_true = pd.DataFrame(X_opt[:, :n_components], 
                          columns=[f"PC{i+1}" for i in range(n_components)])
    df_true['Class'] = ['Cancer' if l == 1 else 'Normal' for l in y]
    
    df_cluster = pd.DataFrame(X_opt[:, :n_components], 
                             columns=[f"PC{i+1}" for i in range(n_components)])
    df_cluster['Cluster'] = [f"Cluster {l}" for l in kmeans_labels]
    
    # Plot parallel coordinates by true class
    plt.figure(figsize=(20, 10))
    
    # Plot by true labels
    plt.subplot(2, 1, 1)
    pd.plotting.parallel_coordinates(df_true, 'Class', colormap='coolwarm', alpha=0.3)
    plt.title("Parallel Coordinates by True Class")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Plot by cluster
    plt.subplot(2, 1, 2)
    pd.plotting.parallel_coordinates(df_cluster, 'Cluster', colormap='viridis', alpha=0.3)
    plt.title("Parallel Coordinates by Cluster")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}_parallel_coordinates.png")
    plt.close()
    
    print("Saved parallel coordinates visualizations")
    
    # Create separate parallel coordinates plots for each class with sample reduction for clarity
    for class_name, class_value in zip(['Normal', 'Cancer'], [0, 1]):
        # Get samples for this class
        class_indices = np.where(y == class_value)[0]
        # Take a random sample of 100 points (or less if fewer points available)
        sample_size = min(100, len(class_indices))
        sample_indices = np.random.choice(class_indices, size=sample_size, replace=False)
        
        # Create sampled dataframe
        sampled_df = df_true.iloc[sample_indices].copy()
        
        plt.figure(figsize=(15, 8))
        pd.plotting.parallel_coordinates(sampled_df, 'Class', colormap='coolwarm', alpha=0.5)
        plt.title(f"Parallel Coordinates - {class_name} Class (Sample of {sample_size} points)")
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}_parallel_coordinates_{class_name.lower()}.png")
        plt.close()
        
        print(f"Saved parallel coordinates for {class_name} class")

def plot_component_distributions(X_opt, y, kmeans_labels, save_path):
    """
    Plot the distribution of each principal component separated by class.
    
    Parameters:
    -----------
    X_opt : array-like
        Transformed data
    y : array-like
        True labels
    kmeans_labels : array-like
        Clustering labels
    save_path : str
        Path to save the output plots
    """
    n_components = min(6, X_opt.shape[1])  # Show first 6 components
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame(X_opt[:, :n_components], 
                     columns=[f"PC{i+1}" for i in range(n_components)])
    df['Class'] = ['Cancer' if l == 1 else 'Normal' for l in y]
    df['Cluster'] = [f"Cluster {l}" for l in kmeans_labels]
    
    # Plot distributions by class
    plt.figure(figsize=(20, 15))
    
    for i in range(n_components):
        plt.subplot(2, 3, i+1)
        sns.kdeplot(data=df, x=f"PC{i+1}", hue="Class", fill=True, common_norm=False, alpha=0.5)
        plt.title(f"Distribution of PC{i+1} by Class")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}_component_distributions_by_class.png")
    plt.close()
    
    # Plot distributions by cluster
    plt.figure(figsize=(20, 15))
    
    for i in range(n_components):
        plt.subplot(2, 3, i+1)
        sns.kdeplot(data=df, x=f"PC{i+1}", hue="Cluster", fill=True, common_norm=False, alpha=0.5)
        plt.title(f"Distribution of PC{i+1} by Cluster")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}_component_distributions_by_cluster.png")
    plt.close()
    
    print("Saved component distribution visualizations")

def plot_confusion_matrix_visualization(y, kmeans_labels, save_path):
    """
    Create enhanced confusion matrix visualization.
    
    Parameters:
    -----------
    y : array-like
        True labels
    kmeans_labels : array-like
        Clustering labels
    save_path : str
        Path to save the output plots
    """
    from sklearn.metrics import confusion_matrix
    
    # Map cluster labels to classes for best alignment
    # Determine which cluster corresponds best to which class
    contingency_table = confusion_matrix(y, kmeans_labels)
    
    # Find the mapping that minimizes classification error
    class0_count = np.sum(y == 0)
    class1_count = np.sum(y == 1)
    
    if contingency_table[0, 0] + contingency_table[1, 1] >= contingency_table[0, 1] + contingency_table[1, 0]:
        # No remapping needed
        cluster_to_class_map = {0: 0, 1: 1}
        aligned_matrix = contingency_table
    else:
        # Swap clusters
        cluster_to_class_map = {0: 1, 1: 0}
        aligned_matrix = contingency_table[:, ::-1]
    
    # Normalize by class size
    cm_norm_by_class = aligned_matrix.astype('float') / aligned_matrix.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    plt.figure(figsize=(15, 12))
    
    plt.subplot(2, 2, 1)
    sns.heatmap(aligned_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Counts)")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Cluster")
    plt.xticks([0.5, 1.5], ['Normal', 'Cancer'])
    plt.yticks([0.5, 1.5], ['Normal', 'Cancer'])
    
    plt.subplot(2, 2, 2)
    sns.heatmap(cm_norm_by_class, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
    plt.title("Confusion Matrix (Normalized by Row)")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Cluster")
    plt.xticks([0.5, 1.5], ['Normal', 'Cancer'])
    plt.yticks([0.5, 1.5], ['Normal', 'Cancer'])
    
    # Calculate and display additional metrics
    accuracy = (aligned_matrix[0, 0] + aligned_matrix[1, 1]) / np.sum(aligned_matrix)
    sensitivity = aligned_matrix[1, 1] / (aligned_matrix[1, 0] + aligned_matrix[1, 1]) if aligned_matrix[1, 0] + aligned_matrix[1, 1] > 0 else 0
    specificity = aligned_matrix[0, 0] / (aligned_matrix[0, 0] + aligned_matrix[0, 1]) if aligned_matrix[0, 0] + aligned_matrix[0, 1] > 0 else 0
    precision = aligned_matrix[1, 1] / (aligned_matrix[0, 1] + aligned_matrix[1, 1]) if aligned_matrix[0, 1] + aligned_matrix[1, 1] > 0 else 0
    
    plt.subplot(2, 2, 3)
    metrics_display = {
        'Metric': ['Accuracy', 'Sensitivity', 'Specificity', 'Precision'],
        'Value': [accuracy, sensitivity, specificity, precision]
    }
    metrics_df = pd.DataFrame(metrics_display)
    metrics_df.set_index('Metric', inplace=True)
    
    # Plot metrics as a table
    for i, metric in enumerate(metrics_df.index):
        plt.text(0.1, 0.8 - 0.1*i, f"{metric}: {metrics_df.loc[metric, 'Value']:.4f}", 
                 fontsize=14)
    
    plt.axis('off')
    plt.title("Classification Metrics")
    
    # Add a note on cluster mapping
    map_description = f"Cluster mapping: Cluster 0 → {'Normal' if cluster_to_class_map[0] == 0 else 'Cancer'}, Cluster 1 → {'Cancer' if cluster_to_class_map[1] == 1 else 'Normal'}"
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.5, map_description, fontsize=14)
    plt.text(0.1, 0.3, f"Total samples: {np.sum(aligned_matrix)}", fontsize=14)
    plt.text(0.1, 0.2, f"Normal class: {class0_count} samples", fontsize=14)
    plt.text(0.1, 0.1, f"Cancer class: {class1_count} samples", fontsize=14)
    plt.axis('off')
    plt.title("Data Summary")
    
    plt.tight_layout()
    plt.savefig(f"{save_path}_confusion_matrix.png")
    plt.close()
    
    print("Saved detailed confusion matrix visualization")

def plot_tsne_visualization(X_opt, y, kmeans_labels, save_path):
    """
    Create t-SNE visualization for better cluster separation.
    
    Parameters:
    -----------
    X_opt : array-like
        Transformed data
    y : array-like
        True labels
    kmeans_labels : array-like
        Clustering labels
    save_path : str
        Path to save the output plots
    """
    # Apply t-SNE
    print("\nApplying t-SNE for visualization...")
    # Use a lower perplexity if sample size is small
    perplexity = min(30, X_opt.shape[0] // 5)
    X_tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(X_opt)
    
    # Plot t-SNE results
    plt.figure(figsize=(20, 10))
    
    # By cluster
    plt.subplot(1, 2, 1)
    for i in range(2):  # Assuming 2 clusters
        plt.scatter(X_tsne[kmeans_labels == i, 0], X_tsne[kmeans_labels == i, 1],
                   alpha=0.6, s=20, label=f'Cluster {i}')
    
    plt.title("t-SNE Visualization by Cluster")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # By true class
    plt.subplot(1, 2, 2)
    for i, label in enumerate(np.unique(y)):
        class_name = 'Cancer' if label == 1 else 'Normal'
        plt.scatter(X_tsne[y == label, 0], X_tsne[y == label, 1],
                   alpha=0.6, s=20, label=class_name)
    
    plt.title("t-SNE Visualization by True Class")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_path}_tsne.png")
    plt.close()
    
    print("Saved t-SNE visualization")

def main():
    """
    Main function for enhanced visualizations of ABIDE dataset clustering.
    """
    print("=== Enhanced Visualization of ABIDE Clustering Results ===")
    
    # Load ABIDE dataset
    X, y, feature_names = load_abide_data("ABIDE2(updated).csv")
    
    if X is None:
        print("Error loading the dataset. Exiting.")
        return
    
    print(f"Loaded dataset with {X.shape[0]} samples and {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Apply the combined optimal approach
    print("\nApplying the combined optimal approach for feature engineering...")
    X_opt = apply_combined_optimal_approach(X, feature_names, n_components=20)
    
    # Apply K-Means clustering
    print("\nPerforming K-Means clustering on transformed data...")
    kmeans = MyKMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_opt)
    
    # Base path for saving visualizations
    base_save_path = 'plots/abide_visualization'
    
    # Print total number of points being visualized
    print(f"\nCreating visualizations for {X_opt.shape[0]} data points...")
    
    # Apply all visualization methods
    plot_zoomed_regions(X_opt, y, kmeans.labels_, base_save_path)
    plot_density_visualization(X_opt, y, kmeans.labels_, base_save_path)
    plot_3d_visualizations(X_opt, y, kmeans.labels_, base_save_path)
    plot_parallel_coordinates(X_opt, y, kmeans.labels_, base_save_path)
    plot_component_distributions(X_opt, y, kmeans.labels_, base_save_path)
    plot_confusion_matrix_visualization(y, kmeans.labels_, base_save_path)
    plot_tsne_visualization(X_opt, y, kmeans.labels_, base_save_path)
    
    # Summary of visualizations created
    print("\n=== Visualization Summary ===")
    print("1. Multiple zoomed region views")
    print("2. Density visualizations with hexbin and KDE plots")
    print("3. 3D visualizations with multiple angles")
    print("4. Parallel coordinates for multi-dimensional analysis")
    print("5. Component distribution analysis")
    print("6. Enhanced confusion matrix and metrics")
    print("7. t-SNE visualization for non-linear embedding")
    
    print("\nAll visualizations saved in the 'plots' directory")
    print("=== Visualization Complete ===")

if __name__ == "__main__":
    main() 