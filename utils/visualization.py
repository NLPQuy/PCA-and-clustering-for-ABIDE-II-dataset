import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

# Create output directory for plots
os.makedirs('plots', exist_ok=True)

def plot_pca_components_variance(explained_variance_ratio, cumulative_explained_variance_ratio, save_path=None):
    """
    Plot the explained variance ratio and cumulative explained variance ratio.
    
    Parameters:
    -----------
    explained_variance_ratio : array-like
        Explained variance ratio of each principal component
    cumulative_explained_variance_ratio : array-like
        Cumulative explained variance ratio
    save_path : str, optional
        Path to save the figure. If None, the figure is shown but not saved.
    """
    plt.figure(figsize=(12, 5))
    
    # Plot individual explained variance
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Component')
    plt.xticks(range(1, min(len(explained_variance_ratio) + 1, 11)))
    plt.grid(True, alpha=0.3)
    
    # Plot cumulative explained variance
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_explained_variance_ratio) + 1), 
             cumulative_explained_variance_ratio, marker='o')
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% Threshold')
    plt.axhline(y=0.9, color='g', linestyle='--', label='90% Threshold')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance')
    plt.xticks(range(1, min(len(cumulative_explained_variance_ratio) + 1, 11)))
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    plt.show()

def plot_pca_2d(X_transformed, labels=None, title="PCA: First Two Principal Components", save_path=None):
    """
    Plot the first two principal components with class labels if provided.
    
    Parameters:
    -----------
    X_transformed : array-like
        Data transformed by PCA
    labels : array-like, optional
        Class labels for coloring the points
    title : str, optional
        Title for the plot
    save_path : str, optional
        Path to save the figure. If None, the figure is shown but not saved.
    """
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for i, color in zip(unique_labels, colors):
            mask = (labels == i)
            plt.scatter(X_transformed[mask, 0], X_transformed[mask, 1], 
                       color=color, alpha=0.7, label=f'Class {i}')
        
        plt.legend()
    else:
        plt.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.7)
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    plt.show()

def plot_clusters_2d(X_transformed, cluster_labels, true_labels=None, title="Clustering Results", save_path=None):
    """
    Plot clustering results in 2D space with the first two principal components.
    
    Parameters:
    -----------
    X_transformed : array-like
        Data transformed by PCA (first two components)
    cluster_labels : array-like
        Cluster assignments
    true_labels : array-like, optional
        True class labels for comparison
    title : str, optional
        Title for the plot
    save_path : str, optional
        Path to save the figure. If None, the figure is shown but not saved.
    """
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    
    # Create colormap
    cmap = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
    cluster_colors = ListedColormap(cmap)
    
    plt.figure(figsize=(16, 6))
    
    # Plot clustering results
    plt.subplot(1, 2, 1)
    sc = plt.scatter(X_transformed[:, 0], X_transformed[:, 1], 
                   c=cluster_labels, cmap=cluster_colors, alpha=0.7)
    plt.colorbar(sc, label='Cluster')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'{title} (Clusters)')
    plt.grid(True, alpha=0.3)
    
    # Plot true labels if provided
    if true_labels is not None:
        unique_labels = np.unique(true_labels)
        n_labels = len(unique_labels)
        
        # Create another colormap
        true_colors = plt.cm.viridis(np.linspace(0, 1, n_labels))
        true_colormap = ListedColormap(true_colors)
        
        plt.subplot(1, 2, 2)
        sc2 = plt.scatter(X_transformed[:, 0], X_transformed[:, 1], 
                        c=true_labels, cmap=true_colormap, alpha=0.7)
        plt.colorbar(sc2, label='True Class')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'{title} (True Labels)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    plt.show() 