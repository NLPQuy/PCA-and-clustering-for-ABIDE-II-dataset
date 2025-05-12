#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced clustering pipeline for ABIDE dataset using:
PCA -> t-SNE/UMAP -> K-means clustering

Pipeline steps:
1. Load and preprocess ABIDE data
2. Apply PCA to reduce to 15 components
3. Further reduce dimensionality with t-SNE or UMAP
4. Apply K-means clustering
5. Evaluate and visualize results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Try to import UMAP, with a fallback if not installed
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with 'pip install umap-learn'")

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Import functions from data_processing.py
from data_processing import load_data, preprocess_data, get_optimal_preprocessing

def load_processed_data(file_path="ABIDE2_processed.csv", group_column='group'):
    """
    Load already processed ABIDE data
    
    Parameters:
    -----------
    file_path : str
        Path to the processed ABIDE CSV file (output from data_processing.py)
    group_column : str
        Name of the column containing group labels
        
    Returns:
    --------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target labels (for evaluation)
    feature_names : list
        Names of the features
    metadata : pandas.DataFrame
        Metadata columns from the dataset
    """
    print(f"Loading processed data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Check if group column exists
    if group_column not in df.columns:
        raise ValueError(f"Group column '{group_column}' not found in the dataset")
    
    # Extract labels and convert to numeric (0 for Normal, 1 for Cancer)
    group_values = df[group_column].values
    if group_values.dtype == np.dtype('O'):  # If object dtype (strings)
        print("Converting group labels to numeric values...")
        y = np.array([1 if val == 'Cancer' else 0 for val in group_values])
    else:
        y = group_values
    
    # Separate metadata (non-feature columns) and features (columns with 'fs' prefix)
    metadata_cols = [col for col in df.columns if not col.startswith('fs')]
    feature_cols = [col for col in df.columns if col.startswith('fs')]
    
    print(f"Found {len(feature_cols)} feature columns and {len(metadata_cols)} metadata columns")
    
    # Extract features
    X = df[feature_cols].values
    feature_names = feature_cols
    metadata = df[metadata_cols]
    
    print(f"Processed data loaded: {X.shape}")
    return X, y, feature_names, metadata

def detect_and_handle_outliers(X, threshold=3.0, method='winsorize'):
    """
    Detect and handle outliers in the data
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    threshold : float
        Z-score threshold for outlier detection
    method : str
        Method for handling outliers: 'winsorize', 'remove', or None
        
    Returns:
    --------
    X_clean : numpy.ndarray
        Data with outliers handled
    outlier_mask : numpy.ndarray
        Boolean mask of outliers (True if outlier)
    """
    # Calculate Z-scores
    z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
    
    # Find outliers (samples with any feature having Z-score > threshold)
    outlier_mask = np.any(z_scores > threshold, axis=1)
    outlier_count = np.sum(outlier_mask)
    
    print(f"Detected {outlier_count} samples ({outlier_count/len(X)*100:.1f}%) with outliers")
    
    if method is None or method.lower() == 'none':
        return X, outlier_mask
    
    elif method.lower() == 'remove':
        # Only remove if less than 30% are outliers
        if outlier_count < len(X) * 0.3:
            X_clean = X[~outlier_mask]
            print(f"Removed {outlier_count} outlier samples")
            return X_clean, outlier_mask
        else:
            print("Too many outliers to remove. Try 'winsorize' instead.")
            return X, outlier_mask
    
    elif method.lower() == 'winsorize':
        # Cap outlier values at the threshold
        X_clean = X.copy()
        for i in range(X.shape[1]):
            col_mean = np.mean(X[:, i])
            col_std = np.std(X[:, i])
            lower_bound = col_mean - threshold * col_std
            upper_bound = col_mean + threshold * col_std
            
            # Cap values
            X_clean[:, i] = np.clip(X_clean[:, i], lower_bound, upper_bound)
        
        print(f"Winsorized {outlier_count} outlier samples")
        return X_clean, outlier_mask
    
    else:
        raise ValueError(f"Unknown outlier handling method: {method}")

def apply_pca(X, n_components=15, random_state=42, whiten=True):
    """
    Apply PCA dimensionality reduction
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    n_components : int
        Number of PCA components to keep
    random_state : int
        Random seed for reproducibility
    whiten : bool
        Whether to apply whitening to the components
        
    Returns:
    --------
    X_pca : numpy.ndarray
        Reduced feature matrix
    pca : sklearn.decomposition.PCA
        Fitted PCA model
    """
    print(f"Applying PCA with {n_components} components (whiten={whiten})...")
    start_time = time.time()
    
    pca = PCA(n_components=n_components, random_state=random_state, whiten=whiten)
    X_pca = pca.fit_transform(X)
    
    print(f"PCA completed in {time.time() - start_time:.2f} seconds")
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
            pca.explained_variance_ratio_, alpha=0.5, color='blue')
    plt.step(range(1, len(pca.explained_variance_ratio_) + 1), 
             np.cumsum(pca.explained_variance_ratio_), where='mid', color='red')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.savefig('plots/pca_explained_variance.png')
    plt.close()
    
    return X_pca, pca

def apply_tsne(X, n_components=2, perplexity=30, random_state=42):
    """
    Apply t-SNE dimensionality reduction
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix (preferably PCA-reduced)
    n_components : int
        Number of t-SNE components to output
    perplexity : float
        Perplexity parameter for t-SNE
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X_tsne : numpy.ndarray
        t-SNE embedded features
    """
    print(f"Applying t-SNE with {n_components} components (perplexity={perplexity})...")
    start_time = time.time()
    
    tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                random_state=random_state, n_iter=1000)
    X_tsne = tsne.fit_transform(X)
    
    print(f"t-SNE completed in {time.time() - start_time:.2f} seconds")
    return X_tsne

def apply_umap(X, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Apply UMAP dimensionality reduction
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix (preferably PCA-reduced)
    n_components : int
        Number of UMAP components to output
    n_neighbors : int
        Number of neighbors for UMAP
    min_dist : float
        Minimum distance parameter for UMAP
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X_umap : numpy.ndarray
        UMAP embedded features
    """
    if not UMAP_AVAILABLE:
        print("UMAP not available. Install with 'pip install umap-learn'")
        return None
    
    print(f"Applying UMAP with {n_components} components (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    start_time = time.time()
    
    umap = UMAP(n_components=n_components, n_neighbors=n_neighbors, 
                min_dist=min_dist, random_state=random_state)
    X_umap = umap.fit_transform(X)
    
    print(f"UMAP completed in {time.time() - start_time:.2f} seconds")
    return X_umap

def apply_kmeans(X, n_clusters=2, random_state=42, n_init=10):
    """
    Apply K-means clustering
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix (PCA -> t-SNE/UMAP reduced)
    n_clusters : int
        Number of clusters
    random_state : int
        Random seed for reproducibility
    n_init : int
        Number of initializations
        
    Returns:
    --------
    kmeans : sklearn.cluster.KMeans
        Fitted K-means model
    """
    print(f"Applying K-means with {n_clusters} clusters...")
    start_time = time.time()
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    kmeans.fit(X)
    
    print(f"K-means completed in {time.time() - start_time:.2f} seconds")
    return kmeans

def evaluate_clustering(labels_true, labels_pred, method_name="Clustering"):
    """
    Evaluate clustering results against true labels
    
    Parameters:
    -----------
    labels_true : numpy.ndarray
        True cluster labels (if available)
    labels_pred : numpy.ndarray
        Predicted cluster labels
    method_name : str
        Name of the method for reporting
        
    Returns:
    --------
    metrics : dict
        Dictionary of evaluation metrics
    """
    # Calculate clustering metrics
    ari = adjusted_rand_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    
    # For classification metrics, we need to align cluster labels with true labels
    # Since k-means doesn't know the meaning of clusters, we need to match them to the true labels
    
    # Convert labels to binary if they're not already
    if isinstance(labels_true[0], str) or labels_true.dtype == np.dtype('O'):
        labels_true_binary = np.array([1 if label == 'Cancer' else 0 for label in labels_true])
    else:
        labels_true_binary = labels_true.astype(int)
        
    # Check if we need to flip predicted labels to match true labels
    # For example, if cluster 0 corresponds to cancer (1) and cluster 1 to normal (0)
    cluster_0_for_class_1 = np.sum((labels_pred == 0) & (labels_true_binary == 1))
    cluster_1_for_class_1 = np.sum((labels_pred == 1) & (labels_true_binary == 1))
    
    # If majority of cluster 0 is class 1 (Cancer), we need to flip
    flip_labels = cluster_0_for_class_1 > cluster_1_for_class_1
    
    if flip_labels:
        labels_pred_aligned = 1 - labels_pred  # Flip 0->1 and 1->0
    else:
        labels_pred_aligned = labels_pred
    
    # Calculate classification metrics
    accuracy = accuracy_score(labels_true_binary, labels_pred_aligned)
    # Specify zero_division to handle cases where a class might not be predicted
    precision = precision_score(labels_true_binary, labels_pred_aligned, zero_division=0)
    recall = recall_score(labels_true_binary, labels_pred_aligned, zero_division=0)
    f1 = f1_score(labels_true_binary, labels_pred_aligned, zero_division=0)
    
    # Get confusion matrix
    cm = confusion_matrix(labels_true_binary, labels_pred_aligned)
    
    print(f"{method_name} evaluation:")
    print(f"  Adjusted Rand Index: {ari:.4f}")
    print(f"  Normalized Mutual Information: {nmi:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    
    return {
        "ARI": ari, 
        "NMI": nmi,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }

def plot_results(X_2d, y_true, y_pred, title="Clustering Results", 
                 file_name="clustering_results.png", metrics=None):
    """
    Plot clustering results in a 2D space
    
    Parameters:
    -----------
    X_2d : numpy.ndarray
        2D feature matrix (t-SNE or UMAP output)
    y_true : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted cluster assignments
    title : str
        Plot title
    file_name : str
        Output file name
    metrics : dict, optional
        Dictionary of evaluation metrics to include in the plot
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Ensure labels are numeric for coloring
    if isinstance(y_true[0], str) or y_true.dtype == np.dtype('O'):
        # Convert string labels to numeric
        y_true_numeric = np.array([1 if label == 'Cancer' else 0 for label in y_true])
    else:
        y_true_numeric = y_true.astype(int)
    
    # Create a custom colormap if needed
    cmap = plt.cm.get_cmap('viridis', len(np.unique(y_true_numeric)))
    
    # Plot true labels
    scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true_numeric, cmap=cmap, alpha=0.8)
    ax1.set_title("True Labels")
    ax1.set_xlabel("Component 1")
    ax1.set_ylabel("Component 2")
    
    # Create custom legend
    unique_labels = np.unique(y_true_numeric)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=cmap(i), markersize=10,
                               label=f"Class {i} ({'Cancer' if i == 1 else 'Normal'})") 
                        for i in unique_labels]
    ax1.legend(handles=legend_elements, title="Classes")
    
    # Plot predicted clusters
    scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred, cmap='plasma', alpha=0.8)
    ax2.set_title("Predicted Clusters")
    ax2.set_xlabel("Component 1")
    ax2.set_ylabel("Component 2")
    legend2 = ax2.legend(*scatter2.legend_elements(), title="Clusters")
    ax2.add_artist(legend2)
    
    # Add metrics to the title if provided
    if metrics:
        metric_text = f"ARI={metrics['ARI']:.4f}, Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}"
        title = f"{title}\n{metric_text}"
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"plots/{file_name}")
    plt.close()

def run_pca_tsne_kmeans_pipeline(X, y, pca_components=15, tsne_perplexity=30, 
                                n_clusters=2, random_state=42):
    """
    Run the full PCA -> t-SNE -> K-means pipeline
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        True labels (for evaluation)
    pca_components : int
        Number of PCA components
    tsne_perplexity : float
        Perplexity parameter for t-SNE
    n_clusters : int
        Number of clusters for K-means
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    results : dict
        Dictionary of results
    """
    print("\n===== Running PCA -> t-SNE -> K-means Pipeline =====")
    
    # Step 1: Apply PCA
    X_pca, pca_model = apply_pca(X, n_components=pca_components, 
                                 random_state=random_state, whiten=True)
    
    # Step 2: Apply t-SNE
    X_tsne = apply_tsne(X_pca, n_components=2, perplexity=tsne_perplexity, 
                       random_state=random_state)
    
    # Step 3: Apply K-means
    kmeans = apply_kmeans(X_pca, n_clusters=n_clusters, 
                          random_state=random_state, n_init=10)
    
    # Step 4: Evaluate
    metrics = evaluate_clustering(y, kmeans.labels_, method_name="PCA -> t-SNE -> K-means")
    
    # Step 5: Visualize
    plot_results(X_tsne, y, kmeans.labels_, 
                title=f"PCA ({pca_components}) -> t-SNE -> K-means",
                file_name="pca_tsne_kmeans_results.png",
                metrics=metrics)
    
    return {
        "pca_model": pca_model,
        "X_pca": X_pca,
        "X_tsne": X_tsne,
        "kmeans": kmeans,
        "metrics": metrics
    }

def run_pca_umap_kmeans_pipeline(X, y, pca_components=15, umap_neighbors=15, 
                               umap_min_dist=0.1, n_clusters=2, random_state=42):
    """
    Run the full PCA -> UMAP -> K-means pipeline
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        True labels (for evaluation)
    pca_components : int
        Number of PCA components
    umap_neighbors : int
        Number of neighbors for UMAP
    umap_min_dist : float
        Minimum distance parameter for UMAP
    n_clusters : int
        Number of clusters for K-means
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    results : dict
        Dictionary of results
    """
    if not UMAP_AVAILABLE:
        print("UMAP not available. Install with 'pip install umap-learn'")
        return None
    
    print("\n===== Running PCA -> UMAP -> K-means Pipeline =====")
    
    # Step 1: Apply PCA
    X_pca, pca_model = apply_pca(X, n_components=pca_components, 
                                random_state=random_state, whiten=True)
    
    # Step 2: Apply UMAP
    X_umap = apply_umap(X_pca, n_components=2, n_neighbors=umap_neighbors, 
                       min_dist=umap_min_dist, random_state=random_state)
    
    if X_umap is None:
        return None
    
    # Step 3: Apply K-means
    kmeans = apply_kmeans(X_pca, n_clusters=n_clusters, 
                         random_state=random_state, n_init=10)
    
    # Step 4: Evaluate
    metrics = evaluate_clustering(y, kmeans.labels_, method_name="PCA -> UMAP -> K-means")
    
    # Step 5: Visualize
    plot_results(X_umap, y, kmeans.labels_, 
                title=f"PCA ({pca_components}) -> UMAP -> K-means",
                file_name="pca_umap_kmeans_results.png",
                metrics=metrics)
    
    return {
        "pca_model": pca_model,
        "X_pca": X_pca,
        "X_umap": X_umap,
        "kmeans": kmeans,
        "metrics": metrics
    }

def compare_hyperparameters(X, y, pca_components=15):
    """
    Compare different hyperparameter combinations
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        True labels (for evaluation)
    pca_components : int
        Number of PCA components to use
    """
    print("\n===== Hyperparameter Comparison =====")
    
    # Apply PCA once for all experiments
    X_pca, _ = apply_pca(X, n_components=pca_components, 
                        random_state=42, whiten=True)
    
    # t-SNE perplexity values to try
    perplexities = [5, 15, 30, 50, 100]
    tsne_results = []
    
    # Try different t-SNE perplexity values
    for perplexity in perplexities:
        print(f"\nTrying t-SNE with perplexity={perplexity}")
        X_tsne = apply_tsne(X_pca, perplexity=perplexity, random_state=42)
        
        kmeans = apply_kmeans(X_pca, n_clusters=2, random_state=42)
        metrics = evaluate_clustering(y, kmeans.labels_, 
                                     f"t-SNE (perplexity={perplexity}) -> KMeans")
        
        tsne_results.append({
            "perplexity": perplexity,
            "metrics": metrics,
            "X_tsne": X_tsne,
            "labels": kmeans.labels_
        })
    
    # Try different UMAP parameters if available
    if UMAP_AVAILABLE:
        # UMAP parameters to try
        umap_params = [
            {"n_neighbors": 5, "min_dist": 0.1},
            {"n_neighbors": 15, "min_dist": 0.1},
            {"n_neighbors": 30, "min_dist": 0.1},
            {"n_neighbors": 15, "min_dist": 0.01},
            {"n_neighbors": 15, "min_dist": 0.5}
        ]
        umap_results = []
        
        for params in umap_params:
            print(f"\nTrying UMAP with n_neighbors={params['n_neighbors']}, min_dist={params['min_dist']}")
            X_umap = apply_umap(X_pca, n_neighbors=params['n_neighbors'], 
                              min_dist=params['min_dist'], random_state=42)
            
            kmeans = apply_kmeans(X_pca, n_clusters=2, random_state=42)
            metrics = evaluate_clustering(y, kmeans.labels_, 
                                       f"UMAP (n={params['n_neighbors']}, d={params['min_dist']}) -> KMeans")
            
            umap_results.append({
                "params": params,
                "metrics": metrics,
                "X_umap": X_umap,
                "labels": kmeans.labels_
            })
        
        # Plot best UMAP result based on F1-score
        best_umap = max(umap_results, key=lambda x: x["metrics"]["f1_score"])
        best_umap_params = best_umap["params"]
        plot_results(best_umap["X_umap"], y, best_umap["labels"],
                   title=f"Best UMAP (n={best_umap_params['n_neighbors']}, d={best_umap_params['min_dist']})",
                   file_name="best_umap_results.png",
                   metrics=best_umap["metrics"])
    
    # Plot best t-SNE result based on F1-score
    best_tsne = max(tsne_results, key=lambda x: x["metrics"]["f1_score"])
    plot_results(best_tsne["X_tsne"], y, best_tsne["labels"],
               title=f"Best t-SNE (perplexity={best_tsne['perplexity']})",
               file_name="best_tsne_results.png",
               metrics=best_tsne["metrics"])
    
    # Print summary
    print("\n===== Hyperparameter Comparison Summary =====")
    print("\nt-SNE Results:")
    for result in tsne_results:
        print(f"  Perplexity {result['perplexity']}: "
              f"ARI={result['metrics']['ARI']:.4f}, "
              f"Accuracy={result['metrics']['accuracy']:.4f}, "
              f"F1={result['metrics']['f1_score']:.4f}")
    
    if UMAP_AVAILABLE:
        print("\nUMAP Results:")
        for result in umap_results:
            params = result['params']
            print(f"  n_neighbors={params['n_neighbors']}, min_dist={params['min_dist']}: "
                  f"ARI={result['metrics']['ARI']:.4f}, "
                  f"Accuracy={result['metrics']['accuracy']:.4f}, "
                  f"F1={result['metrics']['f1_score']:.4f}")
    
    # Create and save comparison plots
    if UMAP_AVAILABLE and umap_results:
        plt.figure(figsize=(15, 10))
        
        # t-SNE comparison
        plt.subplot(2, 2, 1)
        plt.plot([r['perplexity'] for r in tsne_results], [r['metrics']['ARI'] for r in tsne_results], 'o-', label='ARI')
        plt.plot([r['perplexity'] for r in tsne_results], [r['metrics']['NMI'] for r in tsne_results], 's--', label='NMI')
        plt.xlabel('Perplexity')
        plt.ylabel('Score')
        plt.title('t-SNE Clustering Metrics')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # t-SNE classification metrics
        plt.subplot(2, 2, 2)
        plt.plot([r['perplexity'] for r in tsne_results], [r['metrics']['accuracy'] for r in tsne_results], 'o-', label='Accuracy')
        plt.plot([r['perplexity'] for r in tsne_results], [r['metrics']['precision'] for r in tsne_results], 's--', label='Precision')
        plt.plot([r['perplexity'] for r in tsne_results], [r['metrics']['recall'] for r in tsne_results], '^-', label='Recall')
        plt.plot([r['perplexity'] for r in tsne_results], [r['metrics']['f1_score'] for r in tsne_results], 'd-', label='F1')
        plt.xlabel('Perplexity')
        plt.ylabel('Score')
        plt.title('t-SNE Classification Metrics')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if umap_results:
            # UMAP clustering comparison
            plt.subplot(2, 2, 3)
            x_values = range(len(umap_results))
            plt.plot(x_values, [r['metrics']['ARI'] for r in umap_results], 'o-', label='ARI')
            plt.plot(x_values, [r['metrics']['NMI'] for r in umap_results], 's--', label='NMI')
            plt.xticks(x_values, [f"n={r['params']['n_neighbors']}\nd={r['params']['min_dist']}" for r in umap_results], rotation=45)
            plt.xlabel('UMAP Parameters')
            plt.ylabel('Score')
            plt.title('UMAP Clustering Metrics')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # UMAP classification metrics
            plt.subplot(2, 2, 4)
            plt.plot(x_values, [r['metrics']['accuracy'] for r in umap_results], 'o-', label='Accuracy')
            plt.plot(x_values, [r['metrics']['precision'] for r in umap_results], 's--', label='Precision')
            plt.plot(x_values, [r['metrics']['recall'] for r in umap_results], '^-', label='Recall')
            plt.plot(x_values, [r['metrics']['f1_score'] for r in umap_results], 'd-', label='F1')
            plt.xticks(x_values, [f"n={r['params']['n_neighbors']}\nd={r['params']['min_dist']}" for r in umap_results], rotation=45)
            plt.xlabel('UMAP Parameters')
            plt.ylabel('Score')
            plt.title('UMAP Classification Metrics')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/classification_metrics_comparison.png')
        plt.close()

def main():
    """Main function to run the pipeline"""
    # 1. Load the processed data from data_processing.py instead of raw data
    print("\n===== Using data from data_processing.py =====")
    try:
        # First try to load already processed data
        X, y, feature_names, metadata = load_processed_data(
            file_path="ABIDE2_processed.csv", 
            group_column='group'
        )
        print("Successfully loaded pre-processed data from ABIDE2_processed.csv")
    except (FileNotFoundError, ValueError) as e:
        print(f"Warning: {e}")
        print("Could not load pre-processed data. Running data_processing.py pipeline...")
        
        # If processed data is not available, run the preprocessing
        from data_processing import load_data, preprocess_data, get_optimal_preprocessing
        
        # Load raw data
        df = load_data()
        
        # Determine optimal preprocessing parameters
        preprocess_params = get_optimal_preprocessing(df)
        
        # Process the data
        processed_df = preprocess_data(df, preprocess_params)
        
        # Save processed data if it wasn't already saved
        processed_df.to_csv('ABIDE2_processed.csv', index=False)
        
        # Extract features and target
        metadata_cols = [col for col in processed_df.columns if not col.startswith('fs')]
        feature_cols = [col for col in processed_df.columns if col.startswith('fs')]
        
        X = processed_df[feature_cols].values
        
        # Convert group values to numeric
        group_values = processed_df['group'].values
        if group_values.dtype == np.dtype('O'):  # If object dtype (strings)
            print("Converting group labels to numeric values...")
            y = np.array([1 if val == 'Cancer' else 0 for val in group_values])
        else:
            y = group_values
            
        feature_names = feature_cols
        metadata = processed_df[metadata_cols]
    
    # No need for additional outlier handling since that was done in data_processing.py
    X_clean = X
    
    # 3. Run PCA -> t-SNE -> K-means pipeline
    tsne_results = run_pca_tsne_kmeans_pipeline(
        X_clean, y, 
        pca_components=15,
        tsne_perplexity=30,
        n_clusters=2, 
        random_state=42
    )
    
    # 4. Run PCA -> UMAP -> K-means pipeline (if UMAP is available)
    if UMAP_AVAILABLE:
        umap_results = run_pca_umap_kmeans_pipeline(
            X_clean, y, 
            pca_components=15,
            umap_neighbors=15, 
            umap_min_dist=0.1,
            n_clusters=2, 
            random_state=42
        )
    
    # 5. Compare different hyperparameters
    compare_hyperparameters(X_clean, y, pca_components=15)
    
    print("\nAll pipeline steps completed. Results saved in the 'plots' directory.")

if __name__ == "__main__":
    main() 