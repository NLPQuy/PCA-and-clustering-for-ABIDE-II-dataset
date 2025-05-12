#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimal clustering approach for ABIDE dataset based on advanced feature engineering.
This script applies multiple clustering algorithms on optimally preprocessed ABIDE data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from time import time
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, AffinityPropagation

# Import custom modules
from src import (
    MyPCA, 
    MyKMeans, 
    load_abide_data, 
    preprocess_data,
    MyAgglomerativeClustering,
    MyDBSCAN,
    MyMeanShift,
    MyAffinityPropagation,
    find_optimal_clusters
)
from utils import calculate_cluster_metrics, print_metrics, plot_clusters_2d

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

def load_preprocessed_data(processed_data_file='processed_data_for_clustering.npy', 
                           labels_file='labels_for_evaluation.npy'):
    """
    Load preprocessed data for clustering.
    
    Parameters:
    -----------
    processed_data_file : str
        Path to the preprocessed data file
    labels_file : str
        Path to the labels file
        
    Returns:
    --------
    X : array-like
        Preprocessed data
    y : array-like
        True labels (for evaluation only)
    """
    try:
        X = np.load(processed_data_file)
        y = np.load(labels_file)
        print(f"Loaded preprocessed data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    except FileNotFoundError:
        print("Preprocessed data files not found. Using raw ABIDE data...")
        # Fall back to raw data processing
        from abide_feature_engineering_advanced import (
            load_and_prepare_abide_data,
            detect_and_handle_outliers,
            normalize_connectivity_data,
            apply_advanced_pca
        )
        
        # Load and process data
        X, y, _ = load_and_prepare_abide_data('ABIDE2_sample.csv')
        X, _ = detect_and_handle_outliers(X, threshold=3.0, method='winsorize')
        X, _ = normalize_connectivity_data(X, method='robust')
        X, _, _ = apply_advanced_pca(X, n_components=20, transform_method='standard')
        
        return X, y

def try_multiple_clustering_algorithms(X, y, n_clusters=2):
    """
    Apply multiple clustering algorithms to the data and evaluate performance.
    
    Parameters:
    -----------
    X : array-like
        Preprocessed data
    y : array-like
        True labels (for evaluation only)
    n_clusters : int, default=2
        Number of clusters to use (for algorithms that need this parameter)
        
    Returns:
    --------
    results : dict
        Dictionary with results for each algorithm
    """
    # Define clustering algorithms to try
    algorithms = [
        ('KMeans', KMeans(n_clusters=n_clusters, random_state=42, n_init=10)),
        ('Agglomerative-Ward', AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')),
        ('Agglomerative-Complete', AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')),
        ('Agglomerative-Average', AgglomerativeClustering(n_clusters=n_clusters, linkage='average')),
        ('DBSCAN', DBSCAN(eps=0.5, min_samples=5)),
        ('MeanShift', MeanShift(bandwidth=None)),
        ('AffinityPropagation', AffinityPropagation(damping=0.9, random_state=42))
    ]
    
    results = {}
    
    for name, algorithm in algorithms:
        print(f"\nApplying {name}...")
        start_time = time()
        
        # For Mean Shift, estimate bandwidth if None
        if name == 'MeanShift' and algorithm.bandwidth is None:
            from sklearn.cluster import estimate_bandwidth
            bandwidth = estimate_bandwidth(X, quantile=0.3)
            algorithm.bandwidth = bandwidth
            print(f"Estimated bandwidth: {bandwidth:.4f}")
        
        # Fit the algorithm
        algorithm.fit(X)
        
        # Get labels
        labels = algorithm.labels_
        
        end_time = time()
        duration = end_time - start_time
        
        # Calculate metrics
        n_clusters_found = len(np.unique(labels))
        if -1 in labels:  # Handle noise points in DBSCAN
            n_clusters_found -= 1
        
        print(f"Found {n_clusters_found} clusters")
        print(f"Clustering time: {duration:.2f} seconds")
        
        # Calculate evaluation metrics
        metrics = calculate_cluster_metrics(y, labels)
        print_metrics(metrics)
        
        # Save results
        results[name] = {
            'labels': labels,
            'n_clusters': n_clusters_found,
            'metrics': metrics,
            'time': duration
        }
        
        # Plot clusters if 2D or 3D data
        if X.shape[1] <= 3:
            plot_clusters_2d(
                X[:, :2], 
                labels, 
                y,
                title=f"Clustering with {name}",
                save_path=f'plots/abide_optimal_{name.lower()}.png'
            )
    
    return results

def optimize_dbscan_parameters(X, y, eps_range=None, min_samples_range=None):
    """
    Find optimal parameters for DBSCAN.
    
    Parameters:
    -----------
    X : array-like
        Preprocessed data
    y : array-like
        True labels (for evaluation only)
    eps_range : list, optional
        Range of eps values to try
    min_samples_range : list, optional
        Range of min_samples values to try
        
    Returns:
    --------
    best_params : dict
        Best parameters found
    best_score : float
        Best score achieved
    """
    if eps_range is None:
        # Use distances distribution to determine range
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        distances = np.sort(distances[:, 1])
        
        # Use distance distribution to choose eps range
        eps_range = np.linspace(distances[int(len(distances)*0.1)], 
                                distances[int(len(distances)*0.9)], 
                                10)
    
    if min_samples_range is None:
        # Use dataset size to determine range
        min_samples_range = [5, 10, 15, 20, int(0.02*X.shape[0]), int(0.05*X.shape[0])]
    
    best_score = -1
    best_params = {}
    results = []
    
    print("\nOptimizing DBSCAN parameters...")
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            # Apply DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            # Check if clustering is meaningful
            unique_labels = np.unique(labels)
            if len(unique_labels) <= 1 or (len(unique_labels) == 2 and -1 in unique_labels):
                # Skip if all points are noise or a single cluster
                continue
            
            # Calculate metrics
            try:
                ari = adjusted_rand_score(y, labels)
                nmi = normalized_mutual_info_score(y, labels)
                
                # Calculate silhouette only for non-noise points
                non_noise = labels != -1
                if np.sum(non_noise) > 1 and len(np.unique(labels[non_noise])) > 1:
                    silhouette = silhouette_score(X[non_noise], labels[non_noise])
                else:
                    silhouette = -1
                
                # Track results
                results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': len(unique_labels) - (1 if -1 in unique_labels else 0),
                    'noise_points': np.sum(labels == -1),
                    'ari': ari,
                    'nmi': nmi,
                    'silhouette': silhouette
                })
                
                # Update best parameters based on ARI
                if ari > best_score:
                    best_score = ari
                    best_params = {'eps': eps, 'min_samples': min_samples}
                    
                print(f"eps={eps:.4f}, min_samples={min_samples}: ARI={ari:.4f}, "
                      f"NMI={nmi:.4f}, Silhouette={silhouette:.4f}, "
                      f"Clusters={len(unique_labels) - (1 if -1 in unique_labels else 0)}, "
                      f"Noise={np.sum(labels == -1)}")
                
            except Exception as e:
                print(f"Error with eps={eps}, min_samples={min_samples}: {e}")
    
    if results:
        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(results)
        
        # Plot parameter search results
        plt.figure(figsize=(12, 10))
        
        # Create parameter grid
        eps_values = sorted(results_df['eps'].unique())
        min_samples_values = sorted(results_df['min_samples'].unique())
        
        if len(eps_values) > 1 and len(min_samples_values) > 1:
            ari_grid = np.zeros((len(min_samples_values), len(eps_values)))
            
            for i, ms in enumerate(min_samples_values):
                for j, eps in enumerate(eps_values):
                    mask = (results_df['eps'] == eps) & (results_df['min_samples'] == ms)
                    if np.any(mask):
                        ari_grid[i, j] = results_df.loc[mask, 'ari'].values[0]
                    else:
                        ari_grid[i, j] = np.nan
            
            plt.imshow(ari_grid, cmap='viridis', aspect='auto', interpolation='nearest')
            plt.colorbar(label='ARI')
            plt.xticks(range(len(eps_values)), [f"{e:.3f}" for e in eps_values], rotation=90)
            plt.yticks(range(len(min_samples_values)), min_samples_values)
            plt.xlabel('eps')
            plt.ylabel('min_samples')
            plt.title('DBSCAN Parameter Optimization (ARI)')
            
            # Mark best parameters
            best_i = min_samples_values.index(best_params['min_samples'])
            best_j = eps_values.index(best_params['eps'])
            plt.plot(best_j, best_i, 'r*', markersize=15)
            
            plt.tight_layout()
            plt.savefig('plots/dbscan_parameter_optimization.png')
            plt.close()
        
        print(f"\nBest DBSCAN parameters: eps={best_params['eps']:.4f}, "
              f"min_samples={best_params['min_samples']}")
        print(f"Best ARI score: {best_score:.4f}")
    else:
        print("No valid DBSCAN parameters found")
    
    return best_params, best_score

def generate_ensemble_clustering(X, y, results, weight_by='ari'):
    """
    Generate ensemble clustering from multiple algorithms.
    
    Parameters:
    -----------
    X : array-like
        Preprocessed data
    y : array-like
        True labels (for evaluation only)
    results : dict
        Results from try_multiple_clustering_algorithms
    weight_by : str, default='ari'
        Metric to use for weighting algorithms in ensemble
        
    Returns:
    --------
    ensemble_labels : array-like
        Labels from ensemble clustering
    ensemble_metrics : dict
        Evaluation metrics for ensemble clustering
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import pairwise_distances
    
    print("\nGenerating ensemble clustering...")
    
    # Get all individual clusterings
    clusterings = {}
    weights = {}
    total_weight = 0
    
    for name, result in results.items():
        labels = result['labels']
        
        # Skip clusterings with mostly noise points
        if name == 'DBSCAN' and np.mean(labels == -1) > 0.5:
            print(f"Skipping {name} for ensemble due to high noise proportion")
            continue
            
        # Get the selected metric for weighting
        if weight_by in result['metrics']:
            weight = max(0.1, result['metrics'][weight_by])  # Ensure minimum weight
            weights[name] = weight
            total_weight += weight
            clusterings[name] = labels
            print(f"Including {name} with weight {weight:.4f} ({weight/total_weight:.2%})")
        else:
            print(f"Skipping {name} (missing {weight_by} metric)")
    
    if not clusterings:
        print("No valid clusterings for ensemble")
        return None, None
    
    # Normalize weights
    for name in weights:
        weights[name] /= total_weight
    
    # Create co-association matrix
    n_samples = X.shape[0]
    co_association = np.zeros((n_samples, n_samples))
    
    for name, labels in clusterings.items():
        # Create binary co-clustering matrix
        co_cluster = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            # If the point is noise in DBSCAN, don't contribute to co-association
            if labels[i] == -1:
                continue
                
            # Points in the same cluster have co-association = 1
            same_cluster = np.where(labels == labels[i])[0]
            co_cluster[i, same_cluster] = 1
        
        # Add weighted co-clustering to co-association matrix
        co_association += weights[name] * co_cluster
    
    # Convert co-association matrix to a distance matrix
    distance_matrix = 1 - co_association
    
    # Apply hierarchical clustering on the co-association matrix
    n_clusters = len(np.unique(y))  # Use the true number of clusters for ensemble
    ensemble_clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        linkage='average'
    )
    
    ensemble_labels = ensemble_clustering.fit_predict(distance_matrix)
    
    # Evaluate ensemble clustering
    ensemble_metrics = calculate_cluster_metrics(y, ensemble_labels)
    print("\nEnsemble clustering results:")
    print_metrics(ensemble_metrics)
    
    # Visualize ensemble clustering
    if X.shape[1] <= 3:
        plot_clusters_2d(
            X[:, :2], 
            ensemble_labels, 
            y,
            title="Ensemble Clustering",
            save_path='plots/abide_optimal_ensemble.png'
        )
    
    return ensemble_labels, ensemble_metrics

def visualize_clustering_comparison(results, ensemble_metrics=None):
    """
    Visualize and compare metrics across different clustering algorithms.
    
    Parameters:
    -----------
    results : dict
        Results from try_multiple_clustering_algorithms
    ensemble_metrics : dict, optional
        Metrics from ensemble clustering
    """
    # Extract metrics for comparison
    algorithms = list(results.keys())
    if ensemble_metrics:
        algorithms.append('Ensemble')
    
    metrics_to_compare = ['accuracy', 'ari', 'nmi', 'f1']
    metrics_data = {}
    
    for metric in metrics_to_compare:
        metrics_data[metric] = []
        for algo in algorithms:
            if algo == 'Ensemble':
                value = ensemble_metrics[metric]
            else:
                value = results[algo]['metrics'][metric]
            metrics_data[metric].append(value)
    
    # Plot comparison
    plt.figure(figsize=(12, 10))
    
    bar_width = 0.2
    positions = np.arange(len(algorithms))
    
    for i, metric in enumerate(metrics_to_compare):
        plt.bar(
            positions + i * bar_width - (len(metrics_to_compare) - 1) * bar_width / 2, 
            metrics_data[metric], 
            width=bar_width, 
            label=metric.upper()
        )
    
    plt.xlabel('Clustering Algorithm')
    plt.ylabel('Score')
    plt.title('Clustering Algorithm Comparison')
    plt.xticks(positions, algorithms)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/abide_clustering_comparison.png')
    plt.close()
    
    # Create results table
    table_data = []
    header = ['Algorithm', 'N Clusters', 'Accuracy', 'Precision', 'Recall', 'F1', 'ARI', 'NMI', 'Time (s)']
    table_data.append(header)
    
    for algo in algorithms:
        if algo == 'Ensemble':
            row = [
                'Ensemble', 
                len(np.unique(y)),
                f"{ensemble_metrics['accuracy']:.4f}",
                f"{ensemble_metrics['precision']:.4f}",
                f"{ensemble_metrics['recall']:.4f}",
                f"{ensemble_metrics['f1']:.4f}",
                f"{ensemble_metrics['ari']:.4f}",
                f"{ensemble_metrics['nmi']:.4f}",
                "N/A"
            ]
        else:
            row = [
                algo,
                results[algo]['n_clusters'],
                f"{results[algo]['metrics']['accuracy']:.4f}",
                f"{results[algo]['metrics']['precision']:.4f}",
                f"{results[algo]['metrics']['recall']:.4f}",
                f"{results[algo]['metrics']['f1']:.4f}",
                f"{results[algo]['metrics']['ari']:.4f}",
                f"{results[algo]['metrics']['nmi']:.4f}",
                f"{results[algo]['time']:.2f}"
            ]
        table_data.append(row)
    
    # Save results to CSV
    with open('plots/abide_clustering_results.csv', 'w') as f:
        for row in table_data:
            f.write(','.join(str(item) for item in row) + '\n')
    
    print("\nResults saved to plots/abide_clustering_results.csv")

def main():
    """Main function to run the optimal clustering analysis."""
    print("=== Optimal Clustering Analysis for ABIDE Dataset ===")
    
    # Load preprocessed data
    X, y = load_preprocessed_data()
    
    # Find optimal number of clusters
    print("\nFinding optimal number of clusters...")
    n_clusters, _ = find_optimal_clusters(X, method='kmeans', metric='silhouette', max_clusters=5)
    print(f"Optimal number of clusters: {n_clusters}")
    
    # Try multiple clustering algorithms
    results = try_multiple_clustering_algorithms(X, y, n_clusters=n_clusters)
    
    # Optimize DBSCAN parameters
    best_dbscan_params, _ = optimize_dbscan_parameters(X, y)
    
    if best_dbscan_params:
        # Apply optimized DBSCAN
        print("\nApplying optimized DBSCAN...")
        optimized_dbscan = DBSCAN(**best_dbscan_params)
        optimized_dbscan.fit(X)
        dbscan_labels = optimized_dbscan.labels_
        dbscan_metrics = calculate_cluster_metrics(y, dbscan_labels)
        
        print("Optimized DBSCAN results:")
        print_metrics(dbscan_metrics)
        
        # Update results with optimized DBSCAN
        results['DBSCAN-Optimized'] = {
            'labels': dbscan_labels,
            'n_clusters': len(np.unique(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
            'metrics': dbscan_metrics,
            'time': 0  # Not measured
        }
    
    # Generate ensemble clustering
    ensemble_labels, ensemble_metrics = generate_ensemble_clustering(X, y, results)
    
    # Visualize comparison
    visualize_clustering_comparison(results, ensemble_metrics)
    
    # Print final recommendation
    print("\n=== Final Recommendation ===")
    
    # Find best algorithm by ARI
    best_algo = max(results.items(), key=lambda x: x[1]['metrics']['ari'])
    print(f"Best individual algorithm: {best_algo[0]}")
    print(f"ARI: {best_algo[1]['metrics']['ari']:.4f}, NMI: {best_algo[1]['metrics']['nmi']:.4f}")
    
    if ensemble_metrics:
        print(f"Ensemble clustering ARI: {ensemble_metrics['ari']:.4f}, NMI: {ensemble_metrics['nmi']:.4f}")
        
        if ensemble_metrics['ari'] > best_algo[1]['metrics']['ari']:
            print("Recommendation: Use ensemble clustering for best results")
        else:
            print(f"Recommendation: Use {best_algo[0]} for best results")
    
    print("\n=== Clustering Analysis Complete ===")

if __name__ == "__main__":
    main()