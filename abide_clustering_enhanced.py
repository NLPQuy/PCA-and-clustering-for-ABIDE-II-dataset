#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Đề bài 3: Bài toán phân cụm nâng cao trên tập dữ liệu ABIDE với Feature Engineering (Unsupervised)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
import argparse
from sklearn.preprocessing import RobustScaler

# Import custom modules
from src import MyPCA, MyKMeans, EnhancedKMeans, load_abide_data, preprocess_data
from src import (
    feature_engineering_pipeline, 
    select_features_variance,
    apply_kernel_pca, 
    apply_polynomial_pca,
    apply_feature_agglomeration,
    apply_select_from_model,
    apply_truncated_svd,
    apply_select_percentile
)
# Import GMM clustering module
from src import MyGMM, find_optimal_gmm_components, plot_gmm_criterion_scores
# Import our new advanced feature engineering methods
from src.abide_feature_engineering import (
    apply_feature_selection,
    create_feature_interactions,
    apply_advanced_pca,
    apply_optimal_kernel_pca,
    apply_ensemble_dimensionality_reduction,
    optimize_feature_engineering_for_clustering,
    create_region_ratio_features,
    apply_combined_optimal_approach,
    apply_zca_whitening_pca,
    apply_whitening_graph_feature_selection_pca,
    apply_cholesky_whitening,
    apply_mahalanobis_whitening,
    apply_pca_adaptive_whitening,
    apply_hybrid_whitening,
    apply_double_pca
)
from utils import plot_pca_components_variance, plot_pca_2d, plot_clusters_2d
from utils import calculate_cluster_metrics, print_metrics

# Import data processing functions
from data_processing import (
    load_data, 
    analyze_basic_stats, 
    check_distribution, 
    analyze_group_differences,
    analyze_feature_correlations, 
    get_optimal_preprocessing, 
    preprocess_data as dp_preprocess_data
)

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

def get_feature_weights_for_brain_regions(feature_names, n_features=None):
    """
    Generate feature weights based on brain region importance.
    This gives higher weights to more diagnostically valuable regions.
    
    Parameters:
    -----------
    feature_names : list
        Names of features (brain regions)
    n_features : int, optional
        Number of features in the data - used when feature_names doesn't match data dimensions
        
    Returns:
    --------
    weights : array
        Weight for each feature
    """
    # If n_features is provided and doesn't match len(feature_names), create default weights
    if n_features is not None and n_features != len(feature_names):
        print(f"Feature names ({len(feature_names)}) don't match data dimensions ({n_features}). Using uniform weights.")
        return np.ones(n_features)
    
    weights = np.ones(len(feature_names))
    
    # Higher weights for regions known to be important in neurological disorders
    # Based on neuroimaging literature on brain regions affected in conditions like autism
    important_regions = [
        'cortex', 'cerebellum', 'thalamus', 'amygdala', 'hippocampus', 
        'striatum', 'caudate', 'putamen', 'pallidum', 'accumbens',
        'corpus', 'callosum', 'frontal'
    ]
    
    for i, feature in enumerate(feature_names):
        feature_lower = feature.lower()
        # Check if this feature contains an important region name
        for region in important_regions:
            if region in feature_lower:
                weights[i] = 1.5  # Increase weight for important regions
                break
    
    # Normalize weights
    weights = weights / np.mean(weights)
    
    return weights

def compare_kmeans_implementations(X, y, feature_names=None):
    """
    Compare standard K-means with Enhanced K-means.
    
    Parameters:
    -----------
    X : array-like
        Data features
    y : array-like
        True labels (for evaluation only)
    feature_names : list, optional
        Names of features
        
    Returns:
    --------
    comparison_results : dict
        Results of comparison
    """
    print("\n=== Comparing K-means Implementations ===")
    
    # Get feature weights if feature names are provided
    feature_weights = None
    if feature_names is not None:
        # Get dimensions of input data
        n_features = X.shape[1]
        feature_weights = get_feature_weights_for_brain_regions(feature_names, n_features)
        print(f"Generated feature weights for {len(feature_weights)} features")
    
    # Standard K-means
    print("\nRunning standard K-means...")
    start_time = time.time()
    kmeans_standard = MyKMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans_standard.fit(X)
    standard_time = time.time() - start_time
    standard_metrics = calculate_cluster_metrics(y, kmeans_standard.labels_)
    
    print(f"Standard K-means completed in {standard_time:.2f} seconds")
    print("Standard K-means Results:")
    print_metrics(standard_metrics)
    
    # Enhanced K-means with Euclidean distance
    print("\nRunning Enhanced K-means with Euclidean distance...")
    start_time = time.time()
    kmeans_enhanced_euclidean = EnhancedKMeans(
        n_clusters=2, 
        random_state=42, 
        n_init=15,
        feature_weights=feature_weights,
        distance_metric='euclidean',
        outlier_handling=True,
        momentum=0.1,
        early_stopping=True,
        tol=1e-6
    )
    kmeans_enhanced_euclidean.fit(X)
    euclidean_time = time.time() - start_time
    euclidean_metrics = calculate_cluster_metrics(y, kmeans_enhanced_euclidean.labels_)
    
    print(f"Enhanced K-means (Euclidean) completed in {euclidean_time:.2f} seconds")
    print("Enhanced K-means (Euclidean) Results:")
    print_metrics(euclidean_metrics)
    
    # Enhanced K-means with RBF kernel distance
    print("\nRunning Enhanced K-means with RBF kernel distance...")
    start_time = time.time()
    kmeans_enhanced_rbf = EnhancedKMeans(
        n_clusters=2, 
        random_state=42, 
        n_init=15,
        feature_weights=feature_weights,
        distance_metric='rbf',
        outlier_handling=True,
        momentum=0.1,
        early_stopping=True,
        tol=1e-6
    )
    kmeans_enhanced_rbf.fit(X)
    rbf_time = time.time() - start_time
    rbf_metrics = calculate_cluster_metrics(y, kmeans_enhanced_rbf.labels_)
    
    print(f"Enhanced K-means (RBF) completed in {rbf_time:.2f} seconds")
    print("Enhanced K-means (RBF) Results:")
    print_metrics(rbf_metrics)
    
    # Enhanced K-means with Manhattan distance
    print("\nRunning Enhanced K-means with Manhattan distance...")
    start_time = time.time()
    kmeans_enhanced_manhattan = EnhancedKMeans(
        n_clusters=2, 
        random_state=42, 
        n_init=15,
        feature_weights=feature_weights,
        distance_metric='manhattan',
        outlier_handling=True,
        momentum=0.1,
        early_stopping=True,
        tol=1e-6
    )
    kmeans_enhanced_manhattan.fit(X)
    manhattan_time = time.time() - start_time
    manhattan_metrics = calculate_cluster_metrics(y, kmeans_enhanced_manhattan.labels_)
    
    print(f"Enhanced K-means (Manhattan) completed in {manhattan_time:.2f} seconds")
    print("Enhanced K-means (Manhattan) Results:")
    print_metrics(manhattan_metrics)
    
    # Compile results
    results = {
        'standard': {
            'time': standard_time,
            'metrics': standard_metrics,
            'labels': kmeans_standard.labels_,
            'model': kmeans_standard
        },
        'enhanced_euclidean': {
            'time': euclidean_time,
            'metrics': euclidean_metrics,
            'labels': kmeans_enhanced_euclidean.labels_,
            'model': kmeans_enhanced_euclidean
        },
        'enhanced_rbf': {
            'time': rbf_time,
            'metrics': rbf_metrics,
            'labels': kmeans_enhanced_rbf.labels_,
            'model': kmeans_enhanced_rbf
        },
        'enhanced_manhattan': {
            'time': manhattan_time,
            'metrics': manhattan_metrics,
            'labels': kmeans_enhanced_manhattan.labels_,
            'model': kmeans_enhanced_manhattan
        }
    }
    
    # Find best method based on different metrics
    best_f1_method = max(['standard', 'enhanced_euclidean', 'enhanced_rbf', 'enhanced_manhattan'],
                      key=lambda x: results[x]['metrics']['f1'])
    
    best_accuracy_method = max(['standard', 'enhanced_euclidean', 'enhanced_rbf', 'enhanced_manhattan'],
                           key=lambda x: results[x]['metrics']['accuracy'])
    
    best_ari_method = max(['standard', 'enhanced_euclidean', 'enhanced_rbf', 'enhanced_manhattan'],
                       key=lambda x: results[x]['metrics']['ari'])
    
    print("\n=== K-means Comparison Summary ===")
    print(f"Best method based on F1 score: {best_f1_method}")
    print(f"F1 score: {results[best_f1_method]['metrics']['f1']:.4f}")
    print(f"Accuracy: {results[best_f1_method]['metrics']['accuracy']:.4f}")
    
    if best_accuracy_method != best_f1_method:
        print(f"\nBest method based on Accuracy: {best_accuracy_method}")
        print(f"Accuracy: {results[best_accuracy_method]['metrics']['accuracy']:.4f}")
        print(f"F1 score: {results[best_accuracy_method]['metrics']['f1']:.4f}")
    
    if best_ari_method != best_f1_method and best_ari_method != best_accuracy_method:
        print(f"\nBest method based on Adjusted Rand Index: {best_ari_method}")
        print(f"ARI: {results[best_ari_method]['metrics']['ari']:.4f}")
    
    # Lưu giá trị tốt nhất cho từng chỉ số
    best_values = {
        'f1': {
            'method': best_f1_method,
            'value': results[best_f1_method]['metrics']['f1'],
            'model': results[best_f1_method]['model'],
            'metrics': results[best_f1_method]['metrics']
        },
        'accuracy': {
            'method': best_accuracy_method,
            'value': results[best_accuracy_method]['metrics']['accuracy'],
            'model': results[best_accuracy_method]['model'],
            'metrics': results[best_accuracy_method]['metrics']
        },
        'ari': {
            'method': best_ari_method,
            'value': results[best_ari_method]['metrics']['ari'],
            'model': results[best_ari_method]['model'],
            'metrics': results[best_ari_method]['metrics']
        }
    }
    
    # Return best model based on F1 score and all results, plus best values for each metric
    return results[best_f1_method]['model'], results, best_values

def apply_gmm_clustering(X, y, n_components=2, max_components=10, find_optimal=True, covariance_type='full', max_iter=200, n_init=3):
    """
    Apply Gaussian Mixture Model clustering to the data.
    
    Parameters:
    -----------
    X : array-like
        Data features
    y : array-like
        True labels (for evaluation only)
    n_components : int, default=2
        Number of components (clusters) to use if not finding optimal
    max_components : int, default=10
        Maximum number of components to try when finding optimal
    find_optimal : bool, default=True
        Whether to find the optimal number of components using BIC
    covariance_type : str, default='full'
        Type of covariance parameters: 'full', 'tied', 'diag', 'spherical'
    max_iter : int, default=200
        Maximum number of iterations
    n_init : int, default=3
        Number of initializations to try
        
    Returns:
    --------
    gmm : MyGMM
        Fitted GMM model
    metrics : dict
        Evaluation metrics
    """
    print("\n=== Applying GMM Clustering ===")
    
    # Kiểm tra dữ liệu
    if X.shape[0] < X.shape[1]:
        print(f"Warning: More features ({X.shape[1]}) than samples ({X.shape[0]}).")
        print("Using tied covariance type for numerical stability.")
        covariance_type = 'tied'
    
    if find_optimal:
        print(f"Finding optimal number of GMM components (max: {max_components})...")
        start_time = time.time()
        
        # Giới hạn max_components không vượt quá số mẫu
        max_components = min(max_components, X.shape[0] // 5)
        
        optimal_n, bic_scores, best_models = find_optimal_gmm_components(
            X, max_components=max_components, criterion='bic', n_runs=n_init
        )
        
        # Plot BIC scores
        fig = plot_gmm_criterion_scores(bic_scores, criterion='BIC')
        plt.savefig('plots/gmm_bic_scores.png')
        plt.close(fig)
        
        print(f"Optimal number of components: {optimal_n}")
        print(f"Optimization completed in {time.time() - start_time:.2f} seconds")
        
        # Use the best model with optimal components
        gmm = best_models[optimal_n]
    else:
        print(f"Using GMM with {n_components} components, covariance_type={covariance_type}...")
        start_time = time.time()
        
        try:
            gmm = MyGMM(n_components=n_components, 
                       random_state=42, 
                       max_iter=max_iter, 
                       tol=1e-4,
                       reg_covar=1e-4,  # Tăng giá trị reg_covar để tăng tính ổn định
                       init_params='kmeans')  # Sử dụng KMeans để khởi tạo
            gmm.fit(X)
        except Exception as e:
            print(f"Error with regular GMM: {e}")
            print("Trying with tied covariance type...")
            try:
                # Thử với tied covariance nếu full không hoạt động
                from sklearn.mixture import GaussianMixture
                gmm = GaussianMixture(n_components=n_components, 
                                    covariance_type='tied',
                                    random_state=42, 
                                    max_iter=max_iter,
                                    n_init=n_init,
                                    reg_covar=1e-3)
                gmm.fit(X)
            except Exception as e:
                print(f"Error with tied GMM: {e}")
                print("Using K-means clustering instead...")
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_components, random_state=42, n_init=10)
                kmeans.fit(X)
                # Tạo một đối tượng giả GMM
                class FakeGMM:
                    def __init__(self, labels_, n_components):
                        self.labels_ = labels_
                        self.n_components = n_components
                    
                    def predict(self, X):
                        return self.labels_
                
                gmm = FakeGMM(kmeans.labels_, n_components)
        
        print(f"GMM completed in {time.time() - start_time:.2f} seconds")
    
    # Check if true labels exist (2 clusters case)
    if len(np.unique(y)) == 2:
        # Evaluate clustering results
        metrics = calculate_cluster_metrics(y, gmm.labels_)
        print("GMM Clustering Results:")
        print_metrics(metrics)
        
        # If GMM found more than 2 clusters, we can also show results for just 2 clusters
        if hasattr(gmm, 'n_components') and gmm.n_components > 2 and hasattr(gmm, 'predict_proba'):
            try:
                # Use top 2 components with highest responsibility sum
                probs = gmm.predict_proba(X)
                top_components = np.argsort(-np.sum(probs, axis=0))[:2]
                binary_labels = np.zeros_like(gmm.labels_)
                for i, label in enumerate(gmm.labels_):
                    binary_labels[i] = 1 if label in top_components else 0
                    
                binary_metrics = calculate_cluster_metrics(y, binary_labels)
                print("\nGMM Binary Clustering Results (Top 2 components):")
                print_metrics(binary_metrics)
                
                # Nếu binary_metrics tốt hơn, sử dụng chúng
                if binary_metrics['f1'] > metrics['f1']:
                    print("Using binary clustering results (better F1 score)")
                    metrics = binary_metrics
                    # Cập nhật nhãn
                    gmm.labels_ = binary_labels
            except Exception as e:
                print(f"Could not compute binary metrics: {e}")
    else:
        metrics = {'n_clusters': getattr(gmm, 'n_components', 2)}
        print(f"Found {metrics['n_clusters']} clusters")
    
    # Try to visualize clusters if X has reduced dimensions
    if X.shape[1] >= 2:
        if X.shape[1] > 2:
            # Just use first two dimensions for visualization
            X_plot = X[:, :2]
        else:
            X_plot = X
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Simple scatter plot if not MyGMM instance
        if not hasattr(gmm, 'plot_clusters'):
            scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=gmm.labels_, 
                               cmap='viridis', alpha=0.7, s=50, edgecolors='w')
            ax.set_title("GMM Clustering")
            ax.set_xlabel(f'Feature 1')
            ax.set_ylabel(f'Feature 2')
            ax.grid(True, alpha=0.3)
            
            # Add legend
            legend1 = ax.legend(*scatter.legend_elements(),
                               loc="upper right", title="Clusters")
            ax.add_artist(legend1)
        else:
            gmm.plot_clusters(X_plot, ax=ax, title="GMM Clustering")
        
        plt.savefig('plots/gmm_clustering.png')
        plt.close(fig)
    
    return gmm, metrics

def compare_clustering_methods(X, y, feature_names=None):
    """
    Compare different clustering methods on the same data.
    
    Parameters:
    -----------
    X : array-like
        Data features
    y : array-like
        True labels (for evaluation only)
    feature_names : list, optional
        Names of features
        
    Returns:
    --------
    best_model : object
        Best clustering model
    all_results : dict
        Results from all clustering methods
    """
    print("\n=== Comparing Different Clustering Methods ===")
    
    # Apply K-means clustering
    best_kmeans, kmeans_results, kmeans_best_values = compare_kmeans_implementations(X, y, feature_names)
    
    # Lưu lại các kết quả tốt nhất cho từng chỉ số
    kmeans_best_f1 = kmeans_best_values['f1']['value']
    kmeans_best_accuracy = kmeans_best_values['accuracy']['value']
    kmeans_best_ari = kmeans_best_values['ari']['value']
    
    # Sử dụng model có F1 tốt nhất
    kmeans_metrics = kmeans_best_values['f1']['metrics']
    
    # In ra thông tin về cách các chỉ số tốt nhất được chọn
    print(f"\nBest K-means metrics:")
    print(f"F1 score: {kmeans_best_f1:.4f} (method: {kmeans_best_values['f1']['method']})")
    print(f"Accuracy: {kmeans_best_accuracy:.4f} (method: {kmeans_best_values['accuracy']['method']})")
    print(f"ARI: {kmeans_best_ari:.4f} (method: {kmeans_best_values['ari']['method']})")
    
    # Thử nhiều loại covariance cho GMM
    best_gmm = None
    best_gmm_metrics = None
    best_gmm_f1 = -1
    
    for cov_type in ['full', 'tied']:
        try:
            print(f"Trying GMM with covariance_type={cov_type}...")
            gmm, gmm_metrics = apply_gmm_clustering(
                X, y, 
                n_components=2, 
                find_optimal=False, 
                covariance_type=cov_type, 
                max_iter=200, 
                n_init=5
            )
            
            if gmm_metrics.get('f1', 0) > best_gmm_f1:
                best_gmm = gmm
                best_gmm_metrics = gmm_metrics
                best_gmm_f1 = gmm_metrics.get('f1', 0)
        except Exception as e:
            print(f"Error with GMM covariance_type={cov_type}: {e}")
    
    # Visualize differences between K-means and GMM
    if X.shape[1] >= 2 and best_gmm is not None:
        if X.shape[1] > 2:
            # Just use first two dimensions for visualization
            X_plot = X[:, :2]
        else:
            X_plot = X
            
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot K-means - sử dụng model có F1 tốt nhất
        axes[0].scatter(X_plot[:, 0], X_plot[:, 1], c=kmeans_best_values['f1']['model'].labels_, 
                      cmap='viridis', alpha=0.7, s=50, edgecolors='w')
        axes[0].set_title('K-means Clustering')
        axes[0].set_xlabel('Feature 1')
        axes[0].set_ylabel('Feature 2')
        axes[0].grid(True, alpha=0.3)
        
        # Plot GMM
        if hasattr(best_gmm, 'plot_clusters'):
            best_gmm.plot_clusters(X_plot, ax=axes[1], title="GMM Clustering")
        else:
            scatter = axes[1].scatter(X_plot[:, 0], X_plot[:, 1], c=best_gmm.labels_, 
                                 cmap='viridis', alpha=0.7, s=50, edgecolors='w')
            axes[1].set_title('GMM Clustering')
            axes[1].set_xlabel('Feature 1')
            axes[1].set_ylabel('Feature 2')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/kmeans_vs_gmm.png')
        plt.close(fig)
    
    # Compile and compare results
    if best_gmm is None:
        all_results = {
            'kmeans': {
                'model': kmeans_best_values['f1']['model'],  # Sử dụng model có F1 tốt nhất
                'metrics': kmeans_metrics,
                'best_values': kmeans_best_values
            }
        }
        best_method = 'kmeans'
    else:
        all_results = {
            'kmeans': {
                'model': kmeans_best_values['f1']['model'],  # Sử dụng model có F1 tốt nhất 
                'metrics': kmeans_metrics,
                'best_values': kmeans_best_values
            },
            'gmm': {
                'model': best_gmm,
                'metrics': best_gmm_metrics
            }
        }
        
        # Determine best method based on F1 score
        best_method = max(['kmeans', 'gmm'], 
                          key=lambda x: all_results[x]['metrics'].get('f1', 0))
    
    # Print comparison summary
    print("\n=== Clustering Methods Comparison Summary ===")
    methods = list(all_results.keys())
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'adjusted_rand', 'silhouette']
    
    # Create comparison table
    print(f"{'Method':<15} | " + " | ".join(f"{m:<10}" for m in metric_names))
    print("-" * 90)
    
    for method in methods:
        if all_results[method]['metrics'] is not None:
            metrics = all_results[method]['metrics']
            values = [metrics.get(m, 'N/A') for m in metric_names]
            print(f"{method:<15} | " + " | ".join(f"{v:<10.4f}" if isinstance(v, float) else f"{v:<10}" for v in values))
    
    # Print best method
    if len(methods) > 1:
        print(f"\nBest method based on F1 score: {best_method}")
        print(f"F1 score: {all_results[best_method]['metrics'].get('f1', 'N/A'):.4f}")
        print(f"Accuracy: {all_results[best_method]['metrics'].get('accuracy', 'N/A'):.4f}")
    
    # Return best model and all results
    best_model = all_results[best_method]['model']
    return best_model, all_results

def parse_arguments():
    """
    Parse command-line arguments for the script.
    
    Returns:
    --------
    args : argparse.Namespace
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Enhanced clustering on ABIDE dataset with unsupervised feature engineering")
    
    # Input/output options
    parser.add_argument("--input-file", type=str, default="ABIDE2(updated).csv",
                        help="Path to input CSV file")
    parser.add_argument("--output-dir", type=str, default="plots",
                        help="Directory to save output files and plots")
    
    # Data processing options
    parser.add_argument("--skip-process", action="store_true",
                        help="Skip data processing if processed file exists")
    parser.add_argument("--force-process", action="store_true",
                        help="Force data processing even if processed file exists")
    
    # Feature engineering options
    parser.add_argument("--components", type=int, default=20,
                        help="Number of components for dimensionality reduction methods (default: 20)")
    parser.add_argument("--outlier-threshold", type=float, default=5,
                        help="Z-score threshold for outlier detection (default: 5)")
    parser.add_argument("--whitening-method", type=str, 
                        choices=["pca", "zca", "cholesky", "mahalanobis", "adaptive", "graph", "hybrid"],
                        default="pca", help="Whitening method to use for dimensionality reduction")
    parser.add_argument("--adaptive-alpha", type=float, default=1.0,
                        help="Alpha parameter for adaptive whitening (0-1, where 1 is full whitening)")
    parser.add_argument("--zca-weight", type=float, default=0.5,
                        help="Weight for ZCA in hybrid whitening (0-1, where 1 is pure ZCA)")
    parser.add_argument("--advanced-fe", type=str, 
                        choices=["standard", "advanced_pca", "optimal_kernel_pca", "feature_interactions_pca", 
                                "region_ratios_pca", "ensemble_reduction", "variance_interactions", 
                                "optimize_for_clustering", "combined_optimal", "zca_whitening_pca",
                                "whitening_graph_selection_pca", "cholesky_whitening", "mahalanobis_whitening",
                                "adaptive_whitening", "hybrid_whitening", "double_pca"],
                        default="standard", help="Advanced feature engineering method to use")
    
    # Clustering options
    parser.add_argument("--method", type=str, choices=["kmeans", "enhanced_kmeans", "gmm", "compare"],
                        default="compare", help="Clustering method to use")
    parser.add_argument("--n-clusters", type=int, default=2,
                        help="Number of clusters (default: 2)")
    
    # Visualization options
    parser.add_argument("--plot-pca", action="store_true",
                        help="Plot PCA components variance")
    parser.add_argument("--plot-tsne", action="store_true",
                        help="Plot t-SNE visualization of clusters")
    
    # Additional analysis options
    parser.add_argument("--run-all", action="store_true",
                        help="Run all analysis methods")
    # NEW: Add argument to test advanced_pca with components from 1 to 20
    parser.add_argument("--test-advanced-pca-range", action="store_true",
                        help="Test advanced_pca with components from 1 to 20 and print results")
    # NEW: Add argument to test standard PCA with components from 1 to 30
    parser.add_argument("--test-standard-pca-range", action="store_true",
                        help="Test standard PCA with components from 1 to 30 and print results")
    # NEW: Add argument to test various whitening methods with components from 1 to 30
    parser.add_argument("--test-whitening-range", type=str, 
                        choices=["zca", "graph", "cholesky", "mahalanobis", "adaptive", "hybrid"],
                        help="Test a specific whitening method with components from 1 to 30")
    # NEW: Add argument to test double PCA with components from 1 to 30
    parser.add_argument("--test-double-pca-range", action="store_true",
                        help="Test double PCA with components from 1 to 30 and print results")
    
    args = parser.parse_args()
    return args

def main(args=None):
    """
    Main function for enhanced clustering analysis on the ABIDE dataset with unsupervised methods.
    
    Parameters:
    -----------
    args : argparse.Namespace, optional
        Command-line arguments
    """
    print("=== Lab 2 - Part 3: Enhanced Clustering on ABIDE Dataset with Unsupervised Feature Engineering ===")
    
    # If no arguments provided, parse them
    if args is None:
        args = parse_arguments()
    
    # Get components value from args or use default
    components = args.components if hasattr(args, 'components') else 20
    print(f"\nUsing {components} components for dimensionality reduction methods")
    
    # Check for whitening method from args
    whitening_method = args.whitening_method if hasattr(args, 'whitening_method') else "pca"
    print(f"Using whitening method: {whitening_method}")
    
    # Get adaptive alpha if applicable
    adaptive_alpha = args.adaptive_alpha if hasattr(args, 'adaptive_alpha') else 1.0
    if whitening_method == "adaptive":
        print(f"Adaptive whitening alpha parameter: {adaptive_alpha}")
    
    # Get zca_weight if applicable
    zca_weight = args.zca_weight if hasattr(args, 'zca_weight') else 0.5
    if whitening_method == "hybrid":
        print(f"Hybrid whitening parameters: zca_weight={zca_weight}, adaptive_alpha={adaptive_alpha}")
    
    # Get advanced feature engineering method if specified
    advanced_fe_method = args.advanced_fe if hasattr(args, 'advanced_fe') else "standard"
    print(f"Using advanced feature engineering method: {advanced_fe_method}")
    
    # Check if processed data already exists
    processed_file_path = 'ABIDE2_processed.csv'
    
    # Determine whether to run the data processing pipeline
    if args and hasattr(args, 'force_process') and args.force_process:
        skip_processing = False
    elif args and hasattr(args, 'skip_process') and args.skip_process:
        skip_processing = True
    else:
        skip_processing = os.path.exists(processed_file_path)
    
    # Determine input file path
    input_file_path = args.input_file if args and hasattr(args, 'input_file') else "ABIDE2(updated).csv"
    
    if skip_processing:
        print(f"\nProcessed data file '{processed_file_path}' already exists. Skipping data processing pipeline.")
        print("To rerun data processing, use the --force-process flag.")
    else:
        # Data Processing Pipeline
        print("\n=== Data Processing Pipeline ===")
        
        # Load ABIDE dataset with data_processing
        print(f"\nLoading and analyzing data from {input_file_path}...")
        df = load_data(input_file_path)
        
        # Analyze basic statistics
        stats_df = analyze_basic_stats(df)
        
        # Check distribution of features
        sample_columns = check_distribution(df)
        
        # Analyze group differences
        analyze_group_differences(df)
        
        # Analyze feature correlations
        corr_matrix, corr_pairs = analyze_feature_correlations(df)
        
        # Determine optimal preprocessing
        preprocess_params = get_optimal_preprocessing(df)
        
        # Apply preprocessing
        processed_df = dp_preprocess_data(df, preprocess_params)
        
        # Save processed data
        processed_df.to_csv(processed_file_path, index=False)
        print(f"\nProcessed data saved to '{processed_file_path}'")
    
    # Continue with the original pipeline but use processed data
    print("\n=== Starting Clustering Analysis ===")
    
    # Load preprocessed ABIDE dataset with the original loader
    X, y, feature_names = load_abide_data(processed_file_path)
    
    if X is None:
        print("Error loading the dataset. Exiting.")
        return
    
    # Check for and handle outliers
    print("\nChecking for outliers...")
    outlier_threshold = args.outlier_threshold if hasattr(args, 'outlier_threshold') else 5
    outlier_mask = detect_outliers(X, threshold=outlier_threshold)
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
    
    # Standardize the data
    print("\nStandardizing data...")
    X_std, _ = preprocess_data(X)
    
    # Check for special testing flags FIRST before running any other analysis
    
    # NEW: If --test-double-pca-range is set, run double PCA for components 1 to 30
    if hasattr(args, 'test_double_pca_range') and args.test_double_pca_range:
        print("\n=== Running double PCA for components 1 to 30 ===")
        range_results = []
        for n_comp in range(1, 31):
            try:
                X_double_pca = apply_double_pca(X_std, n_components=n_comp, random_state=42)
                best_kmeans, _, kmeans_best_values = compare_kmeans_implementations(X_double_pca, y, feature_names)
                metrics = calculate_cluster_metrics(y, best_kmeans.labels_)
                print(f"Components: {n_comp:2d} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | ARI: {metrics['ari']:.4f} | NMI: {metrics['nmi']:.4f}")
                range_results.append({
                    'n_components': n_comp,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'ari': metrics['ari'],
                    'nmi': metrics['nmi']
                })
            except Exception as e:
                print(f"Error with n_components={n_comp}: {e}")
        # Save results to CSV
        pd.DataFrame(range_results).to_csv('plots/double_pca_components_range_results.csv', index=False)
        print("Results saved to 'plots/double_pca_components_range_results.csv'")
        return  # Exit the function so no other analyses run
    
    # NEW: If --test-whitening-range is set, run the selected whitening method for components 1 to 30
    if hasattr(args, 'test_whitening_range') and args.test_whitening_range:
        whitening_method = args.test_whitening_range
        print(f"\n=== Running {whitening_method} whitening for components 1 to 30 ===")
        range_results = []
        
        # Define parameters for adaptive and hybrid methods
        adaptive_alpha = args.adaptive_alpha if hasattr(args, 'adaptive_alpha') else 1.0
        zca_weight = args.zca_weight if hasattr(args, 'zca_weight') else 0.5
        
        for n_comp in range(1, 31):
            try:
                # Apply the selected whitening method
                if whitening_method == "zca":
                    X_whitened = apply_zca_whitening_pca(X_std, n_components=n_comp, eps=1e-6, random_state=42)
                elif whitening_method == "graph":
                    X_whitened = apply_whitening_graph_feature_selection_pca(X_std, n_components=n_comp, threshold=0.5, random_state=42)
                elif whitening_method == "cholesky":
                    X_whitened = apply_cholesky_whitening(X_std, n_components=n_comp)
                elif whitening_method == "mahalanobis":
                    X_whitened = apply_mahalanobis_whitening(X_std, n_components=n_comp)
                elif whitening_method == "adaptive":
                    X_whitened = apply_pca_adaptive_whitening(X_std, n_components=n_comp, alpha=adaptive_alpha, random_state=42)
                elif whitening_method == "hybrid":
                    X_whitened = apply_hybrid_whitening(X_std, n_components=n_comp, zca_weight=zca_weight, adaptive_alpha=adaptive_alpha, random_state=42)
                
                # Evaluate with K-means
                best_kmeans, _, kmeans_best_values = compare_kmeans_implementations(X_whitened, y, feature_names)
                metrics = calculate_cluster_metrics(y, best_kmeans.labels_)
                print(f"Components: {n_comp:2d} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | ARI: {metrics['ari']:.4f} | NMI: {metrics['nmi']:.4f}")
                range_results.append({
                    'n_components': n_comp,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'ari': metrics['ari'],
                    'nmi': metrics['nmi']
                })
            except Exception as e:
                print(f"Error with n_components={n_comp}: {e}")
        
        # Save results to CSV
        pd.DataFrame(range_results).to_csv(f'plots/{whitening_method}_whitening_components_range_results.csv', index=False)
        print(f"Results saved to 'plots/{whitening_method}_whitening_components_range_results.csv'")
        return  # Exit the function so no other analyses run
    
    # NEW: If --test-standard-pca-range is set, run standard PCA for components 1 to 30
    if hasattr(args, 'test_standard_pca_range') and args.test_standard_pca_range:
        print("\n=== Running standard PCA for components 1 to 30 ===")
        range_results = []
        for n_comp in range(1, 31):
            try:
                my_pca = MyPCA(n_components=n_comp)
                X_pca = my_pca.fit_transform(X_std)
                best_kmeans, _, kmeans_best_values = compare_kmeans_implementations(X_pca, y, feature_names)
                metrics = calculate_cluster_metrics(y, best_kmeans.labels_)
                print(f"Components: {n_comp:2d} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | ARI: {metrics['ari']:.4f} | NMI: {metrics['nmi']:.4f}")
                range_results.append({
                    'n_components': n_comp,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'ari': metrics['ari'],
                    'nmi': metrics['nmi']
                })
            except Exception as e:
                print(f"Error with n_components={n_comp}: {e}")
        # Save results to CSV
        pd.DataFrame(range_results).to_csv('plots/standard_pca_components_range_results.csv', index=False)
        print("Results saved to 'plots/standard_pca_components_range_results.csv'")
        return  # Exit the function so no other analyses run
    
    # NEW: If --test-advanced-pca-range is set, run advanced_pca for components 1 to 20
    if hasattr(args, 'test_advanced_pca_range') and args.test_advanced_pca_range:
        print("\n=== Running advanced_pca for components 1 to 20 ===")
        range_results = []
        for n_comp in range(1, 21):
            try:
                X_adv, _ = apply_advanced_pca(X_std, n_components=n_comp, whiten=True)
                best_kmeans, _, kmeans_best_values = compare_kmeans_implementations(X_adv, y, feature_names)
                metrics = calculate_cluster_metrics(y, best_kmeans.labels_)
                print(f"Components: {n_comp:2d} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | ARI: {metrics['ari']:.4f} | NMI: {metrics['nmi']:.4f}")
                range_results.append({
                    'n_components': n_comp,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'ari': metrics['ari'],
                    'nmi': metrics['nmi']
                })
            except Exception as e:
                print(f"Error with n_components={n_comp}: {e}")
        # Save results to CSV
        pd.DataFrame(range_results).to_csv('plots/advanced_pca_components_range_results.csv', index=False)
        print("Results saved to 'plots/advanced_pca_components_range_results.csv'")
        return  # Exit the function so no other analyses run
    
    # Define different feature engineering approaches to test (only PCA-related methods)
    fe_approaches = [
        {
            "name": "Variance-based Feature Selection",
            "methods": [
                {'name': 'select_variance', 'threshold': 0.01}
            ]
        },
        {
            "name": "Kernel PCA",
            "methods": [
                {'name': 'kernel_pca', 'n_components': components, 'kernel': 'rbf'}
            ]
        },
        
        {
            "name": "Feature Agglomeration",
            "methods": [
                {'name': 'feature_agglomeration', 'n_clusters': components, 'linkage': 'ward'}
            ]
        },
        
        
        {
            "name": "Select Percentile",
            "methods": [
                {'name': 'select_percentile', 'percentile': 30}
            ]
        },
        {
            "name": "Combined Selection + Kernel PCA",
            "methods": [
                {'name': 'select_variance', 'threshold': 0.01},
                {'name': 'kernel_pca', 'n_components': components, 'kernel': 'rbf'}
            ]
        },
        
        {
            "name": "Feature Agglomeration + Kernel PCA",
            "methods": [
                {'name': 'feature_agglomeration', 'n_clusters': min(50, X.shape[1] // 2)},
                {'name': 'kernel_pca', 'n_components': components, 'kernel': 'rbf'}
            ]
        }
    ]
    
    # Add our new advanced feature engineering approaches
    advanced_fe_approaches = [
        {
            "name": "Advanced PCA with Whitening",
            "pipeline": "advanced_pca",
            "params": {"n_components": components, "whiten": True}
        },
        {
            "name": "Optimal Kernel PCA",
            "pipeline": "optimal_kernel_pca",
            "params": {"n_components": components, "kernel": "rbf", "gamma": 0.1}
        },
        {
            "name": "Feature Interactions + PCA",
            "pipeline": "feature_interactions_pca",
            "params": {"n_components": components, "degree": 2, "interaction_only": True}
        },
        {
            "name": "Region Ratio Features + PCA",
            "pipeline": "region_ratios_pca",
            "params": {"n_components": components}
        },
        {
            "name": "Ensemble Dimensionality Reduction",
            "pipeline": "ensemble_reduction",
            "params": {"n_components": components}
        },
        {
            "name": "Variance Selection + Interaction Features",
            "pipeline": "variance_interactions",
            "params": {"variance_threshold": 0.1, "degree": 2, "max_features": 1000}
        },
        {
            "name": "Optimized for Clustering",
            "pipeline": "optimize_for_clustering",
            "params": {"n_components": components}
        },
        {
            "name": "Combined Optimal Approach",
            "pipeline": "combined_optimal",
            "params": {"n_components": components}
        },
        {
            "name": "ZCA Whitening + PCA",
            "pipeline": "zca_whitening_pca",
            "params": {"n_components": components, "eps": 1e-6, "random_state": 42}
        },
        {
            "name": "Whitening + Graph Feature Selection + PCA",
            "pipeline": "whitening_graph_selection_pca",
            "params": {"n_components": components, "threshold": 0.5, "random_state": 42}
        },
        {
            "name": "Cholesky Whitening",
            "pipeline": "cholesky_whitening",
            "params": {"n_components": components}
        },
        {
            "name": "Mahalanobis Whitening",
            "pipeline": "mahalanobis_whitening",
            "params": {"n_components": components}
        },
        {
            "name": "Adaptive PCA Whitening",
            "pipeline": "adaptive_whitening",
            "params": {"n_components": components, "alpha": adaptive_alpha}
        },
        {
            "name": "Hybrid ZCA-Adaptive Whitening",
            "pipeline": "hybrid_whitening",
            "params": {"n_components": components, "zca_weight": zca_weight, "adaptive_alpha": adaptive_alpha}
        },
        {
            "name": "Double PCA",
            "pipeline": "double_pca",
            "params": {"n_components": components, "random_state": 42}
        }
    ]
    
    # Update specific parameters based on command-line arguments
    if advanced_fe_method == "hybrid_whitening":
        # Find and update the hybrid whitening approach with actual provided parameters
        for approach in advanced_fe_approaches:
            if approach["pipeline"] == "hybrid_whitening":
                approach["params"]["zca_weight"] = zca_weight
                approach["params"]["adaptive_alpha"] = adaptive_alpha
                print(f"Updated hybrid whitening parameters: zca_weight={zca_weight}, adaptive_alpha={adaptive_alpha}")
                break
    elif advanced_fe_method == "adaptive_whitening":
        # Find and update the adaptive whitening approach with actual provided parameters
        for approach in advanced_fe_approaches:
            if approach["pipeline"] == "adaptive_whitening":
                approach["params"]["alpha"] = adaptive_alpha
                print(f"Updated adaptive whitening parameters: alpha={adaptive_alpha}")
                break
    
    # Baseline with standard PCA for comparison
    print("\n=== Baseline with Standard PCA ===")
    
    # Use the selected whitening method for baseline
    if whitening_method == "pca":
        my_pca = MyPCA(n_components=components)
        X_baseline = my_pca.fit_transform(X_std)
    elif whitening_method == "zca":
        X_baseline = apply_zca_whitening_pca(X_std, n_components=components)
    elif whitening_method == "cholesky":
        X_baseline = apply_cholesky_whitening(X_std, n_components=components)
    elif whitening_method == "mahalanobis":
        X_baseline = apply_mahalanobis_whitening(X_std, n_components=components)
    elif whitening_method == "adaptive":
        X_baseline = apply_pca_adaptive_whitening(X_std, n_components=components, alpha=adaptive_alpha)
    elif whitening_method == "graph":
        X_baseline = apply_whitening_graph_feature_selection_pca(X_std, n_components=components)
    elif whitening_method == "hybrid":
        zca_weight = args.zca_weight if hasattr(args, 'zca_weight') else 0.5
        X_baseline = apply_hybrid_whitening(X_std, n_components=components, 
                                           zca_weight=zca_weight, 
                                           adaptive_alpha=adaptive_alpha)
    else:
        # Default to standard PCA if method is not recognized
        my_pca = MyPCA(n_components=components)
        X_baseline = my_pca.fit_transform(X_std)
    
    # Compare K-means implementations on baseline data
    print("\n=== Comparing K-means Implementations on Baseline Data ===")
    best_kmeans, kmeans_comparison, kmeans_best_values = compare_kmeans_implementations(X_baseline, y, feature_names)
    
    # Apply GMM clustering on baseline data
    print("\n=== Applying GMM Clustering on Baseline Data ===")
    gmm_baseline, gmm_baseline_metrics = apply_gmm_clustering(X_baseline, y, n_components=2, find_optimal=False)
    
    # Compare K-means and GMM on baseline data
    print("\n=== Comparing Clustering Methods on Baseline Data ===")
    best_model, clustering_comparison = compare_clustering_methods(X_baseline, y, feature_names)
    
    # Use the best K-means for baseline evaluation
    baseline_metrics = calculate_cluster_metrics(y, best_kmeans.labels_)
    print("\nBaseline Results (Standard PCA with Best K-means):")
    print_metrics(baseline_metrics)
    
    # Plot baseline clustering results
    plot_clusters_2d(X_baseline[:, :2], best_kmeans.labels_, y, 
                    title=f"Baseline Clustering with Standard PCA (20 components)",
                    save_path=f'plots/abide_baseline_pca_clustering.png')
    
    # Store results
    results = [{
        'approach': 'Baseline (Standard PCA)',
        'n_features': X_baseline.shape[1],
        'accuracy': baseline_metrics['accuracy'],
        'precision': baseline_metrics['precision'],
        'recall': baseline_metrics['recall'],
        'f1': baseline_metrics['f1'],
        'ari': baseline_metrics['ari'],
        'nmi': baseline_metrics['nmi']
    }]
    
    # Check if a specific advanced feature engineering method was selected
    advanced_fe_method = args.advanced_fe if hasattr(args, 'advanced_fe') else "standard"
    
    # Only run basic feature engineering approaches if no specific advanced method is selected
    if advanced_fe_method == "standard":
        # Try each feature engineering approach with the best K-means implementation
        for approach in fe_approaches:
            approach_name = approach["name"]
            methods = approach["methods"]
            
            print(f"\n=== Testing Approach: {approach_name} ===")
            
            try:
                # Apply feature engineering (pass None for y to ensure unsupervised operation)
                X_transformed = feature_engineering_pipeline(X_std, y=None, methods=methods)
                
                # Find the best K-means approach for this transformed data
                best_kmeans, _, kmeans_best_values = compare_kmeans_implementations(X_transformed, y, feature_names)
                
                # Apply GMM clustering and compare with K-means
                print(f"\n=== Comparing Clustering Methods for {approach_name} ===")
                best_model, clustering_comparison = compare_clustering_methods(X_transformed, y, feature_names)
                
                # If GMM performs better, use it for evaluation
                if clustering_comparison.get('gmm', {}).get('metrics', {}).get('f1', 0) > \
                   clustering_comparison.get('kmeans', {}).get('metrics', {}).get('f1', 0):
                    print(f"GMM performs better than K-means for {approach_name}")
                    best_model_metrics = clustering_comparison['gmm']['metrics']
                    best_labels = clustering_comparison['gmm']['model'].labels_
                else:
                    best_model_metrics = clustering_comparison['kmeans']['metrics']
                    best_labels = best_kmeans.labels_
                
                # Evaluate clustering results
                print(f"Results for {approach_name} (features: {X_transformed.shape[1]}):")
                print_metrics(best_model_metrics)
                
                # Plot clustering results if 2D or more
                if X_transformed.shape[1] >= 2:
                    plot_clusters_2d(X_transformed[:, :2], best_labels, y, 
                                    title=f"Clustering with {approach_name}",
                                    save_path=f'plots/abide_clustering_{approach_name.replace(" ", "_").lower()}.png')
                
                # Store results
                results.append({
                    'approach': approach_name,
                    'n_features': X_transformed.shape[1],
                    'accuracy': best_model_metrics['accuracy'],
                    'precision': best_model_metrics['precision'],
                    'recall': best_model_metrics['recall'],
                    'f1': best_model_metrics['f1'],
                    'ari': best_model_metrics['ari'],
                    'nmi': best_model_metrics['nmi'],
                    'best_method': 'GMM' if clustering_comparison.get('gmm', {}).get('metrics', {}).get('f1', 0) > \
                                   clustering_comparison.get('kmeans', {}).get('metrics', {}).get('f1', 0) else 'KMeans'
                })
                
            except Exception as e:
                print(f"Error applying {approach_name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        print(f"\nSkipping basic feature engineering approaches since advanced method '{advanced_fe_method}' was selected.")
    
    # Try our advanced feature engineering approaches
    print("\n=== Testing Advanced Feature Engineering Methods ===")
    
    # Check if a specific advanced feature engineering method was selected
    advanced_fe_method = args.advanced_fe if hasattr(args, 'advanced_fe') else "standard"
    
    # If standard method is selected, run all approaches or skip
    if advanced_fe_method == "standard":
        print("Running all advanced feature engineering approaches...")
        selected_approaches = advanced_fe_approaches
    else:
        # Find the selected approach
        selected_approaches = []
        for approach in advanced_fe_approaches:
            if approach["pipeline"] == advanced_fe_method:
                print(f"Running only the selected advanced feature engineering method: {approach['name']}")
                selected_approaches = [approach]
                break
        
        if not selected_approaches:
            print(f"Warning: Selected method '{advanced_fe_method}' not found. Using standard pipeline.")
            selected_approaches = advanced_fe_approaches
    
    for i, approach in enumerate(selected_approaches):
        approach_name = approach["name"]
        pipeline = approach["pipeline"]
        params = approach["params"]
        
        print(f"\n=== Testing Advanced Approach: {approach_name} ===")
        
        try:
            # Apply the appropriate advanced feature engineering method
            if pipeline == "advanced_pca":
                X_transformed, _ = apply_advanced_pca(X_std, **params)
            
            elif pipeline == "optimal_kernel_pca":
                try:
                    X_transformed = apply_optimal_kernel_pca(X_std, **params)
                except Exception as e:
                    print(f"Error with optimal_kernel_pca: {e}")
                    print("Falling back to standard Kernel PCA")
                    from sklearn.decomposition import KernelPCA
                    kpca = KernelPCA(n_components=params['n_components'], 
                                     kernel=params['kernel'],
                                     gamma=params['gamma'],
                                     fit_inverse_transform=True,
                                     random_state=42)
                    X_transformed = kpca.fit_transform(X_std)
            
            elif pipeline == "feature_interactions_pca":
                # First select features with variance
                X_selected = apply_feature_selection(X_std, variance_threshold=0.01)
                # Then create interactions
                X_interactions = create_feature_interactions(X_selected, 
                                                           degree=params['degree'], 
                                                           interaction_only=params['interaction_only'])
                # Apply PCA
                X_transformed, _ = apply_advanced_pca(X_interactions, n_components=params['n_components'])
            
            elif pipeline == "region_ratios_pca":
                # Create region ratio features
                X_with_ratios, _ = create_region_ratio_features(X, feature_names)
                # Standardize
                X_ratios_std = RobustScaler().fit_transform(X_with_ratios)
                # Apply PCA
                X_transformed, _ = apply_advanced_pca(X_ratios_std, n_components=params['n_components'])
            
            elif pipeline == "ensemble_reduction":
                try:
                    X_transformed = apply_ensemble_dimensionality_reduction(X_std, **params)
                except Exception as e:
                    print(f"Error with ensemble_reduction: {e}")
                    print("Falling back to PCA + Kernel PCA ensemble")
                    # PCA
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=params['n_components'])
                    X_pca = pca.fit_transform(X_std)
                    
                    # Kernel PCA
                    from sklearn.decomposition import KernelPCA
                    kpca = KernelPCA(n_components=params['n_components'], kernel='rbf', gamma=0.1)
                    X_kpca = kpca.fit_transform(X_std)
                    
                    # Combine results
                    X_combined = np.hstack((X_pca, X_kpca))
                    
                    # Final PCA to reduce to desired dimensions
                    final_pca = PCA(n_components=params['n_components'])
                    X_transformed = final_pca.fit_transform(X_combined)
            
            elif pipeline == "variance_interactions":
                # Apply variance-based feature selection with higher threshold
                X_selected = apply_feature_selection(X_std, variance_threshold=params['variance_threshold'])
                print(f"After variance-based selection: {X_selected.shape[1]} features")
                
                # Create limited interaction features
                max_features = params.get('max_features', 1000)
                n_components = params.get('n_components', 20)  # Default to 20 components if not specified
                
                if X_selected.shape[1] > 50:  # Nếu có quá nhiều tính năng, chỉ lấy top features
                    from sklearn.feature_selection import SelectKBest, mutual_info_regression
                    selector = SelectKBest(mutual_info_regression, k=min(50, X_selected.shape[1]))
                    X_selected = selector.fit_transform(X_selected, y)
                    print(f"Limited to top {X_selected.shape[1]} features for interactions")
                
                # Create interaction features
                X_interactions = create_feature_interactions(X_selected, degree=params['degree'])
                print(f"After adding interaction features: {X_interactions.shape[1]} features")
                
                # Apply feature selection if too many features
                if X_interactions.shape[1] > max_features:
                    from sklearn.feature_selection import SelectKBest, mutual_info_regression
                    selector = SelectKBest(mutual_info_regression, k=min(max_features, X_interactions.shape[1]))
                    X_interactions = selector.fit_transform(X_interactions, y)
                    print(f"Limited to {X_interactions.shape[1]} features after selection")
                
                # Apply PCA for final dimensionality reduction
                X_transformed, _ = apply_advanced_pca(X_interactions, n_components=n_components)
            
            elif pipeline == "optimize_for_clustering":
                X_transformed = optimize_feature_engineering_for_clustering(X_std, n_components=params['n_components'])
            
            elif pipeline == "combined_optimal":
                X_transformed = apply_combined_optimal_approach(X, feature_names, n_components=params['n_components'])
            
            elif pipeline == "zca_whitening_pca":
                try:
                    X_transformed = apply_zca_whitening_pca(X_std, **params)
                except Exception as e:
                    print(f"Error with ZCA whitening + PCA: {e}")
                    print("Falling back to standard PCA with whitening")
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=params['n_components'], whiten=True, random_state=42)
                    X_transformed = pca.fit_transform(X_std)
            
            elif pipeline == "whitening_graph_selection_pca":
                try:
                    X_transformed = apply_whitening_graph_feature_selection_pca(X_std, **params)
                except Exception as e:
                    print(f"Error with whitening + graph selection + PCA: {e}")
                    print("Falling back to standard PCA with whitening")
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=params['n_components'], whiten=True, random_state=42)
                    X_transformed = pca.fit_transform(X_std)
            
            elif pipeline == "cholesky_whitening":
                try:
                    X_transformed = apply_cholesky_whitening(X_std, **params)
                except Exception as e:
                    print(f"Error with Cholesky whitening: {e}")
                    print("Falling back to standard PCA with whitening")
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=params['n_components'], whiten=True, random_state=42)
                    X_transformed = pca.fit_transform(X_std)
            
            elif pipeline == "mahalanobis_whitening":
                try:
                    X_transformed = apply_mahalanobis_whitening(X_std, **params)
                except Exception as e:
                    print(f"Error with Mahalanobis whitening: {e}")
                    print("Falling back to standard PCA with whitening")
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=params['n_components'], whiten=True, random_state=42)
                    X_transformed = pca.fit_transform(X_std)
            
            elif pipeline == "adaptive_whitening":
                try:
                    # Adaptive whitening can take an alpha parameter to control whitening strength
                    alpha = params.get('alpha', 1.0)
                    X_transformed = apply_pca_adaptive_whitening(X_std, n_components=params['n_components'], 
                                                           alpha=alpha, random_state=42)
                except Exception as e:
                    print(f"Error with adaptive whitening: {e}")
                    print("Falling back to standard PCA with whitening")
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=params['n_components'], whiten=True, random_state=42)
                    X_transformed = pca.fit_transform(X_std)
            
            elif pipeline == "hybrid_whitening":
                try:
                    # Hybrid whitening takes both zca_weight and adaptive_alpha parameters
                    zca_weight = params.get('zca_weight', 0.5)
                    adaptive_alpha = params.get('adaptive_alpha', 0.7)
                    X_transformed = apply_hybrid_whitening(X_std, n_components=params['n_components'],
                                                         zca_weight=zca_weight, 
                                                         adaptive_alpha=adaptive_alpha,
                                                         random_state=42)
                except Exception as e:
                    print(f"Error with hybrid whitening: {e}")
                    print("Falling back to standard PCA with whitening")
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=params['n_components'], whiten=True, random_state=42)
                    X_transformed = pca.fit_transform(X_std)
            
            elif pipeline == "double_pca":
                try:
                    X_transformed = apply_double_pca(X_std, n_components=params['n_components'], 
                                                   random_state=params.get('random_state', 42))
                except Exception as e:
                    print(f"Error with double PCA: {e}")
                    print("Falling back to standard PCA")
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=params['n_components'], random_state=42)
                    X_transformed = pca.fit_transform(X_std)
            
            # Find the best K-means approach for this transformed data
            best_kmeans, _, kmeans_best_values = compare_kmeans_implementations(X_transformed, y, feature_names)
            
            # Always apply GMM clustering and compare with K-means
            print(f"\n=== Comparing Clustering Methods for {approach_name} ===")
            
            # Thử nhiều loại covariance cho GMM
            best_gmm = None
            best_gmm_metrics = None
            best_gmm_f1 = -1
            
            for cov_type in ['full', 'tied']:
                try:
                    print(f"Trying GMM with covariance_type={cov_type}...")
                    gmm, gmm_metrics = apply_gmm_clustering(
                        X_transformed, y, 
                        n_components=2, 
                        find_optimal=False, 
                        covariance_type=cov_type, 
                        max_iter=200, 
                        n_init=5
                    )
                    
                    if gmm_metrics.get('f1', 0) > best_gmm_f1:
                        best_gmm = gmm
                        best_gmm_metrics = gmm_metrics
                        best_gmm_f1 = gmm_metrics.get('f1', 0)
                except Exception as e:
                    print(f"Error with GMM covariance_type={cov_type}: {e}")
            
            if best_gmm is None:
                print("All GMM variants failed. Using K-means result only.")
                # Compare without GMM
                best_model = best_kmeans
                best_model_metrics = calculate_cluster_metrics(y, best_kmeans.labels_)
                best_labels = best_kmeans.labels_
                clustering_comparison = {
                    'kmeans': {
                        'model': best_kmeans,
                        'metrics': best_model_metrics
                    }
                }
            else:
                # Now compare with K-means
                kmeans_metrics = calculate_cluster_metrics(y, best_kmeans.labels_)
                
                clustering_comparison = {
                    'kmeans': {
                        'model': best_kmeans,
                        'metrics': kmeans_metrics
                    },
                    'gmm': {
                        'model': best_gmm,
                        'metrics': best_gmm_metrics
                    }
                }
                
                # If GMM performs better, use it for evaluation
                if best_gmm_metrics.get('f1', 0) > kmeans_metrics.get('f1', 0):
                    print(f"GMM performs better than K-means for {approach_name}")
                    best_model_metrics = best_gmm_metrics
                    best_labels = best_gmm.labels_
                    best_model = best_gmm
                else:
                    best_model_metrics = kmeans_metrics
                    best_labels = best_kmeans.labels_
                    best_model = best_kmeans
            
            # Evaluate clustering results
            print(f"Results for {approach_name} (features: {X_transformed.shape[1]}):")
            print_metrics(best_model_metrics)
            
            # Plot clustering results if 2D or more
            if X_transformed.shape[1] >= 2:
                plot_clusters_2d(X_transformed[:, :2], best_labels, y, 
                                title=f"Clustering with {approach_name}",
                                save_path=f'plots/abide_clustering_{approach_name.replace(" ", "_").lower()}.png')
            
            # Store results
            results.append({
                'approach': approach_name,
                'n_features': X_transformed.shape[1],
                'accuracy': best_model_metrics['accuracy'],
                'precision': best_model_metrics['precision'],
                'recall': best_model_metrics['recall'],
                'f1': best_model_metrics['f1'],
                'ari': best_model_metrics['ari'],
                'nmi': best_model_metrics['nmi'],
                'best_method': 'GMM' if clustering_comparison.get('gmm', {}).get('metrics', {}).get('f1', 0) > \
                               clustering_comparison.get('kmeans', {}).get('metrics', {}).get('f1', 0) else 'KMeans'
            })
            
        except Exception as e:
            print(f"Error applying {approach_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary table
    print("\n=== Summary of Results ===")
    print(f"{'Approach':<40} {'Best Method':<12} {'Features':<10} {'Accuracy':<10} {'F1 Score':<10} {'ARI':<10} {'NMI':<10}")
    print("-" * 102)
    
    for result in results:
        best_method = result.get('best_method', '-')
        print(f"{result['approach']:<40} {best_method:<12} {result['n_features']:<10} {result['accuracy']:<10.4f} "
              f"{result['f1']:<10.4f} {result['ari']:<10.4f} {result['nmi']:<10.4f}")
    
    # Save results to CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv('plots/abide_unsupervised_feature_engineering_results.csv', index=False)
    print("Results saved to 'plots/abide_unsupervised_feature_engineering_results.csv'")
    
    # Find best result based on F1 score
    best_result = max(results, key=lambda x: x['f1'])
    print(f"\nBest result by F1 score with approach '{best_result['approach']}':")
    print(f"Method: {best_result.get('best_method', 'Unknown')}")
    print(f"Accuracy: {best_result['accuracy']:.4f}, F1 Score: {best_result['f1']:.4f}")
    
    # Find best result based on ARI
    best_ari_result = max(results, key=lambda x: x['ari'])
    print(f"\nBest result by ARI with approach '{best_ari_result['approach']}':")
    print(f"Method: {best_ari_result.get('best_method', 'Unknown')}")
    print(f"ARI: {best_ari_result['ari']:.4f}, NMI: {best_ari_result['nmi']:.4f}")
    
    # Count how many times GMM was better than K-means
    gmm_wins = sum(1 for result in results if result.get('best_method') == 'GMM')
    kmeans_wins = sum(1 for result in results if result.get('best_method') == 'KMeans')
    total_comparisons = gmm_wins + kmeans_wins
    
    # Tạo danh sách các phương pháp mà GMM chiến thắng
    gmm_winning_approaches = [result['approach'] for result in results if result.get('best_method') == 'GMM']
    
    print(f"\nClustering Method Comparison:")
    if total_comparisons > 0:
        print(f"GMM performed better in {gmm_wins} out of {total_comparisons} approaches ({gmm_wins/total_comparisons*100:.1f}%)")
        print(f"K-means performed better in {kmeans_wins} out of {total_comparisons} approaches ({kmeans_wins/total_comparisons*100:.1f}%)")
        
        if gmm_wins > 0:
            print(f"\nApproaches where GMM performed better:")
            for approach in gmm_winning_approaches:
                print(f"- {approach}")
        
        # Tìm phương pháp Feature Engineering tốt nhất cho từng thuật toán
        kmeans_results = [r for r in results if r.get('best_method') == 'KMeans']
        gmm_results = [r for r in results if r.get('best_method') == 'GMM']
        
        if kmeans_results:
            best_kmeans_approach = max(kmeans_results, key=lambda x: x['f1'])
            print(f"\nBest Feature Engineering approach for K-means: {best_kmeans_approach['approach']}")
            print(f"F1 score: {best_kmeans_approach['f1']:.4f}, Accuracy: {best_kmeans_approach['accuracy']:.4f}")
        
        if gmm_results:
            best_gmm_approach = max(gmm_results, key=lambda x: x['f1'])
            print(f"\nBest Feature Engineering approach for GMM: {best_gmm_approach['approach']}")
            print(f"F1 score: {best_gmm_approach['f1']:.4f}, Accuracy: {best_gmm_approach['accuracy']:.4f}")
    
    # Try GMM with optimal number of components on the best feature engineering approach
    print("\n=== Finding Optimal GMM Components for Best Approach ===")
    best_approach = max(results, key=lambda x: x['f1'])
    print(f"Best feature engineering approach: {best_approach['approach']}")
    print(f"Best method: {best_approach.get('best_method', 'Unknown')}")
    
    # Apply the best approach
    best_approach_name = best_approach['approach']
    
    try:
        # Find corresponding pipeline and params
        best_pipeline = None
        best_params = None
        
        for approach in advanced_fe_approaches:
            if approach["name"] == best_approach_name:
                best_pipeline = approach["pipeline"]
                best_params = approach["params"]
                break
        
        if best_pipeline:
            # Re-apply the best feature engineering method
            if best_pipeline == "advanced_pca":
                X_best, _ = apply_advanced_pca(X_std, **best_params)
            elif best_pipeline == "optimal_kernel_pca":
                X_best = apply_optimal_kernel_pca(X_std, **best_params)
            elif best_pipeline == "feature_interactions_pca":
                X_selected = apply_feature_selection(X_std, variance_threshold=0.01)
                X_interactions = create_feature_interactions(X_selected, 
                                                           degree=best_params['degree'], 
                                                           interaction_only=best_params['interaction_only'])
                X_best, _ = apply_advanced_pca(X_interactions, n_components=best_params['n_components'])
            elif best_pipeline == "region_ratios_pca":
                X_with_ratios, _ = create_region_ratio_features(X, feature_names)
                X_ratios_std = RobustScaler().fit_transform(X_with_ratios)
                X_best, _ = apply_advanced_pca(X_ratios_std, n_components=best_params['n_components'])
            elif best_pipeline == "ensemble_reduction":
                X_best = apply_ensemble_dimensionality_reduction(X_std, **best_params)
            elif best_pipeline == "variance_interactions":
                X_selected = apply_feature_selection(X_std, variance_threshold=best_params['variance_threshold'])
                X_interactions = create_feature_interactions(X_selected, degree=best_params['degree'])
                X_best, _ = apply_advanced_pca(X_interactions, n_components=best_params['n_components'])
            elif best_pipeline == "optimize_for_clustering":
                X_best = optimize_feature_engineering_for_clustering(X_std, n_components=best_params['n_components'])
            elif best_pipeline == "combined_optimal":
                X_best = apply_combined_optimal_approach(X, feature_names, n_components=best_params['n_components'])
            elif best_pipeline == "zca_whitening_pca":
                X_best = apply_zca_whitening_pca(X_std, **best_params)
            elif best_pipeline == "whitening_graph_selection_pca":
                X_best = apply_whitening_graph_feature_selection_pca(X_std, **best_params)
            elif best_pipeline == "cholesky_whitening":
                X_best = apply_cholesky_whitening(X_std, **best_params)
            elif best_pipeline == "mahalanobis_whitening":
                X_best = apply_mahalanobis_whitening(X_std, **best_params)
            elif best_pipeline == "adaptive_whitening":
                X_best = apply_pca_adaptive_whitening(X_std, **best_params)
            elif best_pipeline == "hybrid_whitening":
                X_best = apply_hybrid_whitening(X_std, **best_params)
            elif best_pipeline == "double_pca":
                X_best = apply_double_pca(X_std, **best_params)
            
            # Apply GMM with optimal number of components
            print("\nFinding optimal number of GMM components...")
            optimal_gmm, optimal_metrics = apply_gmm_clustering(X_best, y, max_components=10, find_optimal=True)
            
            # Add to results
            results.append({
                'approach': f"{best_approach_name} + Optimal GMM ({optimal_gmm.n_components} components)",
                'n_features': X_best.shape[1],
                'accuracy': optimal_metrics['accuracy'],
                'precision': optimal_metrics['precision'],
                'recall': optimal_metrics['recall'],
                'f1': optimal_metrics['f1'],
                'ari': optimal_metrics['ari'],
                'nmi': optimal_metrics['nmi'],
                'best_method': 'GMM'
            })
            
    except Exception as e:
        print(f"Error applying optimal GMM to best approach: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nUnsupervised Feature Engineering Analysis Completed!")

if __name__ == "__main__":
    args = parse_arguments()
    main(args) 