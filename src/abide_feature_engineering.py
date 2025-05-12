#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced feature engineering module for ABIDE dataset.
This module provides specialized feature engineering techniques to improve class separation,
particularly for clustering applications.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.cluster import FeatureAgglomeration
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import TruncatedSVD, FastICA
from sklearn.pipeline import Pipeline
from scipy import linalg
import networkx as nx
from sklearn.metrics.pairwise import rbf_kernel

def apply_feature_selection(X, y=None, variance_threshold=0.01, select_k=None):
    """
    Apply feature selection to reduce dimensionality and remove less informative features.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    y : array-like, shape (n_samples,), optional
        Target classes for supervised feature selection
    variance_threshold : float, default=0.01
        Threshold for variance-based feature selection
    select_k : int, optional
        Number of top features to select if using supervised selection
        
    Returns:
    --------
    X_selected : array-like
        Data after feature selection
    """
    # First, apply variance threshold to remove near-constant features
    var_selector = VarianceThreshold(threshold=variance_threshold)
    X_var_selected = var_selector.fit_transform(X)
    
    print(f"After variance-based selection: {X_var_selected.shape[1]} features (removed {X.shape[1] - X_var_selected.shape[1]} features)")
    
    # If y is provided and select_k is specified, apply supervised feature selection
    if y is not None and select_k is not None:
        selector = SelectKBest(f_classif, k=select_k)
        X_selected = selector.fit_transform(X_var_selected, y)
        print(f"After supervised selection: {X_selected.shape[1]} features")
        return X_selected
    
    return X_var_selected

def create_feature_interactions(X, degree=2, include_bias=False, interaction_only=True):
    """
    Create interaction features to capture non-linear relationships.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    degree : int, default=2
        The degree of the polynomial features
    include_bias : bool, default=False
        Whether to include a bias column (all 1s)
    interaction_only : bool, default=True
        If True, only interaction features are produced, not pure polynomial features
        
    Returns:
    --------
    X_interaction : array-like
        Data with interaction features
    """
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias, interaction_only=interaction_only)
    X_interaction = poly.fit_transform(X)
    
    print(f"After adding interaction features: {X_interaction.shape[1]} features")
    return X_interaction

def apply_feature_agglomeration(X, n_clusters=50):
    """
    Apply feature agglomeration to group similar features.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    n_clusters : int, default=50
        Number of feature clusters
        
    Returns:
    --------
    X_agg : array-like
        Data after feature agglomeration
    """
    agglo = FeatureAgglomeration(n_clusters=n_clusters)
    X_agg = agglo.fit_transform(X)
    
    print(f"After feature agglomeration: {X_agg.shape[1]} features")
    return X_agg

def apply_advanced_pca(X, n_components=None, whiten=True, random_state=42):
    """
    Apply PCA with optimal settings for improved class separation.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    n_components : int, optional
        Number of components to keep, if None, will keep 95% of variance
    whiten : bool, default=True
        Whether to whiten the data
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    X_pca : array-like
        PCA transformed data
    pca : PCA object
        The fitted PCA object for later inspection
    """
    if n_components is None:
        # Use enough components to explain 95% of the variance
        pca = PCA(n_components=0.95, whiten=whiten, random_state=random_state)
    else:
        pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
    
    X_pca = pca.fit_transform(X)
    
    print(f"After PCA: {X_pca.shape[1]} components")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    return X_pca, pca

def apply_optimal_kernel_pca(X, n_components=20, kernel='rbf', gamma=None):
    """
    Apply Kernel PCA with optimal settings for non-linear dimensionality reduction.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    n_components : int, default=20
        Number of components to extract
    kernel : str, default='rbf'
        Kernel type - 'rbf', 'poly', 'sigmoid', 'cosine', or 'linear'
    gamma : float, optional
        Kernel coefficient, if None, defaults to 1/n_features
        
    Returns:
    --------
    X_kpca : array-like
        Kernel PCA transformed data
    """
    kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, 
                     fit_inverse_transform=True, random_state=42)
    X_kpca = kpca.fit_transform(X)
    
    print(f"After Kernel PCA ({kernel}): {X_kpca.shape[1]} components")
    return X_kpca

def apply_feature_importance_selection(X, y, n_estimators=100, max_features=50):
    """
    Use a Random Forest to select features based on importance scores.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    y : array-like, shape (n_samples,)
        Target classes
    n_estimators : int, default=100
        Number of trees in the forest
    max_features : int, default=50
        Maximum number of features to select
        
    Returns:
    --------
    X_selected : array-like
        Data with selected features
    """
    # Train a Random Forest classifier
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf.fit(X, y)
    
    # Select top features based on importance
    selector = SelectFromModel(rf, max_features=max_features, threshold=-np.inf)
    X_selected = selector.fit_transform(X, y)
    
    print(f"After Random Forest feature selection: {X_selected.shape[1]} features")
    return X_selected

def apply_multi_method_manifold_learning(X, n_components=2):
    """
    Apply multiple manifold learning methods and return the best result based on class separation.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    n_components : int, default=2
        Number of components for the embedding
        
    Returns:
    --------
    dict : A dictionary containing each embedding result
    """
    results = {}
    
    # Apply t-SNE
    try:
        # For large datasets, use a lower perplexity
        perplexity = min(30, X.shape[0] // 5)
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        results['tsne'] = tsne.fit_transform(X)
        print(f"t-SNE applied, output shape: {results['tsne'].shape}")
    except Exception as e:
        print(f"t-SNE failed: {e}")
    
    # Apply Isomap
    try:
        # For large datasets, use a smaller number of neighbors
        n_neighbors = min(15, X.shape[0] // 10)
        isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors)
        results['isomap'] = isomap.fit_transform(X)
        print(f"Isomap applied, output shape: {results['isomap'].shape}")
    except Exception as e:
        print(f"Isomap failed: {e}")
    
    # Apply Locally Linear Embedding
    try:
        n_neighbors = min(15, X.shape[0] // 10)
        lle = LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors, 
                                    random_state=42)
        results['lle'] = lle.fit_transform(X)
        print(f"LLE applied, output shape: {results['lle'].shape}")
    except Exception as e:
        print(f"LLE failed: {e}")
    
    # Apply Spectral Embedding
    try:
        n_neighbors = min(15, X.shape[0] // 10)
        spectral = SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors, 
                                    random_state=42)
        results['spectral'] = spectral.fit_transform(X)
        print(f"Spectral Embedding applied, output shape: {results['spectral'].shape}")
    except Exception as e:
        print(f"Spectral Embedding failed: {e}")
    
    return results

def apply_auto_feature_engineering(X, y=None, target_dim=20, include_interactions=True):
    """
    Automatic feature engineering pipeline that combines multiple techniques.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    y : array-like, shape (n_samples,), optional
        Target classes
    target_dim : int, default=20
        Target dimension for the final feature set
    include_interactions : bool, default=True
        Whether to include feature interactions
        
    Returns:
    --------
    X_transformed : array-like
        Transformed features
    """
    # Start with scaling
    scaled_data = RobustScaler().fit_transform(X)
    
    # Feature selection to remove low-variance features
    X_selected = apply_feature_selection(scaled_data, variance_threshold=0.01)
    
    # Add interaction features if requested and if dimensionality is not too high
    if include_interactions and X_selected.shape[1] < 100:
        X_with_interactions = create_feature_interactions(X_selected, degree=2, interaction_only=True)
    else:
        X_with_interactions = X_selected
    
    # Feature agglomeration to reduce dimensionality
    if X_with_interactions.shape[1] > target_dim * 3:
        n_clusters = min(target_dim * 2, X_with_interactions.shape[1] // 2)
        X_agg = apply_feature_agglomeration(X_with_interactions, n_clusters=n_clusters)
    else:
        X_agg = X_with_interactions
    
    # Apply PCA for final dimensionality reduction
    X_pca, _ = apply_advanced_pca(X_agg, n_components=target_dim)
    
    return X_pca

def apply_ica_for_separation(X, n_components=20):
    """
    Apply Independent Component Analysis (ICA) to enhance feature separation.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    n_components : int, default=20
        Number of components
        
    Returns:
    --------
    X_ica : array-like
        ICA transformed data
    """
    ica = FastICA(n_components=n_components, random_state=42)
    X_ica = ica.fit_transform(X)
    
    print(f"After ICA: {X_ica.shape[1]} components")
    return X_ica

def apply_ensemble_dimensionality_reduction(X, n_components=20):
    """
    Apply ensemble dimensionality reduction combining PCA, Kernel PCA and ICA.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    n_components : int, default=20
        Number of components for each method
        
    Returns:
    --------
    X_ensemble : array-like
        Ensemble transformed data
    """
    # Apply multiple dimensionality reduction techniques
    X_pca, _ = apply_advanced_pca(X, n_components=n_components)
    X_kpca = apply_optimal_kernel_pca(X, n_components=n_components)
    X_ica = apply_ica_for_separation(X, n_components=n_components)
    
    # Combine the results
    X_combined = np.hstack([X_pca, X_kpca, X_ica])
    
    # Apply final PCA to reduce to the desired dimensionality
    X_final, _ = apply_advanced_pca(X_combined, n_components=n_components)
    
    print(f"After ensemble dimensionality reduction: {X_final.shape[1]} components")
    return X_final

def visualize_feature_importance(X, y, feature_names=None, top_n=20):
    """
    Visualize feature importance using a Random Forest classifier.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    y : array-like, shape (n_samples,)
        Target classes
    feature_names : list, optional
        List of feature names
    top_n : int, default=20
        Number of top features to display
    """
    # Train a Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importances
    importances = rf.feature_importances_
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:top_n]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(top_n), importances[top_indices])
    plt.xticks(range(top_n), [feature_names[i] for i in top_indices], rotation=90)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    plt.close()
    
    print(f"Feature importance visualization saved to 'plots/feature_importance.png'")
    
    return dict(zip([feature_names[i] for i in top_indices], importances[top_indices]))

def evaluate_feature_engineering(X, y, methods):
    """
    Evaluate different feature engineering methods for class separation.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    y : array-like, shape (n_samples,)
        Target classes
    methods : list of str
        List of methods to evaluate
        
    Returns:
    --------
    results : dict
        Dictionary with transformed data for each method
    """
    results = {}
    
    # Standardize data
    X_std = StandardScaler().fit_transform(X)
    
    for method in methods:
        if method == 'pca':
            results[method], _ = apply_advanced_pca(X_std, n_components=2)
        elif method == 'kernel_pca':
            results[method] = apply_optimal_kernel_pca(X_std, n_components=2)
        elif method == 'ica':
            results[method] = apply_ica_for_separation(X_std, n_components=2)
        elif method == 'isomap':
            try:
                n_neighbors = min(15, X.shape[0] // 10)
                isomap = Isomap(n_components=2, n_neighbors=n_neighbors)
                results[method] = isomap.fit_transform(X_std)
            except Exception as e:
                print(f"Isomap failed: {e}")
        elif method == 'tsne':
            try:
                perplexity = min(30, X.shape[0] // 5)
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                results[method] = tsne.fit_transform(X_std)
            except Exception as e:
                print(f"t-SNE failed: {e}")
        elif method == 'ensemble':
            results[method] = apply_ensemble_dimensionality_reduction(X_std, n_components=2)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    for i, (method, X_transformed) in enumerate(results.items()):
        plt.subplot(2, 3, i+1)
        for class_label in np.unique(y):
            plt.scatter(X_transformed[y == class_label, 0], X_transformed[y == class_label, 1], 
                        alpha=0.7, label=f'Class {class_label}')
        plt.title(method)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('plots/feature_engineering_comparison.png')
    plt.close()
    
    print(f"Comparison visualization saved to 'plots/feature_engineering_comparison.png'")
    
    return results

def optimize_feature_engineering_for_clustering(X, y=None, n_components=20):
    """
    Optimize feature engineering specifically for clustering.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    y : array-like, shape (n_samples,), optional
        Target classes (if available)
    n_components : int, default=20
        Number of components for the final feature set
        
    Returns:
    --------
    X_optimized : array-like
        Optimized features for clustering
    """
    # Start with robust scaling to handle outliers
    X_scaled = RobustScaler().fit_transform(X)
    
    # Apply variance threshold to remove near-constant features
    var_selector = VarianceThreshold(threshold=0.01)
    X_var = var_selector.fit_transform(X_scaled)
    print(f"After variance selection: {X_var.shape[1]} features")
    
    # For high-dimensional data, use feature agglomeration
    if X_var.shape[1] > 100:
        agglo = FeatureAgglomeration(n_clusters=min(100, X_var.shape[1] // 2))
        X_agglo = agglo.fit_transform(X_var)
        print(f"After feature agglomeration: {X_agglo.shape[1]} features")
    else:
        X_agglo = X_var
    
    # Apply kernel PCA with RBF kernel for non-linear mapping
    kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=10/X_agglo.shape[1], 
                     random_state=42)
    X_kpca = kpca.fit_transform(X_agglo)
    print(f"After Kernel PCA: {X_kpca.shape[1]} components")
    
    # If we have labels, we can further select components that best separate the classes
    if y is not None:
        # Use truncated SVD as a final step for class separation
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        X_optimized = svd.fit_transform(X_kpca)
        print(f"After final optimization: {X_optimized.shape[1]} components")
        print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")
    else:
        X_optimized = X_kpca
    
    return X_optimized

def create_region_ratio_features(X, feature_names):
    """
    Create ratio features between different brain regions to capture relative differences.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    feature_names : list
        List of feature names
        
    Returns:
    --------
    X_with_ratios : array-like
        Data with additional ratio features
    new_feature_names : list
        Updated list of feature names
    """
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(X, columns=feature_names)
    
    # Identify regions (assuming naming convention like 'fsArea_L_V1_ROI', 'fsArea_R_V1_ROI')
    left_regions = [col for col in df.columns if '_L_' in col]
    right_regions = [col for col in df.columns if '_R_' in col]
    
    print(f"Found {len(left_regions)} left regions and {len(right_regions)} right regions")
    
    # Create ratios between corresponding left and right regions
    new_features = {}
    for l_region in left_regions:
        # Try to find the corresponding right region
        r_region = l_region.replace('_L_', '_R_')
        if r_region in right_regions:
            # Create ratio feature (left / right)
            ratio_name = f"ratio_{l_region.split('_')[2]}"
            # Avoid division by zero
            new_features[ratio_name] = df[l_region] / df[r_region].replace(0, np.nan).fillna(df[r_region].mean())
    
    # Create a DataFrame with the new features
    ratio_df = pd.DataFrame(new_features)
    
    # Combine with original features
    combined_df = pd.concat([df, ratio_df], axis=1)
    
    print(f"Added {len(new_features)} ratio features")
    
    return combined_df.values, list(combined_df.columns)

def identify_discriminative_region_combinations(X, y, feature_names):
    """
    Identify combinations of brain regions that best discriminate between classes.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    y : array-like, shape (n_samples,)
        Target classes
    feature_names : list
        List of feature names
        
    Returns:
    --------
    top_combinations : list
        List of tuples containing the most discriminative feature pairs
    """
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Calculate feature importances
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    
    # Get the top 20 features
    top_indices = importances.argsort()[-20:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    
    # Calculate feature combinations (products) for the top features
    combinations = []
    for i in range(len(top_features)):
        for j in range(i+1, len(top_features)):
            feature1 = top_features[i]
            feature2 = top_features[j]
            
            # Create interaction feature
            interaction_name = f"{feature1}_{feature2}"
            df[interaction_name] = df[feature1] * df[feature2]
            
            # Calculate class separation based on this interaction
            f_value, p_value = f_classif(df[[interaction_name]], df['target'])
            combinations.append((feature1, feature2, f_value[0], p_value[0]))
    
    # Sort by F-value (higher is better)
    combinations.sort(key=lambda x: x[2], reverse=True)
    
    # Return the top 10 combinations
    top_combinations = combinations[:10]
    
    print("Top discriminative region combinations:")
    for feature1, feature2, f_value, p_value in top_combinations:
        print(f"{feature1} × {feature2}: F={f_value:.2f}, p={p_value:.4f}")
    
    return top_combinations

def apply_combined_optimal_approach(X, feature_names, n_components=20):
    """
    Apply the combined optimal approach using Region Ratio Features, Feature Interactions,
    and Ensemble Dimensionality Reduction.
    
    This function implements the recommended approach based on analysis of clustering results:
    1. Create region ratio features (best ARI)
    2. Apply feature selection with variance threshold
    3. Create feature interactions between important features
    4. Apply both standard PCA and kernel PCA
    5. Combine results and apply final PCA
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    feature_names : list
        List of feature names for the features in X
    n_components : int, default=20
        Number of components for the final output
        
    Returns:
    --------
    X_final : array-like
        Transformed data using the optimal approach
    """
    print(f"\n=== Applying Combined Optimal Approach ===")
    print(f"Input data shape: {X.shape}")
    
    # Step 1: Create region ratio features (best ARI)
    print("\nStep 1: Creating region ratio features...")
    X_with_ratios, ratio_feature_names = create_region_ratio_features(X, feature_names)
    print(f"Data with ratio features shape: {X_with_ratios.shape}")
    
    # Robust scaling to handle outliers
    print("\nApplying robust scaling...")
    X_scaled = RobustScaler().fit_transform(X_with_ratios)
    
    # Step 2: Feature selection with variance threshold
    print("\nStep 2: Applying feature selection...")
    var_threshold = 0.01
    X_selected = apply_feature_selection(X_scaled, variance_threshold=var_threshold)
    print(f"After feature selection shape: {X_selected.shape}")
    
    # Step 3: Create feature interactions between important features
    print("\nStep 3: Creating feature interactions...")
    # Only create interactions if not too many features to avoid dimensionality explosion
    if X_selected.shape[1] < 100:
        X_with_interactions = create_feature_interactions(X_selected, degree=2, interaction_only=True)
        print(f"After adding interactions shape: {X_with_interactions.shape}")
    else:
        # If too many features, skip interactions
        X_with_interactions = X_selected
        print(f"Too many features for interactions, skipping this step.")
    
    # Apply Feature Agglomeration if dimensionality is still very high
    if X_with_interactions.shape[1] > 1000:
        print("\nReducing dimensionality with Feature Agglomeration...")
        n_clusters = min(1000, X_with_interactions.shape[1] // 2)
        agglo = FeatureAgglomeration(n_clusters=n_clusters, linkage='ward')
        X_with_interactions = agglo.fit_transform(X_with_interactions)
        print(f"After feature agglomeration shape: {X_with_interactions.shape}")
    
    # Step 4: Apply both advanced PCA with whitening and Kernel PCA
    print("\nStep 4: Applying multiple dimensionality reduction techniques...")
    
    # Calculate intermediate components (more than final to allow for later selection)
    intermediate_components = min(n_components * 2, X_with_interactions.shape[1])
    
    # Initialize the list to store transformed data
    transformed_data = []
    
    # Apply advanced PCA with whitening
    try:
        X_pca, pca = apply_advanced_pca(X_with_interactions, 
                                       n_components=intermediate_components, 
                                       whiten=True)
        print(f"PCA output shape: {X_pca.shape}")
        transformed_data.append(X_pca)
    except Exception as e:
        print(f"Error applying PCA: {e}")
    
    # Apply optimal Kernel PCA with adaptive gamma parameter
    try:
        gamma_value = 10.0 / X_with_interactions.shape[1]  # Adaptive gamma based on data dimension
        X_kpca = apply_optimal_kernel_pca(X_with_interactions, 
                                         n_components=intermediate_components, 
                                         kernel='rbf', 
                                         gamma=gamma_value)
        print(f"Kernel PCA output shape: {X_kpca.shape}")
        transformed_data.append(X_kpca)
    except Exception as e:
        print(f"Error applying RBF Kernel PCA: {e}")
    
    # Apply polynomial Kernel PCA for capturing higher-order relationships
    try:
        # Use smaller gamma for polynomial kernel to avoid singular matrix
        gamma_value = 1.0 / X_with_interactions.shape[1]
        X_poly_kpca = apply_optimal_kernel_pca(X_with_interactions, 
                                              n_components=intermediate_components, 
                                              kernel='poly', 
                                              gamma=gamma_value)
        print(f"Polynomial Kernel PCA output shape: {X_poly_kpca.shape}")
        transformed_data.append(X_poly_kpca)
    except Exception as e:
        print(f"Error applying Polynomial Kernel PCA: {e}")
        # If polynomial kernel fails, try sigmoid kernel instead
        try:
            X_sigmoid_kpca = apply_optimal_kernel_pca(X_with_interactions, 
                                                    n_components=intermediate_components, 
                                                    kernel='sigmoid', 
                                                    gamma=gamma_value)
            print(f"Sigmoid Kernel PCA output shape: {X_sigmoid_kpca.shape}")
            transformed_data.append(X_sigmoid_kpca)
        except Exception as e:
            print(f"Error applying Sigmoid Kernel PCA: {e}")
    
    # Step 5: Combine results and apply final PCA
    print("\nStep 5: Combining results and applying final dimensionality reduction...")
    
    # Check if we have any transformed data
    if len(transformed_data) == 0:
        print("All dimensionality reduction methods failed. Applying standard PCA directly.")
        X_final, _ = apply_advanced_pca(X_with_interactions, n_components=n_components)
        return X_final
    
    if len(transformed_data) == 1:
        # If only one method succeeded, use its output directly
        print("Only one dimensionality reduction method succeeded.")
        X_final = transformed_data[0][:, :n_components]
        return X_final
    
    # Concatenate the outputs horizontally
    X_combined = np.hstack(transformed_data)
    print(f"Combined output shape: {X_combined.shape}")
    
    # Apply final PCA to get the desired number of components
    try:
        X_final, final_pca = apply_advanced_pca(X_combined, n_components=n_components, whiten=True)
        print(f"Final output shape: {X_final.shape}")
        
        # Calculate % variance explained by final components
        variance_explained = np.sum(final_pca.explained_variance_ratio_)
        print(f"Variance explained by final {n_components} components: {variance_explained:.4f} ({variance_explained*100:.2f}%)")
    except Exception as e:
        print(f"Error applying final PCA: {e}")
        # If final PCA fails, use truncated SVD instead
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        X_final = svd.fit_transform(X_combined)
        print(f"Using Truncated SVD instead. Final output shape: {X_final.shape}")
        variance_explained = np.sum(svd.explained_variance_ratio_)
        print(f"Variance explained by final {n_components} components: {variance_explained:.4f} ({variance_explained*100:.2f}%)")
    
    return X_final

def apply_zca_whitening_pca(X, n_components=20, eps=1e-6, random_state=42):
    """
    Apply ZCA whitening before performing PCA to decorrelate features.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    n_components : int, default=20
        Number of principal components to keep after PCA
    eps : float, default=1e-6
        Small value added to avoid division by zero
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    X_zca_pca : array-like
        Data after ZCA whitening and PCA
    """
    print("Applying ZCA whitening followed by PCA...")
    
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov = np.cov(X_centered, rowvar=False)
    
    # Singular Value Decomposition
    U, S, V = linalg.svd(cov)
    
    # Apply ZCA whitening transformation
    components = U.dot(np.diag(1.0 / np.sqrt(S + eps))).dot(U.T)
    X_whitened = X_centered.dot(components)
    
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=n_components, random_state=random_state)
    X_zca_pca = pca.fit_transform(X_whitened)
    
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"After ZCA whitening + PCA: {X_zca_pca.shape[1]} components")
    print(f"Explained variance ratio after dimensionality reduction: {explained_variance:.4f}")
    
    return X_zca_pca

def apply_cholesky_whitening(X, n_components=20, random_state=42):
    """
    Apply Cholesky whitening followed by PCA for dimensionality reduction.
    Cholesky whitening uses the Cholesky decomposition of the inverse covariance matrix.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    n_components : int, default=20
        Number of principal components to keep after PCA
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    X_chol_pca : array-like
        Data after Cholesky whitening and PCA
    """
    print("Applying Cholesky whitening followed by PCA...")
    
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov = np.cov(X_centered, rowvar=False)
    
    # Add small regularization to ensure positive-definiteness
    cov += 1e-6 * np.eye(cov.shape[0])
    
    # Cholesky decomposition of covariance matrix
    L = linalg.cholesky(cov)
    
    # Apply Cholesky whitening
    L_inv = linalg.inv(L)
    X_whitened = X_centered.dot(L_inv.T)
    
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=n_components, random_state=random_state)
    X_chol_pca = pca.fit_transform(X_whitened)
    
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"After Cholesky whitening + PCA: {X_chol_pca.shape[1]} components")
    print(f"Explained variance ratio after dimensionality reduction: {explained_variance:.4f}")
    
    return X_chol_pca

def apply_mahalanobis_whitening(X, n_components=20, random_state=42):
    """
    Apply Mahalanobis whitening followed by PCA.
    This method applies decorrelation and normalization using the Mahalanobis distance.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    n_components : int, default=20
        Number of principal components to keep after PCA
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    X_mahal_pca : array-like
        Data after Mahalanobis whitening and PCA
    """
    print("Applying Mahalanobis whitening followed by PCA...")
    
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov = np.cov(X_centered, rowvar=False)
    
    # Add small regularization to ensure positive-definiteness
    cov += 1e-6 * np.eye(cov.shape[0])
    
    # Calculate the inverse square root of the covariance matrix for Mahalanobis transformation
    evals, evecs = linalg.eigh(cov)
    evals = np.maximum(evals, 1e-6)  # Ensure all eigenvalues are positive
    P = evecs.dot(np.diag(1.0 / np.sqrt(evals))).dot(evecs.T)
    
    # Apply Mahalanobis whitening
    X_whitened = X_centered.dot(P)
    
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=n_components, random_state=random_state)
    X_mahal_pca = pca.fit_transform(X_whitened)
    
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"After Mahalanobis whitening + PCA: {X_mahal_pca.shape[1]} components")
    print(f"Explained variance ratio after dimensionality reduction: {explained_variance:.4f}")
    
    return X_mahal_pca

def apply_pca_adaptive_whitening(X, n_components=20, alpha=1.0, eps=1e-6, random_state=42):
    """
    Apply PCA with adaptive whitening based on eigenvalue magnitudes.
    The parameter alpha controls the strength of whitening based on eigenvalue size.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    n_components : int, default=20
        Number of principal components to keep after PCA
    alpha : float, default=1.0
        Whitening strength parameter. 1.0 is full whitening, 0 is no whitening.
    eps : float, default=1e-6
        Small value added to avoid division by zero
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    X_adaptive_whitened : array-like
        Data after adaptive whitening and PCA
    """
    print(f"Applying PCA with adaptive whitening (alpha={alpha})...")
    
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov = np.cov(X_centered, rowvar=False)
    
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = linalg.eigh(cov)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Apply adaptive whitening to modify eigenvalues
    modified_eigenvalues = eigenvalues ** (1 - alpha)
    
    # Calculate transformation matrix
    transform = eigenvectors.dot(np.diag(1.0 / np.sqrt(eigenvalues + eps)))
    
    # Apply adaptive whitening transformation
    X_whitened = X_centered.dot(transform.dot(np.diag(modified_eigenvalues)))
    
    # Reduce dimensionality if needed
    if n_components < X_whitened.shape[1]:
        X_adaptive_whitened = X_whitened[:, :n_components]
    else:
        X_adaptive_whitened = X_whitened
    
    print(f"After adaptive whitening PCA: {X_adaptive_whitened.shape[1]} components")
    return X_adaptive_whitened

def apply_hybrid_whitening(X, n_components=20, zca_weight=0.5, adaptive_alpha=0.7, eps=1e-6, random_state=42):
    """
    Apply hybrid whitening that combines ZCA and adaptive PCA whitening.
    This method preserves interpretability of features while controlling whitening strength.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    n_components : int, default=20
        Number of principal components to keep
    zca_weight : float, default=0.5
        Weight for ZCA component (0-1). Higher values preserve more original feature space.
    adaptive_alpha : float, default=0.7
        Adaptive whitening strength parameter (0-1). Higher values apply more whitening.
    eps : float, default=1e-6
        Small value added to avoid division by zero
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    X_hybrid : array-like
        Data after hybrid whitening and dimensionality reduction
    """
    print(f"Applying hybrid whitening (ZCA weight={zca_weight}, adaptive alpha={adaptive_alpha})...")
    
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov = np.cov(X_centered, rowvar=False)
    
    # SVD decomposition for eigenvectors and eigenvalues
    U, S, _ = linalg.svd(cov)
    
    # Ensure eigenvalues are positive and sorted
    S = np.maximum(S, eps)
    
    # Calculate ZCA transformation component
    zca_components = U.dot(np.diag(1.0 / np.sqrt(S))).dot(U.T)
    
    # Calculate adaptive transformation component:
    # Modify eigenvalues for adaptive whitening
    modified_S = S ** (1 - adaptive_alpha)
    adaptive_components = U.dot(np.diag(1.0 / np.sqrt(S))).dot(np.diag(modified_S))
    
    # Apply weighted hybrid whitening transformation
    if zca_weight <= 0:
        # Pure adaptive whitening
        X_whitened = X_centered.dot(adaptive_components)
    elif zca_weight >= 1:
        # Pure ZCA whitening
        X_whitened = X_centered.dot(zca_components)
    else:
        # Weighted combination of both methods
        # Apply ZCA portion
        X_zca = X_centered.dot(zca_components)
        
        # Apply adaptive portion
        X_adaptive = X_centered.dot(adaptive_components)
        
        # Combine with weighted average
        X_whitened = zca_weight * X_zca + (1 - zca_weight) * X_adaptive
    
    # Apply dimensionality reduction if needed
    if n_components < X_whitened.shape[1]:
        # Use PCA for final dimensionality reduction
        pca = PCA(n_components=n_components, random_state=random_state)
        X_hybrid = pca.fit_transform(X_whitened)
        explained_variance = np.sum(pca.explained_variance_ratio_)
        print(f"After dimensionality reduction: explained variance = {explained_variance:.4f}")
    else:
        X_hybrid = X_whitened
    
    print(f"After hybrid whitening: {X_hybrid.shape[1]} components")
    return X_hybrid

def apply_whitening_graph_feature_selection_pca(X, n_components=20, threshold=0.5, random_state=42):
    """
    Apply a combination of whitening, graph-based feature selection, and PCA.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    n_components : int, default=20
        Number of components for the final PCA
    threshold : float, default=0.5
        Threshold for feature selection in correlation graph
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    X_transformed : array-like
        Transformed data
    """
    # First, apply ZCA whitening
    X_whitened = zca_whitening(X)
    
    # Build a feature correlation graph
    corr_matrix = np.abs(np.corrcoef(X_whitened.T))
    
    # Create a network from the correlation matrix
    G = nx.from_numpy_array(corr_matrix)
    
    # Calculate centrality measures
    centrality = nx.eigenvector_centrality_numpy(G)
    
    # Select features with high centrality
    selected_features = [i for i, c in enumerate(centrality.values()) if c > threshold]
    
    # Ensure we have at least n_components features
    if len(selected_features) < n_components:
        print(f"Warning: Only {len(selected_features)} features selected. Adding more based on centrality ranking.")
        # Sort features by centrality
        sorted_features = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        additional_features = [f[0] for f in sorted_features[:n_components]]
        # Combine and deduplicate
        selected_features = list(set(selected_features + additional_features))
    
    # Select the identified features
    X_selected = X_whitened[:, selected_features]
    
    print(f"Selected {X_selected.shape[1]} features based on graph centrality")
    
    # Apply final PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    X_transformed = pca.fit_transform(X_selected)
    
    print(f"Final shape after whitening + graph selection + PCA: {X_transformed.shape}")
    
    return X_transformed

def apply_double_pca(X, n_components=20, random_state=42):
    """
    Apply a double PCA transformation: first PCA without specifying components,
    followed by a second PCA with a specified number of components.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    n_components : int, default=20
        Number of components for the second PCA
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    X_transformed : array-like
        Transformed data after double PCA
    """
    print(f"Applying double PCA with final {n_components} components...")
    
    # First PCA without restricting the number of components
    # We'll let scikit-learn determine the optimal number internally
    pca1 = PCA(random_state=random_state)
    X_pca1 = pca1.fit_transform(X)
    
    print(f"First PCA produced {X_pca1.shape[1]} components")
    print(f"Explained variance ratio: {pca1.explained_variance_ratio_.sum():.4f}")
    
    # Second PCA with specified number of components
    pca2 = PCA(n_components=n_components, random_state=random_state)
    X_pca2 = pca2.fit_transform(X_pca1)
    
    print(f"Second PCA produced {X_pca2.shape[1]} components")
    print(f"Explained variance ratio of second PCA: {pca2.explained_variance_ratio_.sum():.4f}")
    print(f"Total explained variance: {pca1.explained_variance_ratio_.sum() * pca2.explained_variance_ratio_.sum():.4f}")
    
    return X_pca2

# Add more specialized feature engineering functions as needed 