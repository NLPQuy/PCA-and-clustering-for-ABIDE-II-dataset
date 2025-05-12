#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ABIDE Feature Engineering Demo
This script demonstrates advanced feature engineering techniques to improve class separation
for the ABIDE dataset, particularly for clustering applications.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Import our custom feature engineering module
from src.abide_feature_engineering import (
    apply_feature_selection,
    create_feature_interactions,
    apply_advanced_pca,
    apply_optimal_kernel_pca,
    optimize_feature_engineering_for_clustering,
    evaluate_feature_engineering,
    create_region_ratio_features,
    identify_discriminative_region_combinations,
    apply_ensemble_dimensionality_reduction,
    visualize_feature_importance
)

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

def load_abide_data(file_path):
    """
    Load ABIDE dataset from CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    X : array-like
        Feature matrix
    y : array-like
        Target labels (binary: 0 for Normal, 1 for Cancer)
    feature_names : list
        List of feature names
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Extract target (diagnoses)
    target_col = 'group'
    if target_col in df.columns:
        # Encode 'Normal' as 0, 'Cancer' as 1
        le = LabelEncoder()
        y = le.fit_transform(df[target_col])
        class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"Class mapping: {class_mapping}")
    else:
        raise ValueError(f"Target column '{target_col}' not found in the dataset.")
    
    # Extract features (exclude non-feature columns)
    non_feature_cols = ['Unnamed: 0', 'site', 'subject', 'age', 'group']
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    
    # Create feature matrix
    X = df[feature_cols].values
    
    print(f"Loaded dataset with {X.shape[0]} samples and {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y, feature_cols

def evaluate_class_separation(X, y, method_name='Original'):
    """
    Simple function to visualize class separation using first two components.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix (at least 2D)
    y : array-like
        Target labels
    method_name : str
        Name of the method used for feature engineering
    """
    # Use only first two dimensions for plotting
    X_2d = X[:, :2]
    
    # Plot
    plt.figure(figsize=(8, 6))
    for class_label in np.unique(y):
        class_name = 'Cancer' if class_label == 1 else 'Normal'
        plt.scatter(X_2d[y == class_label, 0], X_2d[y == class_label, 1], 
                   alpha=0.7, label=class_name)
    
    plt.title(f'Class Separation with {method_name}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.savefig(f'plots/class_separation_{method_name.replace(" ", "_").lower()}.png')
    plt.close()
    
    print(f"Class separation visualization saved for {method_name}")

def demonstrate_feature_engineering():
    """
    Main function to demonstrate feature engineering techniques.
    """
    print("=== ABIDE Feature Engineering Demo ===")
    
    # Load ABIDE sample dataset
    X, y, feature_names = load_abide_data("ABIDE2_sample.csv")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print("\n=== 1. Basic Analysis ===")
    
    # Standardize the data
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    
    # Apply basic PCA
    X_pca, pca = apply_advanced_pca(X_train_std, n_components=2)
    
    # Evaluate initial class separation
    evaluate_class_separation(X_pca, y_train, method_name='Basic PCA')
    
    print("\n=== 2. Feature Selection ===")
    
    # Apply feature selection
    X_selected = apply_feature_selection(X_train_std, y_train, variance_threshold=0.01, select_k=100)
    
    # Apply PCA on selected features
    X_selected_pca, _ = apply_advanced_pca(X_selected, n_components=2)
    
    # Evaluate class separation after feature selection
    evaluate_class_separation(X_selected_pca, y_train, method_name='Feature Selection + PCA')
    
    print("\n=== 3. Feature Interactions ===")
    
    # Add interaction features to the selected features
    X_with_interactions = create_feature_interactions(X_selected, degree=2, interaction_only=True)
    
    # Apply PCA on features with interactions
    X_interactions_pca, _ = apply_advanced_pca(X_with_interactions, n_components=2)
    
    # Evaluate class separation with interaction features
    evaluate_class_separation(X_interactions_pca, y_train, method_name='Feature Interactions + PCA')
    
    print("\n=== 4. Kernel PCA ===")
    
    # Apply Kernel PCA
    X_kpca = apply_optimal_kernel_pca(X_selected, n_components=2, kernel='rbf')
    
    # Evaluate class separation with Kernel PCA
    evaluate_class_separation(X_kpca, y_train, method_name='Kernel PCA')
    
    print("\n=== 5. Region Ratio Features ===")
    
    # Create ratio features between left and right brain regions
    X_with_ratios, ratio_feature_names = create_region_ratio_features(X_train, feature_names)
    
    # Standardize
    X_ratios_std = StandardScaler().fit_transform(X_with_ratios)
    
    # Apply PCA on data with ratio features
    X_ratios_pca, _ = apply_advanced_pca(X_ratios_std, n_components=2)
    
    # Evaluate class separation with ratio features
    evaluate_class_separation(X_ratios_pca, y_train, method_name='Region Ratios + PCA')
    
    print("\n=== 6. Identify Discriminative Region Combinations ===")
    
    # Find most discriminative feature combinations
    top_combinations = identify_discriminative_region_combinations(X_train_std, y_train, feature_names)
    
    print("\n=== 7. Ensemble Dimensionality Reduction ===")
    
    # Apply ensemble dimensionality reduction
    X_ensemble = apply_ensemble_dimensionality_reduction(X_selected, n_components=2)
    
    # Evaluate class separation with ensemble approach
    evaluate_class_separation(X_ensemble, y_train, method_name='Ensemble Reduction')
    
    print("\n=== 8. Feature Importance Visualization ===")
    
    # Visualize feature importance
    top_features = visualize_feature_importance(X_train_std, y_train, feature_names, top_n=20)
    
    print("\n=== 9. Comparative Evaluation ===")
    
    # Evaluate different feature engineering methods
    methods = ['pca', 'kernel_pca', 'ica', 'tsne', 'isomap', 'ensemble']
    results = evaluate_feature_engineering(X_selected, y_train, methods)
    
    print("\n=== 10. Optimized Features for Clustering ===")
    
    # Apply optimized feature engineering for clustering
    X_optimized = optimize_feature_engineering_for_clustering(X_train_std, y_train, n_components=2)
    
    # Evaluate class separation with optimized features
    evaluate_class_separation(X_optimized, y_train, method_name='Optimized for Clustering')
    
    print("\n=== Demo Complete ===")
    print("Feature engineering visualizations have been saved to the 'plots' directory.")

if __name__ == "__main__":
    demonstrate_feature_engineering() 