import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

def load_iris_data():
    """
    Load the Iris dataset from scikit-learn.
    
    Returns:
    --------
    X : array-like, shape (n_samples, n_features)
        The data matrix
    y : array-like, shape (n_samples,)
        The class labels
    features : list
        List of feature names
    target_names : list
        List of target class names
    """
    iris = load_iris()
    X = iris['data']
    y = iris['target']
    features = iris['feature_names']
    target_names = iris['target_names']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Feature names: {features}")
    print(f"Target classes: {target_names}")
    
    return X, y, features, target_names

def load_abide_data(filepath):
    """
    Load the ABIDE II dataset from a CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the ABIDE II CSV file
        
    Returns:
    --------
    X : array-like, shape (n_samples, n_features)
        The data matrix
    y : array-like, shape (n_samples,)
        The class labels (0 for 'Normal', 1 for 'Cancer')
    feature_names : list
        List of feature names
    """
    # Load the data
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading ABIDE data: {e}")
        return None, None, None
    
    # Display dataset information
    print(f"Dataset shape: {df.shape}")
    print(f"Number of samples: {df.shape[0]}")
    print(f"Number of features: {df.shape[1]}")
    
    # Check if 'group' column exists
    if 'group' not in df.columns:
        print("Warning: 'group' column not found in dataset")
        return None, None, None
    
    # Convert labels to numerical format: 'Normal' -> 0, 'Cancer' -> 1
    label_mapping = {'Normal': 0, 'Cancer': 1}
    y = df['group'].map(label_mapping).values
    
    # Remove the 'group' column from features
    feature_df = df.drop(columns=['group'])
    
    # Check for non-numeric columns and remove them
    non_numeric_cols = []
    for col in feature_df.columns:
        if not np.issubdtype(feature_df[col].dtype, np.number):
            non_numeric_cols.append(col)
    
    if non_numeric_cols:
        print(f"Removing {len(non_numeric_cols)} non-numeric columns:")
        print(", ".join(non_numeric_cols[:5]) + ("..." if len(non_numeric_cols) > 5 else ""))
        feature_df = feature_df.drop(columns=non_numeric_cols)
    
    # Check for remaining columns after removal
    if feature_df.shape[1] == 0:
        print("Error: No numeric features remain after removing non-numeric columns")
        return None, None, None
    
    # Get feature names and data matrix
    feature_names = feature_df.columns.tolist()
    X = feature_df.values
    
    print(f"Final dataset shape after preprocessing: {X.shape}")
    
    # Print class distribution
    unique_labels, counts = np.unique(y, return_counts=True)
    print("Class distribution:")
    for label, count in zip(unique_labels, counts):
        label_name = 'Normal' if label == 0 else 'Cancer'
        print(f"  {label_name}: {count} ({count/len(y)*100:.2f}%)")
    
    return X, y, feature_names

def preprocess_data(X, standardize=True):
    """
    Preprocess the data.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data to preprocess
    standardize : bool, default=True
        Whether to standardize the data
        
    Returns:
    --------
    X_processed : array-like, shape (n_samples, n_features)
        Preprocessed data
    scaler : StandardScaler or None
        The scaler used for standardization
    """
    # Handle missing values if any
    X_processed = np.nan_to_num(X)
    
    # Standardize the data
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X_processed)
        print("Data standardized.")
    
    return X_processed, scaler 