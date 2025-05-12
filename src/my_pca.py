import numpy as np

class MyPCA:
    """
    Custom PCA implementation without using scikit-learn.
    """
    
    def __init__(self, n_components=None):
        """
        Initialize the PCA model.
        
        Parameters:
        -----------
        n_components : int, optional
            Number of components to keep. If None, all components are kept.
        """
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.cumulative_explained_variance_ratio_ = None
        
    def fit(self, X):
        """
        Fit the PCA model to the data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        self : object
            Returns the instance itself
        """
        n_samples, n_features = X.shape
        
        # Store mean of each feature
        self.mean_ = np.mean(X, axis=0)
        
        # Center the data
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store components (eigenvectors)
        if self.n_components is None:
            self.n_components = n_features
        
        self.components_ = eigenvectors[:, :self.n_components].T
        
        # Calculate explained variance
        self.explained_variance_ = eigenvalues[:self.n_components]
        
        # Calculate explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        # Calculate cumulative explained variance ratio
        self.cumulative_explained_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)
        
        return self
    
    def transform(self, X):
        """
        Transform the data using PCA.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to transform
            
        Returns:
        --------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data
        """
        # Center the data
        X_centered = X - self.mean_
        
        # Project the data onto the principal components
        X_transformed = np.dot(X_centered, self.components_.T)
        
        return X_transformed
    
    def fit_transform(self, X):
        """
        Fit the model and transform the data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to fit and transform
            
        Returns:
        --------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data
        """
        self.fit(X)
        return self.transform(X) 