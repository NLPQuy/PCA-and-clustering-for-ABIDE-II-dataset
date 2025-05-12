import numpy as np

class MyKMeans:
    """
    Custom K-Means clustering implementation without using scikit-learn.
    """
    
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None, n_init=10):
        """
        Initialize the K-Means clustering model.
        
        Parameters:
        -----------
        n_clusters : int, default=8
            Number of clusters to form and centroids to generate
        max_iter : int, default=300
            Maximum number of iterations
        tol : float, default=1e-4
            Tolerance for stopping criterion
        random_state : int, default=None
            Seed for random number generator
        n_init : int, default=10
            Number of initializations to try
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        
    def _init_centroids(self, X):
        """
        Initialize centroids using k-means++ method.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to initialize centroids from
            
        Returns:
        --------
        centroids : array, shape (n_clusters, n_features)
            Initial centroids
        """
        n_samples, n_features = X.shape
        
        # Choose first centroid randomly
        centroids = [X[np.random.randint(n_samples)]]
        
        # Choose remaining centroids
        for _ in range(1, self.n_clusters):
            # Compute squared distances to closest centroid
            min_sq_dists = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
            
            # Choose next centroid with probability proportional to squared distance
            probs = min_sq_dists / min_sq_dists.sum()
            cumprobs = np.cumsum(probs)
            r = np.random.rand()
            ind = np.searchsorted(cumprobs, r)
            centroids.append(X[ind])
        
        return np.array(centroids)
        
    def fit(self, X):
        """
        Fit the K-Means model to the data.
        
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
        
        if n_samples < self.n_clusters:
            print(f"Warning: n_samples={n_samples} < n_clusters={self.n_clusters}. Setting n_clusters to {n_samples}")
            self.n_clusters = n_samples
        
        # Set random seed if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        best_inertia = np.inf
        best_labels = None
        best_centroids = None
        
        # Try multiple initializations
        for init in range(self.n_init):
            # Initialize centroids using k-means++
            centroids = self._init_centroids(X)
            
            # Initialize previous centroids for convergence check
            prev_centroids = np.zeros_like(centroids)
            
            # Main loop
            for i in range(self.max_iter):
                # Assign points to clusters
                labels = self._assign_clusters(X, centroids)
                
                # Save previous centroids
                prev_centroids = np.copy(centroids)
                
                # Update centroids
                for j in range(self.n_clusters):
                    cluster_points = X[labels == j]
                    if len(cluster_points) > 0:
                        centroids[j] = np.mean(cluster_points, axis=0)
                
                # Check for empty clusters and reinitialize if needed
                empty_clusters = []
                for j in range(self.n_clusters):
                    if np.sum(labels == j) == 0:
                        empty_clusters.append(j)
                
                if empty_clusters:
                    # Find the cluster with most points
                    largest_cluster = np.argmax([np.sum(labels == j) for j in range(self.n_clusters)])
                    points_in_largest = X[labels == largest_cluster]
                    
                    # For each empty cluster, pick a point from the largest cluster
                    for empty_idx in empty_clusters:
                        if len(points_in_largest) > 0:
                            idx = np.random.randint(len(points_in_largest))
                            centroids[empty_idx] = points_in_largest[idx]
                            # Remove the point to avoid duplicates if there are multiple empty clusters
                            points_in_largest = np.delete(points_in_largest, idx, axis=0)
                
                # Check convergence
                if np.linalg.norm(centroids - prev_centroids) < self.tol:
                    break
            
            # Calculate inertia for this initialization
            inertia = self._calculate_inertia(X, labels, centroids)
            
            # Keep track of the best result
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centroids = centroids
        
        # Set final model attributes
        self.centroids_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        
        return self
    
    def _assign_clusters(self, X, centroids=None):
        """
        Assign each data point to the closest centroid.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data
        centroids : array-like, shape (n_clusters, n_features), optional
            Centroids to use. If None, use the fitted centroids
            
        Returns:
        --------
        labels : array, shape (n_samples,)
            Cluster assignments
        """
        if centroids is None:
            centroids = self.centroids_
            
        # Calculate distances to centroids
        distances = np.zeros((X.shape[0], len(centroids)))
        for i, centroid in enumerate(centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        
        # Assign to nearest centroid
        return np.argmin(distances, axis=1)
    
    def _calculate_inertia(self, X, labels=None, centroids=None):
        """
        Calculate sum of squared distances to closest centroid.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data
        labels : array-like, shape (n_samples,), optional
            Cluster assignments. If None, use the fitted labels
        centroids : array-like, shape (n_clusters, n_features), optional
            Centroids to use. If None, use the fitted centroids
            
        Returns:
        --------
        inertia : float
            Sum of squared distances
        """
        if labels is None:
            labels = self.labels_
        
        if centroids is None:
            centroids = self.centroids_
            
        inertia = 0.0
        for i in range(len(centroids)):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum(np.linalg.norm(cluster_points - centroids[i], axis=1) ** 2)
        
        return inertia
    
    def predict(self, X):
        """
        Predict the closest cluster for each sample in X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            New data
            
        Returns:
        --------
        labels : array, shape (n_samples,)
            Cluster assignments
        """
        return self._assign_clusters(X) 