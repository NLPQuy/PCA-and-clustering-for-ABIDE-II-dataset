import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import cdist

class EnhancedKMeans:
    """
    Enhanced K-Means clustering implementation optimized for brain imaging data.
    
    Improvements include:
    1. Smart initialization with k-means++ and multiple seeds
    2. Weighted distance metrics suitable for anatomical features
    3. Adaptive learning rates for centroid updates
    4. Outlier handling with median-based updates
    5. Momentum-based updates to escape local minima
    6. Early convergence detection
    7. Support for brain region feature weighting
    """
    
    def __init__(self, n_clusters=2, max_iter=300, tol=1e-6, random_state=None, 
                 n_init=15, feature_weights=None, distance_metric='euclidean',
                 outlier_handling=True, momentum=0.1, early_stopping=True):
        """
        Initialize the Enhanced K-Means clustering model.
        
        Parameters:
        -----------
        n_clusters : int, default=2
            Number of clusters to form and centroids to generate
        max_iter : int, default=300
            Maximum number of iterations
        tol : float, default=1e-6
            Tolerance for stopping criterion
        random_state : int, default=None
            Seed for random number generator
        n_init : int, default=15
            Number of initializations to try
        feature_weights : array-like, default=None
            Weights for each feature (useful for brain region prioritization)
        distance_metric : str, default='euclidean'
            Distance metric to use ('euclidean', 'manhattan', 'cosine', 'rbf')
        outlier_handling : bool, default=True
            Whether to use median-based updates for handling outliers
        momentum : float, default=0.1
            Momentum coefficient for centroid updates
        early_stopping : bool, default=True
            Whether to stop early if no improvement is detected
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init
        self.feature_weights = feature_weights
        self.distance_metric = distance_metric
        self.outlier_handling = outlier_handling
        self.momentum = momentum
        self.early_stopping = early_stopping
        
        # Model attributes
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        self.previous_updates = None
        
    def _init_centroids(self, X):
        """
        Initialize centroids using enhanced k-means++ method.
        
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
        
        # Apply feature weights if provided
        X_weighted = X.copy()
        if self.feature_weights is not None:
            # Check if feature_weights matches the number of features in X
            # If not, we can't use them and should warn the user
            if len(self.feature_weights) == n_features:
                X_weighted = X * self.feature_weights
            else:
                print(f"Warning: feature_weights shape ({len(self.feature_weights)}) doesn't match "
                      f"data features ({n_features}). Using unweighted data for initialization.")
        
        # Choose first centroid randomly from points that are not outliers
        if self.outlier_handling:
            # Simple outlier detection: use points within 3 std devs for first centroid
            means = np.mean(X_weighted, axis=0)
            stds = np.std(X_weighted, axis=0)
            is_outlier = np.any(np.abs(X_weighted - means) > 3 * stds, axis=1)
            candidate_indices = np.where(~is_outlier)[0]
            if len(candidate_indices) > 0:
                first_idx = candidate_indices[np.random.randint(len(candidate_indices))]
            else:
                first_idx = np.random.randint(n_samples)
        else:
            first_idx = np.random.randint(n_samples)
        
        centroids = [X[first_idx]]
        
        # Choose remaining centroids with probability proportional to distance
        for _ in range(1, self.n_clusters):
            # Compute distances to closest centroid
            min_dists = np.min([self._calculate_distance(X_weighted, np.array([c])) for c in centroids], axis=0)
            
            # Square distances for standard k-means++
            min_sq_dists = min_dists ** 2
            
            # Choose next centroid with probability proportional to squared distance
            probs = min_sq_dists / min_sq_dists.sum()
            cumprobs = np.cumsum(probs)
            r = np.random.rand()
            ind = np.searchsorted(cumprobs, r)
            centroids.append(X[ind])
        
        return np.array(centroids)
    
    def _calculate_distance(self, X, centroids):
        """
        Calculate distances from points to centroids using the specified metric.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data points
        centroids : array-like, shape (n_centroids, n_features)
            Centroid points
            
        Returns:
        --------
        distances : array, shape (n_samples, n_centroids)
            Distances from each point to each centroid
        """
        if self.distance_metric == 'euclidean':
            return cdist(X, centroids, 'euclidean')
        elif self.distance_metric == 'manhattan':
            return cdist(X, centroids, 'cityblock')
        elif self.distance_metric == 'cosine':
            return cdist(X, centroids, 'cosine')
        elif self.distance_metric == 'rbf':
            # RBF kernel distances (1 - kernel value)
            kernel_vals = rbf_kernel(X, centroids, gamma=1.0)
            return 1.0 - kernel_vals
        else:
            # Default to euclidean
            return cdist(X, centroids, 'euclidean')
        
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
            
        # Apply feature weights if provided
        X_weighted = X.copy()
        centroids_weighted = centroids.copy()
        
        if self.feature_weights is not None:
            # Check if feature_weights matches the number of features in X
            if len(self.feature_weights) == X.shape[1]:
                X_weighted = X * self.feature_weights
                centroids_weighted = centroids * self.feature_weights
            else:
                print(f"Warning: feature_weights shape ({len(self.feature_weights)}) doesn't match "
                      f"data features ({X.shape[1]}). Using unweighted data for assignments.")
        
        # Calculate distances to centroids
        distances = self._calculate_distance(X_weighted, centroids_weighted)
        
        # Assign to nearest centroid
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X, labels, current_centroids, iteration):
        """
        Update centroids with advanced techniques like momentum and outlier handling.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data
        labels : array-like, shape (n_samples,)
            Cluster assignments
        current_centroids : array-like, shape (n_clusters, n_features)
            Current centroid positions
        iteration : int
            Current iteration number
            
        Returns:
        --------
        centroids : array, shape (n_clusters, n_features)
            Updated centroids
        centroid_shifts : array, shape (n_clusters, n_features)
            Change in centroid positions
        """
        new_centroids = np.zeros_like(current_centroids)
        
        # Calculate adaptive learning rate (decreases with iterations)
        learning_rate = 1.0 / (1.0 + 0.1 * iteration)
        
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            
            if len(cluster_points) > 0:
                if self.outlier_handling and len(cluster_points) > 5:
                    # Use median for outlier robustness
                    new_position = np.median(cluster_points, axis=0)
                else:
                    # Use mean for standard update
                    new_position = np.mean(cluster_points, axis=0)
                
                # Apply momentum if not first iteration
                if self.previous_updates is not None:
                    update_vector = new_position - current_centroids[i]
                    update_vector = learning_rate * update_vector + self.momentum * self.previous_updates[i]
                    new_centroids[i] = current_centroids[i] + update_vector
                else:
                    new_centroids[i] = new_position
            else:
                # Handle empty clusters
                new_centroids[i] = current_centroids[i]
        
        # Calculate centroid shifts
        centroid_shifts = new_centroids - current_centroids
        
        # Save updates for momentum
        self.previous_updates = centroid_shifts
        
        return new_centroids, centroid_shifts
    
    def _handle_empty_clusters(self, X, labels, centroids):
        """
        Handle empty clusters more intelligently.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data
        labels : array-like, shape (n_samples,)
            Cluster assignments
        centroids : array-like, shape (n_clusters, n_features)
            Current centroids
            
        Returns:
        --------
        centroids : array, shape (n_clusters, n_features)
            Updated centroids with empty clusters handled
        """
        empty_clusters = []
        for j in range(self.n_clusters):
            if np.sum(labels == j) == 0:
                empty_clusters.append(j)
        
        if empty_clusters:
            # Find distances from points to centroids
            distances = self._calculate_distance(X, centroids)
            
            for empty_idx in empty_clusters:
                # Find the cluster with the most dispersed points
                cluster_indices = []
                for j in range(self.n_clusters):
                    if j not in empty_clusters and np.sum(labels == j) > 1:
                        cluster_indices.append(j)
                
                if cluster_indices:
                    # Calculate dispersion for each non-empty cluster
                    dispersions = []
                    for j in cluster_indices:
                        points_in_cluster = X[labels == j]
                        if len(points_in_cluster) > 1:
                            # Calculate average distance to cluster centroid
                            dispersion = np.mean(distances[labels == j, j])
                            dispersions.append((j, dispersion))
                    
                    if dispersions:
                        # Find the most dispersed cluster
                        most_dispersed = max(dispersions, key=lambda x: x[1])[0]
                        
                        # Find the point furthest from its centroid
                        points_in_cluster = X[labels == most_dispersed]
                        dist_to_centroid = distances[labels == most_dispersed, most_dispersed]
                        furthest_point_idx = np.argmax(dist_to_centroid)
                        
                        # Use this point as the new centroid for the empty cluster
                        centroids[empty_idx] = points_in_cluster[furthest_point_idx]
                    else:
                        # Fallback: pick a random point
                        idx = np.random.randint(X.shape[0])
                        centroids[empty_idx] = X[idx]
                else:
                    # Fallback: pick a random point
                    idx = np.random.randint(X.shape[0])
                    centroids[empty_idx] = X[idx]
        
        return centroids
        
    def fit(self, X):
        """
        Fit the Enhanced K-Means model to the data.
        
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
        best_n_iter = None
        
        # Try multiple initializations
        for init in range(self.n_init):
            # Initialize centroids
            centroids = self._init_centroids(X)
            
            # Reset momentum updates
            self.previous_updates = None
            
            # Track centroid shifts for early stopping
            prev_inertia = np.inf
            no_improvement_count = 0
            
            # Main loop
            for i in range(self.max_iter):
                # Assign points to clusters
                labels = self._assign_clusters(X, centroids)
                
                # Handle empty clusters if any
                centroids = self._handle_empty_clusters(X, labels, centroids)
                
                # Update centroids
                new_centroids, centroid_shifts = self._update_centroids(X, labels, centroids, i)
                
                # Calculate inertia
                inertia = self._calculate_inertia(X, labels, new_centroids)
                
                # Check for early stopping
                if self.early_stopping:
                    if inertia < prev_inertia:
                        # Reset counter if improving
                        no_improvement_count = 0
                    else:
                        # Increment counter if not improving
                        no_improvement_count += 1
                        
                    # Stop if no improvement for several iterations
                    if no_improvement_count >= 5:
                        break
                
                # Update centroids
                centroids = new_centroids
                prev_inertia = inertia
                
                # Check convergence
                if np.max(np.abs(centroid_shifts)) < self.tol:
                    break
            
            # Calculate final inertia for this initialization
            labels = self._assign_clusters(X, centroids)
            inertia = self._calculate_inertia(X, labels, centroids)
            
            # Keep track of the best result
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centroids = centroids
                best_n_iter = i + 1
        
        # Set final model attributes
        self.centroids_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        
        return self
    
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
            
        # Apply feature weights if provided
        X_weighted = X.copy()
        centroids_weighted = centroids.copy()
        
        if self.feature_weights is not None:
            # Check if feature_weights matches the number of features in X
            if len(self.feature_weights) == X.shape[1]:
                X_weighted = X * self.feature_weights
                centroids_weighted = centroids * self.feature_weights
            else:
                # Use unweighted data silently for inertia calculation
                pass
        
        inertia = 0.0
        if self.distance_metric == 'euclidean':
            for i in range(len(centroids)):
                cluster_points = X_weighted[labels == i]
                if len(cluster_points) > 0:
                    squared_dists = np.sum((cluster_points - centroids_weighted[i])**2, axis=1)
                    inertia += np.sum(squared_dists)
        else:
            # For other metrics, use the distance calculation function
            distances = self._calculate_distance(X_weighted, centroids_weighted)
            for i in range(len(centroids)):
                cluster_points_indices = np.where(labels == i)[0]
                if len(cluster_points_indices) > 0:
                    inertia += np.sum(distances[cluster_points_indices, i])
        
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
    
    def fit_predict(self, X):
        """
        Fit the model and predict cluster for each sample in X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        labels : array, shape (n_samples,)
            Cluster assignments
        """
        self.fit(X)
        return self.labels_
    
    def get_feature_importance(self, X):
        """
        Calculate feature importance based on the separation they provide.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data
            
        Returns:
        --------
        importance : array, shape (n_features,)
            Importance score for each feature
        """
        if self.centroids_ is None:
            raise ValueError("Model must be fitted before calculating feature importance")
            
        n_features = X.shape[1]
        importance = np.zeros(n_features)
        
        # Calculate the variance of each feature for each cluster
        for i in range(self.n_clusters):
            cluster_points = X[self.labels_ == i]
            if len(cluster_points) > 1:
                # Calculate the variance of each feature in this cluster
                cluster_var = np.var(cluster_points, axis=0)
                # Inverse of variance indicates importance (lower variance = more important)
                importance += 1.0 / (cluster_var + 1e-10)
        
        # Normalize importance
        importance = importance / np.sum(importance)
        
        return importance 