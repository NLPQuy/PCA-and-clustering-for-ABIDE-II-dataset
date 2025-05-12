import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

class MyAgglomerativeClustering:
    """
    Thuật toán phân cụm phân cấp (Hierarchical) với chiến lược Agglomerative (bottom-up)
    """
    
    def __init__(self, n_clusters=2, linkage='ward'):
        """
        Khởi tạo thuật toán phân cụm phân cấp Agglomerative.
        
        Parameters:
        -----------
        n_clusters : int, default=2
            Số lượng cụm cần tạo
        linkage : {'ward', 'complete', 'average', 'single'}, default='ward'
            Phương pháp tính khoảng cách giữa các cụm
            - 'ward': tối thiểu hóa phương sai trong các cụm
            - 'complete': khoảng cách lớn nhất giữa các điểm của hai cụm
            - 'average': khoảng cách trung bình giữa các điểm của hai cụm
            - 'single': khoảng cách nhỏ nhất giữa các điểm của hai cụm
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
        
    def _compute_distance_matrix(self, X):
        """Tính ma trận khoảng cách giữa các điểm dữ liệu"""
        n_samples = X.shape[0]
        dist_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.sqrt(np.sum((X[i] - X[j]) ** 2))
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
                
        return dist_matrix
    
    def _compute_cluster_distance(self, cluster1, cluster2, distances, method):
        """Tính khoảng cách giữa hai cụm dựa trên phương pháp đã chọn"""
        distances_between_clusters = []
        
        for i in cluster1:
            for j in cluster2:
                distances_between_clusters.append(distances[i, j])
        
        if method == 'single':
            return np.min(distances_between_clusters)
        elif method == 'complete':
            return np.max(distances_between_clusters)
        elif method == 'average':
            return np.mean(distances_between_clusters)
        else:  # ward - đơn giản hóa
            return np.mean(distances_between_clusters)
    
    def fit(self, X):
        """
        Thực hiện phân cụm phân cấp trên dữ liệu.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Dữ liệu đầu vào
            
        Returns:
        --------
        self : object
            Trả về đối tượng phân cụm đã được huấn luyện
        """
        n_samples = X.shape[0]
        
        # Ban đầu, mỗi điểm dữ liệu là một cụm riêng biệt
        current_clusters = [[i] for i in range(n_samples)]
        
        # Tính ma trận khoảng cách giữa các điểm
        distances = self._compute_distance_matrix(X)
        
        # Hợp nhất các cụm cho đến khi đạt được số lượng cụm mong muốn
        while len(current_clusters) > self.n_clusters:
            min_distance = float('inf')
            merge_i, merge_j = -1, -1
            
            # Tìm cặp cụm có khoảng cách nhỏ nhất
            for i in range(len(current_clusters)):
                for j in range(i + 1, len(current_clusters)):
                    dist = self._compute_cluster_distance(
                        current_clusters[i], 
                        current_clusters[j], 
                        distances, 
                        self.linkage
                    )
                    
                    if dist < min_distance:
                        min_distance = dist
                        merge_i, merge_j = i, j
            
            # Hợp nhất hai cụm gần nhất
            current_clusters[merge_i].extend(current_clusters[merge_j])
            current_clusters.pop(merge_j)
        
        # Gán nhãn cho từng điểm dữ liệu
        labels = np.zeros(n_samples, dtype=int)
        for i, cluster in enumerate(current_clusters):
            for sample_idx in cluster:
                labels[sample_idx] = i
        
        self.labels_ = labels
        return self

class MyDBSCAN:
    """
    Thuật toán phân cụm DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    """
    
    def __init__(self, eps=0.5, min_samples=5):
        """
        Khởi tạo thuật toán phân cụm DBSCAN.
        
        Parameters:
        -----------
        eps : float, default=0.5
            Bán kính hàng xóm (epsilon) - khoảng cách tối đa giữa hai mẫu để một mẫu được coi là trong vùng lân cận của mẫu khác
        min_samples : int, default=5
            Số lượng mẫu tối thiểu trong một vùng lân cận để một điểm được coi là core point
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        
    def fit(self, X):
        """
        Thực hiện phân cụm DBSCAN trên dữ liệu.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Dữ liệu đầu vào
            
        Returns:
        --------
        self : object
            Trả về đối tượng phân cụm đã được huấn luyện
        """
        n_samples = X.shape[0]
        
        # Tìm các điểm lân cận trong bán kính eps
        neighbors = NearestNeighbors(radius=self.eps).fit(X)
        distances, indices = neighbors.radius_neighbors(X)
        
        # Khởi tạo nhãn ban đầu (không kết nối = -1)
        labels = np.full(n_samples, -1)
        
        # Theo dõi các nhãn đã được thăm
        visited = np.zeros(n_samples, dtype=bool)
        
        # Bắt đầu với cluster_id = 0
        cluster_id = 0
        
        # Xem xét từng điểm
        for i in range(n_samples):
            # Bỏ qua các điểm đã thăm
            if visited[i]:
                continue
            
            # Đánh dấu điểm hiện tại là đã thăm
            visited[i] = True
            
            # Lấy các điểm lân cận
            neighbors_indices = indices[i]
            
            # Nếu không đủ láng giềng, đánh dấu là nhiễu
            if len(neighbors_indices) < self.min_samples:
                labels[i] = -1  # Noise
                continue
            
            # Khởi tạo cụm mới
            labels[i] = cluster_id
            
            # Các điểm lân cận cần được xử lý
            neighbors_to_process = list(neighbors_indices)
            neighbors_to_process.remove(i)  # Loại bỏ điểm hiện tại
            
            # Mở rộng cụm
            while neighbors_to_process:
                neighbor = neighbors_to_process.pop(0)
                
                # Nếu chưa thăm
                if not visited[neighbor]:
                    # Đánh dấu là đã thăm
                    visited[neighbor] = True
                    
                    # Lấy láng giềng của láng giềng
                    new_neighbors = indices[neighbor]
                    
                    # Nếu là core point, thêm láng giềng vào danh sách xử lý
                    if len(new_neighbors) >= self.min_samples:
                        for new_neighbor in new_neighbors:
                            if new_neighbor not in neighbors_to_process and not visited[new_neighbor]:
                                neighbors_to_process.append(new_neighbor)
                
                # Thêm vào cụm hiện tại nếu chưa được gán
                if labels[neighbor] == -1:
                    labels[neighbor] = cluster_id
            
            # Tăng ID cụm cho cụm tiếp theo
            cluster_id += 1
        
        self.labels_ = labels
        return self

class MyMeanShift:
    """
    Thuật toán phân cụm Mean Shift
    """
    
    def __init__(self, bandwidth=None, max_iter=300, tol=1e-3):
        """
        Khởi tạo thuật toán phân cụm Mean Shift.
        
        Parameters:
        -----------
        bandwidth : float, default=None
            Bán kính kernel cho phân cụm. Nếu None, sẽ tự động tính toán.
        max_iter : int, default=300
            Số lần lặp tối đa
        tol : float, default=1e-3
            Dung sai để kiểm tra hội tụ
        """
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None
        self.labels_ = None
        
    def _estimate_bandwidth(self, X):
        """Ước tính bandwidth nếu không được cung cấp"""
        # Sử dụng heuristic đơn giản: khoảng cách trung bình giữa các điểm
        distances = cdist(X[:min(100, X.shape[0])], X[:min(100, X.shape[0])])
        return np.median(distances) / 2
        
    def _gaussian_kernel(self, distance, bandwidth):
        """Tính trọng số kernel cho một khoảng cách"""
        return np.exp(-0.5 * (distance / bandwidth) ** 2)
    
    def fit(self, X):
        """
        Thực hiện phân cụm Mean Shift trên dữ liệu.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Dữ liệu đầu vào
            
        Returns:
        --------
        self : object
            Trả về đối tượng phân cụm đã được huấn luyện
        """
        n_samples, n_features = X.shape
        
        # Ước tính bandwidth nếu không được cung cấp
        if self.bandwidth is None:
            self.bandwidth = self._estimate_bandwidth(X)
            print(f"Estimated bandwidth: {self.bandwidth}")
        
        # Khởi tạo các điểm cũ là dữ liệu đầu vào
        points = X.copy()
        
        # Áp dụng mean shift cho từng điểm
        for iteration in range(self.max_iter):
            # Lưu lại các điểm cũ để kiểm tra hội tụ
            old_points = points.copy()
            
            # Cập nhật vị trí của từng điểm
            for i in range(n_samples):
                # Tính khoảng cách từ điểm hiện tại đến tất cả các điểm trong X
                distances = np.sqrt(np.sum((X - points[i]) ** 2, axis=1))
                
                # Áp dụng kernel để tính trọng số
                weights = self._gaussian_kernel(distances, self.bandwidth)
                
                # Cập nhật vị trí điểm bằng trung bình có trọng số
                points[i] = np.sum(X * weights[:, np.newaxis], axis=0) / np.sum(weights)
            
            # Kiểm tra hội tụ
            shift = np.sqrt(np.sum((points - old_points) ** 2, axis=1))
            if np.all(shift < self.tol):
                break
        
        # Xác định các tâm cụm bằng cách gom nhóm các điểm rất gần nhau
        cluster_centers = []
        labels = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Kiểm tra xem điểm này có gần các tâm cụm hiện có không
            closest_center = None
            min_distance = self.bandwidth / 2  # Ngưỡng để gộp tâm cụm
            
            for j, center in enumerate(cluster_centers):
                distance = np.sqrt(np.sum((points[i] - center) ** 2))
                if distance < min_distance:
                    closest_center = j
                    min_distance = distance
            
            if closest_center is not None:
                # Gán điểm này vào cụm gần nhất
                labels[i] = closest_center
            else:
                # Tạo một tâm cụm mới
                cluster_centers.append(points[i])
                labels[i] = len(cluster_centers) - 1
        
        self.cluster_centers_ = np.array(cluster_centers)
        self.labels_ = labels.astype(int)
        return self

class MyAffinityPropagation:
    """
    Thuật toán phân cụm Affinity Propagation
    """
    
    def __init__(self, damping=0.5, max_iter=200, convergence_iter=15, preference=None):
        """
        Khởi tạo thuật toán phân cụm Affinity Propagation.
        
        Parameters:
        -----------
        damping : float, default=0.5
            Hệ số giảm chấn (0.5 <= damping < 1.0)
        max_iter : int, default=200
            Số lần lặp tối đa
        convergence_iter : int, default=15
            Số lần lặp để xác định hội tụ
        preference : array-like, default=None
            Preference cho từng điểm. Nếu None, sẽ sử dụng median của similarity.
        """
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.preference = preference
        self.cluster_centers_indices_ = None
        self.cluster_centers_ = None
        self.labels_ = None
    
    def fit(self, X):
        """
        Thực hiện phân cụm Affinity Propagation trên dữ liệu.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Dữ liệu đầu vào
            
        Returns:
        --------
        self : object
            Trả về đối tượng phân cụm đã được huấn luyện
        """
        n_samples = X.shape[0]
        
        # Tính ma trận similarity (sử dụng khoảng cách âm giữa các điểm)
        S = -cdist(X, X, 'sqeuclidean')
        
        # Nếu preference không được cung cấp, sử dụng median của similarity
        if self.preference is None:
            preference = np.median(S)
        else:
            preference = self.preference
        
        # Đặt preference trên đường chéo
        np.fill_diagonal(S, preference)
        
        # Khởi tạo messages
        A = np.zeros((n_samples, n_samples))  # Availability
        R = np.zeros((n_samples, n_samples))  # Responsibility
        
        # Lưu trữ trạng thái cũ để kiểm tra hội tụ
        old_labels = np.zeros(n_samples)
        convergence_count = 0
        
        # Main loop
        for iteration in range(self.max_iter):
            # Cập nhật responsibilities
            R_old = R.copy()
            
            # R(i,k) = S(i,k) - max_{k' != k} [A(i,k') + S(i,k')]
            AS = A + S
            idx = np.arange(n_samples)
            tmp = AS.copy()
            
            for i in range(n_samples):
                # Tìm giá trị lớn nhất và lớn thứ hai
                tmp[i, idx] = -np.inf
                first_max_idx = np.argmax(AS[i])
                tmp[i, first_max_idx] = AS[i, first_max_idx]
                tmp[i, idx] = -np.inf
                
                # Tính responsibilities
                R[i] = S[i] - np.max(AS[i])
            
            # Áp dụng damping
            R = self.damping * R_old + (1 - self.damping) * R
            
            # Cập nhật availabilities
            A_old = A.copy()
            
            # A(i,k) = min [0, R(k,k) + sum_{i' != i,k} max(0, R(i',k))]
            # A(k,k) = sum_{i' != k} max(0, R(i',k))
            
            # Tính max(0, R)
            R_pos = np.maximum(0, R)
            
            # Cập nhật A(k,k)
            for k in range(n_samples):
                A[k, k] = np.sum(R_pos[:, k]) - R_pos[k, k]
            
            # Cập nhật A(i,k) cho i != k
            for i in range(n_samples):
                for k in range(n_samples):
                    if i != k:
                        A[i, k] = min(0, R[k, k] + np.sum(R_pos[:, k]) - R_pos[i, k] - R_pos[k, k])
            
            # Áp dụng damping
            A = self.damping * A_old + (1 - self.damping) * A
            
            # Xác định exemplars và labels
            E = A + R
            labels = np.argmax(E, axis=1)
            
            # Kiểm tra hội tụ
            if np.array_equal(labels, old_labels):
                convergence_count += 1
                if convergence_count >= self.convergence_iter:
                    break
            else:
                convergence_count = 0
            
            old_labels = labels.copy()
        
        # Tìm các exemplars (cluster centers)
        cluster_centers_indices = np.unique(labels)
        n_clusters = len(cluster_centers_indices)
        
        # Gán exemplars làm trung tâm cụm
        cluster_centers = X[cluster_centers_indices]
        
        self.cluster_centers_indices_ = cluster_centers_indices
        self.cluster_centers_ = cluster_centers
        self.labels_ = labels
        
        return self

def find_optimal_clusters(X, method='kmeans', metric='silhouette', max_clusters=10):
    """
    Tìm số lượng cụm tối ưu sử dụng phương pháp và thước đo được chỉ định.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Dữ liệu đầu vào
    method : {'kmeans', 'agglomerative', 'dbscan'}, default='kmeans'
        Phương pháp phân cụm muốn sử dụng
    metric : {'silhouette', 'elbow', 'gap'}, default='silhouette'
        Thước đo để đánh giá số lượng cụm
    max_clusters : int, default=10
        Số lượng cụm tối đa để thử
        
    Returns:
    --------
    optimal_n_clusters : int
        Số lượng cụm tối ưu
    scores : list
        Danh sách các điểm đánh giá cho các số lượng cụm khác nhau
    """
    from src.my_kmeans import MyKMeans
    
    # Phải có ít nhất 2 cụm
    min_clusters = 2
    
    # Đảm bảo số lượng cụm không vượt quá số mẫu
    max_clusters = min(max_clusters, X.shape[0] - 1)
    
    scores = []
    cluster_range = range(min_clusters, max_clusters + 1)
    
    if metric == 'silhouette':
        for n_clusters in cluster_range:
            # Khởi tạo mô hình phân cụm theo phương pháp được chọn
            if method == 'kmeans':
                model = MyKMeans(n_clusters=n_clusters, random_state=42)
            elif method == 'agglomerative':
                model = MyAgglomerativeClustering(n_clusters=n_clusters)
            else:
                raise ValueError(f"Phương pháp {method} không được hỗ trợ")
            
            # Huấn luyện mô hình
            model.fit(X)
            
            # Tính điểm silhouette
            if len(np.unique(model.labels_)) > 1:  # Cần ít nhất 2 cụm khác nhau
                score = silhouette_score(X, model.labels_)
            else:
                score = -1  # Không chia được thành đủ 2 cụm
            
            scores.append(score)
            print(f"Số cụm: {n_clusters}, Silhouette Score: {score:.4f}")
            
        # Chọn số lượng cụm có điểm silhouette cao nhất
        optimal_n_clusters = cluster_range[np.argmax(scores)]
        
    elif metric == 'elbow':
        for n_clusters in cluster_range:
            # Chỉ hỗ trợ KMeans cho Elbow method
            if method == 'kmeans':
                model = MyKMeans(n_clusters=n_clusters, random_state=42)
                model.fit(X)
                scores.append(model.inertia_)
                print(f"Số cụm: {n_clusters}, Inertia: {model.inertia_:.4f}")
            else:
                raise ValueError(f"Phương pháp {method} không được hỗ trợ cho Elbow method")
            
        # Sử dụng giá trị gần "khuỷu tay" (heuristic)
        diffs = np.diff(scores)
        diffs_of_diffs = np.diff(diffs)
        optimal_n_clusters = cluster_range[np.argmin(diffs_of_diffs) + 1]
    
    else:
        raise ValueError(f"Thước đo {metric} không được hỗ trợ")
    
    # Vẽ đồ thị
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, scores, 'bo-')
    plt.axvline(x=optimal_n_clusters, color='r', linestyle='--')
    plt.xlabel('Số lượng cụm')
    plt.ylabel('Điểm đánh giá' if metric == 'silhouette' else 'Inertia')
    plt.title(f'Đánh giá số lượng cụm tối ưu ({metric})')
    plt.grid(True)
    plt.savefig(f'plots/optimal_clusters_{method}_{metric}.png')
    plt.close()
    
    return optimal_n_clusters, scores 