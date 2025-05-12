import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class MyGMM(BaseEstimator, ClusterMixin):
    """
    Thuật toán phân cụm Gaussian Mixture Model (GMM).
    
    GMM giả định rằng dữ liệu được sinh ra từ hỗn hợp các phân phối chuẩn (Gaussian)
    nhiều chiều. Thuật toán sử dụng Expectation-Maximization (EM) để tìm
    các tham số tối ưu cho mô hình.
    """
    
    def __init__(self, n_components=2, max_iter=100, tol=1e-3, random_state=None, 
                 reg_covar=1e-6, init_params='kmeans'):
        """
        Khởi tạo thuật toán GMM.
        
        Parameters:
        -----------
        n_components : int, default=2
            Số lượng phân phối Gaussian (số lượng cụm)
        max_iter : int, default=100
            Số lượng vòng lặp tối đa của thuật toán EM
        tol : float, default=1e-3
            Ngưỡng hội tụ - thuật toán dừng khi log-likelihood không thay đổi quá giá trị này
        random_state : int hoặc None, default=None
            Seed cho số ngẫu nhiên, đảm bảo kết quả có thể lặp lại
        reg_covar : float, default=1e-6
            Giá trị điều chuẩn thêm vào đường chéo của ma trận hiệp phương sai để đảm bảo tính ổn định
        init_params : str, default='kmeans'
            Phương pháp khởi tạo các tham số: 'kmeans' hoặc 'random'
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.reg_covar = reg_covar
        self.init_params = init_params
        
        # Các tham số của mô hình
        self.weights_ = None  # Trọng số của các thành phần
        self.means_ = None    # Vector trung bình của các thành phần
        self.covariances_ = None  # Ma trận hiệp phương sai của các thành phần
        self.labels_ = None   # Nhãn cụm cho dữ liệu
        self.responsibilities_ = None  # Xác suất hậu nghiệm
        self.converged_ = False  # Trạng thái hội tụ
        self.n_iter_ = 0      # Số vòng lặp đã thực hiện
        self.lower_bound_ = -np.inf  # Giá trị log-likelihood
        
    def _initialize_parameters(self, X):
        """
        Khởi tạo các tham số cho mô hình GMM
        """
        n_samples, n_features = X.shape
        
        # Thiết lập seed ngẫu nhiên nếu được cung cấp
        rng = np.random.RandomState(self.random_state)
        
        # Khởi tạo trọng số đều cho mỗi thành phần
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        # Khởi tạo các vector trung bình
        if self.init_params == 'kmeans':
            # Sử dụng K-means để khởi tạo (hiệu quả hơn)
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_components, n_init=1, 
                          random_state=self.random_state)
            kmeans.fit(X)
            self.means_ = kmeans.cluster_centers_
        else:
            # Khởi tạo ngẫu nhiên từ các điểm dữ liệu
            indices = rng.choice(n_samples, self.n_components, replace=False)
            self.means_ = X[indices]
        
        # Khởi tạo ma trận hiệp phương sai
        # Sử dụng ma trận đơn vị nhân với phương sai của dữ liệu
        covariances = []
        for _ in range(self.n_components):
            # Tạo ma trận hiệp phương sai đường chéo dựa trên phương sai của dữ liệu
            cov = np.diag(np.var(X, axis=0) + self.reg_covar)
            covariances.append(cov)
        self.covariances_ = np.array(covariances)
    
    def _e_step(self, X):
        """
        Expectation step: tính xác suất hậu nghiệm (responsibilities)
        """
        n_samples, _ = X.shape
        
        # Tính log-density cho mỗi mẫu dưới mỗi phân phối Gaussian
        weighted_log_prob = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            # Tính log-likelihood của dữ liệu dưới phân phối Gaussian k
            try:
                # Sử dụng multivariate_normal.logpdf để tính log-likelihood
                log_prob = multivariate_normal.logpdf(
                    X, mean=self.means_[k], cov=self.covariances_[k]
                )
                # Thêm log của trọng số
                weighted_log_prob[:, k] = log_prob + np.log(self.weights_[k])
            except:
                # Xử lý lỗi nếu ma trận hiệp phương sai không xác định dương
                # (không khả nghịch)
                weighted_log_prob[:, k] = -np.inf
        
        # Loại bỏ các giá trị quá nhỏ để tránh underflow
        log_prob_norm = np.zeros(n_samples)
        
        # Sử dụng log-sum-exp trick để tính logarithm của tổng một cách ổn định số học
        log_prob_norm = np.log(np.sum(np.exp(weighted_log_prob - np.max(weighted_log_prob, axis=1, keepdims=True)), axis=1))
        log_prob_norm += np.max(weighted_log_prob, axis=1)
        
        # Xác suất hậu nghiệm (responsibilities) cho mỗi thành phần
        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        resp = np.exp(log_resp)
        
        # Cập nhật log-likelihood
        self.lower_bound_ = np.mean(log_prob_norm)
        
        return resp
    
    def _m_step(self, X, resp):
        """
        Maximization step: cập nhật các tham số của mô hình
        """
        n_samples, _ = X.shape
        
        # Cập nhật trọng số: trung bình của xác suất hậu nghiệm cho mỗi thành phần
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        self.weights_ = nk / n_samples
        
        # Cập nhật các vector trung bình
        self.means_ = np.dot(resp.T, X) / nk[:, np.newaxis]
        
        # Cập nhật ma trận hiệp phương sai
        for k in range(self.n_components):
            # Tính hiệp phương sai dựa trên xác suất hậu nghiệm
            diff = X - self.means_[k]
            weighted_diff = diff * np.sqrt(resp[:, k, np.newaxis])
            
            # Cập nhật ma trận hiệp phương sai
            self.covariances_[k] = np.dot(weighted_diff.T, weighted_diff) / nk[k]
            
            # Thêm giá trị điều chuẩn vào đường chéo để đảm bảo tính ổn định
            self.covariances_[k].flat[::self.covariances_[k].shape[0] + 1] += self.reg_covar
    
    def fit(self, X):
        """
        Huấn luyện mô hình GMM trên dữ liệu X sử dụng thuật toán EM.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Dữ liệu đầu vào
            
        Returns:
        --------
        self : object
            Trả về đối tượng đã được huấn luyện
        """
        # Khởi tạo các tham số
        self._initialize_parameters(X)
        
        # Theo dõi sự hội tụ thông qua log-likelihood
        prev_lower_bound = -np.inf
        
        # Thuật toán EM
        for n_iter in range(1, self.max_iter + 1):
            # E-step: tính xác suất hậu nghiệm
            resp = self._e_step(X)
            
            # M-step: cập nhật các tham số
            self._m_step(X, resp)
            
            # Kiểm tra sự hội tụ
            change = self.lower_bound_ - prev_lower_bound
            
            if abs(change) < self.tol:
                self.converged_ = True
                break
                
            prev_lower_bound = self.lower_bound_
        
        # Lưu số vòng lặp đã thực hiện
        self.n_iter_ = n_iter
        
        # Lưu xác suất hậu nghiệm cuối cùng
        self.responsibilities_ = resp
        
        # Gán nhãn cho dữ liệu (nhãn của thành phần có xác suất hậu nghiệm cao nhất)
        self.labels_ = resp.argmax(axis=1)
        
        return self
    
    def predict(self, X):
        """
        Dự đoán nhãn cụm cho dữ liệu mới.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Dữ liệu mới cần dự đoán
            
        Returns:
        --------
        labels : array-like, shape (n_samples,)
            Nhãn cụm được dự đoán cho mỗi mẫu
        """
        if self.means_ is None:
            raise ValueError("Model has not been fitted yet.")
            
        n_samples, _ = X.shape
        weighted_log_prob = np.zeros((n_samples, self.n_components))
        
        # Tính log-probability cho mỗi thành phần
        for k in range(self.n_components):
            try:
                log_prob = multivariate_normal.logpdf(
                    X, mean=self.means_[k], cov=self.covariances_[k]
                )
                weighted_log_prob[:, k] = log_prob + np.log(self.weights_[k])
            except:
                weighted_log_prob[:, k] = -np.inf
        
        # Dự đoán nhãn là thành phần có log-probability cao nhất
        return weighted_log_prob.argmax(axis=1)
    
    def predict_proba(self, X):
        """
        Dự đoán xác suất hậu nghiệm cho mỗi thành phần.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Dữ liệu mới cần dự đoán
            
        Returns:
        --------
        responsibilities : array-like, shape (n_samples, n_components)
            Xác suất hậu nghiệm (responsibility) của mỗi thành phần cho mỗi mẫu
        """
        if self.means_ is None:
            raise ValueError("Model has not been fitted yet.")
            
        n_samples, _ = X.shape
        weighted_log_prob = np.zeros((n_samples, self.n_components))
        
        # Tính log-probability cho mỗi thành phần
        for k in range(self.n_components):
            try:
                log_prob = multivariate_normal.logpdf(
                    X, mean=self.means_[k], cov=self.covariances_[k]
                )
                weighted_log_prob[:, k] = log_prob + np.log(self.weights_[k])
            except:
                weighted_log_prob[:, k] = -np.inf
        
        # Chuyển log-probabilities thành probabilities
        log_prob_norm = np.log(np.sum(np.exp(weighted_log_prob - np.max(weighted_log_prob, axis=1, keepdims=True)), axis=1))
        log_prob_norm += np.max(weighted_log_prob, axis=1)
        
        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        resp = np.exp(log_resp)
        
        return resp
    
    def score_samples(self, X):
        """
        Tính log-likelihood của mỗi mẫu dưới mô hình.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Dữ liệu cần tính điểm
            
        Returns:
        --------
        log_prob : array-like, shape (n_samples,)
            Log-likelihood của mỗi mẫu dưới mô hình
        """
        if self.means_ is None:
            raise ValueError("Model has not been fitted yet.")
            
        n_samples, _ = X.shape
        weighted_log_prob = np.zeros((n_samples, self.n_components))
        
        # Tính log-probability cho mỗi thành phần
        for k in range(self.n_components):
            try:
                log_prob = multivariate_normal.logpdf(
                    X, mean=self.means_[k], cov=self.covariances_[k]
                )
                weighted_log_prob[:, k] = log_prob + np.log(self.weights_[k])
            except:
                weighted_log_prob[:, k] = -np.inf
        
        # Sử dụng log-sum-exp trick để tính logarithm của tổng một cách ổn định số học
        log_prob_norm = np.log(np.sum(np.exp(weighted_log_prob - np.max(weighted_log_prob, axis=1, keepdims=True)), axis=1))
        log_prob_norm += np.max(weighted_log_prob, axis=1)
        
        return log_prob_norm
    
    def bic(self, X):
        """
        Tính chỉ số Bayesian Information Criterion (BIC) cho mô hình.
        BIC được sử dụng để chọn số lượng thành phần tối ưu.
        Giá trị BIC càng thấp càng tốt.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Dữ liệu cần tính BIC
            
        Returns:
        --------
        bic : float
            Giá trị BIC của mô hình
        """
        n_samples, n_features = X.shape
        log_likelihood = np.sum(self.score_samples(X))
        
        # Số lượng tham số tự do trong mô hình
        # - n_components - 1 : số tham số cho trọng số (weights)
        # - n_components * n_features : số tham số cho các vector trung bình
        # - n_components * n_features * (n_features + 1) / 2 : số tham số cho ma trận hiệp phương sai
        n_parameters = (self.n_components - 1) + \
                      (self.n_components * n_features) + \
                      (self.n_components * n_features * (n_features + 1) / 2)
        
        return -2 * log_likelihood + n_parameters * np.log(n_samples)
    
    def aic(self, X):
        """
        Tính chỉ số Akaike Information Criterion (AIC) cho mô hình.
        AIC cũng được sử dụng để chọn số lượng thành phần tối ưu.
        Giá trị AIC càng thấp càng tốt.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Dữ liệu cần tính AIC
            
        Returns:
        --------
        aic : float
            Giá trị AIC của mô hình
        """
        n_samples, n_features = X.shape
        log_likelihood = np.sum(self.score_samples(X))
        
        # Số lượng tham số tự do trong mô hình (giống như trong BIC)
        n_parameters = (self.n_components - 1) + \
                      (self.n_components * n_features) + \
                      (self.n_components * n_features * (n_features + 1) / 2)
        
        return -2 * log_likelihood + 2 * n_parameters
    
    def plot_clusters(self, X, axis1=0, axis2=1, ax=None, title="GMM Clustering", cmap="viridis"):
        """
        Vẽ đồ thị phân cụm GMM.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Dữ liệu cần vẽ
        axis1, axis2 : int, default=0, 1
            Chỉ số của các đặc trưng cần vẽ
        ax : matplotlib.axes.Axes, default=None
            Axes để vẽ. Nếu None, sẽ tạo mới
        title : str, default="GMM Clustering"
            Tiêu đề của đồ thị
        cmap : str, default="viridis"
            Bảng màu sử dụng cho đồ thị
            
        Returns:
        --------
        ax : matplotlib.axes.Axes
            Axes đã vẽ
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        
        # Vẽ các điểm dữ liệu với màu tương ứng với cụm
        ax.scatter(X[:, axis1], X[:, axis2], c=self.labels_, cmap=cmap, 
                  alpha=0.7, s=50, edgecolors='w')
        
        # Vẽ các trung tâm của Gaussian
        ax.scatter(self.means_[:, axis1], self.means_[:, axis2], 
                  c='red', s=100, marker='X', edgecolors='k')
        
        # Vẽ ellipse tương ứng với ma trận hiệp phương sai của mỗi thành phần Gaussian
        from matplotlib.patches import Ellipse
        for i in range(self.n_components):
            if self.covariances_[i][axis1, axis2] == 0:
                continue
                
            # Tính các giá trị riêng và vector riêng của ma trận hiệp phương sai 2x2
            covar = np.array([
                [self.covariances_[i][axis1, axis1], self.covariances_[i][axis1, axis2]],
                [self.covariances_[i][axis1, axis2], self.covariances_[i][axis2, axis2]]
            ])
            
            eig_vals, eig_vecs = np.linalg.eigh(covar)
            
            # Góc của trục chính (trong độ)
            angle = np.degrees(np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0]))
            
            # Độ dài của trục chính = 2 * sqrt(eigenvalue) * (hệ số cho khoảng tin cậy)
            width, height = 2 * np.sqrt(eig_vals) * 2  # 2 là hệ số khoảng tin cậy ~95%
            
            # Tạo ellipse
            ellipse = Ellipse(
                xy=(self.means_[i, axis1], self.means_[i, axis2]),
                width=width, height=height, angle=angle,
                facecolor='none', edgecolor='r', linestyle='--', linewidth=2
            )
            ax.add_patch(ellipse)
        
        ax.set_title(title)
        ax.set_xlabel(f'Feature {axis1}')
        ax.set_ylabel(f'Feature {axis2}')
        ax.grid(True, alpha=0.3)
        
        return ax

def find_optimal_gmm_components(X, max_components=10, criterion='bic', n_runs=3):
    """
    Tìm số lượng thành phần tối ưu cho mô hình GMM dựa trên BIC hoặc AIC.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Dữ liệu đầu vào
    max_components : int, default=10
        Số lượng thành phần tối đa để thử
    criterion : {'bic', 'aic'}, default='bic'
        Tiêu chí để chọn mô hình tối ưu
    n_runs : int, default=3
        Số lần chạy cho mỗi số lượng thành phần để chọn mô hình tốt nhất
        
    Returns:
    --------
    optimal_n_components : int
        Số lượng thành phần tối ưu
    scores : list
        Danh sách các giá trị criterion cho mỗi số lượng thành phần
    best_models : dict
        Từ điển chứa mô hình tốt nhất cho mỗi số lượng thành phần
    """
    scores = []
    best_models = {}
    
    # Thử các số lượng thành phần
    for n_components in range(1, max_components + 1):
        best_score = np.inf
        best_model = None
        
        # Chạy nhiều lần cho mỗi số lượng thành phần
        for _ in range(n_runs):
            gmm = MyGMM(n_components=n_components, random_state=np.random.randint(0, 1000))
            gmm.fit(X)
            
            # Tính điểm theo tiêu chí đã chọn
            if criterion == 'bic':
                score = gmm.bic(X)
            else:  # 'aic'
                score = gmm.aic(X)
            
            # Cập nhật mô hình tốt nhất
            if score < best_score:
                best_score = score
                best_model = gmm
        
        scores.append(best_score)
        best_models[n_components] = best_model
    
    # Tìm số lượng thành phần có điểm thấp nhất
    optimal_n_components = np.argmin(scores) + 1
    
    return optimal_n_components, scores, best_models

def plot_gmm_criterion_scores(scores, criterion='BIC'):
    """
    Vẽ đồ thị các giá trị AIC/BIC để xác định số lượng thành phần tối ưu.
    
    Parameters:
    -----------
    scores : list
        Danh sách các giá trị criterion cho mỗi số lượng thành phần
    criterion : str, default='BIC'
        Tên của tiêu chí ('BIC' hoặc 'AIC')
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Đối tượng Figure chứa đồ thị
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Vẽ đồ thị điểm theo số lượng thành phần
    ax.plot(range(1, len(scores) + 1), scores, 'o-', linewidth=2, markersize=8)
    ax.set_xticks(range(1, len(scores) + 1))
    
    # Đánh dấu điểm tối thiểu
    min_idx = np.argmin(scores)
    ax.plot(min_idx + 1, scores[min_idx], 'ro', markersize=12, 
           markerfacecolor='none', markeredgewidth=2, 
           label=f'Optimal: {min_idx + 1} components')
    
    ax.set_xlabel('Number of Components')
    ax.set_ylabel(f'{criterion} Score')
    ax.set_title(f'{criterion} Scores for Different Number of GMM Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig 