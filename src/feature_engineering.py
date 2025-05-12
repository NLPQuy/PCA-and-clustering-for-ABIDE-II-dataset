import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, VarianceThreshold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import KernelPCA, NMF, FastICA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import MiniBatchDictionaryLearning, TruncatedSVD
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy import linalg
import networkx as nx
from sklearn.metrics.pairwise import rbf_kernel

def create_polynomial_features(X, degree=2):
    """
    Tạo các đặc trưng đa thức từ dữ liệu gốc.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Dữ liệu đầu vào
    degree : int, default=2
        Bậc của đa thức
        
    Returns:
    --------
    X_poly : array-like, shape (n_samples, n_new_features)
        Dữ liệu sau khi tạo đặc trưng đa thức
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    return X_poly

def select_features_variance(X, threshold=0.01):
    """
    Chọn đặc trưng dựa trên phương sai (không giám sát).
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Dữ liệu đầu vào
    threshold : float, default=0.01
        Ngưỡng phương sai tối thiểu
        
    Returns:
    --------
    X_selected : array-like, shape (n_samples, k)
        Dữ liệu sau khi chọn đặc trưng
    """
    selector = VarianceThreshold(threshold=threshold)
    X_selected = selector.fit_transform(X)
    
    # Lấy chỉ mục của các đặc trưng được chọn
    selected_indices = selector.get_support(indices=True)
    
    print(f"Đã chọn {len(selected_indices)} đặc trưng có phương sai > {threshold} từ {X.shape[1]} đặc trưng gốc")
    
    return X_selected, selected_indices

def apply_kernel_pca(X, n_components=2, kernel='rbf', gamma=None):
    """
    Áp dụng Kernel PCA để giảm chiều dữ liệu.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Dữ liệu đầu vào
    n_components : int, default=2
        Số thành phần chính cần giữ lại
    kernel : str, default='rbf'
        Loại kernel: 'linear', 'poly', 'rbf', 'sigmoid', 'cosine'
    gamma : float, default=None
        Tham số kernel cho 'rbf', 'poly' và 'sigmoid'
        
    Returns:
    --------
    X_kpca : array-like, shape (n_samples, n_components)
        Dữ liệu sau khi áp dụng Kernel PCA
    """
    kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma)
    X_kpca = kpca.fit_transform(X)
    print(f"Đã áp dụng Kernel PCA với kernel='{kernel}', giảm từ {X.shape[1]} xuống {n_components} chiều")
    return X_kpca

def apply_nmf(X, n_components=2):
    """
    Áp dụng Non-negative Matrix Factorization (NMF) để giảm chiều dữ liệu không âm.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Dữ liệu đầu vào (phải không âm)
    n_components : int, default=2
        Số thành phần cần giữ lại
        
    Returns:
    --------
    X_nmf : array-like, shape (n_samples, n_components)
        Dữ liệu sau khi áp dụng NMF
    """
    # Đảm bảo dữ liệu không âm
    X_non_neg = np.maximum(0, X)
    
    nmf = NMF(n_components=n_components, random_state=42)
    X_nmf = nmf.fit_transform(X_non_neg)
    print(f"Đã áp dụng NMF, giảm từ {X.shape[1]} xuống {n_components} chiều")
    return X_nmf

def apply_ica(X, n_components=2):
    """
    Áp dụng Independent Component Analysis (ICA) để tách thành phần độc lập.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Dữ liệu đầu vào
    n_components : int, default=2
        Số thành phần độc lập cần giữ lại
        
    Returns:
    --------
    X_ica : array-like, shape (n_samples, n_components)
        Dữ liệu sau khi áp dụng ICA
    """
    ica = FastICA(n_components=n_components, random_state=42)
    X_ica = ica.fit_transform(X)
    print(f"Đã áp dụng ICA, giảm từ {X.shape[1]} xuống {n_components} chiều")
    return X_ica

def apply_random_projection(X, n_components=2):
    """
    Áp dụng Random Projection để giảm chiều dữ liệu.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Dữ liệu đầu vào
    n_components : int, default=2
        Số chiều đầu ra
        
    Returns:
    --------
    X_rp : array-like, shape (n_samples, n_components)
        Dữ liệu sau khi áp dụng Random Projection
    """
    rp = GaussianRandomProjection(n_components=n_components, random_state=42)
    X_rp = rp.fit_transform(X)
    print(f"Đã áp dụng Random Projection, giảm từ {X.shape[1]} xuống {n_components} chiều")
    return X_rp

def apply_tsne(X, n_components=2, perplexity=30):
    """
    Áp dụng t-SNE để trực quan hóa dữ liệu cao chiều.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Dữ liệu đầu vào
    n_components : int, default=2
        Số chiều đầu ra
    perplexity : float, default=30
        Tham số perplexity của t-SNE
        
    Returns:
    --------
    X_tsne : array-like, shape (n_samples, n_components)
        Dữ liệu sau khi áp dụng t-SNE
    """
    # Giảm kích thước dữ liệu trước khi chạy t-SNE nếu cần thiết
    if X.shape[1] > 50:
        print("Dữ liệu có quá nhiều chiều. Đang giảm chiều trước khi áp dụng t-SNE...")
        X = KernelPCA(n_components=50, kernel='rbf').fit_transform(X)
    
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X)
    print(f"Đã áp dụng t-SNE, giảm từ {X.shape[1]} xuống {n_components} chiều")
    return X_tsne

def apply_isomap(X, n_components=2, n_neighbors=5):
    """
    Áp dụng Isomap để giảm chiều dữ liệu.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Dữ liệu đầu vào
    n_components : int, default=2
        Số chiều đầu ra
    n_neighbors : int, default=5
        Số lượng láng giềng sử dụng trong thuật toán
        
    Returns:
    --------
    X_isomap : array-like, shape (n_samples, n_components)
        Dữ liệu sau khi áp dụng Isomap
    """
    isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    X_isomap = isomap.fit_transform(X)
    print(f"Đã áp dụng Isomap, giảm từ {X.shape[1]} xuống {n_components} chiều")
    return X_isomap

def apply_lle(X, n_components=2, n_neighbors=5):
    """
    Áp dụng Locally Linear Embedding (LLE) để giảm chiều dữ liệu.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Dữ liệu đầu vào
    n_components : int, default=2
        Số chiều đầu ra
    n_neighbors : int, default=5
        Số lượng láng giềng sử dụng trong thuật toán
        
    Returns:
    --------
    X_lle : array-like, shape (n_samples, n_components)
        Dữ liệu sau khi áp dụng LLE
    """
    lle = LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors, random_state=42)
    X_lle = lle.fit_transform(X)
    print(f"Đã áp dụng LLE, giảm từ {X.shape[1]} xuống {n_components} chiều")
    return X_lle

def apply_polynomial_pca(X, degree=2, n_components=2):
    """
    Tạo đặc trưng đa thức trước khi áp dụng PCA để nắm bắt các mối quan hệ phi tuyến.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Dữ liệu đầu vào
    degree : int, default=2
        Bậc đa thức
    n_components : int, default=2
        Số lượng thành phần chính cần giữ lại
        
    Returns:
    --------
    X_poly_pca : array-like, shape (n_samples, n_components)
        Dữ liệu sau khi áp dụng biến đổi đa thức và PCA
    """
    # Tạo đặc trưng đa thức
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Áp dụng PCA
    pca = PCA(n_components=n_components)
    X_poly_pca = pca.fit_transform(X_poly)
    
    print(f"Đã áp dụng biến đổi đa thức bậc {degree} và PCA, từ {X.shape[1]} đặc trưng gốc, kết quả là {n_components} thành phần chính")
    return X_poly_pca

def apply_feature_agglomeration(X, n_clusters=10, linkage='ward'):
    """
    Gom cụm các đặc trưng tương tự nhau để giảm số chiều.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Dữ liệu đầu vào
    n_clusters : int, default=10
        Số lượng cụm đặc trưng cần giữ lại
    linkage : {'ward', 'complete', 'average'}, default='ward'
        Chiến lược liên kết cho gom cụm
        
    Returns:
    --------
    X_agg : array-like, shape (n_samples, n_clusters)
        Dữ liệu sau khi gom cụm đặc trưng
    """
    agglo = FeatureAgglomeration(n_clusters=n_clusters, linkage=linkage)
    X_agg = agglo.fit_transform(X)
    
    print(f"Đã áp dụng Feature Agglomeration, giảm từ {X.shape[1]} xuống {n_clusters} đặc trưng")
    return X_agg

def apply_select_from_model(X, y=None, estimator=None, threshold='mean', max_features=None):
    """
    Chọn đặc trưng dựa trên importance từ mô hình dự đoán (có thể dùng với y=None).
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Dữ liệu đầu vào
    y : array-like, shape (n_samples,), optional
        Nhãn lớp cho các mẫu (có thể để None)
    estimator : object, default=None
        Mô hình ước lượng importance, mặc định sẽ dùng ExtraTreesClassifier
    threshold : string, float, optional, default='mean'
        Ngưỡng để chọn đặc trưng
    max_features : int or None, optional, default=None
        Số lượng đặc trưng tối đa cần giữ lại
        
    Returns:
    --------
    X_selected : array-like
        Dữ liệu sau khi chọn đặc trưng
    """
    if estimator is None:
        if y is None:
            # Nếu không có nhãn lớp, sử dụng ExtraTrees theo phương pháp không giám sát
            estimator = ExtraTreesClassifier(n_estimators=100, random_state=42)
            # Tạo nhãn ngẫu nhiên chỉ để chạy mô hình
            random_y = np.random.randint(0, 2, size=X.shape[0])
            estimator.fit(X, random_y)
        else:
            estimator = ExtraTreesClassifier(n_estimators=100, random_state=42)
            estimator.fit(X, y)
    
    selector = SelectFromModel(estimator, threshold=threshold, max_features=max_features)
    X_selected = selector.fit_transform(X, y)
    
    print(f"Đã áp dụng SelectFromModel, giảm từ {X.shape[1]} xuống {X_selected.shape[1]} đặc trưng")
    return X_selected

def apply_dictionary_learning(X, n_components=10, alpha=1.0, batch_size=200):
    """
    Áp dụng Dictionary Learning để tìm các thành phần chính cơ sở cho dữ liệu.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Dữ liệu đầu vào
    n_components : int, default=10
        Số lượng từ điển (thành phần) cần học
    alpha : float, default=1.0
        Tham số điều chỉnh độ thưa
    batch_size : int, default=200
        Kích thước batch cho minibatch learning
        
    Returns:
    --------
    X_dict : array-like, shape (n_samples, n_components)
        Dữ liệu được biến đổi qua dictionary learning
    """
    # Đảm bảo dữ liệu không âm
    min_val = X.min()
    if min_val < 0:
        X_shifted = X - min_val
    else:
        X_shifted = X
    
    # Áp dụng dictionary learning
    dict_learning = MiniBatchDictionaryLearning(n_components=n_components, 
                                              alpha=alpha, 
                                              batch_size=batch_size,
                                              random_state=42)
    X_dict = dict_learning.fit_transform(X_shifted)
    
    print(f"Đã áp dụng Dictionary Learning, giảm từ {X.shape[1]} xuống {n_components} thành phần")
    return X_dict

def apply_truncated_svd(X, n_components=10):
    """
    Áp dụng Truncated SVD (tương tự PCA nhưng không yêu cầu centered data).
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Dữ liệu đầu vào
    n_components : int, default=10
        Số lượng thành phần cần giữ lại
        
    Returns:
    --------
    X_svd : array-like, shape (n_samples, n_components)
        Dữ liệu sau khi áp dụng Truncated SVD
    """
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_svd = svd.fit_transform(X)
    
    explained_variance_ratio = svd.explained_variance_ratio_.sum()
    print(f"Đã áp dụng Truncated SVD, giảm từ {X.shape[1]} xuống {n_components} thành phần")
    print(f"Tỷ lệ phương sai giải thích được: {explained_variance_ratio:.4f}")
    
    return X_svd

def apply_select_percentile(X, y=None, score_func=chi2, percentile=10):
    """
    Chọn đặc trưng dựa trên phần trăm đặc trưng tốt nhất theo hàm điểm.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Dữ liệu đầu vào
    y : array-like, shape (n_samples,), optional
        Nhãn lớp cho các mẫu
    score_func : callable, default=chi2
        Hàm tính điểm cho các đặc trưng
    percentile : int, default=10
        Phần trăm đặc trưng cần giữ lại
        
    Returns:
    --------
    X_selected : array-like
        Dữ liệu sau khi chọn đặc trưng
    """
    if y is None:
        # Nếu không có nhãn, tạo nhãn giả để sử dụng kỹ thuật có giám sát
        y_dummy = np.random.randint(0, 2, size=X.shape[0])
    else:
        y_dummy = y
    
    # Đảm bảo dữ liệu không âm nếu sử dụng chi2
    if score_func == chi2:
        min_val = X.min()
        if min_val < 0:
            X_non_neg = X - min_val
        else:
            X_non_neg = X
        X_to_select = X_non_neg
    else:
        X_to_select = X
    
    selector = SelectPercentile(score_func, percentile=percentile)
    X_selected = selector.fit_transform(X_to_select, y_dummy)
    
    print(f"Đã áp dụng Select Percentile ({percentile}%), giảm từ {X.shape[1]} xuống {X_selected.shape[1]} đặc trưng")
    return X_selected

def feature_engineering_pipeline(X, y=None, methods=None, components=20):
    """
    Pipeline xử lý feature engineering với nhiều phương pháp khác nhau.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Dữ liệu đầu vào
    y : array-like, shape (n_samples,), optional
        Nhãn lớp, không sử dụng trong các phương pháp không giám sát
    methods : list of dict, default=None
        Danh sách các phương pháp cần áp dụng và tham số của chúng
    components : int, default=20
        Số lượng thành phần (components) mặc định cho các phương pháp giảm chiều dữ liệu
        
    Returns:
    --------
    X_transformed : array-like
        Dữ liệu sau khi áp dụng các phương pháp feature engineering
    """
    if methods is None:
        methods = [
            {'name': 'scaling', 'type': 'standard'},
            {'name': 'kernel_pca', 'n_components': components, 'kernel': 'rbf'}
        ]
    
    X_transformed = X.copy()
    
    for method in methods:
        name = method.get('name', '')
        
        if name == 'scaling':
            scaler_type = method.get('type', 'standard')
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'robust':
                scaler = RobustScaler()
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
            elif scaler_type == 'maxabs':
                scaler = MaxAbsScaler()
            X_transformed = scaler.fit_transform(X_transformed)
            print(f"Đã áp dụng {scaler_type} scaling")
            
        elif name == 'polynomial':
            degree = method.get('degree', 2)
            X_transformed = create_polynomial_features(X_transformed, degree=degree)
            print(f"Đã tạo đặc trưng đa thức bậc {degree}, tăng từ {X.shape[1]} lên {X_transformed.shape[1]} chiều")
            
        elif name == 'select_variance':
            threshold = method.get('threshold', 0.01)
            X_transformed, _ = select_features_variance(X_transformed, threshold=threshold)
            
        elif name == 'kernel_pca':
            n_components = method.get('n_components', min(components, X_transformed.shape[1]))
            kernel = method.get('kernel', 'rbf')
            gamma = method.get('gamma', None)
            X_transformed = apply_kernel_pca(X_transformed, n_components=n_components, 
                                          kernel=kernel, gamma=gamma)
            
        elif name == 'nmf':
            n_components = method.get('n_components', min(components, X_transformed.shape[1]))
            X_transformed = apply_nmf(X_transformed, n_components=n_components)
            
        elif name == 'ica':
            n_components = method.get('n_components', min(components, X_transformed.shape[1]))
            X_transformed = apply_ica(X_transformed, n_components=n_components)
            
        elif name == 'random_projection':
            n_components = method.get('n_components', min(components, X_transformed.shape[1]))
            X_transformed = apply_random_projection(X_transformed, n_components=n_components)
            
        elif name == 'tsne':
            if X_transformed.shape[0] > 10000:
                print("Cảnh báo: t-SNE không phù hợp với tập dữ liệu lớn")
                continue
                
            n_components = method.get('n_components', min(components, 3))  # t-SNE thường giới hạn ở 2-3 components
            perplexity = method.get('perplexity', 30)
            X_transformed = apply_tsne(X_transformed, n_components=n_components, perplexity=perplexity)
            
        elif name == 'isomap':
            n_components = method.get('n_components', min(components, X_transformed.shape[1]))
            n_neighbors = method.get('n_neighbors', 5)
            X_transformed = apply_isomap(X_transformed, n_components=n_components, n_neighbors=n_neighbors)
            
        elif name == 'lle':
            n_components = method.get('n_components', min(components, X_transformed.shape[1]))
            n_neighbors = method.get('n_neighbors', 5)
            X_transformed = apply_lle(X_transformed, n_components=n_components, n_neighbors=n_neighbors)

        elif name == 'polynomial_pca':
            n_components = method.get('n_components', min(components, X_transformed.shape[1]))
            degree = method.get('degree', 2)
            X_transformed = apply_polynomial_pca(X_transformed, degree=degree, n_components=n_components)
            
        elif name == 'feature_agglomeration':
            n_clusters = method.get('n_clusters', min(components, X_transformed.shape[1]))
            linkage = method.get('linkage', 'ward')
            X_transformed = apply_feature_agglomeration(X_transformed, n_clusters=n_clusters, linkage=linkage)
            
        elif name == 'select_from_model':
            threshold = method.get('threshold', 'mean')
            max_features = method.get('max_features', None)
            X_transformed = apply_select_from_model(X_transformed, y, threshold=threshold, max_features=max_features)
            
        elif name == 'dictionary_learning':
            n_components = method.get('n_components', min(components, X_transformed.shape[1]))
            alpha = method.get('alpha', 1.0)
            batch_size = method.get('batch_size', 200)
            X_transformed = apply_dictionary_learning(X_transformed, n_components=n_components, 
                                                   alpha=alpha, batch_size=batch_size)
            
        elif name == 'truncated_svd':
            n_components = method.get('n_components', min(components, X_transformed.shape[1]))
            X_transformed = apply_truncated_svd(X_transformed, n_components=n_components)
            
        elif name == 'select_percentile':
            percentile = method.get('percentile', 10)
            X_transformed = apply_select_percentile(X_transformed, y, percentile=percentile)
            
    return X_transformed 

def apply_zca_whitening_pca(X, n_components=20, eps=1e-6):
    """
    Áp dụng ZCA whitening trước khi thực hiện PCA để làm giảm tương quan giữa các đặc trưng.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Dữ liệu đầu vào
    n_components : int, default=20
        Số thành phần chính cần giữ lại sau khi áp dụng PCA
    eps : float, default=1e-6
        Giá trị nhỏ thêm vào để tránh chia cho 0
        
    Returns:
    --------
    X_zca_pca : array-like, shape (n_samples, n_components)
        Dữ liệu sau khi áp dụng ZCA whitening và PCA
    """
    print("Áp dụng ZCA whitening và PCA...")
    # Chuẩn hóa dữ liệu
    X_centered = X - np.mean(X, axis=0)
    
    # Tính ma trận hiệp phương sai
    cov = np.cov(X_centered, rowvar=False)
    
    # Tính giá trị riêng và vector riêng
    U, S, V = linalg.svd(cov)
    
    # ZCA whitening
    components = U.dot(np.diag(1.0 / np.sqrt(S + eps))).dot(U.T)
    X_whitened = X_centered.dot(components)
    
    # Áp dụng PCA để giảm chiều
    pca = PCA(n_components=n_components)
    X_zca_pca = pca.fit_transform(X_whitened)
    
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"Đã áp dụng ZCA whitening + PCA, giảm từ {X.shape[1]} xuống {n_components} thành phần")
    print(f"Tỷ lệ phương sai giải thích được: {explained_variance:.4f}")
    
    return X_zca_pca

def apply_whitening_graph_feature_selection_pca(X, n_components=20, n_neighbors=10, threshold=0.5):
    """
    Áp dụng whitening, sau đó dùng graph-based feature selection rồi PCA.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Dữ liệu đầu vào
    n_components : int, default=20
        Số thành phần chính cần giữ lại sau khi áp dụng PCA
    n_neighbors : int, default=10
        Số lượng láng giềng gần nhất để xây dựng đồ thị
    threshold : float, default=0.5
        Ngưỡng để lọc cạnh trong đồ thị
        
    Returns:
    --------
    X_graph_pca : array-like, shape (n_samples, n_components)
        Dữ liệu sau khi áp dụng whitening, graph-based feature selection và PCA
    """
    print("Áp dụng whitening, graph-based feature selection và PCA...")
    
    # Whitening data
    X_centered = X - np.mean(X, axis=0)
    pca_whiten = PCA(whiten=True)
    X_whitened = pca_whiten.fit_transform(X_centered)
    
    # Tính ma trận tương đồng giữa các đặc trưng
    n_features = X_whitened.shape[1]
    
    # Tính correlation matrix
    corr_matrix = np.corrcoef(X_whitened.T)
    
    # Xây dựng đồ thị từ ma trận tương đồng
    G = nx.Graph()
    
    # Thêm nút cho mỗi đặc trưng
    for i in range(n_features):
        G.add_node(i)
    
    # Thêm cạnh với trọng số là tương quan giữa các đặc trưng
    for i in range(n_features):
        for j in range(i+1, n_features):
            correlation = abs(corr_matrix[i, j])
            if correlation > threshold:
                G.add_edge(i, j, weight=correlation)
    
    # Tính độ trung tâm (centrality) cho mỗi nút
    centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06, weight='weight')
    
    # Sắp xếp các đặc trưng theo độ trung tâm
    sorted_features = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    
    # Chọn các đặc trưng có độ trung tâm cao
    selected_features = [feature[0] for feature in sorted_features[:min(n_features, 100)]]
    
    # Chọn các đặc trưng đã chọn từ dữ liệu gốc
    X_selected = X_whitened[:, selected_features]
    
    # Áp dụng PCA để giảm chiều
    pca = PCA(n_components=min(n_components, X_selected.shape[1]))
    X_graph_pca = pca.fit_transform(X_selected)
    
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"Đã áp dụng whitening + graph-based selection + PCA, từ {X.shape[1]} xuống {n_components} thành phần")
    print(f"Đã chọn {len(selected_features)} đặc trưng từ đồ thị với ngưỡng {threshold}")
    print(f"Tỷ lệ phương sai giải thích được: {explained_variance:.4f}")
    
    return X_graph_pca 