#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bài toán phân cụm nâng cao trên tập dữ liệu ABIDE với nhiều thuật toán phân cụm khác nhau
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Import custom modules
from src import (
    MyPCA, 
    MyKMeans, 
    load_abide_data, 
    preprocess_data,
    apply_kernel_pca,
    apply_polynomial_pca,
    apply_truncated_svd,
    MyAgglomerativeClustering,
    MyDBSCAN,
    MyMeanShift,
    MyAffinityPropagation,
    find_optimal_clusters
)
from utils import plot_pca_components_variance, plot_pca_2d, plot_clusters_2d
from utils import calculate_cluster_metrics, print_metrics

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

def detect_outliers(X, threshold=3):
    """
    Phát hiện outliers trong dữ liệu sử dụng phương pháp Z-score.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Dữ liệu cần kiểm tra outliers
    threshold : float, default=3
        Ngưỡng Z-score để phát hiện outlier
        
    Returns:
    --------
    outlier_mask : array, shape (n_samples,)
        Mặt nạ boolean của outliers (True cho outliers)
    """
    # Tính Z-scores cho mỗi đặc trưng
    z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
    
    # Tìm các mẫu có bất kỳ đặc trưng nào có Z-score > threshold
    outlier_mask = np.any(z_scores > threshold, axis=1)
    
    return outlier_mask

def main():
    """
    Hàm chính để phân tích phân cụm trên tập dữ liệu ABIDE với nhiều thuật toán khác nhau.
    """
    print("=== Lab 2 - Phân cụm trên ABIDE Dataset với nhiều thuật toán phân cụm khác nhau ===")
    
    # Load ABIDE dataset
    X, y, feature_names = load_abide_data("ABIDE2(updated).csv")
    
    if X is None:
        print("Lỗi khi tải tập dữ liệu. Kết thúc.")
        return
    
    # Kiểm tra và xử lý outliers
    print("\nKiểm tra outliers...")
    outlier_mask = detect_outliers(X, threshold=5)
    outlier_count = np.sum(outlier_mask)
    print(f"Phát hiện {outlier_count} outliers trong tổng số {X.shape[0]} mẫu ({outlier_count/X.shape[0]*100:.2f}%)")
    
    if outlier_count > 0:
        # Tùy chọn 1: Loại bỏ outliers
        if outlier_count < X.shape[0] * 0.3:  # Chỉ loại bỏ nếu ít hơn 30% là outliers
            X_clean = X[~outlier_mask]
            y_clean = y[~outlier_mask]
            print(f"Đã loại bỏ {outlier_count} outliers. Kích thước mới của tập dữ liệu: {X_clean.shape}")
            X = X_clean
            y = y_clean
        else:
            # Tùy chọn 2: Sử dụng robust scaling thay vì standard scaling
            print("Quá nhiều outliers để loại bỏ. Sử dụng robust scaling thay vì standard scaling.")
            scaler = RobustScaler()
            X = scaler.fit_transform(X)
            print("Dữ liệu đã được robust-scaled.")
    
    # Chuẩn hóa dữ liệu
    print("\nChuẩn hóa dữ liệu...")
    X_std, _ = preprocess_data(X)
    
    # Tìm số lượng cụm tối ưu cho KMeans
    print("\n=== Tìm số lượng cụm tối ưu cho KMeans ===")
    optimal_n_clusters, _ = find_optimal_clusters(
        X_std, method='kmeans', metric='silhouette', max_clusters=10
    )
    print(f"Số lượng cụm tối ưu cho KMeans: {optimal_n_clusters}")
    
    # Định nghĩa các phương pháp giảm chiều để thử nghiệm
    dim_reduction_methods = [
        {
            "name": "Standard PCA",
            "function": lambda X, n_components: MyPCA(n_components=n_components).fit_transform(X),
            "n_components": 20
        },
        {
            "name": "Kernel PCA (RBF)",
            "function": lambda X, n_components: apply_kernel_pca(X, n_components=n_components, kernel='rbf'),
            "n_components": 20
        },
        {
            "name": "Polynomial PCA",
            "function": lambda X, n_components: apply_polynomial_pca(X, n_components=n_components, degree=2),
            "n_components": 20
        },
        {
            "name": "Truncated SVD",
            "function": lambda X, n_components: apply_truncated_svd(X, n_components=n_components),
            "n_components": 20
        }
    ]
    
    # Định nghĩa các thuật toán phân cụm để thử nghiệm
    clustering_algorithms = [
        {
            "name": "K-Means",
            "function": lambda n_clusters: MyKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        },
        {
            "name": "Agglomerative Clustering (Ward)",
            "function": lambda n_clusters: MyAgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        },
        {
            "name": "DBSCAN",
            "function": lambda n_clusters: MyDBSCAN(eps=0.5, min_samples=5)  # n_clusters không được sử dụng
        },
        {
            "name": "Mean Shift",
            "function": lambda n_clusters: MyMeanShift(bandwidth=None)  # n_clusters không được sử dụng
        },
        {
            "name": "Affinity Propagation",
            "function": lambda n_clusters: MyAffinityPropagation(damping=0.9)  # n_clusters không được sử dụng
        }
    ]
    
    # Lưu kết quả
    results = []
    
    # Thử từng phương pháp giảm chiều
    for reduction_method in dim_reduction_methods:
        method_name = reduction_method["name"]
        reduction_func = reduction_method["function"]
        n_components = reduction_method["n_components"]
        
        print(f"\n=== Thử nghiệm với phương pháp giảm chiều: {method_name} ===")
        
        # Áp dụng phương pháp giảm chiều
        start_time = time.time()
        X_reduced = reduction_func(X_std, n_components)
        reduction_time = time.time() - start_time
        print(f"Thời gian giảm chiều: {reduction_time:.2f} giây")
        
        # Thử nghiệm với từng thuật toán phân cụm
        for algorithm in clustering_algorithms:
            algo_name = algorithm["name"]
            algo_func = algorithm["function"]
            
            print(f"\nÁp dụng {algo_name} trên dữ liệu đã giảm chiều với {method_name}...")
            
            try:
                # Khởi tạo và huấn luyện thuật toán phân cụm
                start_time = time.time()
                
                # Với các thuật toán cần số lượng cụm
                if algo_name in ["K-Means", "Agglomerative Clustering (Ward)"]:
                    model = algo_func(optimal_n_clusters)
                else:
                    # Với các thuật toán không cần số lượng cụm
                    model = algo_func(None)
                
                model.fit(X_reduced)
                clustering_time = time.time() - start_time
                
                # Đánh giá kết quả phân cụm
                metrics = calculate_cluster_metrics(y, model.labels_)
                print(f"Kết quả với {algo_name} trên {method_name}:")
                print_metrics(metrics)
                print(f"Thời gian phân cụm: {clustering_time:.2f} giây")
                
                # Lưu kết quả
                result = {
                    'dim_reduction': method_name,
                    'clustering': algo_name,
                    'n_features': X_reduced.shape[1],
                    'n_clusters': len(np.unique(model.labels_)),
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'ari': metrics['ari'],
                    'nmi': metrics['nmi'],
                    'reduction_time': reduction_time,
                    'clustering_time': clustering_time
                }
                results.append(result)
                
                # Vẽ kết quả phân cụm nếu dữ liệu có ít nhất 2D và ít hơn 1,000 điểm
                if X_reduced.shape[1] >= 2 and X_reduced.shape[0] < 1000:
                    plot_clusters_2d(
                        X_reduced[:, :2], 
                        model.labels_, 
                        y,
                        title=f"Phân cụm với {algo_name} trên {method_name}",
                        save_path=f'plots/abide_clustering_{method_name.replace(" ", "_").lower()}_{algo_name.replace(" ", "_").lower()}.png'
                    )
                
            except Exception as e:
                print(f"Lỗi khi áp dụng {algo_name} với {method_name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Tạo DataFrame từ kết quả
    results_df = pd.DataFrame(results)
    
    # Lưu kết quả vào file CSV
    results_df.to_csv('plots/abide_clustering_comparison_results.csv', index=False)
    print("\nĐã lưu kết quả vào 'plots/abide_clustering_comparison_results.csv'")
    
    # In bảng tóm tắt kết quả
    print("\n=== Tóm tắt kết quả ===")
    summary = results_df.sort_values('f1', ascending=False).reset_index(drop=True)
    print(summary[['dim_reduction', 'clustering', 'n_clusters', 'accuracy', 'f1', 'ari', 'nmi']].to_string(index=True))
    
    # Tìm phương pháp tốt nhất dựa trên F1-score
    best_result = results_df.loc[results_df['f1'].idxmax()]
    print(f"\nPhương pháp tốt nhất theo F1 score: {best_result['dim_reduction']} + {best_result['clustering']}")
    print(f"F1 Score: {best_result['f1']:.4f}, ARI: {best_result['ari']:.4f}, NMI: {best_result['nmi']:.4f}")
    
    # Tìm phương pháp tốt nhất dựa trên ARI
    best_ari_result = results_df.loc[results_df['ari'].idxmax()]
    print(f"\nPhương pháp tốt nhất theo ARI: {best_ari_result['dim_reduction']} + {best_ari_result['clustering']}")
    print(f"ARI: {best_ari_result['ari']:.4f}, NMI: {best_ari_result['nmi']:.4f}, F1: {best_ari_result['f1']:.4f}")
    
    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(14, 8))
    
    # Tạo pivot table để so sánh các phương pháp
    heatmap_data = results_df.pivot_table(
        index='clustering', 
        columns='dim_reduction', 
        values='f1',
        aggfunc='max'
    )
    
    # Vẽ heatmap
    ax = plt.axes()
    im = ax.imshow(heatmap_data.values, cmap='YlGnBu')
    
    # Thêm nhãn
    ax.set_xticks(np.arange(len(heatmap_data.columns)))
    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_xticklabels(heatmap_data.columns)
    ax.set_yticklabels(heatmap_data.index)
    
    # Xoay nhãn trục x
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Thêm giá trị vào các ô
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            value = heatmap_data.values[i, j]
            if not np.isnan(value):
                text = ax.text(j, i, f"{value:.3f}",
                               ha="center", va="center", color="black" if value < 0.7 else "white")
    
    plt.colorbar(im, label='F1 Score')
    plt.title('So sánh hiệu suất phân cụm với các phương pháp khác nhau')
    plt.tight_layout()
    plt.savefig('plots/abide_clustering_comparison_heatmap.png')
    plt.close()
    
    print("\nĐã hoàn thành phân tích phân cụm!")

if __name__ == "__main__":
    main() 